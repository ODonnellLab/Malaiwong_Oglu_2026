"""
Combined per-condition SOS barplot + ECDF subpanels.

Top panel: grouped barplot — genotypes on x-axis, 4 conditions as bar groups.
  Bars:            mean of daily medians ± SEM (day = unit of replication)
  Colored dots:    individual animal response times (translucent, condition color)
  Hollow circles:  NR animals censored at 20 s (open, condition color edge)
  Black dots:      per-day medians
  Stars:           LMM BH-adjusted significance vs N2 within each condition

Bottom row: overlapping ECDF for each condition (genotype canonical colors).

Usage:
    python sos_condition_plots.py --data-dir /path/to/SOS --out-dir data/
"""

import argparse
import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.transforms import blended_transform_factory

import sos_style
from sos_style import NR_CEILING

STRAIN_ID_TO_GENO = {"N2": "N2", "PHX3900": "cest-2.1", "RB1161": "tbh-1"}
GENO_ORDER  = ["N2", "cest-2.1", "tbh-1"]
GENO_COLORS = {g: sos_style.COLORS[g] for g in GENO_ORDER}

CONDITION_CONFIGS = [
    dict(data_file="1c_3minOffFood_33OctVSNoOdor_SOSdata.csv",
         stats_prefix="1c_3minOffFood_33OctVSNoOdor_SOSdata",
         condition="3 min off-food", odor="no",
         label="3 min / no odor",    color="#BBBBBB"),
    dict(data_file="1c_3minOffFood_33OctVSNoOdor_SOSdata.csv",
         stats_prefix="1c_3minOffFood_33OctVSNoOdor_SOSdata",
         condition="3 min off-food", odor="33% octanol",
         label="3 min / octanol",    color="#E8974A"),
    dict(data_file="1c_N2cest-2.1tbh-1_33octanol_SOSdata.csv",
         stats_prefix="1c_N2cest-2.1tbh-1_33octanol_SOSdata",
         condition="control",        odor="33% octanol",
         label="20 min / octanol",   color="#2166AC"),
    dict(data_file="1c_N2cest2.1tbh-1_10nonanone_SOSdata.csv",
         stats_prefix="1c_N2cest2.1tbh-1_10nonanone_SOSdata",
         condition="control",        odor="10% nonanone",
         label="20 min / nonanone",  color="#1B7837"),
]

# Bar geometry — spacing large because conditions are nested within genotypes
_BAR_STEP = 0.20                                 # center-to-center within a genotype
_COND_OFF = [_BAR_STEP * (j - 1.5) for j in range(4)]  # [-0.30,-0.10,+0.10,+0.30]
_GENO_X   = {g: i * 1.05 for i, g in enumerate(GENO_ORDER)}
_X_LO     = min(_GENO_X.values()) - 0.40
_X_HI     = max(_GENO_X.values()) + 0.40


# ── data loading ──────────────────────────────────────────────────────────────

def load_condition_data(data_dir, cfg):
    df  = pd.read_csv(os.path.join(data_dir, cfg["data_file"]))
    sub = df[
        (df["Condition"] == cfg["condition"]) &
        (df["Odor"]      == cfg["odor"])      &
        (df["Bacteria"]  == "OP50")
    ].copy()
    sub["Genotype"] = sub["Genotype"].replace({"WT": "N2", "Wild type": "N2"})
    sub = sub[sub["Genotype"].isin(GENO_ORDER)].copy()

    nr_counts = (
        sub[sub["NR"].notna()]
        .groupby(["Genotype", "Date"])["NR"]
        .max().reset_index().rename(columns={"NR": "nr_count"})
    )
    nr_counts["nr_count"] = nr_counts["nr_count"].clip(lower=0).astype(int)

    rt = (sub[sub["Response.time"].notna()]
          .drop_duplicates(subset=["Genotype", "Date", "Response.time"])
          .copy())
    rt["is_nr"] = False

    nr_rows = []
    for _, row in nr_counts.iterrows():
        for _ in range(int(row["nr_count"])):
            nr_rows.append({"Genotype": row["Genotype"], "Date": row["Date"],
                            "Response.time": NR_CEILING, "is_nr": True})
    if nr_rows:
        rt = pd.concat([rt, pd.DataFrame(nr_rows)], ignore_index=True)

    return rt[["Genotype", "Date", "Response.time", "is_nr"]]


def daily_medians(animal_df):
    return (animal_df.groupby(["Genotype", "Date"])["Response.time"]
            .median().reset_index().rename(columns={"Response.time": "median_rt"}))


def bar_stats(dm_df):
    def _sem(x):
        return x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0
    return (dm_df.groupby("Genotype")["median_rt"]
            .agg(mean="mean", sem=_sem)
            .reindex(GENO_ORDER))


def load_lmm_stats(data_dir, cfg):
    means = pd.read_csv(os.path.join(data_dir, cfg["stats_prefix"] + "_LMMmeans.csv"))
    stats = pd.read_csv(os.path.join(data_dir, cfg["stats_prefix"] + "_LMMstats.csv"))
    means["genotype"] = means["Strain_ID"].map(STRAIN_ID_TO_GENO)
    if "Odor" in means.columns:
        means = means[means["Odor"] == cfg["odor"]].copy()
        stats = stats[stats["Odor"] == cfg["odor"]].copy()
    means = means.set_index("genotype")
    pvals = {}
    for _, row in stats.iterrows():
        parts = row["contrast"].split(" - ")
        if len(parts) == 2:
            g1 = STRAIN_ID_TO_GENO.get(parts[0].strip(), parts[0].strip())
            g2 = STRAIN_ID_TO_GENO.get(parts[1].strip(), parts[1].strip())
            pvals[(g1, g2)] = row["p.adj.BH"]
            pvals[(g2, g1)] = row["p.adj.BH"]
    return means, pvals


# ── barplot ───────────────────────────────────────────────────────────────────

def draw_combined_barplot(ax, all_data):
    rng = np.random.default_rng(42)

    for ci, cdata in enumerate(all_data):
        color  = cdata["cfg"]["color"]
        offset = _COND_OFF[ci]
        for g in GENO_ORDER:
            xc = _GENO_X[g] + offset

            sub = cdata["animal_df"][cdata["animal_df"]["Genotype"] == g]
            if not sub.empty:
                rt = sub["Response.time"].values
                nr = sub["is_nr"].values
                xs = xc + rng.uniform(-0.05, 0.05, len(sub))
                ax.scatter(xs[~nr], rt[~nr], s=4, color=color,
                           alpha=0.25, zorder=2, lw=0)
                if nr.any():
                    ax.scatter(xs[nr], rt[nr], s=5, facecolors="none",
                               edgecolors=color, alpha=0.45, zorder=2, lw=0.5)

            bs = cdata["bs_df"]
            if g in bs.index and not np.isnan(bs.loc[g, "mean"]):
                m, se = bs.loc[g, "mean"], bs.loc[g, "sem"]
                ax.bar(xc, m, width=sos_style.BAR_W, color=color, alpha=0.72,
                       zorder=3, edgecolor="none")
                ax.errorbar(xc, m, yerr=se,
                            fmt="none", color="black", capsize=2.5,
                            lw=1.0, zorder=4)

            dm = cdata["dm_df"][cdata["dm_df"]["Genotype"] == g]
            if not dm.empty:
                xs_dm = xc + rng.uniform(-0.04, 0.04, len(dm))
                ax.scatter(xs_dm, dm["median_rt"].values, s=13,
                           color="black", zorder=5, lw=0)

            if g != "N2":
                p = cdata["pvals"].get(("N2", g), np.nan)
                s = sos_style.stars(p)
                if s:
                    ax.text(xc, NR_CEILING * 1.09, s, ha="center", va="bottom",
                            fontsize=8, fontweight="bold", color="black", zorder=6)

    ax.axhline(NR_CEILING, color="gray", lw=0.6, ls=":", alpha=0.5)
    ax.set_xlim(_X_LO, _X_HI)
    ax.set_ylim(0, NR_CEILING * 1.38)
    ax.set_xticks(list(_GENO_X.values()))
    sos_style.set_xticklabels(ax, GENO_ORDER, fontsize=9)
    ax.set_yticks([0, 5, 10, 15, 20])
    ax.set_ylabel("Response time (s)", fontsize=9)
    ax.spines[["top", "right", "bottom"]].set_visible(False)
    ax.tick_params(axis="x", bottom=False, labelsize=9)
    ax.tick_params(axis="y", labelsize=8)

    trans    = blended_transform_factory(ax.transData, ax.transAxes)
    seg_half = abs(_COND_OFF[0]) + sos_style.BAR_W / 2 + 0.03
    for i, g in enumerate(GENO_ORDER):
        xc = _GENO_X[g]
        lo = _X_LO if i == 0                    else xc - seg_half
        hi = _X_HI if i == len(GENO_ORDER) - 1 else xc + seg_half
        ax.plot([lo, hi], [0, 0], color="black", lw=0.8,
                clip_on=False, transform=trans)

    cond_handles = [
        mpatches.Patch(color=cfg["color"], alpha=0.8, label=cfg["label"])
        for cfg in CONDITION_CONFIGS
    ]
    return cond_handles, sos_style.dot_legend_handles()


# ── main ──────────────────────────────────────────────────────────────────────

def _pvals_from_stats(stats_df):
    pvals = {}
    for _, row in stats_df.iterrows():
        parts = row["contrast"].split(" - ")
        if len(parts) == 2:
            g1 = STRAIN_ID_TO_GENO.get(parts[0].strip(), parts[0].strip())
            g2 = STRAIN_ID_TO_GENO.get(parts[1].strip(), parts[1].strip())
            pvals[(g1, g2)] = row["p.adj.BH"]
            pvals[(g2, g1)] = row["p.adj.BH"]
    return pvals


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default=None,
                        help="Raw SOS CSV directory; required on first run or to rescan")
    parser.add_argument("--out-dir",  default="data/")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    animal_cache = os.path.join(args.out_dir, "sos_condition_animal_data.parquet")
    stats_cache  = os.path.join(args.out_dir, "sos_condition_lmm_stats.csv")

    if args.data_dir:
        animal_dfs, stats_dfs = [], []
        for cfg in CONDITION_CONFIGS:
            adf = load_condition_data(args.data_dir, cfg)
            adf["condition"] = cfg["label"]
            animal_dfs.append(adf)
            src = pd.read_csv(
                os.path.join(args.data_dir, cfg["stats_prefix"] + "_LMMstats.csv"))
            if "Odor" in src.columns:
                src = src[src["Odor"] == cfg["odor"]].copy()
            src["condition"] = cfg["label"]
            stats_dfs.append(src)
        animal_combined = pd.concat(animal_dfs, ignore_index=True)
        stats_combined  = pd.concat(stats_dfs,  ignore_index=True)
        animal_combined.to_parquet(animal_cache, index=False)
        stats_combined.to_csv(stats_cache, index=False, float_format="%.6g")
        print(f"Cached: {animal_cache}")
        print(f"Cached: {stats_cache}")
    else:
        if not os.path.exists(animal_cache):
            parser.error("No cache found — supply --data-dir to scan raw SOS files")
        animal_combined = pd.read_parquet(animal_cache)
        stats_combined  = pd.read_csv(stats_cache)

    all_data = []
    for cfg in CONDITION_CONFIGS:
        adf  = animal_combined[animal_combined["condition"] == cfg["label"]].copy()
        dmdf = daily_medians(adf)
        bsdf = bar_stats(dmdf)
        sc   = stats_combined[stats_combined["condition"] == cfg["label"]]
        pv   = _pvals_from_stats(sc)
        all_data.append(dict(cfg=cfg, animal_df=adf, dm_df=dmdf,
                             bs_df=bsdf, pvals=pv))

    # Fixed-inch layout — width computed from content
    S  = sos_style
    bar_w    = (_X_HI - _X_LO) * S.BAR_SCALE
    ecdf_blk = 2 * S.ECDF_W + S.ECDF_HGAP
    fig_w    = S.L_MAR + bar_w + S.BAR_LGD + S.LGD_W + S.LGD_ECDF + ecdf_blk + S.R_MAR
    fig_h    = S.T_MAR + S.CONTENT_H + S.B_FLAT

    fig = plt.figure(figsize=(fig_w, fig_h))

    def _ax(l, b, w, h):
        return S.make_ax(fig, fig_w, fig_h, l, b, w, h)

    ax_bar = _ax(S.L_MAR, S.B_FLAT, bar_w, S.CONTENT_H)

    ecdf_l  = S.L_MAR + bar_w + S.BAR_LGD + S.LGD_W + S.LGD_ECDF
    ax_ecdf = [
        _ax(ecdf_l,                   S.B_FLAT + S.ECDF_H + S.ECDF_VGAP, S.ECDF_W, S.ECDF_H),
        _ax(ecdf_l + S.ECDF_W + S.ECDF_HGAP, S.B_FLAT + S.ECDF_H + S.ECDF_VGAP, S.ECDF_W, S.ECDF_H),
        _ax(ecdf_l,                   S.B_FLAT,                            S.ECDF_W, S.ECDF_H),
        _ax(ecdf_l + S.ECDF_W + S.ECDF_HGAP, S.B_FLAT,                    S.ECDF_W, S.ECDF_H),
    ]

    cond_handles, dot_handles = draw_combined_barplot(ax_bar, all_data)
    ax_bar.text(-0.10, 1.04, "A", transform=ax_bar.transAxes,
                fontsize=10, fontweight="bold", va="bottom")

    lgd_x   = (S.L_MAR + bar_w + S.BAR_LGD) / fig_w
    lgd_top = (S.B_FLAT + S.CONTENT_H) / fig_h
    lgd_bot = S.B_FLAT / fig_h
    fig.legend(handles=cond_handles, fontsize=7, framealpha=0.9,
               title="Starvation / odorant", title_fontsize=7,
               bbox_to_anchor=(lgd_x, lgd_top), loc="upper left",
               bbox_transform=fig.transFigure, borderaxespad=0)
    fig.legend(handles=dot_handles, fontsize=7, framealpha=0.9,
               bbox_to_anchor=(lgd_x, lgd_bot), loc="lower left",
               bbox_transform=fig.transFigure, borderaxespad=0)

    for i, (cdata, ax) in enumerate(zip(all_data, ax_ecdf)):
        S.draw_ecdf(ax, cdata["animal_df"], GENO_ORDER, GENO_COLORS,
                    title=cdata["cfg"]["label"], geno_col="Genotype")
        ax.text(-0.18, 1.08, chr(ord("B") + i), transform=ax.transAxes,
                fontsize=8, fontweight="bold", va="bottom")

    for ext in ("pdf", "png"):
        path = os.path.join(args.out_dir, f"sos_condition_plots.{ext}")
        kw   = dict(bbox_inches="tight")
        if ext == "png":
            kw["dpi"] = 300
        fig.savefig(path, **kw)
        print(f"Saved: {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
