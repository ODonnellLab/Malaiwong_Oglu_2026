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

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"]  = 42

NR_CEILING = 20.0

STRAIN_ID_TO_GENO = {"N2": "N2", "PHX3900": "cest-2.1", "RB1161": "tbh-1"}
GENO_ORDER   = ["N2", "cest-2.1", "tbh-1"]
GENO_COLORS  = {"N2": "#888888", "cest-2.1": "#984F9F", "tbh-1": "#8E3E29"}

CONDITION_CONFIGS = [
    dict(data_file="1c_3minOffFood_33OctVSNoOdor_SOSdata.csv",
         stats_prefix="1c_3minOffFood_33OctVSNoOdor_SOSdata",
         condition="3 min off-food", odor="no",
         label="3 min / no odor",    short="3 min\nno odor",  color="#BBBBBB"),
    dict(data_file="1c_3minOffFood_33OctVSNoOdor_SOSdata.csv",
         stats_prefix="1c_3minOffFood_33OctVSNoOdor_SOSdata",
         condition="3 min off-food", odor="33% octanol",
         label="3 min / octanol",    short="3 min\noctanol",  color="#E8974A"),
    dict(data_file="1c_N2cest-2.1tbh-1_33octanol_SOSdata.csv",
         stats_prefix="1c_N2cest-2.1tbh-1_33octanol_SOSdata",
         condition="control",        odor="33% octanol",
         label="20 min / octanol",   short="20 min\noctanol", color="#2166AC"),
    dict(data_file="1c_N2cest2.1tbh-1_10nonanone_SOSdata.csv",
         stats_prefix="1c_N2cest2.1tbh-1_10nonanone_SOSdata",
         condition="control",        odor="10% nonanone",
         label="20 min / nonanone",  short="20 min\nnonanone", color="#1B7837"),
]

# Bar geometry
_BAR_W    = 0.28
_BAR_STEP = 0.32                               # bar_width + small gap
_COND_OFF = [_BAR_STEP * (j - 1.5) for j in range(4)]   # [-0.48,-0.16,+0.16,+0.48]
_GENO_X   = {g: i * 2.2 for i, g in enumerate(GENO_ORDER)}  # cluster centers


def _stars(p):
    if pd.isna(p): return ""
    if p < 0.001:  return "***"
    if p < 0.01:   return "**"
    if p < 0.05:   return "*"
    return ""


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
    """Mean ± SEM of daily medians per genotype."""
    def _sem(x):
        return x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0
    return (dm_df.groupby("Genotype")["median_rt"]
            .agg(mean="mean", sem=_sem)
            .reindex(GENO_ORDER))


def load_lmm_stats(data_dir, cfg):
    """Return (means_df indexed by genotype, pvals dict)."""
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
    """
    all_data: list of dicts, one per condition —
        cfg, animal_df, dm_df, lmm_means, pvals

    Bars:   LMM geometric mean ± SE (from _LMMmeans.csv)
    Dots:   individual animals (translucent, condition color; NR = open circle)
    Black:  per-day medians (scaled by n_days for size)
    Stars:  LMM BH-adjusted significance vs N2
    """
    rng = np.random.default_rng(42)

    for ci, cdata in enumerate(all_data):
        color  = cdata["cfg"]["color"]
        offset = _COND_OFF[ci]
        for g in GENO_ORDER:
            xc = _GENO_X[g] + offset

            # Individual animals
            sub = cdata["animal_df"][cdata["animal_df"]["Genotype"] == g]
            if not sub.empty:
                rt = sub["Response.time"].values
                nr = sub["is_nr"].values
                xs = xc + rng.uniform(-0.09, 0.09, len(sub))
                ax.scatter(xs[~nr], rt[~nr], s=4, color=color,
                           alpha=0.25, zorder=2, lw=0)
                if nr.any():
                    ax.scatter(xs[nr], rt[nr], s=5, facecolors="none",
                               edgecolors=color, alpha=0.45, zorder=2, lw=0.5)

            # Bar: mean of daily medians ± SEM
            bs = cdata["bs_df"]
            if g in bs.index and not np.isnan(bs.loc[g, "mean"]):
                m, se = bs.loc[g, "mean"], bs.loc[g, "sem"]
                ax.bar(xc, m, width=_BAR_W, color=color, alpha=0.72,
                       zorder=3, edgecolor="none")
                ax.errorbar(xc, m, yerr=se,
                            fmt="none", color="black", capsize=2.5,
                            lw=1.0, zorder=4)

            # Daily medians: black dots, size proportional to n days
            dm = cdata["dm_df"][cdata["dm_df"]["Genotype"] == g]
            if not dm.empty:
                xs_dm = xc + rng.uniform(-0.06, 0.06, len(dm))
                ax.scatter(xs_dm, dm["median_rt"].values, s=20,
                           color="black", zorder=5, lw=0)

            # Significance stars vs N2
            if g != "N2":
                p = cdata["pvals"].get(("N2", g), np.nan)
                s = _stars(p)
                if s:
                    ax.text(xc, NR_CEILING * 1.09, s, ha="center", va="bottom",
                            fontsize=8, fontweight="bold", color="black", zorder=6)

    ax.axhline(NR_CEILING, color="gray", lw=0.6, ls=":", alpha=0.5)

    x_lo = min(_GENO_X.values()) - 0.7
    x_hi = max(_GENO_X.values()) + 0.7
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(0, NR_CEILING * 1.38)
    ax.set_xticks(list(_GENO_X.values()))
    ax.set_xticklabels(GENO_ORDER, fontsize=9)
    ax.set_ylabel("Response time (s)", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)

    # Condition legend
    cond_handles = [
        mpatches.Patch(color=cfg["color"], alpha=0.8, label=cfg["label"])
        for cfg in CONDITION_CONFIGS
    ]
    l1 = ax.legend(handles=cond_handles, fontsize=7.5, framealpha=0.9,
                   loc="upper right", title="Starvation / odorant",
                   title_fontsize=7.5)
    ax.add_artist(l1)

    # Dot legend
    import matplotlib.lines as mlines
    dot_handles = [
        mlines.Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
                      markersize=4, label="Individual animal"),
        mlines.Line2D([0], [0], marker="o", color="w", markerfacecolor="none",
                      markeredgecolor="gray", markersize=4,
                      label="NR (censored 20 s)"),
        mlines.Line2D([0], [0], marker="o", color="w", markerfacecolor="black",
                      markersize=6, label="Daily median"),
    ]
    ax.legend(handles=dot_handles, fontsize=7, framealpha=0.9, loc="upper left")


# ── ECDF ─────────────────────────────────────────────────────────────────────

def draw_ecdf(ax, animal_df, title=""):
    for g in GENO_ORDER:
        vals = (animal_df[animal_df["Genotype"] == g]["Response.time"]
                .dropna().sort_values().values)
        if len(vals) == 0:
            continue
        n     = len(vals)
        x     = np.concatenate([[0], vals, [NR_CEILING]])
        y     = np.concatenate([[0], np.arange(1, n + 1) / n, [1.0]])
        color = GENO_COLORS.get(g, "#aaaaaa")
        ax.step(x, y, where="post", color=color, lw=1.8, label=g)

    ax.axvline(NR_CEILING, color="gray", lw=0.6, ls=":", alpha=0.5)
    ax.set_xlabel("Response time (s)", fontsize=7.5)
    ax.set_ylabel("Cumulative fraction", fontsize=7.5)
    ax.set_xlim(0, NR_CEILING + 0.3)
    ax.set_ylim(0, 1.02)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6.5, framealpha=0.9, loc="lower right")
    ax.set_title(title, fontsize=8, pad=3)


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
        # Rescan: load from raw CSVs and write caches
        animal_dfs, stats_dfs = [], []
        for cfg in CONDITION_CONFIGS:
            adf = load_condition_data(args.data_dir, cfg)
            adf = adf.copy()
            adf["condition"] = cfg["label"]
            animal_dfs.append(adf)

            _, stats_raw = load_lmm_stats(args.data_dir, cfg)
            # Reconstruct a DataFrame from the pvals dict for caching
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

    # Build per-condition data structures
    all_data = []
    for cfg in CONDITION_CONFIGS:
        adf  = animal_combined[animal_combined["condition"] == cfg["label"]].copy()
        dmdf = daily_medians(adf)
        bsdf = bar_stats(dmdf)
        sc   = stats_combined[stats_combined["condition"] == cfg["label"]]
        pv   = _pvals_from_stats(sc)
        all_data.append(dict(cfg=cfg, animal_df=adf, dm_df=dmdf,
                             bs_df=bsdf, pvals=pv))

    # Figure: barplot on top, 2×2 ECDF grid below
    fig = plt.figure(figsize=(8.5, 11))
    gs  = fig.add_gridspec(
        3, 2,
        height_ratios=[1.6, 1.0, 1.0],
        left=0.10, right=0.97, top=0.95, bottom=0.04,
        hspace=0.50, wspace=0.35)

    ax_bar  = fig.add_subplot(gs[0, :])
    ax_ecdf = [
        fig.add_subplot(gs[1, 0]),   # no odor
        fig.add_subplot(gs[1, 1]),   # 3 min octanol
        fig.add_subplot(gs[2, 0]),   # 20 min octanol
        fig.add_subplot(gs[2, 1]),   # nonanone
    ]

    draw_combined_barplot(ax_bar, all_data)
    ax_bar.text(-0.06, 1.03, "A", transform=ax_bar.transAxes,
                fontsize=12, fontweight="bold", va="bottom")

    for i, (cdata, ax) in enumerate(zip(all_data, ax_ecdf)):
        draw_ecdf(ax, cdata["animal_df"], title=cdata["cfg"]["label"])
        ax.text(-0.15, 1.06, chr(ord("B") + i), transform=ax.transAxes,
                fontsize=10, fontweight="bold", va="bottom")

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
