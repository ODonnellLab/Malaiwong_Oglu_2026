"""
Epistasis SOS barplot + paired ECDFs — Figure 1e, Malaiwong & Oglu 2026.

Figure width is computed from content; ECDF panels match sos_condition_plots.py.

  A (left):      barplot — 8 genotypes in epistasis pairs
  B (top right): ECDF — genotypes WITHOUT cest-2.1 mutation
  C (bot right): ECDF — genotypes WITH cest-2.1 mutation

Usage:
    python sos_epistasis_plots.py --out-dir data/sos_figures/
    python sos_epistasis_plots.py --data-dir /path/to/raw --out-dir data/sos_figures/
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.transforms import blended_transform_factory

import sos_style
from sos_style import NR_CEILING

DATA_FILE  = "1e_Epistasis_SOSdata_updated.csv"
STATS_FILE = "1e_Epistasis_SOSdata_updated_LMMstats.csv"
OUT_STEM   = "1e_Epistasis_SOSdata_updated"

GENO_NORMALIZE = {"WT": "N2", "Wild type": "N2"}

GENO_ORDER = [
    "N2",    "cest-2.1",
    "tbh-1", "tbh-1; cest-2.1",
    "bas-1", "bas-1; cest-2.1",
    "tph-1", "tph-1; cest-2.1",
]

GENO_COLORS = {g: sos_style.COLORS[g] for g in GENO_ORDER}

# Double mutants: lighter bars/dots to distinguish from single mutants
GENO_ALPHA_BAR = {g: (0.50 if "cest-2.1" in g and g != "cest-2.1" else 0.72)
                  for g in GENO_ORDER}
GENO_ALPHA_DOT = {g: (0.18 if "cest-2.1" in g and g != "cest-2.1" else 0.25)
                  for g in GENO_ORDER}

ECDF_TOP = ["N2",       "tbh-1",           "bas-1",           "tph-1"]
ECDF_BOT = ["cest-2.1", "tbh-1; cest-2.1", "bas-1; cest-2.1", "tph-1; cest-2.1"]

# Dashed lines for double mutants in ECDF
ECDF_LS = {g: ("--" if "cest-2.1" in g and g != "cest-2.1" else "-") for g in GENO_ORDER}

# Bar geometry — pairs separated by modest gap (no nested sub-conditions)
INTRA_SPACING = 0.20   # within each epistasis pair
INTER_SPACING = 0.42   # between pairs

_PAIRS = [
    ("N2",    "cest-2.1"),
    ("tbh-1", "tbh-1; cest-2.1"),
    ("bas-1", "bas-1; cest-2.1"),
    ("tph-1", "tph-1; cest-2.1"),
]
_GENO_X = {}
_x = 0.0
for _g1, _g2 in _PAIRS:
    _GENO_X[_g1] = _x
    _GENO_X[_g2] = _x + INTRA_SPACING
    _x += INTRA_SPACING + INTER_SPACING

X_LO = min(_GENO_X.values()) - 0.30
X_HI = max(_GENO_X.values()) + 0.30


# ── data loading ──────────────────────────────────────────────────────────────

def load_data(data_dir):
    csv_path   = os.path.join(data_dir, DATA_FILE)
    stats_path = os.path.join(data_dir, STATS_FILE)

    df = pd.read_csv(csv_path)
    df["norm_geno"] = df["Genotype"].map(lambda g: GENO_NORMALIZE.get(g, g))
    df = df[df["Bacteria"] == "OP50"].copy()

    sid_to_geno = (df[["Strain_ID", "norm_geno"]].drop_duplicates()
                   .set_index("Strain_ID")["norm_geno"].to_dict())
    n2_sid = next(s for s, g in sid_to_geno.items() if g == "N2")

    nr_counts = (
        df[df["NR"].notna()]
        .groupby(["norm_geno", "Date"])["NR"]
        .max().reset_index().rename(columns={"NR": "nr_count"})
    )
    nr_counts["nr_count"] = nr_counts["nr_count"].clip(lower=0).astype(int)

    rt = (df[df["Response.time"].notna()]
          .drop_duplicates(subset=["norm_geno", "Date", "Response.time"])
          .copy())
    rt["is_nr"] = False

    nr_rows = []
    for _, row in nr_counts.iterrows():
        for _ in range(int(row["nr_count"])):
            nr_rows.append({"norm_geno": row["norm_geno"], "Date": row["Date"],
                            "Response.time": NR_CEILING, "is_nr": True})
    if nr_rows:
        rt = pd.concat([rt, pd.DataFrame(nr_rows)], ignore_index=True)

    pvals = {}
    if os.path.exists(stats_path):
        stats = pd.read_csv(stats_path)
        geno_to_sid = {v: k for k, v in sid_to_geno.items()}
        for g in GENO_ORDER:
            if g == "N2":
                continue
            sid = geno_to_sid.get(g)
            if sid is None:
                continue
            row = stats[stats["contrast"].str.contains(sid, na=False) &
                        stats["contrast"].str.contains(n2_sid, na=False)]
            if not row.empty:
                pvals[g] = row.iloc[0]["p.adj.BH"]

    return rt, pvals


# ── barplot ───────────────────────────────────────────────────────────────────

def draw_barplot(ax, animal_df, pvals):
    rng = np.random.default_rng(42)

    for g in GENO_ORDER:
        xc    = _GENO_X[g]
        color = GENO_COLORS[g]
        sub   = animal_df[animal_df["norm_geno"] == g]
        if sub.empty:
            continue

        rt = sub["Response.time"].values
        nr = sub["is_nr"].values
        xs = xc + rng.uniform(-0.07, 0.07, len(sub))
        ax.scatter(xs[~nr], rt[~nr], s=4, color=color,
                   alpha=GENO_ALPHA_DOT[g], zorder=2, lw=0)
        if nr.any():
            ax.scatter(xs[nr], rt[nr], s=5, facecolors="none",
                       edgecolors=color, alpha=0.45, zorder=2, lw=0.5)

        dm  = sub.groupby("Date")["Response.time"].median().values
        xsd = xc + rng.uniform(-0.05, 0.05, len(dm))
        ax.scatter(xsd, dm, s=13, color="black", zorder=5, lw=0)

        m   = dm.mean()
        sem = dm.std(ddof=1) / np.sqrt(len(dm)) if len(dm) > 1 else 0.0
        ax.bar(xc, m, width=sos_style.BAR_W, color=color,
               alpha=GENO_ALPHA_BAR[g], zorder=3, edgecolor="none")
        ax.errorbar(xc, m, yerr=sem, fmt="none",
                    color="black", capsize=2.5, lw=1.0, zorder=4)

        s = sos_style.stars(pvals.get(g, np.nan))
        if s:
            ax.text(xc, NR_CEILING * 1.09, s, ha="center", va="bottom",
                    fontsize=8, fontweight="bold", color="black", zorder=6)

    ax.axhline(NR_CEILING, color="gray", lw=0.6, ls=":", alpha=0.5)
    ax.set_xlim(X_LO, X_HI)
    ax.set_ylim(0, NR_CEILING * 1.38)
    ax.set_xticks(list(_GENO_X.values()))
    sos_style.set_xticklabels(ax, GENO_ORDER, fontsize=7.5, rotation=40, ha="right")
    ax.set_yticks([0, 5, 10, 15, 20])
    ax.set_ylabel("Response time (s)", fontsize=9)
    ax.spines[["top", "right", "bottom"]].set_visible(False)
    ax.tick_params(axis="x", bottom=False, labelsize=7.5)
    ax.tick_params(axis="y", labelsize=8)

    trans = blended_transform_factory(ax.transData, ax.transAxes)
    for g1, g2 in _PAIRS:
        lo = _GENO_X[g1] - sos_style.BAR_W / 2 - 0.04
        hi = _GENO_X[g2] + sos_style.BAR_W / 2 + 0.04
        ax.plot([lo, hi], [0, 0], color="black", lw=0.8,
                clip_on=False, transform=trans)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default="/Users/mikeodonnell/Downloads/raw_figure_data",
                        help="Directory with raw SOS CSV and LMMstats files")
    parser.add_argument("--out-dir",  default="data/sos_figures/")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    animal_df, pvals = load_data(args.data_dir)

    # Fixed-inch layout — figure width computed from content
    S     = sos_style
    bar_w = (X_HI - X_LO) * S.BAR_SCALE
    fig_w = S.L_MAR + bar_w + S.BAR_ECDF_ROT + S.ECDF_W + S.R_MAR
    fig_h = S.T_MAR + S.CONTENT_H + S.B_ROT

    fig = plt.figure(figsize=(fig_w, fig_h))

    def _ax(l, b, w, h):
        return S.make_ax(fig, fig_w, fig_h, l, b, w, h)

    ax_bar = _ax(S.L_MAR, S.B_ROT, bar_w, S.CONTENT_H)
    ecdf_l = S.L_MAR + bar_w + S.BAR_ECDF_ROT
    ax_top = _ax(ecdf_l, S.B_ROT + S.ECDF_H + S.ECDF_VGAP, S.ECDF_W, S.ECDF_H)
    ax_bot = _ax(ecdf_l, S.B_ROT,                            S.ECDF_W, S.ECDF_H)

    draw_barplot(ax_bar, animal_df, pvals)
    ax_bar.text(-0.08, 1.04, "A", transform=ax_bar.transAxes,
                fontsize=10, fontweight="bold", va="bottom")

    S.draw_ecdf(ax_top, animal_df, ECDF_TOP, GENO_COLORS,
                title="Without cest-2.1", geno_col="norm_geno", geno_ls=ECDF_LS)
    ax_top.text(-0.22, 1.08, "B", transform=ax_top.transAxes,
                fontsize=8, fontweight="bold", va="bottom")

    S.draw_ecdf(ax_bot, animal_df, ECDF_BOT, GENO_COLORS,
                title="With cest-2.1", geno_col="norm_geno", geno_ls=ECDF_LS)
    ax_bot.text(-0.22, 1.08, "C", transform=ax_bot.transAxes,
                fontsize=8, fontweight="bold", va="bottom")

    for ext in ("pdf", "png"):
        path = os.path.join(args.out_dir, f"{OUT_STEM}.{ext}")
        kw   = dict(bbox_inches="tight")
        if ext == "png":
            kw["dpi"] = 300
        fig.savefig(path, **kw)
        print(f"Saved: {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
