"""
Combined SOS (stimulus-response) analysis for Malaiwong & Oglu 2026.

Loads per-animal 33% octanol response time data across all figure CSV files,
harmonizes genotype names, deduplicates, censors non-responders at 20 s, and
fits a log-normal linear mixed-effects model (LME) via R/lme4+emmeans to
obtain per-genotype time ratios relative to N2. These are then correlated with
forward-run speed fold-changes from the postural locomotion LME.

NOTE: per-figure SOS analysis (bar plots, ECDF, pairwise brackets) is handled
by ODLabPlotTools (R package, github.com/ODonnellLab/tools). This script
performs the cross-dataset combined analysis for the speed–response correlation
only.

Usage:
    python sos_analysis.py --data-dir /path/to/raw_figure_data --out-dir data/
    python sos_analysis.py --out-dir data/          # replot from cache
    python sos_analysis.py --out-dir data/ --refit  # re-run LME only
    python sos_analysis.py --out-dir data/ --data-dir /path --refresh
"""

import argparse
import glob
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats as sp_stats
try:
    from adjustText import adjust_text as _adjust_text
    _HAS_ADJUSTTEXT = True
except ImportError:
    _HAS_ADJUSTTEXT = False

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"]  = 42

# ── genotype harmonization ────────────────────────────────────────────────────

GENOTYPE_MAP = {
    "WT":                   "N2",
    "Wild type":            "N2",
    "octr-; ser-6":         "octr-1+ser-6",
    "octr-; ser-3; ser-6":  "octr-1+ser-3+ser-6",
    "tph-1; cest-2.1":      "tph-1+cest-2.1",
    "bas-1; cest-2.1":      "bas-1+cest-2.1",
    "tbh-1; cest-2.1":      "cest2.1 + tbh-1",
    "tdc-1; tbh-1":         "tdc-1+tbh-1",
    "slc-17.1":             "slc.17.1a",
    "SLC-17.1":             "slc.17.1a",
    "slc-17.1 backcrossed": "slc.17.1a",
    "T05A1.5":              "TO5A1.5",
    "fcmt-1 (MOY113)":      "fcmt-1",
}

DROP_GENOTYPES = {
    "T05A1.5 rescue in CAN line 1", "T05A1.5 rescue in CAN line 2",
    "gba-4 rescue in CAN line 1", "gba-4 rescue in CAN line 2",
    "gba-4 rescue in CAN line 3",
    "cat-1_CAN", "cat-1_RC", "cat-1_RIC",
    "cest-2.1_GUT", "cest-2.1_RIC",
    "slc-17.1 backcrossed rescue in CAN line 1",
    "slc-17.1 backcrossed rescue in CAN line 2",
    "slc-17.1 backcrossed rescue in CAN line 3",
    "slc-17.1 backcrossed rescue in CAN line 4",
    "slc-17.1 rescue in CAN",
}

NR_CEILING = 20.0

# Display names for the correlation plot — use SOS semicolon convention
# Only entries that differ from the internal (speed-data) name are listed.
DISPLAY_NAME = {
    "octr-1+ser-6":         "octr-1; ser-6",
    "octr-1+ser-3+ser-6":   "octr-1; ser-3; ser-6",
    "tph-1+cest-2.1":       "tph-1; cest-2.1",
    "bas-1+cest-2.1":       "bas-1; cest-2.1",
    "cest2.1 + tbh-1":      "cest-2.1; tbh-1",
    "tdc-1+tbh-1":          "tdc-1; tbh-1",
    "slc.17.1a":            "slc-17.1",
    "TO5A1.5":              "T05A1.5",
}


# Canonical genotype → hex color from ODLabPlotTools/R/generate_genotype_colors.R
# Keyed by display name (SOS semicolon convention).
_CANONICAL_COLORS = {
    "N2":                   "#888888",
    "cest-2.1":             "#984F9F",
    "tbh-1":                "#8E3E29",
    "tdc-1":                "#C15C3E",
    "cest-2.1; tbh-1":      "#8E3E29",
    "tdc-1; tbh-1":         "#C15C3E",
    "glo-1":                "#CC6677",
    "fcmt-1":               "#44AA99",
    "cat-1":                "#332288",
    "bas-1":                "#1A2C7A",
    "bas-1; cest-2.1":      "#1A2C7A",
    "tph-1":                "#6699CC",
    "tph-1; cest-2.1":      "#6699CC",
    "cest-1.2":             "#4EAE49",
    "sqv-7":                "#5F96CA",
    "ugt-64":               "#546FB5",
    "gba-4":                "#3954A5",
    "ctg-1":                "#999933",
    "slc-17.1":             "#46883F",
    "T05A1.5":              "#4CB448",
    "T10C6.6":              "#547A4C",
    "octr-1":               "#117733",
    "octr-1; ser-6":        "#3D9959",
    "octr-1; ser-3; ser-6": "#66BB82",
    "ser-3":                "#88DDAA",
    "ser-6":                "#AAEEBB",
    "nhr-49":               "#DDCC77",
    "ser-2":                "#AA4499",
    "tyra-2":               "#88CCEE",
}


def _geno_color(internal_name):
    display = DISPLAY_NAME.get(internal_name, internal_name)
    return _CANONICAL_COLORS.get(display, _CANONICAL_COLORS.get(internal_name, "#AAAAAA"))


def _stars(q):
    if pd.isna(q):       return ""
    if q < 0.001:        return "***"
    if q < 0.01:         return "**"
    if q < 0.05:         return "*"
    return ""


# ── data loading ──────────────────────────────────────────────────────────────

def load_sos_data(data_dir):
    """Load, filter, deduplicate, and harmonize all SOS CSV files."""
    files = glob.glob(os.path.join(data_dir, "*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    dfs = []
    for f in sorted(files):
        df = pd.read_csv(f)
        df["source_file"] = os.path.basename(f)
        dfs.append(df)
    raw = pd.concat(dfs, ignore_index=True)

    # Filter to on-food control, OP50, 33% octanol
    ctrl = raw[
        (raw["Condition"] == "control") &
        (raw["Bacteria"] == "OP50") &
        (raw["Odor"] == "33% octanol")
    ].copy()

    # ── NR counts per Genotype×Date ───────────────────────────────────────────
    # The NR count appears in the NR column of the first row per group.
    # Some files have a dedicated blank row (Response.time = NaN) for the NR
    # entry; others put it on a real data row. Extract the count before
    # filtering to Response.time, then ignore the blank rows as data.
    # Use max() across files to collapse cross-file duplicates without inflation.
    nr_counts = (
        ctrl[ctrl["NR"].notna()]
        .groupby(["Genotype", "Date"])["NR"]
        .max()
        .reset_index()
        .rename(columns={"NR": "nr_count"})
    )
    nr_counts["nr_count"] = nr_counts["nr_count"].clip(lower=0).astype(int)

    # ── individual response times ─────────────────────────────────────────────
    rt = ctrl[ctrl["Response.time"].notna()].copy()
    rt = rt.drop_duplicates(subset=["Genotype", "Date", "Response.time"])

    # ── harmonize genotype names ──────────────────────────────────────────────
    rt["Genotype"]        = rt["Genotype"].replace(GENOTYPE_MAP)
    nr_counts["Genotype"] = nr_counts["Genotype"].replace(GENOTYPE_MAP)

    rt        = rt[~rt["Genotype"].isin(DROP_GENOTYPES)].copy()
    nr_counts = nr_counts[~nr_counts["Genotype"].isin(DROP_GENOTYPES)].copy()

    # ── add NR animals censored at 20 s ──────────────────────────────────────
    rt["is_nr"] = False
    nr_rows = []
    for _, row in nr_counts.iterrows():
        for _ in range(row["nr_count"]):
            nr_rows.append({
                "Genotype":      row["Genotype"],
                "Date":          row["Date"],
                "Response.time": NR_CEILING,
                "is_nr":         True,
            })
    if nr_rows:
        rt = pd.concat([rt, pd.DataFrame(nr_rows)], ignore_index=True)

    rt = rt.rename(columns={"Genotype": "genotype", "Date": "date",
                             "Response.time": "response_time"})
    rt["recording_id"] = rt["genotype"] + "__" + rt["date"].astype(str)

    print(f"  {len(rt):,} animals, {rt['genotype'].nunique()} genotypes, "
          f"{rt['recording_id'].nunique()} recordings")
    print(f"  NR censored at {NR_CEILING} s: "
          f"{rt['is_nr'].sum()} ({rt['is_nr'].mean() * 100:.1f}%)")
    return rt[["genotype", "date", "recording_id", "response_time", "is_nr"]]


def make_recording_summary(animal_df):
    g = animal_df.groupby(["genotype", "date", "recording_id"])
    return g["response_time"].agg(
        n="count",
        n_nr=lambda x: animal_df.loc[x.index, "is_nr"].sum(),
        median_rt="median",
        mean_rt="mean",
    ).reset_index()


# ── log-normal LME via R/lme4 + emmeans ──────────────────────────────────────
# Model: log(response_time) ~ genotype + (1|date) + (1|recording_id)
# Random date effect: with 97 dates the between-date variance is well-estimated
# and random effects are more appropriate than 97 fixed date parameters.
# recording_id = genotype × date (plate-level random intercept).
# emmeans back-transforms to time ratios vs N2 (geometric mean seconds).

_R_SCRIPT = r"""
suppressMessages(library(lme4))
suppressMessages(library(lmerTest))
suppressMessages(library(emmeans))

d <- read.csv("{animal_csv}", stringsAsFactors = FALSE)
d$genotype     <- factor(d$genotype)
d$genotype     <- relevel(d$genotype, ref = "N2")
d$date         <- factor(d$date)
d$recording_id <- factor(d$recording_id)

cat(sprintf("LME: %d animals, %d genotypes, %d dates, %d recordings\n",
            nrow(d), nlevels(d$genotype), nlevels(d$date),
            nlevels(d$recording_id)))

# Log-normal model: random date + plate (recording_id) intercepts
fit <- lmer(log(response_time) ~ genotype + (1 | date) + (1 | recording_id),
            data    = d,
            REML    = TRUE,
            control = lmerControl(optimizer = "bobyqa"))

if (isSingular(fit)) {{
  message("Note: singular fit (date+recording_id) — dropping recording_id")
  fit <- lmer(log(response_time) ~ genotype + (1 | date),
              data    = d,
              REML    = TRUE,
              control = lmerControl(optimizer = "bobyqa"))
}}

# emmeans on log scale, then back-transform to time ratios vs N2
emm <- emmeans(fit, ~ genotype)
contrasts_vs_n2 <- contrast(emm, method = "trt.vs.ctrl", ref = "N2",
                             adjust = "BH") |> as.data.frame()
means <- summary(emm, type = "response") |> as.data.frame()

write.csv(contrasts_vs_n2, "{contrasts_csv}", row.names = FALSE)
write.csv(means,           "{means_csv}",     row.names = FALSE)
cat("Done.\n")
"""


def fit_lme_sos(animal_df, out_dir):
    animal_csv   = os.path.join(out_dir, "_sos_lme_animals.csv")
    contrasts_csv = os.path.join(out_dir, "_sos_lme_contrasts.csv")
    means_csv    = os.path.join(out_dir, "_sos_lme_means.csv")
    r_script     = os.path.join(out_dir, "_sos_lme_fit.R")

    animal_df.to_csv(animal_csv, index=False)

    script = _R_SCRIPT.format(
        animal_csv=animal_csv,
        contrasts_csv=contrasts_csv,
        means_csv=means_csv,
    )
    with open(r_script, "w") as fh:
        fh.write(script)

    print("  Running log-normal LME in R…")
    result = subprocess.run(["Rscript", r_script], capture_output=True, text=True)
    for line in result.stdout.splitlines():
        print(f"    [R] {line}")
    if result.returncode != 0:
        print(result.stderr[-2000:])
        raise RuntimeError("R/lme4 failed")

    contrasts = pd.read_csv(contrasts_csv)
    means     = pd.read_csv(means_csv)

    # Parse genotype from contrast label: "(genotype) - N2" → "genotype"
    contrasts["genotype"] = (contrasts["contrast"]
                             .str.replace(r" - N2$", "", regex=True)
                             .str.strip("()"))

    # time_ratio = exp(estimate); >1 = slower than N2
    contrasts["time_ratio"]    = np.exp(contrasts["estimate"])
    contrasts["time_ratio_lo"] = np.exp(contrasts["estimate"] - 1.96 * contrasts["SE"])
    contrasts["time_ratio_hi"] = np.exp(contrasts["estimate"] + 1.96 * contrasts["SE"])

    # Add N2 row
    n2_mean_s = means.loc[means["genotype"] == "N2", "response"].iloc[0]
    n2_row = pd.DataFrame([{
        "genotype": "N2", "time_ratio": 1.0,
        "time_ratio_lo": 1.0, "time_ratio_hi": 1.0,
        "p": np.nan, "q": np.nan,
    }])
    # emmeans adjust="BH" writes adjusted p-values into p.value in-place
    stat_df = pd.concat([
        n2_row,
        contrasts[["genotype", "time_ratio", "time_ratio_lo", "time_ratio_hi",
                   "p.value"]].rename(columns={"p.value": "q"}).assign(p=np.nan)
    ], ignore_index=True)

    return stat_df, means, n2_mean_s


# ── distribution plot (strip + ECDF) ─────────────────────────────────────────

_DOT_N2       = "#2166ac"
_DOT_MATCHED  = "#555555"
_DOT_UNMATCHED = "#aaaaaa"
_DIAMOND_MUT  = "#333333"


def make_sos_distribution_plot(animal_df, recording_df, sos_stat, n2_rt_geom_s,
                                out_path, title="Response time (ratio vs N2)"):
    import matplotlib.lines as mlines

    stat   = sos_stat.set_index("genotype")
    order  = stat.sort_values("time_ratio").index.tolist()
    present = set(animal_df["genotype"].unique())
    # Insert N2 at its natural ratio-1.0 position (sorted, so near middle)
    order  = [g for g in order if g in present and g != "N2"]
    # find where N2 fits (ratio=1.0)
    ratios = [stat.loc[g, "time_ratio"] if g in stat.index else np.nan for g in order]
    n2_pos = next((i for i, r in enumerate(ratios) if not pd.isna(r) and r >= 1.0),
                  len(order))
    order.insert(n2_pos, "N2")

    n_geno = len(order)
    ytick  = {g: i for i, g in enumerate(order)}

    # Per-date N2 median response time for normalizing strip-plot dots
    n2_by_date = (animal_df[animal_df["genotype"] == "N2"]
                  .groupby("date")["response_time"].median().to_dict())
    n2_grand   = animal_df[animal_df["genotype"] == "N2"]["response_time"].median()

    rec = recording_df.copy()
    rec["n2_ref"]       = rec["date"].map(n2_by_date).fillna(n2_grand)
    rec["rt_norm"]      = rec["median_rt"] / rec["n2_ref"]
    rec["date_matched"] = rec["date"].isin(n2_by_date)

    fig_h = max(8, n_geno * 0.45)
    fig, ax = plt.subplots(figsize=(5.5, fig_h))
    fig.suptitle(title, fontsize=11, y=1.01)

    # ── left panel: strip plot ────────────────────────────────────────────────
    for _, row in rec.iterrows():
        g = row["genotype"]
        if g not in ytick or pd.isna(row["rt_norm"]):
            continue
        y = ytick[g]
        x = row["rt_norm"]
        if g == "N2":
            ax.plot(x, y, "o", mfc=_DOT_N2, mec=_DOT_N2, ms=5, alpha=0.65, lw=0, zorder=2)
        else:
            ec = _DOT_MATCHED if row["date_matched"] else _DOT_UNMATCHED
            fc = ec if row["date_matched"] else "none"
            ax.plot(x, y, "o", mfc=fc, mec=ec, ms=5, alpha=0.65, lw=0, zorder=2)

    for g in order:
        y = ytick[g]
        ax.axhline(y - 0.5, color="lightgray", lw=0.3, zorder=0)

        if g == "N2":
            vals = rec[rec["genotype"] == "N2"]["rt_norm"].dropna()
            if vals.empty:
                continue
            ctr = vals.mean()
            sem = vals.sem() if len(vals) > 1 else 0.0
            ax.plot([ctr - sem, ctr + sem], [y, y], color=_DOT_N2,
                    lw=2.5, solid_capstyle="round", zorder=4)
            ax.plot(ctr, y, "D", color=_DOT_N2, ms=7, zorder=5, mec="white", mew=0.5)
            ax.text(ctr, y + 0.19, f"{n2_rt_geom_s:.1f} s", fontsize=6.5,
                    va="bottom", ha="center", color=_DOT_N2, alpha=0.85, zorder=7)
            continue

        if g not in stat.index or pd.isna(stat.loc[g, "time_ratio"]):
            continue
        r      = stat.loc[g]
        ctr    = r["time_ratio"]
        lo, hi = r["time_ratio_lo"], r["time_ratio_hi"]
        q      = r.get("q", np.nan)
        s      = _stars(q)
        if s:
            ax.text(hi + 0.05, y, s, fontsize=11, va="center", ha="left",
                    color="#111111", fontweight="bold", zorder=6)
        ax.plot([lo, hi], [y, y], color=_DIAMOND_MUT, lw=2.5,
                solid_capstyle="round", zorder=4)
        ax.plot(ctr, y, "D", color=_DIAMOND_MUT, ms=7, zorder=5, mec="white", mew=0.5)
        ax.text(ctr, y + 0.19, f"{ctr:.2f}", fontsize=6.5, va="bottom",
                ha="center", color=_DIAMOND_MUT, alpha=0.85, zorder=7)

    ax.axvline(1.0, color="gray", lw=0.8, ls="--", alpha=0.5, zorder=0)
    ax.set_xlabel("Response time (ratio vs N2)", fontsize=9)
    ax.set_xlim(0, 3.0)
    ax.set_ylim(-0.8, n_geno - 0.2)
    ax.set_yticks(list(ytick.values()))
    ax.set_yticklabels([DISPLAY_NAME.get(g, g) for g in order], fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", labelsize=8)

    leg_handles = [
        mlines.Line2D([], [], color=_DOT_MATCHED,   marker="o", ls="none",
                      label="Trial — same-date N2"),
        mlines.Line2D([], [], color=_DOT_UNMATCHED, marker="o", mfc="none", ls="none",
                      label="Trial — grand-mean N2"),
        mlines.Line2D([], [], color=_DIAMOND_MUT,   marker="D", ls="none",
                      mec="white", mew=0.5, label="LME estimate ± 95% CI"),
        mlines.Line2D([], [], color=_DOT_N2,        marker="D", ls="none",
                      mec="white", mew=0.5, label="N2 mean ± SEM"),
    ]
    ax.legend(handles=leg_handles, fontsize=7.5, framealpha=0.9,
              loc="upper right")

    plt.tight_layout()
    pdf_path = out_path.replace(".png", ".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved figure: {pdf_path}")
    print(f"Saved figure: {out_path}")
    plt.close(fig)


# ── overlapping ECDF figure ───────────────────────────────────────────────────

def _draw_ecdf_panel(ax, animal_df, sos_stat, n2_rt_geom_s):
    """Draw overlapping ECDFs onto ax. Canonical colors; significant genotypes labeled."""
    stat = sos_stat.set_index("genotype")
    sig  = {g for g in stat.index
            if g != "N2" and not pd.isna(stat.loc[g, "q"]) and stat.loc[g, "q"] < 0.05}
    sig_order = (stat.loc[list(sig), "time_ratio"].sort_values().index.tolist()
                 if sig else [])

    for g in animal_df["genotype"].unique():
        if g == "N2":
            continue
        vals = (animal_df[animal_df["genotype"] == g]["response_time"]
                .dropna().sort_values().values)
        if len(vals) == 0:
            continue
        n      = len(vals)
        ecdf_x = np.concatenate([[0], vals, [NR_CEILING]])
        ecdf_y = np.concatenate([[0], np.arange(1, n + 1) / n, [1.0]])

        if g in sig:
            color = _geno_color(g)
            label = f"{DISPLAY_NAME.get(g, g)} {_stars(stat.loc[g, 'q'])}"
            ax.step(ecdf_x, ecdf_y, where="post",
                    color=color, lw=1.4, alpha=0.85, label=label, zorder=3)
        else:
            ax.step(ecdf_x, ecdf_y, where="post",
                    color="lightgray", lw=0.7, alpha=0.5, zorder=1)

    # N2 on top
    n2_vals = (animal_df[animal_df["genotype"] == "N2"]["response_time"]
               .dropna().sort_values().values)
    n2_x = np.concatenate([[0], n2_vals, [NR_CEILING]])
    n2_y = np.concatenate([[0], np.arange(1, len(n2_vals) + 1) / len(n2_vals), [1.0]])
    ax.step(n2_x, n2_y, where="post", color=_CANONICAL_COLORS["N2"], lw=2.0,
            alpha=0.9, label=f"N2  (geom mean {n2_rt_geom_s:.1f} s)", zorder=4)

    ax.axvline(NR_CEILING, color="gray", lw=0.6, ls=":", alpha=0.5)
    ax.set_xlabel("Response time (s)\nNR censored at 20 s", fontsize=9)
    ax.set_ylabel("Cumulative fraction", fontsize=9)
    ax.set_xlim(0, NR_CEILING + 0.5)
    ax.set_ylim(0, 1.02)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=7, framealpha=0.9, loc="lower right",
              title="Significant (q < 0.05)", title_fontsize=7)


def make_sos_ecdf_plot(animal_df, sos_stat, n2_rt_geom_s, out_path,
                       title="33% octanol response time — all genotypes"):
    fig, ax = plt.subplots(figsize=(7, 5))
    _draw_ecdf_panel(ax, animal_df, sos_stat, n2_rt_geom_s)
    ax.set_title(title, fontsize=10)
    plt.tight_layout()
    pdf_path = out_path.replace(".png", ".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved figure: {pdf_path}")
    print(f"Saved figure: {out_path}")
    plt.close(fig)


def make_combined_supplemental(animal_df, recording_df, sos_stat, speed_stat,
                                n2_rt_geom_s, n2_speed_mm, out_path,
                                title="33% octanol response time"):
    """
    3-panel supplemental figure on 8.5 × 11:
      Left  (full height): strip plot of per-trial response time
      Top right:           overlapping ECDF (all genotypes)
      Bottom right:        speed vs response-time correlation
    """
    import matplotlib.lines as mlines

    stat    = sos_stat.set_index("genotype")
    order   = stat.sort_values("time_ratio").index.tolist()
    present = set(animal_df["genotype"].unique())
    order   = [g for g in order if g in present and g != "N2"]
    ratios  = [stat.loc[g, "time_ratio"] if g in stat.index else np.nan for g in order]
    n2_pos  = next((i for i, r in enumerate(ratios) if not pd.isna(r) and r >= 1.0),
                   len(order))
    order.insert(n2_pos, "N2")

    n_geno = len(order)
    ytick  = {g: i for i, g in enumerate(order)}

    n2_by_date = (animal_df[animal_df["genotype"] == "N2"]
                  .groupby("date")["response_time"].median().to_dict())
    n2_grand   = animal_df[animal_df["genotype"] == "N2"]["response_time"].median()

    rec = recording_df.copy()
    rec["n2_ref"]       = rec["date"].map(n2_by_date).fillna(n2_grand)
    rec["rt_norm"]      = rec["median_rt"] / rec["n2_ref"]
    rec["date_matched"] = rec["date"].isin(n2_by_date)

    fig = plt.figure(figsize=(8.5, 11))
    gs  = fig.add_gridspec(
        2, 2,
        width_ratios=[1.4, 1],
        height_ratios=[1, 1],
        left=0.17, right=0.97, top=0.96, bottom=0.05,
        wspace=0.30, hspace=0.35)
    ax_strip = fig.add_subplot(gs[:, 0])   # left column, full height
    ax_ecdf  = fig.add_subplot(gs[0, 1])   # top right
    ax_corr  = fig.add_subplot(gs[1, 1])   # bottom right

    fig.suptitle(title, fontsize=11, y=0.99)

    # ── panel labels ──────────────────────────────────────────────────────────
    for ax, lbl in [(ax_strip, "A"), (ax_ecdf, "B"), (ax_corr, "C")]:
        ax.text(-0.18, 1.02, lbl, transform=ax.transAxes,
                fontsize=12, fontweight="bold", va="bottom", ha="left")

    # ── strip plot ────────────────────────────────────────────────────────────
    for _, row in rec.iterrows():
        g = row["genotype"]
        if g not in ytick or pd.isna(row["rt_norm"]):
            continue
        y, x = ytick[g], row["rt_norm"]
        if g == "N2":
            ax_strip.plot(x, y, "o", mfc=_DOT_N2, mec=_DOT_N2,
                          ms=5, alpha=0.65, lw=0, zorder=2)
        else:
            ec = _DOT_MATCHED if row["date_matched"] else _DOT_UNMATCHED
            fc = ec if row["date_matched"] else "none"
            ax_strip.plot(x, y, "o", mfc=fc, mec=ec, ms=5, alpha=0.65, lw=0, zorder=2)

    for g in order:
        y = ytick[g]
        ax_strip.axhline(y - 0.5, color="lightgray", lw=0.3, zorder=0)
        if g == "N2":
            vals = rec[rec["genotype"] == "N2"]["rt_norm"].dropna()
            if vals.empty:
                continue
            ctr = vals.mean()
            sem = vals.sem() if len(vals) > 1 else 0.0
            ax_strip.plot([ctr - sem, ctr + sem], [y, y], color=_DOT_N2,
                          lw=2.5, solid_capstyle="round", zorder=4)
            ax_strip.plot(ctr, y, "D", color=_DOT_N2, ms=7, zorder=5, mec="white", mew=0.5)
            ax_strip.text(ctr, y + 0.19, f"{n2_rt_geom_s:.1f} s", fontsize=6.5,
                          va="bottom", ha="center", color=_DOT_N2, zorder=7)
            continue
        if g not in stat.index or pd.isna(stat.loc[g, "time_ratio"]):
            continue
        r      = stat.loc[g]
        ctr    = r["time_ratio"]
        lo, hi = r["time_ratio_lo"], r["time_ratio_hi"]
        s      = _stars(r.get("q", np.nan))
        if s:
            ax_strip.text(hi + 0.05, y, s, fontsize=9, va="center", ha="left",
                          color="#111111", fontweight="bold", zorder=6)
        ax_strip.plot([lo, hi], [y, y], color=_DIAMOND_MUT, lw=2.5,
                      solid_capstyle="round", zorder=4)
        ax_strip.plot(ctr, y, "D", color=_DIAMOND_MUT, ms=7, zorder=5, mec="white", mew=0.5)
        ax_strip.text(ctr, y + 0.19, f"{ctr:.2f}", fontsize=6.5, va="bottom",
                      ha="center", color=_DIAMOND_MUT, zorder=7)

    ax_strip.axvline(1.0, color="gray", lw=0.8, ls="--", alpha=0.5, zorder=0)
    ax_strip.set_xlabel("Response time (ratio vs N2)", fontsize=9)
    ax_strip.set_xlim(0, 3.0)
    ax_strip.set_ylim(-0.8, n_geno - 0.2)
    ax_strip.set_yticks(list(ytick.values()))
    ax_strip.set_yticklabels([DISPLAY_NAME.get(g, g) for g in order], fontsize=7.5)
    ax_strip.spines[["top", "right"]].set_visible(False)
    ax_strip.tick_params(axis="x", labelsize=8)
    leg_handles = [
        mlines.Line2D([], [], color=_DOT_MATCHED,   marker="o", ls="none",
                      label="Trial — same-date N2"),
        mlines.Line2D([], [], color=_DOT_UNMATCHED, marker="o", mfc="none", ls="none",
                      label="Trial — grand-mean N2"),
        mlines.Line2D([], [], color=_DIAMOND_MUT,   marker="D", ls="none",
                      mec="white", mew=0.5, label="LME estimate ± 95% CI"),
        mlines.Line2D([], [], color=_DOT_N2,        marker="D", ls="none",
                      mec="white", mew=0.5, label="N2 mean ± SEM"),
    ]
    ax_strip.legend(handles=leg_handles, fontsize=7, framealpha=0.9, loc="upper right")

    # ── ECDF ─────────────────────────────────────────────────────────────────
    _draw_ecdf_panel(ax_ecdf, animal_df, sos_stat, n2_rt_geom_s)

    # ── correlation ───────────────────────────────────────────────────────────
    _draw_correlation_panel(ax_corr, sos_stat, speed_stat, n2_rt_geom_s, n2_speed_mm)

    pdf_path = out_path.replace(".png", ".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved figure: {pdf_path}")
    print(f"Saved figure: {out_path}")
    plt.close(fig)


# ── correlation plot ──────────────────────────────────────────────────────────

def _draw_correlation_panel(ax, sos_stat, speed_stat, n2_rt_geom_s, n2_speed_mm):
    """Draw speed vs response-time correlation onto ax. Returns (pr, pp, sr, sp)."""
    speed = speed_stat[speed_stat["metric"] == "speed"].set_index("genotype")
    sos   = sos_stat.set_index("genotype")

    xs, ys, labels = [], [], []
    x_lo, x_hi, y_lo, y_hi = [], [], [], []
    for g in sorted(set(speed.index) & set(sos.index) - {"N2"}):
        fc_s = speed.loc[g, "fold_change"]
        tr_r = sos.loc[g, "time_ratio"]
        if pd.isna(fc_s) or pd.isna(tr_r):
            continue
        xs.append(fc_s); ys.append(tr_r)
        labels.append(DISPLAY_NAME.get(g, g))
        x_lo.append(speed.loc[g, "fc_lo"] if "fc_lo" in speed.columns else fc_s)
        x_hi.append(speed.loc[g, "fc_hi"] if "fc_hi" in speed.columns else fc_s)
        y_lo.append(sos.loc[g, "time_ratio_lo"] if "time_ratio_lo" in sos.columns else tr_r)
        y_hi.append(sos.loc[g, "time_ratio_hi"] if "time_ratio_hi" in sos.columns else tr_r)

    xs   = np.array(xs);   ys   = np.array(ys)
    x_lo = np.array(x_lo); x_hi = np.array(x_hi)
    y_lo = np.array(y_lo); y_hi = np.array(y_hi)
    n    = len(xs)

    pearson_r,  pearson_p  = sp_stats.pearsonr(xs, ys)
    spearman_r, spearman_p = sp_stats.spearmanr(xs, ys)

    loo_label, loo_pr, loo_pp = None, pearson_r, pearson_p
    for i, lab in enumerate(labels):
        mask = np.ones(n, dtype=bool); mask[i] = False
        pr_i, pp_i = sp_stats.pearsonr(xs[mask], ys[mask])
        if abs(pr_i - pearson_r) > abs(loo_pr - pearson_r):
            loo_label, loo_pr, loo_pp = lab, pr_i, pp_i

    ax.errorbar(xs, ys,
                xerr=[xs - x_lo, x_hi - xs],
                yerr=[ys - y_lo, y_hi - ys],
                fmt="none", ecolor="#4393c3", elinewidth=0.8, capsize=2,
                alpha=0.2, zorder=2)
    ax.scatter(xs, ys, s=30, color="#4393c3", zorder=3, alpha=0.85)

    texts = [ax.text(x, y, lab, fontsize=4.5, ha="center", va="bottom",
                     color="#333333", zorder=4)
             for x, y, lab in zip(xs, ys, labels)]
    if _HAS_ADJUSTTEXT:
        _adjust_text(texts, ax=ax,
                     arrowprops=dict(arrowstyle="-", color="#aaaaaa", lw=0.4),
                     expand=(1.15, 1.3), force_text=(0.05, 0.08))

    m, b = np.polyfit(xs, ys, 1)
    xfit = np.linspace(xs.min() - 0.05, xs.max() + 0.05, 100)
    ax.plot(xfit, m * xfit + b, color="#b2182b", lw=1, ls="--", alpha=0.7)
    ax.axhline(1, color="gray", lw=0.6, ls=":", alpha=0.5)
    ax.axvline(1, color="gray", lw=0.6, ls=":", alpha=0.5)

    n2_speed_um = n2_speed_mm * 1000
    ax.set_xlabel(
        f"Forward-run speed (fold-change vs N2)\nN2 mean = {n2_speed_um:.0f} µm/s",
        fontsize=8)
    ax.set_ylabel(
        f"Response time (ratio vs N2)\nN2 geometric mean = {n2_rt_geom_s:.1f} s",
        fontsize=8)

    r2 = pearson_r ** 2
    loo_line = (f"\nexcl. {loo_label}: r = {loo_pr:.2f}  p = {loo_pp:.3f}"
                if loo_label else "")
    stats_txt = (f"Pearson  r = {pearson_r:.2f}  R² = {r2:.2f}  p = {pearson_p:.3f}\n"
                 f"Spearman ρ = {spearman_r:.2f}  p = {spearman_p:.3f}\n"
                 f"n = {n} genotypes{loo_line}")
    ax.text(0.97, 0.97, stats_txt, transform=ax.transAxes, fontsize=6.5,
            va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="lightgray", alpha=0.9))
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=7)
    return pearson_r, pearson_p, spearman_r, spearman_p


def make_correlation_plot(sos_stat, speed_stat, n2_rt_geom_s, n2_speed_mm,
                          out_path,
                          title="Forward-run speed vs. stimulus response time"):
    fig, ax = plt.subplots(figsize=(6, 5))
    pr, pp, sr, sp_ = _draw_correlation_panel(
        ax, sos_stat, speed_stat, n2_rt_geom_s, n2_speed_mm)
    ax.set_title(title, fontsize=10)
    plt.tight_layout()
    pdf_path = out_path.replace(".png", ".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved figure: {pdf_path}")
    print(f"Saved figure: {out_path}")
    plt.close(fig)
    return pr, pp, sr, sp_


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default=None,
                        help="Directory containing raw SOS CSV files")
    parser.add_argument("--out-dir",  default="data")
    parser.add_argument("--locomotion-dir", default="../locomotion/data",
                        help="Directory containing postural_comparison_stats.csv and "
                             "recording_data.parquet from the locomotion subproject")
    parser.add_argument("--refresh",  action="store_true",
                        help="Force re-scan even if cache exists")
    parser.add_argument("--refit",    action="store_true",
                        help="Re-run R/lme4 even if stats cache exists")
    parser.add_argument("--title",    default="Forward-run speed vs. stimulus response time")
    args = parser.parse_args()
    if args.refresh:
        args.refit = True
    if args.refresh and not args.data_dir:
        parser.error("--refresh requires --data-dir")

    os.makedirs(args.out_dir, exist_ok=True)

    animal_cache    = os.path.join(args.out_dir, "sos_animal_data.parquet")
    recording_cache = os.path.join(args.out_dir, "sos_recording_data.parquet")
    stats_cache     = os.path.join(args.out_dir, "sos_stats.csv")

    # ── load or rescan ────────────────────────────────────────────────────────
    if os.path.exists(animal_cache) and not args.refresh:
        print("Loading cached SOS data…")
        animal_df    = pd.read_parquet(animal_cache)
        recording_df = pd.read_parquet(recording_cache)
        print(f"  {len(animal_df):,} animals, {animal_df['genotype'].nunique()} genotypes")
    else:
        if not args.data_dir:
            parser.error("No cache found — supply --data-dir to scan raw SOS files")
        print(f"Scanning SOS data from {args.data_dir}…")
        animal_df    = load_sos_data(args.data_dir)
        recording_df = make_recording_summary(animal_df)
        animal_df.to_parquet(animal_cache,    index=False)
        recording_df.to_parquet(recording_cache, index=False)
        print(f"  Cached to {animal_cache}")

    n2_rt_mean   = animal_df[animal_df["genotype"] == "N2"]["response_time"].mean()
    n2_rt_median = animal_df[animal_df["genotype"] == "N2"]["response_time"].median()
    # Always compute geometric mean from raw data (not arithmetic mean or emmeans)
    n2_rt_geom   = np.exp(np.log(
        animal_df[animal_df["genotype"] == "N2"]["response_time"]).mean())
    print(f"\nN2: mean {n2_rt_mean:.2f} s, median {n2_rt_median:.2f} s, "
          f"geom mean {n2_rt_geom:.2f} s "
          f"({(animal_df['genotype'] == 'N2').sum()} animals, "
          f"{animal_df[animal_df['genotype'] == 'N2']['is_nr'].sum()} NR)")

    # ── LME ───────────────────────────────────────────────────────────────────
    if os.path.exists(stats_cache) and not args.refit:
        print("\nLoading cached LME stats (use --refit to rerun)…")
        stat_df = pd.read_csv(stats_cache)
    else:
        print("\nFitting log-normal LME via R/lme4 + emmeans…")
        stat_df, means_df, _ = fit_lme_sos(animal_df, args.out_dir)
        stat_df.to_csv(stats_cache, index=False, float_format="%.4f")
        print(f"Saved stats: {stats_cache}")

    # ── speed LME stats and N2 speed ─────────────────────────────────────────
    speed_stats_path = os.path.join(args.locomotion_dir, "postural_comparison_stats.csv")
    if not os.path.exists(speed_stats_path):
        sys.exit(f"Speed stats not found at {speed_stats_path}\n"
                 f"Run locomotion/batch_postural_comparison.py first, or pass --locomotion-dir.")
    speed_stat = pd.read_csv(speed_stats_path)

    rec_path = os.path.join(args.locomotion_dir, "recording_data.parquet")
    if os.path.exists(rec_path):
        rec_df = pd.read_parquet(rec_path)
        n2_speed_mm = rec_df[rec_df["genotype"] == "N2"]["speed"].mean()
    else:
        n2_speed_mm = np.nan

    # ── Supplemental figure: strip + ECDF + correlation (8.5 × 11) ──────────
    supp_out = os.path.join(args.out_dir, "supplemental_sos_combined.png")
    make_combined_supplemental(
        animal_df, recording_df, stat_df, speed_stat,
        n2_rt_geom_s=n2_rt_geom,
        n2_speed_mm=n2_speed_mm,
        out_path=supp_out,
        title="33% octanol response time")

    # ── correlation plot ───────────────────────────────────────────────────────
    fig_out = os.path.join(args.out_dir, "speed_sos_correlation.png")
    pr, pp, sr, sp_ = make_correlation_plot(
        stat_df, speed_stat,
        n2_rt_geom_s=n2_rt_geom,
        n2_speed_mm=n2_speed_mm,
        out_path=fig_out, title=args.title)

    # ── summary CSV ───────────────────────────────────────────────────────────
    speed_s = speed_stat[speed_stat["metric"] == "speed"].set_index("genotype")
    sos_s   = stat_df.set_index("genotype")
    rec_g   = recording_df.groupby("genotype").agg(
        n_recordings=("date", "nunique"),
        median_rt=("median_rt", "median")).reset_index()

    rows = []
    for g in sorted(set(speed_s.index) | set(sos_s.index)):
        row = {"genotype": g}
        if g in speed_s.index:
            row["speed_fc"] = speed_s.loc[g, "fold_change"]
            row["speed_q"]  = speed_s.loc[g, "q"]
        if g in sos_s.index:
            row["rt_ratio"] = sos_s.loc[g, "time_ratio"]
            row["rt_q"]     = sos_s.loc[g, "q"] if "q" in sos_s.columns else np.nan
        r = rec_g[rec_g["genotype"] == g]
        if not r.empty:
            row["n_recordings"] = r["n_recordings"].iloc[0]
            row["median_rt_s"]  = r["median_rt"].iloc[0]
        rows.append(row)

    summary = pd.DataFrame(rows)
    summary_path = os.path.join(args.out_dir, "speed_sos_summary.csv")
    summary.to_csv(summary_path, index=False, float_format="%.4f")
    print(f"Saved summary: {summary_path}")

    n_corr = sum(
        1 for g in set(speed_s.index) & set(sos_s.index) - {"N2"}
        if not pd.isna(speed_s.loc[g, "fold_change"])
        and not pd.isna(sos_s.loc[g, "time_ratio"])
    )
    print(f"\nCorrelation (n = {n_corr} genotypes):")
    print(f"  Pearson  r = {pr:.3f}  p = {pp:.4f}")
    print(f"  Spearman ρ = {sr:.3f}  p = {sp_:.4f}")


if __name__ == "__main__":
    main()
