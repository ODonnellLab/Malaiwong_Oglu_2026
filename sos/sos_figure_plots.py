"""
Per-figure SOS barplots + ECDFs for Malaiwong & Oglu 2026.

Generates one figure per *SOSdata*.csv (excluding 1c_ multi-condition files
and superseded originals). Figure width is computed from the number of bars;
ECDF panel dimensions match sos_condition_plots.py exactly (sos_style.py).

Usage:
    python sos_figure_plots.py --data-dir /path/to/raw_figure_data \\
                               --out-dir data/sos_figures/
    python sos_figure_plots.py --data-dir /path/to/raw_figure_data \\
                               --out-dir data/sos_figures/ \\
                               --file 1d_cest-1.2_SOSdata.csv
"""

import argparse
import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.transforms import blended_transform_factory

import sos_style
from sos_style import NR_CEILING

# Center-to-center spacing between adjacent bars (data units).
# Kept tight — same order of magnitude as within-cluster spacing in
# sos_condition_plots.py. Rescue/paired figures still get x-axis breaks
# between groups, but bar density is uniform throughout.
_BAR_STEP = 0.22
_X_PAD    = 0.20   # margin beyond first/last bar


GENOTYPE_NORMALIZE = {
    "Wild type":            "N2",
    "WT":                   "N2",
    "SLC-17.1":             "slc-17.1",
    "slc-17.1 backcrossed": "slc-17.1",
    "fcmt-1 (MOY113)":      "fcmt-1",
    "octr-; ser-6":         "octr-1; ser-6",
    "octr-; ser-3; ser-6":  "octr-1; ser-3; ser-6",
}


def _normalize(genotype):
    return GENOTYPE_NORMALIZE.get(genotype, genotype)


def _parent_genotype(genotype):
    g = _normalize(genotype)
    m = re.match(r'^(.+?)\s+(?:backcrossed\s+)?rescue\s+in\s+CAN(?:\s+line\s+\d+)?$',
                 g, re.IGNORECASE)
    if m:
        return _normalize(m.group(1).strip())
    m = re.match(r'^(.+?)_([A-Z]{2,4})$', g)
    if m and m.group(2) in {"RIC", "CAN", "RC", "GUT", "INT"}:
        return _normalize(m.group(1))
    return g


def _display_label(genotype):
    g = _normalize(genotype)
    m = re.match(r'^(.+?)\s+rescue\s+in\s+CAN\s+line\s+(\d+)$', g, re.IGNORECASE)
    if m:
        return f"{_normalize(m.group(1).strip())} [CAN L{m.group(2)}]"
    m = re.match(r'^(.+?)\s+backcrossed\s+rescue\s+in\s+CAN\s+line\s+(\d+)$',
                 g, re.IGNORECASE)
    if m:
        return f"{_normalize(m.group(1).strip())} bc [CAN L{m.group(2)}]"
    m = re.match(r'^(.+?)\s+rescue\s+in\s+CAN$', g, re.IGNORECASE)
    if m:
        return f"{_normalize(m.group(1).strip())} [CAN]"
    m = re.match(r'^(.+?)_([A-Z]{2,4})$', g)
    if m and m.group(2) in {"RIC", "CAN", "RC", "GUT", "INT"}:
        return f"{_normalize(m.group(1))} [{m.group(2)}]"
    return g


def _geno_color(genotype):
    parent = _parent_genotype(genotype)
    return sos_style.COLORS.get(parent,
           sos_style.COLORS.get(genotype, "#AAAAAA"))


# ── data loading ──────────────────────────────────────────────────────────────

def load_sos_csv(csv_path):
    df = pd.read_csv(csv_path)
    if "no. of NR" in df.columns and "NR" not in df.columns:
        df = df.rename(columns={"no. of NR": "NR"})
    multi_bact = len(df["Bacteria"].dropna().unique()) > 1
    if not multi_bact:
        df = df[df["Bacteria"] == "OP50"].copy()

    nr_counts = (
        df[df["NR"].notna()]
        .groupby(["Strain_ID", "Genotype", "Condition", "Date", "Odor"])["NR"]
        .max().reset_index().rename(columns={"NR": "nr_count"})
    )
    nr_counts["nr_count"] = nr_counts["nr_count"].clip(lower=0).astype(int)

    rt = df[df["Response.time"].notna()].copy()
    rt = rt.drop_duplicates(subset=["Strain_ID", "Genotype", "Date",
                                    "Condition", "Odor", "Response.time"])
    rt["is_nr"] = False

    nr_rows = []
    for _, row in nr_counts.iterrows():
        for _ in range(row["nr_count"]):
            nr_rows.append({
                "Strain_ID": row["Strain_ID"], "Genotype": row["Genotype"],
                "Condition": row["Condition"], "Date": row["Date"],
                "Odor": row["Odor"], "Response.time": NR_CEILING, "is_nr": True,
            })
    if nr_rows:
        rt = pd.concat([rt, pd.DataFrame(nr_rows)], ignore_index=True)

    rt["norm_genotype"] = rt["Genotype"].apply(_normalize)
    rt["display_label"] = rt["Genotype"].apply(_display_label)
    rt["is_rescue"] = rt["Condition"].str.lower().str.strip() == "rescue"

    if multi_bact:
        rt["_cond"] = rt["Condition"].str.strip()
        rt["Strain_ID"] = rt["Strain_ID"] + "__" + rt["_cond"]
        rt["display_label"] = rt.apply(
            lambda r: r["display_label"] if r["_cond"].lower() == "control"
                      else f"{r['display_label']} [{r['_cond']}]", axis=1)
        rt["is_rescue"] = False

    return rt


def load_lmm_stats(stats_path):
    if not os.path.exists(stats_path):
        return {}
    df = pd.read_csv(stats_path)
    result = {}
    if "comparison_type" in df.columns:
        within = df[df["comparison_type"] == "within_condition"]
        for _, row in within.iterrows():
            contrast = str(row.get("contrast", ""))
            cond = str(row.get("Condition", "")).strip()
            parts = [p.strip() for p in contrast.split(" - ")]
            if len(parts) != 2:
                continue
            result[frozenset({f"{parts[0]}__{cond}", f"{parts[1]}__{cond}"})] = row.get("p.adj.BH", np.nan)
        return result
    for _, row in df.iterrows():
        contrast = str(row.get("contrast", ""))
        parts = [p.strip() for p in contrast.split(" - ")]
        if len(parts) != 2:
            continue
        result[frozenset(parts)] = row.get("p.adj.BH", np.nan)
    return result


# ── genotype ordering ─────────────────────────────────────────────────────────

def _genotype_order(animal_df):
    strains = (animal_df[["Strain_ID", "display_label", "norm_genotype", "is_rescue"]]
               .drop_duplicates("Strain_ID"))

    def _rank(row):
        if row["is_rescue"]:
            parent = _parent_genotype(row["norm_genotype"])
            try:
                return sos_style.CANONICAL_ORDER.index(parent) + 0.5
            except ValueError:
                return len(sos_style.CANONICAL_ORDER) + 0.5
        try:
            return sos_style.CANONICAL_ORDER.index(row["norm_genotype"])
        except ValueError:
            return len(sos_style.CANONICAL_ORDER)

    def _cond_rank(sid):
        if "__" in str(sid):
            return 0 if str(sid).split("__", 1)[1].lower() == "control" else 1
        return 0

    strains = strains.copy()
    strains["_cond_rank"] = strains["Strain_ID"].apply(_cond_rank)
    strains["_rank"]      = strains.apply(_rank, axis=1)
    strains = strains.sort_values(["_cond_rank", "_rank"])
    return list(strains.itertuples(index=False))


# ── config loading ────────────────────────────────────────────────────────────

def load_config(csv_path):
    """Load optional per-dataset YAML grouping config.

    Config file must be named <csv_stem>_config.yaml and live in the same
    directory as the CSV. Returns an empty dict if no config exists.
    """
    cfg_path = os.path.splitext(csv_path)[0] + "_config.yaml"
    if not os.path.exists(cfg_path):
        return {}
    import yaml
    with open(cfg_path) as f:
        return yaml.safe_load(f) or {}


# ── figure ────────────────────────────────────────────────────────────────────

def make_figure(csv_path, out_dir):
    S    = sos_style
    stem = os.path.splitext(os.path.basename(csv_path))[0]
    stats_path = csv_path.replace(".csv", "_LMMstats.csv")

    animal_df = load_sos_csv(csv_path)
    lmm_stats = load_lmm_stats(stats_path)
    order     = _genotype_order(animal_df)
    n_strains = len(order)

    odors  = sorted(animal_df["Odor"].dropna().unique())
    n_ecdf = len(odors)

    rng = np.random.default_rng(42)

    # Bar x positions — config-driven if available, otherwise flat uniform spacing
    cfg        = load_config(csv_path)
    cfg_groups = cfg.get("groups")           # list of lists of Strain_IDs, or None
    intra      = cfg.get("intra_spacing", _BAR_STEP)
    inter      = cfg.get("inter_spacing", _BAR_STEP * 2)

    sid_to_row = {r.Strain_ID: r for r in order}

    if cfg_groups:
        # Explicit ordering and spacing from config
        order = [sid_to_row[sid]
                 for group in cfg_groups
                 for sid in group
                 if sid in sid_to_row]
        x_pos = {}
        x = 0.0
        for gi, group in enumerate(cfg_groups):
            valid = [sid for sid in group if sid in sid_to_row]
            for bi, sid in enumerate(valid):
                x_pos[sid] = x
                if bi < len(valid) - 1:
                    x += intra
            if gi < len(cfg_groups) - 1 and valid:
                x += inter
    else:
        # Flat: tight uniform spacing, all bars in one group
        x_pos = {row.Strain_ID: i * intra for i, row in enumerate(order)}

    n_strains = len(order)
    x_lo = min(x_pos.values()) - _X_PAD
    x_hi = max(x_pos.values()) + _X_PAD

    # Figure dimensions — ECDF panels always ECDF_W × ECDF_H
    bar_w        = (x_hi - x_lo) * S.BAR_SCALE
    total_ecdf_h = n_ecdf * S.ECDF_H + max(0, n_ecdf - 1) * S.ECDF_VGAP
    content_h    = max(S.CONTENT_H, total_ecdf_h)
    fig_w        = S.L_MAR + bar_w + S.BAR_ECDF_ROT + S.ECDF_W + S.R_MAR
    fig_h        = S.T_MAR + content_h + S.B_ROT

    fig   = plt.figure(figsize=(fig_w, fig_h))
    ax_bar = S.make_ax(fig, fig_w, fig_h, S.L_MAR, S.B_ROT, bar_w, content_h)

    # ECDF axes — centered vertically within content area
    ecdf_l        = S.L_MAR + bar_w + S.BAR_ECDF_ROT
    center_offset = (content_h - total_ecdf_h) / 2
    ax_ecdf = []
    for i in range(n_ecdf):
        b = S.B_ROT + center_offset + (n_ecdf - 1 - i) * (S.ECDF_H + S.ECDF_VGAP)
        ax_ecdf.append(S.make_ax(fig, fig_w, fig_h, ecdf_l, b, S.ECDF_W, S.ECDF_H))

    # ── barplot ───────────────────────────────────────────────────────────────
    n2_sid = next((r.Strain_ID for r in order if r.norm_genotype == "N2"), "N2")

    for i, row in enumerate(order):
        sid   = row.Strain_ID
        geno  = row.norm_genotype
        lbl   = row.display_label
        is_r  = row.is_rescue
        color = _geno_color(geno)
        a_bar = 0.55 if is_r else 0.72
        a_dot = 0.20 if is_r else 0.25
        xc    = x_pos[sid]

        sub = animal_df[animal_df["Strain_ID"] == sid]
        if sub.empty:
            continue

        rt  = sub["Response.time"].values
        nr  = sub["is_nr"].values
        xs  = xc + rng.uniform(-0.06, 0.06, len(sub))
        ax_bar.scatter(xs[~nr], rt[~nr], s=4, color=color, alpha=a_dot, zorder=2, lw=0)
        if nr.any():
            ax_bar.scatter(xs[nr], rt[nr], s=5, facecolors="none",
                           edgecolors=color, alpha=0.45, zorder=2, lw=0.6)

        dm  = sub.groupby("Date")["Response.time"].median().values
        xsd = xc + rng.uniform(-0.04, 0.04, len(dm))
        ax_bar.scatter(xsd, dm, s=13, color="black", zorder=5, lw=0)

        m   = dm.mean()
        sem = dm.std(ddof=1) / np.sqrt(len(dm)) if len(dm) > 1 else 0.0
        ax_bar.bar(xc, m, width=S.BAR_W, color=color, alpha=a_bar,
                   zorder=3, edgecolor="none")
        ax_bar.errorbar(xc, m, yerr=sem, fmt="none",
                        color="black", capsize=2.5, lw=1.0, zorder=4)

        p = lmm_stats.get(frozenset({sid, n2_sid}), np.nan)
        s = S.stars(p)
        if s:
            ax_bar.text(xc, NR_CEILING * 1.09, s, ha="center", va="bottom",
                        fontsize=8, fontweight="bold", color="black", zorder=6)

    # Dashed separator between condition groups (multi-bacteria figures)
    prev_cond = None
    for row in order:
        cond = str(row.Strain_ID).split("__", 1)[1] if "__" in str(row.Strain_ID) else None
        if prev_cond is not None and cond != prev_cond:
            ax_bar.axvline(x_pos[row.Strain_ID] - _BAR_STEP / 2,
                           color="gray", lw=0.8, ls="--", alpha=0.4, zorder=1)
        prev_cond = cond

    # Rescue vs parent significance brackets
    rescue_rows   = [(i, row) for i, row in enumerate(order) if row.is_rescue]
    ctrl_by_geno  = {r.norm_genotype: (r.Strain_ID, x_pos[r.Strain_ID])
                     for r in order if not r.is_rescue and r.norm_genotype != "N2"}
    bracket_y0 = NR_CEILING * 1.18
    bracket_dy = NR_CEILING * 0.09
    tick_h     = NR_CEILING * 0.025
    k_bracket  = 0
    for ri, row in rescue_rows:
        parent_geno = _parent_genotype(row.norm_genotype)
        if parent_geno not in ctrl_by_geno:
            continue
        parent_sid, px = ctrl_by_geno[parent_geno]
        p = lmm_stats.get(frozenset({row.Strain_ID, parent_sid}), np.nan)
        s = S.stars(p)
        if not s:
            continue
        rx = x_pos[row.Strain_ID]
        y  = bracket_y0 + k_bracket * bracket_dy
        ax_bar.plot([px, px, rx, rx], [y - tick_h, y, y, y - tick_h],
                    lw=0.8, color="black", clip_on=False)
        ax_bar.text((px + rx) / 2, y + 0.1, s, ha="center", va="bottom",
                    fontsize=8, fontweight="bold", color="black", zorder=6)
        k_bracket += 1

    ylim_top = NR_CEILING * max(1.38, 1.18 + max(k_bracket, 1) * 0.09 + 0.06)
    ax_bar.axhline(NR_CEILING, color="gray", lw=0.6, ls=":", alpha=0.5)
    ax_bar.set_xlim(x_lo, x_hi)
    ax_bar.set_ylim(0, ylim_top)

    # x-axis line: one segment per config group (with breaks between groups);
    # flat continuous line when no config.
    trans    = blended_transform_factory(ax_bar.transData, ax_bar.transAxes)
    seg_half = intra / 2 - 0.01          # segment ends just inside the step boundary
    if cfg_groups and len(cfg_groups) > 1:
        drawn = [[sid for sid in g if sid in x_pos] for g in cfg_groups]
        drawn = [g for g in drawn if g]
        for gi, grp in enumerate(drawn):
            lo = x_lo if gi == 0              else x_pos[grp[0]]  - seg_half
            hi = x_hi if gi == len(drawn) - 1 else x_pos[grp[-1]] + seg_half
            ax_bar.plot([lo, hi], [0, 0], color="black", lw=0.8,
                        clip_on=False, transform=trans)
    else:
        ax_bar.plot([x_lo, x_hi], [0, 0], color="black", lw=0.8,
                    clip_on=False, transform=trans)

    ax_bar.set_xticks(list(x_pos.values()))
    S.set_xticklabels(ax_bar, [row.display_label for row in order],
                      fontsize=7.5, rotation=40, ha="right")
    ax_bar.set_yticks([0, 5, 10, 15, 20])
    ax_bar.set_ylabel("Response time (s)", fontsize=9)
    ax_bar.spines[["top", "right", "bottom"]].set_visible(False)
    ax_bar.tick_params(axis="x", bottom=False, labelsize=7.5)
    ax_bar.tick_params(axis="y", labelsize=8)
    ax_bar.text(-0.08, 1.04, "A", transform=ax_bar.transAxes,
                fontsize=10, fontweight="bold", va="bottom")

    # ── ECDF panels ───────────────────────────────────────────────────────────
    panel_labels = "BCDEFGH"
    for pi, (odor, ax) in enumerate(zip(odors, ax_ecdf)):
        sub_odor = animal_df[animal_df["Odor"] == odor]
        for row in order:
            sid  = row.Strain_ID
            geno = row.norm_genotype
            cond = str(sid).split("__", 1)[1] if "__" in str(sid) else None
            vals = (sub_odor[sub_odor["Strain_ID"] == sid]["Response.time"]
                    .dropna().sort_values().values)
            if len(vals) == 0:
                continue
            n      = len(vals)
            ecdf_x = np.concatenate([[0], vals, [NR_CEILING]])
            ecdf_y = np.concatenate([[0], np.arange(1, n + 1) / n, [1.0]])
            color  = _geno_color(geno)
            alpha  = 0.55 if row.is_rescue else 0.9
            ls     = "--" if (cond and cond.lower() != "control") else "-"
            ax.step(ecdf_x, ecdf_y, where="post", color=color, lw=1.0,
                    alpha=alpha, ls=ls, label=row.display_label)

        title = odor if len(odors) > 1 else ""
        S.style_ecdf_ax(ax, title=title)
        ax.text(-0.22, 1.08, panel_labels[pi], transform=ax.transAxes,
                fontsize=8, fontweight="bold", va="bottom")

    os.makedirs(out_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        path = os.path.join(out_dir, f"{stem}.{ext}")
        kw = dict(bbox_inches="tight")
        if ext == "png":
            kw["dpi"] = 300
        fig.savefig(path, **kw)
        print(f"  Saved: {path}")
    plt.close(fig)


# ── main ──────────────────────────────────────────────────────────────────────

_SKIP_PREFIXES = ("1c_",)
_SKIP_EXACT    = {"2e_fcmt-1_SOSdata.csv",
                  "1e_Epistasis_SOSdata_updated.csv"}   # handled by sos_epistasis_plots.py


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", required=True,
                        help="Directory containing raw SOS CSV files")
    parser.add_argument("--out-dir",  default="data/figures")
    parser.add_argument("--file",     default=None,
                        help="Process a single named CSV file only")
    args = parser.parse_args()

    if args.file:
        files = [os.path.join(args.data_dir, args.file)]
    else:
        files = sorted(glob.glob(os.path.join(args.data_dir, "*SOSdata*.csv")))
        files = [f for f in files
                 if not re.search(r"_(LMMstats|LMMmeans|NRstats)\.csv$",
                                  os.path.basename(f))]
        files = [f for f in files
                 if not any(os.path.basename(f).startswith(p) for p in _SKIP_PREFIXES)]
        files = [f for f in files
                 if os.path.basename(f) not in _SKIP_EXACT]

    print(f"Processing {len(files)} file(s) → {args.out_dir}")
    for f in files:
        print(f"\n--- {os.path.basename(f)} ---")
        try:
            make_figure(f, args.out_dir)
        except Exception as e:
            import traceback
            print(f"  ERROR: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
