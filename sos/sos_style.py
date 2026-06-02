"""
Shared aesthetics, layout constants, and drawing helpers for all SOS figures.

Import this module in each SOS figure script. Changing a value here changes it
across all figures at once.
"""

import matplotlib
import matplotlib.lines as mlines
import numpy as np
import pandas as pd

# ── Global style ──────────────────────────────────────────────────────────────
matplotlib.rcParams["pdf.fonttype"]    = 42
matplotlib.rcParams["ps.fonttype"]     = 42
matplotlib.rcParams["font.family"]     = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]

NR_CEILING = 20.0

# Strain IDs that should NOT be italicised (proper names, not gene names)
_STRAIN_IDS = {"N2"}

# ── Layout constants ──────────────────────────────────────────────────────────
# All dimensions in inches. Edit here to update ALL SOS figures at once.
#
# Design rules (Malaiwong & Oglu 2026):
#   - All bars have the same physical width (set by BAR_SCALE).
#   - All ECDF panels are the same physical size (ECDF_W × ECDF_H).
#   - Figure width expands with more bars; ECDF block stays fixed.

L_MAR     = 0.76   # left margin: y-label + tick labels
R_MAR     = 0.17   # right margin
T_MAR     = 0.30   # top margin
B_FLAT    = 0.54   # bottom margin, flat x-labels
B_ROT     = 0.72   # bottom margin, rotated x-labels
BAR_SCALE = 0.68   # barplot inches per x data unit — fixes physical bar width
ECDF_W    = 1.543  # ECDF panel width
ECDF_H    = 0.847  # ECDF panel height
ECDF_VGAP = 0.466  # vertical gap between ECDF rows
ECDF_HGAP = 0.694  # horizontal gap between ECDF columns (2×2 grid)
CONTENT_H = 2 * ECDF_H + ECDF_VGAP   # 2.160" — fixed content height
LGD_W     = 1.20   # legend column width (when present)
BAR_LGD   = 0.20   # barplot right → legend left gap
LGD_ECDF  = 0.20   # legend right → ECDF left gap
BAR_ECDF     = 0.20   # barplot right → ECDF left gap, flat x-labels
BAR_ECDF_ROT = 0.42   # barplot right → ECDF left gap, rotated x-labels (extra room for panel letters)
BAR_W     = 0.16   # bar width in data units

# ── Canonical genotype colors ─────────────────────────────────────────────────
COLORS = {
    "N2":                   "#888888",
    "WT":                   "#888888",
    "cest-2.1":             "#984F9F",
    "tbh-1":                "#8E3E29",
    "tbh-1; cest-2.1":      "#8E3E29",
    "tdc-1":                "#C15C3E",
    "tdc-1; tbh-1":         "#C15C3E",
    "bas-1":                "#1A2C7A",
    "bas-1; cest-2.1":      "#1A2C7A",
    "tph-1":                "#6699CC",
    "tph-1; cest-2.1":      "#6699CC",
    "glo-1":                "#CC6677",
    "fcmt-1":               "#44AA99",
    "cat-1":                "#332288",
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

# Canonical genotype ordering (for sort rank)
CANONICAL_ORDER = list(COLORS.keys())


# ── Helper functions ──────────────────────────────────────────────────────────

def stars(p):
    """BH-adjusted p-value → significance star string."""
    if pd.isna(p): return ""
    if p < 0.001:  return "***"
    if p < 0.01:   return "**"
    if p < 0.05:   return "*"
    return ""


def make_ax(fig, fig_w, fig_h, l, b, w, h):
    """Add an axes using inch coordinates (l/b/w/h in inches)."""
    return fig.add_axes([l / fig_w, b / fig_h, w / fig_w, h / fig_h])


def _apply_italic(text_objects):
    """Italicise text objects whose label is a gene name (not a strain ID)."""
    for t in text_objects:
        if t.get_text() not in _STRAIN_IDS:
            t.set_fontstyle("italic")


def set_xticklabels(ax, labels, **kw):
    """Set x-tick labels with gene names in italic, strain IDs upright."""
    ticks = ax.set_xticklabels(labels, **kw)
    _apply_italic(ticks)
    return ticks


def draw_ecdf(ax, animal_df, geno_list, geno_colors,
              title="", geno_col="Genotype", geno_ls=None):
    """Draw overlapping ECDF step curves for the listed genotypes.

    Parameters
    ----------
    geno_col : column in animal_df containing the genotype label
    geno_ls  : dict mapping genotype → linestyle (default solid for all)
    """
    for g in geno_list:
        vals = (animal_df[animal_df[geno_col] == g]["Response.time"]
                .dropna().sort_values().values)
        if len(vals) == 0:
            continue
        n  = len(vals)
        x  = np.concatenate([[0], vals, [NR_CEILING]])
        y  = np.concatenate([[0], np.arange(1, n + 1) / n, [1.0]])
        ls = geno_ls.get(g, "-") if geno_ls else "-"
        ax.step(x, y, where="post",
                color=geno_colors.get(g, "#aaaaaa"), lw=1.0, ls=ls, label=g)

    ax.axvline(NR_CEILING, color="gray", lw=0.6, ls=":", alpha=0.5)
    ax.set_xlabel("Response time (s)", fontsize=6.5)
    ax.set_ylabel("Cumulative fraction", fontsize=6.5)
    ax.set_xlim(0, NR_CEILING + 0.3)
    ax.set_ylim(0, 1.02)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=6)
    ax.grid(which="major", axis="both", color="0.85", lw=0.4, zorder=0)
    ax.minorticks_on()
    ax.grid(which="minor", axis="both", color="0.92", lw=0.25, zorder=0)
    ax.set_axisbelow(True)
    lgd = ax.legend(fontsize=5.5, framealpha=0.9, loc="lower right")
    _apply_italic(lgd.get_texts())
    ax.set_title(title, fontsize=7, pad=2)


def style_ecdf_ax(ax, title=""):
    """Apply standard ECDF axis styling after curves have been drawn.

    Use this when the caller draws its own step curves (e.g. rescue figures
    with Strain_ID-level filtering) but still wants consistent ECDF styling.
    """
    ax.axvline(NR_CEILING, color="gray", lw=0.6, ls=":", alpha=0.5)
    ax.set_xlabel("Response time (s)", fontsize=6.5)
    ax.set_ylabel("Cumulative fraction", fontsize=6.5)
    ax.set_xlim(0, NR_CEILING + 0.3)
    ax.set_ylim(0, 1.02)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=6)
    ax.grid(which="major", axis="both", color="0.85", lw=0.4, zorder=0)
    ax.minorticks_on()
    ax.grid(which="minor", axis="both", color="0.92", lw=0.25, zorder=0)
    ax.set_axisbelow(True)
    lgd = ax.legend(fontsize=5.5, framealpha=0.9, loc="lower right")
    if lgd:
        _apply_italic(lgd.get_texts())
    if title:
        ax.set_title(title, fontsize=7, pad=2)


def dot_legend_handles():
    """Standard handles for the individual-animal / NR / daily-median legend."""
    return [
        mlines.Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
                      markersize=4, label="Individual animal"),
        mlines.Line2D([0], [0], marker="o", color="w", markerfacecolor="none",
                      markeredgecolor="gray", markersize=4,
                      label="NR (censored 20 s)"),
        mlines.Line2D([0], [0], marker="o", color="w", markerfacecolor="black",
                      markersize=6, label="Daily median"),
    ]
