"""
blend_ap_axis_figure.py
=======================
AP-axis proximity profile: minimum distance to intestine as a function of
body position for CAN (L/R separately), PDE, HSN, and AVM.

Data source: data/nml_vertices/blend_neurite_nn_distances.csv
             data/nml_vertices/blend_world_verts.csv (intestine Y extent)

Produced by blend_intestine_proximity.py; run that script first.

Usage
-----
    python blend_ap_axis_figure.py [--out-dir data/nml_vertices]
                                   [--bin-size 10]
                                   [--threshold 10]

Outputs
-------
  ap_axis_proximity.png / .pdf
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── CLI ───────────────────────────────────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument('--out-dir',   default='data/nml_vertices')
ap.add_argument('--bin-size',  type=float, default=10.0,
                help='Y-axis bin width in µm (default 10)')
ap.add_argument('--threshold', type=float, default=10.0,
                help='Contact distance threshold in µm (default 10)')
args = ap.parse_args()
OUT       = Path(args.out_dir)
BIN_UM    = args.bin_size
THRESH_UM = args.threshold
BLEND_TO_UM = 114.8

# ── Load data ─────────────────────────────────────────────────────────────────
nn = pd.read_csv(OUT / 'blend_neurite_nn_distances.csv')
nn['y_um'] = nn['y'] * BLEND_TO_UM

verts = pd.read_csv(OUT / 'blend_world_verts.csv')
int_v = verts[verts['mesh'].str.match(r'^int\d')]
INT_Y_MIN = int_v['y'].min() * BLEND_TO_UM
INT_Y_MAX = int_v['y'].max() * BLEND_TO_UM

# Nerve ring reference line: anterior extent of RIH approximates the
# anterior boundary of the nerve ring.
NERVE_RING_Y = nn[nn['mesh'] == 'RIH']['y_um'].min()

# Y-range (in µm) where HSNR/AVM share vertices with PDEL/PDER
# Diagnosed in prior analysis — treat this zone as unreliable for those neurons
SHARED_Y_MIN = -285.0
SHARED_Y_MAX = -97.0

# ── Helper: AP-axis profile for one or more mesh names ───────────────────────
GAP_DOTTED_UM = 40.0   # runs of NaN wider than this are shown as dotted

def ap_profile(mesh_names, bin_um=BIN_UM, dedup=False):
    """
    Return (y_centers, d_min, is_interpolated) arrays for the given meshes.

    y_centers      : bin centre positions in µm
    d_min          : minimum distance to intestine within each bin (µm)
    is_interpolated: True only for bins that span a LARGE structural gap in the
                     neuron mesh (> GAP_DOTTED_UM).  Small NaN bins arising
                     from sparse face-centroid spacing (~20 µm inter-point
                     interval vs 10 µm bins) are filled silently and shown
                     solid — marking every one of them as dotted produces a
                     faint striped artifact rather than a readable profile.
    """
    sub = nn[nn['mesh'].isin(mesh_names)].copy()
    if dedup:
        sub = sub.drop_duplicates(subset=['x', 'y', 'z'])

    y_all = sub['y_um'].values
    d_all = sub['d_nn_um'].values

    y_min = np.floor(y_all.min() / bin_um) * bin_um
    y_max = np.ceil(y_all.max()  / bin_um) * bin_um
    edges   = np.arange(y_min, y_max + bin_um, bin_um)
    centers = (edges[:-1] + edges[1:]) / 2

    d_min = np.full(len(centers), np.nan)
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (y_all >= lo) & (y_all < hi)
        if mask.any():
            d_min[i] = d_all[mask].min()

    # Identify large structural gaps (runs of NaN > GAP_DOTTED_UM wide)
    nan_mask   = np.isnan(d_min)
    is_interp  = np.zeros(len(centers), dtype=bool)
    in_run     = False
    run_start  = 0
    for i, n in enumerate(nan_mask):
        if n and not in_run:
            run_start = i; in_run = True
        elif not n and in_run:
            run_width_um = (i - run_start) * bin_um
            if run_width_um > GAP_DOTTED_UM:
                is_interp[run_start:i] = True
            in_run = False
    if in_run:
        run_width_um = (len(nan_mask) - run_start) * bin_um
        if run_width_um > GAP_DOTTED_UM:
            is_interp[run_start:] = True

    # Linear interpolation across all NaN gaps (don't extrapolate)
    valid = ~nan_mask
    if valid.sum() >= 2:
        f = interp1d(centers[valid], d_min[valid],
                     kind='linear', bounds_error=False, fill_value=np.nan)
        d_min[nan_mask] = f(centers[nan_mask])

    return centers, d_min, is_interp

# ── Build profiles ────────────────────────────────────────────────────────────
# Use the mesh column (not cls) so PDEL/PDER are kept separate.
# dedup=False here because blend_intestine_proximity.py already handles the
# shared-vertex issue via the pde_shared_zone flag.
profiles = {
    'CANL': ap_profile(['CANL']),
    'CANR': ap_profile(['CANR']),
    'PDEL': ap_profile(['PDEL']),
    'PDER': ap_profile(['PDER']),
    'HSNL': ap_profile(['HSNL']),
    'AVM':  ap_profile(['AVM']),
}

# ── Colours ───────────────────────────────────────────────────────────────────
STYLE = {
    'CANL': dict(color='#2166ac', lw=1.8, ls='-',  label='CANL'),
    'CANR': dict(color='#6baed6', lw=1.8, ls='--', label='CANR'),
    'PDEL': dict(color='#e08030', lw=1.5, ls='-',  label='PDEL'),
    'PDER': dict(color='#f4a460', lw=1.5, ls='--', label='PDER'),
    'HSNL': dict(color='#74add1', lw=1.2, ls='-',  label='HSNL'),
    'AVM':  dict(color='#74add1', lw=1.2, ls='--', label='AVM'),
}

# ── Figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8.5, 4.0))

# Shared-mesh zone drawn first (lower z); hatching is faint so green shows over it
ax.axvspan(SHARED_Y_MIN, SHARED_Y_MAX, facecolor='none', hatch='///',
           edgecolor='#ddb5b5', lw=0.5, zorder=0,
           label='Shared-mesh zone\n(PDE/HSN L/R indistinguishable)')

# Intestine span on top of hatching
ax.axvspan(INT_Y_MIN, INT_Y_MAX, color='#d4edda', alpha=0.50, zorder=1,
           label='Intestine extent')

# Nerve ring reference
ax.axvline(NERVE_RING_Y, color='#999999', lw=0.8, ls=':', zorder=2)
ax.text(NERVE_RING_Y + 8, 58, 'Nerve\nring', ha='left', va='top',
        fontsize=6.5, color='#777777')

# Threshold line
ax.axhline(THRESH_UM, color='#cc3300', lw=1.0, ls=':', zorder=5,
           label=f'{THRESH_UM:.0f} µm threshold')

# Neuron profiles
for name, (yc, dm, interp) in profiles.items():
    st = STYLE[name]
    valid_seg = ~np.isnan(dm)

    # Solid line through data-backed bins
    solid_yc = np.where(valid_seg & ~interp, yc, np.nan)
    solid_dm = np.where(valid_seg & ~interp, dm, np.nan)

    # Dotted line through interpolated bins
    interp_yc = np.where(valid_seg & interp, yc, np.nan)
    interp_dm = np.where(valid_seg & interp, dm, np.nan)

    ax.plot(solid_yc, solid_dm,
            color=st['color'], lw=st['lw'], ls=st['ls'], zorder=6)
    ax.plot(interp_yc, interp_dm,
            color=st['color'], lw=st['lw'], ls=':', alpha=0.5, zorder=6)

    # Invisible line for legend entry (solid style)
    ax.plot([], [], color=st['color'], lw=st['lw'], ls=st['ls'],
            label=st['label'])

ax.set_xlabel('Body position (µm, anterior → posterior)', fontsize=10)
ax.set_ylabel('Min. distance to intestine (µm)', fontsize=10)
ax.set_title('Neurite proximity to intestine along body axis\n'
             '(Virtual Worm Blender morphology, Blender-native coordinates)',
             fontsize=10, loc='left')
ax.set_ylim(0, 60)
ax.spines[['top', 'right']].set_visible(False)

# Legend — two columns to save vertical space
ax.legend(fontsize=8, ncol=2, framealpha=0.9, loc='upper right')

fig.tight_layout()

out_stem = OUT / 'ap_axis_proximity'
fig.savefig(str(out_stem) + '.png', dpi=180, bbox_inches='tight')
fig.savefig(str(out_stem) + '.pdf', bbox_inches='tight')
print(f"Saved {out_stem}.png / .pdf")
plt.close()

# ── Combined figure: AP-axis profile + neuron Y-extent (neuron_y_distribution) ──
# Merge CANL/CANR → CAN and PDEL/PDER → PDE for the extent panel
BILATERAL_MERGE = {'CANL': 'CAN', 'CANR': 'CAN', 'PDEL': 'PDE', 'PDER': 'PDE'}
nn['display_cls'] = nn['cls'].apply(lambda c: BILATERAL_MERGE.get(c, c))

# Order merged classes by minimum d_min_um (matches Panel A of blend_intestine_proximity)
metrics_df = pd.read_csv(OUT / 'blend_intestine_proximity_metrics.csv')
metrics_df['display_cls'] = metrics_df['neuron_class'].apply(
    lambda c: BILATERAL_MERGE.get(c, c))
cls_order_merged = (metrics_df.groupby('display_cls')['d_min_um'].min()
                    .sort_values().index.tolist())

# Y extent per merged class
extent = (nn.groupby('display_cls')
           .agg(y_min=('y_um', 'min'), y_max=('y_um', 'max'))
           .reindex(cls_order_merged))

# Colors match blend_intestine_proximity.py (CAN blue, PDE orange)
COL_CAN_BAR  = '#2166ac'
COL_PDE_BAR  = '#e08030'
COL_PART_BAR = '#74add1'
COL_REST_BAR = '#bdbdbd'

def bar_color(cls):
    if cls == 'CAN':                       return COL_CAN_BAR
    if cls == 'PDE':                       return COL_PDE_BAR
    if cls in {'HSN', 'AVM', 'VC4/5'}:   return COL_PART_BAR
    return COL_REST_BAR

fig2, (ax_top, ax_bot) = plt.subplots(
    2, 1, figsize=(8.5, 6.5), sharex=True,
    gridspec_kw={'height_ratios': [2, 1]})
fig2.subplots_adjust(hspace=0.12)

# ── Top panel: AP-axis proximity profiles ─────────────────────────────────────
# Hatching first (lower z), then green intestine zone on top
ax_top.axvspan(SHARED_Y_MIN, SHARED_Y_MAX, facecolor='none', hatch='///',
               edgecolor='#ddb5b5', lw=0.5, zorder=0,
               label='Shared-mesh zone\n(PDE/HSN L/R indistinguishable)')
ax_top.axvspan(INT_Y_MIN, INT_Y_MAX, color='#d4edda', alpha=0.50, zorder=1,
               label='Intestine extent')
ax_top.axvline(NERVE_RING_Y, color='#999999', lw=0.8, ls=':', zorder=2)
ax_top.axhline(THRESH_UM, color='#cc3300', lw=1.0, ls=':', zorder=5,
               label=f'{THRESH_UM:.0f} µm threshold')

for name, (yc, dm, interp) in profiles.items():
    st = STYLE[name]
    valid_seg = ~np.isnan(dm)
    solid_yc  = np.where(valid_seg & ~interp, yc, np.nan)
    solid_dm  = np.where(valid_seg & ~interp, dm, np.nan)
    interp_yc = np.where(valid_seg & interp,  yc, np.nan)
    interp_dm = np.where(valid_seg & interp,  dm, np.nan)
    ax_top.plot(solid_yc,  solid_dm,  color=st['color'], lw=st['lw'], ls=st['ls'],  zorder=6)
    ax_top.plot(interp_yc, interp_dm, color=st['color'], lw=st['lw'], ls=':', alpha=0.5, zorder=6)
    ax_top.plot([], [], color=st['color'], lw=st['lw'], ls=st['ls'], label=st['label'])

ax_top.set_ylabel('Min. distance to intestine (µm)', fontsize=9)
ax_top.set_ylim(0, 60)
ax_top.spines[['top', 'right']].set_visible(False)
ax_top.legend(fontsize=7.5, ncol=2, framealpha=0.9, loc='upper right')
ax_top.set_title('A   Neurite proximity to intestine along body axis',
                 fontsize=9.5, loc='left', fontweight='bold')

# ── Bottom panel: neuron Y-extent bars ────────────────────────────────────────
ax_bot.axvspan(SHARED_Y_MIN, SHARED_Y_MAX, facecolor='none', hatch='///',
               edgecolor='#ddb5b5', lw=0.5, zorder=0)
ax_bot.axvspan(INT_Y_MIN, INT_Y_MAX, color='#d4edda', alpha=0.50, zorder=1)
ax_bot.axvline(NERVE_RING_Y, color='#999999', lw=0.8, ls=':', zorder=2)
ax_bot.text(NERVE_RING_Y, len(cls_order_merged) - 0.2, 'Nerve\nring',
            ha='center', va='top', fontsize=6.5, color='#777777', zorder=5)

for i, cls in enumerate(cls_order_merged):
    row = extent.loc[cls]
    ax_bot.barh(i, row['y_max'] - row['y_min'], left=row['y_min'],
                height=0.65, color=bar_color(cls), edgecolor='white', lw=0.4, zorder=3)

ax_bot.set_yticks(range(len(cls_order_merged)))
ax_bot.set_yticklabels(cls_order_merged, fontsize=9, fontstyle='italic')
ax_bot.set_xlabel('Body position (µm, Blender world coordinates)', fontsize=9)
ax_bot.spines[['top', 'right']].set_visible(False)
ax_bot.set_title('B   cat-1⁺ neuron process extent',
                 fontsize=9.5, loc='left', fontweight='bold')

handles_b = [
    mpatches.Patch(facecolor='#d4edda', edgecolor='#5a9c6a', label='Intestine extent'),
    mpatches.Patch(color=COL_CAN_BAR,  label='CAN (full body)'),
    mpatches.Patch(color=COL_PDE_BAR,  label='PDE (full body)'),
    mpatches.Patch(color=COL_PART_BAR, label='Partial morphology (HSN, AVM, VC4/5)'),
    mpatches.Patch(color=COL_REST_BAR, label='Head-confined / soma only'),
]
ax_bot.legend(handles=handles_b, fontsize=7.5, loc='lower right', framealpha=0.9)

out2 = OUT / 'neuron_y_distribution'
fig2.savefig(str(out2) + '.png', dpi=180, bbox_inches='tight')
fig2.savefig(str(out2) + '.pdf', bbox_inches='tight')
print(f"Saved {out2}.png / .pdf")
plt.close(fig2)
