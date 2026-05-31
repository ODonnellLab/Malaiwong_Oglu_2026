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

# Y-range (in µm) where HSNR/AVM share vertices with PDEL/PDER
# Diagnosed in prior analysis — treat this zone as unreliable for those neurons
SHARED_Y_MIN = -285.0
SHARED_Y_MAX = -97.0

# ── Helper: AP-axis profile for one or more mesh names ───────────────────────
def ap_profile(mesh_names, bin_um=BIN_UM, dedup=False):
    """
    Return (y_centers, d_min, is_interpolated) arrays for the given meshes.

    y_centers      : bin centre positions in µm
    d_min          : minimum distance to intestine within each bin (µm)
    is_interpolated: True for bins filled by linear interpolation (no raw data)
    """
    sub = nn[nn['mesh'].isin(mesh_names)].copy()
    if dedup:
        sub = sub.drop_duplicates(subset=['x', 'y', 'z'])

    y_all = sub['y_um'].values
    d_all = sub['d_nn_um'].values

    y_min = np.floor(y_all.min() / bin_um) * bin_um
    y_max = np.ceil(y_all.max()  / bin_um) * bin_um
    edges = np.arange(y_min, y_max + bin_um, bin_um)
    centers = (edges[:-1] + edges[1:]) / 2

    d_min = np.full(len(centers), np.nan)
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (y_all >= lo) & (y_all < hi)
        if mask.any():
            d_min[i] = d_all[mask].min()

    is_interp = np.isnan(d_min)

    # Linear interpolation across NaN gaps (but don't extrapolate)
    valid = ~np.isnan(d_min)
    if valid.sum() >= 2:
        f = interp1d(centers[valid], d_min[valid],
                     kind='linear', bounds_error=False, fill_value=np.nan)
        d_min[is_interp] = f(centers[is_interp])

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
    'HSNL': dict(color='#777777', lw=1.2, ls='-',  label='HSNL'),
    'AVM':  dict(color='#aaaaaa', lw=1.2, ls='-',  label='AVM'),
}

# ── Figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8.5, 4.0))

# Intestine span (shaded background)
ax.axvspan(INT_Y_MIN, INT_Y_MAX, color='#d4edda', alpha=0.5, zorder=0,
           label='Intestine extent')

# Shared-mesh unreliable zone
ax.axvspan(SHARED_Y_MIN, SHARED_Y_MAX, color='#ffd9d9', alpha=0.6, zorder=1,
           label='Shared-mesh zone\n(PDE/HSN L/R indistinguishable)')

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
ax.set_ylim(bottom=0)
ax.spines[['top', 'right']].set_visible(False)

# Legend — two columns to save vertical space
ax.legend(fontsize=8, ncol=2, framealpha=0.9, loc='upper right')

fig.tight_layout()

out_stem = OUT / 'ap_axis_proximity'
fig.savefig(str(out_stem) + '.png', dpi=180, bbox_inches='tight')
fig.savefig(str(out_stem) + '.pdf', bbox_inches='tight')
print(f"Saved {out_stem}.png / .pdf")
plt.close()
