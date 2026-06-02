"""
blend_render.py
===============
3D render of CAN neurites alongside the C. elegans intestine using
Virtual Worm Blender morphology data.

Requires blend_intestine_proximity.py to have been run first.

Usage
-----
    python blend_render.py [--out-dir data/nml_vertices] [--y-slice 150]

Outputs
-------
  can_intestine_render.png / .pdf
"""

import argparse
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401 (registers projection)
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ── CLI ───────────────────────────────────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument('--out-dir',  default='data/nml_vertices')
ap.add_argument('--y-slice',  type=float, default=150.0,
                help='Body position (µm) for cross-section panel (default 150)')
args    = ap.parse_args()
OUT     = Path(args.out_dir)
Y_SLICE = args.y_slice
Y_WIN   = 30.0   # ± µm window for cross-section

BLEND_TO_UM = 114.8

# ── Load data ─────────────────────────────────────────────────────────────────
nn  = pd.read_csv(OUT / 'blend_neurite_nn_distances.csv')
v   = pd.read_csv(OUT / 'blend_world_verts.csv')
fd  = pd.read_csv(OUT / 'blend_world_faces.csv')

int_raw = v[v['mesh'].str.match(r'^int\d')].copy()
int_raw['y_um'] = int_raw['y'] * BLEND_TO_UM

def cell_idx(name):
    m = re.match(r'int(\d+)', name)
    return int(m.group(1)) if m else 0

int_raw['ci'] = int_raw['mesh'].apply(cell_idx)

# CAN paths (face centroids + gap interpolations from nn distances file)
def can_path(name):
    df = nn[nn['mesh'] == name].sort_values('y')
    r  = df[~df['interpolated']][['x', 'y', 'z']].values * BLEND_TO_UM
    i  = df[ df['interpolated']][['x', 'y', 'z']].values * BLEND_TO_UM
    return r, i

canl_r, canl_i = can_path('CANL')
canr_r, canr_i = can_path('CANR')

# ── Build intestine Poly3D faces ──────────────────────────────────────────────
int_meshes = sorted(int_raw['mesh'].unique())
cmap_int   = plt.cm.YlGn
norm_int   = plt.Normalize(1, 9)

def build_polys(meshes):
    polys, colours = [], []
    for name in meshes:
        ci  = cell_idx(name)
        col = cmap_int(norm_int(ci))
        mv  = v[v['mesh'] == name].set_index('vid')
        mf  = fd[fd['mesh'] == name]
        for _, row in mf.iterrows():
            vids = [int(row['v0']), int(row['v1']), int(row['v2'])]
            if pd.notna(row['v3']) and row['v3'] != '':
                vids.append(int(row['v3']))
            try:
                pts = mv.loc[vids, ['x', 'y', 'z']].values.astype(float) * BLEND_TO_UM
                polys.append(pts)
                colours.append(col)
            except KeyError:
                pass
    return polys, colours

polys, cols = build_polys(int_meshes)

# ── Figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(15, 6))

def draw_3d(ax, elev, azim, title):
    pc = Poly3DCollection(polys, alpha=0.18, linewidths=0.1)
    pc.set_facecolor([(*c[:3], 0.18) for c in cols])
    pc.set_edgecolor([(*c[:3], 0.35) for c in cols])
    ax.add_collection3d(pc)

    ax.plot(*canl_r.T, color='#1a5fa8', lw=2.5, zorder=8, label='CANL')
    if len(canl_i):
        ax.plot(*canl_i.T, color='#1a5fa8', lw=1.8, ls=':', alpha=0.55, zorder=8)
    ax.plot(*canr_r.T, color='#6db3e8', lw=2.5, zorder=8, label='CANR')
    if len(canr_i):
        ax.plot(*canr_i.T, color='#6db3e8', lw=1.8, ls=':', alpha=0.55, zorder=8)

    all_pts = np.vstack([p for p in polys] + [canl_r, canr_r])
    ax.set_xlim(all_pts[:, 0].min(), all_pts[:, 0].max())
    ax.set_ylim(all_pts[:, 1].min(), all_pts[:, 1].max())
    ax.set_zlim(all_pts[:, 2].min(), all_pts[:, 2].max())
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel('X (µm)', fontsize=7, labelpad=1)
    ax.set_ylabel('Y / A-P (µm)', fontsize=7, labelpad=1)
    ax.set_zlabel('Z (µm)', fontsize=7, labelpad=1)
    ax.set_title(title, fontsize=9, pad=4)
    ax.tick_params(labelsize=5, pad=0)
    ax.set_box_aspect([1, 5, 1])

ax1 = fig.add_subplot(1, 3, 1, projection='3d')
draw_3d(ax1, elev=15, azim=-80, title='Side view (looking along X)')
ax1.legend(fontsize=7, loc='upper left', framealpha=0.8)

ax2 = fig.add_subplot(1, 3, 2, projection='3d')
draw_3d(ax2, elev=28, azim=-115, title='Oblique 3D')

# ── Cross-section panel ───────────────────────────────────────────────────────
ax3 = fig.add_subplot(1, 3, 3)
sl  = int_raw[int_raw['y_um'].between(Y_SLICE - Y_WIN, Y_SLICE + Y_WIN)]

sc = ax3.scatter(sl['x'] * BLEND_TO_UM, sl['z'] * BLEND_TO_UM,
                 c=sl['ci'], cmap=cmap_int, vmin=1, vmax=9,
                 s=18, alpha=0.8, zorder=3, label='Intestine')

for pts, col, name in [(canl_r, '#1a5fa8', 'CANL'), (canr_r, '#6db3e8', 'CANR')]:
    mask = np.abs(pts[:, 1] - Y_SLICE) < Y_WIN
    sub  = pts[mask]
    if len(sub):
        ax3.scatter(sub[:, 0], sub[:, 2], s=80, c=col, zorder=7,
                    edgecolors='#222', linewidths=0.7, label=name)

ax3.axhline(0, color='#ccc', lw=0.5, ls='--')
ax3.axvline(0, color='#ccc', lw=0.5, ls='--')
ax3.set_xlabel('X / L-R (µm)', fontsize=9)
ax3.set_ylabel('Z / D-V (µm)', fontsize=9)
ax3.set_title(f'Cross-section Y≈{Y_SLICE:.0f} µm', fontsize=9)
ax3.set_aspect('equal')
ax3.legend(fontsize=8, loc='upper right')
ax3.spines[['top', 'right']].set_visible(False)
plt.colorbar(sc, ax=ax3, label='int cell (1→9)', shrink=0.55)

fig.suptitle('CAN neurites alongside the C. elegans intestine\n'
             '(Virtual Worm Blender morphology; face-centroid neurite paths)',
             fontsize=10, y=1.01)
fig.tight_layout()

out_stem = OUT / 'can_intestine_render'
fig.savefig(str(out_stem) + '.png', dpi=220, bbox_inches='tight')
fig.savefig(str(out_stem) + '.pdf', bbox_inches='tight')
print(f"Saved {out_stem}.png / .pdf")
plt.close()
