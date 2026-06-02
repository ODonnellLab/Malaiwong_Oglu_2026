"""
blend_intestine_proximity.py
============================
Nearest-neighbour distance analysis between cat-1+ neuron process meshes
and the C. elegans intestine — performed entirely in Virtual Worm Blender
world coordinates (no NeuroML coordinate transform required).

Data source
-----------
Virtual_Worm_February_2012.blend, extracted via extract_world_verts.py:
    data/nml_vertices/blend_world_verts.csv      — vertex clouds (world space)
    data/nml_vertices/blend_world_centroids.csv  — per-object centroids

Distance metric
---------------
Two regimes depending on where the neuron vertex sits relative to the intestine:

  Within intestine Y range (± Y_WIN_UM buffer):
      Cross-sectional XZ distance to the nearest intestine vertex at the same
      body level (within ±Y_WIN_UM along the body axis).  The Y-window is
      expanded up to Y_WIN_MAX_UM when the intestine mesh has a gap at that
      level.  This prevents false-close matches from unrelated body positions.

  Outside intestine Y range (anterior to the intestine):
      Full 3D nearest-neighbour distance.  Neurons here are in the head ganglia
      and their closest intestine point is legitimately at the intestine's
      anterior tip; a 3D query is physically meaningful.

Gap interpolation
-----------------
Both the neuron meshes (CANL/CANR: up to 110 µm gap) and the intestine
(up to 51 µm gap) are sparse along the body axis.  Linear interpolation is
applied to fill gaps > GAP_THRESH_UM before computing distances.  Interpolated
vertices are flagged in the output (interpolated=True).

Neuron mesh coverage (from Blender file)
-----------------------------------------
Full body-length morphology : CANL, CANR, PDEL, PDER
Substantial partial          : HSN (L/R, head→mid-body), AVM (head→anterior body)
Head-confined partial        : ADF, ASI, ADE, NSM, CEP
Soma-only (nerve ring)       : RIC, RIM, RIH, RIR, VC4/5

Note on shared Blender meshes
------------------------------
PDEL/PDER: 68.8 % of vertices are byte-identical (fasciculated in anterior body).
  L/R distinction is unreliable in y ≈ −285 to −97 µm; flag this zone.
CANL/CANR: zero shared vertices — fully distinct, L/R analysis is valid.

Scale conversion
----------------
Blender world units → µm  using the isotropic scale derived from:
  - Y-axis regression over 20 compact calibration neurons: 112.78 µm/unit
  - X-axis bilateral CAN pair (L vs R separation):         116.77 µm/unit
Mean: BLEND_TO_UM = 114.8 µm/Blender unit  (within 2 % of either estimate)

Usage
-----
Prerequisites: run extract_world_verts.py first (requires Blender).

    python blend_intestine_proximity.py [--out-dir data/nml_vertices]
                                        [--contact-threshold 10]
                                        [--y-win 25]
                                        [--gap-thresh 20]

Outputs (in --out-dir)
-----------------------
  blend_neurite_nn_distances.csv        — per-vertex distances (incl. interp.)
  blend_intestine_proximity_metrics.csv — per-neuron-class summary
  blend_intestine_proximity.png / .pdf  — figure
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── CLI ──────────────────────────────────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument('--out-dir',          default='data/nml_vertices')
ap.add_argument('--contact-threshold', type=float, default=10.0,
                help='Contact distance threshold in µm (default 10)')
ap.add_argument('--y-win',            type=float, default=25.0,
                help='Initial Y-window for cross-sectional query in µm (default 25)')
ap.add_argument('--gap-thresh',       type=float, default=20.0,
                help='Gap threshold for interpolation in µm (default 20)')
args      = ap.parse_args()
OUT       = Path(args.out_dir)
THRESH_UM = args.contact_threshold
Y_WIN_UM  = args.y_win
Y_WIN_MAX_UM = 80.0        # maximum window expansion for gaps
GAP_THRESH_UM = args.gap_thresh
INTERP_STEP_UM = 10.0      # spacing of inserted interpolated vertices

BLEND_TO_UM = 114.8        # µm per Blender world unit (isotropic scale)

# Y range where PDEL/PDER share vertices (fasciculated zone)
PDE_SHARED_Y_MIN_UM = -285.0
PDE_SHARED_Y_MAX_UM = -97.0

# ── Neuron → class mapping ───────────────────────────────────────────────────
# CANL/CANR and PDEL/PDER are kept as separate entries; all other bilaterals
# are merged because they share substantial mesh geometry.
CLASS_MAP = {
    'CANL' :'CANL',  'CANR' :'CANR',
    'PDEL' :'PDEL',  'PDER' :'PDER',
    'HSNL' :'HSN',   'HSNR' :'HSN',
    'AVM'  :'AVM',
    'RICL' :'RIC',   'RICR' :'RIC',
    'RIML' :'RIM',   'RIMR' :'RIM',
    'RIH'  :'RIH',
    'RIR'  :'RIR',
    'NSML' :'NSM',   'NSMR' :'NSM',
    'ADFL' :'ADF',   'ADFR' :'ADF',
    'ASIL' :'ASI',   'ASIR' :'ASI',
    'ADEL' :'ADE',   'ADER' :'ADE',
    'CEPDL':'CEP',   'CEPDR':'CEP',  'CEPVL':'CEP', 'CEPVR':'CEP',
    'VC4'  :'VC4/5', 'VC5'  :'VC4/5',
}

COVERAGE = {
    'CANL':'full body',  'CANR':'full body',
    'PDEL':'full body',  'PDER':'full body',
    'HSN' :'partial (head→mid)',  'AVM':'partial (head→ant.)',
    'ADF' :'head', 'ASI':'head', 'ADE':'head', 'NSM':'head', 'CEP':'head',
    'RIC' :'soma only', 'RIM':'soma only',
    'RIH' :'soma only', 'RIR':'soma only', 'VC4/5':'soma only',
}

# ── Load data ────────────────────────────────────────────────────────────────
# blend_world_verts.csv     — columns: mesh, vid, x, y, z
# blend_world_face_cents.csv — columns: mesh, x, y, z  (one row per face)
#
# Intestine distance query — use raw VERTICES, not face centroids.
#   Each vertex is a real surface point on the intestine wall.  A large quad
#   face centroid sits 25 µm from its corners; using vertices gives the true
#   minimum wall distance.  We want every corner of every face available.
#
# Neuron path representation — use face CENTROIDS, not raw vertices.
#   Neuron meshes are strips of quads whose vertex pairs straddle the axon.
#   Raw vertex pairs zigzag around the cross-section when sorted by Y; face
#   centroids sit at the midpoint of each quad and naturally trace the axon axis.

verts      = pd.read_csv(OUT / 'blend_world_verts.csv')
face_cents = pd.read_csv(OUT / 'blend_world_face_cents.csv')

# Intestine: dense surface samples via bilinear face interpolation.
# The raw Blender mesh is low-poly (rendering resolution); using only
# corner vertices leaves gaps of up to 50 µm between samples, causing
# NN distances to miss the true nearest wall point.  We subdivide each
# face at FACE_SUBDIV×FACE_SUBDIV internal grid points so the maximum
# inter-sample gap is ~face_size / FACE_SUBDIV ≈ 3–5 µm.
FACE_SUBDIV = 8   # interior grid per face edge (8×8 = 64 pts/face)

faces_df = pd.read_csv(OUT / 'blend_world_faces.csv')
# vid lookup: index into verts on (mesh, vid)
verts_idx = verts.set_index(['mesh', 'vid'])

def dense_face_samples(mesh_name, subdiv=FACE_SUBDIV):
    """
    Bilinear/barycentric sampling of all faces for one mesh object.
    Returns (N, 3) float array of world-space XYZ points.
    """
    mf  = faces_df[faces_df['mesh'] == mesh_name]
    mv  = verts[verts['mesh'] == mesh_name].set_index('vid')
    pts = []
    ts  = np.linspace(0, 1, subdiv + 2)[1:-1]   # interior parameter values
    for _, row in mf.iterrows():
        vids = [int(row['v0']), int(row['v1']), int(row['v2'])]
        is_quad = pd.notna(row['v3']) and row['v3'] != ''
        if is_quad:
            vids.append(int(row['v3']))
        coords = mv.loc[vids, ['x', 'y', 'z']].values.astype(float)
        if is_quad:
            p0, p1, p2, p3 = coords   # assumed order: bl, br, tr, tl
            for s in ts:
                for t in ts:
                    # bilinear interpolation
                    p = ((1-s)*(1-t)*p0 + s*(1-t)*p1 +
                         s*t    *p2 + (1-s)*t*p3)
                    pts.append(p)
        else:
            p0, p1, p2 = coords
            for s in ts:
                for t in ts[ts <= 1 - s]:   # stay inside triangle
                    p = (1-s-t)*p0 + s*p1 + t*p2
                    pts.append(p)
    return np.array(pts) if pts else np.empty((0, 3))

int_mesh_names = (verts[verts['mesh'].str.match(r'^int\d')]['mesh']
                  .unique().tolist())
int_dense_parts = [dense_face_samples(m) for m in int_mesh_names]
int_dense = np.vstack([p for p in int_dense_parts if len(p)])
print(f"Intestine dense samples: {len(int_dense):,}  "
      f"({FACE_SUBDIV}×{FACE_SUBDIV} per face, {len(int_mesh_names)} cells)")

all_neurons = list(CLASS_MAP.keys())

# Neurons: face centroids for path tracing + gap interpolation
neu_fc = face_cents[face_cents['mesh'].isin(all_neurons)].copy()
neu_fc['cls'] = neu_fc['mesh'].map(CLASS_MAP)
print(f"Neuron face-cents:      {len(neu_fc)} across {neu_fc['cls'].nunique()} classes")

# ── Intestine Y extent ───────────────────────────────────────────────────────
INT_Y_MIN_BL = int_dense[:, 1].min()
INT_Y_MAX_BL = int_dense[:, 1].max()
INT_Y_MIN_UM = INT_Y_MIN_BL * BLEND_TO_UM
INT_Y_MAX_UM = INT_Y_MAX_BL * BLEND_TO_UM
print(f"Intestine Y range: {INT_Y_MIN_UM:.0f} to {INT_Y_MAX_UM:.0f} µm")

# ── Gap-fill neuron face-centroid paths ───────────────────────────────────────
# Neuron face centroids sorted by Y trace the axon axis cleanly.
# Gaps > GAP_THRESH_UM are filled with linearly interpolated centroids
# (flagged interpolated=True) so the AP-axis profile is continuous.
#
# Intestine face centroids are NOT gap-filled: each int* cell is a closed
# ring; sorting by Y and interpolating between ring sections would place
# phantom points inside the intestinal lumen.  Instead, the cross-sectional
# query window expands to bridge intestine gaps (up to Y_WIN_MAX_UM).

def fill_path_gaps(fc_df, gap_thresh_um=GAP_THRESH_UM, step_um=INTERP_STEP_UM):
    """
    Insert linearly interpolated face-centroid points through Y gaps.
    Returns fc_df (interpolated=False) + new rows (interpolated=True).
    """
    groups = []
    gap_bl  = gap_thresh_um / BLEND_TO_UM
    step_bl = step_um       / BLEND_TO_UM
    for name, grp in fc_df.groupby('mesh'):
        cls_val = grp['cls'].iloc[0]
        fc_s    = grp.sort_values('y').reset_index(drop=True)
        pts     = fc_s[['x', 'y', 'z']].values
        orig    = fc_s.assign(interpolated=False)
        new_rows = []
        for i in range(len(pts) - 1):
            p0, p1 = pts[i], pts[i + 1]
            dy = p1[1] - p0[1]
            if dy > gap_bl:
                n_steps = max(1, int(np.ceil(dy / step_bl)) - 1)
                for k in range(1, n_steps + 1):
                    t = k / (n_steps + 1)
                    new_rows.append({
                        'mesh': name, 'cls': cls_val,
                        'x': p0[0] + t * (p1[0] - p0[0]),
                        'y': p0[1] + t * (p1[1] - p0[1]),
                        'z': p0[2] + t * (p1[2] - p0[2]),
                        'interpolated': True,
                    })
        chunk = (pd.concat([orig, pd.DataFrame(new_rows)], ignore_index=True)
                 if new_rows else orig)
        groups.append(chunk)
    return pd.concat(groups, ignore_index=True)

neu_fill = fill_path_gaps(neu_fc)
neu_fill['y_um'] = neu_fill['y'] * BLEND_TO_UM
n_interp = int(neu_fill['interpolated'].sum())
print(f"Neuron after gap-fill: {len(neu_fill)}  (+{n_interp} interpolated)")

# Display filter cutoff: anterior edge of CEP soma.
# Approximated as the most anterior per-mesh centroid (mean vertex y) across
# the 4 CEP meshes.  Vertex-mean is biased toward the soma (the largest
# structure), giving a clean separation from the anteriorly-projecting
# sensory dendrites that inflate distances.
cep_centroid_y = (verts[verts['mesh'].isin(['CEPDL','CEPDR','CEPVL','CEPVR'])]
                  .groupby('mesh')['y'].mean()
                  .min())
NERVE_RING_Y_UM = float(cep_centroid_y * BLEND_TO_UM)
print(f"CEP soma cutoff (min mesh centroid Y): {NERVE_RING_Y_UM:.1f} µm")

# ── Cross-sectional distance ──────────────────────────────────────────────────
# Two regimes based on neuron Y relative to the intestine:
#
#   Inside [INT_Y_MIN, INT_Y_MAX]:
#       XZ cross-sectional distance to the nearest intestine face centroid at
#       the same body level (Y-window, expanding up to Y_WIN_MAX_UM for gaps).
#       Prevents false matches to distant body positions caused by body curvature.
#
#   Outside [INT_Y_MIN, INT_Y_MAX]:
#       Full 3D distance to nearest intestine face centroid (finds anterior tip).
#       Head neurons legitimately approach the intestine at its anterior end.

int_arr = int_dense                              # dense intestine surface
int_y_s = int_arr[np.argsort(int_arr[:, 1])]   # Y-sorted
int_ys  = int_y_s[:, 1]

Y_WIN_BL     = Y_WIN_UM     / BLEND_TO_UM
Y_WIN_MAX_BL = Y_WIN_MAX_UM / BLEND_TO_UM

def dist_to_intestine(neu_pts_arr):
    """
    Parameters
    ----------
    neu_pts_arr : (N, 3) float array, Blender world units

    Returns
    -------
    dists_um : (N,) float array, distances in µm
    """
    dists = np.empty(len(neu_pts_arr))
    for i, pt in enumerate(neu_pts_arr):
        y_n = pt[1]

        if INT_Y_MIN_BL <= y_n <= INT_Y_MAX_BL:
            # Cross-sectional regime: expand Y window until intestine found
            for win in (Y_WIN_BL, Y_WIN_BL * 2, Y_WIN_MAX_BL):
                lo = np.searchsorted(int_ys, y_n - win)
                hi = np.searchsorted(int_ys, y_n + win)
                if hi > lo:
                    slc  = int_y_s[lo:hi]
                    xz_d = np.sqrt((slc[:, 0] - pt[0])**2 +
                                   (slc[:, 2] - pt[2])**2)
                    dists[i] = xz_d.min() * BLEND_TO_UM
                    break
            else:
                # Fallback: full 3D (large gap, window maxed out)
                dists[i] = (np.sqrt(((int_arr - pt)**2).sum(1)).min()
                            * BLEND_TO_UM)
        else:
            # Outside intestine Y range: 3D distance to nearest intestine tip
            dists[i] = (np.sqrt(((int_arr - pt)**2).sum(1)).min()
                        * BLEND_TO_UM)

    return dists

print("\nComputing cross-sectional distances …")
neu_pts  = neu_fill[['x', 'y', 'z']].values
dists_um = dist_to_intestine(neu_pts)

neu_fill             = neu_fill.copy()
neu_fill['d_nn_um']  = dists_um
neu_fill['d_nn_bl']  = dists_um / BLEND_TO_UM

# Flag PDE shared-mesh zone (L/R indistinguishable here)
neu_fill['pde_shared_zone'] = (
    neu_fill['mesh'].isin(['PDEL', 'PDER']) &
    neu_fill['y_um'].between(PDE_SHARED_Y_MIN_UM, PDE_SHARED_Y_MAX_UM)
)

neu_fill.to_csv(OUT / 'blend_neurite_nn_distances.csv', index=False)
print(f"Saved blend_neurite_nn_distances.csv  ({len(neu_fill)} rows)")

# ── Per-class metrics ─────────────────────────────────────────────────────────
# Contact length: sum of AP-axis profile bins where d_min < threshold.
# This is more accurate than max(Y_close) - min(Y_close) because it only
# counts regions where the neuron is ACTUALLY within threshold — not bridging
# across mid-body gaps where it has drifted away.  Gaps in the face-centroid
# path are filled by linear interpolation (same logic as blend_ap_axis_figure.py)
# so that real neurite path coverage is not underestimated.
# Head neurons outside the intestine Y range use 3D distance to the anterior
# intestine tip; that distance appears in their bins and is correctly excluded
# by the threshold (they are typically > 30 µm from the intestine).

from scipy.interpolate import interp1d as _interp1d

BIN_UM_CONTACT  = 10.0     # AP-axis bin width for contact length (µm)
GAP_FILL_MAX_UM = 80.0     # max gap to interpolate across

def _contact_length(grp, thresh_um=THRESH_UM, bin_um=BIN_UM_CONTACT):
    y_um = grp['y_um'].values
    d_um = grp['d_nn_um'].values
    if len(y_um) < 2:
        return 0.0
    y_min = np.floor(y_um.min() / bin_um) * bin_um
    y_max = np.ceil(y_um.max()  / bin_um) * bin_um
    edges   = np.arange(y_min, y_max + bin_um, bin_um)
    centers = (edges[:-1] + edges[1:]) / 2
    d_min   = np.full(len(centers), np.nan)
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (y_um >= lo) & (y_um < hi)
        if mask.any():
            d_min[i] = d_um[mask].min()
    valid = ~np.isnan(d_min)
    if valid.sum() >= 2:
        f = _interp1d(centers[valid], d_min[valid], kind='linear',
                      bounds_error=False, fill_value=np.nan)
        nan_mask = np.isnan(d_min)
        d_min[nan_mask] = f(centers[nan_mask])
    return float(np.nansum(d_min < thresh_um) * bin_um)

rows = []
for cls, grp in neu_fill.groupby('cls'):
    d    = grp['d_nn_um'].values
    d_bl = grp['d_nn_bl'].values
    rows.append({
        'neuron_class':       cls,
        'coverage':           COVERAGE.get(cls, ''),
        'd_min_um':           float(np.min(d)),
        'd_p10_um':           float(np.percentile(d, 10)),
        'd_median_um':        float(np.median(d)),
        'd_p90_um':           float(np.percentile(d, 90)),
        'frac_within_thresh': float(np.mean(d_bl <= THRESH_UM / BLEND_TO_UM)),
        'contact_span_um':    _contact_length(grp),
        'n_verts':            len(d),
    })

metrics = (pd.DataFrame(rows)
             .sort_values('d_min_um')
             .reset_index(drop=True))
metrics.to_csv(OUT / 'blend_intestine_proximity_metrics.csv', index=False)

print(f"\nProximity metrics (threshold = {THRESH_UM} µm):")
print(metrics[['neuron_class', 'coverage', 'd_min_um', 'd_median_um',
               'contact_span_um', 'frac_within_thresh']].to_string(index=False))

# ── Figure ────────────────────────────────────────────────────────────────────
COL_CANL    = '#2166ac'   # dark blue
COL_CANR    = '#2166ac'   # same dark blue; row position distinguishes L vs R
COL_PDE     = '#e08030'
COL_PARTIAL = '#74add1'
COL_HEAD    = '#bdbdbd'

def cls_color(cls):
    if cls == 'CANL': return COL_CANL
    if cls == 'CANR': return COL_CANR
    if cls in ('PDEL', 'PDER'):  return COL_PDE
    if cls == 'VC4/5':           return COL_PARTIAL   # mid-body, not head-confined
    cov = COVERAGE.get(cls, '')
    if 'partial' in cov:         return COL_PARTIAL
    return COL_HEAD

fig, ax = plt.subplots(figsize=(7.0, 5.5))

cls_order = metrics['neuron_class'].tolist()
for i, cls in enumerate(cls_order):
    # Exclude vertices anterior to the nerve ring (sensory dendrites projecting
    # toward the nose tip); keep all processes at or posterior to the nerve ring
    cls_data = neu_fill[(neu_fill['cls'] == cls) & (neu_fill['y_um'] >= NERVE_RING_Y_UM)]
    if cls_data.empty:
        cls_data = neu_fill[neu_fill['cls'] == cls]
    d   = cls_data['d_nn_um'].values
    col = cls_color(cls)
    ax.scatter(d, [i] * len(d), s=3, color=col, alpha=0.3, zorder=2)
    ax.plot([np.percentile(d, 10), np.percentile(d, 90)], [i, i],
            color=col, lw=1.5, alpha=0.6, zorder=3)
    ax.plot([np.median(d)], [i], 'o', color=col, ms=6, zorder=4,
            markeredgecolor='white', markeredgewidth=0.5)

ax.axvline(THRESH_UM, color='#cc3300', lw=1.2, ls='--')
ax.set_yticks(range(len(cls_order)))
ax.set_yticklabels(cls_order, fontsize=9, fontstyle='italic')
ax.set_xlabel('Cross-sectional distance to intestine (µm)', fontsize=9)
ax.set_title('Distance to intestine (cross-sectional within intestine region)',
             fontsize=9.5, loc='left', fontweight='bold')
ax.text(0.01, -0.10,
        f'Vertices anterior to CEP soma excluded (body position < {NERVE_RING_Y_UM:.0f} µm)',
        transform=ax.transAxes, fontsize=7, color='#777777', va='top')
ax.spines[['top', 'right']].set_visible(False)
# Cap at 98th percentile of filtered data; floor at 50 µm so head-neuron
# soma distances (30–47 µm) remain visible
cap = np.percentile(
    neu_fill[neu_fill['y_um'] >= NERVE_RING_Y_UM]['d_nn_um'].values, 98)
ax.set_xlim(0, max(cap, 50.0))


fig.tight_layout()
out_stem = OUT / 'blend_intestine_proximity'
fig.savefig(str(out_stem) + '.png', dpi=180, bbox_inches='tight')
fig.savefig(str(out_stem) + '.pdf', bbox_inches='tight')
print(f"\nFigure saved to {out_stem}.png / .pdf")
plt.close()
