"""
_nml_intestine_proximity.py  [DEPRECATED]
=========================================
Superseded by blend_intestine_proximity.py (Blender-native coordinates).
Retained for reference only; delete after confirming blend_intestine_proximity.py
produces equivalent results.

Original description:
Nearest-neighbour distance analysis: cat-1+ neuron processes vs. C. elegans intestine.

Supports the spatial filter argument (step 2) in the Malaiwong & Oglu 2026 rebuttal:
"Of the 14 cat-1+ neuron types, which have neurite processes anatomically adjacent
to the intestine (where octopamine glucosides are produced)?"

Data sources
------------
Neurite positions : NeuroML c302 segment coordinates (data/nml_vertices/neuron_segments.csv)
Intestine geometry: Virtual Worm Blender model (Virtual_Worm_February_2012.blend),
                    extracted via extract_world_verts.py →
                    data/nml_vertices/blend_world_verts.csv
                    data/nml_vertices/blend_world_centroids.csv

Coordinate alignment
--------------------
Blender (Virtual Worm) and NeuroML c302 use different coordinate systems.
We derive a per-axis linear transform (scale + offset) by regressing Blender
world-space soma centroids against the NeuroML soma positions of the same neurons,
using only neurons whose processes are compact and confined to the head region
(so the mesh centroid ≈ soma position).

Excluded from calibration: CAN, HSN, AVM — these have long body-length axons
that shift the Blender centroid far from the soma.

Usage
-----
Prerequisites:
  1. Run extract_world_verts.py via Blender (one-time step):
       /Applications/Blender.app/Contents/MacOS/blender \\
           --background Virtual_Worm_February_2012.blend \\
           --python extract_world_verts.py
  2. Then run this script normally:
       python nml_intestine_proximity.py [--out-dir data/nml_vertices]

Outputs (in --out-dir)
----------------------
  intestine_verts_nml.csv        — intestine vertices transformed to NeuroML µm
  neurite_nn_distances.csv       — per-segment-point NN distance to intestine
  intestine_proximity_metrics.csv— per-neuron-type summary metrics
  intestine_proximity.png / .pdf — figure
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial import cKDTree
from scipy import stats
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams["pdf.fonttype"]    = 42
matplotlib.rcParams["ps.fonttype"]     = 42
matplotlib.rcParams["font.family"]     = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator

# ── CLI ──────────────────────────────────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument('--out-dir', default='data/nml_vertices')
ap.add_argument('--contact-threshold', type=float, default=10.0,
                help='µm cutoff for "in contact" (default 10)')
args = ap.parse_args()
OUT = Path(args.out_dir)
OUT.mkdir(parents=True, exist_ok=True)

THRESH = args.contact_threshold   # µm

# ── Neuron class map ──────────────────────────────────────────────────────────
CLASS_MAP = {
    'CANL':'CAN','CANR':'CAN',
    'PDEL':'PDE','PDER':'PDE',
    'HSNL':'HSN','HSNR':'HSN',
    'AVM':'AVM',
    'RICL':'RIC','RICR':'RIC',
    'RIML':'RIM','RIMR':'RIM',
    'RIH':'RIH','RIR':'RIR',
    'NSML':'NSM','NSMR':'NSM',
    'ADFL':'ADF','ADFR':'ADF',
    'ASIL':'ASI','ASIR':'ASI',
    'ADEL':'ADE','ADER':'ADE',
    'CEPDL':'CEP','CEPDR':'CEP','CEPVL':'CEP','CEPVR':'CEP',
    'VC4':'VC4/5','VC5':'VC4/5',
}

# ── 1. Load neurite segments ──────────────────────────────────────────────────
segs = pd.read_csv(OUT / 'neuron_segments.csv')
segs['cls'] = segs['neuron'].map(CLASS_MAP)
segs = segs.dropna(subset=['cls'])

# Use both proximal and distal endpoints of each segment as neurite sample points
prox = segs[['cls','neuron','seg_id','prox_x','prox_y','prox_z']].rename(
    columns={'prox_x':'x','prox_y':'y','prox_z':'z'})
dist = segs[['cls','neuron','seg_id','dist_x','dist_y','dist_z']].rename(
    columns={'dist_x':'x','dist_y':'y','dist_z':'z'})
prox['endpoint'] = 'prox'
dist['endpoint'] = 'dist'
neurites = pd.concat([prox, dist], ignore_index=True).drop_duplicates(
    subset=['neuron','seg_id','x','y','z'])

# ── 2. Coordinate calibration: Blender world → NeuroML µm ────────────────────
centroids = pd.read_csv(OUT / 'blend_world_centroids.csv')
soma_nml  = (segs[segs['seg_id'] == 0]
             [['neuron','prox_x','prox_y','prox_z']]
             .rename(columns={'prox_x':'nx','prox_y':'ny','prox_z':'nz'})
             .drop_duplicates('neuron'))
soma_blend = (centroids[centroids['mesh'].isin(soma_nml['neuron'])]
              [['mesh','cx','cy','cz']]
              .rename(columns={'mesh':'neuron','cx':'bx','cy':'by','cz':'bz'}))
paired = soma_nml.merge(soma_blend, on='neuron')

# Calibration neurons: compact, head-confined (centroid ≈ soma).
# Exclude CAN, HSN, AVM whose long axons shift centroid far from soma.
EXCLUDE_CAL = {'CAN','CANL','CANR','HSN','HSNL','HSNR','AVM','VC4','VC5'}
cal = paired[~paired['neuron'].isin(EXCLUDE_CAL)].copy()
print(f"Calibration control points: {len(cal)} neurons")

# Y: free regression — spans the full A-P axis, well-constrained.
# X, Z: force isotropic scale (= Y slope). Compact head calibration neurons
# cluster in a tiny lateral range so free regression fits noise (R²≈0.46/0.64).
# With slope fixed, the intercept is just mean(NML - slope·Blend), i.e. the
# mean lateral offset between coordinate origins.
transforms = {}
by_vals = cal['by'].values
ny_vals = cal['ny'].values
y_slope, y_intercept, y_r, _, _ = stats.linregress(by_vals, ny_vals)
y_resid = ny_vals - (y_slope * by_vals + y_intercept)
print(f"Calibration control points: {len(cal)} neurons")
print(f"  Y: NML_y = {y_slope:.2f}×Blend_y + {y_intercept:.2f}"
      f"  R²={y_r**2:.4f}  RMSE={np.sqrt(np.mean(y_resid**2)):.1f} µm")
transforms['y'] = (y_slope, y_intercept)

for axis in ('x', 'z'):
    b = cal[f'b{axis}'].values
    n = cal[f'n{axis}'].values
    slope = y_slope                      # isotropic: same scale as Y
    intercept = float(np.mean(n - slope * b))
    residuals = n - (slope * b + intercept)
    rmse = np.sqrt(np.mean(residuals**2))
    r = float(np.corrcoef(b, n)[0, 1]) if b.std() > 0 else 0.0
    transforms[axis] = (slope, intercept)
    print(f"  {axis.upper()}: NML_{axis} = {slope:.2f}×Blend_{axis} + {intercept:.2f}"
          f"  (isotropic scale, offset from {len(cal)}-neuron mean)"
          f"  R²={r**2:.4f}  RMSE={rmse:.1f} µm")

def blend_to_nml(bx, by, bz):
    sx, ox = transforms['x']
    sy, oy = transforms['y']
    sz, oz = transforms['z']
    return sx*bx + ox, sy*by + oy, sz*bz + oz

# ── 3. Transform intestine vertices to NeuroML space ─────────────────────────
verts = pd.read_csv(OUT / 'blend_world_verts.csv')
int_meshes = [m for m in verts['mesh'].unique()
              if m.startswith('int') or m == 'Pharyn-Intest-Valve']
intestine_blend = verts[verts['mesh'].isin(int_meshes)].copy()

intestine_blend['nx'], intestine_blend['ny'], intestine_blend['nz'] = blend_to_nml(
    intestine_blend['x'].values,
    intestine_blend['y'].values,
    intestine_blend['z'].values)

intestine_nml = intestine_blend[['mesh','nx','ny','nz']].rename(
    columns={'nx':'x','ny':'y','nz':'z'})
intestine_nml.to_csv(OUT / 'intestine_verts_nml.csv', index=False)
print(f"\nIntestine vertices in NeuroML coords: {len(intestine_nml)}")
print(f"  y range: {intestine_nml.y.min():.1f} – {intestine_nml.y.max():.1f} µm")
print(f"  x range: {intestine_nml.x.min():.1f} – {intestine_nml.x.max():.1f} µm")
print(f"  z range: {intestine_nml.z.min():.1f} – {intestine_nml.z.max():.1f} µm")

# ── 4. Nearest-neighbour distances ───────────────────────────────────────────
int_pts = intestine_nml[['x','y','z']].values
tree    = cKDTree(int_pts)

neu_pts = neurites[['x','y','z']].values
dists, nn_idx = tree.query(neu_pts, workers=-1)

neurites = neurites.copy()
neurites['d_nn']     = dists
neurites['nn_int_y'] = int_pts[nn_idx, 1]   # y of the closest intestine point

neurites.to_csv(OUT / 'neurite_nn_distances.csv', index=False)

# ── 5. Arc-length via parent→child inter-node distances ──────────────────────
# Most NeuroML segments here are "point segments" (prox == dist), so the
# morphology is encoded as a tree of 3D nodes where arc length lives in the
# parent→child edges, not within individual segments.
# Strategy: for each segment, its node = distal point; edge length =
# Euclidean distance from parent's distal to this segment's distal.
# Root segment (no parent) contributes prox→dist length only.

# Build node d_nn lookup: for each (neuron, seg_id) → d_nn of its distal point
dist_d_lookup = (neurites[neurites['endpoint'] == 'dist']
                 .set_index(['neuron', 'seg_id'])['d_nn'])

segs2 = segs.copy()
segs2['cls'] = segs2['neuron'].map(CLASS_MAP)
segs2 = segs2.dropna(subset=['cls'])

# Parent node positions: dist of parent segment
parent_pos = (segs2[['neuron', 'seg_id', 'dist_x', 'dist_y', 'dist_z']]
              .rename(columns={'seg_id':'parent_seg',
                               'dist_x':'par_x','dist_y':'par_y','dist_z':'par_z'}))
segs2 = segs2.merge(parent_pos, on=['neuron','parent_seg'], how='left')

# Edge length: parent.dist → child.dist (= child.prox for point segs)
segs2['edge_len'] = np.where(
    segs2['par_x'].isna(),
    # root: use prox→dist length
    np.sqrt((segs2.dist_x-segs2.prox_x)**2 +
            (segs2.dist_y-segs2.prox_y)**2 +
            (segs2.dist_z-segs2.prox_z)**2),
    # non-root: parent.dist → child.dist
    np.sqrt((segs2.dist_x-segs2.par_x)**2 +
            (segs2.dist_y-segs2.par_y)**2 +
            (segs2.dist_z-segs2.par_z)**2)
)

# d_nn per segment: minimum d_nn across ALL endpoints of that segment.
# For point segs (prox==dist) the dedup keeps one entry (prox); for proper
# segs both endpoints exist. Group by (neuron, seg_id) over all endpoints.
seg_min_d = (neurites.groupby(['neuron','seg_id'])['d_nn'].min()
             .rename('d_node'))
segs2 = segs2.join(seg_min_d, on=['neuron','seg_id'])

# Parent's d_nn (also needed for edge proximity)
parent_d = seg_min_d.rename('d_parent')
segs2['parent_seg_int'] = segs2['parent_seg'].astype('Int64')
segs2 = segs2.join(parent_d, on=['neuron','parent_seg_int'])
segs2['d_seg_min'] = segs2[['d_node','d_parent']].min(axis=1)

# ── 6. Per-neuron-class summary metrics ───────────────────────────────────────
rows = []
for cls, grp in neurites.groupby('cls'):
    d = grp['d_nn'].values
    seg_grp = segs2[segs2['cls']==cls]
    contact_len = seg_grp.loc[seg_grp['d_seg_min'] <= THRESH, 'edge_len'].sum()
    total_len   = seg_grp['edge_len'].sum()

    rows.append({
        'neuron_class':   cls,
        'd_min_um':       float(np.min(d)),
        'd_p25_um':       float(np.percentile(d, 25)),
        'd_median_um':    float(np.median(d)),
        'd_p75_um':       float(np.percentile(d, 75)),
        'frac_within_thresh': float(np.mean(d <= THRESH)),
        'contact_length_um': float(contact_len),
        'total_length_um':   float(total_len),
        'n_points':       len(d),
    })

metrics = pd.DataFrame(rows).sort_values('d_min_um')
metrics.to_csv(OUT / 'intestine_proximity_metrics.csv', index=False)

print(f"\nProximity metrics (threshold = {THRESH} µm):")
print(metrics[['neuron_class','d_min_um','d_median_um',
               'contact_length_um','total_length_um']].to_string(index=False))

# ── 7. Figure ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(11, 5),
                          gridspec_kw={'width_ratios':[1.8, 1.4, 1]})

# Colour by minimum distance
def dist_color(d):
    if d <= 10:   return '#2166ac'   # dark blue — close
    if d <= 30:   return '#74add1'   # light blue — moderate
    return '#bdbdbd'                 # grey — distant

# Panel A: distribution of NN distances per neuron class
ax = axes[0]
cls_order = metrics['neuron_class'].tolist()
for i, cls in enumerate(cls_order):
    d = neurites[neurites['cls']==cls]['d_nn'].values
    col = dist_color(metrics.loc[metrics['neuron_class']==cls, 'd_min_um'].values[0])
    ax.scatter(d, [i]*len(d), s=4, color=col, alpha=0.4, zorder=2)
    ax.plot([np.median(d)], [i], 'o', color=col, ms=6, zorder=3,
            markeredgecolor='white', markeredgewidth=0.5)

ax.axvline(THRESH, color='#e08030', lw=1, ls='--', label=f'{THRESH} µm threshold')
ax.set_yticks(range(len(cls_order)))
ax.set_yticklabels(cls_order, fontsize=8.5, fontstyle='italic')
ax.set_xlabel('Distance to nearest intestine vertex (µm)', fontsize=8.5)
ax.set_title('A   NN distances to intestine', fontsize=9.5, loc='left', fontweight='bold')
ax.legend(fontsize=7.5, loc='upper right')
ax.spines[['top','right']].set_visible(False)
ax.set_xlim(left=0)

# Panel B: contact length bar chart
ax2 = axes[1]
cls_b = metrics.sort_values('contact_length_um', ascending=False)
cols_b = [dist_color(r['d_min_um']) for _, r in cls_b.iterrows()]
ax2.barh(range(len(cls_b)), cls_b['contact_length_um'], color=cols_b,
         edgecolor='white', lw=0.4, height=0.7)
ax2.set_yticks(range(len(cls_b)))
ax2.set_yticklabels(cls_b['neuron_class'], fontsize=8.5, fontstyle='italic')
ax2.set_xlabel(f'Process length within {THRESH} µm of intestine (µm)', fontsize=8.5)
ax2.set_title('B   Contact length', fontsize=9.5, loc='left', fontweight='bold')
ax2.spines[['top','right']].set_visible(False)

# Panel C: cross-section scatter at y ∈ [0, +250] µm (body region with intestine)
ax3 = axes[2]
mid_int = intestine_nml[(intestine_nml['y'] >= 0) & (intestine_nml['y'] <= 250)]
ax3.scatter(mid_int['x'], mid_int['z'], s=3, color='#fda060', alpha=0.5, zorder=1, label='Intestine')
mid_neu = neurites[(neurites['y'] >= 0) & (neurites['y'] <= 250)]
mid_neu = mid_neu[mid_neu['cls'].isin(['CAN','PDE','HSN'])]
cpal = {'CAN':'#2166ac','PDE':'#2166ac','HSN':'#74add1'}
for cls, grp in mid_neu.groupby('cls'):
    ax3.scatter(grp['x'], grp['z'], s=8, color=cpal.get(cls,'#bdbdbd'),
                alpha=0.6, zorder=2, label=cls)
ax3.set_xlabel('X (µm)', fontsize=8.5)
ax3.set_ylabel('Z (µm)', fontsize=8.5)
ax3.set_title('C   Cross-section\n(y = 0–250 µm)', fontsize=9.5, loc='left', fontweight='bold')
ax3.legend(fontsize=7, markerscale=1.5, framealpha=0.9)
ax3.spines[['top','right']].set_visible(False)
ax3.axhline(0, color='#ddd', lw=0.5); ax3.axvline(0, color='#ddd', lw=0.5)

fig.tight_layout(w_pad=1.5)
fig.savefig(OUT / 'intestine_proximity.png', dpi=180, bbox_inches='tight')
fig.savefig(OUT / 'intestine_proximity.pdf', bbox_inches='tight')
print(f"\nFigure saved to {OUT}/intestine_proximity.png")
plt.close()
