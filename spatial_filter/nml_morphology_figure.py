"""
Figure: cat-1+ neuron process morphology vs. intestine (NeuroML coordinates).
Spatial filter — step 2 of CAN identification rebuttal argument.

NOTE: requires data/neuron_segments.csv which is not currently in the repo.
This script is retained for reference; use blend_ap_axis_figure.py for the
current version of this figure.
"""
import argparse
import pandas as pd, numpy as np
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams["pdf.fonttype"]    = 42
matplotlib.rcParams["ps.fonttype"]     = 42
matplotlib.rcParams["font.family"]     = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator
import matplotlib.patheffects as pe

ap = argparse.ArgumentParser()
ap.add_argument('--input-dir', default='data',
                help='Directory containing neuron_segments.csv')
ap.add_argument('--out-dir',   default='data',
                help='Output directory for figures')
args = ap.parse_args()

from pathlib import Path
INPUT_DIR = Path(args.input_dir)
OUT_DIR   = Path(args.out_dir)

# ── Load & prep ──────────────────────────────────────────────────────────────
segs = pd.read_csv(INPUT_DIR / 'neuron_segments.csv')
for ax_ in ('x','y','z'):
    segs[f'm{ax_}'] = (segs[f'prox_{ax_}'] + segs[f'dist_{ax_}']) / 2

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
segs['cls'] = segs['neuron'].map(CLASS_MAP)

summary = segs.groupby('cls').agg(
    y_min=('my','min'), y_max=('my','max'),
    x_min=('mx','min'), x_max=('mx','max'),
    z_min=('mz','min'), z_max=('mz','max'),
).reset_index()

# Intestine bounds (NeuroML µm; pharynx-intestinal valve ≈ y = −130)
INT_Y0, INT_Y1 = -130, 380
INT_X0, INT_X1 = -16, +16   # intestine cross-section x
INT_Z0, INT_Z1 = -10, +20   # intestine cross-section z (slightly dorsal)

# Category: requires longitudinal overlap AND radial plausibility
# AVM is excluded by radial filter (runs ventrally, z < −44 in overlap zone)
FORCE_NONE = {'AVM'}   # confirmed ventral nerve cord, no intestinal contact

def categorize(row):
    if row['cls'] in FORCE_NONE:
        return 'none'
    # longitudinal overlap fraction of intestine length
    lo = max(row.y_min, INT_Y0)
    hi = min(row.y_max, INT_Y1)
    frac = max(0, hi - lo) / (INT_Y1 - INT_Y0)
    if frac > 0.5:
        return 'full'
    elif frac > 0.1:
        return 'partial'
    return 'none'

summary['cat'] = summary.apply(categorize, axis=1)

CAT_ORDER = {'full':0,'partial':1,'none':2}
summary['co'] = summary['cat'].map(CAT_ORDER)
summary = summary.sort_values(['co','y_max'], ascending=[True,False]).reset_index(drop=True)

COL  = {'full':'#2166ac','partial':'#74add1','none':'#bdbdbd'}
ECOL = {'full':'#1a4e8a','partial':'#5090b8','none':'#999999'}

# ── Figure layout ────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(8.5, 5.2))
gs  = fig.add_gridspec(1, 2, width_ratios=[2.6, 1], wspace=0.35,
                        left=0.12, right=0.97, top=0.88, bottom=0.10)
axA = fig.add_subplot(gs[0])
axB = fig.add_subplot(gs[1])

# ── Panel A: longitudinal extent ─────────────────────────────────────────────
BODY_MIN, BODY_MAX = -360, 425

# Intestine shading
axA.axvspan(INT_Y0, INT_Y1, color='#fdefd8', alpha=0.85, zorder=0)
axA.axvline(INT_Y0, color='#d07820', lw=0.9, ls='--', zorder=1)
axA.axvline(INT_Y1, color='#d07820', lw=0.9, ls='--', zorder=1)

# Anatomy reference lines (below plot area as x-axis annotations)
REF = [(-270,'Nerve\nring'), (-200,'Pharynx\nend'), (0,'Mid-\nbody'), (380,'Tail')]
for y, lbl in REF:
    axA.axvline(y, color='#bbbbbb', lw=0.5, ls=':', zorder=0)

# Bars
for i, row in summary.iterrows():
    col  = COL[row['cat']]
    ecol = ECOL[row['cat']]
    axA.barh(i, row.y_max - row.y_min, left=row.y_min,
             height=0.65, color=col, edgecolor=ecol, lw=0.5, zorder=2)

axA.set_yticks(range(len(summary)))
axA.set_yticklabels(summary['cls'], fontsize=9, fontstyle='italic')
axA.set_xlim(BODY_MIN, BODY_MAX)
axA.set_xlabel('Position along body axis (µm, NeuroML coordinates)', fontsize=8.5)
axA.spines[['top','right']].set_visible(False)
axA.xaxis.set_minor_locator(MultipleLocator(50))
axA.tick_params(axis='x', which='major', labelsize=8)

# Anatomy labels just above x-axis
for y, lbl in REF:
    axA.text(y, -0.8, lbl, ha='center', va='top', fontsize=6.0, color='#888888',
             transform=axA.get_xaxis_transform(), clip_on=False)

# Intestine label inside shading
axA.text((INT_Y0+INT_Y1)/2, len(summary)-0.3, 'Intestine\nregion',
         ha='center', va='top', fontsize=7.5, color='#c05500',
         style='italic', zorder=3)

axA.set_title('A     Longitudinal process extent — 14 cat-1⁺ neuron types',
              fontsize=9.5, loc='left', fontweight='bold', pad=6)

# Legend
handles = [
    mpatches.Patch(facecolor='#fdefd8', edgecolor='#d07820', label='Intestine region'),
    mpatches.Patch(color=COL['full'],    label='Intestine-adjacent'),
    mpatches.Patch(color=COL['partial'], label='Partial overlap (HSN)'),
    mpatches.Patch(color=COL['none'],    label='Head-confined'),
]
axA.legend(handles=handles, fontsize=7.5, loc='lower right',
           framealpha=0.92, edgecolor='#cccccc')

# ── Panel B: cross-section y ∈ [0, +300] ────────────────────────────────────
mid = segs[(segs['my'] >= 0) & (segs['my'] <= 300)].copy()
mid['cls'] = mid['neuron'].map(CLASS_MAP)
mid = mid.dropna(subset=['cls'])

# Intestine box
int_rect = mpatches.FancyBboxPatch(
    (INT_X0, INT_Z0), INT_X1-INT_X0, INT_Z1-INT_Z0,
    boxstyle='round,pad=1.5', lw=1.4,
    edgecolor='#d07820', facecolor='#fdefd8', alpha=0.85, zorder=1)
axB.add_patch(int_rect)
axB.text(0, (INT_Z0+INT_Z1)/2+1, 'Intestine', ha='center', va='center',
         fontsize=7, color='#b05000', style='italic', zorder=2)

# Only show neurons that reach this y-zone
present = mid.groupby('cls').size()
focus   = ['CAN','PDE','HSN']   # the only ones with y_max > 0
col_pts = {'CAN':COL['full'],'PDE':COL['full'],'HSN':COL['partial']}

for cls in focus:
    sub = mid[mid['cls']==cls]
    if sub.empty: continue
    axB.scatter(sub['mx'], sub['mz'], s=7, color=col_pts[cls],
                alpha=0.55, zorder=3)
    # Label at median position
    mx, mz = sub['mx'].median(), sub['mz'].median()
    axB.text(mx, mz, cls, fontsize=7.5, ha='center', va='center',
             color=col_pts[cls], fontstyle='italic', fontweight='bold',
             zorder=4,
             path_effects=[pe.withStroke(linewidth=2, foreground='white')])

axB.set_xlabel('X (µm)', fontsize=8.5)
axB.set_ylabel('Z (µm, ventral → dorsal)', fontsize=8.5)
axB.set_xlim(-35, 35)
axB.set_ylim(-75, 55)
axB.axhline(0, color='#dddddd', lw=0.5, zorder=0)
axB.axvline(0, color='#dddddd', lw=0.5, zorder=0)
axB.spines[['top','right']].set_visible(False)
axB.tick_params(labelsize=8)
axB.set_title('B     Cross-section\n(y = 0 – 300 µm)',
              fontsize=9.5, loc='left', fontweight='bold', pad=6)

out = str(OUT_DIR / 'intestine_proximity')
fig.savefig(out + '.png', dpi=180, bbox_inches='tight')
fig.savefig(out + '.pdf', bbox_inches='tight')
print("Saved", out + ".png / .pdf")
plt.close()
