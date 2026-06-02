"""
wormatlas_em_measure.py
=======================
Measure lateral nerve–intestine distance from EM cross-sections (WormImage N2U).

Workflow (ImageJ/Fiji)
----------------------
For each section image:
  1. Open image in ImageJ/Fiji
  2. Draw a straight line across the excretory canal outer boundary (longest axis).
     Analyze → Measure → record length in pixels → ec_diam_px
     (EC outer diameter ≈ 1.8 µm at mid-body, White 1986)
  3. Draw a straight line: nearest lateral nerve membrane → nearest intestinal
     cell membrane (shortest gap).
     Analyze → Measure → record length in pixels → dist_px
  4. Estimate body position from anatomy (see guide below).

Body position estimation guide (adult hermaphrodite, ~1000 µm body length)
---------------------------------------------------------------------------
What you see in the section          body_um estimate   body_range
--------------------------------------------------------------------
Excretory pore region, thin intestine  150–200          (120, 230)
Anterior intestine, no gonad           200–300          (180, 330)
Intestine + anterior gonad arm         300–420          (270, 450)
Intestine + two gonad arms (mid-body)  420–550          (380, 600)
Intestine + large oocytes, vulva       520–620          (460, 660)
Posterior gonad arm, large oocytes     600–750          (550, 800)
Intestine narrowing, near preanal      750–900          (700, 950)

Identity rules
--------------
- Excretory canal: large, nearly clear oval lumen; granular cytoplasm surround.
- Lateral nerve: small round-to-oval neuronal profiles immediately dorsal (and
  sometimes ventral) to the excretory canal lumen. CAN processes are among these.
- Intestine: large cells with dense granular cytoplasm, microvilli on luminal face,
  positioned medial to the excretory canal.

Usage
-----
1. Fill in MEASUREMENTS below.
2. Run: python wormatlas_em_measure.py
3. Outputs: data/wormatlas_em/em_distance_measurements.csv + figure
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# ── Calibration constant ──────────────────────────────────────────────────────
# Excretory canal outer diameter (White 1986 + WormAtlas: ~1.8 µm at mid-body).
# This is used to convert pixel measurements to µm on a per-image basis.
EC_OUTER_DIAM_UM = 1.8

# ── ENTER YOUR MEASUREMENTS HERE ─────────────────────────────────────────────
# Each dict is one image. Fields:
#   image      : WormImage ID (str) or descriptive name, e.g. 'can_b'
#   body_um    : estimated body position µm from nose (use guide above)
#   body_range : (lo, hi) uncertainty range for the x-axis error bar
#   ec_diam_px : EC outer diameter in pixels (your ImageJ measurement)
#   dist_px    : lateral nerve membrane → intestine membrane in pixels
#   dist_um    : alternatively, if scale bar gives direct µm, enter here (set dist_px=None)
#   notes      : anything notable (e.g. 'intestine partially out of frame')
#
MEASUREMENTS = [
    # Example — replace with real data:
    # dict(image='117865',  body_um=400, body_range=(300,500),
    #      ec_diam_px=None, dist_px=None, dist_um=None, notes=''),
    # dict(image='can_b',   body_um=350, body_range=(250,450),
    #      ec_diam_px=None, dist_px=None, dist_um=None, notes='WormAtlas CAN fig'),
]

# ─────────────────────────────────────────────────────────────────────────────

real = [m for m in MEASUREMENTS if m.get('dist_px') is not None or m.get('dist_um') is not None]
if not real:
    print("No measurements entered yet. Edit MEASUREMENTS in this script and re-run.")
    exit(0)

rows = []
for m in real:
    if m.get('dist_um') is not None:
        dist_um = m['dist_um']
        cal = 'scale_bar'
    elif m.get('dist_px') is not None:
        if m.get('ec_diam_px') is None:
            print(f"  WARNING: image '{m['image']}' has dist_px but no ec_diam_px — skipping")
            continue
        px_per_um = m['ec_diam_px'] / EC_OUTER_DIAM_UM
        dist_um = m['dist_px'] / px_per_um
        cal = 'EC_diam'
    else:
        continue

    lo, hi = m.get('body_range', (m['body_um'] - 50, m['body_um'] + 50))
    rows.append({
        'image':    m['image'],
        'body_um':  m['body_um'],
        'body_lo':  lo,
        'body_hi':  hi,
        'dist_um':  dist_um,
        'cal':      cal,
        'notes':    m.get('notes', ''),
    })

if not rows:
    print("No valid measurements to process.")
    exit(0)

df = pd.DataFrame(rows).sort_values('body_um').reset_index(drop=True)
OUT = Path('data/wormatlas_em')
OUT.mkdir(exist_ok=True)
df.to_csv(OUT / 'em_distance_measurements.csv', index=False)

print(f"\nEM measurements: {len(df)} sections")
print(df[['image','body_um','body_lo','body_hi','dist_um','notes']].to_string(index=False))
print(f"\nMedian lateral nerve–intestine distance: {df['dist_um'].median():.2f} µm")
print(f"Range: {df['dist_um'].min():.2f}–{df['dist_um'].max():.2f} µm")
print(f"All < 10 µm threshold: {(df['dist_um'] < 10).all()}")

# ── Figure ────────────────────────────────────────────────────────────────────
# Blender Y coordinate → µm from nose: body_um ≈ blender_y + BLENDER_OFFSET
# Derived from intestine extent: Blender (-268, 438) ≈ nose-relative (150, 860).
BLENDER_OFFSET = 420

# Blender AP-axis distance profiles (CANL and CANR, from Virtual Worm mesh)
BLEND_CANL = [(-300,16),(-200,27),(-100,27),(-50,27),(0,8),(100,5),(200,7),(300,5),(400,2)]
BLEND_CANR = [(-300,3),(-200,4),(-100,3),(0,5),(100,6),(150,10),(200,24),(300,20),(400,6)]

fig, ax = plt.subplots(figsize=(9, 4.5))

# Intestine extent
ax.axvspan(150, 860, color='#d4edda', alpha=0.25, label='Intestine extent (approx.)')

# Blender curves
by, bd = zip(*BLEND_CANL)
ax.plot([y + BLENDER_OFFSET for y in by], bd,
        color='#2166ac', lw=1.5, ls='--', alpha=0.55, label='CANL (Blender model)')
by, bd = zip(*BLEND_CANR)
ax.plot([y + BLENDER_OFFSET for y in by], bd,
        color='#6baed6', lw=1.5, ls='--', alpha=0.55, label='CANR (Blender model)')

# EM spot measurements
xerr_lo = df['body_um'] - df['body_lo']
xerr_hi = df['body_hi'] - df['body_um']
ax.errorbar(
    df['body_um'], df['dist_um'],
    xerr=[xerr_lo, xerr_hi],
    fmt='o', color='#1a5fa8', ms=7,
    ecolor='#1a5fa8', elinewidth=1, capsize=3,
    markeredgecolor='k', markeredgewidth=0.6,
    label='Lateral nerve–intestine (EM, N2U)',
    zorder=6,
)
for _, row in df.iterrows():
    ax.annotate(str(row['image']), (row['body_um'], row['dist_um']),
                fontsize=6, color='#1a5fa8',
                xytext=(4, 4), textcoords='offset points')

ax.axhline(10, color='#cc3300', lw=1, ls=':', label='10 µm proximity threshold')

ax.set_xlabel('Body position (µm from nose)', fontsize=10)
ax.set_ylabel('Distance to intestine (µm)', fontsize=10)
ax.set_title(
    'Lateral nerve–intestine distance along the body\n'
    'Blender Virtual Worm model (dashed) + N2U EM cross-sections (points, ±position estimate)',
    fontsize=9, loc='left'
)
ax.legend(fontsize=8, ncol=2)
ax.spines[['top', 'right']].set_visible(False)
ax.set_ylim(bottom=0)
ax.set_xlim(0, 1050)

fig.tight_layout()
fig.savefig(OUT / 'em_lateral_nerve_intestine.png', dpi=180, bbox_inches='tight')
fig.savefig(OUT / 'em_lateral_nerve_intestine.pdf', bbox_inches='tight')
print(f"\nFigure → {OUT/'em_lateral_nerve_intestine.png'}")
plt.close()
