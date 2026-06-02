"""
DEG enrichment vs neuron-specificity threshold, for all cat-1+ neurons.

Layout: 3 rows (TPM min = 1, 10, 25) × 3 columns (L4, Adult, Ghaddar).
Each panel shows enrichment curves for all available cat-1+ neurons as
pct_max threshold is swept from 1% to 50%.  CAN = blue, others = grey.

For neuron NEU at (tpm_min, pct_max):
  N = all measured genes
  K = genes where NEU_TPM >= tpm_min AND NEU_TPM / gene_max >= pct_max
  n = all DEGs in measured genes
  k = DEGs satisfying the same two conditions
Enrichment = (k/n) / (K/N), p-value from hypergeometric.

Ghaddar gene IDs are 8-digit WBGene (e.g. WBGene00021160); Taylor IDs have a
trailing version digit (e.g. WBGene000211602).  DEG IDs are stripped of the
last character before matching.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import hypergeom
from pathlib import Path


# ── Load Taylor data ──────────────────────────────────────────────────────────
def load_taylor(stem):
    df = pd.read_csv(f'Taylor_{stem}_GenesExpressing-BATCH-thrs2.csv')
    df = df.rename(columns={'Unnamed: 0': 'wbgene'}).set_index('wbgene')
    cell_cols = [c for c in df.columns if c != 'gene_name']
    return df, cell_cols

t_adult_full, ccols_a = load_taylor('Adult')
t_l4_full,    ccols_l = load_taylor('L4')

# Pseudogenes excluded from all DEG lists
PSEUDOGENE_EXCLUDE = {
    'WBGene000453662',   # Y17D7C.3 — pseudogene, absent from Ghaddar
}

t_adult_deg = pd.read_csv('Taylor_Adult_GenesExpressing-BATCH-thrs2_DEGfiltered.csv')
t_l4_deg    = pd.read_csv('Taylor_L4_GenesExpressing-BATCH-thrs2_DEGfiltered.csv')
for df in [t_adult_deg, t_l4_deg]:
    df.rename(columns={'Unnamed: 0': 'wbgene'}, inplace=True)
    df.set_index('wbgene', inplace=True)
    if 'gene_name' in df.columns:
        df.drop(columns='gene_name', inplace=True)
    df.drop(index=df.index.intersection(PSEUDOGENE_EXCLUDE), inplace=True)


# ── Load Ghaddar data ─────────────────────────────────────────────────────────
print('Loading Ghaddar data...')
_graw = pd.read_csv('Ghaddar_prcnt_tpm_bootstrap.csv',
                    usecols=['gene_ID', 'cell_type', 'scaled_TPM'])
g_full_all = _graw.pivot_table(index='gene_ID', columns='cell_type',
                                values='scaled_TPM', aggfunc='first')
g_full_all.index.name = 'wbgene'

# Restrict to neurons present in Taylor adult.
# Start with exact name matches (88), then add Ghaddar merged types that cover
# Taylor neurons — these contribute to the gene_max denominator.
_t_cols = set(c for c in t_adult_full.columns if c != 'gene_name')
g_neuron_cols = [c for c in g_full_all.columns if c in _t_cols]

# Ghaddar merged types → constituent Taylor neuron names (for reference only;
# we include them if the merged column exists in g_full_all)
_GHADDAR_MERGED = [
    'CEP_ADE_PDE',   # CEP, ADE, PDE
    'DA_VA',          # DA, VA
    'DB_VB',          # DB, VB
    'VD_DD',          # VD, DD
    'PHA_PHB',        # PHA, PHB
    'PLM_ALM',        # PLM, ALM
    'PVD_FLP',        # PVD, FLP
    'URX_AQR_PQR',   # URX, AQR, PQR
    'SDQ',            # SDQL, SDQR
    'AWC',            # AWC_OFF, AWC_ON
    'ASE',            # ASEL, ASER
    'IL1',            # IL1_DV, IL1_LR
    'IL2',            # IL2_DV, IL2_LR
    'RMD',            # RMD_DV  (RMD_LR already in exact overlap)
    'RME',            # RME_DV, RME_LR
    'PLN',            # ALN_PLN
]
for _mt in _GHADDAR_MERGED:
    if _mt in g_full_all.columns:
        g_neuron_cols.append(_mt)

g_full = g_full_all[g_neuron_cols].copy()
g_full.index.name = 'wbgene'
gcols = g_neuron_cols

# DEG IDs from Taylor have a trailing version digit; strip to match Ghaddar
g_l4_deg_idx    = pd.Index([x[:-1] for x in t_l4_deg.index])
g_adult_deg_idx = pd.Index([x[:-1] for x in t_adult_deg.index])
g_l4_deg    = pd.DataFrame(index=g_l4_deg_idx)
g_adult_deg = pd.DataFrame(index=g_adult_deg_idx)
print(f'  Ghaddar matrix (neuronal subset): {g_full.shape[0]} genes × {g_full.shape[1]} cell types')
print(f'  Adult DEGs matched: {g_full.index.isin(g_adult_deg_idx).sum()} / {len(g_adult_deg_idx)}')

CAT1_NEURONS = ['ADE', 'ADF', 'AVM', 'CAN', 'CEP', 'HSN', 'NSM', 'PDE',
                'RIC', 'RIH', 'RIM', 'RIR', 'ASI', 'VC_4_5']

TPM_MINS  = [1, 10, 25]
PCT_THRS  = [0.01, 0.05, 0.10, 0.20, 0.33, 0.50]
XLABELS   = ['1%', '5%', '10%', '20%', '33%', '50%']


def compute_enrichment(full, deg, cell_cols, neu, tpm_min, pct_thrs):
    if neu not in full.columns:
        return None

    measured  = full[full[cell_cols].notna().any(axis=1)]
    row_max   = measured[cell_cols].max(axis=1)
    neu_expr  = measured[neu].fillna(0)

    degs_m    = deg[deg.index.isin(measured.index)]
    neu_d     = neu_expr.reindex(degs_m.index)
    row_max_d = row_max.reindex(degs_m.index)

    N = len(measured)
    n = len(degs_m)

    rows = []
    for pct in pct_thrs:
        K = int(((neu_expr >= tpm_min) & ((neu_expr / row_max) >= pct)).sum())
        k = int(((neu_d   >= tpm_min) & ((neu_d   / row_max_d) >= pct)).sum())

        bg  = K / N if N else 0
        obs = k / n if n else 0
        enr = obs / bg if bg else 0
        p   = hypergeom.sf(k - 1, N, K, n) if K > 0 and n > 0 and k > 0 else 1.0

        rows.append(dict(pct=pct, N=N, K=K, n=n, k=k,
                         bg=bg, obs=obs, enr=enr, p=p))
    return rows


# ── Figure: 3 rows (tpm_min) × 2 columns (dataset) ───────────────────────────
DATASETS = [
    ('Taylor L4',    t_l4_full,    t_l4_deg,    ccols_l),
    ('Taylor adult', t_adult_full, t_adult_deg, ccols_a),
    ('Ghaddar (adult)', g_full,  g_adult_deg, gcols),
]

n_rows = len(TPM_MINS)
n_cols = len(DATASETS)
fig, axes = plt.subplots(n_rows, n_cols,
                          figsize=(5.8 * n_cols, 3.8 * n_rows),
                          sharey=False)

x = np.arange(len(PCT_THRS))
X_LABEL_OFFSET = 0.18   # data units right of last x tick

for row_i, tpm_min in enumerate(TPM_MINS):
    for col_i, (ds_name, full, deg, cell_cols) in enumerate(DATASETS):
        ax = axes[row_i, col_i]
        available = [c for c in CAT1_NEURONS if c in cell_cols]

        # ── first pass: plot lines, collect label positions ──────────────────
        label_positions = []   # (y_end, neu, color)
        for neu in available:
            res = compute_enrichment(full, deg, cell_cols, neu, tpm_min, PCT_THRS)
            if res is None:
                continue
            enr    = [r['enr'] for r in res]
            p_vals = [r['p']   for r in res]
            is_can = (neu == 'CAN')

            color = '#1a5fa8' if is_can else '#aaaaaa'
            lw    = 2.0       if is_can else 0.8
            zo    = 5         if is_can else 2
            alpha = 1.0       if is_can else 0.5

            ax.plot(x, enr, color=color, lw=lw, zorder=zo, alpha=alpha)
            for xi, (e, p) in enumerate(zip(enr, p_vals)):
                if p < 0.05:
                    sz = 40 if is_can else 20
                    ax.scatter(xi, e, color=color, s=sz,
                               alpha=1.0 if is_can else 0.7, zorder=6)

            has_sig = any(p < 0.05 for p in p_vals)
            if is_can or has_sig:
                label_positions.append((enr[-1], neu, color, is_can))

        # ── second pass: place labels, spreading overlapping ones ────────────
        # Sort by y, then nudge apart any that are closer than min_gap
        label_positions.sort(key=lambda t: t[0])
        min_gap = 0.10
        placed_y = []
        for y_raw, neu, color, is_can in label_positions:
            y = y_raw
            for py in reversed(placed_y):
                if y - py < min_gap:
                    y = py + min_gap
            placed_y.append(y)
            ax.text(x[-1] + X_LABEL_OFFSET, y, neu,
                    fontsize=6.5 if is_can else 6,
                    color='#1a5fa8' if is_can else '#888888',
                    fontweight='bold' if is_can else 'normal',
                    va='center', clip_on=False, zorder=7)

        ax.axhline(1, color='k', lw=0.7, ls='--', alpha=0.35)
        ax.set_xticks(x)
        ax.set_xticklabels(XLABELS, fontsize=8)
        ax.set_xlim(x[0] - 0.3, x[-1] + 1.4)
        ax.set_xlabel('Neuron / gene-max threshold', fontsize=8)
        ax.set_ylabel('Enrichment', fontsize=8)
        ax.set_title(f'{ds_name}  |  NEU ≥ {tpm_min} TPM', fontsize=9)
        ax.spines[['top', 'right']].set_visible(False)

        res_can = compute_enrichment(full, deg, cell_cols, 'CAN', tpm_min, PCT_THRS)
        if res_can:
            ax.text(0.02, 0.97,
                    f'n={res_can[0]["n"]} DEGs / N={res_can[0]["N"]} genes',
                    transform=ax.transAxes, fontsize=7,
                    ha='left', va='top', color='#666666')

fig.suptitle('cest-2.1 DEG enrichment — all cat-1+ neurons\n'
             'CAN = blue (filled dot = p<0.05); grey = other cat-1+ neurons',
             fontsize=9, y=1.005)
fig.tight_layout()

out = Path('data')
fig.savefig(out / 'can_deg_enrichment_by_threshold.png', dpi=150,
            bbox_inches='tight')
fig.savefig(out / 'can_deg_enrichment_by_threshold.pdf', bbox_inches='tight')
plt.close()
print('Saved data/can_deg_enrichment_by_threshold.png')


HDR = (f'  {"neuron":>8}  {"tpm_min":>7}  {"pct%":>5}  {"N":>5}  {"K":>5}'
       f'  {"n":>4}  {"k":>3}  {"bg%":>6}  {"obs%":>5}  {"enr":>5}  {"p":>7}')

def collect_rows(ds_name, full, deg, cell_cols, neurons):
    rows = []
    for tpm_min in TPM_MINS:
        for neu in neurons:
            res = compute_enrichment(full, deg, cell_cols, neu, tpm_min, PCT_THRS)
            if res is None:
                continue
            for r in res:
                rows.append(dict(dataset=ds_name, neuron=neu, tpm_min=tpm_min,
                                 pct_max=r['pct'], N=r['N'], K=r['K'],
                                 n=r['n'], k=r['k'],
                                 bg_frac=round(r['bg'], 6),
                                 obs_frac=round(r['obs'], 6),
                                 enrichment=round(r['enr'], 4),
                                 p_hypergeom=round(r['p'], 6),
                                 sig=(r['p'] < 0.05)))
    return rows

def print_table(ds_name, rows):
    print(f'\n{"="*60}')
    print(f'{ds_name}')
    print(HDR)
    for r in rows:
        sig = '*' if r['sig'] else ' '
        print(f'  {r["neuron"]:>8}  {r["tpm_min"]:>7}  {r["pct_max"]*100:>4.0f}%'
              f'  {r["N"]:>5}  {r["K"]:>5}  {r["n"]:>4}  {r["k"]:>3}'
              f'  {r["bg_frac"]*100:>5.1f}%  {r["obs_frac"]*100:>4.0f}%'
              f'  {r["enrichment"]:>5.2f}  {r["p_hypergeom"]:>7.4f}{sig}')

# ── Collect all results ───────────────────────────────────────────────────────
all_can_rows, all_neuron_rows = [], []
for ds_name, full, deg, cell_cols in DATASETS:
    available = [c for c in CAT1_NEURONS if c in cell_cols]
    can_rows = collect_rows(ds_name, full, deg, cell_cols, ['CAN'])
    all_rows = collect_rows(ds_name, full, deg, cell_cols, available)
    all_can_rows.extend(can_rows)
    all_neuron_rows.extend(all_rows)

# ── Print tables ──────────────────────────────────────────────────────────────
print('\n\n── TABLE 1: CAN ──')
for ds_name in [d[0] for d in DATASETS]:
    print_table(ds_name, [r for r in all_can_rows if r['dataset'] == ds_name])

print('\n\n── TABLE 2: all cat-1+ neurons ──')
for ds_name in [d[0] for d in DATASETS]:
    print_table(ds_name, [r for r in all_neuron_rows if r['dataset'] == ds_name])

# ── Save CSVs ─────────────────────────────────────────────────────────────────
can_df = pd.DataFrame(all_can_rows)
can_df.to_csv(out / 'deg_enrichment_can_statistics.csv', index=False)

all_df = pd.DataFrame(all_neuron_rows)
all_df.to_csv(out / 'deg_enrichment_all_neurons_statistics.csv', index=False)

print('\nSaved data/deg_enrichment_can_statistics.csv')
print('Saved data/deg_enrichment_all_neurons_statistics.csv')
