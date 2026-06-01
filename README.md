# Malaiwong & Oglu 2026 — *C. elegans* locomotion data

Data and analysis code for the postural locomotion dataset from Malaiwong & Oglu 2026.
117 recordings across 32 genotypes; locomotion quantified using [ODLabTracker](https://github.com/ODonnellLab/ODLabTracker).

---

## Quick start — reproduce figures from cached data

No NAS or external data access required. All per-animal datasets are included in `data/`.

**Postural locomotion figure:**

```bash
git clone https://github.com/ODonnellLab/Malaiwong_Oglu_2026.git
cd Malaiwong_Oglu_2026
pip install git+https://github.com/ODonnellLab/ODLabTracker.git
python batch_postural_comparison.py --out-dir data/ --exclude exclude.csv
```

This loads the cached parquet files in `data/`, skips the LME fit (already cached in
`data/postural_comparison_stats.csv`), and regenerates the figure.

To re-run the LME models (requires R with lme4 and lmerTest):

```bash
python batch_postural_comparison.py --out-dir data/ --exclude exclude.csv --refit
```

**SOS response time — combined conditions figure (supplemental):**

```bash
python sos_condition_plots.py --out-dir data/
```

This loads the cached per-animal SOS data from `data/sos_condition_animal_data.parquet`
and regenerates the figure. To rescan from the raw ODLabPlotTools CSV files:

```bash
python sos_condition_plots.py --data-dir /path/to/SOS --out-dir data/
```

**SOS per-genotype figures (`data/sos_figures/`):**

```bash
python sos_figure_plots.py --data-dir /path/to/raw_figure_data --out-dir data/sos_figures/
```

Regenerates all per-genotype SOS barplot + ECDF figures. For the epistasis figure (1e):

```bash
python sos_epistasis_plots.py --data-dir /path/to/raw_figure_data --out-dir data/sos_figures/
```

---

## Full pipeline — from raw video to figure

If you have access to the raw tracking data on the NAS, you can reproduce the full pipeline
from video files.

### Step 1 — Track videos with ODLabTracker

Install ODLabTracker and run the postural tracker on each video:

```bash
git clone https://github.com/ODonnellLab/ODLabTracker.git
cd ODLabTracker
pip install -e .
python track.py -c configs/IR_medium.yaml -f path/to/video.avi
```

ODLabTracker produces a `tracks.csv` file in a `<video_name>_results/` folder next to each video.
Organize recordings so that the dataset root has one subdirectory per genotype, each containing
dated recording folders:

```
dataset/
├── N2/
│   ├── N2 on food-01222026123456-0000_results/
│   │   └── tracks.csv
│   └── ...
├── tph-1/
│   └── tph-1 on food-02062026134512-0000_results/
│       └── tracks.csv
└── ...
```

For large datasets, use the parallel batch runner (4–6 workers recommended to avoid NAS I/O saturation):

```bash
python run_fasttrack_parallel.py /path/to/dataset -c configs/IR_medium.yaml -j 4
```

### Step 2 — Compute locomotion metrics and fit statistical models

```bash
python batch_postural_comparison.py \
    --data-dir /path/to/dataset \
    --out-dir  results/ \
    --exclude  exclude.csv \
    --title    "Forward-run speed (fold-change vs N2)"
```

This scans all `tracks.csv` files, computes per-particle forward-run speed, reversal rate, and
pirouette rate, normalizes to same-date N2 controls, fits linear mixed-effects models via R/lme4,
and saves the figure and supplemental table to `results/`.

Subsequent runs reuse cached parquet files and skip the NAS scan:

```bash
python batch_postural_comparison.py --out-dir results/ --exclude exclude.csv
```

---

## Repository contents

### Analysis scripts

| Script | Description |
|--------|-------------|
| `batch_postural_comparison.py` | Locomotion analysis — metrics, LME, figure |
| `sos_style.py` | **Shared style module** — layout constants, colors, ECDF helpers used by all SOS scripts |
| `sos_condition_plots.py` | SOS 4-condition barplot + 2×2 ECDF (N2, cest-2.1, tbh-1) |
| `sos_epistasis_plots.py` | SOS epistasis barplot + paired ECDFs (figure 1e) |
| `sos_figure_plots.py` | All other per-genotype SOS barplot + ECDF figures |
| `sos_analysis.py` | SOS strip plot, ECDF, and speed correlation |
| `exclude.csv` | Genotype/date pairs excluded before normalization |
| `test_reproduce.py` | Reproduction test: demo (no NAS) and full NAS rescan |
| `SESSION_NOTES.md` | Figure style conventions and layout decisions |

### Data files

| Path | Description |
|------|-------------|
| `data/supplemental_particle_data.csv` | Per-particle raw metrics (13,709 particles) |
| `data/fwd_frame_data.parquet` | Per-forward-run-frame speed data used by LME |
| `data/particle_data.parquet` | Per-particle cache |
| `data/recording_data.parquet` | Per-recording summary with normalized values |
| `data/postural_comparison_stats.csv` | LME fold-change estimates and FDR q-values |
| `data/postural_comparison.csv` | Per-recording summary CSV |
| `data/postural_comparison.pdf` | Locomotion figure (8.5 × 11 in, vector text) |
| `data/postural_comparison.png` | Locomotion figure (raster) |
| `data/sos_condition_animal_data.parquet` | Per-animal SOS response times — 4 conditions (N2, cest-2.1, tbh-1) |
| `data/sos_condition_lmm_stats.csv` | Pre-computed LMM pairwise contrasts for each condition |
| `data/sos_condition_plots.pdf` | SOS 4-condition supplemental figure |
| `data/sos_figures/` | Per-genotype SOS barplot + ECDF figures (one PDF + PNG per dataset) |
| `data/sos_animal_data.parquet` | Per-animal SOS data — 20 min off-food / 33% octanol, all genotypes |
| `data/sos_stats.csv` | LMM time ratios vs N2 — all genotypes |
| `data/speed_sos_correlation.pdf` | Speed vs SOS response time correlation |
| `data/supplemental_sos_combined.pdf` | SOS strip plot + ECDF + correlation supplemental figure |
| `demo_data/` | 4-genotype subset for install verification (see below) |

---

## Verify your install with demo data

Run the automated reproduction test, which checks particle counts, N2 speed, and LME
fold-changes against expected values:

```bash
python test_reproduce.py          # demo only — no NAS required
```

To also verify the full NAS rescan (reproduces all 117 recordings from raw `tracks.csv` files):

```bash
python test_reproduce.py --data-dir /path/to/raw/dataset
```

Or run the analysis manually on the 4-genotype demo subset:

```bash
python batch_postural_comparison.py \
    --out-dir demo_data/ \
    --exclude exclude.csv \
    --title   "Demo — forward-run speed"
```

This writes `demo_data/postural_comparison.png`. Add `--refit` to re-run the LME models.

---

## Dependencies

- **Python**: ODLabTracker, numpy, pandas, matplotlib, scipy, pyarrow
- **R** (for `--refit`): lme4 (≥ 1.1), lmerTest (≥ 3.1)

Install R packages:

```r
install.packages(c("lme4", "lmerTest"))
```

---

## SOS figure pipeline

All per-genotype SOS figures share a common style defined in `sos_style.py`. The three
figure scripts (`sos_condition_plots.py`, `sos_epistasis_plots.py`, `sos_figure_plots.py`)
import from it so that bar width, ECDF panel size, font, and colors are consistent across
all figures without duplication.

### Regenerating SOS figures

Each script reads cached parquet files from `data/` (or rescans raw CSVs with `--data-dir`):

```bash
# 4-condition barplot (figure 1c supplemental)
python sos_condition_plots.py --out-dir data/

# Epistasis barplot + ECDFs (figure 1e)
python sos_epistasis_plots.py --out-dir data/sos_figures/

# All other per-genotype figures
python sos_figure_plots.py --data-dir /path/to/raw_figure_data --out-dir data/sos_figures/

# Single figure only
python sos_figure_plots.py --data-dir /path/to/raw_figure_data \
                           --out-dir data/sos_figures/ \
                           --file 3e_cest-2.1\ rescues_SOSdata.csv
```

### Grouping configurations

`sos_figure_plots.py` reads an optional YAML config alongside each CSV to control
bar grouping and x-axis segmentation. A config is required for any figure with
rescue lines or paired comparisons; figures without a config use flat uniform spacing.

Config files live in the raw data directory and are named `<csv_stem>_config.yaml`:

```yaml
# Example: 3e_cest-2.1 rescues_SOSdata_config.yaml
groups:
  - [N2]
  - [PHX3900, MOY0066, MOY0065]   # cest-2.1 | cest-2.1 [RIC] | cest-2.1 [GUT]
intra_spacing: 0.22   # center-to-center within a group (data units)
inter_spacing: 0.42   # center-to-center between groups (data units)
```

Groups reference Strain_IDs from the CSV. Within a group bars are drawn at `intra_spacing`
apart; a break in the x-axis line marks the boundary between groups.

### Adding new data and re-running the combined LMM

`sos_analysis.py` pools all per-genotype SOS data into a single combined log-normal
LMM to produce the speed–response correlation figure. It reads from the same raw CSV
directory used by the per-genotype figure scripts.

**Step 1 — Drop the new CSV into the raw data directory**

The raw figure CSVs are not currently included in the repository (data collection is
ongoing); they will be added at submission. Point `--data-dir` at wherever your local
copy lives. The script scans all `*.csv` files in that directory and filters to rows
where `Condition == "control"`, `Bacteria == "OP50"`, `Odor == "33% octanol"`. Any
file following the standard ODLabPlotTools column format will be picked up automatically.
If you are adding a file that was already partially covered by another CSV (e.g. an
updated panel), the script deduplicates on `(Genotype, Date, Response.time)` to
prevent double-counting.

**Step 2 — Check genotype name harmonization**

If the new file introduces a genotype name not seen before, add it to `GENOTYPE_MAP`
in `sos_analysis.py` before running. Unrecognised names pass through unchanged and
will not match the speed data, appearing in the summary but missing a speed fold-change.
To audit names before running:

```bash
python3 -c "
import pandas as pd, glob
raw = pd.concat([pd.read_csv(f) for f in glob.glob('/path/to/raw_figure_data/*.csv')])
ctrl = raw[(raw.Condition=='control') & (raw.Bacteria=='OP50') & (raw.Odor=='33% octanol')]
print(sorted(ctrl.Genotype.unique()))
"
```

Rescue lines and cell-type-specific expression lines should be listed in `DROP_GENOTYPES`
so they are excluded from the combined analysis (they are handled in the per-genotype
figures separately).

**Step 3 — Re-run with `--refresh`**

```bash
python sos_analysis.py \
    --data-dir /path/to/raw_figure_data \
    --out-dir  data/ \
    --refresh
```

`--refresh` forces a full re-scan of all CSVs and re-runs the LMM (equivalent to
`--refit`). Without it the script uses the cached parquet files in `data/` and skips
loading. Use `--refit` alone to re-run the LMM without re-scanning the data.

**Statistical model**

```
log(response_time) ~ genotype + (1 | date) + (1 | recording_id)
```

Fit in R (lme4, REML, bobyqa optimizer). If the model is singular, `recording_id`
is dropped and the model is re-fit with `(1 | date)` only. Contrasts vs N2 are
estimated via emmeans with Benjamini–Hochberg correction. See
`methods_sos_conditions.txt` for the full methods text.

**Outputs** (written to `--out-dir`):

| File | Description |
|------|-------------|
| `sos_animal_data.parquet` | Per-animal cache (re-used on subsequent runs) |
| `sos_recording_data.parquet` | Per-recording (date) summary cache |
| `sos_stats.csv` | LMM time ratios vs N2 |
| `speed_sos_summary.csv` | Combined speed + RT summary per genotype |
| `speed_sos_correlation.pdf/.png` | Correlation figure |

### Modifying the shared style

Edit `sos_style.py` to change any visual property across all SOS figures at once.
Key constants:

| Constant | Value | Effect |
|----------|-------|--------|
| `BAR_SCALE` | 0.68 in/unit | Physical bar width — same across all figures |
| `ECDF_W × ECDF_H` | 1.543 × 0.847 in | ECDF panel size — same across all figures |
| `CONTENT_H` | 2.160 in | Fixed content height (= 2 ECDF rows + gap) |
| `B_FLAT / B_ROT` | 0.54 / 0.72 in | Bottom margin for flat vs rotated x-labels |

See `SESSION_NOTES.md` for full layout conventions and typography rules.

---

## Methods

See [ODLabTracker — Locomotion analysis methods](https://github.com/ODonnellLab/ODLabTracker/blob/main/dev/LOCOMOTION_ANALYSIS_METHODS.md)
for full statistical methods including the LME model structure, normalization strategy,
and R code to reproduce the model directly from `supplemental_particle_data.csv`.
