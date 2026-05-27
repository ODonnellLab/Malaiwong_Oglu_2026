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

| Path | Description |
|------|-------------|
| `batch_postural_comparison.py` | Locomotion analysis — metrics, LME, figure |
| `sos_condition_plots.py` | SOS response time — 4-condition barplot + ECDF supplemental figure |
| `sos_analysis.py` | SOS combined analysis — strip plot, ECDF, speed correlation |
| `exclude.csv` | Genotype/date pairs excluded before normalization |
| `test_reproduce.py` | Reproduction test: demo (no NAS) and full NAS rescan |
| `data/supplemental_particle_data.csv` | Per-particle raw metrics (13,709 particles) |
| `data/fwd_frame_data.parquet` | Per-forward-run-frame speed data used by LME |
| `data/particle_data.parquet` | Per-particle cache |
| `data/recording_data.parquet` | Per-recording summary with normalized values |
| `data/postural_comparison_stats.csv` | LME fold-change estimates and FDR q-values |
| `data/postural_comparison.csv` | Per-recording summary CSV |
| `data/postural_comparison.pdf` | Locomotion figure (8.5 × 11 in, vector text) |
| `data/postural_comparison.png` | Locomotion figure (raster) |
| `data/sos_condition_animal_data.parquet` | Per-animal SOS response times across 4 conditions (N2, cest-2.1, tbh-1) |
| `data/sos_condition_lmm_stats.csv` | Pre-computed LMM pairwise contrasts for each condition |
| `data/sos_condition_plots.pdf` | SOS supplemental figure (8.5 × 11 in, vector text) |
| `data/sos_condition_plots.png` | SOS supplemental figure (raster) |
| `data/sos_animal_data.parquet` | Per-animal SOS data — 20 min off-food / 33% octanol, all genotypes |
| `data/sos_stats.csv` | LMM time ratios vs N2 — all genotypes, 20 min off-food / octanol |
| `data/speed_sos_correlation.pdf` | Speed vs response time correlation figure |
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

## Methods

See [ODLabTracker — Locomotion analysis methods](https://github.com/ODonnellLab/ODLabTracker/blob/main/dev/LOCOMOTION_ANALYSIS_METHODS.md)
for full statistical methods including the LME model structure, normalization strategy,
and R code to reproduce the model directly from `supplemental_particle_data.csv`.
