# Locomotion — postural analysis

Forward-run speed, reversal rate, and pirouette rate across 32 genotypes × 117 recordings.
LME fold-changes relative to N2 with BH-corrected q-values.

## Quick start — reproduce figures from cached data

No NAS access required. Per-animal parquet files are cached in `data/`.

```bash
cd locomotion
python batch_postural_comparison.py
```

Loads cached parquets, skips the LME fit (already cached in `data/postural_comparison_stats.csv`),
and writes `data/postural_comparison.png/.pdf`.

To re-run the LME models (requires R with lme4 + lmerTest):

```bash
python batch_postural_comparison.py --refit
```

To rescan raw tracking data from the NAS and refit everything:

```bash
python batch_postural_comparison.py --data-dir /path/to/dataset --refresh
```

## Full pipeline — from raw video

### Step 1 — Track videos with ODLabTracker

```bash
pip install git+https://github.com/ODonnellLab/ODLabTracker.git
python track.py -c configs/IR_medium.yaml -f path/to/video.avi
```

Organize recordings one subdirectory per genotype, each containing dated recording folders:

```
dataset/
├── N2/
│   ├── N2 on food-01222026123456-0000_results/
│   │   └── tracks.csv
```

### Step 2 — Run postural comparison

```bash
python batch_postural_comparison.py --data-dir /path/to/dataset --out-dir data/ --refresh
```

## Scripts

| Script | Inputs | Outputs |
|--------|--------|---------|
| `batch_postural_comparison.py` | `data/*.parquet` (or raw `--data-dir`) | `data/postural_comparison.csv`, `data/postural_comparison_stats.csv`, `data/postural_comparison.png/.pdf`, `data/supplemental_particle_data.csv` |
| `test_reproduce.py` | `demo_data/` | pass/fail to stdout |

## Files

| File | Description |
|------|-------------|
| `exclude.csv` | Genotype × date exclusion list (QC failures) |
| `demo_data/` | 4-genotype subset for offline reproducibility testing |
| `data/particle_data.parquet` | Per-particle speed data (all genotypes) |
| `data/fwd_frame_data.parquet` | Per-frame forward-run data |
| `data/recording_data.parquet` | Per-recording summary |
| `data/postural_comparison_stats.csv` | LME fold-changes, SE, q-values (used by `sos/sos_analysis.py`) |
| `data/postural_comparison.csv` | Per-recording median speed table |
| `data/postural_comparison.png/.pdf` | Main locomotion figure |

## Dependencies

```
pip install git+https://github.com/ODonnellLab/ODLabTracker.git
pip install numpy pandas matplotlib scipy pyarrow
R: lme4, lmerTest
```
