# Malaiwong & Oglu 2026 — *C. elegans* locomotion and CAN neuron data

Data and analysis code for Malaiwong & Oglu 2026. The repository contains four
independent subprojects, each with its own scripts and `data/` directory.

---

## Installation

```bash
git clone https://github.com/ODonnellLab/Malaiwong_Oglu_2026.git
cd Malaiwong_Oglu_2026
pip install git+https://github.com/ODonnellLab/ODLabTracker.git
pip install numpy pandas matplotlib scipy pyarrow
```

R (≥ 4.0) with `lme4`, `lmerTest`, and `emmeans` is required to re-run mixed-effects models.
All model outputs are cached in `*/data/` so figures can be regenerated without R.

---

## Subprojects

### 1. [`locomotion/`](locomotion/README.md) — Postural locomotion analysis

Forward-run speed, reversal rate, and pirouette rate across 32 genotypes × 117 recordings.
LME fold-changes relative to N2.

```bash
cd locomotion && python batch_postural_comparison.py
```

→ `locomotion/data/postural_comparison.pdf`

---

### 2. [`sos/`](sos/README.md) — SOS octanol avoidance response time

Avoidance response times and speed–SOS correlation across all figure genotypes.

```bash
cd sos && python sos_analysis.py
cd sos && python sos_condition_plots.py
```

→ `sos/data/speed_sos_correlation.pdf`, `sos/data/sos_condition_plots.pdf`

**Note:** `sos_analysis.py` reads locomotion outputs from `../locomotion/data/` by default
(`--locomotion-dir` to override). Run the locomotion pipeline first.

---

### 3. [`deg_enrichment/`](deg_enrichment/README.md) — CAN transcriptomic enrichment

DEG enrichment analysis: are *cest-2.1* downstream effectors enriched in CAN
relative to other cat-1⁺ neurons? Tests across Taylor L4, Taylor adult, and Ghaddar adult
scRNA-seq datasets.

```bash
cd deg_enrichment && python deg_enrichment_by_threshold.py
```

→ `deg_enrichment/data/can_deg_enrichment_by_threshold.pdf`

---

### 4. [`spatial_filter/`](spatial_filter/README.md) — Neurite–intestine proximity

Which cat-1⁺ neuron types have processes adjacent to the intestine?
Nearest-neighbour distance analysis using the Virtual Worm Blender morphology.

```bash
cd spatial_filter
python blend_intestine_proximity.py
python blend_ap_axis_figure.py
```

→ `spatial_filter/data/blend_intestine_proximity.pdf`, `spatial_filter/data/neuron_y_distribution.pdf`

---

## Repository structure

```
Malaiwong_Oglu_2026/
├── locomotion/         # postural speed analysis
│   ├── batch_postural_comparison.py
│   ├── test_reproduce.py
│   ├── exclude.csv     # genotype QC exclusion list
│   ├── demo_data/      # 4-genotype subset for offline testing
│   └── data/           # cached parquets, LME results, figures
├── sos/                # SOS response time + speed correlation
│   ├── sos_analysis.py, sos_figure_plots.py, sos_condition_plots.py
│   ├── sos_epistasis_plots.py, sos_style.py
│   └── data/           # cached parquets, LMM stats, figures
│       └── figures/    # per-figure SOS bar + ECDF plots
├── deg_enrichment/     # CAN DEG enrichment
│   ├── deg_enrichment_by_threshold.py
│   ├── input/          # Taylor + Ghaddar scRNA-seq CSVs
│   └── data/           # enrichment figures and statistics CSVs
└── spatial_filter/     # neurite–intestine proximity
    ├── extract_world_verts.py (Blender headless)
    ├── blend_intestine_proximity.py
    ├── blend_ap_axis_figure.py, blend_render.py
    ├── nml_morphology_figure.py, wormatlas_em_measure.py
    └── data/           # Blender-derived CSVs and proximity figures
```

---

## Reproducibility

Each subproject's `data/` directory contains cached intermediate files so that all
figures can be regenerated without raw tracking data or NAS access. Scripts that
require raw data accept a `--data-dir` or `--refresh` flag.

`locomotion/test_reproduce.py` verifies the locomotion pipeline end-to-end using the
4-genotype `demo_data/` subset:

```bash
cd locomotion && python test_reproduce.py
```
