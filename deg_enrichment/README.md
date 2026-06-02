# DEG enrichment — CAN transcriptomic specificity

Tests whether *cest-2.1* downstream effector genes (DEGs) are enriched in CAN
relative to other cat-1⁺ neuron types, across three single-cell RNA-seq datasets
(Taylor L4, Taylor adult, Ghaddar adult) and a range of expression specificity thresholds.

## Quick start

```bash
cd deg_enrichment
python deg_enrichment_by_threshold.py
```

Reads from `input/`, writes figures and statistics to `data/`.

## Scripts

| Script | Inputs | Outputs |
|--------|--------|---------|
| `deg_enrichment_by_threshold.py` | `input/Taylor_*.csv`, `input/Ghaddar_*.csv` | `data/can_deg_enrichment_by_threshold.pdf/.png`, `data/deg_enrichment_can_statistics.csv`, `data/deg_enrichment_all_neurons_statistics.csv` |

CLI options:
```
--input-dir   input/   Directory containing Taylor and Ghaddar CSVs
--out-dir     data/    Output directory for figures and statistics CSVs
```

## Input files

| File | Source | Description |
|------|--------|-------------|
| `input/Taylor_L4_GenesExpressing-BATCH-thrs2.csv` | Taylor et al. 2021 | L4 scRNA-seq TPM, all neurons |
| `input/Taylor_L4_GenesExpressing-BATCH-thrs2_DEGfiltered.csv` | Derived | L4 rows restricted to *cest-2.1* DEGs |
| `input/Taylor_Adult_GenesExpressing-BATCH-thrs2.csv` | Taylor et al. 2021 | Adult scRNA-seq TPM, all neurons |
| `input/Taylor_Adult_GenesExpressing-BATCH-thrs2_DEGfiltered.csv` | Derived | Adult rows restricted to DEGs |
| `input/Ghaddar_adg0506_Data_S4.csv` | Ghaddar et al. 2023 | Adult scRNA-seq supplemental table |
| `input/Ghaddar_prcnt_tpm_bootstrap.csv` | Ghaddar et al. 2023 | Bootstrap TPM estimates (large file, not in repo) |
| `input/S3_cest2.1DEgenes` | Paper supplement | *cest-2.1* DEG list (placeholder, not in repo) |
| `input/Wang2024_UptakeSynthesisRelease/` | Wang et al. 2024 | Reference data (placeholder, not in repo) |

## Output files

| File | Description |
|------|-------------|
| `data/can_deg_enrichment_by_threshold.pdf/.png` | 4-panel enrichment figure (3 datasets + all-neuron comparison) |
| `data/deg_enrichment_can_statistics.csv` | CAN enrichment statistics per dataset × TPM threshold |
| `data/deg_enrichment_all_neurons_statistics.csv` | Same for all cat-1⁺ neurons |
| `data/Ghaddar_DEG_TPM_wide.csv` | Derived Ghaddar TPM matrix (wide format) |
| `data/deg_binary_matrix.csv` | Binary DEG presence/absence matrix |
| `data/deg_combined_norm_matrix.csv` | Normalized expression matrix |
| `data/deg_goclass_enrichment.csv` | GO class enrichment results |
| `data/deg_hypergeometric_enrichment.csv` | Hypergeometric test results |

## Key method notes

- Enrichment metric: fold-over-mean neuronal expression (`NEU_TPM / gene_mean`)
- Significance: BH FDR correction within each panel (dataset × TPM threshold)
- Ghaddar gene IDs (8-digit WBGene) matched to Taylor by stripping trailing version digit
- Taylor neuronal cell types filtered to 88 exact matches + 16 merged types = 104 total
- See `methods_deg_enrichment.txt` for full statistical methods

## Dependencies

```
pip install numpy pandas matplotlib scipy
```
