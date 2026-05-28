# Figure style notes — Malaiwong & Oglu 2026

## SOS barplot layout conventions (established 2026-05-28)

All SOS barplot figures share a single style module (`sos_style.py`). Edit constants there to update all figures at once.

### Layout rules

- **Bar width is fixed in physical inches** (`sos_style.BAR_SCALE = 0.68 in/data-unit`). All bars across all figures have the same physical width. Figure width expands when there are more bars.
- **ECDF panels are always the same physical size** (`ECDF_W × ECDF_H = 1.543" × 0.847"`). Adding more ECDF panels changes the ECDF block arrangement, not panel size.
- Figure height is fixed by the ECDF stack height (`CONTENT_H = 2.160"`) plus top/bottom margins.
- Figure width is computed from content: `L_MAR + barplot_w + gap(s) + ECDF_block + R_MAR`.

### Bar spacing conventions

- **Genotype is the primary grouping unit.** Use wide inter-genotype spacing (`_GENO_X spacing ~1.05`) only when sub-conditions are nested within each genotype (as in `sos_condition_plots.py`).
- **Pure genotype comparisons** (e.g., epistasis figure) use tight, uniform spacing — only a modest gap between any pairs, no large inter-group gaps.

### Typography

- Global font: **Arial** (set in `sos_style.py` rcParams).
- Gene names on axes and legends are **italic** (`cest-2.1`, `tbh-1`, etc.).
- Strain IDs such as **N2** are **upright** (not italic).
- Applied via `sos_style.set_xticklabels()` and `sos_style.draw_ecdf()` (which italicises legend entries automatically).

### Shared helpers (`sos_style.py`)

| Function | Purpose |
|---|---|
| `stars(p)` | BH p-value → `*`/`**`/`***` string |
| `make_ax(fig, fw, fh, l, b, w, h)` | Add axes using inch coordinates |
| `set_xticklabels(ax, labels, **kw)` | Set labels with italic gene names |
| `draw_ecdf(ax, df, geno_list, colors, ...)` | Shared ECDF drawing with gridlines + italic legend |
| `style_ecdf_ax(ax, title)` | Apply ECDF styling to an axes after custom curve drawing |
| `dot_legend_handles()` | Standard individual / NR / daily-median legend entries |
| `COLORS` | Canonical genotype color dict |
| `CANONICAL_ORDER` | Genotype sort order (derived from `COLORS` key order) |

---

## Grouping configuration system (established 2026-05-28)

`sos_figure_plots.py` reads per-dataset YAML configs from the raw data directory.
The config controls bar grouping, x-axis segmentation, and intra/inter-group spacing.

**When a config is present:** bars are arranged per the `groups` field (lists of Strain_IDs),
with `intra_spacing` within groups and `inter_spacing` between. A break in the x-axis line
marks each group boundary.

**When no config is present:** all bars are drawn flat with uniform tight spacing
(`_BAR_STEP = 0.22`) and a continuous x-axis line.

### Config file naming and location

Config files live alongside the raw CSVs, named `<csv_stem>_config.yaml`.
Currently written for the four rescue/grouped figures:

| Config | Groups |
|--------|--------|
| `3e_cest-2.1 rescues_SOSdata_config.yaml` | `[N2]` \| `[cest-2.1, RIC, GUT]` |
| `4d_RescueCAN(gba-4)_SOSdata_config.yaml` | `[N2]` \| `[gba-4, CAN L1–3]` |
| `5b_cat-1 rescues_SOSdata_config.yaml` | `[N2]` \| `[cat-1, RIC, CAN, RC]` |
| `5d_RescueCAN(transporters)_SOSdata_config.yaml` | `[N2]` \| `[all slc-17.1]` \| `[all T05A1.5]` |

### x-axis segmentation rules

- **Breaks** are drawn only between config-defined groups (≥ 2 groups).
- **No config / single group** → continuous line.
- The epistasis figure (`sos_epistasis_plots.py`) uses its own pair-based segmentation
  (INTRA = 0.20, INTER = 0.42) defined in script constants, not a YAML config.
- `sos_condition_plots.py` always segments at genotype boundaries (one segment per
  genotype, 4 condition bars nested within each).
