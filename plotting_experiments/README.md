# Plotting Experiments

Publication-quality matplotlib visualizations of ERICA clustering results.
Also serves as end-to-end validation of the flattened method API + HDBSCAN support.

## Prerequisites

```bash
pip install -e ".[all]"  # from the ERICA repo root
```

## Quick Start

```bash
cd plotting_experiments
make all  # runs data generation, ERICA pipeline, and all plots
```

## Individual Steps

```bash
make data        # 01: generate/copy datasets (~5 seconds)
make pipeline    # 02: run ERICA on all datasets (~15 minutes)
make plots       # 03-07: core visualization suite (~30 seconds)
make scatter     # 08: cluster scatter plots
make progressions  # 09-10: K progression grids
make batch       # 11: batch plot generation
make gaussian    # gaussian_mixture_study scripts
make literature  # literature_comparison scripts
make erica-stats # erica_statistics scripts (ERICA stat + ARI + AMI)
make clean       # remove data/, results/, figures/
```

## Scripts

| # | Script | What it does |
|---|--------|-------------|
| 01 | `01_generate_datasets.py` | Copies VDX 3-gene CSV, generates sklearn synthetics |
| 02 | `02_run_erica_pipeline.py` | Runs ERICA on each dataset, saves results via joblib |
| 03 | `03_clam_heatmaps.py` | Raw + sorted CLAM heatmaps |
| 04 | `04_stability_strips.py` | Per-sample cluster assignment stacked bars |
| 05 | `05_metrics_curves.py` | CRI/WCRI/TWCRI and ARI/AMI vs K curves |
| 06 | `06_method_comparison.py` | Side-by-side method comparison panels |
| 07 | `07_surfaces.py` | 3D surfaces + 2D heatmap fallbacks |
| 08 | `08_cluster_scatter.py` | 2D scatter plots colored by cluster assignment |
| 09 | `09_k_progression.py` | Method x K progression grids |
| 10 | `10_full_k_progression.py` | Full K progression with all methods |
| 11 | `11_generate_all_plots.py` | Batch generation of per-dataset figure suites |

### Subfolders

| Subfolder | Scripts | Description |
|-----------|---------|-------------|
| `gaussian_mixture_study/` | 01-05 | 4-center Gaussian sigma sweep (0.01 to 10.0) |
| `literature_comparison/` | 01-03 | Error bands, co-assignment heatmaps, PCA scatter (Tibshirani, Monti, Masoero) |
| `erica_statistics/` | 01-08 | ERICA stat + ARI + AMI exploratory analysis, PCSP, ICA |
| `future_experiments/` | — | Proposals for dropout/dimensionality studies |

## HTML Galleries

| Gallery | Location | Figures |
|---------|----------|---------|
| Main Gallery | `2026-03-30-gallery.html` | 80 figures, dataset table, by-dataset breakdowns |
| Entropy & ERICA Stat Playground | `2026-03-30-playground.html` | 101 entropy/CRI experiments |
| ERICA Statistic Deep Dive | `2026-03-30-erica-statistic-playground.html` | 68 distribution/boxplot/trajectory figures |
| Literature Comparison | `literature_comparison/2026-03-31-literature-comparison.html` | Error bands, co-assignment, PCA scatter |
| ERICA Statistics (Exploratory) | `erica_statistics/2026-04-02-erica-statistics.html` | ERICA stat + ARI + AMI, PCSP, ICA/ICAH |

## Output

Figures are saved to `figures/` as both PDF (publication) and PNG (preview).

## Supporting Files

- `style.py` — Shared publication style settings and save utilities
- `metrics_table.tsv` — 374-row metrics export across all datasets and methods
- `DATA_DOCUMENTATION.md` — Dataset format (.npz) and results format (.joblib)
- `PLOTTING_GUIDE.md` — ERICA API usage patterns and dataset notes
- `LITERATURE_FIGURE_SURVEY.md` — Survey of standard figures in clustering stability literature
- `WHAT_WE_DID.md` — Session notes and observations
