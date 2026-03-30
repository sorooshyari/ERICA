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
make data      # 01: generate/copy datasets (~5 seconds)
make pipeline  # 02: run ERICA on all datasets (~15 minutes)
make plots     # 03-07: generate all figures (~30 seconds)
make clean     # remove data/, results/, figures/
```

## Scripts

| # | Script | What it does |
|---|--------|-------------|
| 01 | 01_generate_datasets.py | Copies VDX 3-gene CSV, generates sklearn synthetics |
| 02 | 02_run_erica_pipeline.py | Runs ERICA on each dataset, saves results via joblib |
| 03 | 03_clam_heatmaps.py | Raw + sorted CLAM heatmaps |
| 04 | 04_stability_strips.py | Per-sample cluster assignment stacked bars |
| 05 | 05_metrics_curves.py | CRI/WCRI/TWCRI and ARI/AMI vs K curves |
| 06 | 06_method_comparison.py | Side-by-side method comparison panels |
| 07 | 07_surfaces.py | 3D surfaces + 2D heatmap fallbacks |

## Output

Figures are saved to figures/ as both PDF (publication) and PNG (preview).
