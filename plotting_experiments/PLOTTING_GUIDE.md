# ERICA Plotting Experiments Guide

## Overview

This folder contains scripts for generating publication-quality visualizations of ERICA clustering replicability results. The experiments serve dual purposes:

1. **Visualization**: Publication-ready matplotlib figures for papers and presentations
2. **Validation**: End-to-end testing of ERICA's flattened method API, HDBSCAN auto-K support, and ARI/AMI metrics

## ERICA API Functions Used

### Core Workflow

```python
from erica import ERICA

erica = ERICA(
    data=X,                          # numpy array, samples x features
    k_range=[2, 3, 4, 5, 6],        # K values to test
    n_iterations=200,                # Monte Carlo iterations
    method=['kmeans',                # NEW: list-of-strings API
            'agglomerative_ward',
            'agglomerative_single',
            'hdbscan'],
    hdbscan_params={'min_cluster_size': 10},  # HDBSCAN-specific params
    transpose=False,                 # data is already samples x features
    output_dir='results/erica_workdir',
)
erica.run()
results = erica.get_results()
```

### Accessing Results

| What | How | Returns |
|------|-----|---------|
| CLAM matrix (K-based) | `results['clam_matrices'][(k, method)]` | `np.ndarray` (n_samples x k) |
| CLAM matrix (HDBSCAN) | `results['auto_k']['hdbscan']['clam_matrix']` | `np.ndarray` (n_samples x modal_k) |
| ERICA metrics | `results['metrics'][k][method]['CRI']` | float |
| Parmigiani metrics | `results['metrics'][k][method]['ARI_mean']` | float |
| Optimal K* | `results['k_star']['TWCRI'][method]` | int |
| HDBSCAN modal K | `results['auto_k']['hdbscan']['modal_k']` | int |
| HDBSCAN agreement | `results['auto_k']['hdbscan']['k_agreement_rate']` | float (0-1) |
| K distribution | `results['auto_k']['hdbscan']['k_distribution']` | dict {k: count} |
| Disqualified K values | `results['disqualified_k'][method]` | list of int |
| Config | `results['config']['method']` | list of str |
| Auto-K accessor | `erica.get_auto_k_results('hdbscan')` | dict or None |

### Key Concepts

**CLAM Matrix** (Cluster Labels Alignment Matrix): For each sample, counts how many times it was assigned to each cluster across Monte Carlo iterations. Shape: (n_samples, k). Row sums equal the number of iterations where the sample appeared in the test set.

**Primary Cluster Assignment**: `np.argmax(clam_row)` — the cluster a sample was most frequently assigned to.

**Assignment Confidence**: `max(clam_row) / sum(clam_row)` — proportion of iterations where the sample landed in its primary cluster. 1.0 = perfectly stable, 1/k = random assignment.

**K\* (Optimal K)**: Selected by Algorithm 2 (non-decreasing metric criterion). For each metric (CRI, WCRI, TWCRI), the K* is the largest K where the metric value is non-decreasing. TWCRI K* is the primary recommendation.

## Datasets

### Real Data

| Name | Shape | Source | Notes |
|------|-------|--------|-------|
| `vdx_3gene` | 344 x 3 | Parmigiani et al. | ESR1, ERBB2, AURKA breast cancer gene expression |

### Synthetic Data (sklearn)

| Name | Shape | True K | Generator | Purpose |
|------|-------|--------|-----------|---------|
| `well_separated` | 300 x 5 | 3 | `make_blobs(std=0.8)` | Easy baseline |
| `overlapping` | 300 x 5 | 4 | `make_blobs(std=2.0)` | Blurred boundaries |
| `moons_2d` | 300 x 2 | 2 | `make_moons(noise=0.05)` | Non-convex shapes |
| `circles_2d` | 300 x 2 | 2 | `make_circles(noise=0.05)` | Nested topology |
| `blobs_2d` | 300 x 2 | 3 | `make_blobs(std=0.5)` | Simple 2D baseline |
| `high_dim` | 200 x 50 | 3 | `make_blobs(n_features=50)` | Dimensionality test |

### Gaussian Mixtures

| Name | Shape | True K | Structure | Purpose |
|------|-------|--------|-----------|---------|
| `gmm_spherical` | 450 x 2 | 3 | Equal spherical covariance | Easy GMM baseline |
| `gmm_anisotropic` | 450 x 2 | 3 | Elongated, rotated ellipses | Tests orientation sensitivity |
| `gmm_mixed_variance` | 450 x 2 | 3 | Tight + medium + spread | Tests variance sensitivity |
| `gmm_overlapping_pair` | 500 x 2 | 3 | Two overlapping + one isolated | Ambiguous boundary test |
| `gmm_five_clusters` | 400 x 2 | 5 | Five varied covariances | Complex structure |
| `gmm_high_dim_32` | 300 x 32 | 3 | Isotropic in 32D | Parmigiani dim032-style |

## Scripts

### Phase 1: Data and Pipeline

| Script | What | Inputs | Outputs |
|--------|------|--------|---------|
| `01_generate_datasets.py` | Generate all datasets | `examples/data/VDX_3_SV.csv` | `data/*.npz` |
| `02_run_erica_pipeline.py` | Run ERICA on all datasets | `data/*.npz` | `results/*.joblib` |

### Phase 2: Plot Scripts

| Script | What | Key ERICA Functions Used |
|--------|------|--------------------------|
| `03_clam_heatmaps.py` | Raw + sorted CLAM heatmaps | `clam_matrices[(k, method)]`, `auto_k['hdbscan']['clam_matrix']` |
| `04_stability_strips.py` | Per-sample stacked bars | `clam_matrices[(k, method)]` (normalized to proportions) |
| `05_metrics_curves.py` | CRI/WCRI/TWCRI + ARI/AMI vs K | `metrics[k][method]`, `k_star['TWCRI'][method]` |
| `06_method_comparison.py` | Side-by-side method panels | `metrics[k][method]`, `auto_k['hdbscan']['metrics_at_modal_k']` |
| `07_surfaces.py` | 3D + 2D surface plots | `clam_matrices`, `metrics`, HDBSCAN sweep results |
| `08_cluster_scatter.py` | 2D scatter with cluster colors | `clam_matrices[(k, method)]` (argmax for color) |
| `09_k_progression.py` | K=2..4 grids for moons/circles | `clam_matrices[(k, method)]` for 3 methods |
| `10_full_k_progression.py` | K=2..6 grids for ALL datasets | `clam_matrices[(k, method)]`, `k_star` for highlighting |

### Shared Module

| Module | What | Exports |
|--------|------|---------|
| `style.py` | Publication rcParams | `set_publication_style()`, `save_figure()`, `CMAP_SEQ`, `METHOD_COLORS`, `METRIC_COLORS` |

## Figure Organization

```
figures/
├── k_progressions/              # Full K=2..6 progression grids (13 datasets)
│   ├── k_progression_moons_2d.pdf
│   ├── k_progression_circles_2d.pdf
│   ├── k_progression_gmm_anisotropic.pdf
│   ├── ...
│   └── k_progression_vdx_3gene.pdf
├── clam_*.pdf                   # CLAM heatmaps (script 03)
├── stability_*.pdf              # Stability strips (script 04)
├── metrics_*.pdf                # Metric curves (script 05)
├── method_*.pdf                 # Method comparisons (script 06)
├── kstar_*.pdf                  # K* bar charts (script 06)
├── surface_*.pdf                # 3D surfaces (script 07)
├── landscape_*.pdf              # Metric landscapes (script 07)
├── hdbscan_sensitivity_*.pdf    # HDBSCAN param sweep (script 07)
├── scatter_*.pdf                # Cluster scatter plots (script 08)
└── confidence_*.pdf             # Confidence maps (script 08)
```

## How Clustering Methods Differ

These experiments demonstrate why method choice matters:

**K-Means**: Assumes spherical clusters. Splits non-convex shapes (moons, circles) incorrectly. Works well on isotropic Gaussians.

**Agglomerative Ward**: Minimizes within-cluster variance. Similar limitations to K-Means for non-convex shapes, but handles elongated clusters somewhat better.

**Agglomerative Single Linkage**: Builds clusters by nearest-neighbor chaining. Can follow non-convex shapes (moons) but tends to oversplit on noisy data and is sensitive to outliers.

**HDBSCAN**: Density-based, discovers K automatically. Handles non-convex shapes and varying cluster sizes. Noise points assigned to nearest cluster centroid for CLAM computation.

## Reproducing

```bash
cd plotting_experiments
pip install -e ".[all]"        # from repo root
python 01_generate_datasets.py  # ~5 seconds
python 02_run_erica_pipeline.py # ~30 seconds (200 iterations)
python 10_full_k_progression.py # ~10 seconds, generates all K=2..6 grids
```

Or use the Makefile:
```bash
make all    # runs everything
make clean  # removes data/, results/, figures/
```
