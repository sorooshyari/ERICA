# Design: Plotting Experiments for ERICA

**Date:** 2026-03-29
**Status:** Draft
**Scope:** Create a self-contained plotting experiments folder with publication-quality matplotlib visualizations of ERICA results, using locally installed ERICA and local/generated datasets.

---

## Problem

ERICA has `erica/plotting.py` with plotly-based interactive plots, but no publication-quality matplotlib figures matching Parmigiani et al.'s style. We need a playground to generate test data, run ERICA, and experiment with different plot types (CLAM heatmaps, metric curves, 3D surfaces, etc.) for paper figures.

Additionally, the ERICA package was just updated with a flattened method API and HDBSCAN support. These plotting experiments serve as an end-to-end validation of the new implementation — exercising the new `method` list API, `hdbscan` method, `auto_k_results`, ARI/AMI metrics, and `get_auto_k_results()` accessor against real and synthetic data.

## Constraints

- All scripts import ERICA via `from erica import ERICA` (pip-installed package via `pip install -e .`)
- Scripts must exercise the new API features: list-of-strings `method` parameter, `hdbscan_params`, `get_auto_k_results()`, and ARI/AMI metrics in results
- No remote downloads required at runtime. Use local data (`examples/data/VDX_3_SV.csv`) and sklearn-generated synthetics
- All matplotlib, no plotly. Publication-quality PDFs matching Parmigiani et al.'s style
- Data generation and ERICA pipeline run separately from plot scripts (fast plot iteration)
- Serialization via `joblib.dump`/`joblib.load` (handles ERICA's nested dict/tuple-key structures)

---

## 1. Directory Structure

```
plotting_experiments/
├── data/                     # Generated datasets (gitignored)
├── results/                  # ERICA pipeline output (gitignored)
├── figures/                  # PDF/PNG output (gitignored)
├── style.py                  # Shared publication style settings
├── 01_generate_datasets.py   # Generate/copy all datasets locally
├── 02_run_erica_pipeline.py  # Run ERICA on each dataset, save via joblib
├── 03_clam_heatmaps.py       # Plots A & B: raw + sorted CLAM heatmaps
├── 04_stability_strips.py    # Plot C: per-sample cluster assignment proportions
├── 05_metrics_curves.py      # Plot D: CRI/WCRI/TWCRI vs K with K* and ARI/AMI error bars
├── 06_method_comparison.py   # Plot E: multi-panel method comparison
├── 07_surfaces.py            # Plot F: 3D CLAM surface, metric landscape, HDBSCAN param grid
├── Makefile                  # make data, make pipeline, make plots, make all
└── README.md
```

Add to `.gitignore`:
```
plotting_experiments/data/
plotting_experiments/results/
plotting_experiments/figures/
```

---

## 2. Shared Style Module (`style.py`)

A single module imported by all plot scripts. Avoids duplicating rcParams across 5 files.

```python
# Exports:
# - set_publication_style() — configures rcParams for paper figures
# - SINGLE_COL, DOUBLE_COL — figure width constants (inches)
# - CMAP_SEQ — sequential colormap for heatmaps (viridis, colorblind-safe)
# - CMAP_DIV — diverging colormap
# - METHOD_COLORS — dict mapping method names to colorblind-safe line colors
# - save_figure(fig, name) — saves to figures/ as both PDF and PNG
```

Style settings (matching Parmigiani et al., with colorblind-safe updates):
- Font: DejaVu Sans (matplotlib default, available everywhere)
- Tick labels: 10pt. Axis labels: 12pt. Titles: 14pt.
- `lines.linewidth`: 2
- Colormap for heatmaps: `viridis` (colorblind-safe, replaces Parmigiani's `cool`)
- Line colors: Okabe-Ito colorblind-safe palette for categorical method comparisons
- Figure sizes: single column = 3.5", double column = 7.0" (standard bioinformatics journal)
- DPI: 300 for PDF, 150 for PNG previews
- `savefig.bbox_inches`: `tight`

---

## 3. Dataset Generation (`01_generate_datasets.py`)

All datasets saved to `plotting_experiments/data/` via `np.save` or `np.savez`.

### Real data
- **VDX 3-gene subset**: Copy from `examples/data/VDX_3_SV.csv`. Load with `erica.data.load_data()`, save as numpy array. 344 samples, 3 features (ESR1, ERBB2, AURKA).

### Synthetic data (sklearn-generated, matching Parmigiani's categories)
- **Well-separated blobs**: `make_blobs(n_samples=300, centers=3, cluster_std=0.8, random_state=42)` — 5 features. Easy clustering baseline.
- **Overlapping blobs**: `make_blobs(n_samples=300, centers=4, cluster_std=2.0, random_state=42)` — 5 features. Tests behavior when clusters merge.
- **2D shape datasets** (replacing Lorenzo's Frantti files, which aren't accessible):
  - `make_moons(n_samples=300, noise=0.05)` — non-convex clusters
  - `make_circles(n_samples=300, noise=0.05, factor=0.5)` — nested rings
  - `make_blobs(n_samples=300, centers=3, cluster_std=0.5)` in 2D — well-separated 2D
- **High-dimensional**: `make_blobs(n_samples=200, centers=3, n_features=50, random_state=42)` — tests ERICA on higher-dimensional data

Each dataset saved as `data/{name}.npz` with keys `X` (data array) and `meta` (dict with `n_samples`, `n_features`, `true_k`, `description`, `transpose` flag).

---

## 4. ERICA Pipeline (`02_run_erica_pipeline.py`)

Runs ERICA on each dataset and saves full results via `joblib.dump`.

### New API features exercised

This script validates the recently shipped flattened method API + HDBSCAN support:
- **Flattened `method` parameter**: Uses `method=['kmeans', 'agglomerative_ward', 'hdbscan']` (list-of-strings API, not the old `method='both'` + `linkages`)
- **HDBSCAN auto-K**: Exercises `hdbscan_params` dict, verifies `auto_k_results` in output
- **ARI/AMI metrics**: Verifies `ARI_mean`, `ARI_std`, `AMI_mean`, `AMI_std` appear in K-based metric dicts
- **`get_auto_k_results()`**: Calls accessor to retrieve HDBSCAN modal_k, k_distribution, k_agreement_rate, metrics_at_modal_k
- **Mixed K-based + auto-K**: Runs both paths in a single ERICA instance, validates results dict contains both `metrics` (K-based) and `auto_k` (HDBSCAN) keys

Script prints a validation summary after each dataset run confirming all expected keys/structures are present.

### Per-dataset configuration

| Dataset | method | k_range | n_iterations | transpose | hdbscan_params |
|---------|--------|---------|-------------|-----------|----------------|
| vdx_3gene | `['kmeans', 'agglomerative_ward', 'hdbscan']` | [2,3,4,5,6] | 100 | False (already samples x features after load) | `{'min_cluster_size': 15}` |
| well_separated | same | [2,3,4,5] | 100 | False | `{'min_cluster_size': 10}` |
| overlapping | same | [2,3,4,5,6] | 100 | False | `{'min_cluster_size': 10}` |
| moons_2d | same | [2,3,4] | 100 | False | `{'min_cluster_size': 10}` |
| circles_2d | same | [2,3,4] | 100 | False | `{'min_cluster_size': 10}` |
| blobs_2d | same | [2,3,4,5] | 100 | False | `{'min_cluster_size': 10}` |
| high_dim | same | [2,3,4,5] | 100 | False | `{'min_cluster_size': 10}` |

### HDBSCAN parameter sweep (for surface plot 07)

For VDX 3-gene only, run a grid sweep:
- `min_cluster_size`: [5, 10, 15, 20, 30, 50]
- `min_samples`: [None, 3, 5, 10]

Each combination is a separate ERICA run with `method=['hdbscan']`, K=2..6, 50 iterations. Save as `results/vdx_hdbscan_sweep.joblib`.

### Output format

Each result saved as `results/{dataset_name}.joblib` containing:
```python
{
    'erica_results': erica.get_results(),   # full results dict
    'iteration_labels': {                    # per-(k, method) label pairs for ARI recomputation
        (k, method): result['iteration_labels']
        for (k, method), result in erica.results_.items()
    },
    'auto_k_results': erica.auto_k_results_,  # HDBSCAN results with CLAM, labels, etc.
    'config': {                               # dataset metadata
        'dataset_name': name,
        'n_samples': n,
        'n_features': p,
        'true_k': true_k,
        'transpose': False,
    },
}
```

Per-iteration ARI/AMI scores are recomputable from `iteration_labels` stored here. This avoids storing redundant aggregated-only values.

### Runtime estimate

At 100 iterations per dataset, ~7 datasets: expect ~5-10 minutes total on a modern laptop. The HDBSCAN sweep (24 param combos x 50 iterations) adds ~5 more minutes. Total: ~15 minutes.

Script prints progress per dataset with elapsed time.

---

## 5. Plot Scripts

All scripts follow the same pattern:
```python
from style import set_publication_style, save_figure, CMAP_SEQ, METHOD_COLORS
import joblib

set_publication_style()
results = joblib.load('results/{dataset}.joblib')
# ... create figure ...
save_figure(fig, '{plot_name}')
```

### 03 — CLAM Heatmaps

Two figures per dataset/method/K combination:

**A) Raw CLAM heatmap**: `imshow(clam_matrix)`, samples on Y-axis, clusters on X-axis, color = assignment count. Colorbar on right.

**B) Sorted CLAM heatmap**: Same data but samples sorted by primary cluster assignment (argmax of each row), then within each cluster by assignment strength (descending). Produces clean diagonal blocks for stable clusterings, scattered patterns for unstable ones. This is the more informative view.

Access pattern: K-based CLAMs from `results['erica_results']['clam_matrices'][(k, method)]`. HDBSCAN CLAM from `results['auto_k_results']['hdbscan']['clam_matrix']`.

Default output: sorted heatmap for VDX 3-gene, K=3, kmeans. Generate a multi-panel figure showing K=2,3,4 side by side.

### 04 — Stability Strips

Per-sample horizontal stacked bars showing cluster assignment proportions (CLAM row normalized to sum=1). Sorted by entropy (most stable samples at top, most confused at bottom).

Each sample is one horizontal bar. Colors represent clusters. A sample that's 100% in one cluster is a solid-color bar. A 50/50 split shows two equal segments.

Width: single column (3.5"). Height: scales with sample count (cap at ~100 samples shown, subsample if needed).

### 05 — Metrics Curves

Two panels per dataset:

**Panel 1: ERICA metrics** — `errorbar` plot of CRI, WCRI, TWCRI vs K. Vertical dashed lines at K* for each metric (color-matched). X-axis = K, Y-axis = metric value.

**Panel 2: Parmigiani metrics** — `errorbar` plot of ARI mean +/- std and AMI mean +/- std vs K. Error bars computed by reloading `iteration_labels` from the saved results and calling `compute_parmigiani_metrics` per iteration.

K* lines: one per metric, color-matched to the metric's line. TWCRI K* gets the thickest dashed line (primary recommendation).

### 06 — Method Comparison

Multi-panel figure (one row per metric, columns = K values or vice versa):

**Main figure**: 3 subplots (CRI, WCRI, TWCRI), each showing lines for kmeans, agglomerative_ward, and HDBSCAN-at-modal-K. X-axis = K, Y-axis = metric value.

**Secondary figure**: Grouped bar chart of K* per method per metric.

For HDBSCAN, which has a single modal K (not a curve), show it as a horizontal band or a single point at its modal K with metrics_at_modal_k values.

### 07 — 3D Surfaces + 2D Fallbacks

Three surface types, each with a primary 2D heatmap and an optional 3D `plot_surface` view:

**A) CLAM topography**: X = sample index (sorted by cluster), Y = cluster index, Z = assignment count. Primary view: 2D imshow (this IS the sorted CLAM heatmap from script 03, but rendered as a continuous surface). 3D view: same data as a raised surface.

**B) Metric landscape**: X = K values, Y = methods (indexed 0..N), Z = TWCRI. Primary: 2D heatmap with K on x-axis, method on y-axis, color = TWCRI. 3D: surface interpolated across K and method.

**C) HDBSCAN parameter sensitivity**: X = `min_cluster_size`, Y = `min_samples`, Z = `k_agreement_rate`. Loads from `results/vdx_hdbscan_sweep.joblib`. Primary: 2D heatmap. 3D: parameter sensitivity surface.

For all 3D plots: export both the 2D and 3D versions. 2D heatmaps are the paper-ready defaults; 3D surfaces are for exploration and supplementary materials.

---

## 6. Makefile

```makefile
.PHONY: data pipeline plots all clean

data:
	python 01_generate_datasets.py

pipeline: data
	python 02_run_erica_pipeline.py

plots: pipeline
	python 03_clam_heatmaps.py
	python 04_stability_strips.py
	python 05_metrics_curves.py
	python 06_method_comparison.py
	python 07_surfaces.py

all: plots

clean:
	rm -rf data/ results/ figures/
```

---

## 7. Dependencies

All available via the existing `pip install -e .` plus matplotlib (already a core dep in pyproject.toml):
- `erica` (installed from local repo)
- `numpy`, `matplotlib`, `scikit-learn` (all existing deps)
- `joblib` (comes with scikit-learn)

No new dependencies required.
