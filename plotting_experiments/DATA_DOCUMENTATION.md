# Data Documentation

Generated datasets and ERICA pipeline results for plotting experiments.

## Datasets (`data/`)

All datasets stored as `.npz` files with keys `X` (numpy array, samples x features) and `meta` (dict with `n_samples`, `n_features`, `true_k`, `description`).

### Real Data

| Dataset | Shape | True K | Source | Description |
|---------|-------|--------|--------|-------------|
| `vdx_3gene.npz` | 344 x 3 | Unknown | Parmigiani et al. | VDX breast cancer 3-gene subset (ESR1, ERBB2, AURKA). Loaded from `examples/data/VDX_3_SV.csv` via `np.loadtxt`. Gene expression values range [5.4, 15.6]. |

### Synthetic Data

| Dataset | Shape | True K | Generator | Description |
|---------|-------|--------|-----------|-------------|
| `well_separated.npz` | 300 x 5 | 3 | `make_blobs(centers=3, cluster_std=0.8)` | Well-separated clusters in 5D. Easy baseline for validating ERICA finds the correct K. |
| `overlapping.npz` | 300 x 5 | 4 | `make_blobs(centers=4, cluster_std=2.0)` | Overlapping clusters in 5D. Tests ERICA's behavior when cluster boundaries blur. |
| `moons_2d.npz` | 300 x 2 | 2 | `make_moons(noise=0.05)` | Two interleaving half-moon shapes. Non-convex clusters that K-Means cannot separate but HDBSCAN can. |
| `circles_2d.npz` | 300 x 2 | 2 | `make_circles(noise=0.05, factor=0.5)` | Nested concentric circles. Another non-convex shape that challenges centroid-based methods. |
| `blobs_2d.npz` | 300 x 2 | 3 | `make_blobs(centers=3, cluster_std=0.5, n_features=2)` | Tight 2D blobs. Simple baseline for visual scatter plot confirmation. |
| `high_dim.npz` | 200 x 50 | 3 | `make_blobs(centers=3, n_features=50)` | High-dimensional clusters. Tests whether ERICA metrics remain meaningful in 50D. |

All synthetic datasets use `random_state=42` for reproducibility.

## Pipeline Results (`results/`)

Each result file is a joblib-serialized dict saved by `02_run_erica_pipeline.py`. All runs used `method=['kmeans', 'agglomerative_ward', 'hdbscan']` with 200 Monte Carlo iterations and `transpose=False`.

### Per-Dataset Results

| Result File | K Range | Iterations | K* (TWCRI) KMeans | K* (TWCRI) Ward | HDBSCAN Modal K | HDBSCAN Agreement |
|-------------|---------|------------|-------------------|-----------------|-----------------|-------------------|
| `vdx_3gene.joblib` | [2,3,4,5,6] | 200 | 6 | 6 | 2 | 0.60 |
| `well_separated.joblib` | [2,3,4,5] | 200 | 3 | 3 | 3 | 1.00 |
| `overlapping.joblib` | [2,3,4,5,6] | 200 | 4 | 4 | 3 | 0.57 |
| `moons_2d.joblib` | [2,3,4] | 200 | 4 | 2 | 2 | 0.93 |
| `circles_2d.joblib` | [2,3,4] | 200 | 2 | 2 | 4 | 0.39 |
| `blobs_2d.joblib` | [2,3,4,5] | 200 | 3 | 3 | 3 | 0.99 |
| `high_dim.joblib` | [2,3,4,5] | 200 | 3 | 3 | 3 | 0.72 |

### Result File Structure

Each `.joblib` file contains:

```python
{
    'erica_results': {
        'clam_matrices': {(k, method): np.ndarray, ...},   # tuple keys
        'metrics': {k: {method: {metric: value, ...}}},     # nested dicts
        'k_star': {metric: {method: k_value}},
        'disqualified_k': {method: [k_values]},
        'auto_k': {
            'hdbscan': {
                'modal_k': int,
                'k_agreement_rate': float,
                'k_distribution': {k: count},
                'clam_matrix': np.ndarray,
                'n_iterations_used': int,
                'iteration_labels': {'predicted': [...], 'true': [...]},
                'noise_counts': [int, ...],
                'metrics_at_modal_k': {metric: value},  # if modal_k > 0
            }
        },
        'config': {method: list, k_range: list, ...},
        'results': {(k, method): {clam_matrix, iteration_labels, ...}},
    },
    'auto_k_results': {...},  # shortcut to hdbscan results
    'config': {
        'dataset_name': str,
        'n_samples': int,
        'n_features': int,
        'true_k': int or None,
        'transpose': False,
    },
}
```

### HDBSCAN Parameter Sweep

`vdx_hdbscan_sweep.joblib` contains results from a grid search over HDBSCAN parameters on the VDX 3-gene dataset (100 iterations per combo):

- `min_cluster_size`: [5, 10, 15, 20, 30, 50]
- `min_samples`: [None, 3, 5, 10]
- 24 parameter combinations total

Structure: dict with tuple keys `(min_cluster_size, min_samples)` mapping to:

```python
{
    'modal_k': int,
    'k_agreement_rate': float,
    'k_distribution': {k: count},
    'params': {'min_cluster_size': int, ...},
}
```

## Key Observations

**Well-separated and blobs_2d** are the easiest: both K-Means and Agglomerative Ward recover the true K, and HDBSCAN agrees with near-perfect agreement rates (0.99-1.00).

**Moons_2d** shows an interesting divergence: Ward correctly finds K*=2 (the true structure), but K-Means selects K*=4 (it can't represent non-convex clusters). HDBSCAN correctly finds 2 clusters with 0.93 agreement.

**Circles_2d** is challenging for all methods: HDBSCAN finds 4 clusters with only 0.39 agreement, indicating high instability. K-Means and Ward both select K*=2.

**VDX 3-gene** (real data): Both methods select K*=6 (the maximum tested), suggesting the k_range should be extended. HDBSCAN finds modal K=2 with 0.60 agreement, suggesting a two-group structure at the coarsest level.

**Overlapping** shows reduced agreement across the board (HDBSCAN 0.57), correctly reflecting the ambiguous cluster structure when clusters merge.

## Figures (`figures/`)

23 figures generated as both PDF (publication) and PNG (preview):

| Script | Figures | Description |
|--------|---------|-------------|
| `03_clam_heatmaps.py` | 4 | Raw + sorted CLAM heatmaps, multipanel K comparison, HDBSCAN CLAM |
| `04_stability_strips.py` | 2 | Per-sample cluster assignment stacked bars (KMeans + Ward) |
| `05_metrics_curves.py` | 6 | CRI/WCRI/TWCRI + ARI/AMI error bars for 3 datasets x 2 methods |
| `06_method_comparison.py` | 6 | Side-by-side method comparison + K* bar charts |
| `07_surfaces.py` | 5 | 3D CLAM surface, TWCRI landscape (2D+3D), HDBSCAN sensitivity (2D+3D) |

## Reproducing

```bash
cd plotting_experiments
pip install -e ".[all]"   # from repo root
make clean && make all     # regenerate everything from scratch
```

All datasets use `random_state=42`. ERICA uses `random_seed=123` (default). Results are fully deterministic.
