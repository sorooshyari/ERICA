# Gaussian Mixture Replicability Study

## Purpose

Recapitulate the original synthetic Gaussian mixture experiments from the MCSS/ERICA development (see `clustering_1 (73).pdf`) using the updated ERICA Python package. This validates that ERICA produces the expected behavior: high replicability for well-separated clusters, degrading as variance increases and clusters overlap.

This also serves as a side-by-side comparison with the Parmigiani et al. real-data results (VDX breast cancer dataset).

## Experimental Design

### Synthetic Data: 4-Center Gaussian Mixtures

Four datasets, each with **4 Gaussian components in 100 dimensions**, varying only the standard deviation (sigma). This directly mirrors the original `Gaussian_mix_gen_2-Shawn.ipynb` setup.

| Dataset | Sigma | Description | Expected K* |
|---------|-------|-------------|-------------|
| `gauss4c_sigma0p01` | 0.01 | Extremely tight. Clusters are point masses. | 4 (trivial) |
| `gauss4c_sigma0p1` | 0.1 | Tight. Well-separated relative to center spacing (inter/intra ratio ~9.5). | 4 (easy) |
| `gauss4c_sigma1p0` | 1.0 | Severe overlap. In 100D, expected radius (~10) approaches inter-center distance (~9.5). | 3 or 2 |
| `gauss4c_sigma10p0` | 10.0 | Massive overlap. Clusters indistinguishable (radius ~100 >> spacing ~9.5). | 2 (everything merges) |

**Separation math:** Adjacent centers are separated by `3 * sqrt(10) = 9.49` in Euclidean distance (spacing of 3 across 10 informative dimensions). The expected distance from a cluster center is `sigma * sqrt(d)`. At sigma=1.0 in 100D, this is `1 * sqrt(100) = 10`, so clusters overlap heavily. At sigma=0.1, it is only `1.0`, giving clear separation.

**Data generation parameters:**
- **Dimensions:** 100
- **Samples per cluster:** 100 (400 total per dataset)
- **Number of centers:** 4
- **Center placement:** Center i has value `i * 3` in dimensions 0-9, zero in dimensions 10-99. This gives inter-center Euclidean spacing of 9.49.
- **Covariance:** Isotropic (sigma^2 * I_100) per cluster
- **Random seed:** 42 (for data generation; ERICA uses its own seed=123)

**Status:** `gauss4c_sigma0p01` and `gauss4c_sigma0p1` are already generated in `data/`. The sigma=1.0 and sigma=10.0 datasets need to be created by `01_generate_gaussians.py`.

### Real Data: VDX Breast Cancer (Parmigiani)

The VDX 3-gene subset (ESR1, ERBB2, AURKA) from Parmigiani et al. is already generated and available in `plotting_experiments/data/vdx_3gene.npz` (344 samples, 3 features). ERICA results for VDX are already computed in `plotting_experiments/results/vdx_3gene.joblib`.

### ERICA Pipeline Configuration

All datasets run with the same configuration:

| Parameter | Value |
|-----------|-------|
| `method` | `['kmeans', 'agglomerative_ward', 'agglomerative_single', 'hdbscan']` |
| `k_range` | `[2, 3, 4, 5, 6]` (matching existing results) |
| `n_iterations` | 200 |
| `train_percent` | 0.8 (80-20 split, matching original) |
| `transpose` | False |
| `hdbscan_params` | `{'min_cluster_size': 15}` (matching existing runs; keeps cluster threshold feasible given ~80 samples per cluster in train split) |

**Note on HDBSCAN at high sigma:** At sigma=10.0, data forms a single diffuse blob. HDBSCAN will likely classify most points as noise or find only 1 cluster. This is expected and informative -- it demonstrates HDBSCAN's behavior when no cluster structure exists.

## Analysis Plan

### 1. Per-Dataset K Progression (K=2..6)

For each of the 4 sigma levels, generate a **3-row x 5-column grid**:
- Rows: K-Means, Ward, Single Linkage (3 rows; HDBSCAN shown as a separate single-panel below each grid at its modal K)
- Columns: K = 2, 3, 4, 5, 6
- Cell content: Sorted CLAM heatmap (since data is 100D, not directly plottable as scatter)
- K* panel highlighted with orange border

This shows how each method's cluster assignments evolve as K increases, and whether K* lands at the true K=4.

### 2. Replicability vs Sigma Curve

A summary figure showing how TWCRI at true K=4 degrades across sigma levels:
- X-axis: sigma (log scale: 0.01, 0.1, 1, 10)
- Y-axis: TWCRI at K=4
- One line per method (K-Means, Ward, Single)
- HDBSCAN shown as discrete markers at its modal K's TWCRI (since it does not use a fixed K)
- Shows the "replicability frontier" -- at what sigma does each method lose the signal

Also: ARI at K=4 vs sigma, same format. Include AMI as a supplemental panel.

### 2b. HDBSCAN Noise Fraction vs Sigma

- X-axis: sigma (log scale)
- Y-axis: Mean noise fraction across iterations (from `noise_counts`)
- Shows how HDBSCAN's noise behavior changes as cluster structure degrades

### 3. K* Selection Summary

Table and bar chart showing which K* each method selects at each sigma level:

| Sigma | K-Means K* | Ward K* | Single K* | HDBSCAN modal_k |
|-------|-----------|---------|-----------|-----------------|
| 0.01  | ?         | ?       | ?         | ?               |
| 0.1   | ?         | ?       | ?         | ?               |
| 1.0   | ?         | ?       | ?         | ?               |
| 10.0  | ?         | ?       | ?         | ?               |

### 4. Comparison with VDX Real Data

Side-by-side: VDX metrics curves vs the sigma=0.1 and sigma=1.0 Gaussian mixtures. At sigma=0.1, clusters are well-separated (like clear molecular subtypes); at sigma=1.0, overlap is severe (ratio ~0.95). Comparing VDX to both endpoints contextualizes where real-data ambiguity falls on the synthetic spectrum.

**Note on empty clusters:** At high sigma with K=5,6, single linkage may produce empty clusters. These are tracked via `erica.get_disqualified_k()` and should be noted in results.

## File Organization

```
plotting_experiments/
├── data/                          # ALL datasets (shared, gitignored)
│   ├── vdx_3gene.npz             # Already exists
│   ├── gauss4c_sigma0p01.npz     # Already exists
│   ├── gauss4c_sigma0p1.npz      # Already exists
│   ├── gauss4c_sigma1p0.npz      # To generate
│   └── gauss4c_sigma10p0.npz     # To generate
├── results/                       # ALL ERICA results (shared, gitignored)
│   ├── vdx_3gene.joblib          # Already exists
│   ├── gauss4c_sigma0p01.joblib  # Already exists (re-run if config changes)
│   ├── gauss4c_sigma0p1.joblib   # To compute
│   ├── gauss4c_sigma1p0.joblib   # To compute
│   └── gauss4c_sigma10p0.joblib  # To compute
├── gaussian_mixture_study/        # This experiment's scripts and docs
│   ├── EXPERIMENT_PLAN.md         # This file
│   ├── 01_generate_gaussians.py   # Generate the 4 sigma-level datasets
│   ├── 02_run_erica.py            # Run ERICA pipeline on all 4
│   ├── 03_k_progression_grids.py  # Per-dataset CLAM heatmap progressions
│   ├── 04_sigma_curves.py         # TWCRI/ARI vs sigma summary curves
│   ├── 05_vdx_comparison.py       # Side-by-side with real data
│   └── RESULTS.md                 # Filled in after running (findings, observations)
└── figures/
    └── gaussian_study/            # Output subfolder (gitignored)
        ├── k_prog_sigma0p01.pdf
        ├── k_prog_sigma0p1.pdf
        ├── k_prog_sigma1p0.pdf
        ├── k_prog_sigma10p0.pdf
        ├── twcri_vs_sigma.pdf
        ├── ari_vs_sigma.pdf
        ├── hdbscan_noise_vs_sigma.pdf
        ├── kstar_summary.pdf
        └── vdx_comparison.pdf
```

Data and results are saved to the shared `data/` and `results/` directories (already gitignored) so they're accessible to both this study's scripts and the existing plotting scripts.

## ERICA Functions Invoked

| Step | ERICA API | What it does |
|------|-----------|-------------|
| Data generation | None (pure numpy) | `np.random.multivariate_normal()` |
| Pipeline | `ERICA(data, method=[...], k_range=[2..6], ...)` | Initializes with flattened method list |
| Pipeline | `erica.run()` | Runs MC subsampling, clustering, metrics |
| Results | `erica.get_results()['clam_matrices'][(k, method)]` | CLAM matrix per K/method |
| Results | `erica.get_results()['metrics'][k][method]['TWCRI']` | Replicability metric |
| Results | `erica.get_results()['metrics'][k][method]['ARI_mean']` | Parmigiani ARI metric |
| Results | `erica.get_results()['metrics'][k][method]['AMI_mean']` | Parmigiani AMI metric |
| Results | `erica.get_results()['k_star']['TWCRI'][method]` | Optimal K* |
| Results | `erica.get_auto_k_results('hdbscan')` | HDBSCAN auto-K results |
| Results | `erica.get_auto_k_results('hdbscan')['noise_counts']` | HDBSCAN noise per iteration |

## Relationship to Original Experiments

This study maps directly to the original `Gaussian_mix_gen_2-Shawn.ipynb`:

| Original | ERICA Equivalent |
|----------|-----------------|
| `Gaussian_mix_gen_2-Shawn.ipynb` data gen | `01_generate_gaussians.py` |
| `align_cluster_identities_1.r` | `erica.clustering._align_cluster_identities()` (L2-norm sorting) |
| `clam_matrix_form_1.r` + `sum_to_form_final_CLAM_matrix.r` | `erica.clustering._generate_clam_matrix()` |
| `clam_matrix_sort_xxx.m` scatter/histograms | `03_k_progression_grids.py` sorted CLAM heatmaps |
| Manual K-Means / AC / HDBSCAN runs | `ERICA(method=['kmeans', 'agglomerative_ward', ...])` — single call |

The key improvement: what originally required separate R scripts, manual alignment, and per-method notebooks is now a single `ERICA(...).run()` call with the flattened method list.
