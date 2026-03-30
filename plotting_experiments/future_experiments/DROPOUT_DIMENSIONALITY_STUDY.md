# Experiment: Observation Dropout and Dimensionality Reduction Effects on CLAM Structure

**Status:** Proposed (not yet implemented)
**Author:** Shawn Shirazi
**Date:** 2026-03-30

## Research Question

Does the internal structure revealed by the CLAM matrix change meaningfully as we:
1. Randomly drop observations (samples) and re-run clustering
2. Reduce dimensionality of the feature space

If CLAM structure is robust to these perturbations, it suggests the clustering captures genuine signal rather than artifacts of sample composition or high-dimensional noise.

## Motivation

ERICA's Monte Carlo subsampling already tests stability across random train/test splits. This experiment goes further by asking: what happens to the CLAM matrix itself when we systematically degrade the input data? Two degradation axes:

- **Observation dropout**: Remove 10%, 20%, ..., 50% of samples before running ERICA. Does K* change? Do the same samples remain stably assigned? This tests whether replicability depends on specific samples or reflects population-level structure.

- **Dimensionality reduction**: Project data to fewer dimensions (via PCA or random projection) before clustering. Does CLAM structure survive compression? At what dimensionality does it break down? This tests whether the clustering signal lives in a low-dimensional subspace.

## Proposed Setup

### Datasets

Use the existing toy datasets from `plotting_experiments/data/`:
- `moons_2d` (non-convex, 2D) — easy to visualize dropout effects
- `blobs_2d` (convex, 2D) — baseline comparison
- `well_separated` (5D) — moderate dimensionality
- `high_dim` (50D) — where dimensionality reduction matters most
- `vdx_3gene` (3D, real data) — does real data behave like synthetics?

### Experiment A: Observation Dropout

For each dataset:
1. Start with the full dataset (N samples)
2. For dropout_rate in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
   a. Randomly remove `dropout_rate * N` samples (fixed random seed per rate)
   b. Run ERICA with the same config as the main experiments
   c. Record: K*, TWCRI at each K, ARI, CLAM matrix
3. Compare:
   - K* stability: does optimal K change with dropout?
   - TWCRI trajectory: does replicability degrade gracefully or cliff?
   - Sample-level CLAM comparison: for samples that survive dropout, do their CLAM assignments match the full-data assignments?

### Experiment B: Dimensionality Reduction

For high-dimensional datasets (well_separated, high_dim, vdx_3gene):
1. Apply PCA, retaining [2, 5, 10, 20, all] components
2. Run ERICA on each projected dataset
3. Compare:
   - K* at each dimensionality
   - TWCRI curves: at what PCA rank does the signal emerge?
   - Cross-dimensionality CLAM agreement: do samples get the same primary cluster at d=5 as at d=50?

### Combined Analysis

Cross the two axes: dropout rate x dimensionality. Does dropping observations hurt more in high dimensions (curse of dimensionality) or low dimensions (small sample size)?

## Expected Outputs

### Figures

1. **TWCRI vs dropout rate** — one line per method, one panel per dataset
2. **K* vs dropout rate** — grouped bar chart
3. **CLAM agreement heatmap** — for each (dropout_rate, full_data) pair, compute Adjusted Rand Index between primary cluster assignments. Shows how quickly assignments diverge.
4. **TWCRI vs PCA components** — replicability as a function of retained variance
5. **Dropout x Dimensionality surface** — 2D heatmap, TWCRI as color

### Key Questions This Answers

- Is ERICA's K* selection robust to sample perturbation?
- At what dropout rate does clustering structure collapse?
- Does dimensionality reduction help or hurt replicability? (If noise dimensions dilute signal, PCA might IMPROVE replicability.)
- Can we identify a "minimum viable dimensionality" for a given dataset?

## ERICA Functions Used

Same as the main experiments:
- `ERICA(data, method=[...], k_range=[...])` — standard pipeline
- `get_results()['clam_matrices']` — CLAM matrices for comparison
- `get_results()['k_star']` — K* tracking across perturbation levels
- `get_results()['metrics']` — TWCRI/ARI curves

Additionally, for cross-condition CLAM comparison:
- Primary cluster assignment: `np.argmax(clam_matrix, axis=1)`
- Cross-condition ARI: `sklearn.metrics.adjusted_rand_score(assignments_full, assignments_dropout)`

## Notes

- Random seeds should be controlled so that dropout at rate=0.1 is a strict subset of dropout at rate=0.2 (nested dropout). This ensures monotonic degradation rather than random variation.
- PCA should be fit on the full dataset, then the dropped-out version projected. This simulates the realistic scenario where you have a reference PCA space.
- This experiment is computationally expensive: ~6 dropout rates x ~5 PCA levels x ~5 datasets x ~4 methods = ~600 ERICA runs. Recommend reducing n_iterations to 100 for this study.
