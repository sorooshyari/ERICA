# ERICA Statistics — Exploratory Journal Entry

*2026-04-02 — Extending Parmigiani's ARI/AMI to Monte Carlo subsampling*

## What We're Exploring

ERICA adapts Parmigiani et al.'s (2023) cross-study ARI/AMI comparison to a Monte Carlo subsampling framework. Where Parmigiani compares train-study vs test-study, ERICA compares train-subsample vs test-subsample within the same dataset across B=200 MC iterations.

This produces three complementary views of clustering replicability:

- **ERICA statistic** — per-sample assignment consistency: `max(CLAM[i,:]) / sum(CLAM[i,:])`. Unique to ERICA.
- **ERICA-ARI** — Monte Carlo Adjusted Rand Index. Per-iteration: fit on train, predict on test, refit fresh on test, compute ARI(predicted, refit).
- **ERICA-AMI** — Monte Carlo Adjusted Mutual Information. Same comparison, information-theoretic variant.

The key difference from Parmigiani: these measure **within-dataset replicability under subsampling**, not cross-study replicability. The MC framework gives distributions (200 values) rather than point estimates.

## The Question

How do these three views relate? Do they agree on which datasets/methods/K-values produce replicable clusterings? Where do they diverge, and what does divergence tell us?

## Scripts

| # | Script | Output |
|---|--------|--------|
| 01 | `01_distributions.py` | `dist_{dataset}.{pdf,png}` — 1x3 histograms of ERICA stat / ARI / AMI at K* |
| 02 | `02_statistics_vs_k.py` | `stats_vs_k_{dataset}_{method}.{pdf,png}` — all 3 statistics vs K with error bands |
| 03 | `03_method_comparison.py` | `method_compare_{dataset}.{pdf,png}` — 2x3 grid: methods x statistics |
| 04 | `04_cross_dataset_summary.py` | `cross_dataset_summary.{pdf,png}` — heatmap of all datasets x statistics |
| 05 | `05_sigma_degradation.py` | `sigma_degradation.{pdf,png}` — 3-statistic degradation across Gaussian sigma |
| 06 | `06_scatter_plots.py` | `scatter_erica_*`, `triptych_*`, `scatter_method_compare_*` — spatial views |
| 07 | `07_pcsp.py` | `pcsp_{dataset}_kmeans.{pdf,png}` — Per-Cluster Scatter Plots (from original ERICA) |
| 08 | `08_ica_analysis.py` | `pca_vs_ica_*`, `icah_*` — ICA dim. reduction + Inter-Cluster Assignment Heatmap |

Figures output to `../figures/erica_statistics/`.

## Implementation Note

For K-Means, the train model genuinely predicts on test data (`.predict()`). For Agglomerative (Ward), there is no native `.predict()` — ERICA uses a centroid-based proxy. This approximation is worth keeping in mind when interpreting ERICA-ARI/AMI for Ward.

## Observations

*(To be filled in after reviewing the gallery.)*
