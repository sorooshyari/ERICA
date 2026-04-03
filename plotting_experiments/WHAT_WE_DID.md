# Plotting Experiments — Session Notes

*2026-03-30 — Shawn Shirazi*

---

## What This Is

A working lab notebook. We set up a plotting sandbox to explore which visualizations of ERICA's output are useful for understanding clustering replicability. Most of this is rough — we threw a lot at the wall to see what sticks. Publication-quality this is not. "Does this plot actually tell us anything?" is the bar.

---

## What We Generated

### Datasets

We ran ERICA on 26 datasets. One real (VDX breast cancer 3-gene subset from Parmigiani et al.), the rest synthetic — sklearn toy shapes, Gaussian mixtures with covariances ranging from "politely separated" to "actively hostile," and a 4-center 50D Gaussian with sigma swept from 0.01 to 10.0 (i.e., from "trivially separable" to "asking a water molecule which ocean it belongs to").

The dataset details are in `DATA_DOCUMENTATION.md` and the gallery pages.

### Figures

612 total (PDF + PNG pairs). That's not a typo. Organized in `figures/` with subfolders. Most are variations on a few themes — we were not accused of undersampling the visualization space.

### Scripts

28 Python scripts generating the data, running ERICA, and producing plots (11 top-level + 5 gaussian + 3 literature + 8 erica_statistics + style.py). The Gaussian sigma study, literature comparison, and ERICA statistics exploration each have their own subfolders.

---

## What We Tried

### Standard Plots

The basics: sorted CLAM heatmaps, metrics-vs-K curves, method comparison panels, K* bar charts, stability strips, cluster scatter plots (2D datasets). K progression grids (methods x K=2..6) for every dataset. 3D CLAM surfaces and TWCRI landscapes. These are documented in `PLOTTING_GUIDE.md`.

### Entropy Experiments

We interpolated Shannon entropy of each sample's CLAM row across 2D feature space. This produces "uncertainty fields" showing where clustering is ambiguous. The entropy delta maps (change in entropy from K to K+1) turned out to be interesting — they show spatially which samples benefit vs suffer from adding a cluster. Red regions got more confused, blue regions got clearer.

For high-dimensional data where spatial plots aren't possible, we did entropy delta heatmaps (samples x K transitions) and entropy distribution histograms.

### ERICA Statistic Experiments

The ERICA statistic (formerly CRI) for a sample is the proportion of MC iterations where it was assigned to its primary cluster: max(CLAM[i,:]) / sum(CLAM[i,:]). We explored this from several angles:

- Distribution histograms at each K (how many samples are stable vs confused)
- Per-cluster boxplots (which clusters are internally consistent)
- Sample trajectory heatmaps (each sample's ERICA stat across K=2..6 — probably the most informative of the bunch)
- Per-cluster ERICA stat vs K (which clusters hold together as K increases)
- Method comparison profiles (ranked bar charts)
- Cross-sigma distribution shift (from a spike at 1.0 to uniform noise)

### Gaussian Sigma Study

Recreated the original Gaussian_mix_gen_2 setup: 4 centers in 50D, sigma = 0.01, 0.1, 1.0, 10.0. Ran all methods. The TWCRI-vs-sigma curve shows the expected degradation. At sigma=1.0 (radius/spacing ratio 0.75), methods start to diverge. At sigma=10.0, everything collapses. The details are in `gaussian_mixture_study/EXPERIMENT_PLAN.md`.

### Literature Comparison

Reproduced standard figures from the clustering stability literature (Tibshirani gap statistic, Monti consensus clustering, Masoero/Parmigiani replicability). Error band plots, co-assignment heatmaps, and PCA scatter colored by ERICA statistics. Scripts in `literature_comparison/`.

### ERICA Statistics Exploration

Deep dive into the ERICA statistic, ARI, and AMI across all datasets and methods. Eight scripts covering: distribution analysis (histograms, KDE), statistics-vs-K curves, method comparison panels, cross-dataset summary heatmaps, sigma degradation tracking, scatter plots of statistic relationships, Per-Cluster Stability Profiles (PCSP), and ICA/ICAH (Iterative Cluster Assignment / Iterative Cluster Assignment Heatmaps) analysis. The PCSP and ICA views turned out to be particularly informative — they show per-cluster stability trajectories that the aggregate metrics obscure. Gallery at `erica_statistics/2026-04-02-erica-statistics.html`.

---

## What Seems Useful So Far

Some of these visualizations are more informative than others. Preliminary impressions (subject to revision):

- The **sorted CLAM heatmap** remains the most intuitive view of what ERICA is doing
- The **entropy delta maps** (K to K+1 spatial change) are a nice way to see where adding a cluster helps vs hurts
- The **ERICA statistic trajectory heatmap** (samples x K) shows individual sample stability across K values in a way the aggregate curves don't
- The **per-cluster boxplots** quickly show which clusters are real vs fragile
- The **3D surfaces** look cool but it's not clear they earn their extra dimension. The 2D heatmaps might say the same thing with less matplotlib wrestling

### What Needs Work

- HDBSCAN integration needs validation before we present any HDBSCAN results. It's flagged as experimental throughout.
- The VDX results are interesting but we only used the 3-gene subset. The full dataset (22K genes) and the other cohorts (Mainz, Transbig) would give a more complete picture.
- Some of the colorbar alignments are still off on a few entropy plots.
- We should probably standardize the K range across all datasets (some have K=2..5, others K=2..6).

---

## File Organization

```
plotting_experiments/
├── 11 Python scripts + style.py
├── gaussian_mixture_study/           5 scripts + experiment plan
├── literature_comparison/            3 scripts + HTML gallery
├── erica_statistics/                 8 scripts + HTML gallery
├── future_experiments/               dropout/dimensionality proposal
├── data/                             26 datasets (.npz)
├── results/                          26 result bundles (.joblib)
├── figures/                          612 PDFs + 612 PNGs
│   ├── by_dataset/{name}/            full suite per dataset
│   ├── k_progressions/               K=2..6 grids
│   ├── gaussian_study/               sigma study
│   └── playground/                   entropy/CRI/ERICA stat experiments
│       └── erica_statistic/
├── 2026-03-30-gallery.html           main gallery
├── 2026-03-30-playground.html        entropy & CRI experiments
├── 2026-03-30-erica-statistic-playground.html
├── literature_comparison/2026-03-31-literature-comparison.html
├── erica_statistics/2026-04-02-erica-statistics.html
├── metrics_table.tsv                 374-row metrics export
├── DATA_DOCUMENTATION.md
├── PLOTTING_GUIDE.md
├── WHAT_WE_DID.md                    this file
└── README.md
```

---

## Next Steps

- Validate HDBSCAN implementation (it's giving us looks but we don't trust it yet)
- Run on full VDX + Mainz + Transbig datasets (not just the 3-gene appetizer)
- Observation dropout experiment (proposed in `future_experiments/`)
- Decide which of these 612 figures actually deserve to be in a paper (spoiler: not 612 of them)

---

*"All clusters are wrong, but some are replicable."*
