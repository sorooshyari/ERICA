# What We Did (And Why The Clusters Had It Coming)

*A field guide to the plotting experiments, for those who like their statistics visual and their cluster assignments stable.*

---

## The Big Picture

We built a complete visualization laboratory for ERICA, the clustering replicability tool that asks the question every unsupervised learner is afraid to hear: "But would you find those same clusters again?"

Turns out, some clusters are real. Some are figments of your K-Means' imagination. And some are just noise wearing a trench coat pretending to be structure.

Here's how we figured out which is which.

---

## Act I: The Infrastructure

### The ERICA Upgrade (a.k.a. "The Flattening")

Before we could plot anything, we shipped a major API update:

- **Flattened method parameter**: No more `method='both'` with a mysterious `linkages` list. Now it's `method=['kmeans', 'agglomerative_ward', 'hdbscan']`. You say what you mean, ERICA does what you say.
- **HDBSCAN support**: The density-based kid on the block. Doesn't need you to tell it K. Finds it on its own. Sometimes finds nothing. (That's informative too.) *Note: HDBSCAN integration is still under validation.*
- **ARI/AMI metrics**: Parmigiani's partition comparison metrics, baked in. Because CRI tells you if clusters are stable, but ARI tells you if the train-fitted model generalizes to held-out data. Belt and suspenders.

55 tests. All passing. Zero regressions.

### The Style Module

One `style.py` to rule them all. Viridis colormap (because we care about colorblind colleagues), Okabe-Ito palette for method comparisons, journal-ready figure sizes (3.5" single column, 7.0" double). Every plot script imports from the same source of truth.

---

## Act II: The Datasets

### Real Data: VDX Breast Cancer (344 samples, 3 genes)

From Parmigiani et al.'s replicability paper. Three genes: ESR1, ERBB2, AURKA. These aren't random genes. They're the molecular equivalent of asking someone their age, income, and whether they prefer cats or dogs. Surprisingly informative.

### Parmigiani et al. Complete Dataset Inventory

The original paper uses 23 datasets total:
- **3 breast cancer cohorts** (VDX 344, Mainz 200, Transbig 198 samples) with 5 feature representations each (3-gene, PAM50, PAM50+PCA, top 0.5% variable, top+PCA)
- **8 benchmark shape datasets** (Frantti & Sieranoja 2018: aggregation, compound, d31, flame, jain, pathbased, r15, spiral)
- **6 high-dimensional null Gaussians** (dim 32 through 1024, testing curse of dimensionality)
- **6 additional synthetic GMMs** in the appendix

We currently use VDX 3-gene. The full inventory is documented in the gallery HTML.

### Synthetic Toybox

| Dataset | What It Is | What It Tests |
|---------|-----------|---------------|
| `well_separated` | Three blobs that wouldn't be caught dead overlapping | "Can you pass the easy exam?" |
| `overlapping` | Four blobs at a party where personal space is optional | Boundary ambiguity |
| `moons_2d` | Two interleaving crescents | Non-convex shapes (K-Means' kryptonite) |
| `circles_2d` | Concentric rings | Nested topology (everyone's kryptonite) |
| `blobs_2d` | Three tidy Gaussians in flatland | The control group |
| `high_dim` | Three clusters in 50 dimensions | The curse of dimensionality, domesticated |

K-Means looked at the moons dataset and said K*=4. It literally cut each moon in half. This is not a bug. This is K-Means being honest about its geometric worldview: "I only understand circles, and these aren't circles."

### Gaussian Mixtures

Six 2D Gaussian mixtures with varied covariance structures (spherical, anisotropic, mixed variance, overlapping pair, five clusters, 32D). Plus the main event:

### Gaussian Sigma Study (The Original Recipe)

Four datasets recreating the original `Gaussian_mix_gen_2-Shawn.ipynb` experiments. Each has 4 Gaussian centers in 50 dimensions, 100 samples per center:

| Sigma | Radius/Spacing | Translation |
|-------|---------------|-------------|
| 0.01 | 0.01 | Clusters so tight they're basically delta functions. |
| 0.1 | 0.07 | Comfortably separated. The "control." |
| 1.0 | 0.75 | Tails are touching. This is where things get interesting. |
| 10.0 | 7.45 | One big diffuse blob. Asking "which cluster?" is like asking a water molecule which ocean. |

---

## Act III: The Visualizations

### 612 Figures, 17 Scripts, 3 HTML Galleries

We generated figures across multiple visualization types, organized into three galleries.

### Gallery 1: Main Figures (`2026-03-30-gallery.html`)

Standard ERICA outputs with full dataset index and searchable metrics table (374 rows).

**CLAM Heatmaps**: The CLAM matrix is ERICA's soul. Sort the rows by primary cluster and you get beautiful diagonal blocks. Or a mess. Both are informative.

**Stability Strips**: Horizontal stacked bars showing each sample's cluster assignment proportions, sorted by entropy.

**Metrics Curves**: CRI (now ERICA statistic), WCRI, TWCRI plotted against K, with K* marked. Plus ARI/AMI with error bars.

**Method Comparison**: K-Means vs Ward vs HDBSCAN*, side by side.

**3D Surfaces**: CLAM topography, TWCRI landscapes, HDBSCAN parameter sensitivity.

**Cluster Scatter Plots**: For 2D datasets — points colored by cluster assignment, sized by confidence.

**K Progressions**: The definitive view. 3 methods x 5 K values grid. Watch clusters subdivide as K increases. Available for all 26 datasets with both scatter (2D) and CLAM heatmap (high-D) versions, plus 3D surface progressions.

### Gallery 2: Entropy & CRI Playground (`2026-03-30-playground.html`)

Experimental visualizations exploring what the CLAM matrix tells us about per-sample uncertainty.

**Entropy Fields**: For 2D datasets, Shannon entropy of each sample's CLAM row interpolated across feature space. Shows where clustering is uncertain. Computed as H(i) = -sum(p_j * log2(p_j)) where p_j = CLAM[i,j] / sum(CLAM[i,:]).

**Entropy Delta Maps**: The money visualization. Shows WHERE entropy changes when going from K to K+1. Red = that region got more confused (bad split). Blue = it got clearer (good refinement). This pinpoints the optimal K better than any single metric.

**Entropy Surfaces (3D)**: Raised surfaces over 2D feature space with Z = entropy. The "uncertainty mountains."

**Method Entropy Comparison**: Same dataset, same K, different methods side by side. Shows where each method is uncertain.

**CRI/ERICA Statistic Fields**: Same spatial interpolation but with the ERICA statistic (per-sample CRI) instead of entropy. Green = stable, red = unstable.

**CRI Delta Maps**: Spatial change in per-sample replicability K to K+1. Blue = more replicable, red = less.

**CRI vs Entropy Scatter**: The relationship between the two measures across 12 datasets.

**Confidence Surfaces & Volatility Maps**: Additional per-sample stability views.

### Gallery 3: ERICA Statistic Deep Dive (`2026-03-30-erica-statistic-playground.html`)

Focused on the ERICA statistic (formerly CRI) across all datasets including VDX and the Gaussian sigma study.

**The ERICA Statistic**: For sample i, ERICA_stat(i) = max(CLAM[i,:]) / sum(CLAM[i,:]). The proportion of Monte Carlo iterations in which that sample was assigned to its primary cluster. Ranges from 1/K (random) to 1.0 (perfectly stable).

**Distribution Histograms**: How many samples are stable vs confused at each K, overlaid for all methods.

**Per-Cluster Boxplots**: Which clusters are internally stable vs fragile.

**Sample Trajectory Heatmaps**: Each sample's ERICA statistic across K=2..6. Shows how individual stability changes as K varies. Green bands = always stable. Red-green transitions = samples that lose/gain stability at specific K values.

**Mean ERICA Stat vs K**: Aggregate stability decline curve per method.

**Per-Cluster ERICA Stat vs K**: Which clusters hold together as K increases past the true K.

**Method Comparison Profiles**: Ranked bar charts showing the full distribution of per-sample stability for each method.

**Cross-Sigma Shift**: How the ERICA statistic distribution slides from a spike at 1.0 (sigma=0.01) to a uniform spread (sigma=10.0).

**VDX vs Gaussian Comparison**: Where does real breast cancer data fall on the synthetic spectrum.

### Gaussian Mixture Study (`gaussian_mixture_study/`)

Dedicated subfolder with its own experiment plan, 5 scripts, and figures. Recreates the original MCSS experiments. Key output: TWCRI-vs-sigma curve showing the "replicability cliff."

---

## Act IV: What We Explored

1. **The ERICA statistic works as expected.** Perfect stability at low sigma, degradation as clusters overlap, collapse when there's no structure.

2. **Method choice matters.** K-Means can't do non-convex. Single linkage oversplits. Ward is a good default. HDBSCAN* sometimes says "no clusters" and may be right.

3. **The entropy delta map is the most informative new visualization.** It shows spatially where adding a cluster helps vs hurts. The transition from red to blue pinpoints the optimal K.

4. **The ERICA statistic trajectory heatmap** (samples x K, color = stability) shows which samples are robust across K and which are boundary cases. This is directly useful for identifying "confident" vs "ambiguous" samples in real data.

5. **Per-cluster ERICA statistic vs K** reveals which clusters are structural (hold together across K) vs artifacts (dissolve at K+1).

---

## Future Experiments

The `future_experiments/` folder has a proposal for **Observation Dropout & Dimensionality Reduction**: randomly remove samples and reduce dimensions via PCA, then watch what happens to the CLAM matrix. Hypothesis: if clustering signal lives in a low-dimensional subspace, PCA should *improve* replicability.

---

## File Census

```
plotting_experiments/
├── 12 Python scripts (01-11 + style.py)
├── gaussian_mixture_study/        5 scripts + experiment plan
├── future_experiments/            dropout study proposal
├── data/                          26+ datasets (.npz)
├── results/                       26+ result bundles (.joblib)
├── figures/                       612 PDFs + 612 PNGs
│   ├── by_dataset/                26 subdirectories, full suite each
│   ├── k_progressions/            13 K=2..6 grids
│   ├── gaussian_study/            9 sigma study figures
│   └── playground/                101 entropy/CRI experiments
│       └── erica_statistic/       68 ERICA statistic deep dives
├── 2026-03-30-gallery.html        Main gallery (datasets, metrics table, standard plots)
├── 2026-03-30-playground.html     Entropy & CRI spatial experiments
├── 2026-03-30-erica-statistic-playground.html  ERICA statistic deep dive
├── metrics_table.tsv              374-row complete metrics export
├── DATA_DOCUMENTATION.md          Dataset specs and result structure
├── PLOTTING_GUIDE.md              ERICA API reference for plotting
├── WHAT_WE_DID.md                 This file
└── README.md                      Quick start
```

---

## Gallery Index

| Page | Figures | What's in it |
|------|---------|-------------|
| `2026-03-30-gallery.html` | ~80 | Dataset index, metrics table, CLAM heatmaps, stability, metrics curves, method comparison, scatter, K progressions, surfaces, Gaussian sigma study |
| `2026-03-30-playground.html` | 101 | Entropy fields, entropy delta maps, entropy surfaces, method entropy comparison, CRI fields, CRI deltas, CRI vs entropy, confidence surfaces, volatility |
| `2026-03-30-erica-statistic-playground.html` | 68 | ERICA stat distributions, per-cluster boxplots, trajectory heatmaps, mean vs K, per-cluster vs K, method profiles, cross-sigma, VDX comparison |

---

*"All clusters are wrong, but some are replicable." — George Box, probably, if he'd worked on unsupervised learning.*

<sub>*HDBSCAN integration is under active development and not yet fully validated.</sub>
