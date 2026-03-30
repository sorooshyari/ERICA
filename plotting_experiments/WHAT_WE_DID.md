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
- **HDBSCAN support**: The density-based kid on the block. Doesn't need you to tell it K. Finds it on its own. Sometimes finds nothing. (That's informative too.)
- **ARI/AMI metrics**: Parmigiani's partition comparison metrics, baked in. Because CRI tells you if clusters are stable, but ARI tells you if the train-fitted model generalizes to held-out data. Belt and suspenders.

55 tests. All passing. Zero regressions. The p-value of our confidence is... well, we're not frequentists, but it's small.

### The Style Module

One `style.py` to rule them all. Viridis colormap (because we care about colorblind colleagues), Okabe-Ito palette for method comparisons, journal-ready figure sizes (3.5" single column, 7.0" double). Every plot script imports from the same source of truth.

No more "why does Figure 3 use blue for K-Means but Figure 7 uses orange?" We've all been burned by that particular form of scientific sin.

---

## Act II: The Datasets

### Real Data: VDX Breast Cancer (344 samples, 3 genes)

From Parmigiani et al.'s replicability paper. Three genes: ESR1, ERBB2, AURKA. These aren't random genes. They're the molecular equivalent of asking someone their age, income, and whether they prefer cats or dogs. Surprisingly informative.

ERICA's verdict: K*=6 for both K-Means and Ward (at K_max=6). HDBSCAN finds 2 clusters with 60% agreement. The biology says "it's complicated." The statistics agree.

### Synthetic Toybox

| Dataset | What It Is | What It Tests |
|---------|-----------|---------------|
| `well_separated` | Three blobs that wouldn't be caught dead overlapping | "Can you pass the easy exam?" |
| `overlapping` | Four blobs at a party where personal space is optional | Boundary ambiguity |
| `moons_2d` | Two interleaving crescents | Non-convex shapes (K-Means' kryptonite) |
| `circles_2d` | Concentric rings | Nested topology (everyone's kryptonite) |
| `blobs_2d` | Three tidy Gaussians in flatland | The control group |
| `high_dim` | Three clusters in 50 dimensions | The curse of dimensionality, domesticated |

Fun fact: K-Means looked at the moons dataset and said K*=4. It literally cut each moon in half. This is not a bug. This is K-Means being honest about its geometric worldview: "I only understand circles, and these aren't circles."

Ward linkage got it right (K*=2). Single linkage got it right. HDBSCAN got it right (modal K=2, 93% agreement). The moral: when your data isn't convex, bring a density-based friend.

### Gaussian Mixtures (The Original Recipe)

Four datasets, same recipe, different spice level. Each has 4 Gaussian centers in 50 dimensions:

| Sigma | Radius/Spacing Ratio | Translation |
|-------|---------------------|-------------|
| 0.01 | 0.01 | Clusters so tight they're basically delta functions. A good sneeze would separate them. |
| 0.1 | 0.07 | Comfortably separated. The "control" of the experiment. |
| 1.0 | 0.75 | Tails are touching. This is where things get interesting. |
| 10.0 | 7.45 | One big diffuse blob wearing a "K=4" name tag that nobody believes. |

The separation math: centers are spaced 9.49 apart (Euclidean). A point drawn from a 50D Gaussian with sigma=1.0 is expected to be 7.07 from its center. At sigma=10, it's 70.71. At that point, asking "which cluster are you from?" is like asking a molecule of water which ocean it belongs to.

---

## Act III: The Visualizations

### 80 Figures, 10 Scripts, Zero Regrets

We generated figures across multiple visualization types. Here's the taxonomy:

**CLAM Heatmaps** (script 03): The CLAM matrix is ERICA's soul. Each row is a sample, each column is a cluster, each entry is how many times that sample got assigned to that cluster across 200 Monte Carlo iterations. Sort the rows by primary cluster and you get beautiful diagonal blocks. Or a mess. Both are informative.

**Stability Strips** (script 04): Horizontal stacked bars showing each sample's cluster assignment proportions. Sorted by entropy. The stable samples (entropy near zero) sit smugly at the top. The confused ones (entropy approaching log2(K)) huddle at the bottom, unsure of their identity. We've all been there.

**Metrics Curves** (script 05): CRI, WCRI, TWCRI plotted against K, with K* marked. Plus ARI/AMI with error bars. This is the "does increasing K help or hurt?" plot. When the TWCRI curve goes up, you've found more real structure. When it goes down, you're carving the turkey too thin.

**Method Comparison** (script 06): K-Means vs Ward vs HDBSCAN, side by side. Because the best way to evaluate a clustering method is to see what its competitors think.

**3D Surfaces** (script 07): CLAM topography, TWCRI landscapes, HDBSCAN parameter sensitivity. The 2D heatmaps are for the paper. The 3D surfaces are for the supplementary materials and/or impressing your advisor.

**Cluster Scatter Plots** (script 08): For 2D datasets only (moons, circles, blobs). Points colored by cluster assignment, sized by confidence. The moons plots are *chef's kiss*: you can literally see K-Means drawing vertical lines through curved data.

**K Progressions** (scripts 09, 10): The crown jewel. A grid showing K=2 through K=6, one column per K, one row per method. Watch the clusters subdivide as K increases. At the true K, the CLAM heatmap shows clean blocks. Above it, you see fragmentation. Below it, you see forced merging.

### The Gaussian Sigma Study (gaussian_mixture_study/)

Five dedicated scripts recreating the original Gaussian_mix_gen_2 experiments:

**TWCRI vs Sigma curve**: The money plot. Perfect replicability at sigma=0.01 and 0.1. Graceful degradation at sigma=1.0. Complete collapse at sigma=10.0. This is the "replicability frontier" — the sigma at which each method can no longer distinguish signal from noise.

**K* Summary**: At low sigma, everyone agrees: K*=4 (the truth). At sigma=1.0, single linkage panics and says K*=6. At sigma=10.0, everyone says K*=2 because there's nothing left to find. HDBSCAN at sigma=10 says modal_k=1. It's not wrong. There is, in fact, one big blob.

**HDBSCAN Noise Fraction**: At sigma=0.01, zero noise points. At sigma=10.0, almost everything is noise. HDBSCAN is the canary in the coal mine of cluster structure.

**VDX Comparison**: Where does real breast cancer data fall on the synthetic spectrum? Somewhere between sigma=0.1 (clear) and sigma=1.0 (murky). Biology is messy. Film at 11.

---

## Act IV: What We Learned

1. **ERICA works.** The new flattened API, HDBSCAN support, and ARI/AMI metrics all validated end-to-end across 13 datasets.

2. **Method choice matters.** K-Means is fast but geometrically naive. Ward is robust but still centroid-brained. Single linkage follows chains but oversplits on noise. HDBSCAN is the wise elder who sometimes says "there are no clusters here" and is usually right.

3. **The CLAM matrix is more informative than K* alone.** K* gives you a number. The sorted CLAM heatmap shows you *why* that number was chosen — which samples are stable, which are confused, where the boundaries lie.

4. **High sigma is an honest test.** When there's no structure, ERICA should say "there's no structure." TWCRI collapsing to near-zero and HDBSCAN finding modal_k=1 at sigma=10 is not a failure. It's the correct answer. The most dangerous clustering algorithm is one that always finds clusters.

5. **200 Monte Carlo iterations is plenty.** The standard deviations on ARI/AMI are tight. The CLAM matrices are well-populated. More iterations wouldn't change the story.

---

## File Census

```
plotting_experiments/
├── 11 Python scripts (01-10 + style.py)
├── gaussian_mixture_study/        (5 scripts + experiment plan)
├── future_experiments/            (dropout study proposal)
├── data/                          (13+ datasets, .npz)
├── results/                       (13+ result bundles, .joblib)
├── figures/                       (80+ PDFs and PNGs)
│   ├── k_progressions/            (13 K=2..6 grids)
│   └── gaussian_study/            (9 sigma study figures)
├── DATA_DOCUMENTATION.md          (dataset specs and result structure)
├── PLOTTING_GUIDE.md              (ERICA API reference for plotting)
├── README.md                      (quick start)
└── WHAT_WE_DID.md                 (you are here)
```

---

## What's Next

The `future_experiments/` folder has a proposal for the **Dropout and Dimensionality Reduction Study**: randomly remove samples and reduce dimensions via PCA, then watch what happens to the CLAM matrix. The hypothesis: if the clustering signal lives in a low-dimensional subspace (spoiler: for Gaussians, it does), then PCA should *improve* replicability by stripping noise dimensions. For real data, the answer is less obvious. That's why it's an experiment and not a blog post.

---

*"All clusters are wrong, but some are replicable." — George Box, probably, if he'd worked on unsupervised learning.*
