# What Do the Papers Plot? A Survey of Standard Figures in Clustering Stability Literature

*2026-03-31 — Compiled for CMAC*

We looked at the foundational papers in clustering replicability and validation to see which figure types are standard. The goal: make ERICA's visualizations maximally comparable with what reviewers expect to see.

---

## The Papers

| Short Name | Citation | Metric | Year |
|-----------|---------|--------|------|
| Gap Statistic | Tibshirani, Walther & Hastie. "Estimating the number of clusters in a data set via the gap statistic." *JRSS-B* 63(2):411-423. | Gap(K) | 2001 |
| Ben-Hur | Ben-Hur, Elisseeff & Guyon. "A stability based method for discovering structure in clustered data." *PSB* 2002. | Pairwise similarity CDF | 2002 |
| Clest | Dudoit & Fridlyand. "A prediction-based resampling method for estimating the number of clusters in a dataset." *Genome Biology* 3(7). | d_K (prediction-based) | 2002 |
| Monti | Monti, Tamayo, Mesirov & Golub. "Consensus clustering." *Machine Learning* 52:91-118. | Consensus matrix, CDF area | 2003 |
| Lange | Lange, Roth, Braun & Buhmann. "Stability-based validation of clustering solutions." *Neural Computation* 16(6):1299-1323. | Stability index vs null | 2004 |
| Prediction Strength | Tibshirani & Walther. "Cluster validation by prediction strength." *JCGS* 14(3):511-528. | ps(K) | 2005 |
| Hennig | Hennig. "Cluster-wise assessment of cluster stability." *CSDA* 52(1):258-271. | Per-cluster Jaccard | 2007 |
| von Luxburg | von Luxburg. "Clustering stability: An overview." *FnTML* 2(3):235-274. | (survey) | 2010 |
| Parmigiani | Masoero, Camerlenghi, Favaro, Broderick & Parmigiani. "Cross-study replicability in cluster analysis." *Stat. Science* 38(2):303-316. | ARI/AMI | 2023 |
| Liu review | Liu, Yang, Boutemedjet & Zhou. "Stability estimation for unsupervised clustering: A review." *WIREs Comp Stats* 2022. | (survey) | 2022 |
| ConsensusClusterPlus | Wilkerson & Hayes. *Bioinformatics* 26(12):1572-1573. | (implementation) | 2010 |

---

## Figure Types: What the Field Expects

### 1. Stability Profile Plot (the universal one)

Every paper has this. X = K, Y = their metric. Lines, sometimes with error bars or confidence bands.

| Paper | Y-axis | Error representation | K selection rule |
|-------|--------|---------------------|-----------------|
| Gap Statistic | Gap(K) | ±1 SE bars | First K where Gap(K) ≥ Gap(K+1) - SE |
| Prediction Strength | ps(K) | None shown | Largest K where ps(K) > 0.8 (or 0.9) |
| Clest | d_K | Null reference line | Peak d_K above threshold |
| Lange | Stability index | Null reference curve | Peak above null |
| Parmigiani | ARI or AMI | Bootstrap shaded bands | Peak |
| Fang & Wang | Instability | None | Minimum instability |

**What ERICA has:** `plot_metrics()` with CRI/WCRI/TWCRI lines. `plot_k_star_selection()` with K* marked.

**What ERICA is missing:** Error bands or confidence intervals on the metric lines. Every serious paper in this space shows uncertainty. This is probably the single highest-priority addition.

### 2. CDF of Pairwise Similarity (Ben-Hur / Monti)

Multiple curves overlaid, one per K. X = similarity score (0-1), Y = cumulative proportion. Tight CDF near 1.0 = stable K. Broad/bimodal = unstable.

This is the most visually distinctive figure in the field. If you've read clustering stability papers, you've seen this plot. Immediately recognizable.

**What ERICA has:** Nothing equivalent.

**What it would take:** From the CLAM matrix, compute pairwise co-assignment frequency (how often were samples i and j in the same cluster). This is a sample x sample matrix. The CDF is the distribution of those frequencies at each K.

### 3. Consensus / Co-Assignment Heatmap (Monti)

Sample x sample heatmap. Entry (i,j) = fraction of iterations where i and j were co-clustered. Reordered by hierarchical clustering to reveal block structure. The canonical output of consensus clustering.

**What ERICA has:** `plot_clam_heatmap()` — but this is sample x cluster, not sample x sample. Related but different. The CLAM tells you about each sample's cluster membership. The co-assignment matrix tells you about pairwise relationships between samples.

**What it would take:** For each pair of samples, count how many iterations they appeared in the same cluster when both were in the test set. Straightforward to compute from the existing CLAM data, just needs an O(n^2) loop.

### 4. Per-Cluster Stability Bar Chart (Hennig)

X = cluster index, Y = mean Jaccard similarity between that cluster and its best match across bootstrap resamples. Threshold lines at 0.6 ("dissolved"), 0.75 ("pattern"), 0.85 ("stable") per Hennig's guidelines.

This is the standard output of `clusterboot()` in R. Expected whenever someone draws per-cluster conclusions.

**What ERICA has:** The per-cluster ERICA statistic boxplots we just built in the playground are in this spirit but use a different metric (ERICA stat vs Jaccard). The thresholds would need calibration.

### 5. Per-Sample Local Replicability (Parmigiani)

Scatter plot in PCA or tSNE space, colored by per-sample replicability score. Shows which regions of the data have stable vs unstable cluster assignments.

Parmigiani uses this in Figures 4 and 6 — it's their most scientifically distinctive contribution. The spatial view of "where is clustering reliable?" is particularly valuable for genomics where you want to know which patient subtypes are robust.

**What ERICA has:** The confidence scatter and entropy field plots we built in the playground. These are similar in spirit. The main difference: Parmigiani uses PCA/tSNE for high-D data projection, we only do spatial plots for natively 2D datasets.

### 6. Null Reference Comparison

Real-data metric curve overlaid with the same metric computed on randomly permuted (null) data. The gap between curves indicates genuine structure.

Used by: Gap Statistic (this IS the gap), Clest (d_K relative to null), Lange (stability ratio).

**What ERICA has:** Nothing equivalent. This would strengthen claims about detecting the absence of structure — important for the sigma=10.0 case where we want to show ERICA correctly reports "no clusters."

### 7. Delta Area Plot (Monti)

Bar chart of relative change in area under the CDF curve as K increases. Companion to Figure Type 2. Optimal K = where delta drops to near zero.

**What ERICA has:** Nothing equivalent. Dependent on having the CDF figure first.

### 8. Item Tracking Plot (ConsensusClusterPlus)

Matrix where X = samples, Y = K values (2..Kmax), color = cluster assignment. Shows how samples move between clusters as K grows.

**What ERICA has:** The ERICA statistic trajectory heatmap (samples x K, color = ERICA stat) is conceptually similar but shows stability, not assignment identity. We could make the tracking version by coloring by primary cluster instead of ERICA stat.

### 9. Silhouette Plot

Horizontal bars sorted within clusters by silhouette width. Internal validity metric (not resampling-based). Used as a comparison baseline in Dudoit & Fridlyand. Not in ERICA's remit since ERICA is stability-focused.

---

## Gap Analysis: ERICA vs the Field

| What the Field Expects | ERICA Has? | Priority | Effort |
|----------------------|-----------|----------|--------|
| Metric profile vs K with error bands | Partial (no error bands) | **High** | Low — add ±1 std shading |
| Sample x sample co-assignment heatmap | No | **High** | Medium — O(n^2) computation from CLAM |
| Per-cluster Jaccard stability bars | No (have boxplots instead) | **Medium** | Medium — need Jaccard computation |
| Per-sample scatter in PCA/tSNE space | No (have 2D scatter only) | **Medium** | Low — add PCA projection step |
| Null reference comparison | No | **Medium** | Medium — need permutation baseline |
| CDF of pairwise similarity | No | Lower | Medium — need co-assignment matrix first |
| Delta area plot | No | Lower | Low once CDF exists |
| Item tracking plot | Partial (trajectory heatmap) | Lower | Low — recolor by cluster ID |

---

## Recommendations

If we want ERICA's figures to look like they belong in the same conversation as these papers, the three highest-value additions seem to be:

1. **Error bands on metric-vs-K curves.** Every paper has them. We have the per-iteration data to compute them. This is probably an afternoon of work.

2. **Sample x sample co-assignment heatmap.** The Monti consensus matrix is the most recognizable figure in this space. Building it from ERICA's existing CLAM data is straightforward — we already have the per-iteration assignments.

3. **PCA/tSNE scatter colored by ERICA statistic** for high-D datasets. We do this for 2D datasets already. Adding a PCA projection step would let us do it for VDX and the Gaussian sigma study data too.

After those three, a null reference overlay would be the next thing — it's how you argue that a flat TWCRI curve at sigma=10.0 means "no structure" rather than "ERICA is broken."

---

## Sources

- Tibshirani, Walther & Hastie (2001). JRSS-B 63(2):411-423.
- Ben-Hur, Elisseeff & Guyon (2002). PSB 2002.
- Dudoit & Fridlyand (2002). Genome Biology 3(7).
- Monti, Tamayo, Mesirov & Golub (2003). Machine Learning 52:91-118.
- Lange, Roth, Braun & Buhmann (2004). Neural Computation 16(6):1299-1323.
- Tibshirani & Walther (2005). JCGS 14(3):511-528.
- Hennig (2007). CSDA 52(1):258-271.
- von Luxburg (2010). FnTML 2(3):235-274.
- Wilkerson & Hayes (2010). Bioinformatics 26(12):1572-1573.
- Liu, Yang, Boutemedjet & Zhou (2022). WIREs Computational Statistics.
- Masoero, Camerlenghi, Favaro, Broderick & Parmigiani (2023). Statistical Science 38(2):303-316.
