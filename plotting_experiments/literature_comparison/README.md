# Literature Comparison Experiment

*2026-03-31 — Making ERICA's plots look like they went to the same school as Monti, Tibshirani, and Parmigiani*

## What We're Doing

Adding the three figure types that every reviewer in this space expects to see but ERICA doesn't yet produce. Based on our survey of 11 foundational papers (see `LITERATURE_FIGURE_SURVEY.md`).

## The Three Additions

- **Error bands on metric curves** — Every serious stability paper shows uncertainty on metric-vs-K plots. We have 200 iterations of per-sample data. We just... weren't showing the variance. Embarrassing, like having a confidence interval and keeping it to yourself. (cf. Tibshirani et al. [3], Tibshirani & Walther [2])

- **Sample x sample co-assignment heatmap** — The Monti consensus matrix. THE figure in clustering stability. "How often were samples i and j in the same cluster across iterations?" Reordered to show block-diagonal structure. If you've read a clustering paper in the last 20 years, you've seen this plot. ERICA computes the CLAM matrix (sample x cluster) but not the co-assignment matrix (sample x sample). One is the other's extroverted cousin. (Monti et al. [1], Fig. 2)

- **PCA scatter colored by ERICA statistic** — For high-D datasets where we can't do spatial plots directly. Project to 2D via PCA, color each point by its ERICA statistic. Parmigiani does this with tSNE in their Figures 4 and 6. We'll do PCA because it's deterministic and doesn't require tuning perplexity (life's too short). (Masoero et al. [4], Figs. 4 & 6)

## Datasets

Running on everything we have results for, but the key comparisons:
- VDX 3-gene (the real data anchor)
- Gaussian sigma study (0.01, 0.1, 1.0, 10.0)
- Moons and blobs (2D, for sanity checking)

## Scripts

| Script | What | Based on |
|--------|------|----------|
| `01_error_bands.py` | Metric curves with ±1 std shading | Parmigiani Fig 5, Gap Statistic |
| `02_coassignment_heatmap.py` | Sample x sample co-assignment matrix | Monti 2003 Fig 2 |
| `03_pca_erica_scatter.py` | PCA projection colored by ERICA stat | Parmigiani Fig 4, 6 |

## Outputs

Figures go to `../figures/literature_comparison/`

## References

[1] Monti, S., Tamayo, P., Mesirov, J., and Golub, T. (2003). Consensus Clustering: A Resampling-Based Method for Class Discovery and Visualization of Gene Expression Microarray Data. *Machine Learning*, 52, 91–118. https://doi.org/10.1023/A:1023949509487

[2] Tibshirani, R. and Walther, G. (2005). Cluster Validation by Prediction Strength. *Journal of Computational and Graphical Statistics*, 14(3), 511–528. https://doi.org/10.1198/106186005X59243

[3] Tibshirani, R., Walther, G., and Hastie, T. (2001). Estimating the Number of Clusters in a Data Set via the Gap Statistic. *Journal of the Royal Statistical Society: Series B (Statistical Methodology)*, 63(2), 411–423. https://doi.org/10.1111/1467-9868.00293

[4] Masoero, L., Thomas, E., Parmigiani, G., Tyekucheva, S., and Trippa, L. (2023). Cross-Study Replicability in Cluster Analysis. *Statistical Science*, 38(2), 303–316. https://doi.org/10.1214/22-STS871

[5] Hennig, C. (2007). Cluster-wise Assessment of Cluster Stability. *Computational Statistics and Data Analysis*, 52(1), 258–271. https://doi.org/10.1016/j.csda.2006.11.025
