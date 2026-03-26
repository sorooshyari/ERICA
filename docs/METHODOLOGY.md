# ERICA Methodology

## Overview

ERICA (Evaluating Replicability via Iterative Clustering Assignments) is a robust method for assessing clustering stability and replicability through Monte Carlo subsampling and cluster assignment analysis.

---

## The ERICA Process

ERICA consists of four main steps:
1. Iterative Clustering Subsampling (Monte Carlo)
2. Cluster Identity Alignment
3. CLAM Matrix Generation
4. Metrics Computation (CRI, WCRI, TWCRI)
5. **K\* Selection using Algorithm 2**

### 1. Iterative Clustering Subsampling

For each iteration *b* = 1, 2, ..., *B*:

1. **Random Partitioning**: Randomly partition the dataset *D* into:
   - Training set *D_train^(b)*: Contains *p* × *n* samples (typically *p* = 0.8)
   - Test set *D_test^(b)*: Contains (1 - *p*) × *n* samples

2. **Clustering**: 
   - For K-Means: Fit the clustering model on *D_train^(b)*, predict cluster labels for *D_test^(b)*
   - For Agglomerative: Fit the clustering model on *D_test^(b)* directly (following original methodology)

3. **Store Results**: Save test indices and predicted cluster labels

### 2. Cluster Identity Alignment

The alignment process ensures that cluster labels are consistent across iterations:

1. **Global Centroids**: Compute centroids from clustering the complete dataset
2. **Sort by L2 Norm**: Order global centroids by their L2 norm for consistent reference
3. **Per-Iteration Mapping**: For each iteration *b*:
   - Calculate distances between iteration centroids and sorted global centroids
   - Use greedy assignment to map iteration labels to global labels
   - Transform iteration predictions using this mapping

This alignment is crucial because cluster labels (0, 1, 2, ...) are arbitrary and may not correspond across iterations.

### 3. CLAM Matrix Generation

The CLAM (CLuster Assignment Matrix) *M* is an *n* × *k* matrix where:

```
M[i, j] = number of times sample i was assigned to cluster j across all iterations
```

**Construction:**
- Initialize *M* as zeros matrix (*n* × *k*)
- For each iteration *b* and each test sample *i* in *D_test^(b)*:
  - Increment *M[i, aligned_label(i, b)]*
- Each row sum equals the number of times that sample appeared in test sets

**Properties:**
- Each sample appears in approximately (1 - *p*) × *B* test sets
- Row sums vary due to random sampling
- High values on the diagonal indicate stable assignments

### 4. Metrics Computation

#### CRI (Clustering Replicability Index)

For each cluster *c*:

```
CRI_c = mean(M[i, c] / sum(M[i, :]))
```

where the mean is over all samples *i* primarily assigned to cluster *c*.

**Interpretation:**
- CRI_c = 1.0: Perfect replicability for cluster *c*
- CRI_c ≈ 0.5: Random assignment (no replicability)
- CRI_c < 0.5: Anti-replicability (unusual, suggests issues)

Overall CRI = mean(CRI_c) across all clusters

#### WCRI (Weighted CRI)

Accounts for cluster size imbalance:

```
WCRI_c = CRI_c × (size_c / n)
```

where *size_c* is the number of samples primarily assigned to cluster *c*.

Mean WCRI = mean(WCRI_c) provides a balanced view across clusters.

#### TWCRI (Total Weighted CRI)

```
TWCRI = sum(WCRI_c) over all clusters
```

**Interpretation:**
- TWCRI close to 1.0: Excellent overall replicability
- TWCRI between 0.6-0.8: Good replicability
- TWCRI < 0.6: Poor replicability

TWCRI is particularly useful for selecting optimal *k* as it balances:
- Within-cluster replicability
- Cluster size balance
- Overall stability

#### Parmigiani Metrics (ARI and AMI)

In addition to CLAM-based metrics, ERICA also implements the partition comparison metrics from Parmigiani et al. (2023) "Cross-Study Replicability in Cluster Analysis" (Statistical Science, 38(2): 303-316, DOI: 10.1214/22-STS871).

These metrics compare two clusterings of the test set:
1. **predicted_labels**: Labels assigned by a model fitted on training data
2. **true_labels**: Labels from fitting a fresh model directly on test data

**ARI (Adjusted Rand Index)**

ARI measures agreement between two clusterings, adjusted for chance:

```
ARI = (RI - Expected_RI) / (max_RI - Expected_RI)
```

**Interpretation:**
- ARI = 1.0: Perfect agreement (identical clusterings up to permutation)
- ARI = 0.0: Random agreement (no better than chance)
- ARI < 0: Worse than random (rare)

**AMI (Adjusted Mutual Information)**

AMI measures mutual information between clusterings, adjusted for chance:

```
AMI = (MI - Expected_MI) / (mean(H(U), H(V)) - Expected_MI)
```

where H(U) and H(V) are the entropies of the two clusterings.

**Interpretation:**
- AMI = 1.0: Perfect mutual information
- AMI = 0.0: Independent clusterings

**Usage in ERICA:**

```python
from erica.metrics import compute_ari, compute_ami, aggregate_parmigiani_metrics

# Single iteration
ari = compute_ari(predicted_labels, true_labels)
ami = compute_ami(predicted_labels, true_labels)

# Multiple iterations
summary = aggregate_parmigiani_metrics(ari_scores, ami_scores)
print(f"Mean ARI: {summary['ARI_mean']:.3f} +/- {summary['ARI_std']:.3f}")
```

**When to use which metrics:**

| Metric | Best for | Computation |
|--------|----------|-------------|
| CRI/WCRI/TWCRI | Per-sample stability analysis | From CLAM matrix (post-aggregation) |
| ARI/AMI | Overall partition comparison | Per-iteration (then aggregated) |

---

## Selecting K*

ERICA provides multiple approaches for selecting the optimal number of clusters:

### Method 1: Algorithm 2 - K\* Selection (Recommended)

**Algorithm 2** implements a principled approach to selecting the optimal number of clusters by preferring larger K values when replicability is maintained or improved:

```
Algorithm 2: ERICA Cluster Number Selection

Input: M = {k → metric_value} for k = 2, 3, ..., K_max
Output: K* (optimal number of clusters)

1. Initialize K* = 2
2. For k = 3 to K_max:
    a. If M[k] is not NaN:
        i. If M[k] >= M[k-1], set K* = k
3. Return K*
```

**Key Properties:**
- Prefers larger K when metrics are non-decreasing
- Identifies the most granular stable clustering
- Handles NaN values (failed clustering) gracefully
- Different from simply selecting maximum metric value

**Example:**
```python
M = {2: 0.71, 3: 0.75, 4: 0.74, 5: NaN, 6: 0.78}
# K* = 6 (last non-decreasing value)
```

**Usage:**
```python
from erica import ERICA

erica = ERICA(data, k_range=[2, 3, 4, 5, 6])
results = erica.run()

# K* is automatically computed for all metrics
k_star_twcri = erica.get_k_star('TWCRI')
k_star_cri = erica.get_k_star('CRI')
k_star_wcri = erica.get_k_star('WCRI')
```

**Rationale**: This algorithm identifies the point where increasing cluster granularity no longer compromises replicability, which is often more scientifically meaningful than the global maximum.

### Method 2: Maximum TWCRI

Choose *k* that maximizes TWCRI:

```
k_optimal = argmax_k TWCRI(k)
```

**Rationale**: Balances cluster stability and size distribution.

### Method 3: Elbow Method on CRI

Plot CRI vs *k* and look for an "elbow point" where the rate of improvement decreases.

### Method 4: Stability Threshold

Choose the smallest *k* where CRI > threshold (e.g., 0.75):

```
k_optimal = min{k : CRI(k) > 0.75}
```

### Method 5: Domain Knowledge Integration

Combine ERICA metrics with:
- Biological/scientific interpretability
- Silhouette scores or other internal metrics
- External validation if available

---

## Comparison: K-Means vs Agglomerative

### K-Means in ERICA

**Procedure:**
1. Fit on training set *D_train^(b)*
2. Predict on test set *D_test^(b)*
3. Alignment using K-Means centroids

**Advantages:**
- Fast computation
- Well-defined cluster centers
- Easy alignment process

**Considerations:**
- Assumes spherical clusters
- Sensitive to initialization (mitigated by fixed random_state)

### Agglomerative Clustering in ERICA

**Procedure:**
1. Fit on test set *D_test^(b)* directly
2. Compute centroids from test data clustering
3. Alignment using computed centroids

**Advantages:**
- Can capture non-spherical clusters
- No initialization sensitivity
- Hierarchical structure (if needed)

**Considerations:**
- Slower for large datasets
- Memory intensive
- Linkage method affects results ('single' vs 'ward')

---

## Statistical Properties

### Sampling Distribution

Each iteration provides an independent estimate of cluster assignments under subsampling. The CLAM matrix aggregates these estimates, providing:

1. **Point Estimates**: Primary cluster assignments (argmax of each row)
2. **Uncertainty Estimates**: Distribution of assignments (row values)
3. **Stability Measures**: Concentration around primary assignments

### Convergence

As *B* increases:
- CRI estimates stabilize (lower variance)
- CLAM matrix entries approach their expectation
- Computational cost increases linearly

**Recommended B values:**
- Exploratory analysis: B = 100
- Standard analysis: B = 200
- High-precision analysis: B = 500

### Reproducibility

Setting `random_seed` ensures:
- Identical train/test splits
- Identical clustering results (with fixed cluster algorithm seeds)
- Reproducible metrics

This is implemented via:
```python
random.seed(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
# Plus threading controls for deterministic linear algebra
```

---

## Theoretical Foundations

### Relationship to Bootstrap

ERICA's iterative subsampling is similar to bootstrap resampling, but:
- **Bootstrap**: Samples with replacement
- **ERICA**: Samples without replacement (within each iteration)

This makes ERICA more conservative and reduces redundancy in replicability assessment.

### Connection to Ensemble Learning

ERICA can be viewed as an ensemble approach where:
- Each iteration is a "weak clusterer" trained on a subsample
- The CLAM matrix aggregates predictions
- Primary assignments are the ensemble decision

However, unlike traditional ensembles, ERICA's goal is assessment, not improved clustering.

### Stability Selection Parallel

ERICA shares concepts with stability selection in feature selection:
- Both use subsampling
- Both measure selection stability
- Both provide probabilistic interpretations

In ERICA, "features" are replaced by "cluster assignments" and "selection frequency" becomes "assignment frequency".

---

## Extensions and Variations

### Stratified Subsampling

For datasets with known groups, use stratified sampling:
- Maintain group proportions in train/test splits
- Useful for preventing group-imbalanced subsamples

### Weighted Samples

For samples with different importance:
- Use weighted sampling during subsampling
- Compute weighted CRI metrics

### Hierarchical ERICA

For hierarchical clustering analysis:
- Run ERICA at multiple levels
- Compare stability across hierarchy levels

### Multi-Method Consensus

Combine results from multiple clustering methods:
- Run ERICA with K-Means, Agglomerative, etc.
- Identify samples with consistent assignments
- Use consensus for higher-confidence clustering

---

## Computational Complexity

### Time Complexity

For *n* samples, *d* features, *k* clusters, *B* iterations:

**K-Means:**
- Single K-Means clustering: O(*ndk* × iterations)
- ERICA with K-Means: O(*B* × *ndk* × iterations)
- Alignment: O(*B* × *k*²*d*)
- CLAM generation: O(*B* × *n*)
- **Total**: O(*B* × *ndk* × iterations)

**Agglomerative:**
- Single Agglomerative: O(*n*²*log n*) for most linkages
- ERICA with Agglomerative: O(*B* × *n*²*log n*)
- **Total**: O(*B* × *n*²*log n*)

### Space Complexity

- Original data: O(*nd*)
- CLAM matrix: O(*nk*)
- Indices storage: O(*Bn*)
- Total: O(*nd* + *nk* + *Bn*)

With `optimize_io=True`, avoid storing subsample data, keeping space at O(*nd* + *nk*).

---

## Practical Considerations

### When to Use ERICA

**Good Use Cases:**
- Evaluating clustering stability before downstream analysis
- Comparing different clustering methods objectively
- Selecting optimal *k* with stability as a criterion
- Identifying outliers or ambiguous samples
- Publishing clustering results (for transparency)

**Not Ideal For:**
- Real-time clustering (too computationally intensive)
- Streaming data (requires complete dataset)
- When ground truth labels are available (use supervised metrics instead)

### Interpreting Low CRI

If CRI is low (< 0.6), consider:
1. **Too many clusters**: Try smaller *k*
2. **Poor separation**: Data may not have clear clusters
3. **High noise**: Preprocess or filter data
4. **Wrong method**: Try different clustering algorithms
5. **Insufficient features**: Feature engineering may help

### Handling Large Datasets

For datasets with *n* > 10,000:
1. **Reduce B**: Use B = 100-200 instead of 500
2. **Use K-Means**: Much faster than Agglomerative
3. **Subsample features**: If *d* is large, select informative features
4. **Parallel processing**: Run iterations in parallel (future feature)
5. **Incremental CLAM**: Update CLAM matrix incrementally

---

## References

### Related Methods

1. **Consensus Clustering**: Similar goal but different approach
2. **Stability Selection**: Similar bootstrap philosophy for feature selection
3. **Cross-validation for Clustering**: Related evaluation framework

### Recommended Reading

- Monti et al. (2003). "Consensus Clustering"
- Von Luxburg (2010). "Clustering Stability: An Overview"
- Ben-Hur & Guyon (2003). "Detecting stable clusters using principal component analysis"

---

## Citation

If you use ERICA in your research, please cite:

```bibtex
@software{erica2025,
  title = {ERICA: Evaluating Replicability via Iterative Clustering Assignments},
  author = {Sorooshyari, Siamak and Shirazi, Shawn},
  year = {2025},
  url = {https://github.com/sorooshyari/ERICA}
}
```

---

## Appendix: Mathematical Notation

| Symbol | Meaning |
|--------|---------|
| *n* | Number of samples |
| *d* | Number of features |
| *k* | Number of clusters |
| *B* | Number of iterations |
| *p* | Training proportion (default 0.8) |
| *D* | Complete dataset |
| *D_train^(b)* | Training set for iteration *b* |
| *D_test^(b)* | Test set for iteration *b* |
| *M* | CLAM matrix (*n* × *k*) |
| *M[i, j]* | Assignment count for sample *i* to cluster *j* |
| CRI_c | CRI for cluster *c* |
| WCRI_c | Weighted CRI for cluster *c* |
| TWCRI | Total weighted CRI |


