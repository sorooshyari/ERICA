# ERICA Metrics Guide

This document explains the clustering replicability metrics implemented in ERICA: the **CLAM-based metrics** (CRI, WCRI, TWCRI) and the **Parmigiani metrics** (ARI, AMI). If you've ever wondered whether your clusters are real or just statistical hallucinations, these metrics provide quantitative answers.

## Overview

### ERICA Metrics (CLAM-based)

| Metric | Full Name | Relationship | Purpose |
|--------|-----------|--------------|---------|
| **CRI** | Clustering Replicability Index | **Core ERICA metric** | Measures per-cluster assignment consistency |
| **WCRI** | Weighted CRI | Derived from CRI | Size-weighted CRI for imbalanced clusters |
| **TWCRI** | Total Weighted CRI | Sum of WCRI | Aggregate score for K* selection |

### Parmigiani Metrics (Partition Comparison)

| Metric | Full Name | Range | Purpose |
|--------|-----------|-------|---------|
| **ARI** | Adjusted Rand Index | [-1, 1] | Partition similarity adjusted for chance |
| **AMI** | Adjusted Mutual Information | [0, 1] | Information-theoretic partition comparison |

All metrics have **higher values indicating better replicability** (except ARI can be negative for worse-than-random).

---

## CRI: The Core ERICA Metric

**CRI (Clustering Replicability Index)** is the fundamental metric of ERICA. It quantifies how consistently samples are assigned to their primary cluster across Monte Carlo iterations.

### Definition

For each cluster *c*, CRI measures the average proportion of times that samples assigned to cluster *c* were actually placed in cluster *c* across all iterations:

```
CRI_c = mean( M[i,c] / sum(M[i,:]) )  for all samples i in cluster c
```

where:
- *M* is the CLAM matrix (samples × clusters)
- *M[i,c]* = number of times sample *i* was assigned to cluster *c*
- The mean is computed over all samples primarily assigned to cluster *c*

**Overall CRI** = mean of CRI_c across all clusters

### Interpretation

| CRI Value | Interpretation |
|-----------|----------------|
| 1.0 | Perfect replicability: every sample always assigned to the same cluster |
| 0.8–1.0 | High replicability |
| 0.6–0.8 | Moderate replicability |
| 0.5 | Random assignment (no better than chance) |
| < 0.5 | Poor replicability (unusual—suggests methodological issues) |

### Per-Cluster CRI

CRI is computed for each cluster individually, allowing you to identify which clusters are stable and which are problematic:

```python
from erica import ERICA

erica = ERICA(data=data, k_range=[2, 3, 4, 5])
results = erica.run()

# Get CRI for K=3
metrics = erica.get_metrics(k=3)
print(f"Overall CRI: {metrics['kmeans']['CRI']:.3f}")
print(f"Per-cluster CRI: {metrics['kmeans']['CRI_per_cluster']}")
# Example output: [0.92, 0.87, 0.65]
# Cluster 3 (0.65) is less stable than clusters 1 and 2
```

### When CRI Is Most Useful

- **Diagnosing clustering quality**: Low CRI clusters may be artifacts
- **Comparing cluster stability**: Which clusters are well-defined vs. fuzzy?
- **Scientific interpretation**: Is this subtype real or noise?

---

## WCRI: Weighted CRI

**WCRI** weights CRI by cluster size, providing a size-adjusted view of replicability.

### Definition

For each cluster *c*:

```
WCRI_c = CRI_c × (size_c / n)
```

where:
- *size_c* = number of samples primarily assigned to cluster *c*
- *n* = total number of samples

### Why Weight by Size?

Consider a dataset with two clusters:
- Cluster A: 90 samples, CRI = 0.95
- Cluster B: 10 samples, CRI = 0.50

Unweighted mean CRI = (0.95 + 0.50) / 2 = 0.725

But Cluster A represents 90% of your data. WCRI ensures that the stability of the majority is appropriately reflected.

### When to Use WCRI

- When cluster sizes are highly imbalanced
- When larger clusters are more scientifically relevant
- For a "population-weighted" view of replicability

---

## TWCRI: Total Weighted CRI

**TWCRI** is the sum of all WCRI values, providing a single aggregate score for the entire clustering solution.

### Definition

```
TWCRI = sum(WCRI_c) for all clusters c
```

### Why TWCRI for K* Selection?

TWCRI is the default metric for Algorithm 2 (K* selection) because it:

1. **Incorporates CRI**: Built on the core ERICA metric
2. **Accounts for cluster size**: Via the weighting in WCRI
3. **Provides a single score**: Easy to compare across K values
4. **Penalizes empty clusters**: Via NaN handling in Algorithm 2

### Interpretation

| TWCRI Value | Interpretation |
|-------------|----------------|
| > 0.8 | High overall replicability |
| 0.6–0.8 | Moderate replicability |
| < 0.6 | Low replicability (reconsider K or clustering method) |

### Example: K* Selection with TWCRI

```python
# Get recommended K* based on TWCRI
k_star = erica.get_k_star('TWCRI')
print(f"Recommended K: {k_star['kmeans']}")

# Compare TWCRI across K values
for k in [2, 3, 4, 5]:
    metrics = erica.get_metrics(k=k)
    print(f"K={k}: TWCRI = {metrics['kmeans']['TWCRI']:.3f}")
```

---

## Metric Hierarchy

```
CRI (per cluster)
 │
 ├── Overall CRI = mean(CRI_c)     ← Simple average across clusters
 │
 └── WCRI_c = CRI_c × (size_c/n)   ← Size-weighted CRI
      │
      └── TWCRI = sum(WCRI_c)      ← Aggregate score for K* selection
```

---

## Choosing Which Metric to Report

| Use Case | Recommended Metric |
|----------|-------------------|
| **Primary result** | **CRI** (the core ERICA metric) |
| K* selection | TWCRI (used by Algorithm 2) |
| Identifying weak clusters | CRI per cluster |
| Imbalanced cluster analysis | WCRI |
| Comparing clustering methods | CRI or TWCRI |

For publications, we recommend reporting **CRI** as the primary replicability metric, with TWCRI used for K* selection decisions.

---

## Empty Clusters and NaN Values

When a K value produces empty clusters (clusters with no assigned samples), ERICA marks all metrics as `NaN`:

```
K=8: CRI = NaN (DISQUALIFIED - empty cluster detected)
```

### Why Disqualify?

Empty clusters indicate that:
1. K is too large for the data structure
2. The clustering algorithm couldn't find K distinct groups
3. The K value is inappropriate for this dataset

Per Algorithm 2, NaN values are automatically skipped during K* selection.

### Tracking Disqualified K Values

```python
disqualified = erica.get_disqualified_k()
print(f"Disqualified K values: {disqualified}")
# Output: {'kmeans': [8], 'agglomerative_ward': [7, 8]}
```

---

## Mathematical Formulas

### CRI (the core metric)

$$\text{CRI}_c = \frac{1}{|S_c|} \sum_{i \in S_c} \frac{M_{i,c}}{\sum_{j=1}^{k} M_{i,j}}$$

where $S_c$ is the set of samples primarily assigned to cluster $c$.

### WCRI (derived from CRI)

$$\text{WCRI}_c = \text{CRI}_c \times \frac{|S_c|}{n}$$

### TWCRI (aggregate of WCRI)

$$\text{TWCRI} = \sum_{c=1}^{k} \text{WCRI}_c$$

---

## Parmigiani Metrics (ARI and AMI)

In addition to CLAM-based metrics, ERICA implements the partition comparison metrics from Parmigiani et al. (2023) "Cross-Study Replicability in Cluster Analysis" (Statistical Science, 38(2): 303-316).

### Overview

| Metric | Full Name | Range | Perfect Score |
|--------|-----------|-------|---------------|
| **ARI** | Adjusted Rand Index | [-1, 1] | 1.0 |
| **AMI** | Adjusted Mutual Information | [0, 1] | 1.0 |

### How They Work

The Parmigiani methodology (Algorithm 1) compares two clusterings of the test set:

1. **predicted_labels**: Cluster assignments from a model trained on the training set, applied to test samples
2. **true_labels**: Cluster assignments from a fresh model fitted directly on test samples

```
Train Set ──► Fit Model ──► Predict on Test ──► predicted_labels
                                                      │
Test Set  ──► Fit Fresh Model ──────────────► true_labels
                                                      │
                                              Compare with ARI/AMI
```

### Interpretation

| Value | ARI Interpretation | AMI Interpretation |
|-------|-------------------|-------------------|
| 1.0 | Perfect agreement | Perfect mutual information |
| 0.8+ | High replicability | High replicability |
| 0.5-0.8 | Moderate agreement | Moderate agreement |
| ~0 | Random (no better than chance) | Independent clusterings |
| <0 | Worse than random (ARI only) | N/A |

### Usage Example

```python
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from erica.metrics import compute_ari, compute_ami, aggregate_parmigiani_metrics

# Single iteration
train_data, test_data = train_test_split(data, train_size=0.8)

# Fit on train, predict on test
kmeans_train = KMeans(n_clusters=3).fit(train_data)
predicted_labels = kmeans_train.predict(test_data)

# Fit fresh on test
true_labels = KMeans(n_clusters=3).fit_predict(test_data)

# Compute metrics
ari = compute_ari(predicted_labels, true_labels)
ami = compute_ami(predicted_labels, true_labels)
print(f"ARI: {ari:.3f}, AMI: {ami:.3f}")

# Multiple iterations
ari_scores, ami_scores = [], []
for i in range(100):
    # ... compute for each iteration ...
    ari_scores.append(ari)
    ami_scores.append(ami)

summary = aggregate_parmigiani_metrics(ari_scores, ami_scores)
print(f"Mean ARI: {summary['ARI_mean']:.3f} +/- {summary['ARI_std']:.3f}")
```

### ERICA vs Parmigiani Metrics

| Aspect | ERICA (CRI/WCRI/TWCRI) | Parmigiani (ARI/AMI) |
|--------|------------------------|----------------------|
| **Based on** | CLAM matrix (aggregated assignments) | Pairwise partition comparison |
| **Computed** | Once after all iterations | Per iteration, then aggregated |
| **Measures** | Per-sample assignment consistency | Overall partition similarity |
| **Best for** | Identifying unstable samples/clusters | Overall replicability score |

Both approaches are complementary. Use CRI to diagnose which clusters are problematic; use ARI/AMI for a quick overall replicability assessment.

---

## Code Reference

The metrics are implemented in `erica/metrics.py`:

### ERICA Metrics (CLAM-based)

| Function | Description |
|----------|-------------|
| `compute_cri()` | Compute CRI for each cluster (core metric) |
| `compute_wcri()` | Compute WCRI (weighted CRI) |
| `compute_twcri()` | Compute TWCRI (total weighted CRI) |
| `compute_metrics_for_clam()` | Compute all metrics from a CLAM matrix |
| `select_optimal_k()` | Select K* using Algorithm 2 |

### Parmigiani Metrics (Partition Comparison)

| Function | Description |
|----------|-------------|
| `compute_ari()` | Adjusted Rand Index between two clusterings |
| `compute_ami()` | Adjusted Mutual Information between two clusterings |
| `compute_parmigiani_metrics()` | Compute both ARI and AMI |
| `aggregate_parmigiani_metrics()` | Aggregate scores across iterations |

---

## Further Reading

- [Methodology](METHODOLOGY.md) — Full mathematical derivation
- [API Reference](API_REFERENCE.md) — Function documentation
- [Getting Started](GETTING_STARTED.md) — Tutorial with metric examples
