# Design: Flattened Method API + HDBSCAN Support

**Date:** 2026-03-29
**Status:** Approved
**Scope:** Refactor `method` parameter from string + `linkages` to flat list; add HDBSCAN as first auto-K method

---

## Problem

The current `method` parameter is confusing:
- `method='both'` implies two things but runs three (K-Means + Agglomerative Single + Agglomerative Ward)
- The `linkages` parameter is a separate config that silently controls how many agglomerative variants run
- Adding new clustering methods (HDBSCAN, spectral, birch, etc.) doesn't fit the current `'kmeans' | 'agglomerative' | 'both'` scheme

## Solution

Flatten `method` into a list of explicit method strings. Each string maps to exactly one clustering algorithm configuration. Remove the `linkages` parameter entirely.

---

## 1. Method Parameter API

The `method` parameter accepts a string or list of strings. The `linkages` parameter is removed.

```python
# Single method
ERICA(data, method='kmeans', k_range=[2,3,4,5])

# Multiple methods (explicit)
ERICA(data, method=['kmeans', 'agglomerative_ward', 'hdbscan'], k_range=[2,3,4,5])

# All available methods
ERICA(data, method='all', k_range=[2,3,4,5])

# Backwards compat — prints deprecation warning
ERICA(data, method='both')  # maps to ['kmeans', 'agglomerative_single', 'agglomerative_ward']
```

### Available Method Strings

| Name | sklearn Class | Type |
|---|---|---|
| `'kmeans'` | `KMeans` | K-based |
| `'agglomerative_ward'` | `AgglomerativeClustering(linkage='ward')` | K-based |
| `'agglomerative_single'` | `AgglomerativeClustering(linkage='single')` | K-based |
| `'agglomerative_complete'` | `AgglomerativeClustering(linkage='complete')` | K-based |
| `'agglomerative_average'` | `AgglomerativeClustering(linkage='average')` | K-based |
| `'hdbscan'` | `HDBSCAN` | Auto-K |

`'all'` expands to all of the above. Future methods (spectral, birch, gmm, etc.) add a new string with no API changes.

### Backwards Compatibility

- `method='both'` still works but emits a `DeprecationWarning` and maps to `['kmeans', 'agglomerative_single', 'agglomerative_ward']`
- `method='agglomerative'` still works, emits a `DeprecationWarning`, and maps to `['agglomerative_single', 'agglomerative_ward']` (matching old default linkages)
- `method='kmeans'` works as before (no change)
- The `linkages` parameter is removed from `__init__`. If passed, raise `TypeError` with a migration message.

---

## 2. Two Execution Paths

Methods are classified at init time into two groups:
- `self.k_based_methods`: Methods that take K as input (kmeans, agglomerative_*)
- `self.auto_k_methods`: Methods that discover K on their own (hdbscan)

### K-Based Methods (existing, enhanced)

```
for each K in k_range:
    for each k_based_method:
        for each iteration:
            fit on train → predict on test → predicted_labels
            fit fresh on test → true_labels
            record labels for CLAM matrix
            compute ARI/AMI for this iteration
        build CLAM matrix → compute CRI/WCRI/TWCRI
        aggregate ARI/AMI across iterations
```

Enhancement: K-based methods now also compute and report ARI/AMI per iteration (Parmigiani metrics), aggregated alongside CRI/WCRI/TWCRI.

### Auto-K Methods (new)

```
for each auto_k_method:
    for each iteration:
        fit on train → predict on test → predicted_labels (record discovered K)
        fit fresh on test → true_labels
        compute ARI/AMI for this iteration
    aggregate ARI/AMI across iterations
    find modal K (most common K discovered across iterations)
    build CLAM matrix from iterations where discovered K == modal K
    compute CRI/WCRI/TWCRI from that CLAM matrix
    report K distribution + K agreement rate
```

Iterations where HDBSCAN finds a different K than the mode are excluded from the CLAM matrix but still contribute to ARI/AMI. CRI/WCRI/TWCRI are computed on a consistent K; ARI/AMI capture the full picture including instability.

---

## 3. Results Structure

### K-Based Methods

Same structure as today, with Parmigiani metrics added:

```python
results['metrics'][3]['kmeans'] = {
    'CRI': 0.85,
    'CRI_per_cluster': [0.90, 0.80],
    'WCRI': 0.82,
    'WCRI_per_cluster': [0.45, 0.37],
    'TWCRI': 0.83,
    'cluster_sizes': [55, 45],
    'has_empty_clusters': False,
    'ARI_mean': 0.88,
    'ARI_std': 0.03,
    'AMI_mean': 0.85,
    'AMI_std': 0.04,
}
```

### Auto-K Methods

New top-level key `'auto_k'`:

```python
results['auto_k']['hdbscan'] = {
    'k_distribution': {3: 156, 4: 38, 2: 6},
    'modal_k': 3,
    'k_agreement_rate': 0.78,
    'ARI_mean': 0.82,
    'ARI_std': 0.05,
    'AMI_mean': 0.79,
    'AMI_std': 0.04,
    'metrics_at_modal_k': {
        'CRI': 0.85,
        'CRI_per_cluster': [0.90, 0.82, 0.78],
        'WCRI': 0.80,
        'WCRI_per_cluster': [0.32, 0.28, 0.20],
        'TWCRI': 0.83,
        'cluster_sizes': [42, 35, 25],
        'has_empty_clusters': False,
    },
    'clam_matrix': <np.ndarray>,  # CLAM for modal K iterations only
    'n_iterations_used': 156,     # how many iterations contributed to CLAM
}
```

---

## 4. HDBSCAN Implementation Details

### Train/Test Workflow

For each iteration:
1. Fit `sklearn.cluster.HDBSCAN` on train set
2. Predict test labels via `approximate_predict()` (if available) or nearest-centroid assignment from train labels
3. Fit fresh `HDBSCAN` on test set directly → true_labels
4. Noise points (label = -1): assigned to nearest non-noise cluster via closest centroid
5. Record discovered K (number of non-noise clusters) for both train-fitted and test-fitted

### HDBSCAN Parameters

Expose key HDBSCAN parameters via an optional `hdbscan_params` dict in `__init__`:

```python
ERICA(
    data=data,
    method=['kmeans', 'hdbscan'],
    k_range=[2, 3, 4, 5],
    hdbscan_params={'min_cluster_size': 15, 'min_samples': 5}
)
```

Defaults: `min_cluster_size=15`, `min_samples=None` (sklearn default). These are the only two parameters most users will need to tune.

### Noise Handling

HDBSCAN labels some points as noise (-1). For CLAM matrix and metric computation:
- Noise points are assigned to the nearest non-noise cluster (by centroid distance)
- This ensures every sample has a cluster assignment for CRI computation
- The number of noise points per iteration is tracked and reported

---

## 5. File Changes

### `erica/core.py`
- `__init__`: Replace `method: str` + `linkages: list` with `method: Union[str, List[str]]`. Normalize to list. Deprecation warnings. Expand `'all'`. Classify into `self.k_based_methods` / `self.auto_k_methods`. Add optional `hdbscan_params`.
- `run()`: Two loops — K-based (refactored from existing) and auto-K (new). Both compute ARI/AMI per iteration.
- Remove `linkages` from `__init__`, `_get_config_dict()`, `__repr__()`.
- New method: `get_auto_k_results(method='hdbscan')`.
- Update `get_results()` to include `'auto_k'` key.

### `erica/clustering.py`
- New function: `hdbscan_clustering()` — fits HDBSCAN, handles noise, returns labels + discovered K.
- Refactor `kmeans_clustering()` and `agglomerative_clustering()` to also return per-iteration predicted/true label pairs for ARI/AMI.
- New internal helper: `_assign_noise_to_nearest(labels, data, centroids)`.

### `erica/metrics.py`
- No changes needed (Parmigiani metrics already implemented).

### `erica/utils.py`
- Update validation: accept list or string, validate against known method names.
- Remove `linkages` validation.
- Add `VALID_METHODS` and `AUTO_K_METHODS` constants.

### `erica/__init__.py`
- Update exports and docstring examples.

### `tests/test_erica.py`
- Update existing tests: `method='both'` → list form.
- New tests: list input, `'all'`, `'hdbscan'`, deprecation warning, auto-K results structure, noise handling.

### `examples/`
- Update all existing examples (comments clarifying method values).
- New: `06_hdbscan.py` demonstrating auto-K workflow.

### `docs/`
- Update: API_REFERENCE.md, METRICS_GUIDE.md, GETTING_STARTED.md, README.md, METHODOLOGY.md, docs/README.md.

### `setup.py`
- No dependency changes needed. HDBSCAN is in sklearn since 1.3.0, which is already our minimum.

---

## 6. Migration Guide

For users upgrading:

```python
# Old API
ERICA(data, method='both', linkages=['single', 'ward'])

# New API (equivalent)
ERICA(data, method=['kmeans', 'agglomerative_single', 'agglomerative_ward'])

# Old API still works with deprecation warning
ERICA(data, method='both')  # warns, maps to the above

# Old linkages parameter raises helpful error
ERICA(data, method='kmeans', linkages=['ward'])
# TypeError: "linkages parameter has been removed. Use method=['agglomerative_ward'] instead."
```
