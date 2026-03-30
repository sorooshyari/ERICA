# Flattened Method API + HDBSCAN Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Flatten the `method` parameter from `str` + `linkages` to a list of explicit method strings, and add HDBSCAN as the first auto-K clustering method.

**Architecture:** The change touches 4 source files. `utils.py` gets method constants and a normalization function. `clustering.py` gets the `hdbscan_clustering` function and a noise-handling helper, plus existing clustering functions get enhanced to return per-iteration label pairs for ARI/AMI. `core.py.__init__` accepts the new `method` format and classifies methods; `core.py.run()` splits into K-based and auto-K execution paths.

**Tech Stack:** Python, numpy, pandas, scikit-learn (>=1.3.0 for HDBSCAN), pytest

**Spec:** `docs/superpowers/specs/2026-03-29-flatten-method-api-hdbscan-design.md`

---

### Task 1: Method Constants and Normalization (utils.py)

**Files:**
- Modify: `erica/utils.py`
- Test: `tests/test_erica.py`

- [ ] **Step 1: Write failing tests for method constants and normalize_method**

Add to `tests/test_erica.py`:

```python
import warnings
from erica.utils import (
    VALID_METHODS, K_BASED_METHODS, AUTO_K_METHODS,
    normalize_method,
)


def test_method_constants():
    """Test that method constants are defined and consistent."""
    assert 'kmeans' in VALID_METHODS
    assert 'agglomerative_ward' in VALID_METHODS
    assert 'agglomerative_single' in VALID_METHODS
    assert 'agglomerative_complete' in VALID_METHODS
    assert 'agglomerative_average' in VALID_METHODS
    assert 'hdbscan' in VALID_METHODS
    assert set(VALID_METHODS) == set(K_BASED_METHODS) | set(AUTO_K_METHODS)
    assert 'hdbscan' in AUTO_K_METHODS
    assert 'hdbscan' not in K_BASED_METHODS


def test_normalize_method_single_string():
    """Test normalize_method with a single valid method string."""
    assert normalize_method('kmeans') == ['kmeans']
    assert normalize_method('hdbscan') == ['hdbscan']
    assert normalize_method('agglomerative_ward') == ['agglomerative_ward']


def test_normalize_method_list():
    """Test normalize_method with a list of method strings."""
    result = normalize_method(['kmeans', 'hdbscan'])
    assert result == ['kmeans', 'hdbscan']


def test_normalize_method_all():
    """Test normalize_method with 'all' expands to all methods."""
    result = normalize_method('all')
    assert set(result) == set(VALID_METHODS)


def test_normalize_method_both_deprecated():
    """Test that method='both' emits DeprecationWarning and maps correctly."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = normalize_method('both')
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert 'deprecated' in str(w[0].message).lower()
    assert result == ['kmeans', 'agglomerative_single', 'agglomerative_ward']


def test_normalize_method_agglomerative_deprecated():
    """Test that method='agglomerative' emits DeprecationWarning."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = normalize_method('agglomerative')
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
    assert result == ['agglomerative_single', 'agglomerative_ward']


def test_normalize_method_invalid_string():
    """Test that invalid method string raises ValueError."""
    with pytest.raises(ValueError, match="Unknown method"):
        normalize_method('invalid_method')


def test_normalize_method_invalid_in_list():
    """Test that invalid method in list raises ValueError."""
    with pytest.raises(ValueError, match="Unknown method"):
        normalize_method(['kmeans', 'invalid_method'])


def test_normalize_method_wrong_type():
    """Test that non-string non-list raises TypeError."""
    with pytest.raises(TypeError):
        normalize_method(123)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_erica.py::test_method_constants -v`
Expected: FAIL with ImportError (VALID_METHODS not yet defined)

- [ ] **Step 3: Implement constants and normalize_method in utils.py**

Add at the top of `erica/utils.py`, after the existing imports:

```python
import warnings
from typing import Union

# Valid method strings for the flattened method API
VALID_METHODS = [
    'kmeans',
    'agglomerative_ward',
    'agglomerative_single',
    'agglomerative_complete',
    'agglomerative_average',
    'hdbscan',
]

K_BASED_METHODS = [
    'kmeans',
    'agglomerative_ward',
    'agglomerative_single',
    'agglomerative_complete',
    'agglomerative_average',
]

AUTO_K_METHODS = [
    'hdbscan',
]

_DEPRECATED_METHOD_ALIASES = {
    'both': ['kmeans', 'agglomerative_single', 'agglomerative_ward'],
    'agglomerative': ['agglomerative_single', 'agglomerative_ward'],
}


def normalize_method(method: Union[str, list]) -> list:
    """Normalize method parameter to a list of valid method strings.

    Parameters
    ----------
    method : str or list of str
        Method specification. Can be a single method string, a list of method
        strings, 'all' (expands to all methods), or a deprecated alias ('both',
        'agglomerative') which emits a DeprecationWarning.

    Returns
    -------
    list of str
        Normalized list of method strings.

    Raises
    ------
    ValueError
        If any method string is not recognized.
    TypeError
        If method is not a string or list.
    """
    if isinstance(method, str):
        if method == 'all':
            return list(VALID_METHODS)
        if method in _DEPRECATED_METHOD_ALIASES:
            warnings.warn(
                f"method='{method}' is deprecated. "
                f"Use method={_DEPRECATED_METHOD_ALIASES[method]} instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            return list(_DEPRECATED_METHOD_ALIASES[method])
        if method in VALID_METHODS:
            return [method]
        raise ValueError(
            f"Unknown method '{method}'. Valid methods: {VALID_METHODS}"
        )

    if isinstance(method, list):
        for m in method:
            if m not in VALID_METHODS:
                raise ValueError(
                    f"Unknown method '{m}'. Valid methods: {VALID_METHODS}"
                )
        return list(method)

    raise TypeError(
        f"method must be a string or list of strings, got {type(method).__name__}"
    )
```

Also update `validate_config` to use the new constants. Replace the method and linkages validation blocks:

```python
def validate_config(config: Dict[str, Any]) -> None:
    # ... existing k_range, n_iterations, train_percent validation ...

    # Validate method (accepts new flattened format or deprecated aliases)
    if 'method' in config:
        method = config['method']
        if isinstance(method, str):
            valid = list(VALID_METHODS) + list(_DEPRECATED_METHOD_ALIASES.keys())
            if method not in valid and method != 'all':
                raise ValueError(f"method must be one of {valid} or 'all'")
        elif isinstance(method, list):
            for m in method:
                if m not in VALID_METHODS:
                    raise ValueError(
                        f"Invalid method '{m}'. Must be one of {VALID_METHODS}"
                    )

    # linkages parameter is removed
    if 'linkages' in config:
        raise TypeError(
            "The 'linkages' parameter has been removed. "
            "Use method=['agglomerative_ward'] instead. "
            "See the migration guide for details."
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_erica.py -k "test_method_constants or test_normalize_method" -v`
Expected: All 9 tests PASS

- [ ] **Step 5: Commit**

```bash
git add erica/utils.py tests/test_erica.py
git commit -m "feat: add method constants and normalize_method to utils.py"
```

---

### Task 2: Noise Handling Helper (clustering.py)

**Files:**
- Modify: `erica/clustering.py`
- Test: `tests/test_erica.py`

- [ ] **Step 1: Write failing test for _assign_noise_to_nearest**

Add to `tests/test_erica.py`:

```python
from erica.clustering import _assign_noise_to_nearest


def test_assign_noise_to_nearest_no_noise():
    """Test that labels without noise are returned unchanged."""
    labels = np.array([0, 1, 2, 0, 1])
    data = np.random.rand(5, 3)
    centroids = np.random.rand(3, 3)
    result = _assign_noise_to_nearest(labels, data, centroids)
    np.testing.assert_array_equal(result, labels)


def test_assign_noise_to_nearest_with_noise():
    """Test that noise points (-1) are assigned to nearest centroid."""
    # Two clusters at known positions
    centroids = np.array([[0.0, 0.0], [10.0, 10.0]])
    # Point near cluster 0, point near cluster 1, noise point near cluster 0
    data = np.array([[0.1, 0.1], [9.9, 9.9], [0.2, 0.2]])
    labels = np.array([0, 1, -1])

    result = _assign_noise_to_nearest(labels, data, centroids)

    assert result[0] == 0
    assert result[1] == 1
    assert result[2] == 0  # noise assigned to nearest (cluster 0)


def test_assign_noise_to_nearest_all_noise():
    """Test when all points are noise."""
    centroids = np.array([[0.0, 0.0], [10.0, 10.0]])
    data = np.array([[0.1, 0.1], [9.9, 9.9]])
    labels = np.array([-1, -1])

    result = _assign_noise_to_nearest(labels, data, centroids)

    assert result[0] == 0
    assert result[1] == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_erica.py::test_assign_noise_to_nearest_no_noise -v`
Expected: FAIL with ImportError

- [ ] **Step 3: Implement _assign_noise_to_nearest in clustering.py**

Add before the `_align_cluster_identities` function in `erica/clustering.py`:

```python
def _assign_noise_to_nearest(
    labels: np.ndarray,
    data: np.ndarray,
    centroids: np.ndarray
) -> np.ndarray:
    """Assign noise points (label == -1) to the nearest non-noise cluster.

    Used by HDBSCAN to ensure every sample has a cluster assignment for
    CLAM matrix computation.

    Parameters
    ----------
    labels : np.ndarray
        Cluster labels where -1 indicates noise. Shape: (n_samples,)
    data : np.ndarray
        Data points. Shape: (n_samples, n_features)
    centroids : np.ndarray
        Cluster centroids. Shape: (n_clusters, n_features)

    Returns
    -------
    np.ndarray
        Labels with noise points reassigned to nearest cluster.
    """
    noise_mask = labels == -1
    if not np.any(noise_mask):
        return labels.copy()

    result = labels.copy()
    noise_points = data[noise_mask]

    # Compute distances from noise points to each centroid
    distances = np.linalg.norm(
        noise_points[:, np.newaxis, :] - centroids[np.newaxis, :, :],
        axis=2
    )
    nearest = np.argmin(distances, axis=1)
    result[noise_mask] = nearest

    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_erica.py -k "test_assign_noise" -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add erica/clustering.py tests/test_erica.py
git commit -m "feat: add _assign_noise_to_nearest helper for HDBSCAN noise handling"
```

---

### Task 3: HDBSCAN Clustering Function (clustering.py)

**Files:**
- Modify: `erica/clustering.py`
- Test: `tests/test_erica.py`

- [ ] **Step 1: Write failing tests for hdbscan_clustering**

Add to `tests/test_erica.py`:

```python
from erica.clustering import hdbscan_clustering


def test_hdbscan_clustering_basic():
    """Test basic HDBSCAN clustering returns expected structure."""
    from sklearn.datasets import make_blobs

    data, _ = make_blobs(n_samples=100, n_features=5, centers=3, random_state=42)

    # Create subsampling indices
    with tempfile.TemporaryDirectory() as tmpdir:
        indices_folder = os.path.join(tmpdir, 'indices')
        os.makedirs(indices_folder)

        n_iterations = 10
        n_samples = len(data)
        train_size = int(n_samples * 0.8)

        all_train = []
        all_test = []
        for _ in range(n_iterations):
            perm = np.random.permutation(n_samples)
            all_train.append(perm[:train_size])
            all_test.append(perm[train_size:])

        np.save(os.path.join(indices_folder, 'all_train_indices.npy'),
                np.array(all_train, dtype=object))
        np.save(os.path.join(indices_folder, 'all_test_indices.npy'),
                np.array(all_test, dtype=object))

        result = hdbscan_clustering(
            samples_array=data,
            n_iterations=n_iterations,
            indices_folder=indices_folder,
            output_dir=tmpdir,
            hdbscan_params={'min_cluster_size': 5},
            verbose=False,
        )

    # Check result structure
    assert 'k_distribution' in result
    assert 'modal_k' in result
    assert 'k_agreement_rate' in result
    assert 'clam_matrix' in result
    assert 'n_iterations_used' in result
    assert 'iteration_labels' in result
    assert 'predicted' in result['iteration_labels']
    assert 'true' in result['iteration_labels']
    assert 'noise_counts' in result

    # modal_k should be a positive integer
    assert result['modal_k'] >= 1

    # k_agreement_rate should be between 0 and 1
    assert 0 <= result['k_agreement_rate'] <= 1

    # n_iterations_used should be <= n_iterations
    assert result['n_iterations_used'] <= n_iterations

    # CLAM matrix should have shape (n_samples, modal_k)
    assert result['clam_matrix'].shape[0] == n_samples
    assert result['clam_matrix'].shape[1] == result['modal_k']


def test_hdbscan_clustering_noise_handling():
    """Test that HDBSCAN handles noise points correctly."""
    from sklearn.datasets import make_blobs

    data, _ = make_blobs(n_samples=80, n_features=3, centers=2,
                         cluster_std=0.5, random_state=42)

    with tempfile.TemporaryDirectory() as tmpdir:
        indices_folder = os.path.join(tmpdir, 'indices')
        os.makedirs(indices_folder)

        n_iterations = 5
        n_samples = len(data)
        train_size = int(n_samples * 0.8)

        all_train = []
        all_test = []
        for _ in range(n_iterations):
            perm = np.random.permutation(n_samples)
            all_train.append(perm[:train_size])
            all_test.append(perm[train_size:])

        np.save(os.path.join(indices_folder, 'all_train_indices.npy'),
                np.array(all_train, dtype=object))
        np.save(os.path.join(indices_folder, 'all_test_indices.npy'),
                np.array(all_test, dtype=object))

        result = hdbscan_clustering(
            samples_array=data,
            n_iterations=n_iterations,
            indices_folder=indices_folder,
            output_dir=tmpdir,
            hdbscan_params={'min_cluster_size': 5},
            verbose=False,
        )

    # noise_counts should be a list of non-negative integers
    assert all(n >= 0 for n in result['noise_counts'])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_erica.py::test_hdbscan_clustering_basic -v`
Expected: FAIL with ImportError

- [ ] **Step 3: Implement hdbscan_clustering in clustering.py**

Add the HDBSCAN import at the top of `erica/clustering.py`:

```python
from sklearn.cluster import KMeans, AgglomerativeClustering, HDBSCAN
```

Add the function after `agglomerative_clustering`:

```python
def hdbscan_clustering(
    samples_array: np.ndarray,
    n_iterations: int,
    indices_folder: str,
    output_dir: str,
    hdbscan_params: Optional[Dict] = None,
    verbose: bool = False
) -> Dict:
    """Perform HDBSCAN auto-K clustering with ERICA analysis.

    HDBSCAN discovers the number of clusters automatically. This function
    runs HDBSCAN on each train/test split, tracks the discovered K values,
    computes a CLAM matrix from iterations where K matches the mode, and
    returns ARI/AMI label pairs for all iterations.

    Parameters
    ----------
    samples_array : np.ndarray
        Input data with shape (n_samples, n_features)
    n_iterations : int
        Number of iterations
    indices_folder : str
        Path to folder with train/test indices
    output_dir : str
        Directory for saving outputs
    hdbscan_params : dict, optional
        Parameters for HDBSCAN (min_cluster_size, min_samples).
        Defaults: min_cluster_size=15, min_samples=None (sklearn default)
    verbose : bool, optional
        Whether to print progress, default False

    Returns
    -------
    dict
        Results dictionary containing:
        - 'k_distribution': dict mapping discovered K to count
        - 'modal_k': most common K across iterations
        - 'k_agreement_rate': fraction of iterations with modal K
        - 'clam_matrix': CLAM matrix for modal K iterations (n_samples, modal_k)
        - 'n_iterations_used': number of iterations contributing to CLAM
        - 'iteration_labels': {'predicted': [...], 'true': [...]} per iteration
        - 'noise_counts': list of noise point counts per iteration
        - 'output_folder': path to output directory
    """
    from collections import Counter

    if verbose:
        print("Starting HDBSCAN clustering...")

    n_samples = samples_array.shape[0]
    params = {'min_cluster_size': 15}
    if hdbscan_params:
        params.update(hdbscan_params)

    # Load indices
    train_indices = np.load(
        os.path.join(indices_folder, 'all_train_indices.npy'),
        allow_pickle=True
    )
    test_indices = np.load(
        os.path.join(indices_folder, 'all_test_indices.npy'),
        allow_pickle=True
    )

    # Per-iteration storage
    discovered_ks = []
    all_predicted_labels = []
    all_true_labels = []
    all_reassigned_labels = []  # after noise reassignment, for CLAM
    noise_counts = []

    for iter_idx in range(n_iterations):
        if verbose and iter_idx % 50 == 0:
            print(f"  Iteration {iter_idx + 1}/{n_iterations}")

        train_data, test_data = load_iteration_data(
            iter_idx, samples_array, indices_folder
        )

        # Fit HDBSCAN on train set
        hdb_train = HDBSCAN(**params)
        train_labels = hdb_train.fit_predict(train_data)

        # Compute train centroids (excluding noise)
        unique_train = set(train_labels)
        unique_train.discard(-1)
        if len(unique_train) == 0:
            # All noise on train, skip iteration
            discovered_ks.append(0)
            all_predicted_labels.append(np.array([]))
            all_true_labels.append(np.array([]))
            noise_counts.append(len(test_data))
            continue

        n_clusters_train = max(unique_train) + 1
        train_centroids = np.zeros((n_clusters_train, train_data.shape[1]))
        for c in range(n_clusters_train):
            mask = train_labels == c
            if mask.any():
                train_centroids[c] = train_data[mask].mean(axis=0)

        # Predict test labels via nearest train centroid
        distances = np.linalg.norm(
            test_data[:, np.newaxis, :] - train_centroids[np.newaxis, :, :],
            axis=2
        )
        predicted_labels = np.argmin(distances, axis=1)

        # Fit fresh HDBSCAN on test set
        hdb_test = HDBSCAN(**params)
        test_labels_raw = hdb_test.fit_predict(test_data)

        # Count noise before reassignment
        n_noise = int(np.sum(test_labels_raw == -1))
        noise_counts.append(n_noise)

        # Compute test centroids for noise reassignment
        unique_test = set(test_labels_raw)
        unique_test.discard(-1)
        if len(unique_test) == 0:
            # All test points are noise, use predicted_labels as fallback
            true_labels = predicted_labels.copy()
            discovered_k = n_clusters_train
        else:
            n_clusters_test = max(unique_test) + 1
            test_centroids = np.zeros((n_clusters_test, test_data.shape[1]))
            for c in range(n_clusters_test):
                mask = test_labels_raw == c
                if mask.any():
                    test_centroids[c] = test_data[mask].mean(axis=0)

            true_labels = _assign_noise_to_nearest(
                test_labels_raw, test_data, test_centroids
            )
            discovered_k = n_clusters_test

        discovered_ks.append(discovered_k)
        all_predicted_labels.append(predicted_labels)
        all_true_labels.append(true_labels)
        all_reassigned_labels.append(true_labels)

    # Find modal K
    k_counts = Counter(k for k in discovered_ks if k > 0)
    if not k_counts:
        modal_k = 0
        k_agreement_rate = 0.0
        clam_matrix = np.zeros((n_samples, 1))
        n_used = 0
        k_distribution = {}
    else:
        modal_k = k_counts.most_common(1)[0][0]
        n_modal = k_counts[modal_k]
        k_agreement_rate = n_modal / n_iterations
        k_distribution = dict(k_counts)

        # Build CLAM matrix from iterations where discovered K == modal K
        clam_matrix = np.zeros((n_samples, modal_k))
        n_used = 0
        label_idx = 0  # tracks position in all_reassigned_labels
        for iter_idx in range(n_iterations):
            if discovered_ks[iter_idx] == 0:
                # Skipped iteration (all noise on train)
                continue

            labels = all_reassigned_labels[label_idx]
            label_idx += 1

            if discovered_ks[iter_idx] != modal_k:
                continue
            if len(labels) == 0:
                continue
            if int(labels.max()) >= modal_k:
                continue

            iter_test_idx = test_indices[iter_idx].astype(int)
            for i, sample_idx in enumerate(iter_test_idx):
                if i < len(labels):
                    cluster_id = int(labels[i])
                    if 0 <= cluster_id < modal_k:
                        clam_matrix[sample_idx, cluster_id] += 1
            n_used += 1

    # Save outputs
    output_folder = os.path.join(output_dir, 'hdbscan')
    os.makedirs(output_folder, exist_ok=True)
    np.save(os.path.join(output_folder, 'clam_matrix.npy'), clam_matrix)

    if verbose:
        print(f"  HDBSCAN complete. Modal K={modal_k}, "
              f"agreement={k_agreement_rate:.2f}")

    return {
        'k_distribution': k_distribution,
        'modal_k': modal_k,
        'k_agreement_rate': k_agreement_rate,
        'clam_matrix': clam_matrix,
        'n_iterations_used': n_used,
        'iteration_labels': {
            'predicted': all_predicted_labels,
            'true': all_true_labels,
        },
        'noise_counts': noise_counts,
        'output_folder': output_folder,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_erica.py -k "test_hdbscan" -v`
Expected: Both tests PASS

- [ ] **Step 5: Commit**

```bash
git add erica/clustering.py tests/test_erica.py
git commit -m "feat: add hdbscan_clustering function with noise handling and auto-K"
```

---

### Task 4: Enhance K-Based Clustering for ARI/AMI (clustering.py)

**Files:**
- Modify: `erica/clustering.py`
- Test: `tests/test_erica.py`

- [ ] **Step 1: Write failing tests for label pair returns**

Add to `tests/test_erica.py`:

```python
def test_kmeans_returns_iteration_labels():
    """Test that kmeans_clustering returns per-iteration label pairs."""
    data = np.random.rand(30, 5)

    with tempfile.TemporaryDirectory() as tmpdir:
        from erica.clustering import iterative_clustering_subsampling, kmeans_clustering

        _, indices_folder = iterative_clustering_subsampling(
            data, 30, 5, 24, tmpdir, verbose=False
        )
        result = kmeans_clustering(
            data, k=2, n_iterations=5,
            indices_folder=indices_folder,
            output_dir=tmpdir, verbose=False
        )

    assert 'iteration_labels' in result
    assert 'predicted' in result['iteration_labels']
    assert 'true' in result['iteration_labels']
    assert len(result['iteration_labels']['predicted']) == 5
    assert len(result['iteration_labels']['true']) == 5


def test_agglomerative_returns_iteration_labels():
    """Test that agglomerative_clustering returns per-iteration label pairs."""
    data = np.random.rand(30, 5)

    with tempfile.TemporaryDirectory() as tmpdir:
        from erica.clustering import iterative_clustering_subsampling, agglomerative_clustering

        _, indices_folder = iterative_clustering_subsampling(
            data, 30, 5, 24, tmpdir, verbose=False
        )
        result = agglomerative_clustering(
            data, k=2, linkage='ward', n_iterations=5,
            indices_folder=indices_folder,
            output_dir=tmpdir, verbose=False
        )

    assert 'iteration_labels' in result
    assert 'predicted' in result['iteration_labels']
    assert 'true' in result['iteration_labels']
    assert len(result['iteration_labels']['predicted']) == 5
    assert len(result['iteration_labels']['true']) == 5
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_erica.py::test_kmeans_returns_iteration_labels -v`
Expected: FAIL (KeyError 'iteration_labels')

- [ ] **Step 3: Enhance kmeans_clustering to return label pairs**

In `erica/clustering.py`, inside `kmeans_clustering`:

Add storage lists before the iteration loop (after `iteration_centroids_list = []`):

```python
    all_predicted_labels = []
    all_true_labels = []
```

Inside the iteration loop, after `predictions = kmeans_iter.predict(test_data)`, add:

```python
        all_predicted_labels.append(predictions.copy())

        # Fit fresh model on test data for true_labels (Parmigiani method)
        kmeans_test = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
        true_labels = kmeans_test.fit_predict(test_data)
        all_true_labels.append(true_labels)
```

In the return dict, add the key:

```python
        'iteration_labels': {
            'predicted': all_predicted_labels,
            'true': all_true_labels,
        },
```

- [ ] **Step 4: Enhance agglomerative_clustering to return label pairs**

In `erica/clustering.py`, inside `agglomerative_clustering`:

Add storage lists before the iteration loop (after `iteration_centroids_list = []`):

```python
    all_predicted_labels = []
    all_true_labels = []
```

Change the line `_, test_data = load_iteration_data(...)` to:

```python
        train_data, test_data = load_iteration_data(
            iter_idx, samples_array, indices_folder
        )
```

After `predictions = agg_iter.fit_predict(test_data)`, add:

```python
        # true_labels = test-fitted predictions (already computed)
        all_true_labels.append(predictions.copy())

        # predicted_labels = train-fitted centroids applied to test data
        agg_train = AgglomerativeClustering(n_clusters=k, linkage=linkage)
        train_labels = agg_train.fit_predict(train_data)
        train_centroids_pred = np.zeros((k, train_data.shape[1]))
        for cidx in range(k):
            mask = train_labels == cidx
            if mask.any():
                train_centroids_pred[cidx] = train_data[mask].mean(axis=0)
            else:
                train_centroids_pred[cidx] = train_data.mean(axis=0)
        dists = np.linalg.norm(
            test_data[:, np.newaxis, :] - train_centroids_pred[np.newaxis, :, :],
            axis=2
        )
        pred_labels = np.argmin(dists, axis=1)
        all_predicted_labels.append(pred_labels)
```

In the return dict, add the key:

```python
        'iteration_labels': {
            'predicted': all_predicted_labels,
            'true': all_true_labels,
        },
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_erica.py -k "test_kmeans_returns_iteration_labels or test_agglomerative_returns_iteration_labels" -v`
Expected: Both tests PASS

- [ ] **Step 6: Run existing tests to verify no regressions**

Run: `pytest tests/test_erica.py -v`
Expected: All existing tests still PASS

- [ ] **Step 7: Commit**

```bash
git add erica/clustering.py tests/test_erica.py
git commit -m "feat: enhance K-based clustering to return per-iteration label pairs for ARI/AMI"
```

---

### Task 5: Refactor ERICA.__init__ for Flattened Method API (core.py)

**Files:**
- Modify: `erica/core.py`
- Test: `tests/test_erica.py`

- [ ] **Step 1: Write failing tests for new __init__ behavior**

Add to `tests/test_erica.py`:

```python
def test_erica_init_list_method():
    """Test ERICA initialization with method as a list."""
    data = np.random.rand(50, 10)
    erica = ERICA(
        data=data,
        k_range=[2, 3],
        n_iterations=10,
        method=['kmeans', 'agglomerative_ward'],
    )
    assert erica.k_based_methods == ['kmeans', 'agglomerative_ward']
    assert erica.auto_k_methods == []


def test_erica_init_hdbscan():
    """Test ERICA initialization with HDBSCAN method."""
    data = np.random.rand(50, 10)
    erica = ERICA(
        data=data,
        k_range=[2, 3],
        n_iterations=10,
        method=['kmeans', 'hdbscan'],
    )
    assert 'kmeans' in erica.k_based_methods
    assert 'hdbscan' in erica.auto_k_methods


def test_erica_init_all_method():
    """Test ERICA initialization with method='all'."""
    data = np.random.rand(50, 10)
    erica = ERICA(
        data=data,
        k_range=[2, 3],
        n_iterations=10,
        method='all',
    )
    assert len(erica.k_based_methods) == 5
    assert len(erica.auto_k_methods) == 1


def test_erica_init_both_deprecated():
    """Test ERICA initialization with deprecated method='both' still works."""
    data = np.random.rand(50, 10)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        erica = ERICA(
            data=data,
            k_range=[2, 3],
            n_iterations=10,
            method='both',
        )
        dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(dep_warnings) >= 1

    assert set(erica.k_based_methods) == {
        'kmeans', 'agglomerative_single', 'agglomerative_ward'
    }


def test_erica_init_linkages_rejected():
    """Test that passing linkages raises TypeError with migration message."""
    data = np.random.rand(50, 10)
    with pytest.raises(TypeError, match="linkages.*removed"):
        ERICA(
            data=data,
            k_range=[2, 3],
            n_iterations=10,
            method='kmeans',
            linkages=['ward'],
        )


def test_erica_init_hdbscan_params():
    """Test ERICA initialization with hdbscan_params."""
    data = np.random.rand(50, 10)
    erica = ERICA(
        data=data,
        k_range=[2, 3],
        n_iterations=10,
        method=['hdbscan'],
        hdbscan_params={'min_cluster_size': 10, 'min_samples': 3},
    )
    assert erica.hdbscan_params == {'min_cluster_size': 10, 'min_samples': 3}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_erica.py::test_erica_init_list_method -v`
Expected: FAIL (TypeError, current __init__ expects str for method)

- [ ] **Step 3: Refactor ERICA.__init__ in core.py**

Update the imports at the top of `erica/core.py`:

```python
from erica.clustering import (
    kmeans_clustering,
    agglomerative_clustering,
    hdbscan_clustering,
    iterative_clustering_subsampling,
)
from erica.metrics import compute_metrics_for_clam, select_optimal_k_by_method
from erica.data import prepare_samples_array, validate_dataset
from erica.utils import (
    set_deterministic_mode, compute_config_hash,
    normalize_method, K_BASED_METHODS, AUTO_K_METHODS,
)
```

Replace the `__init__` signature and body:

```python
    def __init__(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        k_range: Optional[List[int]] = None,
        n_iterations: int = 200,
        train_percent: float = 0.8,
        method: Union[str, List[str]] = 'both',
        random_seed: int = 123,
        output_dir: str = './erica_output',
        transpose: bool = True,
        verbose: bool = True,
        hdbscan_params: Optional[Dict] = None,
        **kwargs,
    ):
        """Initialize ERICA analysis.

        Parameters
        ----------
        data : np.ndarray or pd.DataFrame
            Input data for clustering analysis
        k_range : list of int, optional
            Range of k values to test (default: [2, 3, 4, 5])
        n_iterations : int, optional
            Number of subsampling iterations (default: 200)
        train_percent : float, optional
            Percentage of data for training (default: 0.8)
        method : str or list of str, optional
            Clustering method(s). Valid: 'kmeans', 'agglomerative_ward',
            'agglomerative_single', 'agglomerative_complete',
            'agglomerative_average', 'hdbscan', 'all'.
            Deprecated: 'both', 'agglomerative'. Default: 'both'
        random_seed : int, optional
            Random seed for reproducibility (default: 123)
        output_dir : str, optional
            Directory for output files (default: './erica_output')
        transpose : bool, optional
            Whether to transpose the data (default: True)
        verbose : bool, optional
            Print progress messages (default: True)
        hdbscan_params : dict, optional
            Parameters for HDBSCAN (min_cluster_size, min_samples)
        """
        # Reject removed parameters
        if 'linkages' in kwargs:
            raise TypeError(
                "The 'linkages' parameter has been removed. "
                "Use method=['agglomerative_ward', 'agglomerative_single'] "
                "instead. See the migration guide for details."
            )

        self.data = data
        self.k_range = k_range or [2, 3, 4, 5]
        self.n_iterations = n_iterations
        self.train_percent = train_percent
        self.random_seed = random_seed
        self.output_dir = output_dir
        self.transpose = transpose
        self.verbose = verbose
        self.hdbscan_params = hdbscan_params

        # Normalize and classify methods
        self.method_list = normalize_method(method)
        self.method = method  # keep original for backwards compat
        self.k_based_methods = [m for m in self.method_list if m in K_BASED_METHODS]
        self.auto_k_methods = [m for m in self.method_list if m in AUTO_K_METHODS]

        # Deterministic reproducibility
        set_deterministic_mode(random_seed)

        # Prepare and validate data
        self.samples_array = prepare_samples_array(data, transpose=transpose)
        self.n_samples, self.n_features = self.samples_array.shape
        validate_dataset(self.samples_array, min(self.k_range), self.train_percent)

        # Initialize storage
        self.results_ = {}
        self.clam_matrices_ = {}
        self.metrics_ = {}
        self.k_star_ = {}
        self.disqualified_k_ = {}
        self.auto_k_results_ = {}
        self.output_folders_ = []

        os.makedirs(output_dir, exist_ok=True)

        if self.verbose:
            print(f"ERICA initialized:")
            print(f"  Data shape: {self.samples_array.shape}")
            print(f"  K range: {self.k_range}")
            print(f"  Iterations: {self.n_iterations}")
            print(f"  Methods: {self.method_list}")
            print(f"  Random seed: {self.random_seed}")
```

- [ ] **Step 4: Run new init tests to verify they pass**

Run: `pytest tests/test_erica.py -k "test_erica_init_list_method or test_erica_init_hdbscan or test_erica_init_all_method or test_erica_init_both_deprecated or test_erica_init_linkages_rejected or test_erica_init_hdbscan_params" -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Run full test suite to check regressions**

Run: `pytest tests/test_erica.py -v`
Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add erica/core.py tests/test_erica.py
git commit -m "refactor: flatten method parameter in ERICA.__init__, remove linkages"
```

---

### Task 6: Refactor ERICA.run() with ARI/AMI and Auto-K (core.py)

**Files:**
- Modify: `erica/core.py`
- Test: `tests/test_erica.py`

- [ ] **Step 1: Write failing tests for ARI/AMI in results and auto-K**

Add to `tests/test_erica.py`:

```python
def test_erica_run_kbased_ari_ami():
    """Test that K-based run includes ARI/AMI metrics."""
    data = np.random.rand(30, 5)
    erica = ERICA(
        data=data,
        k_range=[2, 3],
        n_iterations=5,
        method='kmeans',
        verbose=False,
    )
    results = erica.run()
    metrics = results['metrics']

    for k in [2, 3]:
        assert k in metrics
        assert 'kmeans' in metrics[k]
        m = metrics[k]['kmeans']
        assert 'ARI_mean' in m
        assert 'ARI_std' in m
        assert 'AMI_mean' in m
        assert 'AMI_std' in m
        assert -1 <= m['ARI_mean'] <= 1
        assert 0 <= m['AMI_mean'] <= 1


def test_erica_run_agglomerative_list():
    """Test ERICA run with agglomerative method as flat string."""
    data = np.random.rand(30, 5)
    erica = ERICA(
        data=data,
        k_range=[2],
        n_iterations=3,
        method=['agglomerative_ward'],
        verbose=False,
    )
    results = erica.run()

    assert 2 in results['metrics']
    assert 'agglomerative_ward' in results['metrics'][2]


def test_erica_run_hdbscan():
    """Test ERICA end-to-end with HDBSCAN."""
    from sklearn.datasets import make_blobs

    data, _ = make_blobs(n_samples=80, n_features=5, centers=3,
                         cluster_std=0.5, random_state=42)
    erica = ERICA(
        data=data,
        k_range=[2, 3],
        n_iterations=10,
        method=['kmeans', 'hdbscan'],
        hdbscan_params={'min_cluster_size': 5},
        transpose=False,
        verbose=False,
    )
    results = erica.run()

    assert 'metrics' in results
    assert 2 in results['metrics']
    assert 'kmeans' in results['metrics'][2]

    assert 'auto_k' in results
    assert 'hdbscan' in results['auto_k']

    hdb = results['auto_k']['hdbscan']
    assert 'modal_k' in hdb
    assert 'k_distribution' in hdb
    assert 'k_agreement_rate' in hdb


def test_get_auto_k_results():
    """Test get_auto_k_results accessor."""
    from sklearn.datasets import make_blobs

    data, _ = make_blobs(n_samples=60, n_features=3, centers=2,
                         cluster_std=0.5, random_state=42)
    erica = ERICA(
        data=data,
        k_range=[2],
        n_iterations=5,
        method=['hdbscan'],
        hdbscan_params={'min_cluster_size': 5},
        transpose=False,
        verbose=False,
    )
    erica.run()

    result = erica.get_auto_k_results('hdbscan')
    assert result is not None
    assert 'modal_k' in result

    assert erica.get_auto_k_results('nonexistent') is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_erica.py::test_erica_run_kbased_ari_ami -v`
Expected: FAIL (ARI_mean not in metrics)

- [ ] **Step 3: Replace run(), _compute_all_metrics, and add accessors in core.py**

Replace `run()`:

```python
    def run(self) -> Dict:
        """Run the complete ERICA analysis."""
        from erica.metrics import (
            compute_parmigiani_metrics, aggregate_parmigiani_metrics
        )

        if self.verbose:
            print(f"\nStarting ERICA analysis at "
                  f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(self.output_dir, f"erica_run_{timestamp}")
        os.makedirs(run_dir, exist_ok=True)

        # Step 1: Iterative clustering subsampling
        if self.verbose:
            print(f"\n[1/4] Performing iterative clustering subsampling...")
        train_size = int(self.n_samples * self.train_percent)
        subsamples_folder, indices_folder = iterative_clustering_subsampling(
            samples_array=self.samples_array,
            num_samples=self.n_samples,
            num_iterations=self.n_iterations,
            subsample_size_train=train_size,
            base_save_folder_str=run_dir,
            verbose=self.verbose
        )

        # Step 2: K-based clustering
        if self.k_based_methods:
            if self.verbose:
                print(f"\n[2/4] Running K-based clustering methods...")
            for k in self.k_range:
                for method_name in self.k_based_methods:
                    if method_name == 'kmeans':
                        if self.verbose:
                            print(f"  Running K-Means for k={k}...")
                        result = kmeans_clustering(
                            samples_array=self.samples_array,
                            k=k,
                            n_iterations=self.n_iterations,
                            indices_folder=indices_folder,
                            output_dir=run_dir,
                            verbose=self.verbose,
                        )
                    elif method_name.startswith('agglomerative_'):
                        linkage = method_name.split('_', 1)[1]
                        if self.verbose:
                            print(f"  Running Agglomerative ({linkage}) for k={k}...")
                        result = agglomerative_clustering(
                            samples_array=self.samples_array,
                            k=k,
                            linkage=linkage,
                            n_iterations=self.n_iterations,
                            indices_folder=indices_folder,
                            output_dir=run_dir,
                            verbose=self.verbose,
                        )
                    else:
                        continue

                    self.clam_matrices_[(k, method_name)] = result['clam_matrix']
                    self.results_[(k, method_name)] = result

        # Step 3: Auto-K clustering
        if self.auto_k_methods:
            if self.verbose:
                print(f"\n[3/4] Running auto-K clustering methods...")
            for method_name in self.auto_k_methods:
                if method_name == 'hdbscan':
                    if self.verbose:
                        print(f"  Running HDBSCAN...")
                    result = hdbscan_clustering(
                        samples_array=self.samples_array,
                        n_iterations=self.n_iterations,
                        indices_folder=indices_folder,
                        output_dir=run_dir,
                        hdbscan_params=self.hdbscan_params,
                        verbose=self.verbose,
                    )
                    self.auto_k_results_[method_name] = result

        # Step 4: Compute metrics
        if self.verbose:
            print(f"\n[4/4] Computing metrics...")
        self.metrics_ = self._compute_all_metrics()

        # Select optimal K (K-based only)
        if self.k_based_methods and self.metrics_:
            self.k_star_ = self._select_optimal_k()

        self.output_folders_.append(run_dir)

        if self.verbose:
            print(f"\nERICA analysis complete!")
            print(f"Results saved to: {run_dir}")
            n_configs = len(self.results_) + len(self.auto_k_results_)
            print(f"Total configurations analyzed: {n_configs}")
            self._print_k_star_summary()

        return self.get_results()
```

Replace `_compute_all_metrics()`:

```python
    def _compute_all_metrics(self) -> Dict:
        """Compute CRI, WCRI, TWCRI, and Parmigiani metrics for all results."""
        from erica.metrics import (
            compute_parmigiani_metrics, aggregate_parmigiani_metrics
        )

        metrics_by_k = {}
        for (k, method_name), result in self.results_.items():
            clam_matrix = result['clam_matrix']
            metrics = compute_metrics_for_clam(clam_matrix, k)

            # Compute ARI/AMI from iteration label pairs
            if 'iteration_labels' in result:
                predicted_list = result['iteration_labels']['predicted']
                true_list = result['iteration_labels']['true']
                ari_scores = []
                ami_scores = []
                for pred, true in zip(predicted_list, true_list):
                    pm = compute_parmigiani_metrics(pred, true)
                    ari_scores.append(pm['ARI'])
                    ami_scores.append(pm['AMI'])
                agg = aggregate_parmigiani_metrics(ari_scores, ami_scores)
                metrics['ARI_mean'] = agg['ARI_mean']
                metrics['ARI_std'] = agg['ARI_std']
                metrics['AMI_mean'] = agg['AMI_mean']
                metrics['AMI_std'] = agg['AMI_std']

            if k not in metrics_by_k:
                metrics_by_k[k] = {}
            metrics_by_k[k][method_name] = metrics

            if metrics.get('has_empty_clusters', False):
                if method_name not in self.disqualified_k_:
                    self.disqualified_k_[method_name] = []
                if k not in self.disqualified_k_[method_name]:
                    self.disqualified_k_[method_name].append(k)

        for method_name in self.disqualified_k_:
            self.disqualified_k_[method_name].sort()

        # Auto-K metrics
        for method_name, result in self.auto_k_results_.items():
            if 'iteration_labels' in result:
                predicted_list = result['iteration_labels']['predicted']
                true_list = result['iteration_labels']['true']
                ari_scores = []
                ami_scores = []
                for pred, true in zip(predicted_list, true_list):
                    if len(pred) > 0 and len(true) > 0:
                        pm = compute_parmigiani_metrics(pred, true)
                        ari_scores.append(pm['ARI'])
                        ami_scores.append(pm['AMI'])
                if ari_scores:
                    agg = aggregate_parmigiani_metrics(ari_scores, ami_scores)
                    result['ARI_mean'] = agg['ARI_mean']
                    result['ARI_std'] = agg['ARI_std']
                    result['AMI_mean'] = agg['AMI_mean']
                    result['AMI_std'] = agg['AMI_std']

            if result.get('modal_k', 0) > 0:
                clam_metrics = compute_metrics_for_clam(
                    result['clam_matrix'], result['modal_k']
                )
                result['metrics_at_modal_k'] = clam_metrics

        return metrics_by_k
```

Add `get_auto_k_results` method:

```python
    def get_auto_k_results(self, method: str = 'hdbscan') -> Optional[Dict]:
        """Get auto-K clustering results for a specific method.

        Parameters
        ----------
        method : str, optional
            Auto-K method name (default: 'hdbscan')

        Returns
        -------
        dict or None
            Results dictionary or None if method not found
        """
        return self.auto_k_results_.get(method)
```

Update `get_results`:

```python
    def get_results(self) -> Dict:
        """Get all analysis results."""
        return {
            'clam_matrices': self.clam_matrices_,
            'metrics': self.metrics_,
            'k_star': self.k_star_,
            'disqualified_k': self.disqualified_k_,
            'auto_k': self.auto_k_results_,
            'config': self._get_config_dict(),
            'output_folders': self.output_folders_,
            'results': self.results_,
        }
```

Update `_get_config_dict`:

```python
    def _get_config_dict(self) -> Dict:
        """Return ERICA configuration as dictionary."""
        return {
            'k_range': self.k_range,
            'n_iterations': self.n_iterations,
            'train_percent': self.train_percent,
            'method': self.method_list,
            'random_seed': self.random_seed,
            'n_samples': self.n_samples,
            'n_features': self.n_features,
            'hdbscan_params': self.hdbscan_params,
            'config_hash': compute_config_hash({
                'k_range': self.k_range,
                'n_iterations': self.n_iterations,
                'train_percent': self.train_percent,
                'method': sorted(self.method_list),
                'random_seed': self.random_seed,
            })
        }
```

Update `__repr__`:

```python
    def __repr__(self) -> str:
        return (
            f"ERICA(n_samples={self.n_samples}, n_features={self.n_features}, "
            f"k_range={self.k_range}, n_iterations={self.n_iterations}, "
            f"methods={self.method_list})"
        )
```

- [ ] **Step 4: Run new tests to verify they pass**

Run: `pytest tests/test_erica.py -k "test_erica_run_kbased_ari_ami or test_erica_run_agglomerative_list or test_erica_run_hdbscan or test_get_auto_k_results" -v`
Expected: All 4 PASS

- [ ] **Step 5: Run full test suite**

Run: `pytest tests/test_erica.py -v`
Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add erica/core.py tests/test_erica.py
git commit -m "feat: add ARI/AMI to K-based metrics, auto-K execution path, and result accessors"
```

---

### Task 7: Update Exports (__init__.py)

**Files:**
- Modify: `erica/__init__.py`

- [ ] **Step 1: Update imports and __all__ in __init__.py**

Add to the clustering imports:

```python
from erica.clustering import (
    kmeans_clustering,
    agglomerative_clustering,
    hdbscan_clustering,
    iterative_clustering_subsampling,
)
```

Add to the utils imports:

```python
from erica.utils import (
    set_deterministic_mode,
    compute_config_hash,
    normalize_method,
    VALID_METHODS,
    K_BASED_METHODS,
    AUTO_K_METHODS,
)
```

Add to `__all__`:

```python
    'hdbscan_clustering',
    'normalize_method',
    'VALID_METHODS',
    'K_BASED_METHODS',
    'AUTO_K_METHODS',
```

- [ ] **Step 2: Verify imports work**

Run: `python3 -c "from erica import hdbscan_clustering, normalize_method, VALID_METHODS; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Run full test suite**

Run: `pytest tests/test_erica.py -v`
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add erica/__init__.py
git commit -m "feat: export hdbscan_clustering, normalize_method, and method constants"
```

---

### Task 8: Integration Tests

**Files:**
- Test: `tests/test_erica.py`

- [ ] **Step 1: Write comprehensive integration tests**

Add to `tests/test_erica.py`:

```python
def test_erica_full_integration_all_methods():
    """Integration test: run ERICA with multiple K-based + auto-K methods."""
    from sklearn.datasets import make_blobs

    data, _ = make_blobs(n_samples=60, n_features=4, centers=3,
                         cluster_std=0.8, random_state=42)
    erica = ERICA(
        data=data,
        k_range=[2, 3],
        n_iterations=5,
        method=['kmeans', 'agglomerative_ward', 'hdbscan'],
        hdbscan_params={'min_cluster_size': 5},
        transpose=False,
        verbose=False,
    )
    results = erica.run()

    # K-based metrics
    for k in [2, 3]:
        assert 'kmeans' in results['metrics'][k]
        assert 'agglomerative_ward' in results['metrics'][k]
        for method in ['kmeans', 'agglomerative_ward']:
            m = results['metrics'][k][method]
            assert 'CRI' in m
            assert 'TWCRI' in m
            assert 'ARI_mean' in m
            assert 'AMI_mean' in m

    # K* computed
    assert 'TWCRI' in results['k_star']
    assert 'kmeans' in results['k_star']['TWCRI']
    assert 'agglomerative_ward' in results['k_star']['TWCRI']

    # Auto-K results
    assert 'hdbscan' in results['auto_k']
    hdb = results['auto_k']['hdbscan']
    assert 'modal_k' in hdb
    assert 'k_distribution' in hdb
    assert 'clam_matrix' in hdb

    # Config reflects new API
    assert isinstance(results['config']['method'], list)
    assert 'hdbscan' in results['config']['method']


def test_erica_backwards_compat_both():
    """Test that method='both' still produces valid results."""
    data = np.random.rand(30, 5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        erica = ERICA(
            data=data,
            k_range=[2, 3],
            n_iterations=3,
            method='both',
            verbose=False,
        )
    results = erica.run()

    for k in [2, 3]:
        assert 'kmeans' in results['metrics'][k]
        assert 'agglomerative_single' in results['metrics'][k]
        assert 'agglomerative_ward' in results['metrics'][k]
```

- [ ] **Step 2: Run integration tests**

Run: `pytest tests/test_erica.py -k "test_erica_full_integration or test_erica_backwards_compat_both" -v`
Expected: Both PASS

- [ ] **Step 3: Run the complete test suite**

Run: `pytest tests/test_erica.py -v`
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add tests/test_erica.py
git commit -m "test: add integration tests for flattened API and backwards compat"
```

---

### Task 9: Example Script (06_hdbscan.py)

**Files:**
- Create: `examples/06_hdbscan.py`

- [ ] **Step 1: Write the HDBSCAN example**

Create `examples/06_hdbscan.py`:

```python
"""
Example 06: HDBSCAN Auto-K Clustering with ERICA

Demonstrates how to use HDBSCAN as an auto-K clustering method alongside
traditional K-based methods. HDBSCAN discovers the number of clusters
automatically, while ERICA evaluates replicability across subsamples.
"""

import numpy as np
from sklearn.datasets import make_blobs
from erica import ERICA

# Generate synthetic data with 3 clusters
data, true_labels = make_blobs(
    n_samples=200, n_features=5, centers=3,
    cluster_std=1.0, random_state=42
)

print("=" * 60)
print("ERICA with HDBSCAN Auto-K Clustering")
print("=" * 60)

# Run ERICA with both K-Means and HDBSCAN
erica = ERICA(
    data=data,
    k_range=[2, 3, 4, 5],
    n_iterations=50,
    method=['kmeans', 'hdbscan'],
    hdbscan_params={'min_cluster_size': 10},
    transpose=False,
    verbose=True,
)

results = erica.run()

# --- K-based results ---
print("\n--- K-Based Results (K-Means) ---")
k_star = erica.get_k_star('TWCRI')
print(f"Optimal K* (TWCRI): {k_star.get('kmeans', 'N/A')}")

for k in [2, 3, 4, 5]:
    m = results['metrics'][k]['kmeans']
    print(f"  K={k}: TWCRI={m['TWCRI']:.4f}, "
          f"ARI={m['ARI_mean']:.4f} +/- {m['ARI_std']:.4f}")

# --- Auto-K results ---
print("\n--- Auto-K Results (HDBSCAN) ---")
hdb = results['auto_k']['hdbscan']
print(f"Modal K: {hdb['modal_k']}")
print(f"K distribution: {hdb['k_distribution']}")
print(f"K agreement rate: {hdb['k_agreement_rate']:.2f}")
print(f"Iterations used for CLAM: {hdb['n_iterations_used']}")

if 'ARI_mean' in hdb:
    print(f"ARI: {hdb['ARI_mean']:.4f} +/- {hdb['ARI_std']:.4f}")
    print(f"AMI: {hdb['AMI_mean']:.4f} +/- {hdb['AMI_std']:.4f}")

if 'metrics_at_modal_k' in hdb:
    mk = hdb['metrics_at_modal_k']
    print(f"TWCRI at modal K: {mk.get('TWCRI', 'N/A')}")

print(f"\nNoise points per iteration (mean): "
      f"{np.mean(hdb['noise_counts']):.1f}")
```

- [ ] **Step 2: Verify example runs**

Run: `cd /Users/shawnshirazi/LocalExperiments/ERICA && python3 examples/06_hdbscan.py`
Expected: Output showing K-Means and HDBSCAN results without errors

- [ ] **Step 3: Commit**

```bash
git add examples/06_hdbscan.py
git commit -m "docs: add HDBSCAN example script (example 06)"
```
