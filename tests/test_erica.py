"""
Basic tests for ERICA package.
"""

import pytest
import numpy as np
import pandas as pd
import os
import tempfile
from erica import ERICA
from erica.data import load_data, prepare_samples_array
from erica.metrics import select_optimal_k, select_optimal_k_by_method
from erica.clustering import _assign_noise_to_nearest, hdbscan_clustering


def test_erica_import():
    """Test that ERICA can be imported."""
    import erica
    assert hasattr(erica, 'ERICA')


def test_erica_initialization():
    """Test ERICA initialization with sample data."""
    # Create sample data
    data = np.random.rand(50, 10)
    
    # Initialize ERICA
    erica = ERICA(
        data=data,
        k_range=[2, 3],
        n_iterations=10,  # Small number for testing
        method='kmeans'
    )
    
    assert erica.data is not None
    assert erica.k_range == [2, 3]
    assert erica.n_iterations == 10
    assert erica.method == 'kmeans'


def test_erica_run():
    """Test ERICA run method."""
    # Create sample data
    data = np.random.rand(30, 5)
    
    # Initialize and run ERICA
    erica = ERICA(
        data=data,
        k_range=[2, 3],
        n_iterations=5,  # Small number for testing
        method='kmeans'
    )
    
    results = erica.run()
    
    # Check that results are returned
    assert results is not None
    assert isinstance(results, dict)


def test_erica_metrics():
    """Test ERICA metrics computation."""
    # Create sample data
    data = np.random.rand(20, 4)
    
    # Initialize and run ERICA
    erica = ERICA(
        data=data,
        k_range=[2],
        n_iterations=3,  # Small number for testing
        method='kmeans'
    )
    
    erica.run()
    metrics = erica.get_metrics()
    
    # Check that metrics are computed
    assert metrics is not None
    assert isinstance(metrics, dict)


def test_csv_loading_samples_in_rows():
    """Test CSV loading with samples in rows (standard format)."""
    # Create temporary CSV with samples in rows
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write('sample_id,feature1,feature2,feature3\n')
        for i in range(50):
            f.write(f'sample_{i},{np.random.rand()},{np.random.rand()},{np.random.rand()}\n')
        temp_file = f.name
    
    try:
        # Load and prepare data
        data = load_data(temp_file)
        samples_array = prepare_samples_array(data, transpose=False)
        
        # Should have 50 samples and 3 features
        assert samples_array.shape[0] == 50
        assert samples_array.shape[1] == 3
    finally:
        os.unlink(temp_file)


def test_csv_loading_features_in_rows():
    """Test CSV loading with features in rows (genomics format)."""
    # Create temporary CSV with features in rows
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        # Create 50 features x 100 samples
        f.write('gene_id,' + ','.join([f'sample{i}' for i in range(100)]) + '\n')
        for i in range(50):
            values = ','.join([str(np.random.rand()) for _ in range(100)])
            f.write(f'gene_{i},{values}\n')
        temp_file = f.name
    
    try:
        # Load and prepare data
        data = load_data(temp_file)
        samples_array = prepare_samples_array(data, transpose=True)
        
        # Should have 100 samples and 50 features (transposed)
        assert samples_array.shape[0] == 100
        assert samples_array.shape[1] == 50
    finally:
        os.unlink(temp_file)


def test_csv_transpose_parameter():
    """Test explicit transpose parameter."""
    # Create temporary CSV
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write('id,col1,col2,col3\n')
        for i in range(20):
            f.write(f'row_{i},{np.random.rand()},{np.random.rand()},{np.random.rand()}\n')
        temp_file = f.name
    
    try:
        data = load_data(temp_file)
        
        # Test transpose=False
        samples_no = prepare_samples_array(data, transpose=False)
        assert samples_no.shape == (20, 3)
        
        # Test transpose=True
        samples_yes = prepare_samples_array(data, transpose=True)
        assert samples_yes.shape == (3, 20)
    finally:
        os.unlink(temp_file)


def test_erica_with_csv():
    """Test ERICA end-to-end with CSV data."""
    # Create temporary CSV
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write('sample_id,feature1,feature2,feature3\n')
        for i in range(30):
            f.write(f'sample_{i},{np.random.rand()},{np.random.rand()},{np.random.rand()}\n')
        temp_file = f.name
    
    try:
        # Load data
        data = load_data(temp_file)
        
        # Run ERICA
        erica = ERICA(
            data=data,
            k_range=[2, 3],
            n_iterations=5,
            method='kmeans',
            transpose=False
        )
        
        results = erica.run()
        
        # Check results
        assert results is not None
        assert erica.n_samples == 30
        assert erica.n_features == 3
    finally:
        os.unlink(temp_file)


def test_select_optimal_k_basic():
    """Test basic K* selection with simple metric dictionary."""
    # Example from the algorithm specification
    M = {2: 0.71, 3: 0.75, 4: 0.74, 5: float('nan'), 6: 0.78}
    k_star = select_optimal_k(M)
    
    # Should select K=6 (last non-decreasing value)
    assert k_star == 6


def test_select_optimal_k_monotonic_increasing():
    """Test K* selection with monotonically increasing metrics."""
    M = {2: 0.71, 3: 0.75, 4: 0.80, 5: 0.85}
    k_star = select_optimal_k(M)
    
    # Should select the largest K since all are non-decreasing
    assert k_star == 5


def test_select_optimal_k_monotonic_decreasing():
    """Test K* selection with monotonically decreasing metrics."""
    M = {2: 0.85, 3: 0.80, 4: 0.75, 5: 0.70}
    k_star = select_optimal_k(M)
    
    # Should select K=2 since no subsequent values are >= previous
    assert k_star == 2


def test_select_optimal_k_with_nan():
    """Test K* selection handles NaN values correctly."""
    M = {2: 0.71, 3: float('nan'), 4: 0.75, 5: 0.80}
    k_star = select_optimal_k(M)
    
    # Should skip NaN and continue evaluation
    # K=4 (0.75) is not >= K=2 (0.71), but K=5 (0.80) >= K=4 (0.75)
    assert k_star == 5


def test_select_optimal_k_all_nan():
    """Test K* selection when all values except first are NaN."""
    M = {2: 0.71, 3: float('nan'), 4: float('nan'), 5: float('nan')}
    k_star = select_optimal_k(M)
    
    # Should return K=2 since no valid comparisons can be made
    assert k_star == 2


def test_select_optimal_k_with_gaps():
    """Test K* selection with non-consecutive K values."""
    M = {2: 0.71, 4: 0.75, 6: 0.80, 8: 0.78}
    k_star = select_optimal_k(M)
    
    # Should handle gaps in K values correctly
    # K=4 >= K=2, K=6 >= K=4, but K=8 < K=6
    assert k_star == 6


def test_select_optimal_k_equal_values():
    """Test K* selection with equal consecutive values."""
    M = {2: 0.75, 3: 0.75, 4: 0.75, 5: 0.75}
    k_star = select_optimal_k(M)
    
    # Should select the largest K since all are equal (non-decreasing)
    assert k_star == 5


def test_select_optimal_k_empty_dict():
    """Test K* selection raises error on empty dictionary."""
    with pytest.raises(ValueError):
        select_optimal_k({})


def test_select_optimal_k_single_value():
    """Test K* selection with single K value."""
    M = {2: 0.75}
    k_star = select_optimal_k(M)
    
    # Should return the only K available
    assert k_star == 2


def test_select_optimal_k_with_kmax():
    """Test K* selection with k_max parameter."""
    M = {2: 0.71, 3: 0.75, 4: 0.80, 5: 0.85, 6: 0.90}
    k_star = select_optimal_k(M, k_max=4)
    
    # Should only consider up to K=4
    assert k_star == 4


def test_select_optimal_k_by_method():
    """Test K* selection for multiple methods."""
    metrics = {
        2: {'kmeans': {'TWCRI': 0.71}},
        3: {'kmeans': {'TWCRI': 0.75}},
        4: {'kmeans': {'TWCRI': 0.74}}
    }
    
    optimal_k = select_optimal_k_by_method(metrics, 'TWCRI')
    
    assert 'kmeans' in optimal_k
    assert optimal_k['kmeans'] == 3


def test_select_optimal_k_by_method_multiple_methods():
    """Test K* selection with multiple clustering methods."""
    metrics = {
        2: {
            'kmeans': {'TWCRI': 0.71},
            'agglomerative_ward': {'TWCRI': 0.68}
        },
        3: {
            'kmeans': {'TWCRI': 0.75},
            'agglomerative_ward': {'TWCRI': 0.72}
        },
        4: {
            'kmeans': {'TWCRI': 0.74},
            'agglomerative_ward': {'TWCRI': 0.76}
        }
    }
    
    optimal_k = select_optimal_k_by_method(metrics, 'TWCRI')
    
    assert optimal_k['kmeans'] == 3
    assert optimal_k['agglomerative_ward'] == 4


def test_select_optimal_k_by_method_invalid_metric():
    """Test K* selection raises error for invalid metric name."""
    metrics = {2: {'kmeans': {'TWCRI': 0.71}}}
    
    with pytest.raises(ValueError):
        select_optimal_k_by_method(metrics, 'INVALID_METRIC')


def test_erica_k_star_integration():
    """Test that ERICA computes K* automatically."""
    # Create sample data
    data = np.random.rand(30, 5)
    
    # Initialize and run ERICA
    erica = ERICA(
        data=data,
        k_range=[2, 3, 4],
        n_iterations=5,
        method='kmeans',
        verbose=False
    )
    
    results = erica.run()
    
    # Check that K* is computed
    assert 'k_star' in results
    assert results['k_star'] is not None
    assert isinstance(results['k_star'], dict)
    
    # Check that all metrics have K* values
    assert 'CRI' in results['k_star']
    assert 'WCRI' in results['k_star']
    assert 'TWCRI' in results['k_star']
    
    # Check that get_k_star method works
    k_star_twcri = erica.get_k_star('TWCRI')
    assert 'kmeans' in k_star_twcri
    assert k_star_twcri['kmeans'] in [2, 3, 4]


def test_erica_k_star_multiple_methods():
    """Test K* computation with multiple clustering methods."""
    # Create sample data
    data = np.random.rand(30, 5)

    # Initialize and run ERICA with both methods
    erica = ERICA(
        data=data,
        k_range=[2, 3],
        n_iterations=3,
        method=['kmeans', 'agglomerative_ward'],
        verbose=False
    )
    
    results = erica.run()
    
    # Check K* for both methods
    k_star_twcri = results['k_star']['TWCRI']
    assert 'kmeans' in k_star_twcri
    assert 'agglomerative_ward' in k_star_twcri


def test_empty_cluster_disqualification():
    """Test that K values with empty clusters are disqualified (marked as NaN)."""
    from erica.metrics import compute_metrics_for_clam
    
    # Create a CLAM matrix where one cluster is empty (k=3, but only 2 clusters used)
    # Samples 0-2 assigned to cluster 0, samples 3-5 assigned to cluster 1, cluster 2 is empty
    clam_matrix = np.array([
        [50, 10, 0],  # Sample 0: primarily cluster 0
        [45, 15, 0],  # Sample 1: primarily cluster 0
        [48, 12, 0],  # Sample 2: primarily cluster 0
        [5, 55, 0],   # Sample 3: primarily cluster 1
        [8, 52, 0],   # Sample 4: primarily cluster 1
        [10, 50, 0],  # Sample 5: primarily cluster 1
    ])
    
    metrics = compute_metrics_for_clam(clam_matrix, k=3)
    
    # Check that metrics are marked as NaN due to empty cluster
    assert np.isnan(metrics['CRI']), "CRI should be NaN when there's an empty cluster"
    assert np.isnan(metrics['WCRI']), "WCRI should be NaN when there's an empty cluster"
    assert np.isnan(metrics['TWCRI']), "TWCRI should be NaN when there's an empty cluster"
    assert metrics['has_empty_clusters'] is True
    assert metrics['cluster_sizes'][2] == 0


def test_no_empty_cluster_averaging():
    """Test that CRI is averaged only over non-empty clusters when no empty clusters exist."""
    from erica.metrics import compute_metrics_for_clam
    
    # Create a CLAM matrix with all clusters populated
    clam_matrix = np.array([
        [50, 10],  # Sample 0: primarily cluster 0
        [45, 15],  # Sample 1: primarily cluster 0
        [5, 55],   # Sample 2: primarily cluster 1
    ])
    
    metrics = compute_metrics_for_clam(clam_matrix, k=2)
    
    # Check that metrics are valid (not NaN)
    assert not np.isnan(metrics['CRI'])
    assert not np.isnan(metrics['WCRI'])
    assert not np.isnan(metrics['TWCRI'])
    assert metrics['has_empty_clusters'] is False
    
    # Verify CRI is computed correctly
    cri_per_cluster = metrics['CRI_per_cluster']
    expected_cri = np.mean(cri_per_cluster)
    assert abs(metrics['CRI'] - expected_cri) < 1e-6


def test_k_star_skips_empty_clusters():
    """Test that K* selection skips K values with empty clusters (NaN metrics)."""
    # Simulate metrics where K=4 has an empty cluster (NaN)
    metrics_by_k = {
        2: {'kmeans': {'TWCRI': 0.71, 'CRI': 0.75, 'WCRI': 0.70}},
        3: {'kmeans': {'TWCRI': 0.75, 'CRI': 0.78, 'WCRI': 0.72}},
        4: {'kmeans': {'TWCRI': float('nan'), 'CRI': float('nan'), 'WCRI': float('nan')}},  # Empty cluster
        5: {'kmeans': {'TWCRI': 0.78, 'CRI': 0.80, 'WCRI': 0.76}},
    }
    
    # Test with TWCRI
    optimal_k = select_optimal_k_by_method(metrics_by_k, 'TWCRI')
    
    # Should skip K=4 (NaN) and select K=5
    assert optimal_k['kmeans'] == 5
    
    # Test with CRI
    optimal_k_cri = select_optimal_k_by_method(metrics_by_k, 'CRI')
    assert optimal_k_cri['kmeans'] == 5
    
    # Test with WCRI
    optimal_k_wcri = select_optimal_k_by_method(metrics_by_k, 'WCRI')
    assert optimal_k_wcri['kmeans'] == 5


def test_get_disqualified_k():
    """Test that ERICA tracks and returns disqualified K values."""
    # Create sample data
    data = np.random.rand(30, 5)
    
    # Initialize and run ERICA
    erica = ERICA(
        data=data,
        k_range=[2, 3],
        n_iterations=5,
        method='kmeans',
        verbose=False
    )
    
    results = erica.run()
    
    # Check that disqualified_k is in results
    assert 'disqualified_k' in results
    assert isinstance(results['disqualified_k'], dict)
    
    # Check that get_disqualified_k method works
    disqualified = erica.get_disqualified_k()
    assert isinstance(disqualified, dict)
    
    # Check that we can get disqualified K for specific method
    disqualified_kmeans = erica.get_disqualified_k('kmeans')
    assert isinstance(disqualified_kmeans, list)
    
    # For this random data, we likely won't have empty clusters,
    # but the structure should be correct
    if 'kmeans' in disqualified:
        assert all(isinstance(k, int) for k in disqualified['kmeans'])


import warnings
from erica.utils import (
    VALID_METHODS, K_BASED_METHODS, AUTO_K_METHODS,
    normalize_method,
)


def test_method_constants():
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
    assert normalize_method('kmeans') == ['kmeans']
    assert normalize_method('hdbscan') == ['hdbscan']
    assert normalize_method('agglomerative_ward') == ['agglomerative_ward']


def test_normalize_method_list():
    result = normalize_method(['kmeans', 'hdbscan'])
    assert result == ['kmeans', 'hdbscan']


def test_normalize_method_all():
    result = normalize_method('all')
    assert set(result) == set(VALID_METHODS)


def test_normalize_method_both_deprecated():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = normalize_method('both')
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert 'deprecated' in str(w[0].message).lower()
    assert result == ['kmeans', 'agglomerative_single', 'agglomerative_ward']


def test_normalize_method_agglomerative_deprecated():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = normalize_method('agglomerative')
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
    assert result == ['agglomerative_single', 'agglomerative_ward']


def test_normalize_method_invalid_string():
    with pytest.raises(ValueError, match="Unknown method"):
        normalize_method('invalid_method')


def test_normalize_method_invalid_in_list():
    with pytest.raises(ValueError, match="Unknown method"):
        normalize_method(['kmeans', 'invalid_method'])


def test_normalize_method_wrong_type():
    with pytest.raises(TypeError):
        normalize_method(123)


def test_assign_noise_to_nearest_no_noise():
    labels = np.array([0, 1, 2, 0, 1])
    data = np.random.rand(5, 3)
    centroids = np.random.rand(3, 3)
    result = _assign_noise_to_nearest(labels, data, centroids)
    np.testing.assert_array_equal(result, labels)


def test_assign_noise_to_nearest_with_noise():
    centroids = np.array([[0.0, 0.0], [10.0, 10.0]])
    data = np.array([[0.1, 0.1], [9.9, 9.9], [0.2, 0.2]])
    labels = np.array([0, 1, -1])
    result = _assign_noise_to_nearest(labels, data, centroids)
    assert result[0] == 0
    assert result[1] == 1
    assert result[2] == 0


def test_assign_noise_to_nearest_all_noise():
    centroids = np.array([[0.0, 0.0], [10.0, 10.0]])
    data = np.array([[0.1, 0.1], [9.9, 9.9]])
    labels = np.array([-1, -1])
    result = _assign_noise_to_nearest(labels, data, centroids)
    assert result[0] == 0
    assert result[1] == 1


def test_hdbscan_clustering_basic():
    from sklearn.datasets import make_blobs
    data, _ = make_blobs(n_samples=100, n_features=5, centers=3, random_state=42)

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
            samples_array=data, n_iterations=n_iterations,
            indices_folder=indices_folder, output_dir=tmpdir,
            hdbscan_params={'min_cluster_size': 5}, verbose=False,
        )

    assert 'k_distribution' in result
    assert 'modal_k' in result
    assert 'k_agreement_rate' in result
    assert 'clam_matrix' in result
    assert 'n_iterations_used' in result
    assert 'noise_counts' in result
    assert result['modal_k'] >= 1
    assert 0 <= result['k_agreement_rate'] <= 1
    assert result['n_iterations_used'] <= n_iterations
    assert result['clam_matrix'].shape[0] == n_samples
    assert result['clam_matrix'].shape[1] == result['modal_k']


def test_hdbscan_clam_uses_test_set_semantics():
    """HDBSCAN CLAM rows must accumulate at test_indices, not train_indices.

    Semantic parity with K-Means / Agglomerative: each iteration fits on
    train, predicts on test, and accumulates predictions at test_indices.

    For modal-K iterations, every test point is assigned to exactly one
    train centroid -> aligned cluster (no -1s in test_predicted_labels),
    so each row's CLAM mass should equal the number of modal-K iterations
    in which that sample appeared in the TEST set. Under the broken
    train-fit-at-train-indices behavior, that equality would not hold and
    samples-mostly-in-train would have HIGHER CLAM mass than samples-mostly-
    in-test (the opposite of correct semantics).
    """
    from sklearn.datasets import make_blobs
    rng = np.random.RandomState(0)
    data, _ = make_blobs(
        n_samples=120, n_features=4, centers=3, cluster_std=0.5,
        random_state=42,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        indices_folder = os.path.join(tmpdir, 'indices')
        os.makedirs(indices_folder)
        n_iterations = 8
        n_samples = len(data)
        train_size = int(n_samples * 0.8)
        all_train = []
        all_test = []
        for _ in range(n_iterations):
            perm = rng.permutation(n_samples)
            all_train.append(perm[:train_size])
            all_test.append(perm[train_size:])
        np.save(os.path.join(indices_folder, 'all_train_indices.npy'),
                np.array(all_train, dtype=object))
        np.save(os.path.join(indices_folder, 'all_test_indices.npy'),
                np.array(all_test, dtype=object))

        result = hdbscan_clustering(
            samples_array=data, n_iterations=n_iterations,
            indices_folder=indices_folder, output_dir=tmpdir,
            hdbscan_params={'min_cluster_size': 5}, verbose=False,
        )

        clam = result['clam_matrix']
        modal_k = result['modal_k']

        # Determine which iterations participated in CLAM accumulation
        # (their discovered K equals modal_k). We re-derive participation
        # from the recorded train sets: a sample's CLAM row sum should
        # equal the number of participating iterations in which it was
        # in the TEST set.
        #
        # We don't have direct access to discovered_ks here; instead we
        # use n_iterations_used as the count of participating iterations
        # and compute test-membership counts across ALL iterations as an
        # upper bound. The strict invariant we test:
        #   row_sum[i] <= (number of iterations where i was in test set)
        # plus the aggregate equality:
        #   sum(row_sums) == n_iterations_used * test_size
        # The aggregate equality is the load-bearing test-vs-train check.
        n_used = result['n_iterations_used']
        test_size = len(all_test[0])

        # Aggregate: total CLAM mass = n_used * test_size (test-set semantics).
        # If the bug were present (accumulation at train_indices), the total
        # would be n_used * train_size instead, which is much larger.
        total_mass = clam.sum()
        assert total_mass == n_used * test_size, (
            f"Total CLAM mass {total_mass} != n_used * test_size "
            f"{n_used * test_size}; train-set accumulation would give "
            f"{n_used * train_size}"
        )

        # Per-sample upper bound: row sum cannot exceed times-in-test count
        # (each in-test appearance contributes at most 1 across modal-K iters).
        times_in_test = np.zeros(n_samples, dtype=int)
        for test_arr in all_test:
            times_in_test[np.asarray(test_arr, dtype=int)] += 1
        row_sums = clam.sum(axis=1)
        assert np.all(row_sums <= times_in_test), (
            "Some samples have CLAM mass > times-in-test count, indicating "
            "accumulation at the wrong index set"
        )

        # And complementary: row sum cannot exceed times-in-test, but if the
        # bug were present, row_sums would correlate with times-in-TRAIN.
        # Concretely, a sample mostly in train would have row sum > times-in-test.
        times_in_train = n_iterations - times_in_test
        # Expect no sample to have CLAM mass exceeding its train-membership count
        # only weakly under the bug; the strong check is the aggregate above.
        # Here we just sanity check shapes.
        assert clam.shape == (n_samples, modal_k)
        assert times_in_train.sum() + times_in_test.sum() == n_samples * n_iterations


def test_hdbscan_clustering_noise_handling():
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
            samples_array=data, n_iterations=n_iterations,
            indices_folder=indices_folder, output_dir=tmpdir,
            hdbscan_params={'min_cluster_size': 5}, verbose=False,
        )

    assert all(n >= 0 for n in result['noise_counts'])


def test_erica_init_list_method():
    data = np.random.rand(50, 10)
    erica = ERICA(
        data=data, k_range=[2, 3], n_iterations=10,
        method=['kmeans', 'agglomerative_ward'],
    )
    assert erica.k_based_methods == ['kmeans', 'agglomerative_ward']
    assert erica.auto_k_methods == []


def test_erica_init_hdbscan():
    data = np.random.rand(50, 10)
    erica = ERICA(
        data=data, k_range=[2, 3], n_iterations=10,
        method=['kmeans', 'hdbscan'],
    )
    assert 'kmeans' in erica.k_based_methods
    assert 'hdbscan' in erica.auto_k_methods


def test_erica_init_all_method():
    data = np.random.rand(50, 10)
    erica = ERICA(
        data=data, k_range=[2, 3], n_iterations=10,
        method='all',
    )
    assert len(erica.k_based_methods) == 5
    assert len(erica.auto_k_methods) == 1


def test_erica_init_both_deprecated():
    data = np.random.rand(50, 10)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        erica = ERICA(
            data=data, k_range=[2, 3], n_iterations=10,
            method='both',
        )
        dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(dep_warnings) >= 1
    assert set(erica.k_based_methods) == {
        'kmeans', 'agglomerative_single', 'agglomerative_ward'
    }


def test_erica_init_linkages_rejected():
    data = np.random.rand(50, 10)
    with pytest.raises(TypeError, match="linkages.*removed"):
        ERICA(
            data=data, k_range=[2, 3], n_iterations=10,
            method='kmeans', linkages=['ward'],
        )


def test_erica_init_hdbscan_params():
    data = np.random.rand(50, 10)
    erica = ERICA(
        data=data, k_range=[2, 3], n_iterations=10,
        method=['hdbscan'],
        hdbscan_params={'min_cluster_size': 10, 'min_samples': 3},
    )
    assert erica.hdbscan_params == {'min_cluster_size': 10, 'min_samples': 3}


def test_erica_run_agglomerative_list():
    data = np.random.rand(30, 5)
    erica = ERICA(
        data=data, k_range=[2], n_iterations=3,
        method=['agglomerative_ward'], verbose=False,
    )
    results = erica.run()
    assert 2 in results['metrics']
    assert 'agglomerative_ward' in results['metrics'][2]


def test_erica_run_hdbscan():
    from sklearn.datasets import make_blobs
    data, _ = make_blobs(n_samples=80, n_features=5, centers=3,
                         cluster_std=0.5, random_state=42)
    erica = ERICA(
        data=data, k_range=[2, 3], n_iterations=10,
        method=['kmeans', 'hdbscan'],
        hdbscan_params={'min_cluster_size': 5},
        transpose=False, verbose=False,
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
    from sklearn.datasets import make_blobs
    data, _ = make_blobs(n_samples=60, n_features=3, centers=2,
                         cluster_std=0.5, random_state=42)
    erica = ERICA(
        data=data, k_range=[2], n_iterations=5,
        method=['hdbscan'],
        hdbscan_params={'min_cluster_size': 5},
        transpose=False, verbose=False,
    )
    erica.run()
    result = erica.get_auto_k_results('hdbscan')
    assert result is not None
    assert 'modal_k' in result
    assert erica.get_auto_k_results('nonexistent') is None


def test_erica_full_integration_all_methods():
    from sklearn.datasets import make_blobs
    data, _ = make_blobs(n_samples=60, n_features=4, centers=3,
                         cluster_std=0.8, random_state=42)
    erica = ERICA(
        data=data, k_range=[2, 3], n_iterations=5,
        method=['kmeans', 'agglomerative_ward', 'hdbscan'],
        hdbscan_params={'min_cluster_size': 5},
        transpose=False, verbose=False,
    )
    results = erica.run()

    for k in [2, 3]:
        assert 'kmeans' in results['metrics'][k]
        assert 'agglomerative_ward' in results['metrics'][k]
        for method in ['kmeans', 'agglomerative_ward']:
            m = results['metrics'][k][method]
            assert 'CRI' in m
            assert 'TWCRI' in m

    assert 'TWCRI' in results['k_star']
    assert 'kmeans' in results['k_star']['TWCRI']
    assert 'agglomerative_ward' in results['k_star']['TWCRI']

    assert 'hdbscan' in results['auto_k']
    hdb = results['auto_k']['hdbscan']
    assert 'modal_k' in hdb
    assert 'k_distribution' in hdb
    assert 'clam_matrix' in hdb

    assert isinstance(results['config']['method'], list)
    assert 'hdbscan' in results['config']['method']


def test_erica_backwards_compat_both():
    data = np.random.rand(30, 5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        erica = ERICA(
            data=data, k_range=[2, 3], n_iterations=3,
            method='both', verbose=False,
        )
    results = erica.run()
    for k in [2, 3]:
        assert 'kmeans' in results['metrics'][k]
        assert 'agglomerative_single' in results['metrics'][k]
        assert 'agglomerative_ward' in results['metrics'][k]


if __name__ == "__main__":
    pytest.main([__file__])
