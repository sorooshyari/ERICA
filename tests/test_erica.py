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
        method='both',
        linkages=['ward'],
        verbose=False
    )
    
    results = erica.run()
    
    # Check K* for both methods
    k_star_twcri = results['k_star']['TWCRI']
    assert 'kmeans' in k_star_twcri
    assert 'agglomerative_ward' in k_star_twcri


if __name__ == "__main__":
    pytest.main([__file__])
