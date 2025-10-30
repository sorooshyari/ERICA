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
        samples_array = prepare_samples_array(data, transpose='auto')
        
        # Should have 50 samples and 3 features
        assert samples_array.shape[0] == 50
        assert samples_array.shape[1] == 3
    finally:
        os.unlink(temp_file)


def test_csv_loading_features_in_rows():
    """Test CSV loading with features in rows (genomics format)."""
    # Create temporary CSV with features in rows (more columns than rows)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        # Create 50 features x 100 samples (will auto-transpose)
        f.write('gene_id,' + ','.join([f'sample{i}' for i in range(100)]) + '\n')
        for i in range(50):
            values = ','.join([str(np.random.rand()) for _ in range(100)])
            f.write(f'gene_{i},{values}\n')
        temp_file = f.name
    
    try:
        # Load and prepare data
        data = load_data(temp_file)
        samples_array = prepare_samples_array(data, transpose='auto')
        
        # Should have 100 samples and 50 features (auto-transposed)
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
        
        # Test transpose='no'
        samples_no = prepare_samples_array(data, transpose='no')
        assert samples_no.shape == (20, 3)
        
        # Test transpose='yes'
        samples_yes = prepare_samples_array(data, transpose='yes')
        assert samples_yes.shape == (3, 20)
        
        # Test transpose='auto' (should detect samples in rows)
        samples_auto = prepare_samples_array(data, transpose='auto')
        assert samples_auto.shape == (20, 3)
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
            transpose='auto'
        )
        
        results = erica.run()
        
        # Check results
        assert results is not None
        assert erica.n_samples == 30
        assert erica.n_features == 3
    finally:
        os.unlink(temp_file)


if __name__ == "__main__":
    pytest.main([__file__])
