"""
Basic tests for ERICA package.
"""

import pytest
import numpy as np
from erica import ERICA


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


if __name__ == "__main__":
    pytest.main([__file__])
