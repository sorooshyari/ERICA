"""ERICA - Evaluating Replicability via Iterative Clustering Assignments

A comprehensive tool for analyzing clustering replicability using iterative 
clustering assignments to evaluate the stability and consistency of clustering 
results across different data subsamples.

Basic Usage:
    >>> from erica import ERICA
    >>> import numpy as np
    >>> 
    >>> # Load your data
    >>> data = np.random.rand(100, 50)  # 100 samples, 50 features
    >>> 
    >>> # Run ERICA analysis
    >>> erica = ERICA(
    ...     data=data,
    ...     k_range=[2, 3, 4, 5],
    ...     n_iterations=200,
    ...     method='both'
    ... )
    >>> erica.run()
    >>> 
    >>> # Get results
    >>> results = erica.get_results()
    >>> clam_matrix = erica.get_clam_matrix(k=3)

Advanced Usage:
    >>> from erica.clustering import kmeans_clustering, agglomerative_clustering
    >>> from erica.metrics import compute_cri, compute_wcri, compute_twcri
    >>> from erica.plotting import plot_metrics, plot_optimal_k
    >>> 
    >>> # Use individual components
    >>> results = kmeans_clustering(data, k=3, n_iterations=200)
    >>> metrics = compute_cri(results['clam_matrix'])
"""

__version__ = "0.1.0"
__author__ = "Siamak Sorooshyari, Shawn Shirazi"
__license__ = "MIT"

# Import main API
from erica.core import ERICA

# Import key clustering functions
from erica.clustering import (
    kmeans_clustering,
    agglomerative_clustering,
    iterative_clustering_subsampling,
)

# Import metrics functions
from erica.metrics import (
    compute_cri,
    compute_wcri,
    compute_twcri,
    compute_metrics_for_clam,
    select_optimal_k,
    select_optimal_k_by_method,
)

# Import plotting functions (if available)
try:
    from erica.plotting import (
        plot_metrics,
        plot_optimal_k,
        create_metrics_plots,
        plot_k_star_selection,
        plot_k_star_by_method,
    )
except ImportError:
    # Plotting dependencies not installed
    pass

# Import data utilities
from erica.data import (
    load_data,
    prepare_samples_array,
    validate_dataset,
)

# Import utilities
from erica.utils import (
    set_deterministic_mode,
    compute_config_hash,
)

__all__ = [
    # Main class
    'ERICA',
    
    # Clustering
    'kmeans_clustering',
    'agglomerative_clustering',
    'iterative_clustering_subsampling',
    
    # Metrics
    'compute_cri',
    'compute_wcri',
    'compute_twcri',
    'compute_metrics_for_clam',
    'select_optimal_k',
    'select_optimal_k_by_method',
    
    # Plotting (conditional)
    'plot_metrics',
    'plot_optimal_k',
    'create_metrics_plots',
    'plot_k_star_selection',
    'plot_k_star_by_method',
    
    # Data utilities
    'load_data',
    'prepare_samples_array',
    'validate_dataset',
    
    # Utilities
    'set_deterministic_mode',
    'compute_config_hash',
    
    # Version
    '__version__',
]
