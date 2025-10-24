"""Utility functions for ERICA.

This module provides helper functions for:
- Deterministic mode setup
- Configuration hashing
- Logging
"""

import os
import random
import numpy as np
import json
from hashlib import sha256
from typing import Dict, Any


def set_deterministic_mode(seed: int) -> None:
    """Set deterministic mode for reproducible results.
    
    This function configures:
    - Python random seed
    - NumPy random seed
    - Threading environment variables for deterministic linear algebra
    
    Parameters
    ----------
    seed : int
        Random seed value
        
    Examples
    --------
    >>> set_deterministic_mode(123)
    >>> # All subsequent random operations will be reproducible
    
    Notes
    -----
    Setting deterministic mode ensures that:
    1. Random subsampling is reproducible
    2. Clustering results are identical across runs
    3. Linear algebra operations are deterministic
    
    This is crucial for scientific reproducibility and debugging.
    """
    try:
        # Set environment variables for deterministic multithreading
        os.environ.setdefault("PYTHONHASHSEED", str(seed))
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
        os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    except Exception:
        pass
    
    # Set random seeds
    random.seed(seed)
    np.random.seed(seed)


def compute_config_hash(config: Dict[str, Any]) -> str:
    """Compute hash of configuration for reproducibility tracking.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary
        
    Returns
    -------
    str
        12-character hash string
        
    Examples
    --------
    >>> config = {'k_range': [2, 3, 4], 'n_iterations': 200, 'random_seed': 123}
    >>> hash_str = compute_config_hash(config)
    >>> len(hash_str)
    12
    
    Notes
    -----
    The hash is based on key parameters that affect results:
    - k range
    - Number of iterations
    - Random seed
    - Clustering method
    - Train/test split percentage
    
    This hash can be used to uniquely identify analysis runs
    and verify reproducibility.
    """
    # Create minimal config for hashing (only parameters that affect results)
    minimal_config = {
        'k_range': config.get('k_range'),
        'n_iterations': config.get('n_iterations'),
        'train_percent': config.get('train_percent'),
        'random_seed': config.get('random_seed'),
        'method': config.get('method'),
        'linkages': config.get('linkages'),
    }
    
    # Convert to sorted JSON string
    config_str = json.dumps(minimal_config, sort_keys=True, default=str)
    
    # Compute SHA256 hash and return first 12 characters
    hash_obj = sha256(config_str.encode('utf-8'))
    return hash_obj.hexdigest()[:12]


def format_time_estimate(seconds: float) -> str:
    """Format time in seconds to human-readable string.
    
    Parameters
    ----------
    seconds : float
        Time in seconds
        
    Returns
    -------
    str
        Formatted time string
        
    Examples
    --------
    >>> format_time_estimate(45)
    '45s'
    >>> format_time_estimate(125)
    '2m 5s'
    >>> format_time_estimate(3665)
    '1h 1m'
    """
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def get_version_info() -> Dict[str, str]:
    """Get version information for ERICA and dependencies.
    
    Returns
    -------
    dict
        Dictionary with version strings
        
    Examples
    --------
    >>> info = get_version_info()
    >>> print(info['erica'])
    """
    import erica
    
    versions = {
        'erica': erica.__version__
    }
    
    try:
        import sklearn
        versions['scikit-learn'] = sklearn.__version__
    except ImportError:
        versions['scikit-learn'] = 'not installed'
    
    try:
        import numpy
        versions['numpy'] = numpy.__version__
    except ImportError:
        versions['numpy'] = 'not installed'
    
    try:
        import pandas
        versions['pandas'] = pandas.__version__
    except ImportError:
        versions['pandas'] = 'not installed'
    
    try:
        import plotly
        versions['plotly'] = plotly.__version__
    except ImportError:
        versions['plotly'] = 'not installed'
    
    return versions


def check_dependencies() -> Dict[str, bool]:
    """Check if all required and optional dependencies are available.
    
    Returns
    -------
    dict
        Dictionary mapping dependency names to availability (True/False)
        
    Examples
    --------
    >>> deps = check_dependencies()
    >>> if not deps['plotly']:
    ...     print("Install plotly for plotting: pip install erica-clustering[plots]")
    """
    dependencies = {}
    
    # Required dependencies
    required = ['numpy', 'pandas', 'sklearn']
    for dep in required:
        try:
            __import__(dep)
            dependencies[dep] = True
        except ImportError:
            dependencies[dep] = False
    
    # Optional dependencies
    #optional = ['plotly', 'matplotlib', 'gradio']
    for dep in optional:
        try:
            __import__(dep)
            dependencies[dep] = True
        except ImportError:
            dependencies[dep] = False
    
    return dependencies


def validate_config(config: Dict[str, Any]) -> None:
    """Validate configuration parameters.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary to validate
        
    Raises
    ------
    ValueError
        If any configuration parameter is invalid
        
    Examples
    --------
    >>> config = {
    ...     'k_range': [2, 3, 4],
    ...     'n_iterations': 200,
    ...     'train_percent': 0.8,
    ...     'method': 'both'
    ... }
    >>> validate_config(config)
    """
    # Validate k_range
    if 'k_range' in config:
        k_range = config['k_range']
        if not isinstance(k_range, list) or len(k_range) == 0:
            raise ValueError("k_range must be a non-empty list")
        if any(k < 2 for k in k_range):
            raise ValueError("All k values must be >= 2")
    
    # Validate n_iterations
    if 'n_iterations' in config:
        if config['n_iterations'] < 1:
            raise ValueError("n_iterations must be >= 1")
    
    # Validate train_percent
    if 'train_percent' in config:
        if not (0 < config['train_percent'] < 1):
            raise ValueError("train_percent must be between 0 and 1")
    
    # Validate method
    if 'method' in config:
        valid_methods = ['kmeans', 'agglomerative', 'both']
        if config['method'] not in valid_methods:
            raise ValueError(f"method must be one of {valid_methods}")
    
    # Validate linkages
    if 'linkages' in config:
        valid_linkages = ['single', 'complete', 'average', 'ward']
        linkages = config['linkages']
        if not isinstance(linkages, list):
            raise ValueError("linkages must be a list")
        for linkage in linkages:
            if linkage not in valid_linkages:
                raise ValueError(f"Invalid linkage '{linkage}'. Must be one of {valid_linkages}")


def create_run_directory(base_dir: str, prefix: str = 'erica_run') -> str:
    """Create a timestamped run directory.
    
    Parameters
    ----------
    base_dir : str
        Base directory path
    prefix : str, optional
        Prefix for run directory name, default 'erica_run'
        
    Returns
    -------
    str
        Path to created directory
        
    Examples
    --------
    >>> run_dir = create_run_directory('./output')
    >>> print(run_dir)  # doctest: +SKIP
    ./output/erica_run_20250103_143052
    """
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, f"{prefix}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    return run_dir


def save_config(config: Dict[str, Any], filepath: str) -> None:
    """Save configuration to JSON file.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary
    filepath : str
        Output file path
        
    Examples
    --------
    >>> config = {'k_range': [2, 3, 4], 'n_iterations': 200}
    >>> save_config(config, 'config.json')
    """
    os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=2, default=str)


def load_config(filepath: str) -> Dict[str, Any]:
    """Load configuration from JSON file.
    
    Parameters
    ----------
    filepath : str
        Configuration file path
        
    Returns
    -------
    dict
        Loaded configuration
        
    Examples
    --------
    >>> config = load_config('config.json')
    """
    with open(filepath, 'r') as f:
        return json.load(f)


