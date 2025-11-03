"""Data loading and preprocessing utilities for ERICA.

This module handles:
- Loading data from various formats (.npy, .csv, DataFrame)
- Preparing samples arrays for clustering
- Data validation
"""

import os
import numpy as np
import pandas as pd
from typing import Union, Dict, Tuple, Optional


def load_data(filepath: str) -> Union[np.ndarray, pd.DataFrame]:
    """Load data from file.
    
    Supports .npy and .csv files with automatic type detection and conversion.
    
    Parameters
    ----------
    filepath : str
        Path to data file (.npy or .csv)
        
    Returns
    -------
    np.ndarray or pd.DataFrame
        Loaded data
        
    Raises
    ------
    FileNotFoundError
        If file doesn't exist
    ValueError
        If file format is unsupported
        
    Examples
    --------
    >>> data = load_data('my_data.npy')
    >>> data = load_data('my_data.csv')
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    ext = os.path.splitext(filepath)[1].lower()
    
    if ext == '.npy':
        return _load_npy(filepath)
    elif ext == '.csv':
        return _load_csv(filepath)
    else:
        raise ValueError(f"Unsupported file format: {ext}")


def _load_npy(filepath: str) -> Union[np.ndarray, pd.DataFrame]:
    """Load .npy file."""
    loaded = np.load(filepath, allow_pickle=True)
    
    # Handle 0-d array containing a dict (common format)
    if loaded.shape == () and loaded.dtype == object:
        loaded = loaded.item()
    
    # Handle different .npy formats
    if isinstance(loaded, dict):
        if 'all' in loaded:
            return loaded['all']
        else:
            # Convert dict to DataFrame
            return pd.DataFrame(loaded)
    else:
        return loaded


def _load_csv(filepath: str) -> pd.DataFrame:
    """Load .csv file with intelligent type inference."""
    try:
        # Try loading with header detection
        df = pd.read_csv(filepath, low_memory=False)
    except Exception:
        # Fallback: load without headers
        df = pd.read_csv(filepath, header=None, low_memory=False)
    
    # Try to convert string columns to numeric
    for col in df.columns:
        if df[col].dtype == 'object':
            numeric_series = pd.to_numeric(df[col], errors='coerce')
            # If most values converted successfully, use numeric version
            if numeric_series.notna().sum() / len(numeric_series) > 0.8:
                df[col] = numeric_series
    
    return df


def prepare_samples_array(
    data: Union[np.ndarray, pd.DataFrame],
    transpose: bool = True
) -> np.ndarray:
    """Convert data to numeric samples array suitable for clustering.
    
    This function:
    1. Detects and removes header rows
    2. Removes non-numeric first columns (gene IDs, etc.)
    3. Converts to numeric where possible
    4. Optionally transposes data if features are in rows
    
    Parameters
    ----------
    data : np.ndarray or pd.DataFrame
        Input data
    transpose : bool, optional
        Whether to transpose the data (default: True)
        - True: Assumes features in rows, samples in columns (genomics format - default)
        - False: Assumes samples in rows, features in columns (standard ML format)
        
    Returns
    -------
    np.ndarray
        Numeric array with shape (n_samples, n_features)
        
    Raises
    ------
    ValueError
        If no numeric data can be extracted
        
    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'sample_id': ['sample1', 'sample2'],
    ...     'feature1': [1.0, 2.0],
    ...     'feature2': [3.0, 4.0]
    ... })
    >>> array = prepare_samples_array(df, transpose=False)
    >>> print(array.shape)  # (2, 2) - 2 samples, 2 features
    (2, 2)
    """
    # If already numpy array, check if numeric
    if isinstance(data, np.ndarray):
        if np.issubdtype(data.dtype, np.number):
            return data
        else:
            # Try converting to DataFrame for processing
            data = pd.DataFrame(data)
    
    # Process DataFrame
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Data must be numpy array or pandas DataFrame")
    
    # Check if first row looks like a header
    if not data.empty and data.shape[1] > 1:
        first_row_values = data.iloc[0, 1:]
        non_numeric_count = pd.to_numeric(first_row_values, errors='coerce').isna().sum()
        if non_numeric_count > len(first_row_values) / 2:
            # First row is likely a header, skip it
            data = data.iloc[1:].reset_index(drop=True)
    
    # Remove non-numeric first column (likely gene IDs or sample names)
    if not data.empty and not np.issubdtype(data.iloc[:, 0].dtype, np.number):
        data = data.iloc[:, 1:]
    
    # Convert all columns to numeric where possible
    converted_df = data.copy()
    for col in data.columns:
        if not np.issubdtype(data[col].dtype, np.number):
            try:
                numeric_series = pd.to_numeric(data[col], errors='coerce')
                # Keep conversion if at least 50% of values are valid
                if numeric_series.isna().sum() / len(numeric_series) <= 0.5:
                    converted_df[col] = numeric_series
            except Exception:
                pass
    
    # Select only numeric columns
    numeric_df = converted_df.select_dtypes(include=[np.number])
    
    if numeric_df.empty:
        raise ValueError(
            f"No numeric data found in input. "
            f"Original shape: {data.shape}, "
            f"After processing: {numeric_df.shape}"
        )
    
    # Apply transposition if requested
    if transpose:
        samples_array = numeric_df.values.T
    else:
        samples_array = numeric_df.values
    
    if samples_array.shape[0] == 0:
        raise ValueError(
            f"Final array has 0 samples after processing. "
            f"Original DataFrame shape: {data.shape}, "
            f"Numeric DataFrame shape: {numeric_df.shape}, "
            f"Transposed: {transpose}"
        )
    
    return samples_array


def validate_dataset(
    samples_array: np.ndarray,
    min_k: int,
    train_percent: float
) -> None:
    """Validate dataset meets minimum requirements for clustering.
    
    Parameters
    ----------
    samples_array : np.ndarray
        Data array with shape (n_samples, n_features)
    min_k : int
        Minimum number of clusters to test
    train_percent : float
        Training data percentage
        
    Raises
    ------
    ValueError
        If dataset doesn't meet requirements
        
    Examples
    --------
    >>> data = np.random.rand(100, 50)
    >>> validate_dataset(data, min_k=2, train_percent=0.8)
    """
    n_samples, n_features = samples_array.shape
    
    # Check minimum samples
    if n_samples < 3:
        hint = ""
        if n_features > 10 and n_features > n_samples:
            hint = (
                f"\n\nHINT: Your data has {n_features} features but only {n_samples} samples. "
                f"This might indicate incorrect data orientation. "
                f"Try toggling the transpose parameter (True/False) when initializing ERICA."
            )
        raise ValueError(
            f"Dataset has only {n_samples} samples. "
            f"Need at least 3 samples for meaningful clustering.{hint}"
        )
    
    # Check samples vs k
    if n_samples < min_k:
        hint = ""
        if n_features > n_samples and n_features >= min_k:
            hint = (
                f"\n\nHINT: Your data has {n_samples} samples but {n_features} features. "
                f"This might indicate incorrect data orientation. "
                f"Try toggling the transpose parameter (True/False) when initializing ERICA."
            )
        raise ValueError(
            f"Dataset has {n_samples} samples but k={min_k} clusters requested. "
            f"Need at least k samples.{hint}"
        )
    
    # Check training subset size
    train_size = int(n_samples * train_percent)
    test_size = n_samples - train_size
    
    if train_size < min_k:
        raise ValueError(
            f"Training subset size ({train_size}) is smaller than "
            f"number of clusters ({min_k}). "
            f"Try reducing k or increasing training percentage."
        )
    
    if test_size < 1:
        raise ValueError(
            f"Test subset size ({test_size}) is too small. "
            f"Try reducing the training percentage."
        )
    
    # Check for NaN or Inf values
    if np.any(np.isnan(samples_array)):
        raise ValueError("Dataset contains NaN values. Please clean your data.")
    
    if np.any(np.isinf(samples_array)):
        raise ValueError("Dataset contains Inf values. Please clean your data.")
    
    # Check variance
    if np.all(samples_array == samples_array[0, :]):
        raise ValueError("Dataset has zero variance. All samples are identical.")


def save_clam_matrix(
    clam_matrix: np.ndarray,
    filepath: str,
    format: str = 'csv'
) -> None:
    """Save CLAM matrix to file.
    
    Parameters
    ----------
    clam_matrix : np.ndarray
        CLAM matrix to save
    filepath : str
        Output file path
    format : str, optional
        Output format ('csv' or 'npy'), default 'csv'
        
    Examples
    --------
    >>> clam = np.array([[50, 10], [45, 15]])
    >>> save_clam_matrix(clam, 'clam_k2.csv')
    """
    os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
    
    if format == 'csv':
        pd.DataFrame(clam_matrix).to_csv(filepath, index=False, header=False)
    elif format == 'npy':
        np.save(filepath, clam_matrix)
    else:
        raise ValueError(f"Unsupported format: {format}")


def load_clam_matrix(filepath: str) -> np.ndarray:
    """Load CLAM matrix from file.
    
    Parameters
    ----------
    filepath : str
        Path to CLAM matrix file
        
    Returns
    -------
    np.ndarray
        Loaded CLAM matrix
        
    Examples
    --------
    >>> clam = load_clam_matrix('clam_k2.csv')
    """
    ext = os.path.splitext(filepath)[1].lower()
    
    if ext == '.csv':
        return np.loadtxt(filepath, delimiter=',')
    elif ext == '.npy':
        return np.load(filepath)
    else:
        raise ValueError(f"Unsupported file format: {ext}")


def get_dataset_info(
    data: Union[np.ndarray, pd.DataFrame],
    transpose: bool = True
) -> Dict:
    """Get summary information about a dataset.
    
    Parameters
    ----------
    data : np.ndarray or pd.DataFrame
        Input data
    transpose : bool, optional
        Whether to transpose the data (default: True)
        
    Returns
    -------
    dict
        Dictionary with dataset information
        
    Examples
    --------
    >>> data = np.random.rand(100, 50)
    >>> info = get_dataset_info(data)
    >>> print(info['n_samples'], info['n_features'])
    100 50
    """
    samples_array = prepare_samples_array(data, transpose=transpose)
    n_samples, n_features = samples_array.shape
    
    return {
        'n_samples': n_samples,
        'n_features': n_features,
        'shape': samples_array.shape,
        'dtype': str(samples_array.dtype),
        'has_nan': bool(np.any(np.isnan(samples_array))),
        'has_inf': bool(np.any(np.isinf(samples_array))),
        'min_value': float(np.min(samples_array)),
        'max_value': float(np.max(samples_array)),
        'mean_value': float(np.mean(samples_array)),
        'std_value': float(np.std(samples_array))
    }


