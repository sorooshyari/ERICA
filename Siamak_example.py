"""
Comprehensive examples for ERICA Clustering.
Updated to load data from .npy or .csv files.
"""

import os
import numpy as np
import pandas as pd  # NEW: for reading CSV files
from erica.core import ERICA


def load_data_matrix(filename: str, key: str = 'all') -> np.ndarray:
    """
    Load a matrix from a .npy file containing a dictionary OR from a .csv file.

    Parameters
    ----------
    filename : str
        Path to the .npy or .csv file.
    key : str, optional
        Key to select from .npy dictionary file (ignored for CSV files).

    Returns
    -------
    np.ndarray
        The data matrix.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")

    ext = os.path.splitext(filename)[1].lower()

    if ext == '.npy':
        # Load from .npy dictionary
        data_dict = np.load(filename, allow_pickle=True).item()
        if key not in data_dict:
            raise ValueError(f"Key '{key}' not found in {filename}. Available keys: {list(data_dict.keys())}")
        matrix = data_dict[key]
        if isinstance(matrix, list):
            matrix = np.array(matrix)
        return matrix

    elif ext == '.csv':
        # Load from CSV
        df = pd.read_csv(filename, header=None)
        matrix = df.to_numpy(dtype=float)
        return matrix

    else:
        raise ValueError(f"Unsupported file type: {ext}. Supported: .npy, .csv")


def example_1_basic_usage():
    print("=" * 60)
    print("Example 1: Basic ERICA Analysis")
    print("=" * 60)

    # Modify the filename as needed
    filename = "vdx_dict.npy"  # or "data.csv"      #Siamak commented out
    #filename = "VDX_3_SV.csv"  # or "data.csv"       #Siamak added
    #filename = "LiuWeissman_2023_DE_then_PCA.csv"  # or "data.csv"       #Siamak added
    data_matrix = load_data_matrix(filename, key='all')

    print(f"Data shape: {data_matrix.shape}")

    # Initialize ERICA
    erica = ERICA(
        data=data_matrix,
        k_range=[2, 3, 4, 5, 6, 7, 8],
        n_iterations=200,
        method='both',
        random_seed=123,
        verbose=True
    )

    # Run ERICA
    results = erica.run()

    print("\nExample 1 completed successfully!")
    print("Metrics:")
    print(results['metrics'])

    return erica, results['metrics']


def main():
    try:
        erica1, metrics1 = example_1_basic_usage()
    except Exception as e:
        print("\nError occurred:", e)


if __name__ == "__main__":
    main()
