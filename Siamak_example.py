"""
Comprehensive examples for ERICA Clustering.
Updated to load data from vdx_dict.npy.
"""

import os
import numpy as np
from erica.core import ERICA

def load_data_matrix(filename: str, key: str = 'all') -> np.ndarray:
    """Load a matrix from a .npy file containing a dictionary."""
    data_dict = np.load(filename, allow_pickle=True).item()
    if key not in data_dict:
        raise ValueError(f"Key '{key}' not found in {filename}. Available keys: {list(data_dict.keys())}")
    matrix = data_dict[key]
    if isinstance(matrix, list):
        matrix = np.array(matrix)
    return matrix

def example_1_basic_usage():
    print("="*60)
    print("Example 1: Basic ERICA Analysis")
    print("="*60)

    data_matrix = load_data_matrix("vdx_dict.npy", key='all')    # Load the data from vdx_dict.npy
    print(f"Data shape: {data_matrix.shape}")

    # Initialize ERICA
    erica = ERICA(
        data=data_matrix,
        k_range=[2, 3, 4],   #Siamak commented
        #k_range=[2, 3, 4, 5, 6, 7, 8],   #Siamak added
        n_iterations=200,
        #method='kmeans',  #Siamak commented
        #method='both',  #Siamak added
        method='agglomerative',  #Siamak added
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

