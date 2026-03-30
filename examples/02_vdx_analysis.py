"""
VDX Breast Cancer Dataset Analysis Example

This example demonstrates how to use ERICA to analyze real-world gene expression data.
It automatically downloads the required VDX dataset from the ERICA repository if missing.
"""


import sys
import os
# Add parent directory to path to ensure we import the local erica package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from erica import ERICA
from erica.data import check_and_download_example_data, load_data

def main():
    print("=" * 60)
    print("ERICA VDX Breast Cancer Analysis Example")
    print("=" * 60)
    
    # 1. Ensure data is available (will prompt to download if missing)
    print("\n1. Checking for VDX dataset...")
    try:
        # Check main dataset (full genes)
        filename = check_and_download_example_data("vdx_dict.npy")
    except FileNotFoundError:
        print("Analysis cannot proceed without data.")
        return

    # 2. Load the data
    print(f"\n2. Loading data from {filename}...")
    # vdx_dict.npy contains genes in rows, samples in columns
    # load_data handles .npy dictionary structure automatically
    data = load_data(filename)
    
    # 3. Initialize ERICA
    # Note: transpose=True is default, which is correct for genomic data (genes x samples)
    print("\n3. Initializing ERICA...")
    erica = ERICA(
        data=data,
        k_range=[2, 3, 4, 5, 6, 7],  # Test K=2 to K=7
        n_iterations=200,            # 200 iterations for stability
        method='both',               # K-Means + Agglomerative (Single & Ward linkage)
        random_seed=123,             # Deterministic for reproducibility
        verbose=True
    )
    
    # 4. Run Analysis
    print("\n4. Running analysis (this may take a few minutes)...")
    results = erica.run()
    
    # 5. Display Results
    print("\n5. Analysis Complete. Results:")
    print("=" * 60)
    
    # Get recommended K* values
    k_star = erica.get_k_star('TWCRI')
    print("\nOptimal K* Recommendations (Algorithm 2):")
    for method, k in k_star.items():
        print(f"  - {method}: K* = {k}")
    
    # Show detailed metrics for the recommended K
    best_method = 'kmeans'
    if best_method in k_star:
        best_k = k_star[best_method]
        metrics = erica.get_metrics(k=best_k)[best_method]
        
        print(f"\nDetailed Metrics for {best_method} at K={best_k}:")
        print(f"  CRI:   {metrics['CRI']:.4f}")
        print(f"  WCRI:  {metrics['WCRI']:.4f}")
        print(f"  TWCRI: {metrics['TWCRI']:.4f}")
    
    print("\n" + "=" * 60)
    print("Example complete!")

if __name__ == "__main__":
    main()
