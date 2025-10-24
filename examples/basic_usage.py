"""Basic ERICA usage example.

This example demonstrates the simplest way to use ERICA for clustering
replicability analysis.
"""

import numpy as np
from erica import ERICA

def main():
    """Run basic ERICA analysis."""
    print("=" * 60)
    print("ERICA Basic Usage Example")
    print("=" * 60)
    
    # Generate sample data (100 samples, 50 features)
    print("\n1. Generating sample data...")
    np.random.seed(42)
    data = np.random.rand(100, 50)
    print(f"   Data shape: {data.shape}")
    
    # Create ERICA instance
    print("\n2. Setting up ERICA...")
    erica = ERICA(
        data=data,
        k_range=[2, 3, 4, 5],
        n_iterations=100,  # Using fewer iterations for quick demo
        method='kmeans',    # Using only K-Means for speed
        verbose=True
    )
    
    # Run analysis
    print("\n3. Running analysis...")
    results = erica.run()
    
    # Display results
    print("\n4. Results:")
    print("=" * 60)
    
    for k in [2, 3, 4, 5]:
        metrics = erica.get_metrics(k=k)
        if 'kmeans' in metrics:
            m = metrics['kmeans']
            print(f"\nK={k}:")
            print(f"  CRI:   {m['CRI']:.4f}")
            print(f"  WCRI:  {m['WCRI']:.4f}")
            print(f"  TWCRI: {m['TWCRI']:.4f}")
    
    # Find optimal k
    print("\n5. Finding optimal k...")
    from erica.metrics import find_optimal_k
    
    all_metrics = erica.get_metrics()
    optimal_k, twcri_value = find_optimal_k(all_metrics, metric_name='TWCRI')
    print(f"   Optimal k = {optimal_k} (TWCRI = {twcri_value:.4f})")
    
    # Save results
    print("\n6. Saving results...")
    erica.save_results('erica_results.json')
    print("   Results saved to: erica_results.json")
    
    # Save CLAM matrix
    from erica.data import save_clam_matrix
    clam = erica.get_clam_matrix(k=optimal_k, method='kmeans')
    save_clam_matrix(clam, f'clam_k{optimal_k}.csv', format='csv')
    print(f"   CLAM matrix saved to: clam_k{optimal_k}.csv")
    
    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()


