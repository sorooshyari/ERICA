"""Advanced ERICA usage example.

This example demonstrates using individual ERICA components for
custom workflows and advanced analysis.
"""

import numpy as np
from erica.clustering import iterative_clustering_subsampling, kmeans_clustering
from erica.metrics import compute_metrics_for_clam, find_optimal_k
from erica.data import prepare_samples_array, validate_dataset, save_clam_matrix
from erica.utils import set_deterministic_mode, compute_config_hash
from erica.plotting import plot_metrics, plot_clam_heatmap


def main():
    """Run advanced ERICA analysis with individual components."""
    print("=" * 60)
    print("ERICA Advanced Usage Example")
    print("=" * 60)
    
    # Set deterministic mode
    print("\n1. Setting up deterministic mode...")
    set_deterministic_mode(42)
    print("   Random seed set to 42 for reproducibility")
    
    # Generate and prepare data
    print("\n2. Preparing data...")
    data = np.random.rand(100, 50)
    samples = prepare_samples_array(data)
    print(f"   Data shape: {samples.shape}")
    
    # Validate dataset
    print("\n3. Validating dataset...")
    try:
        validate_dataset(samples, min_k=2, train_percent=0.8)
        print("   ✓ Dataset validation passed")
    except ValueError as e:
        print(f"   ✗ Dataset validation failed: {e}")
        return
    
    # Configuration
    config = {
        'k_range': [2, 3, 4],
        'n_iterations': 100,
        'train_percent': 0.8,
        'random_seed': 42,
        'method': 'kmeans'
    }
    
    config_hash = compute_config_hash(config)
    print(f"\n4. Configuration hash: {config_hash}")
    
    # Step-by-step analysis
    n_samples = len(samples)
    train_size = int(n_samples * config['train_percent'])
    
    # Perform subsampling
    print(f"\n5. Performing iterative subsampling...")
    print(f"   Iterations: {config['n_iterations']}")
    print(f"   Train size: {train_size}, Test size: {n_samples - train_size}")
    
    subsamples_folder, indices_folder = iterative_clustering_subsampling(
        samples_array=samples,
        num_samples=n_samples,
        num_iterations=config['n_iterations'],
        subsample_size_train=train_size,
        base_save_folder_str='./advanced_output',
        optimize_io=True,
        verbose=True
    )
    
    # Run clustering for each k
    print(f"\n6. Running K-Means clustering for each k...")
    results = {}
    
    for k in config['k_range']:
        print(f"\n   Processing k={k}...")
        result = kmeans_clustering(
            samples_array=samples,
            k=k,
            n_iterations=config['n_iterations'],
            indices_folder=indices_folder,
            output_dir='./advanced_output',
            random_state=42,
            verbose=False
        )
        results[k] = result
        print(f"   ✓ Completed k={k}")
    
    # Compute metrics for all k values
    print(f"\n7. Computing metrics...")
    metrics_by_k = {}
    
    for k, result in results.items():
        clam_matrix = result['clam_matrix']
        metrics = compute_metrics_for_clam(clam_matrix, k)
        metrics_by_k[k] = metrics
        
        print(f"\n   K={k}:")
        print(f"     CRI:   {metrics['CRI']:.4f}")
        print(f"     WCRI:  {metrics['WCRI']:.4f}")
        print(f"     TWCRI: {metrics['TWCRI']:.4f}")
        print(f"     Cluster sizes: {metrics['cluster_sizes']}")
    
    # Find optimal k
    print(f"\n8. Finding optimal k...")
    optimal_k, twcri_value = find_optimal_k(metrics_by_k, metric_name='TWCRI')
    print(f"   Optimal k = {optimal_k}")
    print(f"   TWCRI = {twcri_value:.4f}")
    
    # Create visualizations
    print(f"\n9. Creating visualizations...")
    
    # Prepare data for plotting
    k_values = list(config['k_range'])
    cri_values = [metrics_by_k[k]['CRI'] for k in k_values]
    wcri_values = [metrics_by_k[k]['WCRI'] for k in k_values]
    twcri_values = [metrics_by_k[k]['TWCRI'] for k in k_values]
    
    # Plot metrics
    try:
        fig_metrics = plot_metrics(k_values, cri_values, wcri_values, twcri_values)
        fig_metrics.write_html('advanced_metrics_plot.html')
        print("   ✓ Metrics plot saved to: advanced_metrics_plot.html")
        
        # Plot CLAM heatmap for optimal k
        clam_optimal = results[optimal_k]['clam_matrix']
        fig_clam = plot_clam_heatmap(clam_optimal, k=optimal_k)
        fig_clam.write_html(f'advanced_clam_k{optimal_k}.html')
        print(f"   ✓ CLAM heatmap saved to: advanced_clam_k{optimal_k}.html")
    except ImportError:
        print("   ⚠ Plotly not installed, skipping visualization")
        print("     Install with: pip install erica-clustering[plots]")
    
    # Save CLAM matrices
    print(f"\n10. Saving CLAM matrices...")
    for k, result in results.items():
        clam_matrix = result['clam_matrix']
        save_clam_matrix(clam_matrix, f'advanced_clam_k{k}.csv', format='csv')
        save_clam_matrix(clam_matrix, f'advanced_clam_k{k}.npy', format='npy')
        print(f"   ✓ Saved CLAM matrices for k={k}")
    
    # Summary
    print("\n" + "=" * 60)
    print("Advanced Analysis Summary")
    print("=" * 60)
    print(f"Configuration hash: {config_hash}")
    print(f"Analyzed k values: {config['k_range']}")
    print(f"Optimal k: {optimal_k} (TWCRI = {twcri_value:.4f})")
    print(f"Output directory: ./advanced_output")
    print("=" * 60)
    
    return results, metrics_by_k, optimal_k


if __name__ == "__main__":
    results, metrics, optimal_k = main()


