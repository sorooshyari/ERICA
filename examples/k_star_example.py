"""
Example demonstrating K* (optimal K) selection using ERICA Algorithm 2.

This example shows how to:
1. Run ERICA analysis with multiple K values
2. Automatically compute K* for each metric (CRI, WCRI, TWCRI)
3. Access and use the K* values
4. Visualize the K* selection (optional)
"""

import numpy as np
import sys
import os
# Add parent directory to path to ensure we import the local erica package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from erica import ERICA
from erica.metrics import select_optimal_k

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic data with 3 natural clusters
n_samples_per_cluster = 30
n_features = 10

# Create 3 distinct clusters
cluster1 = np.random.randn(n_samples_per_cluster, n_features) + np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
cluster2 = np.random.randn(n_samples_per_cluster, n_features) + np.array([5, 5, 5, 5, 5, 5, 5, 5, 5, 5])
cluster3 = np.random.randn(n_samples_per_cluster, n_features) + np.array([-5, -5, -5, -5, -5, -5, -5, -5, -5, -5])

# Combine clusters
data = np.vstack([cluster1, cluster2, cluster3])

print("="*70)
print("K* SELECTION EXAMPLE")
print("="*70)
print(f"\nDataset: {data.shape[0]} samples, {data.shape[1]} features")
print("True number of clusters: 3\n")

# Initialize ERICA with a range of K values
erica = ERICA(
    data=data,
    k_range=[2, 3, 4, 5, 6],
    n_iterations=100,
    method='kmeans',
    random_seed=42,
    verbose=True
)

# Run analysis
print("\nRunning ERICA analysis...")
results = erica.run()

# The K* values are automatically computed and displayed
print("\n" + "="*70)
print("ACCESSING K* VALUES PROGRAMMATICALLY")
print("="*70)

# Access K* for each metric
for metric_name in ['CRI', 'WCRI', 'TWCRI']:
    k_star_dict = erica.get_k_star(metric_name)
    print(f"\n{metric_name}:")
    for method, k_star in k_star_dict.items():
        # Get the metric value at K*
        metric_value = results['metrics'][k_star][method][metric_name]
        print(f"  Method: {method}")
        print(f"  K* = {k_star}")
        print(f"  {metric_name} at K* = {metric_value:.6f}")

# Demonstrate standalone K* selection
print("\n" + "="*70)
print("STANDALONE K* SELECTION")
print("="*70)

# Extract TWCRI values for kmeans
twcri_dict = {}
for k in erica.k_range:
    if k in results['metrics'] and 'kmeans' in results['metrics'][k]:
        twcri_dict[k] = results['metrics'][k]['kmeans']['TWCRI']

print(f"\nTWCRI values by K: {twcri_dict}")

# Select optimal K
k_star = select_optimal_k(twcri_dict)
print(f"Selected K* = {k_star}")
print(f"TWCRI at K* = {twcri_dict[k_star]:.6f}")

# Show all K values and their metrics for comparison
print("\n" + "="*70)
print("DETAILED METRICS COMPARISON")
print("="*70)
print(f"\n{'K':<5} {'CRI':<12} {'WCRI':<12} {'TWCRI':<12}")
print("-" * 45)
for k in sorted(results['metrics'].keys()):
    if 'kmeans' in results['metrics'][k]:
        metrics = results['metrics'][k]['kmeans']
        print(f"{k:<5} {metrics['CRI']:<12.6f} {metrics['WCRI']:<12.6f} {metrics['TWCRI']:<12.6f}")

# Optional: Create visualization if plotly is available
try:
    from erica.plotting import plot_k_star_selection
    
    print("\n" + "="*70)
    print("VISUALIZATION")
    print("="*70)
    print("\nGenerating K* selection plot...")
    
    # Get K* for TWCRI
    k_star_twcri = erica.get_k_star('TWCRI')['kmeans']
    
    # Create plot
    fig = plot_k_star_selection(
        metric_dict=twcri_dict,
        k_star=k_star_twcri,
        metric_name='TWCRI',
        method_name='kmeans'
    )
    
    # Save plot
    output_file = 'k_star_selection_plot.html'
    fig.write_html(output_file)
    print(f"Plot saved to: {output_file}")
    print("Open this file in a web browser to view the interactive plot.")
    
except ImportError:
    print("\nNote: Install plotly to generate visualizations:")
    print("  pip install plotly")

print("\n" + "="*70)
print("INTERPRETATION")
print("="*70)
print("""
Algorithm 2 (K* Selection) works by:
1. Starting with K_star = 2
2. For each subsequent K, if the metric is >= the previous valid metric, update K_star
3. This prefers larger K values when stability is maintained or improved

This approach differs from simply selecting the maximum metric value.
It identifies the most granular clustering that maintains replicability,
which is often more useful than the globally optimal metric value.
""")

