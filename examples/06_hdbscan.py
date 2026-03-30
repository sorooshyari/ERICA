"""
Example 06: HDBSCAN Auto-K Clustering with ERICA

Demonstrates how to use HDBSCAN as an auto-K clustering method alongside
traditional K-based methods. HDBSCAN discovers the number of clusters
automatically, while ERICA evaluates replicability across subsamples.
"""

import numpy as np
from sklearn.datasets import make_blobs
from erica import ERICA

# Generate synthetic data with 3 clusters
data, true_labels = make_blobs(
    n_samples=200, n_features=5, centers=3,
    cluster_std=1.0, random_state=42
)

print("=" * 60)
print("ERICA with HDBSCAN Auto-K Clustering")
print("=" * 60)

# Run ERICA with both K-Means and HDBSCAN
erica = ERICA(
    data=data,
    k_range=[2, 3, 4, 5],
    n_iterations=50,
    method=['kmeans', 'hdbscan'],
    hdbscan_params={'min_cluster_size': 10},
    transpose=False,
    verbose=True,
)

results = erica.run()

# --- K-based results ---
print("\n--- K-Based Results (K-Means) ---")
k_star = erica.get_k_star('TWCRI')
print(f"Optimal K* (TWCRI): {k_star.get('kmeans', 'N/A')}")

for k in [2, 3, 4, 5]:
    m = results['metrics'][k]['kmeans']
    print(f"  K={k}: TWCRI={m['TWCRI']:.4f}, "
          f"ARI={m['ARI_mean']:.4f} +/- {m['ARI_std']:.4f}")

# --- Auto-K results ---
print("\n--- Auto-K Results (HDBSCAN) ---")
hdb = results['auto_k']['hdbscan']
print(f"Modal K: {hdb['modal_k']}")
print(f"K distribution: {hdb['k_distribution']}")
print(f"K agreement rate: {hdb['k_agreement_rate']:.2f}")
print(f"Iterations used for CLAM: {hdb['n_iterations_used']}")

if 'ARI_mean' in hdb:
    print(f"ARI: {hdb['ARI_mean']:.4f} +/- {hdb['ARI_std']:.4f}")
    print(f"AMI: {hdb['AMI_mean']:.4f} +/- {hdb['AMI_std']:.4f}")

if 'metrics_at_modal_k' in hdb:
    mk = hdb['metrics_at_modal_k']
    print(f"TWCRI at modal K: {mk.get('TWCRI', 'N/A')}")

print(f"\nNoise points per iteration (mean): "
      f"{np.mean(hdb['noise_counts']):.1f}")
