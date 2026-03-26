"""Example: Computing Parmigiani Metrics (ARI and AMI)

This example demonstrates how to compute the Adjusted Rand Index (ARI) and
Adjusted Mutual Information (AMI) metrics from Parmigiani et al. (2023)
"Cross-Study Replicability in Cluster Analysis".

These metrics measure clustering replicability by comparing:
- predicted_labels: What a train-fitted model predicts for test samples
- true_labels: What a fresh model fitted on test samples produces

High ARI/AMI indicates that clustering structure transfers well from
training to test data.

Reference:
    Parmigiani et al. "Cross-Study Replicability in Cluster Analysis"
    Statistical Science, 38(2): 303-316 (May 2023)
    DOI: 10.1214/22-STS871
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

# Import Parmigiani metrics from ERICA
from erica.metrics import (
    compute_ari,
    compute_ami,
    compute_parmigiani_metrics,
    aggregate_parmigiani_metrics,
)


def main():
    # =========================================================================
    # Generate synthetic data with clear cluster structure
    # =========================================================================
    print("=" * 60)
    print("Parmigiani Metrics Example")
    print("=" * 60)

    np.random.seed(42)

    # Create 3 well-separated clusters
    n_samples_per_cluster = 100
    n_features = 10

    # Cluster centers
    centers = np.array([
        [0] * n_features,
        [5] * n_features,
        [10] * n_features,
    ])

    # Generate samples around each center
    data = np.vstack([
        np.random.randn(n_samples_per_cluster, n_features) + centers[0],
        np.random.randn(n_samples_per_cluster, n_features) + centers[1],
        np.random.randn(n_samples_per_cluster, n_features) + centers[2],
    ])

    print(f"\nData shape: {data.shape}")
    print(f"Number of clusters: 3")

    # =========================================================================
    # Single iteration: Parmigiani Algorithm 1
    # =========================================================================
    print("\n" + "-" * 60)
    print("Single Iteration (Algorithm 1)")
    print("-" * 60)

    # Step 1: Split data into train and test sets
    train_data, test_data = train_test_split(
        data, train_size=0.8, random_state=42
    )
    print(f"\nTrain samples: {len(train_data)}")
    print(f"Test samples: {len(test_data)}")

    # Step 2: Fit clustering model on training data
    k = 3
    kmeans_train = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans_train.fit(train_data)

    # Step 3: Predict labels for test data using train-fitted model
    predicted_labels = kmeans_train.predict(test_data)

    # Step 4: Fit fresh model directly on test data
    kmeans_test = KMeans(n_clusters=k, random_state=42, n_init=10)
    true_labels = kmeans_test.fit_predict(test_data)

    # Step 5: Compute ARI and AMI
    ari = compute_ari(predicted_labels, true_labels)
    ami = compute_ami(predicted_labels, true_labels)

    print(f"\nResults:")
    print(f"  ARI: {ari:.4f}")
    print(f"  AMI: {ami:.4f}")

    # Or use the convenience function
    metrics = compute_parmigiani_metrics(predicted_labels, true_labels)
    print(f"\nUsing compute_parmigiani_metrics():")
    print(f"  ARI: {metrics['ARI']:.4f}")
    print(f"  AMI: {metrics['AMI']:.4f}")

    # =========================================================================
    # Multiple iterations: Monte Carlo estimation
    # =========================================================================
    print("\n" + "-" * 60)
    print("Monte Carlo Estimation (Multiple Iterations)")
    print("-" * 60)

    n_iterations = 100
    ari_scores = []
    ami_scores = []

    print(f"\nRunning {n_iterations} iterations...")

    for i in range(n_iterations):
        # Random train/test split each iteration
        train_data, test_data = train_test_split(
            data, train_size=0.8, random_state=i
        )

        # Fit on train, predict on test
        kmeans_train = KMeans(n_clusters=k, random_state=i, n_init=10)
        kmeans_train.fit(train_data)
        predicted_labels = kmeans_train.predict(test_data)

        # Fit fresh on test
        kmeans_test = KMeans(n_clusters=k, random_state=i, n_init=10)
        true_labels = kmeans_test.fit_predict(test_data)

        # Compute metrics for this iteration
        metrics = compute_parmigiani_metrics(predicted_labels, true_labels)
        ari_scores.append(metrics['ARI'])
        ami_scores.append(metrics['AMI'])

    # Aggregate results across all iterations
    summary = aggregate_parmigiani_metrics(ari_scores, ami_scores)

    print(f"\nAggregated Results ({summary['n_iterations']} iterations):")
    print(f"  ARI: {summary['ARI_mean']:.4f} +/- {summary['ARI_std']:.4f}")
    print(f"  AMI: {summary['AMI_mean']:.4f} +/- {summary['AMI_std']:.4f}")

    # =========================================================================
    # Interpretation guide
    # =========================================================================
    print("\n" + "-" * 60)
    print("Interpretation Guide")
    print("-" * 60)
    print("""
    ARI (Adjusted Rand Index):
      - 1.0:  Perfect agreement between train-fitted and test-fitted clusters
      - 0.0:  Random agreement (no better than chance)
      - <0:   Worse than random (rare)

    AMI (Adjusted Mutual Information):
      - 1.0:  Perfect mutual information (identical clusterings)
      - 0.0:  Independent clusterings

    High values (>0.8) indicate:
      - Clustering structure learned from training generalizes well
      - Stable, replicable cluster assignments

    Low values (<0.5) suggest:
      - Poor generalization from train to test
      - Unstable clustering structure
      - Consider: different K, different algorithm, or data preprocessing
    """)

    # =========================================================================
    # Compare different K values
    # =========================================================================
    print("-" * 60)
    print("Comparing Different K Values")
    print("-" * 60)

    for k in [2, 3, 4, 5]:
        ari_scores = []
        ami_scores = []

        for i in range(50):  # Fewer iterations for speed
            train_data, test_data = train_test_split(
                data, train_size=0.8, random_state=i
            )

            kmeans_train = KMeans(n_clusters=k, random_state=i, n_init=10)
            kmeans_train.fit(train_data)
            predicted_labels = kmeans_train.predict(test_data)

            kmeans_test = KMeans(n_clusters=k, random_state=i, n_init=10)
            true_labels = kmeans_test.fit_predict(test_data)

            metrics = compute_parmigiani_metrics(predicted_labels, true_labels)
            ari_scores.append(metrics['ARI'])
            ami_scores.append(metrics['AMI'])

        summary = aggregate_parmigiani_metrics(ari_scores, ami_scores)
        print(f"\n  K={k}:")
        print(f"    ARI: {summary['ARI_mean']:.4f} +/- {summary['ARI_std']:.4f}")
        print(f"    AMI: {summary['AMI_mean']:.4f} +/- {summary['AMI_std']:.4f}")

    print("\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
