"""Metrics computation for ERICA clustering replicability analysis.

This module implements two families of replicability metrics:

ERICA Metrics (CLAM-based):
- CRI (Clustering Replicability Index)
- WCRI (Weighted CRI)
- TWCRI (Total Weighted CRI)

Parmigiani Metrics (partition comparison):
- ARI (Adjusted Rand Index)
- AMI (Adjusted Mutual Information)

The Parmigiani metrics are based on the methodology from:
    Parmigiani et al. "Cross-Study Replicability in Cluster Analysis"
    Statistical Science, 38(2): 303-316 (May 2023)
    DOI: 10.1214/22-STS871
    https://arxiv.org/pdf/2202.01910.pdf

Reference implementation: https://github.com/lorenzomasoero/clustering_replicability
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score

# Small epsilon for numerical stability
_EPS = 1e-9


def compute_cri(clam_matrix: np.ndarray, k: int) -> np.ndarray:
    """Compute CRI (Clustering Replicability Index) for each cluster.
    
    CRI measures how consistently samples are assigned to their primary cluster
    across iterations. A CRI value of 1.0 indicates perfect replicability.
    
    Parameters
    ----------
    clam_matrix : np.ndarray
        CLAM matrix with shape (n_samples, k)
    k : int
        Number of clusters
        
    Returns
    -------
    np.ndarray
        CRI value for each cluster (length k)
        
    Examples
    --------
    >>> clam = np.array([[50, 10], [45, 15], [5, 55]])
    >>> cri = compute_cri(clam, k=2)
    >>> print(cri)  # doctest: +SKIP
    [0.833, 0.917]
    
    Notes
    -----
    CRI is calculated as the mean of (max_count / total_count) for all samples
    in each cluster, where max_count is the count for the cluster's primary
    assignment.
    """
    # Get primary cluster assignments (cluster with max count for each sample)
    primary_clusters = np.argmax(clam_matrix, axis=1)
    
    # Sum across all iterations for each sample
    sum_row = clam_matrix.sum(axis=1)
    sum_row[sum_row == 0] = _EPS  # Avoid division by zero
    
    # Calculate CRI for each cluster
    cri_values = np.zeros(k)
    
    for cluster_idx in range(k):
        # Find samples primarily assigned to this cluster
        cluster_mask = (primary_clusters == cluster_idx)
        cluster_samples = np.where(cluster_mask)[0]
        
        if len(cluster_samples) == 0:
            cri_values[cluster_idx] = 0.0
            continue
        
        # Calculate proportion of times each sample was assigned to this cluster
        proportions = clam_matrix[cluster_samples, cluster_idx] / sum_row[cluster_samples]
        cri_values[cluster_idx] = np.mean(proportions)
    
    return cri_values


def compute_wcri(clam_matrix: np.ndarray, k: int) -> Tuple[np.ndarray, float]:
    """Compute WCRI (Weighted CRI) for each cluster.
    
    WCRI weights the CRI by cluster sizes to account for cluster imbalance.
    This provides a more balanced view of replicability when clusters have
    different sizes.
    
    Parameters
    ----------
    clam_matrix : np.ndarray
        CLAM matrix with shape (n_samples, k)
    k : int
        Number of clusters
        
    Returns
    -------
    tuple of (np.ndarray, float)
        (wcri_per_cluster, mean_wcri) where:
        - wcri_per_cluster: WCRI value for each cluster (length k)
        - mean_wcri: Mean WCRI across all clusters
        
    Examples
    --------
    >>> clam = np.array([[50, 10], [45, 15], [5, 55]])
    >>> wcri_per_cluster, mean_wcri = compute_wcri(clam, k=2)
    """
    # Compute CRI
    cri_values = compute_cri(clam_matrix, k)
    
    # Get primary cluster assignments
    primary_clusters = np.argmax(clam_matrix, axis=1)
    
    # Calculate cluster sizes
    cluster_sizes = np.array([
        np.sum(primary_clusters == i) for i in range(k)
    ])
    
    # Avoid division by zero
    total_size = np.sum(cluster_sizes)
    if total_size == 0:
        return np.zeros(k), 0.0
    
    # Weight CRI by cluster size
    wcri_values = (cri_values * cluster_sizes) / total_size
    mean_wcri = np.mean(wcri_values)
    
    return wcri_values, mean_wcri


def compute_twcri(clam_matrix: np.ndarray, k: int) -> float:
    """Compute TWCRI (Total Weighted CRI).
    
    TWCRI is the sum of weighted CRI values across all clusters, providing
    a single overall measure of clustering replicability.
    
    Parameters
    ----------
    clam_matrix : np.ndarray
        CLAM matrix with shape (n_samples, k)
    k : int
        Number of clusters
        
    Returns
    -------
    float
        Total weighted CRI value
        
    Examples
    --------
    >>> clam = np.array([[50, 10], [45, 15], [5, 55]])
    >>> twcri = compute_twcri(clam, k=2)
    
    Notes
    -----
    TWCRI provides a comprehensive replicability score where higher values
    indicate better overall clustering stability across all clusters.
    """
    wcri_values, _ = compute_wcri(clam_matrix, k)
    return np.sum(wcri_values)


def compute_metrics_for_clam(clam_matrix: np.ndarray, k: int) -> Dict[str, float]:
    """Compute all metrics (CRI, WCRI, TWCRI) for a CLAM matrix.
    
    This is a convenience function that computes all standard ERICA metrics
    in one call.
    
    Parameters
    ----------
    clam_matrix : np.ndarray
        CLAM matrix with shape (n_samples, k)
    k : int
        Number of clusters
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'CRI': Mean CRI across non-empty clusters
        - 'CRI_per_cluster': CRI for each cluster
        - 'WCRI': Mean WCRI across clusters
        - 'WCRI_per_cluster': WCRI for each cluster
        - 'TWCRI': Total weighted CRI
        - 'cluster_sizes': Size of each cluster
        - 'has_empty_clusters': Boolean indicating if any cluster is empty
        
    Notes
    -----
    If any cluster is empty (size 0), the metrics are marked as NaN to indicate
    this K value should be disqualified from K* selection per Algorithm 2, Line 4.
    
    Examples
    --------
    >>> clam = np.array([[50, 10], [45, 15], [5, 55]])
    >>> metrics = compute_metrics_for_clam(clam, k=2)
    >>> print(f"CRI: {metrics['CRI']:.3f}")
    >>> print(f"WCRI: {metrics['WCRI']:.3f}")
    >>> print(f"TWCRI: {metrics['TWCRI']:.3f}")
    """
    cri_values = compute_cri(clam_matrix, k)
    wcri_values, mean_wcri = compute_wcri(clam_matrix, k)
    twcri_value = compute_twcri(clam_matrix, k)
    
    # Get cluster sizes
    primary_clusters = np.argmax(clam_matrix, axis=1)
    cluster_sizes = np.array([
        np.sum(primary_clusters == i) for i in range(k)
    ])
    
    # Check for empty clusters (violates Algorithm 2, Line 4: ∃ k ≥ 1 : X_k = 0)
    has_empty_clusters = np.any(cluster_sizes == 0)
    
    # If there are empty clusters, mark all metrics as NaN to disqualify this K
    if has_empty_clusters:
        cri_mean = float('nan')
        wcri_mean = float('nan')
        twcri_final = float('nan')
    else:
        # Only average non-empty clusters for CRI
        # (WCRI and TWCRI already handle weighting correctly)
        non_empty_mask = cluster_sizes > 0
        cri_mean = float(np.mean(cri_values[non_empty_mask]))
        wcri_mean = float(mean_wcri)
        twcri_final = float(twcri_value)
    
    result = {
        'CRI': cri_mean,
        'CRI_per_cluster': cri_values.tolist(),
        'WCRI': wcri_mean,
        'WCRI_per_cluster': wcri_values.tolist(),
        'TWCRI': twcri_final,
        'cluster_sizes': cluster_sizes.tolist(),
        'k': k,
        'has_empty_clusters': bool(has_empty_clusters)
    }

    # Print results neatly with 6 decimal places and extra newlines between main metrics
    print("\n=== Cluster Index Summary ===")

    for key, value in result.items():
        if isinstance(value, float):
            if np.isnan(value):
                print(f"{key}: NaN (DISQUALIFIED - empty cluster detected)")
            else:
                print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")

        # Add an extra newline after CRI, WCRI, and TWCRI
        if key in ('CRI', 'WCRI', 'TWCRI'):
            print()

    print("==============================\n")

    return result




def compute_stability_score(clam_matrix: np.ndarray) -> float:
    """Compute overall stability score for a CLAM matrix.
    
    This score measures how stable the clustering is across all samples
    by looking at the concentration of assignments.
    
    Parameters
    ----------
    clam_matrix : np.ndarray
        CLAM matrix with shape (n_samples, k)
        
    Returns
    -------
    float
        Stability score between 0 and 1, where 1 is perfectly stable
        
    Notes
    -----
    Stability is calculated as the mean of the maximum proportion
    of assignments across all samples. A score of 1.0 means every
    sample was consistently assigned to the same cluster across
    all iterations.
    """
    # Get max count for each sample
    max_counts = np.max(clam_matrix, axis=1)
    
    # Get total count for each sample
    total_counts = np.sum(clam_matrix, axis=1)
    total_counts[total_counts == 0] = _EPS
    
    # Calculate proportion
    proportions = max_counts / total_counts
    
    return float(np.mean(proportions))


def compute_cluster_purity(clam_matrix: np.ndarray, k: int) -> np.ndarray:
    """Compute purity score for each cluster.

    Purity measures how often samples assigned to a cluster stay in
    that cluster across iterations.

    Parameters
    ----------
    clam_matrix : np.ndarray
        CLAM matrix with shape (n_samples, k)
    k : int
        Number of clusters

    Returns
    -------
    np.ndarray
        Purity score for each cluster (length k)

    Notes
    -----
    Purity is similar to CRI but may be computed slightly differently
    depending on the specific application requirements.
    """
    return compute_cri(clam_matrix, k)


# =============================================================================
# PARMIGIANI METRICS (Partition Comparison)
# =============================================================================
#
# These metrics implement the methodology from Parmigiani et al. (2023)
# "Cross-Study Replicability in Cluster Analysis"
#
# The key idea is to measure how well a clustering model trained on one
# subset of data generalizes to another subset, compared to clustering
# that subset directly. This is done via:
#
# Algorithm 1 (Parmigiani et al.):
# 1. Split data into train and test sets
# 2. Fit clustering model on train set
# 3. Predict cluster labels for test set using train-fitted model
#    -> "predicted_labels"
# 4. Fit a fresh clustering model directly on test set
#    -> "true_labels" (what we'd get if we only had test data)
# 5. Compare predicted_labels vs true_labels using ARI and AMI
#
# High ARI/AMI indicates that the clustering structure learned from
# training data transfers well to unseen test data.
# =============================================================================


def compute_ari(
    predicted_labels: np.ndarray,
    true_labels: np.ndarray
) -> float:
    """Compute Adjusted Rand Index (ARI) between two clusterings.

    ARI measures the similarity between two clustering assignments,
    adjusted for chance. It is used in the Parmigiani et al. methodology
    to compare train-fitted predictions against test-fitted labels.

    Parameters
    ----------
    predicted_labels : np.ndarray
        Cluster labels predicted by a model fitted on training data,
        applied to test samples. Shape: (n_test_samples,)
    true_labels : np.ndarray
        Cluster labels from fitting a fresh model directly on test data.
        Shape: (n_test_samples,)

    Returns
    -------
    float
        ARI score in range [-1, 1]:
        - 1.0: Perfect agreement (identical clusterings)
        - 0.0: Random labeling (expected value for random clusterings)
        - <0: Less agreement than expected by chance

    Notes
    -----
    ARI is symmetric: compute_ari(a, b) == compute_ari(b, a)

    This implements the replicability measure from:
        Parmigiani et al. "Cross-Study Replicability in Cluster Analysis"
        Statistical Science, 38(2): 303-316 (2023)

    In their Algorithm 1, ARI measures how well the clustering structure
    learned from training data transfers to held-out test data.

    Examples
    --------
    >>> # Perfect agreement
    >>> pred = np.array([0, 0, 1, 1, 2, 2])
    >>> true = np.array([0, 0, 1, 1, 2, 2])
    >>> compute_ari(pred, true)
    1.0

    >>> # Labels are permuted but structure is same
    >>> pred = np.array([0, 0, 1, 1, 2, 2])
    >>> true = np.array([1, 1, 2, 2, 0, 0])
    >>> compute_ari(pred, true)
    1.0

    >>> # Completely different structure
    >>> pred = np.array([0, 0, 0, 1, 1, 1])
    >>> true = np.array([0, 1, 0, 1, 0, 1])
    >>> ari = compute_ari(pred, true)
    >>> ari < 0.1  # Low agreement
    True

    See Also
    --------
    compute_ami : Adjusted Mutual Information (alternative measure)
    compute_parmigiani_metrics : Compute both ARI and AMI together
    sklearn.metrics.adjusted_rand_score : Underlying implementation
    """
    # Validate inputs have matching shapes
    predicted_labels = np.asarray(predicted_labels)
    true_labels = np.asarray(true_labels)

    if predicted_labels.shape != true_labels.shape:
        raise ValueError(
            f"Label arrays must have same shape. "
            f"Got predicted_labels: {predicted_labels.shape}, "
            f"true_labels: {true_labels.shape}"
        )

    if len(predicted_labels) == 0:
        raise ValueError("Cannot compute ARI on empty label arrays")

    # Use sklearn's implementation which handles all edge cases
    # ARI is symmetric and invariant to label permutations
    return float(adjusted_rand_score(predicted_labels, true_labels))


def compute_ami(
    predicted_labels: np.ndarray,
    true_labels: np.ndarray,
    average_method: str = 'arithmetic'
) -> float:
    """Compute Adjusted Mutual Information (AMI) between two clusterings.

    AMI measures the mutual information between two clustering assignments,
    adjusted for chance. It quantifies how much knowing one clustering
    tells you about the other.

    Parameters
    ----------
    predicted_labels : np.ndarray
        Cluster labels predicted by a model fitted on training data,
        applied to test samples. Shape: (n_test_samples,)
    true_labels : np.ndarray
        Cluster labels from fitting a fresh model directly on test data.
        Shape: (n_test_samples,)
    average_method : str, optional
        Method to normalize the mutual information. Options:
        - 'arithmetic': Arithmetic mean of entropies (default, used by Parmigiani)
        - 'geometric': Geometric mean of entropies
        - 'min': Minimum of entropies
        - 'max': Maximum of entropies

    Returns
    -------
    float
        AMI score in range [0, 1]:
        - 1.0: Perfect agreement (identical clusterings up to permutation)
        - 0.0: Independent clusterings (no mutual information beyond chance)

    Notes
    -----
    AMI is symmetric: compute_ami(a, b) == compute_ami(b, a)

    AMI and ARI measure related but distinct aspects of clustering similarity:
    - ARI: Based on counting pairs of samples
    - AMI: Based on information-theoretic mutual information

    Both are adjusted for chance, meaning random clusterings yield ~0.

    This implements the replicability measure from:
        Parmigiani et al. "Cross-Study Replicability in Cluster Analysis"
        Statistical Science, 38(2): 303-316 (2023)

    Examples
    --------
    >>> # Perfect agreement
    >>> pred = np.array([0, 0, 1, 1, 2, 2])
    >>> true = np.array([0, 0, 1, 1, 2, 2])
    >>> compute_ami(pred, true)
    1.0

    >>> # Labels permuted but structure preserved
    >>> pred = np.array([0, 0, 1, 1, 2, 2])
    >>> true = np.array([2, 2, 0, 0, 1, 1])
    >>> compute_ami(pred, true)
    1.0

    See Also
    --------
    compute_ari : Adjusted Rand Index (alternative measure)
    compute_parmigiani_metrics : Compute both ARI and AMI together
    sklearn.metrics.adjusted_mutual_info_score : Underlying implementation
    """
    # Validate inputs have matching shapes
    predicted_labels = np.asarray(predicted_labels)
    true_labels = np.asarray(true_labels)

    if predicted_labels.shape != true_labels.shape:
        raise ValueError(
            f"Label arrays must have same shape. "
            f"Got predicted_labels: {predicted_labels.shape}, "
            f"true_labels: {true_labels.shape}"
        )

    if len(predicted_labels) == 0:
        raise ValueError("Cannot compute AMI on empty label arrays")

    # Use sklearn's implementation
    # average_method='arithmetic' matches Parmigiani et al. implementation
    return float(adjusted_mutual_info_score(
        predicted_labels,
        true_labels,
        average_method=average_method
    ))


def compute_parmigiani_metrics(
    predicted_labels: np.ndarray,
    true_labels: np.ndarray
) -> Dict[str, float]:
    """Compute both ARI and AMI for a single iteration.

    This is a convenience function that computes both Parmigiani metrics
    in one call. Use this when processing a single train/test split.

    Parameters
    ----------
    predicted_labels : np.ndarray
        Cluster labels predicted by train-fitted model on test samples.
        Shape: (n_test_samples,)
    true_labels : np.ndarray
        Cluster labels from fitting directly on test samples.
        Shape: (n_test_samples,)

    Returns
    -------
    dict
        Dictionary with keys:
        - 'ARI': Adjusted Rand Index
        - 'AMI': Adjusted Mutual Information

    Examples
    --------
    >>> pred = np.array([0, 0, 1, 1, 2, 2])
    >>> true = np.array([0, 0, 1, 1, 2, 2])
    >>> metrics = compute_parmigiani_metrics(pred, true)
    >>> print(f"ARI: {metrics['ARI']:.3f}, AMI: {metrics['AMI']:.3f}")
    ARI: 1.000, AMI: 1.000

    See Also
    --------
    aggregate_parmigiani_metrics : Aggregate metrics across multiple iterations
    """
    return {
        'ARI': compute_ari(predicted_labels, true_labels),
        'AMI': compute_ami(predicted_labels, true_labels)
    }


def aggregate_parmigiani_metrics(
    ari_scores: List[float],
    ami_scores: List[float]
) -> Dict[str, float]:
    """Aggregate ARI and AMI scores across multiple iterations.

    In the Parmigiani methodology, ARI and AMI are computed for each
    Monte Carlo iteration. This function aggregates them into summary
    statistics (mean and standard deviation).

    Parameters
    ----------
    ari_scores : list of float
        ARI scores from each iteration. Length: n_iterations
    ami_scores : list of float
        AMI scores from each iteration. Length: n_iterations

    Returns
    -------
    dict
        Dictionary with keys:
        - 'ARI_mean': Mean ARI across iterations
        - 'ARI_std': Standard deviation of ARI
        - 'AMI_mean': Mean AMI across iterations
        - 'AMI_std': Standard deviation of AMI
        - 'n_iterations': Number of iterations

    Notes
    -----
    Following Parmigiani et al., the mean ARI/AMI across iterations
    provides a robust estimate of clustering replicability, while the
    standard deviation indicates stability of this estimate.

    Examples
    --------
    >>> # Simulate 5 iterations with high replicability
    >>> ari_scores = [0.85, 0.88, 0.82, 0.86, 0.84]
    >>> ami_scores = [0.80, 0.83, 0.78, 0.81, 0.79]
    >>> summary = aggregate_parmigiani_metrics(ari_scores, ami_scores)
    >>> print(f"Mean ARI: {summary['ARI_mean']:.3f} +/- {summary['ARI_std']:.3f}")
    Mean ARI: 0.850 +/- 0.021

    See Also
    --------
    compute_parmigiani_metrics : Compute metrics for a single iteration
    """
    ari_array = np.array(ari_scores)
    ami_array = np.array(ami_scores)

    return {
        'ARI_mean': float(np.mean(ari_array)),
        'ARI_std': float(np.std(ari_array)),
        'AMI_mean': float(np.mean(ami_array)),
        'AMI_std': float(np.std(ami_array)),
        'n_iterations': len(ari_scores)
    }


def summarize_metrics(
    metrics_by_k: Dict[int, Dict[str, Dict]]
) -> Dict[str, List]:
    """Summarize metrics across all k values and methods.
    
    Parameters
    ----------
    metrics_by_k : dict
        Nested dictionary: {k: {method: metrics_dict}}
        
    Returns
    -------
    dict
        Summary dictionary with lists for each metric
        
    Examples
    --------
    >>> metrics = {
    ...     2: {'kmeans': {'CRI': 0.85, 'WCRI': 0.82, 'TWCRI': 0.88}},
    ...     3: {'kmeans': {'CRI': 0.90, 'WCRI': 0.87, 'TWCRI': 0.92}}
    ... }
    >>> summary = summarize_metrics(metrics)
    """
    k_values = []
    cri_values = []
    wcri_values = []
    twcri_values = []
    methods = []
    
    for k in sorted(metrics_by_k.keys()):
        for method, metrics in metrics_by_k[k].items():
            k_values.append(k)
            methods.append(method)
            cri_values.append(metrics.get('CRI', 0.0))
            wcri_values.append(metrics.get('WCRI', 0.0))
            twcri_values.append(metrics.get('TWCRI', 0.0))
    
    return {
        'k_values': k_values,
        'methods': methods,
        'CRI': cri_values,
        'WCRI': wcri_values,
        'TWCRI': twcri_values
    }




def select_optimal_k(
    metric_dict: Dict[int, float],
    k_max: Optional[int] = None
) -> int:
    """Select optimal K using ERICA Algorithm 2 (non-decreasing metric selection).
    
    This algorithm implements the K_star selection procedure from ERICA paper:
    
    Algorithm 2: Cluster number (K*) selection with ERICA
    1. Input: Metric for considered K values {M_K : K = 2, ..., K^max}
    2. K* ← 2  (initialize)
    3. for K = 3 to K^max do
    4.     if NA ∉ {M_K(k)} then  % is violated if ∃ k ≥ 1 : X_k = 0
    5.         if M_K ≥ M_{K-1} then
    6.             K* ← K
    7.         end if
    8.     end if
    9. end for
    10. return K*
    
    The algorithm prefers the largest K where the metric is non-decreasing,
    ensuring we select the most granular clustering that maintains stability.
    
    Parameters
    ----------
    metric_dict : dict
        Dictionary mapping K values to metric scores (e.g., {2: 0.71, 3: 0.75, ...})
        NaN values are skipped per line 4 of Algorithm 2
    k_max : int, optional
        Maximum K to consider. If None, uses max(metric_dict.keys())
        
    Returns
    -------
    int
        Optimal K value (K_star)
        
    Examples
    --------
    >>> M = {2: 0.71, 3: 0.75, 4: 0.74, 5: float('nan'), 6: 0.78}
    >>> k_star = select_optimal_k(M)
    >>> print(k_star)
    6
    
    >>> M = {2: 0.85, 3: 0.90, 4: 0.88, 5: 0.87}
    >>> k_star = select_optimal_k(M)
    >>> print(k_star)
    3
    
    Notes
    -----
    This algorithm differs from simply selecting the maximum metric value.
    It prefers larger K values when metrics are non-decreasing, which helps
    identify the most detailed stable clustering structure.
    
    Line 4 of Algorithm 2: "if NA ∉ {M_K(k)}" means we skip any K with NaN values.
    This is important when clustering fails for certain K values.
    """
    import math
    
    if not metric_dict:
        raise ValueError("metric_dict cannot be empty")
    
    # Get sorted K values
    k_values = sorted(metric_dict.keys())
    
    if len(k_values) == 0:
        raise ValueError("No valid K values in metric_dict")
    
    # Set k_max if not provided
    if k_max is None:
        k_max = max(k_values)
    
    # Algorithm 2, Line 2: Initialize K_star to minimum K (typically 2)
    k_star = min(k_values)
    
    # Track the most recent valid K and its metric value
    last_valid_k = k_star
    last_valid_metric = metric_dict.get(k_star, float('nan'))
    
    # Handle case where the first K itself is NaN
    if math.isnan(last_valid_metric):
        # Find first valid K (satisfies line 4: NA ∉ {M_K(k)})
        for k in k_values:
            val = metric_dict.get(k, float('nan'))
            if not math.isnan(val):
                k_star = k
                last_valid_k = k
                last_valid_metric = val
                break
    
    # Algorithm 2, Line 3: for K = 3 to K^max do
    for k in k_values:
        if k <= last_valid_k:
            continue
        if k > k_max:
            break
        
        # Algorithm 2, Line 4: if NA ∉ {M_K(k)} then
        # Check if current metric is valid (not NaN)
        current_metric = metric_dict.get(k, float('nan'))
        if math.isnan(current_metric):
            continue  # Skip this K if it contains NA
        
        # Algorithm 2, Line 5: if M_K ≥ M_{K-1} then
        # Compare with the most recent valid metric
        if current_metric >= last_valid_metric:
            # Algorithm 2, Line 6: K* ← K
            k_star = k
        
        # Update last valid K and metric for next comparison
        last_valid_k = k
        last_valid_metric = current_metric
    
    # Algorithm 2, Line 10: return K*
    return k_star


def select_optimal_k_by_method(
    metrics_by_k: Dict[int, Dict[str, Dict]],
    metric_name: str = 'TWCRI'
) -> Dict[str, int]:
    """Select optimal K for each clustering method using Algorithm 2.
    
    Parameters
    ----------
    metrics_by_k : dict
        Nested dictionary: {k: {method: metrics_dict}}
    metric_name : str, optional
        Metric to use for selection ('CRI', 'WCRI', or 'TWCRI'), default 'TWCRI'
        
    Returns
    -------
    dict
        Dictionary mapping method names to their optimal K values
        
    Examples
    --------
    >>> metrics = {
    ...     2: {'kmeans': {'TWCRI': 0.71}},
    ...     3: {'kmeans': {'TWCRI': 0.75}},
    ...     4: {'kmeans': {'TWCRI': 0.74}}
    ... }
    >>> optimal_k = select_optimal_k_by_method(metrics, 'TWCRI')
    >>> print(optimal_k)
    {'kmeans': 3}
    """
    if metric_name not in ['CRI', 'WCRI', 'TWCRI']:
        raise ValueError(f"Invalid metric_name: {metric_name}. Must be 'CRI', 'WCRI', or 'TWCRI'")
    
    # Organize metrics by method
    methods_metrics = {}
    for k, methods_dict in metrics_by_k.items():
        for method, metrics in methods_dict.items():
            if method not in methods_metrics:
                methods_metrics[method] = {}
            if metric_name in metrics:
                methods_metrics[method][k] = metrics[metric_name]
    
    # Select optimal K for each method
    optimal_k_by_method = {}
    for method, metric_dict in methods_metrics.items():
        optimal_k_by_method[method] = select_optimal_k(metric_dict)
    
    return optimal_k_by_method


