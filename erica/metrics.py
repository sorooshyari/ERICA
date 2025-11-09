"""Metrics computation for ERICA clustering replicability analysis.

This module implements the core replicability metrics:
- CRI (Clustering Replicability Index)
- WCRI (Weighted CRI)
- TWCRI (Total Weighted CRI)
"""

import numpy as np
from typing import Dict, Tuple, Optional, List

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


