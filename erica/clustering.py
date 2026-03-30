"""Clustering functions for ERICA replicability analysis.

This module contains all clustering-related functions including:
- Iterative clustering subsampling (Monte Carlo subsampling)
- K-Means clustering with alignment
- Agglomerative (hierarchical) clustering with alignment
- CLAM matrix generation
"""

import os
import random
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Dict
from sklearn.cluster import KMeans, AgglomerativeClustering, HDBSCAN


def iterative_clustering_subsampling(
    samples_array: np.ndarray,
    num_samples: int,
    num_iterations: int,
    subsample_size_train: int,
    base_save_folder_str: str,
    iter_prefix: str = "",
    optimize_io: bool = True,
    verbose: bool = False
) -> Tuple[Optional[str], str]:
    """Perform iterative clustering subsampling (Monte Carlo subsampling).
    
    Creates train/test splits for multiple iterations and stores index lists.
    This is the foundation of ERICA's replicability analysis.
    
    Parameters
    ----------
    samples_array : np.ndarray
        Input data array with shape (n_samples, n_features)
    num_samples : int
        Total number of samples
    num_iterations : int
        Number of Monte Carlo iterations (B)
    subsample_size_train : int
        Size of training subsample
    base_save_folder_str : str
        Base folder path for saving outputs
    iter_prefix : str, optional
        Prefix for subfolder names, default ""
    optimize_io : bool, optional
        If True, only save indices (reduces I/O), default True
    verbose : bool, optional
        Whether to print progress messages, default False
        
    Returns
    -------
    tuple of (str or None, str)
        (subsamples_data_folder, indices_folder)
        subsamples_data_folder is None if optimize_io=True
        
    Examples
    --------
    >>> data = np.random.rand(100, 50)
    >>> train_size = int(100 * 0.8)
    >>> subsamples_folder, indices_folder = iterative_clustering_subsampling(
    ...     data, 100, 200, train_size, './output'
    ... )
    
    Notes
    -----
    This function implements the core Monte Carlo subsampling approach:
    1. For each iteration, randomly split data into train/test sets
    2. Save indices for reproducibility
    3. Optionally save actual data arrays (if optimize_io=False)
    
    The train/test split is performed without replacement within each iteration,
    ensuring every sample appears exactly once per iteration (either in train or test).
    """
    if verbose:
        print("Starting iterative clustering subsampling...")
    
    # Create output directories
    subsamples_data_folder = os.path.join(base_save_folder_str, f'{iter_prefix}subsamples_data')
    indices_folder = os.path.join(base_save_folder_str, f'{iter_prefix}indices')
    
    # Always create indices folder (needed for both modes)
    os.makedirs(indices_folder, exist_ok=True)
    
    # Only create subsamples_data folder if not optimizing I/O
    if not optimize_io:
        os.makedirs(subsamples_data_folder, exist_ok=True)

    all_subset_indices_list: List[np.ndarray] = []
    all_complement_indices_list: List[np.ndarray] = []

    # Perform subsampling iterations
    for B_idx_iter in range(num_iterations):
        if verbose and B_idx_iter % 50 == 0:
            print(f"  Subsampling iteration {B_idx_iter+1}/{num_iterations}")
        
        # Randomly shuffle all indices
        current_indices = random.sample(range(num_samples), num_samples)
        subset_indices_arr = np.array(current_indices[:subsample_size_train])
        complement_indices_arr = np.array(current_indices[subsample_size_train:])

        all_subset_indices_list.append(subset_indices_arr)
        all_complement_indices_list.append(complement_indices_arr)

        if not optimize_io:
            # Save data arrays immediately (legacy mode)
            subset_data_arr = samples_array[subset_indices_arr]
            complement_data_arr = samples_array[complement_indices_arr]
            np.save(os.path.join(subsamples_data_folder, f'subset_data_{B_idx_iter}.npy'), subset_data_arr)
            np.save(os.path.join(subsamples_data_folder, f'complement_data_{B_idx_iter}.npy'), complement_data_arr)

    # Always save indices (required for CLAM generation)
    np.save(
        os.path.join(indices_folder, 'all_train_indices.npy'),
        np.array(all_subset_indices_list, dtype=object)
    )
    np.save(
        os.path.join(indices_folder, 'all_test_indices.npy'),
        np.array(all_complement_indices_list, dtype=object)
    )

    if verbose:
        print("Completed iterative clustering subsampling")
    
    # Return None for subsamples_data_folder if optimized
    return (None if optimize_io else subsamples_data_folder), indices_folder


def load_iteration_data(
    iteration_idx: int,
    samples_array: np.ndarray,
    indices_folder: str
) -> Tuple[np.ndarray, np.ndarray]:
    """Load train/test data for a specific iteration.
    
    Parameters
    ----------
    iteration_idx : int
        Iteration index to load
    samples_array : np.ndarray
        Full dataset array
    indices_folder : str
        Path to folder containing index files
        
    Returns
    -------
    tuple of (np.ndarray, np.ndarray)
        (train_data, test_data) for the specified iteration
        
    Raises
    ------
    FileNotFoundError
        If index files are not found
    """
    try:
        train_indices = np.load(
            os.path.join(indices_folder, 'all_train_indices.npy'),
            allow_pickle=True
        )
        test_indices = np.load(
            os.path.join(indices_folder, 'all_test_indices.npy'),
            allow_pickle=True
        )
        
        # Convert to integer indices
        train_idx = train_indices[iteration_idx].astype(int)
        test_idx = test_indices[iteration_idx].astype(int)
        
        subset_data_arr = samples_array[train_idx]
        complement_data_arr = samples_array[test_idx]
        
        return subset_data_arr, complement_data_arr
        
    except Exception as e:
        raise FileNotFoundError(
            f"Failed to load indices for iteration {iteration_idx}: {str(e)}"
        )


def kmeans_clustering(
    samples_array: np.ndarray,
    k: int,
    n_iterations: int,
    indices_folder: str,
    output_dir: str,
    random_state: int = 0,
    verbose: bool = False
) -> Dict:
    """Perform K-Means clustering with ERICA analysis.
    
    This function implements the complete K-Means clustering pipeline:
    1. Calculate global centroids on full dataset
    2. For each iteration: fit on train, predict on test
    3. Align cluster identities across iterations
    4. Generate CLAM matrix
    
    Parameters
    ----------
    samples_array : np.ndarray
        Input data with shape (n_samples, n_features)
    k : int
        Number of clusters
    n_iterations : int
        Number of iterations
    indices_folder : str
        Path to folder with train/test indices
    output_dir : str
        Directory for saving outputs
    random_state : int, optional
        Random state for K-Means, default 0
    verbose : bool, optional
        Whether to print progress, default False
        
    Returns
    -------
    dict
        Results dictionary containing:
        - 'clam_matrix': Final CLAM matrix
        - 'global_centroids': Sorted global centroids
        - 'aligned_predictions': Aligned prediction matrix
        - 'output_folder': Path to output directory
        
    Examples
    --------
    >>> data = np.random.rand(100, 50)
    >>> result = kmeans_clustering(
    ...     data, k=3, n_iterations=100,
    ...     indices_folder='./indices',
    ...     output_dir='./output'
    ... )
    >>> clam = result['clam_matrix']
    """
    if verbose:
        print(f"Starting K-Means clustering for k={k}...")
    
    n_samples = samples_array.shape[0]
    
    # Load indices
    train_indices = np.load(
        os.path.join(indices_folder, 'all_train_indices.npy'),
        allow_pickle=True
    )
    test_indices = np.load(
        os.path.join(indices_folder, 'all_test_indices.npy'),
        allow_pickle=True
    )
    
    test_size = len(test_indices[0])
    
    # Step 1: Calculate global centroids
    if verbose:
        print("  Calculating global centroids...")
    
    kmeans_global = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
    kmeans_global.fit(samples_array)
    global_centroids = kmeans_global.cluster_centers_
    
    # Sort centroids by L2 norm
    l2_norms = [np.linalg.norm(centroid) for centroid in global_centroids]
    sorted_indices = np.argsort(l2_norms)
    global_centroids_sorted = global_centroids[sorted_indices]
    
    # Step 2: Clustering iterations
    if verbose:
        print(f"  Running {n_iterations} iterations...")
    
    unaligned_predictions = np.zeros((n_iterations, test_size))
    iteration_centroids_list = []
    all_predicted_labels = []
    all_true_labels = []

    for iter_idx in range(n_iterations):
        if verbose and iter_idx % 50 == 0:
            print(f"    Iteration {iter_idx+1}/{n_iterations}")

        # Load data for this iteration
        train_data, test_data = load_iteration_data(
            iter_idx, samples_array, indices_folder
        )

        # Fit on train, predict on test
        kmeans_iter = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
        kmeans_iter.fit(train_data)
        predictions = kmeans_iter.predict(test_data)

        all_predicted_labels.append(predictions.copy())

        # Fit fresh model on test data for true_labels (Parmigiani method)
        kmeans_test = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
        true_labels = kmeans_test.fit_predict(test_data)
        all_true_labels.append(true_labels)

        unaligned_predictions[iter_idx, :] = predictions
        iteration_centroids_list.append(kmeans_iter.cluster_centers_)
    
    # Step 3: Align cluster identities
    if verbose:
        print("  Aligning cluster identities...")
    
    aligned_predictions = _align_cluster_identities(
        unaligned_predictions,
        iteration_centroids_list,
        global_centroids_sorted,
        k
    )
    
    # Step 4: Generate CLAM matrix
    if verbose:
        print("  Generating CLAM matrix...")
    
    clam_matrix = _generate_clam_matrix(
        n_samples, k, n_iterations,
        test_indices, aligned_predictions
    )
    
    # Save outputs
    output_folder = os.path.join(output_dir, f'kmeans_k{k}')
    os.makedirs(output_folder, exist_ok=True)
    
    np.save(os.path.join(output_folder, 'clam_matrix.npy'), clam_matrix)
    np.save(os.path.join(output_folder, 'global_centroids.npy'), global_centroids_sorted)
    pd.DataFrame(clam_matrix).to_csv(
        os.path.join(output_folder, f'clam_k{k}_kmeans.csv'),
        index=False, header=False
    )
    
    if verbose:
        print(f"  K-Means clustering complete for k={k}")
    
    return {
        'clam_matrix': clam_matrix,
        'global_centroids': global_centroids_sorted,
        'aligned_predictions': aligned_predictions,
        'output_folder': output_folder,
        'iteration_labels': {
            'predicted': all_predicted_labels,
            'true': all_true_labels,
        },
    }


def agglomerative_clustering(
    samples_array: np.ndarray,
    k: int,
    linkage: str,
    n_iterations: int,
    indices_folder: str,
    output_dir: str,
    verbose: bool = False
) -> Dict:
    """Perform Agglomerative (Hierarchical) clustering with ERICA analysis.
    
    This function implements the complete Agglomerative clustering pipeline:
    1. Calculate global centroids on full dataset
    2. For each iteration: fit/predict on test data
    3. Align cluster identities across iterations
    4. Generate CLAM matrix
    
    Parameters
    ----------
    samples_array : np.ndarray
        Input data with shape (n_samples, n_features)
    k : int
        Number of clusters
    linkage : str
        Linkage method ('single', 'ward', 'complete', 'average')
    n_iterations : int
        Number of iterations
    indices_folder : str
        Path to folder with train/test indices
    output_dir : str
        Directory for saving outputs
    verbose : bool, optional
        Whether to print progress, default False
        
    Returns
    -------
    dict
        Results dictionary containing:
        - 'clam_matrix': Final CLAM matrix
        - 'global_centroids': Sorted global centroids
        - 'aligned_predictions': Aligned prediction matrix
        - 'output_folder': Path to output directory
        
    Notes
    -----
    Unlike K-Means, Agglomerative clustering in ERICA fits on the TEST data
    per iteration, following the original MCSS methodology.
    """
    if verbose:
        print(f"Starting Agglomerative clustering (linkage={linkage}) for k={k}...")
    
    n_samples = samples_array.shape[0]
    
    # Load indices
    train_indices = np.load(
        os.path.join(indices_folder, 'all_train_indices.npy'),
        allow_pickle=True
    )
    test_indices = np.load(
        os.path.join(indices_folder, 'all_test_indices.npy'),
        allow_pickle=True
    )
    
    test_size = len(test_indices[0])
    
    # Step 1: Calculate global centroids
    if verbose:
        print("  Calculating global centroids...")
    
    agg_global = AgglomerativeClustering(n_clusters=k, linkage=linkage)
    global_labels = agg_global.fit_predict(samples_array)
    
    # Calculate centroids from labels
    global_centroids = np.zeros((k, samples_array.shape[1]))
    for cluster_idx in range(k):
        cluster_mask = (global_labels == cluster_idx)
        if cluster_mask.sum() > 0:
            global_centroids[cluster_idx] = samples_array[cluster_mask].mean(axis=0)
        else:
            global_centroids[cluster_idx] = samples_array.mean(axis=0)
    
    # Sort centroids by L2 norm
    l2_norms = [np.linalg.norm(centroid) for centroid in global_centroids]
    sorted_indices = np.argsort(l2_norms)
    global_centroids_sorted = global_centroids[sorted_indices]
    
    # Step 2: Clustering iterations (fit on TEST data for agglomerative)
    if verbose:
        print(f"  Running {n_iterations} iterations...")
    
    unaligned_predictions = np.zeros((n_iterations, test_size))
    iteration_centroids_list = []
    all_predicted_labels = []
    all_true_labels = []

    for iter_idx in range(n_iterations):
        if verbose and iter_idx % 50 == 0:
            print(f"    Iteration {iter_idx+1}/{n_iterations}")

        # Load data for this iteration
        train_data, test_data = load_iteration_data(
            iter_idx, samples_array, indices_folder
        )

        # Fit/predict on test data
        agg_iter = AgglomerativeClustering(n_clusters=k, linkage=linkage)
        predictions = agg_iter.fit_predict(test_data)

        # true_labels = test-fitted predictions (already computed above)
        all_true_labels.append(predictions.copy())

        # predicted_labels = train-fitted centroids applied to test data
        agg_train = AgglomerativeClustering(n_clusters=k, linkage=linkage)
        train_labels = agg_train.fit_predict(train_data)
        train_centroids_pred = np.zeros((k, train_data.shape[1]))
        for cidx in range(k):
            mask = train_labels == cidx
            if mask.any():
                train_centroids_pred[cidx] = train_data[mask].mean(axis=0)
            else:
                train_centroids_pred[cidx] = train_data.mean(axis=0)
        dists = np.linalg.norm(
            test_data[:, np.newaxis, :] - train_centroids_pred[np.newaxis, :, :],
            axis=2
        )
        pred_labels = np.argmin(dists, axis=1)
        all_predicted_labels.append(pred_labels)

        unaligned_predictions[iter_idx, :] = predictions
        
        # Calculate centroids from test data
        iter_centroids = np.zeros((k, test_data.shape[1]))
        for cluster_idx in range(k):
            cluster_mask = (predictions == cluster_idx)
            if cluster_mask.sum() > 0:
                iter_centroids[cluster_idx] = test_data[cluster_mask].mean(axis=0)
            else:
                iter_centroids[cluster_idx] = test_data.mean(axis=0)
        
        iteration_centroids_list.append(iter_centroids)
    
    # Step 3: Align cluster identities
    if verbose:
        print("  Aligning cluster identities...")
    
    aligned_predictions = _align_cluster_identities(
        unaligned_predictions,
        iteration_centroids_list,
        global_centroids_sorted,
        k
    )
    
    # Step 4: Generate CLAM matrix
    if verbose:
        print("  Generating CLAM matrix...")
    
    clam_matrix = _generate_clam_matrix(
        n_samples, k, n_iterations,
        test_indices, aligned_predictions
    )
    
    # Save outputs
    output_folder = os.path.join(output_dir, f'agglomerative_{linkage}_k{k}')
    os.makedirs(output_folder, exist_ok=True)
    
    np.save(os.path.join(output_folder, 'clam_matrix.npy'), clam_matrix)
    np.save(os.path.join(output_folder, 'global_centroids.npy'), global_centroids_sorted)
    pd.DataFrame(clam_matrix).to_csv(
        os.path.join(output_folder, f'clam_k{k}_agglomerative_{linkage}.csv'),
        index=False, header=False
    )
    
    if verbose:
        print(f"  Agglomerative clustering complete for k={k}")
    
    return {
        'clam_matrix': clam_matrix,
        'global_centroids': global_centroids_sorted,
        'aligned_predictions': aligned_predictions,
        'output_folder': output_folder,
        'iteration_labels': {
            'predicted': all_predicted_labels,
            'true': all_true_labels,
        },
    }


def hdbscan_clustering(
    samples_array: np.ndarray,
    n_iterations: int,
    indices_folder: str,
    output_dir: str,
    hdbscan_params: Optional[Dict] = None,
    verbose: bool = False
) -> Dict:
    """Perform HDBSCAN auto-K clustering with ERICA analysis.

    HDBSCAN discovers the number of clusters automatically. Runs HDBSCAN on
    each train/test split, tracks discovered K values, computes a CLAM matrix
    from iterations where K matches the mode, and returns label pairs for ARI/AMI.

    Parameters
    ----------
    samples_array : np.ndarray
        Input data with shape (n_samples, n_features)
    n_iterations : int
        Number of iterations
    indices_folder : str
        Path to folder with train/test indices
    output_dir : str
        Directory for saving outputs
    hdbscan_params : dict, optional
        Parameters for HDBSCAN. Defaults: min_cluster_size=15
    verbose : bool, optional
        Whether to print progress, default False

    Returns
    -------
    dict
        Results with keys: k_distribution, modal_k, k_agreement_rate,
        clam_matrix, n_iterations_used, iteration_labels, noise_counts,
        output_folder
    """
    from collections import Counter

    if verbose:
        print("Starting HDBSCAN clustering...")

    n_samples = samples_array.shape[0]
    params = {'min_cluster_size': 15}
    if hdbscan_params:
        params.update(hdbscan_params)

    train_indices = np.load(
        os.path.join(indices_folder, 'all_train_indices.npy'),
        allow_pickle=True
    )
    test_indices = np.load(
        os.path.join(indices_folder, 'all_test_indices.npy'),
        allow_pickle=True
    )

    discovered_ks = []
    all_predicted_labels = []
    all_true_labels = []
    noise_counts = []
    # Store (iter_idx, reassigned_labels) for CLAM building
    iter_label_pairs = []

    for iter_idx in range(n_iterations):
        if verbose and iter_idx % 50 == 0:
            print(f"  Iteration {iter_idx + 1}/{n_iterations}")

        train_data, test_data = load_iteration_data(
            iter_idx, samples_array, indices_folder
        )

        # Fit HDBSCAN on train set
        hdb_train = HDBSCAN(**params)
        train_labels = hdb_train.fit_predict(train_data)

        unique_train = set(train_labels)
        unique_train.discard(-1)
        if len(unique_train) == 0:
            discovered_ks.append(0)
            all_predicted_labels.append(np.array([]))
            all_true_labels.append(np.array([]))
            noise_counts.append(len(test_data))
            continue

        n_clusters_train = max(unique_train) + 1
        train_centroids = np.zeros((n_clusters_train, train_data.shape[1]))
        for c in range(n_clusters_train):
            mask = train_labels == c
            if mask.any():
                train_centroids[c] = train_data[mask].mean(axis=0)

        # Predict test labels via nearest train centroid
        distances = np.linalg.norm(
            test_data[:, np.newaxis, :] - train_centroids[np.newaxis, :, :],
            axis=2
        )
        predicted_labels = np.argmin(distances, axis=1)

        # Fit fresh HDBSCAN on test set
        hdb_test = HDBSCAN(**params)
        test_labels_raw = hdb_test.fit_predict(test_data)

        n_noise = int(np.sum(test_labels_raw == -1))
        noise_counts.append(n_noise)

        unique_test = set(test_labels_raw)
        unique_test.discard(-1)
        if len(unique_test) == 0:
            # All test points are noise — no valid true_labels for ARI/AMI.
            # Use empty arrays so this iteration is skipped in ARI aggregation.
            all_predicted_labels.append(np.array([]))
            all_true_labels.append(np.array([]))
            discovered_ks.append(n_clusters_train)
            continue
        else:
            n_clusters_test = max(unique_test) + 1
            test_centroids = np.zeros((n_clusters_test, test_data.shape[1]))
            for c in range(n_clusters_test):
                mask = test_labels_raw == c
                if mask.any():
                    test_centroids[c] = test_data[mask].mean(axis=0)
            true_labels = _assign_noise_to_nearest(
                test_labels_raw, test_data, test_centroids
            )
            discovered_k = n_clusters_test

        discovered_ks.append(discovered_k)
        all_predicted_labels.append(predicted_labels)
        all_true_labels.append(true_labels)
        iter_label_pairs.append((iter_idx, true_labels))

    # Find modal K
    k_counts = Counter(k for k in discovered_ks if k > 0)
    if not k_counts:
        modal_k = 1
        k_agreement_rate = 0.0
        clam_matrix = np.zeros((n_samples, 1))
        n_used = 0
        k_distribution = {}
    else:
        modal_k = k_counts.most_common(1)[0][0]
        n_modal = k_counts[modal_k]
        k_agreement_rate = n_modal / n_iterations
        k_distribution = dict(k_counts)

        clam_matrix = np.zeros((n_samples, modal_k))
        n_used = 0
        for iter_idx, labels in iter_label_pairs:
            if discovered_ks[iter_idx] != modal_k:
                continue
            if len(labels) == 0:
                continue
            if int(labels.max()) >= modal_k:
                continue

            iter_test_idx = test_indices[iter_idx].astype(int)
            for i, sample_idx in enumerate(iter_test_idx):
                if i < len(labels):
                    cluster_id = int(labels[i])
                    if 0 <= cluster_id < modal_k:
                        clam_matrix[sample_idx, cluster_id] += 1
            n_used += 1

    output_folder = os.path.join(output_dir, 'hdbscan')
    os.makedirs(output_folder, exist_ok=True)
    np.save(os.path.join(output_folder, 'clam_matrix.npy'), clam_matrix)

    if verbose:
        print(f"  HDBSCAN complete. Modal K={modal_k}, "
              f"agreement={k_agreement_rate:.2f}")

    return {
        'k_distribution': k_distribution,
        'modal_k': modal_k,
        'k_agreement_rate': k_agreement_rate,
        'clam_matrix': clam_matrix,
        'n_iterations_used': n_used,
        'iteration_labels': {
            'predicted': all_predicted_labels,
            'true': all_true_labels,
        },
        'noise_counts': noise_counts,
        'output_folder': output_folder,
    }


def _assign_noise_to_nearest(
    labels: np.ndarray,
    data: np.ndarray,
    centroids: np.ndarray
) -> np.ndarray:
    """Assign noise points (label == -1) to the nearest non-noise cluster.

    Parameters
    ----------
    labels : np.ndarray
        Cluster labels where -1 indicates noise. Shape: (n_samples,)
    data : np.ndarray
        Data points. Shape: (n_samples, n_features)
    centroids : np.ndarray
        Cluster centroids. Shape: (n_clusters, n_features)

    Returns
    -------
    np.ndarray
        Labels with noise points reassigned to nearest cluster.
    """
    noise_mask = labels == -1
    if not np.any(noise_mask):
        return labels.copy()

    result = labels.copy()
    noise_points = data[noise_mask]

    distances = np.linalg.norm(
        noise_points[:, np.newaxis, :] - centroids[np.newaxis, :, :],
        axis=2
    )
    nearest = np.argmin(distances, axis=1)
    result[noise_mask] = nearest

    return result


def _align_cluster_identities(
    unaligned_predictions: np.ndarray,
    iteration_centroids_list: List[np.ndarray],
    global_centroids_sorted: np.ndarray,
    k: int
) -> np.ndarray:
    """Align cluster identities across iterations to global centroids.
    
    Parameters
    ----------
    unaligned_predictions : np.ndarray
        Unaligned prediction matrix (n_iterations, n_test_samples)
    iteration_centroids_list : list of np.ndarray
        List of centroid arrays for each iteration
    global_centroids_sorted : np.ndarray
        Sorted global centroids (k, n_features)
    k : int
        Number of clusters
        
    Returns
    -------
    np.ndarray
        Aligned prediction matrix (n_iterations, n_test_samples)
        
    Notes
    -----
    This alignment ensures that cluster labels are consistent across iterations
    by matching iteration centroids to global centroids based on minimum distance.
    """
    n_iterations, n_test_samples = unaligned_predictions.shape
    aligned_predictions = np.zeros((n_iterations, n_test_samples))
    
    for iter_idx in range(n_iterations):
        iteration_centroids = iteration_centroids_list[iter_idx]
        unaligned_preds = unaligned_predictions[iter_idx, :].astype(int)
        
        # Calculate distance matrix
        dist_matrix = np.sqrt(
            np.sum(
                (iteration_centroids[:, np.newaxis, :] - 
                 global_centroids_sorted[np.newaxis, :, :])**2,
                axis=2
            )
        )
        
        # Find optimal mapping using greedy assignment
        mapping = np.zeros(k, dtype=int)
        used_global_indices = np.zeros(k, dtype=bool)
        
        for i in range(k):
            available_distances = dist_matrix[i, ~used_global_indices]
            if len(available_distances) == 0:
                mapping[i] = -1
                continue
            
            min_dist_idx_local = np.argmin(available_distances)
            original_global_idx = np.where(~used_global_indices)[0][min_dist_idx_local]
            mapping[i] = original_global_idx
            used_global_indices[original_global_idx] = True
        
        # Apply mapping to predictions
        transformed_preds = np.zeros_like(unaligned_preds, dtype=float)
        for point_idx, original_label in enumerate(unaligned_preds):
            if 0 <= original_label < k and mapping[original_label] != -1:
                transformed_preds[point_idx] = mapping[original_label]
            else:
                transformed_preds[point_idx] = np.nan
        
        aligned_predictions[iter_idx, :] = transformed_preds
    
    return aligned_predictions


def _generate_clam_matrix(
    n_samples: int,
    k: int,
    n_iterations: int,
    test_indices: np.ndarray,
    aligned_predictions: np.ndarray
) -> np.ndarray:
    """Generate CLAM (CLuster Assignment Matrix).
    
    The CLAM matrix is a fundamental output of ERICA analysis, where each
    element (i, j) represents the number of times sample i was assigned
    to cluster j across all iterations.
    
    Parameters
    ----------
    n_samples : int
        Total number of samples
    k : int
        Number of clusters
    n_iterations : int
        Number of iterations
    test_indices : np.ndarray
        Array of test indices for each iteration
    aligned_predictions : np.ndarray
        Aligned prediction matrix (n_iterations, n_test_samples)
        
    Returns
    -------
    np.ndarray
        CLAM matrix with shape (n_samples, k)
        
    Notes
    -----
    The CLAM matrix is the foundation for computing replicability metrics
    (CRI, WCRI, TWCRI). Higher values on the diagonal indicate more stable
    cluster assignments.
    """
    clam_matrix = np.zeros((n_samples, k))
    
    for iter_idx in range(n_iterations):
        iter_test_indices = test_indices[iter_idx]
        iter_aligned_preds = aligned_predictions[iter_idx, :]
        
        for i, sample_idx in enumerate(iter_test_indices):
            aligned_cluster_id = iter_aligned_preds[i]
            if not np.isnan(aligned_cluster_id):
                clam_matrix[sample_idx, int(aligned_cluster_id)] += 1
    
    return clam_matrix


