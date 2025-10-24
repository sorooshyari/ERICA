"""
Core ERICA class for clustering replicability analysis.

This module provides the main ERICA class that orchestrates the entire
clustering replicability analysis workflow.
"""

import os
import numpy as np
import pandas as pd
from typing import Union, List, Dict, Optional
from datetime import datetime

from erica.clustering import (
    kmeans_clustering,
    agglomerative_clustering,
    iterative_clustering_subsampling,
)
from erica.metrics import compute_metrics_for_clam
from erica.data import prepare_samples_array, validate_dataset
from erica.utils import set_deterministic_mode, compute_config_hash


class ERICA:
    """Main class for ERICA (Evaluating Replicability via Iterative Clustering Assignments)."""

    def __init__(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        k_range: Optional[List[int]] = None,
        n_iterations: int = 200,
        train_percent: float = 0.8,
        method: str = 'both',
        linkages: Optional[List[str]] = None,
        random_seed: int = 123,
        output_dir: str = './erica_output',
        verbose: bool = True
    ):
        """Initialize ERICA analysis."""
        self.data = data
        self.k_range = k_range or [2, 3, 4, 5]
        self.n_iterations = n_iterations
        self.train_percent = train_percent
        self.method = method
        self.linkages = linkages or ['single', 'ward']
        self.random_seed = random_seed
        self.output_dir = output_dir
        self.verbose = verbose

        # Deterministic reproducibility
        set_deterministic_mode(random_seed)

        # Prepare and validate data
        self.samples_array = prepare_samples_array(data)
        self.n_samples, self.n_features = self.samples_array.shape
        validate_dataset(self.samples_array, min(self.k_range), self.train_percent)

        # Initialize storage
        self.results_ = {}
        self.clam_matrices_ = {}
        self.metrics_ = {}
        self.output_folders_ = []

        os.makedirs(output_dir, exist_ok=True)

        if self.verbose:
            print(f"ERICA initialized:")
            print(f"  Data shape: {self.samples_array.shape}")
            print(f"  K range: {self.k_range}")
            print(f"  Iterations: {self.n_iterations}")
            print(f"  Method: {self.method}")
            print(f"  Random seed: {self.random_seed}")

    def run(self) -> Dict:
        """Run the complete ERICA analysis."""
        if self.verbose:
            print(f"\nStarting ERICA analysis at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(self.output_dir, f"erica_run_{timestamp}")
        os.makedirs(run_dir, exist_ok=True)

        # Step 1: Iterative clustering subsampling
        if self.verbose:
            print(f"\n[1/3] Performing iterative clustering subsampling...")
        train_size = int(self.n_samples * self.train_percent)
        subsamples_folder, indices_folder = iterative_clustering_subsampling(
            samples_array=self.samples_array,
            num_samples=self.n_samples,
            num_iterations=self.n_iterations,
            subsample_size_train=train_size,
            base_save_folder_str=run_dir,
            verbose=self.verbose
        )

        # Step 2: Clustering for each k and method
        if self.verbose:
            print(f"\n[2/3] Running clustering analysis...")
        methods_to_run = ['kmeans', 'agglomerative'] if self.method == 'both' else [self.method]

        for k in self.k_range:
            for method_name in methods_to_run:
                if method_name == 'kmeans':
                    if self.verbose:
                        print(f"  Running K-Means clustering for k={k}...")
                    result = kmeans_clustering(
                        samples_array=self.samples_array,
                        k=k,
                        n_iterations=self.n_iterations,
                        indices_folder=indices_folder,
                        output_dir=run_dir,
                        verbose=self.verbose
                    )
                    self.clam_matrices_[(k, 'kmeans')] = result['clam_matrix']
                    self.results_[(k, 'kmeans')] = result

                elif method_name == 'agglomerative':
                    for linkage in self.linkages:
                        if self.verbose:
                            print(f"  Running Agglomerative clustering (linkage={linkage}) for k={k}...")
                        result = agglomerative_clustering(
                            samples_array=self.samples_array,
                            k=k,
                            linkage=linkage,
                            n_iterations=self.n_iterations,
                            indices_folder=indices_folder,
                            output_dir=run_dir,
                            verbose=self.verbose
                        )
                        method_key = f'agglomerative_{linkage}'
                        self.clam_matrices_[(k, method_key)] = result['clam_matrix']
                        self.results_[(k, method_key)] = result

        # Step 3: Compute metrics
        if self.verbose:
            print(f"\n[3/3] Computing metrics...")
        self.metrics_ = self._compute_all_metrics()

        # Store output folder
        self.output_folders_.append(run_dir)

        if self.verbose:
            print(f"\nERICA analysis complete!")
            print(f"Results saved to: {run_dir}")
            print(f"Total configurations analyzed: {len(self.results_)}")

        return self.get_results()


    def _compute_all_metrics(self) -> Dict:
        """Compute CRI, WCRI, TWCRI metrics for all results."""
        metrics_by_k = {}
        for (k, method_name), result in self.results_.items():
            clam_matrix = result['clam_matrix']
            metrics = compute_metrics_for_clam(clam_matrix, k)
            if k not in metrics_by_k:
                metrics_by_k[k] = {}
            metrics_by_k[k][method_name] = metrics
        return metrics_by_k


    def get_results(self) -> Dict:
        """Get all analysis results."""
        return {
            'clam_matrices': self.clam_matrices_,
            'metrics': self.metrics_,
            'config': self._get_config_dict(),
            'output_folders': self.output_folders_,
            'results': self.results_
        }

    def get_clam_matrix(self, k: int, method: str = 'kmeans') -> Optional[np.ndarray]:
        """Get CLAM matrix for specific k and method."""
        return self.clam_matrices_.get((k, method))

    def get_metrics(self, k: Optional[int] = None) -> Dict:
        """Get computed metrics."""
        if k is not None:
            return self.metrics_.get(k, {})
        return self.metrics_

    def _get_config_dict(self) -> Dict:
        """Return ERICA configuration as dictionary."""
        return {
            'k_range': self.k_range,
            'n_iterations': self.n_iterations,
            'train_percent': self.train_percent,
            'method': self.method,
            'linkages': self.linkages,
            'random_seed': self.random_seed,
            'n_samples': self.n_samples,
            'n_features': self.n_features,
            'config_hash': compute_config_hash({
                'k_range': self.k_range,
                'n_iterations': self.n_iterations,
                'train_percent': self.train_percent,
                'method': self.method,
                'linkages': self.linkages,
                'random_seed': self.random_seed,
            })
        }

    def __repr__(self) -> str:
        return (
            f"ERICA(n_samples={self.n_samples}, n_features={self.n_features}, "
            f"k_range={self.k_range}, n_iterations={self.n_iterations}, "
            f"method='{self.method}')"
        )

