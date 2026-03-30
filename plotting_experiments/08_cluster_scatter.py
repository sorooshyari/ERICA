"""2D scatter plots of datasets colored by cluster assignment.

Shows how each method assigns samples to clusters for 2D datasets
(moons, circles, blobs_2d). Uses the primary cluster from the CLAM matrix
(argmax of each row = most frequent assignment across iterations).

Note: joblib is used here to load locally-generated ERICA results (not
untrusted external data). These files are produced by 02_run_erica_pipeline.py.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import set_publication_style, save_figure, DOUBLE_COL, SINGLE_COL

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')

CLUSTER_COLORS = ['#4477AA', '#EE6677', '#228833', '#CCBB44', '#66CCEE',
                  '#AA3377', '#BBBBBB', '#332288']


def get_primary_clusters(clam_matrix):
    """Get primary cluster assignment and confidence for each sample."""
    primary = np.argmax(clam_matrix, axis=1)
    row_sums = clam_matrix.sum(axis=1)
    row_sums[row_sums == 0] = 1
    confidence = clam_matrix[np.arange(len(clam_matrix)), primary] / row_sums
    return primary, confidence


def plot_dataset_overview(X, dataset_name):
    """Plot raw dataset with no clustering."""
    fig, ax = plt.subplots(figsize=(SINGLE_COL, SINGLE_COL))
    ax.scatter(X[:, 0], X[:, 1], s=10, c='gray', alpha=0.6)
    ax.set_title(f'{dataset_name} — Raw Data')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_aspect('equal')
    return fig


def plot_cluster_assignments(X, clam_matrix, k, method, dataset_name,
                              show_confidence=True):
    """Scatter plot colored by primary cluster assignment.

    Point size scales with confidence (how consistently this sample
    was assigned to its primary cluster across iterations).
    """
    primary, confidence = get_primary_clusters(clam_matrix)

    fig, ax = plt.subplots(figsize=(SINGLE_COL, SINGLE_COL))

    for c_idx in range(k):
        mask = primary == c_idx
        if not mask.any():
            continue
        color = CLUSTER_COLORS[c_idx % len(CLUSTER_COLORS)]

        if show_confidence:
            sizes = 10 + 40 * confidence[mask]
            ax.scatter(X[mask, 0], X[mask, 1], s=sizes, c=color,
                      label=f'C{c_idx+1}', alpha=0.7, edgecolors='none')
        else:
            ax.scatter(X[mask, 0], X[mask, 1], s=15, c=color,
                      label=f'C{c_idx+1}', alpha=0.7, edgecolors='none')

    ax.set_title(f'{dataset_name} — {method}, K={k}')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.legend(fontsize=7, loc='best', markerscale=1.5)
    ax.set_aspect('equal')
    return fig


def plot_method_comparison_scatter(X, er, k, dataset_name):
    """Side-by-side scatter for kmeans, agglomerative_ward, and HDBSCAN."""
    methods_to_plot = []

    for method in ['kmeans', 'agglomerative_ward']:
        key = (k, method)
        if key in er['clam_matrices']:
            clam = er['clam_matrices'][key]
            methods_to_plot.append((method, clam, k))

    hdb = er.get('auto_k', {}).get('hdbscan', {})
    if hdb.get('k_agreement_rate', 0) > 0 and 'clam_matrix' in hdb:
        methods_to_plot.append(('hdbscan', hdb['clam_matrix'], hdb['modal_k']))

    n = len(methods_to_plot)
    if n == 0:
        return None

    fig, axes = plt.subplots(1, n, figsize=(DOUBLE_COL, DOUBLE_COL / n))
    if n == 1:
        axes = [axes]

    for ax, (method, clam, method_k) in zip(axes, methods_to_plot):
        primary, confidence = get_primary_clusters(clam)

        for c_idx in range(method_k):
            mask = primary == c_idx
            if not mask.any():
                continue
            color = CLUSTER_COLORS[c_idx % len(CLUSTER_COLORS)]
            sizes = 10 + 40 * confidence[mask]
            ax.scatter(X[mask, 0], X[mask, 1], s=sizes, c=color,
                      label=f'C{c_idx+1}', alpha=0.7, edgecolors='none')

        title = f'HDBSCAN (K={method_k})' if method == 'hdbscan' else f'{method} (K={k})'
        ax.set_title(title, fontsize=10)
        ax.set_aspect('equal')
        if ax == axes[0]:
            ax.set_ylabel('Feature 2')
        ax.set_xlabel('Feature 1')
        ax.legend(fontsize=6, loc='best', markerscale=1.2)

    fig.suptitle(f'{dataset_name} — Method Comparison', fontsize=14, y=1.02)
    fig.tight_layout()
    return fig


def plot_confidence_scatter(X, clam_matrix, k, method, dataset_name):
    """Scatter where color = confidence (how stable the assignment is).

    Green = high confidence, red = low confidence (confused).
    """
    _, confidence = get_primary_clusters(clam_matrix)

    fig, ax = plt.subplots(figsize=(SINGLE_COL, SINGLE_COL))
    sc = ax.scatter(X[:, 0], X[:, 1], s=15, c=confidence, cmap='RdYlGn',
                    vmin=0.3, vmax=1.0, edgecolors='none')
    plt.colorbar(sc, ax=ax, label='Assignment confidence')
    ax.set_title(f'{dataset_name} — {method}, K={k}\nConfidence Map')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_aspect('equal')
    return fig


def main():
    set_publication_style()

    datasets_2d = ['moons_2d', 'circles_2d', 'blobs_2d']

    for dataset in datasets_2d:
        data_path = os.path.join(DATA_DIR, f'{dataset}.npz')
        if not os.path.exists(data_path):
            print(f'Skipping {dataset} (no data)')
            continue
        X = np.load(data_path, allow_pickle=True)['X']

        result_path = os.path.join(RESULTS_DIR, f'{dataset}.joblib')
        if not os.path.exists(result_path):
            print(f'Skipping {dataset} (no results)')
            continue
        r = joblib.load(result_path)
        er = r['erica_results']
        true_k = r['config']['true_k']

        print(f'\n{dataset} (true_k={true_k}):')

        # Raw data overview
        fig = plot_dataset_overview(X, dataset)
        save_figure(fig, f'scatter_raw_{dataset}')

        # Cluster assignments at true K
        k_to_show = true_k if true_k else 2
        for method in ['kmeans', 'agglomerative_ward']:
            key = (k_to_show, method)
            if key in er['clam_matrices']:
                fig = plot_cluster_assignments(
                    X, er['clam_matrices'][key], k_to_show, method, dataset)
                save_figure(fig, f'scatter_{dataset}_{method}_k{k_to_show}')

                fig = plot_confidence_scatter(
                    X, er['clam_matrices'][key], k_to_show, method, dataset)
                save_figure(fig, f'confidence_{dataset}_{method}_k{k_to_show}')

        # Method comparison side-by-side
        fig = plot_method_comparison_scatter(X, er, k_to_show, dataset)
        if fig:
            save_figure(fig, f'scatter_compare_{dataset}_k{k_to_show}')

        # Moons: show K-Means at K=2 (correct) vs K=4 (what K* selected)
        if dataset == 'moons_2d':
            print('  Moons special: K-Means K=2 vs K=4')
            for k_show in [2, 4]:
                key = (k_show, 'kmeans')
                if key in er['clam_matrices']:
                    fig = plot_cluster_assignments(
                        X, er['clam_matrices'][key], k_show, 'kmeans', dataset)
                    save_figure(fig, f'scatter_{dataset}_kmeans_k{k_show}')

        # Circles: show all K values for both methods
        if dataset == 'circles_2d':
            print('  Circles special: K=2,3,4 for both methods')
            for k_show in [2, 3, 4]:
                for method in ['kmeans', 'agglomerative_ward']:
                    key = (k_show, method)
                    if key in er['clam_matrices']:
                        fig = plot_cluster_assignments(
                            X, er['clam_matrices'][key], k_show, method, dataset)
                        save_figure(fig, f'scatter_{dataset}_{method}_k{k_show}')

    print('\nCluster scatter plots complete.')


if __name__ == '__main__':
    main()
