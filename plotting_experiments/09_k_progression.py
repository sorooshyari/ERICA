"""K progression scatter plots for moons and circles.

Shows how K-Means, Agglomerative Ward, and Agglomerative Single linkage
cluster assignments change as K steps from 2 to 4. Each dataset gets a
3x3 grid: rows = methods, columns = K values.

Note: Uses joblib to load locally-generated ERICA results from 02_run_erica_pipeline.py.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import set_publication_style, save_figure, DOUBLE_COL

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')

CLUSTER_COLORS = ['#4477AA', '#EE6677', '#228833', '#CCBB44', '#66CCEE',
                  '#AA3377', '#BBBBBB', '#332288']

METHODS = ['kmeans', 'agglomerative_ward', 'agglomerative_single']
METHOD_LABELS = {
    'kmeans': 'K-Means',
    'agglomerative_ward': 'Ward Linkage',
    'agglomerative_single': 'Single Linkage',
}
K_VALUES = [2, 3, 4]


def get_primary_clusters(clam_matrix):
    """Get primary cluster assignment and confidence."""
    primary = np.argmax(clam_matrix, axis=1)
    row_sums = clam_matrix.sum(axis=1)
    row_sums[row_sums == 0] = 1
    confidence = clam_matrix[np.arange(len(clam_matrix)), primary] / row_sums
    return primary, confidence


def plot_k_progression(X, er, dataset_name):
    """Create 3x3 grid: rows=methods, cols=K values."""
    fig, axes = plt.subplots(
        len(METHODS), len(K_VALUES),
        figsize=(DOUBLE_COL, DOUBLE_COL),
        sharex=True, sharey=True,
    )

    for row, method in enumerate(METHODS):
        for col, k in enumerate(K_VALUES):
            ax = axes[row, col]
            key = (k, method)

            if key not in er['clam_matrices']:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                       transform=ax.transAxes, fontsize=9, color='gray')
                ax.set_facecolor('#f8f8f8')
            else:
                clam = er['clam_matrices'][key]
                primary, confidence = get_primary_clusters(clam)

                for c_idx in range(k):
                    mask = primary == c_idx
                    if not mask.any():
                        continue
                    color = CLUSTER_COLORS[c_idx % len(CLUSTER_COLORS)]
                    sizes = 8 + 30 * confidence[mask]
                    ax.scatter(X[mask, 0], X[mask, 1], s=sizes, c=color,
                              alpha=0.7, edgecolors='none')

            if row == 0:
                ax.set_title(f'K = {k}', fontsize=12, fontweight='bold')
            if col == 0:
                ax.set_ylabel(METHOD_LABELS[method], fontsize=10)
            if row == len(METHODS) - 1:
                ax.set_xlabel('Feature 1', fontsize=9)

            ax.set_aspect('equal')
            ax.tick_params(labelsize=7)

    fig.suptitle(f'{dataset_name} — Clustering Progression (K=2 to K=4)',
                 fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    return fig


def main():
    set_publication_style()

    for dataset in ['moons_2d', 'circles_2d']:
        data_path = os.path.join(DATA_DIR, f'{dataset}.npz')
        result_path = os.path.join(RESULTS_DIR, f'{dataset}.joblib')

        if not os.path.exists(data_path) or not os.path.exists(result_path):
            print(f'Skipping {dataset}')
            continue

        X = np.load(data_path, allow_pickle=True)['X']
        r = joblib.load(result_path)
        er = r['erica_results']

        available = set()
        for (k, m) in er['clam_matrices'].keys():
            available.add(m)
        print(f'{dataset}: available methods = {sorted(available)}')

        fig = plot_k_progression(X, er, dataset)
        save_figure(fig, f'k_progression_{dataset}')

    print('\nK progression plots complete.')


if __name__ == '__main__':
    main()
