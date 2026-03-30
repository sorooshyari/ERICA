"""Full K=2..6 progression scatter plots for all 2D datasets.

Generates a methods-x-K grid for every 2D dataset, saved to figures/k_progressions/.
Each grid shows K-Means, Ward, and Single linkage across K=2 to K=6.

For high-dimensional datasets (>2D), generates CLAM heatmap progressions instead.

Note: joblib is used here to load locally-generated ERICA results (not
untrusted external data). These files are produced by 02_run_erica_pipeline.py.

ERICA functions used:
    - ERICA class: orchestrates subsampling, clustering, and metric computation
    - erica.get_results()['clam_matrices'][(k, method)]: CLAM matrix per K/method
    - erica.get_results()['auto_k']['hdbscan']: HDBSCAN auto-K results
    - erica.get_results()['metrics'][k][method]: per-K per-method metrics dict
    - erica.get_results()['k_star']['TWCRI'][method]: optimal K* per method
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import set_publication_style, save_figure, DOUBLE_COL, CMAP_SEQ

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')
PROG_DIR = os.path.join(SCRIPT_DIR, 'figures', 'k_progressions')

CLUSTER_COLORS = ['#4477AA', '#EE6677', '#228833', '#CCBB44', '#66CCEE',
                  '#AA3377', '#BBBBBB', '#332288']

METHODS = ['kmeans', 'agglomerative_ward', 'agglomerative_single']
METHOD_LABELS = {
    'kmeans': 'K-Means',
    'agglomerative_ward': 'Ward',
    'agglomerative_single': 'Single',
}
K_VALUES = [2, 3, 4, 5, 6]


def get_primary_clusters(clam_matrix):
    """Primary cluster = argmax of CLAM row. Confidence = max_count / row_sum."""
    primary = np.argmax(clam_matrix, axis=1)
    row_sums = clam_matrix.sum(axis=1)
    row_sums[row_sums == 0] = 1
    confidence = clam_matrix[np.arange(len(clam_matrix)), primary] / row_sums
    return primary, confidence


def sort_clam(clam_matrix):
    """Sort CLAM rows by primary cluster then assignment strength."""
    primary = np.argmax(clam_matrix, axis=1)
    row_sums = clam_matrix.sum(axis=1)
    row_sums[row_sums == 0] = 1
    strength = clam_matrix[np.arange(len(clam_matrix)), primary] / row_sums
    sort_idx = np.lexsort((-strength, primary))
    return clam_matrix[sort_idx]


def plot_scatter_progression(X, er, dataset_name, true_k, k_star_info):
    """3-row x 5-col grid: scatter plots colored by cluster assignment."""
    fig, axes = plt.subplots(
        len(METHODS), len(K_VALUES),
        figsize=(DOUBLE_COL + 2, DOUBLE_COL - 0.5),
        sharex=True, sharey=True,
    )

    for row, method in enumerate(METHODS):
        for col, k in enumerate(K_VALUES):
            ax = axes[row, col]
            key = (k, method)

            if key not in er['clam_matrices']:
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center',
                       transform=ax.transAxes, color='gray', fontsize=9)
                ax.set_facecolor('#f8f8f8')
            else:
                clam = er['clam_matrices'][key]
                primary, confidence = get_primary_clusters(clam)

                for c_idx in range(k):
                    mask = primary == c_idx
                    if not mask.any():
                        continue
                    color = CLUSTER_COLORS[c_idx % len(CLUSTER_COLORS)]
                    sizes = 6 + 25 * confidence[mask]
                    ax.scatter(X[mask, 0], X[mask, 1], s=sizes, c=color,
                              alpha=0.7, edgecolors='none')

            if row == 0:
                star = ' (true)' if k == true_k else ''
                ax.set_title(f'K={k}{star}', fontsize=10, fontweight='bold')

            if col == 0:
                ks = k_star_info.get(method, '?')
                ax.set_ylabel(f'{METHOD_LABELS[method]}\nK*={ks}', fontsize=9)

            # Highlight K* panels with orange border
            ks_val = k_star_info.get(method)
            if ks_val == k:
                for spine in ax.spines.values():
                    spine.set_edgecolor('#D55E00')
                    spine.set_linewidth(2.5)
                    spine.set_visible(True)

            ax.set_aspect('equal')
            ax.tick_params(labelsize=6)

    fig.suptitle(f'{dataset_name}  (true K={true_k})',
                 fontsize=13, fontweight='bold', y=1.01)
    fig.tight_layout()
    return fig


def plot_clam_progression(er, dataset_name, true_k, k_star_info):
    """3-row x 5-col grid: sorted CLAM heatmaps for high-dimensional data."""
    fig, axes = plt.subplots(
        len(METHODS), len(K_VALUES),
        figsize=(DOUBLE_COL + 2, DOUBLE_COL - 0.5),
    )

    for row, method in enumerate(METHODS):
        for col, k in enumerate(K_VALUES):
            ax = axes[row, col]
            key = (k, method)

            if key not in er['clam_matrices']:
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center',
                       transform=ax.transAxes, color='gray', fontsize=9)
            else:
                clam = sort_clam(er['clam_matrices'][key])
                ax.imshow(clam, aspect='auto', cmap=CMAP_SEQ, interpolation='nearest')

            if row == 0:
                star = ' (true)' if k == true_k else ''
                ax.set_title(f'K={k}{star}', fontsize=10, fontweight='bold')
            if col == 0:
                ks = k_star_info.get(method, '?')
                ax.set_ylabel(f'{METHOD_LABELS[method]}\nK*={ks}', fontsize=9)

            ks_val = k_star_info.get(method)
            if ks_val == k:
                for spine in ax.spines.values():
                    spine.set_edgecolor('#D55E00')
                    spine.set_linewidth(2.5)
                    spine.set_visible(True)

            ax.tick_params(labelsize=6)

    fig.suptitle(f'{dataset_name}  (true K={true_k}) -- CLAM Heatmaps',
                 fontsize=13, fontweight='bold', y=1.01)
    fig.tight_layout()
    return fig


def main():
    set_publication_style()
    os.makedirs(PROG_DIR, exist_ok=True)

    all_datasets = sorted([f.replace('.npz', '') for f in os.listdir(DATA_DIR)
                           if f.endswith('.npz')])

    for dataset in all_datasets:
        result_path = os.path.join(RESULTS_DIR, f'{dataset}.joblib')
        if not os.path.exists(result_path):
            print(f'Skipping {dataset} (no results)')
            continue

        X = np.load(os.path.join(DATA_DIR, f'{dataset}.npz'), allow_pickle=True)['X']
        r = joblib.load(result_path)
        er = r['erica_results']
        true_k = r['config']['true_k']
        n_features = X.shape[1]

        # Get K* for each method (TWCRI)
        k_star_info = {}
        ks = er.get('k_star', {}).get('TWCRI', {})
        for m in METHODS:
            k_star_info[m] = ks.get(m, '?')

        print(f'{dataset} ({X.shape[0]}x{n_features}, true_k={true_k}):')

        if n_features == 2:
            fig = plot_scatter_progression(X, er, dataset, true_k, k_star_info)
        else:
            fig = plot_clam_progression(er, dataset, true_k, k_star_info)

        # Save to k_progressions subfolder
        out_name = f'k_progression_{dataset}'
        for fmt in ('pdf', 'png'):
            path = os.path.join(PROG_DIR, f'{out_name}.{fmt}')
            fig.savefig(path, format=fmt, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f'  Saved: {out_name}')

    print(f'\nAll progressions saved to figures/k_progressions/')
    print(f'Total: {len([f for f in os.listdir(PROG_DIR) if f.endswith(".pdf")])} PDFs')


if __name__ == '__main__':
    main()
