"""K-progression CLAM heatmap grids for the Gaussian mixture study.

For each sigma level, creates a 3-row x 5-col grid of sorted CLAM heatmaps
(rows = kmeans / ward / single, cols = K=2..6). K* panels receive orange
borders. HDBSCAN is shown as a separate single panel below.

Note: Uses joblib to load locally-generated ERICA results from 02_run_erica.py.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from style import set_publication_style, DOUBLE_COL, CMAP_SEQ, METHOD_COLORS

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, '..', 'results')
FIGURES_DIR = os.path.join(SCRIPT_DIR, '..', 'figures', 'gaussian_study')

SIGMAS = [
    ('gauss4c_sigma0p01', 0.01),
    ('gauss4c_sigma0p1',  0.1),
    ('gauss4c_sigma1p0',  1.0),
    ('gauss4c_sigma10p0', 10.0),
]

METHODS = ['kmeans', 'agglomerative_ward', 'agglomerative_single']
METHOD_LABELS = {
    'kmeans': 'K-Means',
    'agglomerative_ward': 'Ward',
    'agglomerative_single': 'Single',
}
K_VALUES = [2, 3, 4, 5, 6]
TRUE_K = 4


def sort_clam(clam):
    """Sort CLAM matrix: argmax for primary cluster, then by (-confidence).

    Returns a copy with rows reordered so that samples within each primary
    cluster are grouped together, with the most confident assignments first.
    """
    primary = np.argmax(clam, axis=1)
    row_sums = clam.sum(axis=1)
    row_sums[row_sums == 0] = 1
    confidence = clam[np.arange(len(clam)), primary] / row_sums
    # lexsort: last key is primary (sort first by primary ascending,
    # then by -confidence so highest confidence comes first)
    order = np.lexsort((-confidence, primary))
    return clam[order]


def save_fig(fig, name, formats=('pdf', 'png')):
    """Save figure to the gaussian_study figures directory."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    for fmt in formats:
        path = os.path.join(FIGURES_DIR, f'{name}.{fmt}')
        fig.savefig(path, format=fmt, bbox_inches='tight', dpi=300)
        print(f'  Saved: {path}')
    plt.close(fig)


def plot_clam_heatmap(ax, clam, is_kstar=False):
    """Plot a sorted CLAM heatmap in the given axes."""
    sorted_clam = sort_clam(clam)
    ax.imshow(sorted_clam, aspect='auto', cmap=CMAP_SEQ,
              vmin=0, vmax=1, interpolation='nearest')
    ax.set_xticks([])
    ax.set_yticks([])

    if is_kstar:
        for spine in ax.spines.values():
            spine.set_edgecolor('#E69F00')
            spine.set_linewidth(3)
            spine.set_visible(True)


def plot_sigma_grid(er, sigma, dataset_name):
    """Create grid figure for one sigma level.

    Main grid: 3 rows (methods) x 5 cols (K=2..6).
    Below: single HDBSCAN panel (if available).
    """
    hdbscan_data = er.get('auto_k', {}).get('hdbscan', None)
    has_hdbscan = hdbscan_data is not None and hdbscan_data.get('clam_matrix') is not None

    # Figure layout
    n_grid_rows = 3
    if has_hdbscan:
        fig = plt.figure(figsize=(DOUBLE_COL + 1.5, 6.5))
        gs = gridspec.GridSpec(
            n_grid_rows + 2, len(K_VALUES),
            figure=fig,
            height_ratios=[1, 1, 1, 0.15, 0.7],
            hspace=0.35, wspace=0.12,
        )
    else:
        fig = plt.figure(figsize=(DOUBLE_COL + 1.5, 5.0))
        gs = gridspec.GridSpec(
            n_grid_rows, len(K_VALUES),
            figure=fig,
            hspace=0.25, wspace=0.12,
        )

    # Get K* values for TWCRI
    k_star_twcri = er.get('k_star', {}).get('TWCRI', {})

    # Main grid
    for row, method in enumerate(METHODS):
        kstar_for_method = k_star_twcri.get(method, None)
        for col, k in enumerate(K_VALUES):
            ax = fig.add_subplot(gs[row, col])
            key = (k, method)
            if key in er['clam_matrices']:
                clam = er['clam_matrices'][key]
                is_kstar = (k == kstar_for_method)
                plot_clam_heatmap(ax, clam, is_kstar=is_kstar)
            else:
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center',
                        transform=ax.transAxes, fontsize=9, color='gray')
                ax.set_facecolor('#f8f8f8')

            if row == 0:
                ax.set_title(f'K={k}', fontsize=10, fontweight='bold')
            if col == 0:
                ax.set_ylabel(METHOD_LABELS[method], fontsize=10)

    # HDBSCAN panel
    if has_hdbscan:
        modal_k = int(hdbscan_data['modal_k'])
        clam_hdb = hdbscan_data['clam_matrix']
        # Span 2 central columns
        col_start = 1
        col_end = 4
        ax_hdb = fig.add_subplot(gs[4, col_start:col_end])
        sorted_clam_hdb = sort_clam(clam_hdb)
        ax_hdb.imshow(sorted_clam_hdb, aspect='auto', cmap=CMAP_SEQ,
                      vmin=0, vmax=1, interpolation='nearest')
        ax_hdb.set_xticks([])
        ax_hdb.set_yticks([])
        ax_hdb.set_ylabel('HDBSCAN', fontsize=10)
        # Orange border for HDBSCAN
        for spine in ax_hdb.spines.values():
            spine.set_edgecolor(METHOD_COLORS['hdbscan'])
            spine.set_linewidth(2)
            spine.set_visible(True)
        ax_hdb.set_title(
            f'HDBSCAN (modal K={modal_k}, '
            f'agreement={hdbscan_data["k_agreement_rate"]:.0%})',
            fontsize=9,
        )

    fig.suptitle(
        f'CLAM Heatmaps \u2014 \u03c3={sigma}, true K={TRUE_K}',
        fontsize=13, fontweight='bold', y=0.98,
    )
    return fig


def main():
    set_publication_style()
    os.makedirs(FIGURES_DIR, exist_ok=True)

    for fname, sigma in SIGMAS:
        result_path = os.path.join(RESULTS_DIR, f'{fname}.joblib')
        if not os.path.exists(result_path):
            print(f'Skipping {fname}: file not found')
            continue

        print(f'Loading {fname} (sigma={sigma}) ...')
        r = joblib.load(result_path)
        er = r['erica_results']

        fig = plot_sigma_grid(er, sigma, fname)
        save_fig(fig, f'k_progression_{fname}')

    print('\nK-progression grids complete.')


if __name__ == '__main__':
    main()
