"""
02_coassignment_heatmap.py

Compute and plot sample x sample co-assignment similarity matrices
inspired by Monti et al. (2003).

For each dataset and K, we compute an assignment-probability inner
product from the CLAM matrix:
    coassign = clam_normed @ clam_normed.T  (n_samples x n_samples)

NOTE: This is the inner product of L1-normalised assignment vectors,
not the exact Monti consensus matrix (which conditions on pairwise
co-occurrence within the same subsample).  The two converge for
well-separated clusters.  See compute_coassignment() docstring.

Rows/columns are reordered via hierarchical clustering to reveal
block-diagonal structure, matching the Monti visual style.

Outputs: ../figures/literature_comparison/coassignment_{dataset}_kmeans.{pdf,png}
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import joblib
from sklearn.preprocessing import normalize
from scipy.cluster.hierarchy import linkage, leaves_list

# ---------------------------------------------------------------------------
# Style imports (from parent plotting_experiments directory)
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from style import set_publication_style, DOUBLE_COL

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, '..', 'results')
FIGURES_DIR = os.path.join(SCRIPT_DIR, '..', 'figures', 'literature_comparison')

# Datasets to process and their display names
DATASETS = [
    ('vdx_3gene',        'VDX 3-gene (real data)'),
    ('gauss4c_sigma0p1', 'Gaussian 4c σ=0.1'),
    ('gauss4c_sigma1p0', 'Gaussian 4c σ=1.0'),
    ('gauss4c_sigma10p0','Gaussian 4c σ=10.0'),
    ('moons_2d',         'Moons 2D'),
    ('blobs_2d',         'Blobs 2D'),
]

METHOD = 'kmeans'


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def compute_coassignment(clam_matrix):
    """Compute a co-assignment similarity matrix from a CLAM matrix.

    This computes the inner product of L1-normalised assignment probability
    vectors:  coassign[i, j] = sum_c  p(i, c) * p(j, c)

    This is NOT identical to the Monti et al. (2003) consensus matrix,
    which conditions on co-occurrence in the same subsample.  The Monti
    matrix requires iteration-level co-occurrence tracking that is not
    available from the aggregated CLAM matrix alone.  The inner-product
    approximation is a related quantity that:

      - IS symmetric and bounded in [0, 1].
      - DOES reveal block-diagonal structure for stable clusterings.
      - Has diagonal values < 1.0 for samples with uncertain assignments
        (the true Monti diagonal is always 1.0).

    For well-separated clusters the two matrices converge.

    Parameters
    ----------
    clam_matrix : np.ndarray, shape (n_samples, k)
        Raw cluster-assignment count matrix from ERICA.

    Returns
    -------
    coassign : np.ndarray, shape (n_samples, n_samples)
        Symmetric matrix in [0, 1] where entry (i, j) is the inner
        product of the assignment probability vectors of samples i and j.
    """
    clam = np.array(clam_matrix, dtype=float)
    # L1-normalize rows to proportions (zero rows stay zero)
    clam_normed = normalize(clam, norm='l1', axis=1)
    coassign = clam_normed @ clam_normed.T
    return coassign


def reorder_coassignment(coassign):
    """Reorder samples using hierarchical clustering to reveal block structure.

    Parameters
    ----------
    coassign : np.ndarray, shape (n, n)
        Symmetric co-assignment matrix.

    Returns
    -------
    reordered : np.ndarray
        Co-assignment matrix with rows and columns reordered.
    order : np.ndarray
        Integer index array that produces the reordered layout.
    """
    # Convert similarity to distance for linkage
    distance = 1.0 - coassign
    # Clip to avoid floating-point negatives on the diagonal
    np.clip(distance, 0.0, None, out=distance)
    # Condense upper triangle for scipy
    n = distance.shape[0]
    condensed = distance[np.triu_indices(n, k=1)]
    Z = linkage(condensed, method='average')
    order = leaves_list(Z)
    reordered = coassign[np.ix_(order, order)]
    return reordered, order


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def choose_k_values(true_k, available_ks):
    """Select three K values to display: true_k-1, true_k, true_k+1 when possible.

    Falls back gracefully when true_k is None or K values are missing.
    """
    if true_k is not None:
        candidates = [true_k - 1, true_k, true_k + 1]
    else:
        # No ground truth: pick three consecutive from the middle of available
        sorted_ks = sorted(available_ks)
        mid = len(sorted_ks) // 2
        start = max(0, mid - 1)
        candidates = sorted_ks[start:start + 3]
        return [k for k in candidates if k in available_ks]
    return [k for k in candidates if k in available_ks]


def plot_coassignment_panel(dataset_name, display_name, result_path, method=METHOD):
    """Generate a three-panel figure of co-assignment heatmaps for one dataset.

    Parameters
    ----------
    dataset_name : str
    display_name : str
    result_path : str
    method : str

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    data = joblib.load(result_path)
    er = data['erica_results']
    config = data['config']
    clam_matrices = er['clam_matrices']
    true_k = config.get('true_k')

    # Available K values for this method
    available_ks = sorted(k for (k, m) in clam_matrices if m == method)
    k_values = choose_k_values(true_k, available_ks)

    if not k_values:
        print(f'  [{dataset_name}] No K values available for method={method}, skipping.')
        return None

    n_panels = len(k_values)
    # Figure: three narrow square panels side by side
    panel_size = 2.4
    fig_width = panel_size * n_panels + 0.6  # extra for colorbar
    fig_height = panel_size + 0.9            # extra for title + colorbar label

    fig = plt.figure(figsize=(fig_width, fig_height))

    # Use gridspec for tight control; reserve a slim column for the colorbar
    gs = gridspec.GridSpec(
        1, n_panels + 1,
        figure=fig,
        width_ratios=[1] * n_panels + [0.08],
        wspace=0.12,
        left=0.06,
        right=0.92,
        top=0.85,
        bottom=0.08,
    )

    im_last = None
    for col, k in enumerate(k_values):
        ax = fig.add_subplot(gs[0, col])
        clam = clam_matrices.get((k, method))
        if clam is None:
            ax.set_visible(False)
            continue

        coassign = compute_coassignment(clam)
        reordered, _ = reorder_coassignment(coassign)

        im = ax.imshow(
            reordered,
            aspect='equal',
            cmap='Blues',
            vmin=0.0,
            vmax=1.0,
            interpolation='nearest',
        )
        im_last = im

        ax.set_title(f'K = {k}', fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])

        # Label leftmost panel
        if col == 0:
            ax.set_ylabel('Samples (reordered)', fontsize=9)
        ax.set_xlabel('Samples (reordered)', fontsize=9)

    # Shared colorbar in the reserved column
    if im_last is not None:
        cax = fig.add_subplot(gs[0, n_panels])
        cbar = fig.colorbar(im_last, cax=cax)
        cbar.set_label('Co-assignment\nsimilarity', fontsize=8)
        cbar.ax.tick_params(labelsize=7)

    fig.suptitle(
        f'Co-assignment similarity — {display_name} ({method})',
        fontsize=13,
        y=0.97,
    )

    return fig


# ---------------------------------------------------------------------------
# Save helper (literature_comparison sub-directory aware)
# ---------------------------------------------------------------------------

def save_figure_local(fig, name, out_dir=FIGURES_DIR, formats=('pdf', 'png')):
    """Save figure to the literature_comparison figures sub-directory."""
    os.makedirs(out_dir, exist_ok=True)
    for fmt in formats:
        path = os.path.join(out_dir, f'{name}.{fmt}')
        fig.savefig(path, format=fmt)
        print(f'  Saved: {path}')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    set_publication_style()

    for dataset_name, display_name in DATASETS:
        result_path = os.path.join(RESULTS_DIR, f'{dataset_name}.joblib')
        if not os.path.exists(result_path):
            print(f'[SKIP] Results not found: {result_path}')
            continue

        print(f'\nProcessing: {dataset_name}')
        fig = plot_coassignment_panel(dataset_name, display_name, result_path, method=METHOD)
        if fig is None:
            continue

        out_name = f'coassignment_{dataset_name}_{METHOD}'
        save_figure_local(fig, out_name)

    print('\nDone.')


if __name__ == '__main__':
    main()
