"""
03_clam_heatmaps.py

Create CLAM matrix heatmap visualizations for ERICA plotting experiments.
Generates sorted and unsorted single-panel views plus a multi-panel comparison.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import set_publication_style, save_figure, CMAP_SEQ, DOUBLE_COL

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def sort_clam(clam_matrix):
    """Sort a CLAM matrix by primary cluster assignment and strength.

    Parameters
    ----------
    clam_matrix : np.ndarray, shape (n_samples, k)
        Raw co-assignment counts matrix.

    Returns
    -------
    sorted_matrix : np.ndarray
        Matrix with rows reordered.
    sort_indices : np.ndarray
        Integer indices that produced the sorted order.
    """
    clam = np.array(clam_matrix, dtype=float)
    row_sums = clam.sum(axis=1)

    # Primary cluster: column with the largest count per row
    primary = np.argmax(clam, axis=1)

    # Strength: max count / row sum (0 for zero-sum rows)
    strength = np.where(row_sums > 0, clam.max(axis=1) / row_sums, 0.0)

    # Sort: primary cluster ascending, then strength descending within cluster
    sort_indices = np.lexsort((-strength, primary))

    sorted_matrix = clam[sort_indices]
    return sorted_matrix, sort_indices


def plot_clam(clam_matrix, title, sorted_view=True, ax=None):
    """Plot a single CLAM matrix as a heatmap.

    Parameters
    ----------
    clam_matrix : np.ndarray, shape (n_samples, k)
    title : str
    sorted_view : bool
        Whether to sort rows before plotting.
    ax : matplotlib Axes or None
        If None a new figure + axes is created and the figure is returned.

    Returns
    -------
    fig : matplotlib Figure (only when ax is None; otherwise None)
    """
    clam = np.array(clam_matrix, dtype=float)

    if sorted_view:
        clam, _ = sort_clam(clam)

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(4, 5))
    else:
        fig = None

    k = clam.shape[1]
    im = ax.imshow(clam, aspect='auto', cmap=CMAP_SEQ, interpolation='nearest')

    # Colorbar
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Co-assignment count')

    # Axes labels
    ax.set_xticks(range(k))
    ax.set_xticklabels([f'Cluster {i + 1}' for i in range(k)], fontsize=9)
    ax.set_ylabel('Sample')
    ax.set_xlabel('Cluster assignment')
    ax.set_title(title)

    if standalone:
        return fig


def plot_clam_multipanel(clam_matrices, k_values, method, dataset_name):
    """Plot sorted CLAM matrices side by side for multiple K values.

    Parameters
    ----------
    clam_matrices : dict
        Mapping (k, method) -> np.ndarray.
    k_values : list of int
    method : str
    dataset_name : str

    Returns
    -------
    fig : matplotlib Figure
    """
    n = len(k_values)
    fig, axes = plt.subplots(1, n, figsize=(DOUBLE_COL, 4.5))
    if n == 1:
        axes = [axes]

    for ax, k in zip(axes, k_values):
        cm = clam_matrices.get((k, method))
        if cm is None:
            ax.set_visible(False)
            continue
        plot_clam(cm, title=f'K = {k}', sorted_view=True, ax=ax)

    fig.suptitle(
        f'CLAM matrices — {dataset_name} ({method})',
        fontsize=12,
        y=1.01,
    )
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    set_publication_style()

    # ------------------------------------------------------------------
    # Load vdx_3gene results
    # ------------------------------------------------------------------
    result_path = os.path.join(RESULTS_DIR, "vdx_3gene.joblib")
    data = joblib.load(result_path)
    er = data["erica_results"]
    clam_matrices = er["clam_matrices"]

    print("Loaded vdx_3gene results.")
    print(f"  clam_matrices keys: {list(clam_matrices.keys())[:6]} ...")

    # ------------------------------------------------------------------
    # 1. Sorted single panel — K=3 kmeans
    # ------------------------------------------------------------------
    print("\nGenerating clam_sorted_vdx_3gene_kmeans_k3 ...")
    cm_k3 = clam_matrices[(3, 'kmeans')]
    fig = plot_clam(cm_k3, title='CLAM (sorted) — VDX 3-gene, K=3, K-Means', sorted_view=True)
    save_figure(fig, 'clam_sorted_vdx_3gene_kmeans_k3')

    # ------------------------------------------------------------------
    # 2. Unsorted (raw) single panel — K=3 kmeans
    # ------------------------------------------------------------------
    print("\nGenerating clam_raw_vdx_3gene_kmeans_k3 ...")
    fig = plot_clam(cm_k3, title='CLAM (raw) — VDX 3-gene, K=3, K-Means', sorted_view=False)
    save_figure(fig, 'clam_raw_vdx_3gene_kmeans_k3')

    # ------------------------------------------------------------------
    # 3. Multi-panel K=2,3,4 kmeans
    # ------------------------------------------------------------------
    print("\nGenerating clam_multipanel_vdx_3gene_kmeans ...")
    fig = plot_clam_multipanel(
        clam_matrices,
        k_values=[2, 3, 4],
        method='kmeans',
        dataset_name='VDX 3-gene',
    )
    save_figure(fig, 'clam_multipanel_vdx_3gene_kmeans')

    # ------------------------------------------------------------------
    # 4. HDBSCAN at modal K (only if k_agreement_rate > 0)
    # ------------------------------------------------------------------
    hdb = er.get('auto_k', {}).get('hdbscan', {})
    k_agreement = hdb.get('k_agreement_rate', 0)
    print(f"\nHDBSCAN k_agreement_rate = {k_agreement}")

    if k_agreement is not None and k_agreement > 0:
        print("Generating clam_sorted_vdx_3gene_hdbscan ...")
        cm_hdb = hdb['clam_matrix']
        modal_k = hdb.get('modal_k', '?')
        fig = plot_clam(
            cm_hdb,
            title=f'CLAM (sorted) — VDX 3-gene, HDBSCAN (modal K={modal_k})',
            sorted_view=True,
        )
        save_figure(fig, 'clam_sorted_vdx_3gene_hdbscan')
    else:
        print("  Skipping HDBSCAN panel (k_agreement_rate == 0 or None).")

    print("\nDone.")


if __name__ == "__main__":
    main()
