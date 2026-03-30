"""
04_stability_strips.py

Create per-sample stability strip plots: horizontal stacked bars showing
cluster assignment proportions, sorted by Shannon entropy (most stable first).
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import set_publication_style, save_figure, SINGLE_COL

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")

# Qualitative colorblind-friendly palette
CLUSTER_COLORS = [
    '#4477AA', '#EE6677', '#228833', '#CCBB44',
    '#66CCEE', '#AA3377', '#BBBBBB', '#332288',
]

MAX_SAMPLES = 100


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def normalize_clam(clam_matrix):
    """Normalize each CLAM row to proportions; skip zero-sum rows.

    Parameters
    ----------
    clam_matrix : np.ndarray, shape (n_samples, k)

    Returns
    -------
    proportions : np.ndarray, shape (n_samples, k)
        Rows with zero sum are left as zeros.
    valid_mask : np.ndarray of bool, shape (n_samples,)
        True where row sum > 0.
    """
    clam = np.array(clam_matrix, dtype=float)
    row_sums = clam.sum(axis=1)
    valid_mask = row_sums > 0
    proportions = np.zeros_like(clam)
    proportions[valid_mask] = clam[valid_mask] / row_sums[valid_mask, np.newaxis]
    return proportions, valid_mask


def shannon_entropy(proportions):
    """Compute per-row Shannon entropy (bits) from a proportion matrix.

    p = 0 contributes 0 to the sum.

    Parameters
    ----------
    proportions : np.ndarray, shape (n_samples, k)

    Returns
    -------
    entropy : np.ndarray, shape (n_samples,)
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        log_p = np.where(proportions > 0, np.log2(proportions), 0.0)
    return -np.sum(proportions * log_p, axis=1)


def plot_stability_strips(clam_matrix, title, max_samples=MAX_SAMPLES):
    """Create a horizontal stacked-bar stability strip plot.

    Each bar represents one sample. Bar segments show the proportion of
    subsample iterations in which the sample was assigned to each cluster.
    Bars are sorted by Shannon entropy ascending (most stable at top).

    Parameters
    ----------
    clam_matrix : np.ndarray, shape (n_samples, k)
    title : str
    max_samples : int
        Maximum number of samples to display.

    Returns
    -------
    fig : matplotlib Figure
    """
    proportions, valid_mask = normalize_clam(clam_matrix)
    entropy = shannon_entropy(proportions)

    n_samples, k = proportions.shape

    # Sort ascending by entropy (most stable = lowest entropy = top of plot)
    sort_order = np.argsort(entropy)

    # Subsample if needed (keep the most-stable end of the sorted list)
    if n_samples > max_samples:
        sort_order = sort_order[:max_samples]

    props_sorted = proportions[sort_order]
    ent_sorted = entropy[sort_order]
    n_display = len(sort_order)

    # Figure height scales with number of samples (min 2 in, max 8 in)
    bar_height = 0.6
    fig_height = max(2.0, min(8.0, n_display * bar_height * 0.12 + 1.5))
    fig_width = SINGLE_COL + 0.5  # slightly wider for labels

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    y_positions = np.arange(n_display)
    colors = [CLUSTER_COLORS[i % len(CLUSTER_COLORS)] for i in range(k)]

    lefts = np.zeros(n_display)
    for c in range(k):
        widths = props_sorted[:, c]
        ax.barh(
            y_positions,
            widths,
            left=lefts,
            height=bar_height,
            color=colors[c],
            label=f'Cluster {c + 1}',
            linewidth=0,
        )
        lefts += widths

    # Axes formatting
    ax.set_xlim(0, 1)
    ax.set_xlabel('Proportion of iterations')
    ax.set_ylabel('Sample (sorted by entropy)')
    ax.set_title(title, fontsize=10)

    # Y-axis: show "stable" / "unstable" anchors instead of sample indices
    ax.set_yticks([0, n_display - 1])
    ax.set_yticklabels(['most stable', 'least stable'], fontsize=8)

    # Entropy annotation on the right
    ax_right = ax.twinx()
    ax_right.set_ylim(ax.get_ylim())
    ax_right.set_yticks([0, n_display - 1])
    ax_right.set_yticklabels(
        [f'H={ent_sorted[0]:.2f}', f'H={ent_sorted[-1]:.2f}'],
        fontsize=7,
    )
    ax_right.set_ylabel('Shannon entropy (bits)', fontsize=8)
    ax_right.spines['top'].set_visible(False)

    # Legend outside plot area
    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.18),
        ncol=min(k, 4),
        fontsize=7,
        frameon=False,
    )

    if n_samples > max_samples:
        ax.text(
            0.5, 1.02,
            f'(showing {n_display} of {n_samples} samples)',
            transform=ax.transAxes,
            ha='center',
            fontsize=7,
            color='gray',
        )

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    set_publication_style()

    result_path = os.path.join(RESULTS_DIR, "vdx_3gene.joblib")
    data = joblib.load(result_path)
    er = data["erica_results"]
    clam_matrices = er["clam_matrices"]

    print("Loaded vdx_3gene results.")
    print(f"  clam_matrices keys: {list(clam_matrices.keys())}")

    configs = [
        {
            'key': (3, 'kmeans'),
            'title': 'Stability Strips — VDX 3-gene, K=3, K-Means',
            'fname': 'stability_vdx_3gene_kmeans_k3',
        },
        {
            'key': (3, 'agglomerative_ward'),
            'title': 'Stability Strips — VDX 3-gene, K=3, Agglomerative (Ward)',
            'fname': 'stability_vdx_3gene_agglomerative_ward_k3',
        },
    ]

    for cfg in configs:
        key = cfg['key']
        cm = clam_matrices[key]
        print(f"\nGenerating {cfg['fname']} (CLAM shape {cm.shape}) ...")
        fig = plot_stability_strips(cm, title=cfg['title'])
        save_figure(fig, cfg['fname'])

    print("\nDone.")


if __name__ == "__main__":
    main()
