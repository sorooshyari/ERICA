"""
07_surfaces.py

Three surface types, each with a 2D heatmap and optional 3D plot_surface:

  A) CLAM topography  — 3D surface of sorted CLAM matrix (K=3, kmeans, VDX 3-gene)
  B) Metric landscape — 2D heatmap + 3D surface of TWCRI across K x method
  C) HDBSCAN sensitivity — 2D heatmap + 3D surface of k_agreement_rate across
                           min_cluster_size x min_samples parameter grid
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import set_publication_style, save_figure, DOUBLE_COL, CMAP_SEQ

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3D projection

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")


# ---------------------------------------------------------------------------
# Helper: CLAM sort (same logic as 03_clam_heatmaps.py)
# ---------------------------------------------------------------------------

def sort_clam(clam_matrix):
    """Sort CLAM rows by primary cluster then descending strength."""
    clam = np.array(clam_matrix, dtype=float)
    row_sums = clam.sum(axis=1)
    primary = np.argmax(clam, axis=1)
    strength = np.where(row_sums > 0, clam.max(axis=1) / row_sums, 0.0)
    sort_indices = np.lexsort((-strength, primary))
    return clam[sort_indices], sort_indices


# ===========================================================================
# A) CLAM topography
# ===========================================================================

def plot_clam_surface(clam_matrix):
    """3D surface of a sorted CLAM matrix.

    X = cluster index (columns), Y = sample index (rows), Z = assignment count.

    Returns
    -------
    fig : matplotlib Figure
    """
    clam_sorted, _ = sort_clam(clam_matrix)
    n_samples, k = clam_sorted.shape

    # Build mesh grids
    cluster_idx = np.arange(k)
    sample_idx = np.arange(n_samples)
    X_mesh, Y_mesh = np.meshgrid(cluster_idx, sample_idx)
    Z = clam_sorted  # shape (n_samples, k)

    fig = plt.figure(figsize=(DOUBLE_COL, 5))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(
        X_mesh, Y_mesh, Z,
        cmap=CMAP_SEQ,
        linewidth=0,
        antialiased=False,
        rcount=min(200, n_samples),
        ccount=k,
    )

    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Co-assignment count')

    ax.set_xlabel('Cluster index', labelpad=8)
    ax.set_ylabel('Sample index (sorted)', labelpad=8)
    ax.set_zlabel('Count', labelpad=6)
    ax.set_title('CLAM Topography — VDX 3-gene, K-Means K=3')
    ax.set_xticks(cluster_idx)
    ax.set_xticklabels([f'C{i+1}' for i in cluster_idx])

    fig.tight_layout()
    return fig


# ===========================================================================
# B) Metric landscape (TWCRI across K x method)
# ===========================================================================

METHODS = ['kmeans', 'agglomerative_ward']
METHOD_LABELS = {
    'kmeans': 'K-Means',
    'agglomerative_ward': 'Agglo. Ward',
}


def _build_twcri_matrix(er):
    """Return (k_values list, twcri 2D array shape (n_methods, n_k))."""
    metrics = er['metrics']
    k_values = sorted(metrics.keys())
    n_methods = len(METHODS)
    n_k = len(k_values)
    twcri = np.full((n_methods, n_k), np.nan)

    for ki, k in enumerate(k_values):
        for mi, method in enumerate(METHODS):
            m_dict = metrics[k].get(method, {})
            v = m_dict.get('TWCRI', np.nan)
            try:
                twcri[mi, ki] = float(v)
            except (TypeError, ValueError):
                twcri[mi, ki] = np.nan

    return k_values, twcri


def plot_metric_landscape_2d(er):
    """2D heatmap of TWCRI (methods × K), cells annotated with values.

    Returns
    -------
    fig : matplotlib Figure
    """
    k_values, twcri = _build_twcri_matrix(er)
    n_methods, n_k = twcri.shape

    # Build a masked array so NaN cells are rendered in grey
    twcri_masked = np.ma.masked_invalid(twcri)
    cmap = plt.get_cmap(CMAP_SEQ).copy()
    cmap.set_bad(color='#cccccc')

    fig, ax = plt.subplots(figsize=(DOUBLE_COL * 0.8, 2.5))
    im = ax.imshow(twcri_masked, aspect='auto', cmap=cmap, vmin=0, vmax=1,
                   interpolation='nearest')

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='TWCRI')

    # Annotate cells
    for mi in range(n_methods):
        for ki in range(n_k):
            v = twcri[mi, ki]
            if np.isnan(v):
                label = 'NaN'
                color = '#555555'
                weight = 'normal'
            else:
                label = f'{v:.2f}'
                # choose white/black text for contrast
                color = 'white' if v < 0.55 else 'black'
                weight = 'normal'
            ax.text(ki, mi, label, ha='center', va='center',
                    fontsize=9, color=color, fontweight=weight)

    ax.set_xticks(range(n_k))
    ax.set_xticklabels([str(k) for k in k_values])
    ax.set_yticks(range(n_methods))
    ax.set_yticklabels([METHOD_LABELS[m] for m in METHODS])
    ax.set_xlabel('K')
    ax.set_ylabel('Method')
    ax.set_title('TWCRI Metric Landscape — VDX 3-gene')

    fig.tight_layout()
    return fig


def plot_metric_landscape_3d(er):
    """3D surface of TWCRI (methods × K).

    Returns
    -------
    fig : matplotlib Figure
    """
    k_values, twcri = _build_twcri_matrix(er)
    n_methods, n_k = twcri.shape

    # Replace NaN with 0 for surface rendering
    twcri_plot = np.where(np.isnan(twcri), 0.0, twcri)

    method_idx = np.arange(n_methods)
    k_idx = np.arange(n_k)
    X_mesh, Y_mesh = np.meshgrid(k_idx, method_idx)
    Z = twcri_plot

    fig = plt.figure(figsize=(DOUBLE_COL, 4.5))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(
        X_mesh, Y_mesh, Z,
        cmap=CMAP_SEQ,
        linewidth=0.3,
        edgecolor='grey',
        alpha=0.9,
        vmin=0, vmax=1,
    )

    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='TWCRI')

    ax.set_xlabel('K', labelpad=8)
    ax.set_ylabel('Method', labelpad=8)
    ax.set_zlabel('TWCRI', labelpad=6)
    ax.set_xticks(k_idx)
    ax.set_xticklabels([str(k) for k in k_values], fontsize=8)
    ax.set_yticks(method_idx)
    ax.set_yticklabels([METHOD_LABELS[m] for m in METHODS], fontsize=8)
    ax.set_zlim(0, 1)
    ax.set_title('TWCRI Metric Landscape 3D — VDX 3-gene')

    fig.tight_layout()
    return fig


# ===========================================================================
# C) HDBSCAN parameter sensitivity
# ===========================================================================

def _parse_sweep(sweep):
    """Parse sweep dict into sorted axis values and 2D grid.

    Returns
    -------
    mcs_vals : list of int
        Unique min_cluster_size values, sorted ascending.
    ms_vals : list
        Unique min_samples values: None first, then integers ascending.
    grid : np.ndarray, shape (len(ms_vals), len(mcs_vals))
        k_agreement_rate values (NaN where missing).
    """
    mcs_set = set()
    ms_set = set()
    for (mcs, ms) in sweep:
        mcs_set.add(mcs)
        ms_set.add(ms)

    mcs_vals = sorted(mcs_set)

    # None first, then integers ascending
    none_vals = [v for v in ms_set if v is None]
    int_vals = sorted(v for v in ms_set if v is not None)
    ms_vals = none_vals + int_vals

    n_ms = len(ms_vals)
    n_mcs = len(mcs_vals)

    mcs_to_col = {v: i for i, v in enumerate(mcs_vals)}
    ms_to_row = {v: i for i, v in enumerate(ms_vals)}

    grid = np.full((n_ms, n_mcs), np.nan)
    for (mcs, ms), info in sweep.items():
        r = ms_to_row[ms]
        c = mcs_to_col[mcs]
        rate = info.get('k_agreement_rate', np.nan)
        try:
            grid[r, c] = float(rate)
        except (TypeError, ValueError):
            grid[r, c] = np.nan

    return mcs_vals, ms_vals, grid


def plot_hdbscan_sensitivity_2d(sweep):
    """2D heatmap of k_agreement_rate across (min_cluster_size, min_samples).

    Zero-agreement cells are annotated with a bold 'X'.

    Returns
    -------
    fig : matplotlib Figure
    """
    mcs_vals, ms_vals, grid = _parse_sweep(sweep)
    n_ms, n_mcs = grid.shape

    cmap = plt.get_cmap(CMAP_SEQ).copy()
    cmap.set_bad(color='#cccccc')
    grid_masked = np.ma.masked_invalid(grid)

    fig, ax = plt.subplots(figsize=(DOUBLE_COL * 0.9, 3.2))
    im = ax.imshow(grid_masked, aspect='auto', cmap=cmap, vmin=0, vmax=1,
                   interpolation='nearest')

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='K-agreement rate')

    # Annotate cells
    for ri in range(n_ms):
        for ci in range(n_mcs):
            v = grid[ri, ci]
            if np.isnan(v):
                continue
            if v == 0.0:
                ax.text(ci, ri, 'X', ha='center', va='center',
                        fontsize=10, color='red', fontweight='bold')
            else:
                color = 'white' if v < 0.55 else 'black'
                ax.text(ci, ri, f'{v:.2f}', ha='center', va='center',
                        fontsize=8, color=color)

    ax.set_xticks(range(n_mcs))
    ax.set_xticklabels([str(v) for v in mcs_vals])
    ax.set_yticks(range(n_ms))
    ax.set_yticklabels(['None' if v is None else str(v) for v in ms_vals])
    ax.set_xlabel('min_cluster_size')
    ax.set_ylabel('min_samples')
    ax.set_title('HDBSCAN Parameter Sensitivity — k-agreement rate')

    fig.tight_layout()
    return fig


def plot_hdbscan_sensitivity_3d(sweep):
    """3D surface of k_agreement_rate across HDBSCAN parameter grid.

    Returns
    -------
    fig : matplotlib Figure
    """
    mcs_vals, ms_vals, grid = _parse_sweep(sweep)

    # Replace NaN with 0 for surface rendering
    grid_plot = np.where(np.isnan(grid), 0.0, grid)

    mcs_idx = np.arange(len(mcs_vals))
    ms_idx = np.arange(len(ms_vals))
    X_mesh, Y_mesh = np.meshgrid(mcs_idx, ms_idx)
    Z = grid_plot

    fig = plt.figure(figsize=(DOUBLE_COL, 4.5))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(
        X_mesh, Y_mesh, Z,
        cmap=CMAP_SEQ,
        linewidth=0.3,
        edgecolor='grey',
        alpha=0.9,
        vmin=0, vmax=1,
    )

    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='K-agreement rate')

    ax.set_xlabel('min_cluster_size', labelpad=10)
    ax.set_ylabel('min_samples', labelpad=10)
    ax.set_zlabel('K-agreement rate', labelpad=6)
    ax.set_xticks(mcs_idx)
    ax.set_xticklabels([str(v) for v in mcs_vals], fontsize=7, rotation=15)
    ax.set_yticks(ms_idx)
    ax.set_yticklabels(['None' if v is None else str(v) for v in ms_vals], fontsize=7)
    ax.set_zlim(0, 1)
    ax.set_title('HDBSCAN Parameter Sensitivity 3D')

    fig.tight_layout()
    return fig


# ===========================================================================
# Main
# ===========================================================================

def main():
    set_publication_style()

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    vdx_path = os.path.join(RESULTS_DIR, "vdx_3gene.joblib")
    sweep_path = os.path.join(RESULTS_DIR, "vdx_hdbscan_sweep.joblib")

    print("Loading vdx_3gene.joblib ...")
    data = joblib.load(vdx_path)
    er = data["erica_results"]

    print("Loading vdx_hdbscan_sweep.joblib ...")
    sweep = joblib.load(sweep_path)

    # ------------------------------------------------------------------
    # A) CLAM topography
    # ------------------------------------------------------------------
    print("\n[A] CLAM topography — K=3 kmeans ...")
    cm_k3 = er['clam_matrices'][(3, 'kmeans')]
    fig_a = plot_clam_surface(cm_k3)
    save_figure(fig_a, 'surface_clam_vdx_3gene_kmeans_k3')

    # ------------------------------------------------------------------
    # B) Metric landscape (TWCRI)
    # ------------------------------------------------------------------
    print("\n[B] Metric landscape — TWCRI 2D heatmap ...")
    fig_b2d = plot_metric_landscape_2d(er)
    save_figure(fig_b2d, 'landscape_twcri_vdx_3gene_2d')

    print("[B] Metric landscape — TWCRI 3D surface ...")
    fig_b3d = plot_metric_landscape_3d(er)
    save_figure(fig_b3d, 'landscape_twcri_vdx_3gene_3d')

    # ------------------------------------------------------------------
    # C) HDBSCAN parameter sensitivity
    # ------------------------------------------------------------------
    print("\n[C] HDBSCAN sensitivity — 2D heatmap ...")
    fig_c2d = plot_hdbscan_sensitivity_2d(sweep)
    save_figure(fig_c2d, 'hdbscan_sensitivity_2d')

    print("[C] HDBSCAN sensitivity — 3D surface ...")
    fig_c3d = plot_hdbscan_sensitivity_3d(sweep)
    save_figure(fig_c3d, 'hdbscan_sensitivity_3d')

    print("\nDone.")


if __name__ == "__main__":
    main()
