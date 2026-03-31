"""PCA scatter plots colored by per-sample ERICA statistic.

Projects high-dimensional datasets to 2D via PCA and colors each point by its
ERICA statistic (per-sample replicability). For 2D datasets (moons_2d,
blobs_2d) the raw coordinates are used directly — PCA would be a no-op.

Based on Parmigiani et al. (2023) Figures 4 and 6 (which use tSNE). We use
PCA because it is deterministic and requires no perplexity tuning.

Three figures per dataset:
  1. pca_erica_{dataset}         — single panel: PCA colored by ERICA stat
  2. pca_triptych_{dataset}      — 1x3: cluster assignment | ERICA stat | entropy
  3. pca_method_compare_{dataset}— one panel per method, colored by ERICA stat

Note: joblib is used to load locally-generated ERICA results (not untrusted
external data). These files are produced by ../02_run_erica_pipeline.py.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import joblib
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from style import set_publication_style, save_figure, SINGLE_COL, DOUBLE_COL

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, '..', 'results')
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
FIGURES_DIR = os.path.join(SCRIPT_DIR, '..', 'figures', 'literature_comparison')

# Datasets to process (ordered roughly by complexity)
DATASETS = [
    'vdx_3gene',
    'gauss4c_sigma0p1',
    'gauss4c_sigma1p0',
    'gauss4c_sigma10p0',
    'well_separated',
    'high_dim',
    'overlapping',
    'moons_2d',
    'blobs_2d',
]

# 2D datasets: skip PCA, use raw coordinates
DATASETS_2D = {'moons_2d', 'blobs_2d'}

# VDX has no true_k; fall back to K=3
VDX_FALLBACK_K = 3

# Methods for method-comparison figure
METHODS_COMPARE = ['kmeans', 'agglomerative_ward', 'agglomerative_single']
METHOD_LABELS = {
    'kmeans': 'K-Means',
    'agglomerative_ward': 'Ward',
    'agglomerative_single': 'Single',
}

# Colorblind-safe palette for cluster assignment panel
CLUSTER_COLORS = ['#4477AA', '#EE6677', '#228833', '#CCBB44',
                  '#66CCEE', '#AA3377', '#BBBBBB', '#332288']


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save(fig, stem):
    """Save figure to literature_comparison figures directory."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    for fmt in ('pdf', 'png'):
        path = os.path.join(FIGURES_DIR, f'{stem}.{fmt}')
        fig.savefig(path, format=fmt)
        print(f'  Saved: {path}')
    plt.close(fig)


def _project(X, dataset_name):
    """Return 2D projection and explained variance ratios.

    For 2D datasets returns raw X and (1.0, 1.0) as dummy variance ratios.
    """
    if dataset_name in DATASETS_2D or X.shape[1] == 2:
        return X, (1.0, 1.0)
    pca = PCA(n_components=2, random_state=0)
    X_2d = pca.fit_transform(X)
    return X_2d, tuple(pca.explained_variance_ratio_)


def _erica_stat(clam):
    """Per-sample ERICA statistic: proportion assigned to primary cluster.

    Returns (primary_cluster_indices, erica_stat_array).
    """
    primary = np.argmax(clam, axis=1)
    sums = clam.sum(axis=1)
    sums[sums == 0] = 1
    stat = clam[np.arange(len(clam)), primary] / sums
    return primary, stat


def _entropy(clam):
    """Per-sample normalised entropy of cluster assignment distribution."""
    sums = clam.sum(axis=1, keepdims=True)
    sums[sums == 0] = 1
    p = clam / sums
    with np.errstate(divide='ignore', invalid='ignore'):
        log_p = np.where(p > 0, np.log(p), 0.0)
    raw_entropy = -(p * log_p).sum(axis=1)
    k = clam.shape[1]
    max_entropy = np.log(k) if k > 1 else 1.0
    return raw_entropy / max_entropy if max_entropy > 0 else raw_entropy


def _variance_label(var_ratio):
    """Axis labels with explained variance, or plain labels for 2D data."""
    r1, r2 = var_ratio
    if r1 == 1.0 and r2 == 1.0:
        return 'Feature 1', 'Feature 2'
    return (f'PC1 ({r1*100:.1f}% var)', f'PC2 ({r2*100:.1f}% var)')


def _choose_k(config, er):
    """Pick K to display: true_k if available, else VDX_FALLBACK_K."""
    true_k = config.get('true_k')
    if true_k is not None:
        return true_k
    fallback = VDX_FALLBACK_K
    if (fallback, 'kmeans') in er['clam_matrices']:
        return fallback
    keys = sorted(er['clam_matrices'].keys())
    return keys[0][0] if keys else None


# ---------------------------------------------------------------------------
# Figure 1: Single panel — PCA colored by ERICA statistic
# ---------------------------------------------------------------------------

def plot_single(X_2d, var_ratio, erica_stat, k, method, dataset_name):
    """Single scatter: points colored by ERICA stat (RdYlGn colormap).

    Point size scales with ERICA stat (larger = more stable).
    """
    xlabel, ylabel = _variance_label(var_ratio)

    fig, ax = plt.subplots(figsize=(SINGLE_COL * 1.4, SINGLE_COL * 1.2))

    sizes = 10 + 40 * erica_stat
    sc = ax.scatter(X_2d[:, 0], X_2d[:, 1],
                    c=erica_stat, cmap='RdYlGn',
                    s=sizes, vmin=0.0, vmax=1.0,
                    alpha=0.75, edgecolors='none', linewidths=0)

    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('ERICA statistic', fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(f'{dataset_name}  ({method}, K={k})', fontsize=11)
    ax.tick_params(labelsize=8)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 2: Triptych — assignment | ERICA stat | entropy
# ---------------------------------------------------------------------------

def plot_triptych(X_2d, var_ratio, clam, k, method, dataset_name):
    """1x3 figure: cluster labels, ERICA stat, and normalised entropy."""
    primary, erica_stat = _erica_stat(clam)
    ent = _entropy(clam)

    xlabel, ylabel = _variance_label(var_ratio)

    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL * 1.3, SINGLE_COL * 1.2))

    # Panel 0: cluster assignment
    ax = axes[0]
    for c_idx in range(k):
        mask = primary == c_idx
        if not mask.any():
            continue
        color = CLUSTER_COLORS[c_idx % len(CLUSTER_COLORS)]
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   s=12, c=color, alpha=0.75,
                   edgecolors='none', label=f'C{c_idx + 1}')
    ax.set_title('Cluster assignment', fontsize=10)
    ax.legend(fontsize=7, loc='best', markerscale=1.3,
              framealpha=0.7, handletextpad=0.3)

    # Panel 1: ERICA stat
    ax = axes[1]
    sizes = 10 + 40 * erica_stat
    sc1 = ax.scatter(X_2d[:, 0], X_2d[:, 1],
                     c=erica_stat, cmap='RdYlGn',
                     s=sizes, vmin=0.0, vmax=1.0,
                     alpha=0.75, edgecolors='none')
    cb1 = fig.colorbar(sc1, ax=ax, fraction=0.046, pad=0.04)
    cb1.set_label('ERICA stat', fontsize=8)
    cb1.ax.tick_params(labelsize=7)
    ax.set_title('ERICA statistic', fontsize=10)

    # Panel 2: entropy
    ax = axes[2]
    sc2 = ax.scatter(X_2d[:, 0], X_2d[:, 1],
                     c=ent, cmap='RdYlGn_r',
                     s=12, vmin=0.0, vmax=1.0,
                     alpha=0.75, edgecolors='none')
    cb2 = fig.colorbar(sc2, ax=ax, fraction=0.046, pad=0.04)
    cb2.set_label('Norm. entropy', fontsize=8)
    cb2.ax.tick_params(labelsize=7)
    ax.set_title('Entropy', fontsize=10)

    for i, ax in enumerate(axes):
        ax.set_xlabel(xlabel, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    axes[0].set_ylabel(ylabel, fontsize=8)

    fig.suptitle(f'{dataset_name}  --  {method}, K={k}', fontsize=12, y=1.02)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 3: Method comparison — one panel per method
# ---------------------------------------------------------------------------

def plot_method_compare(X_2d, var_ratio, clam_by_method, k, dataset_name):
    """One subplot per method, all colored by ERICA stat on the same PCA axes.

    clam_by_method : dict  {method_name: clam_matrix}
    """
    methods = [m for m in METHODS_COMPARE if m in clam_by_method]
    if not methods:
        return None

    n = len(methods)
    fig, axes = plt.subplots(1, n,
                             figsize=(DOUBLE_COL * (n / 3.0) * 1.3,
                                      SINGLE_COL * 1.2),
                             sharey=True)
    if n == 1:
        axes = [axes]

    xlabel, ylabel = _variance_label(var_ratio)

    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
    cmap = plt.get_cmap('RdYlGn')

    for ax, method in zip(axes, methods):
        clam = clam_by_method[method]
        _, erica_stat = _erica_stat(clam)
        sizes = 10 + 40 * erica_stat
        ax.scatter(X_2d[:, 0], X_2d[:, 1],
                   c=erica_stat, cmap=cmap, norm=norm,
                   s=sizes, alpha=0.75, edgecolors='none')
        ax.set_title(METHOD_LABELS.get(method, method), fontsize=10)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    axes[0].set_ylabel(ylabel, fontsize=8)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, fraction=0.025, pad=0.04)
    cbar.set_label('ERICA statistic', fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    fig.suptitle(f'{dataset_name}  --  Method comparison (K={k})',
                 fontsize=12, y=1.02)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def process_dataset(dataset_name):
    data_path = os.path.join(DATA_DIR, f'{dataset_name}.npz')
    result_path = os.path.join(RESULTS_DIR, f'{dataset_name}.joblib')

    if not os.path.exists(data_path):
        print(f'  [SKIP] No data file: {data_path}')
        return
    if not os.path.exists(result_path):
        print(f'  [SKIP] No results file: {result_path}')
        return

    X = np.load(data_path, allow_pickle=True)['X']
    r = joblib.load(result_path)
    er = r['erica_results']
    config = r['config']

    k = _choose_k(config, er)
    if k is None:
        print(f'  [SKIP] No usable K found for {dataset_name}')
        return

    X_2d, var_ratio = _project(X, dataset_name)
    print(f'  {dataset_name}: n={X.shape[0]}, K={k}, '
          f'PC1={var_ratio[0]*100:.1f}%, PC2={var_ratio[1]*100:.1f}%')

    # Figure 1: single panel (kmeans)
    key = (k, 'kmeans')
    if key in er['clam_matrices']:
        clam = er['clam_matrices'][key]
        _, erica_stat = _erica_stat(clam)
        fig = plot_single(X_2d, var_ratio, erica_stat, k, 'kmeans', dataset_name)
        _save(fig, f'pca_erica_{dataset_name}')
    else:
        print(f'  [WARN] No kmeans K={k} CLAM for {dataset_name}')

    # Figure 2: triptych (kmeans)
    if key in er['clam_matrices']:
        clam = er['clam_matrices'][key]
        fig = plot_triptych(X_2d, var_ratio, clam, k, 'kmeans', dataset_name)
        _save(fig, f'pca_triptych_{dataset_name}')

    # Figure 3: method comparison
    clam_by_method = {}
    for method in METHODS_COMPARE:
        mkey = (k, method)
        if mkey in er['clam_matrices']:
            clam_by_method[method] = er['clam_matrices'][mkey]

    if clam_by_method:
        fig = plot_method_compare(X_2d, var_ratio, clam_by_method, k, dataset_name)
        if fig is not None:
            _save(fig, f'pca_method_compare_{dataset_name}')
    else:
        print(f'  [WARN] No CLAM matrices for method comparison on {dataset_name}')


def main():
    set_publication_style()
    os.makedirs(FIGURES_DIR, exist_ok=True)

    for dataset_name in DATASETS:
        print(f'\n{dataset_name}:')
        process_dataset(dataset_name)

    print('\nPCA ERICA scatter complete.')


if __name__ == '__main__':
    main()
