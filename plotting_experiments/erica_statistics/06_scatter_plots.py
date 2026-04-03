"""
06_scatter_plots.py — Spatial Scatter Plots of ERICA Statistics

Three visualization types per dataset:
  1. ERICA stat confidence map — 2D scatter colored by ERICA stat (RdYlGn),
     point size scaled by stat value.
  2. Triptych (1x3) — cluster assignment | ERICA stat | normalized entropy.
  3. Method comparison — side-by-side panels (K-Means | Ward), shared colorbar.

High-dimensional data is projected to 2D via PCA; natively 2D datasets
(moons_2d, blobs_2d) use raw coordinates.

Note: joblib is used to load locally-generated ERICA results (not untrusted
external data). These files are produced by ../02_run_erica_pipeline.py.

Outputs: ../figures/erica_statistics/scatter_erica_{dataset}_{method}.{pdf,png}
         ../figures/erica_statistics/triptych_{dataset}_{method}.{pdf,png}
         ../figures/erica_statistics/scatter_method_compare_{dataset}.{pdf,png}
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import joblib
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from style import set_publication_style, DOUBLE_COL, SINGLE_COL, METRIC_COLORS

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, '..', 'results')
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
FIGURES_DIR = os.path.join(SCRIPT_DIR, '..', 'figures', 'erica_statistics')

DATASETS = [
    'vdx_3gene', 'gauss4c_sigma0p1', 'gauss4c_sigma1p0', 'gauss4c_sigma10p0',
    'well_separated', 'high_dim', 'overlapping', 'moons_2d', 'blobs_2d',
]
DATASETS_2D = {'moons_2d', 'blobs_2d'}
CLUSTER_COLORS = [
    '#4477AA', '#EE6677', '#228833', '#CCBB44',
    '#66CCEE', '#AA3377', '#BBBBBB', '#332288',
]
METHODS = ['kmeans', 'agglomerative_ward']
METHOD_LABELS = {'kmeans': 'K-Means', 'agglomerative_ward': 'Ward'}


def erica_stat_from_clam(clam):
    row_sums = clam.sum(axis=1)
    row_sums[row_sums == 0] = 1
    return clam.max(axis=1) / row_sums


def entropy_from_clam(clam):
    sums = clam.sum(axis=1, keepdims=True)
    sums[sums == 0] = 1
    p = clam / sums
    with np.errstate(divide='ignore', invalid='ignore'):
        log_p = np.where(p > 0, np.log(p), 0.0)
    raw = -(p * log_p).sum(axis=1)
    k = clam.shape[1]
    max_ent = np.log(k) if k > 1 else 1.0
    return raw / max_ent if max_ent > 0 else raw


def project_2d(X, dataset_name):
    if dataset_name in DATASETS_2D or X.shape[1] == 2:
        return X, (1.0, 1.0)
    pca = PCA(n_components=2, random_state=0)
    X_2d = pca.fit_transform(X)
    return X_2d, tuple(pca.explained_variance_ratio_)


def choose_k(config, er, method='kmeans'):
    true_k = config.get('true_k')
    if true_k is not None:
        return true_k
    k_star = er.get('k_star', {}).get('TWCRI', {}).get(method)
    if k_star is not None:
        return k_star
    return 3


def _variance_label(var_ratio):
    r1, r2 = var_ratio
    if r1 == 1.0 and r2 == 1.0:
        return 'Feature 1', 'Feature 2'
    return f'PC1 ({r1*100:.1f}% var)', f'PC2 ({r2*100:.1f}% var)'


def _save(fig, stem):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    for fmt in ('pdf', 'png'):
        path = os.path.join(FIGURES_DIR, f'{stem}.{fmt}')
        fig.savefig(path, format=fmt)
        print(f'  Saved: {path}')
    plt.close(fig)


def plot_erica_confidence(X_2d, var_ratio, erica_stat, k, method, dataset_name):
    xlabel, ylabel = _variance_label(var_ratio)

    fig, ax = plt.subplots(figsize=(SINGLE_COL * 1.4, SINGLE_COL * 1.2))

    sizes = 10 + 40 * erica_stat
    sc = ax.scatter(
        X_2d[:, 0], X_2d[:, 1],
        c=erica_stat, cmap='RdYlGn',
        s=sizes, vmin=0, vmax=1,
        alpha=0.75, edgecolors='none',
    )

    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('ERICA statistic', fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(f'{dataset_name}  ({METHOD_LABELS.get(method, method)}, K={k})',
                 fontsize=11)
    ax.tick_params(labelsize=8)

    fig.tight_layout()
    return fig


def plot_triptych(X_2d, var_ratio, clam, k, method, dataset_name):
    primary = np.argmax(clam, axis=1)
    erica_stat = erica_stat_from_clam(clam)
    ent = entropy_from_clam(clam)
    xlabel, ylabel = _variance_label(var_ratio)

    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL * 1.3, SINGLE_COL * 1.2))

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

    ax = axes[1]
    sizes = 10 + 40 * erica_stat
    sc1 = ax.scatter(
        X_2d[:, 0], X_2d[:, 1],
        c=erica_stat, cmap='RdYlGn',
        s=sizes, vmin=0.0, vmax=1.0,
        alpha=0.75, edgecolors='none',
    )
    cb1 = fig.colorbar(sc1, ax=ax, fraction=0.046, pad=0.04)
    cb1.set_label('ERICA stat', fontsize=8)
    cb1.ax.tick_params(labelsize=7)
    ax.set_title('ERICA statistic', fontsize=10)

    ax = axes[2]
    sc2 = ax.scatter(
        X_2d[:, 0], X_2d[:, 1],
        c=ent, cmap='RdYlGn_r',
        s=12, vmin=0.0, vmax=1.0,
        alpha=0.75, edgecolors='none',
    )
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

    fig.suptitle(
        f'{dataset_name}  --  {METHOD_LABELS.get(method, method)}, K={k}',
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    return fig


def plot_method_compare(X_2d, var_ratio, clam_by_method, k, dataset_name):
    methods_with_data = [m for m in METHODS if m in clam_by_method]
    if not methods_with_data:
        return None

    n = len(methods_with_data)
    fig, axes = plt.subplots(
        1, n,
        figsize=(DOUBLE_COL * (n / 3.0) * 1.3, SINGLE_COL * 1.2),
        sharey=True,
    )
    if n == 1:
        axes = [axes]

    xlabel, ylabel = _variance_label(var_ratio)
    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
    cmap = plt.get_cmap('RdYlGn')

    for ax, method in zip(axes, methods_with_data):
        clam = clam_by_method[method]
        estat = erica_stat_from_clam(clam)
        sizes = 10 + 40 * estat
        ax.scatter(
            X_2d[:, 0], X_2d[:, 1],
            c=estat, cmap=cmap, norm=norm,
            s=sizes, alpha=0.75, edgecolors='none',
        )
        ax.set_title(METHOD_LABELS.get(method, method), fontsize=10)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    axes[0].set_ylabel(ylabel, fontsize=8)

    fig.suptitle(
        f'{dataset_name}  --  Method comparison (K={k})',
        fontsize=12, y=1.0,
    )
    fig.tight_layout(rect=[0, 0.08, 1, 0.95])

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.2, 0.01, 0.6, 0.025])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('ERICA statistic', fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    return fig


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
    r = joblib.load(result_path)  # locally-generated ERICA results
    er = r['erica_results']
    config = r.get('config', {})

    X_2d, var_ratio = project_2d(X, dataset_name)

    for method in METHODS:
        k = choose_k(config, er, method)

        key = (k, method)
        clam = er.get('clam_matrices', {}).get(key)
        if clam is None:
            print(f'  [WARN] No CLAM for K={k}, {method} in {dataset_name}')
            continue

        erica_stat = erica_stat_from_clam(clam)
        print(f'  {dataset_name}/{method}: n={X.shape[0]}, K={k}, '
              f'PC1={var_ratio[0]*100:.1f}%, PC2={var_ratio[1]*100:.1f}%')

        fig = plot_erica_confidence(X_2d, var_ratio, erica_stat, k,
                                    method, dataset_name)
        _save(fig, f'scatter_erica_{dataset_name}_{method}')

        fig = plot_triptych(X_2d, var_ratio, clam, k, method, dataset_name)
        _save(fig, f'triptych_{dataset_name}_{method}')

    k_ref = choose_k(config, er, 'kmeans')
    clam_by_method = {}
    for method in METHODS:
        mkey = (k_ref, method)
        if mkey in er.get('clam_matrices', {}):
            clam_by_method[method] = er['clam_matrices'][mkey]

    if clam_by_method:
        fig = plot_method_compare(X_2d, var_ratio, clam_by_method,
                                  k_ref, dataset_name)
        if fig is not None:
            _save(fig, f'scatter_method_compare_{dataset_name}')


def main():
    set_publication_style()
    os.makedirs(FIGURES_DIR, exist_ok=True)

    for dataset_name in DATASETS:
        print(f'\n{dataset_name}:')
        process_dataset(dataset_name)

    print('\nScatter plots complete.')


if __name__ == '__main__':
    main()
