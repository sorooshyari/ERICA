"""
08_ica_analysis.py — ICA Dimensionality Reduction + Inter-Cluster Assignment Heatmap

Two visualization types:
  1. PCA vs ICA scatter — side-by-side 2D projections colored by ERICA stat.
     FastICA provides a non-orthogonal alternative to PCA for revealing
     cluster structure.  For natively 2D datasets the raw coordinates are
     used in both panels.
  2. ICAH — Inter-Cluster Assignment Heatmap.  The k x k matrix where
     entry [i, j] = mean normalized assignment of cluster i's samples to
     cluster j.  Diagonal = self-assignment (high is good); off-diagonal =
     leakage between clusters.

Note: joblib is used to load locally-generated ERICA results (not untrusted
external data). These files are produced by the project pipeline scripts.

Outputs: ../figures/erica_statistics/pca_vs_ica_{dataset}.{pdf,png}
         ../figures/erica_statistics/icah_{dataset}_{method}.{pdf,png}
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.decomposition import PCA, FastICA

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from style import set_publication_style, DOUBLE_COL, SINGLE_COL, METRIC_COLORS

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, '..', 'results')
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
FIGURES_DIR = os.path.join(SCRIPT_DIR, '..', 'figures', 'erica_statistics')

DATASETS = [
    'vdx_3gene', 'gauss4c_sigma0p01', 'gauss4c_sigma0p1', 'gauss4c_sigma1p0', 'gauss4c_sigma10p0',
    'well_separated', 'high_dim', 'overlapping', 'moons_2d', 'blobs_2d',
]
DATASETS_2D = {'moons_2d', 'blobs_2d'}


def erica_stat_from_clam(clam):
    row_sums = clam.sum(axis=1)
    row_sums[row_sums == 0] = 1
    return clam.max(axis=1) / row_sums


def choose_k(config, er, method='kmeans'):
    true_k = config.get('true_k')
    if true_k is not None:
        return true_k
    k_star = er.get('k_star', {}).get('TWCRI', {}).get(method)
    if k_star is not None:
        return k_star
    return 3


def project_pca(X, dataset_name):
    if dataset_name in DATASETS_2D or X.shape[1] == 2:
        return X, (1.0, 1.0)
    pca = PCA(n_components=2, random_state=0)
    X_2d = pca.fit_transform(X)
    return X_2d, tuple(pca.explained_variance_ratio_)


def project_ica(X, dataset_name):
    if dataset_name in DATASETS_2D or X.shape[1] == 2:
        return X
    ica = FastICA(n_components=2, random_state=0, max_iter=1000)
    try:
        return ica.fit_transform(X)
    except Exception:
        return None


def _save(fig, stem):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    for fmt in ('pdf', 'png'):
        path = os.path.join(FIGURES_DIR, f'{stem}.{fmt}')
        fig.savefig(path, format=fmt)
        print(f'  Saved: {path}')
    plt.close(fig)


def plot_pca_vs_ica(X_pca, var_ratio, X_ica, erica_stat, dataset_name):
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, SINGLE_COL * 1.2))

    sizes = 10 + 40 * erica_stat

    ax = axes[0]
    sc = ax.scatter(
        X_pca[:, 0], X_pca[:, 1],
        c=erica_stat, cmap='RdYlGn',
        s=sizes, vmin=0, vmax=1,
        alpha=0.75, edgecolors='none',
    )
    r1, r2 = var_ratio
    if r1 == 1.0 and r2 == 1.0:
        ax.set_xlabel('Feature 1', fontsize=9)
        ax.set_ylabel('Feature 2', fontsize=9)
    else:
        ax.set_xlabel(f'PC1 ({r1*100:.1f}% var)', fontsize=9)
        ax.set_ylabel(f'PC2 ({r2*100:.1f}% var)', fontsize=9)
    ax.set_title('PCA', fontsize=10)
    ax.tick_params(labelsize=7)

    ax = axes[1]
    ax.scatter(
        X_ica[:, 0], X_ica[:, 1],
        c=erica_stat, cmap='RdYlGn',
        s=sizes, vmin=0, vmax=1,
        alpha=0.75, edgecolors='none',
    )
    ax.set_xlabel('IC1', fontsize=9)
    ax.set_ylabel('IC2', fontsize=9)
    ax.set_title('ICA', fontsize=10)
    ax.tick_params(labelsize=7)

    for a in axes:
        a.spines['top'].set_visible(False)
        a.spines['right'].set_visible(False)

    fig.suptitle(f'{dataset_name} — PCA vs ICA (colored by ERICA stat)',
                 fontsize=11, y=1.02)

    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(sc, cax=cbar_ax)
    cbar.set_label('ERICA statistic', fontsize=9)
    cbar.ax.tick_params(labelsize=7)

    return fig


def compute_icah(clam, k_val):
    row_sums = clam.sum(axis=1, keepdims=True)
    safe_sums = np.where(row_sums == 0, 1, row_sums)
    clam_norm = clam / safe_sums
    primary = np.argmax(clam, axis=1)

    icah = np.zeros((k_val, k_val))
    for src in range(k_val):
        mask = primary == src
        if mask.any():
            icah[src, :] = clam_norm[mask, :].mean(axis=0)
    return icah


def plot_icah(icah, k_val, dataset_name, method):
    fig, ax = plt.subplots(figsize=(SINGLE_COL * 1.3, SINGLE_COL * 1.2))

    im = ax.imshow(icah, cmap='RdYlGn', vmin=0, vmax=1, aspect='equal')

    for i in range(k_val):
        for j in range(k_val):
            val = icah[i, j]
            text_color = 'white' if val < 0.4 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=9, fontweight='bold', color=text_color)

    labels = [f'C{i+1}' for i in range(k_val)]
    ax.set_xticks(np.arange(k_val))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticks(np.arange(k_val))
    ax.set_yticklabels(labels, fontsize=9)

    ax.set_xlabel('Target cluster', fontsize=9)
    ax.set_ylabel('Source cluster', fontsize=9)

    method_label = 'K-Means' if method == 'kmeans' else method.replace('_', ' ').title()
    ax.set_title(
        f'{dataset_name} — {method_label}, K={k_val}\nInter-Cluster Assignment',
        fontsize=10,
    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Mean assignment', fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    fig.tight_layout()
    return fig


def main():
    set_publication_style()
    os.makedirs(FIGURES_DIR, exist_ok=True)

    method = 'kmeans'

    for dataset_name in DATASETS:
        print(f'\n{dataset_name}:')

        data_path = os.path.join(DATA_DIR, f'{dataset_name}.npz')
        result_path = os.path.join(RESULTS_DIR, f'{dataset_name}.joblib')

        if not os.path.exists(data_path):
            print(f'  [SKIP] No data file: {data_path}')
            continue
        if not os.path.exists(result_path):
            print(f'  [SKIP] No results file: {result_path}')
            continue

        X = np.load(data_path, allow_pickle=True)['X']
        r = joblib.load(result_path)  # locally-generated results
        er = r['erica_results']
        config = r.get('config', {})

        k = choose_k(config, er, method)
        key = (k, method)
        clam = er.get('clam_matrices', {}).get(key)
        if clam is None:
            print(f'  [SKIP] No CLAM for K={k}, {method}')
            continue

        erica_stat = erica_stat_from_clam(clam)

        is_natively_2d = dataset_name in DATASETS_2D or X.shape[1] == 2
        X_pca, var_ratio = project_pca(X, dataset_name)
        X_ica = None if is_natively_2d else project_ica(X, dataset_name)

        if is_natively_2d:
            print(f'  [SKIP] PCA vs ICA for natively 2D dataset {dataset_name}')
        elif X_ica is not None:
            fig = plot_pca_vs_ica(X_pca, var_ratio, X_ica, erica_stat,
                                  dataset_name)
            _save(fig, f'pca_vs_ica_{dataset_name}')
        else:
            print(f'  [WARN] ICA failed for {dataset_name}')

        icah = compute_icah(clam, k)
        fig = plot_icah(icah, k, dataset_name, method)
        _save(fig, f'icah_{dataset_name}_{method}')

    print('\nICA analysis complete.')


if __name__ == '__main__':
    main()
