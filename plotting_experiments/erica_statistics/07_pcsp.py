"""
07_pcsp.py — Per-Cluster Scatter Plots (PCSP)

For each cluster k at K*, scatter the normalized assignment values (V-set)
against an index showing how samples assigned to cluster k distribute their
assignment mass across other clusters.

The PCSP computation:
  1. Normalize CLAM rows: clam_norm = clam / row_sum.
  2. Find primary cluster per sample: primary = argmax(clam, axis=1).
  3. For cluster k, gather samples where primary==k.
  4. For each target cluster j != k, collect those samples' normalized
     assignment value in column j.
  5. Concatenate these V-sets and scatter x=index, y=value.

Note: joblib is used to load locally-generated ERICA results (not untrusted
external data). These files are produced by ../02_run_erica_pipeline.py.

Outputs: ../figures/erica_statistics/pcsp_{dataset}_{method}.{pdf,png}
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import joblib

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


def choose_k(config, er, method='kmeans'):
    true_k = config.get('true_k')
    if true_k is not None:
        return true_k
    k_star = er.get('k_star', {}).get('TWCRI', {}).get(method)
    if k_star is not None:
        return k_star
    return 3


def _save(fig, stem):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    for fmt in ('pdf', 'png'):
        path = os.path.join(FIGURES_DIR, f'{stem}.{fmt}')
        fig.savefig(path, format=fmt)
        print(f'  Saved: {path}')
    plt.close(fig)


def plot_pcsp(clam, k_val, dataset_name, method):
    row_sums = clam.sum(axis=1, keepdims=True)
    safe_sums = np.where(row_sums == 0, 1, row_sums)
    clam_norm = clam / safe_sums
    primary = np.argmax(clam, axis=1)

    ncols = min(k_val, 4)
    nrows = int(np.ceil(k_val / ncols))
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(4 * ncols, 3.5 * nrows),
        squeeze=False,
    )
    axes_flat = axes.flatten()

    for idx in range(k_val, nrows * ncols):
        axes_flat[idx].set_visible(False)

    for src_k in range(k_val):
        ax = axes_flat[src_k]
        mask = primary == src_k
        if not mask.any():
            ax.set_title(f'C{src_k+1} (empty)', fontsize=10)
            ax.set_ylim(-0.05, 1.05)
            continue

        segments = []
        boundaries = [0]
        target_labels = []
        for tgt_k in range(k_val):
            if tgt_k == src_k:
                continue
            vals = clam_norm[mask, tgt_k]
            segments.append(vals)
            boundaries.append(boundaries[-1] + len(vals))
            target_labels.append(f'C{tgt_k+1}')

        if not segments:
            ax.set_title(f'C{src_k+1} (n={mask.sum()})', fontsize=10)
            ax.set_ylim(-0.05, 1.05)
            continue

        all_vals = np.concatenate(segments)
        x = np.arange(len(all_vals))
        ax.scatter(x, all_vals, s=8, alpha=0.6, c='#4477AA')

        for b in boundaries[1:-1]:
            ax.axvline(b, linestyle=':', color='#888', linewidth=0.8)

        mid_points = [
            (boundaries[i] + boundaries[i + 1]) / 2
            for i in range(len(target_labels))
        ]
        for mid, label in zip(mid_points, target_labels):
            ax.text(mid, -0.12, label, ha='center', va='top', fontsize=7,
                    color='#555')

        ax.set_title(f'C{src_k+1} (n={mask.sum()})', fontsize=10)
        ax.set_xlabel('Index', fontsize=8)
        if src_k % ncols == 0:
            ax.set_ylabel('Assignment to other clusters', fontsize=8)
        ax.set_ylim(-0.05, 1.05)
        ax.tick_params(labelsize=7)

    method_label = 'K-Means' if method == 'kmeans' else method.replace('_', ' ').title()
    fig.suptitle(f'{dataset_name} — {method_label}, K={k_val} — PCSP',
                 fontsize=12, y=1.02)
    fig.tight_layout()
    return fig


def main():
    set_publication_style()
    os.makedirs(FIGURES_DIR, exist_ok=True)

    method = 'kmeans'
    generated = 0

    for dataset_name in DATASETS:
        result_path = os.path.join(RESULTS_DIR, f'{dataset_name}.joblib')
        if not os.path.exists(result_path):
            print(f'  [SKIP] No results: {dataset_name}')
            continue

        r = joblib.load(result_path)  # locally-generated ERICA results
        er = r['erica_results']
        config = r.get('config', {})

        k = choose_k(config, er, method)
        key = (k, method)
        clam = er.get('clam_matrices', {}).get(key)
        if clam is None:
            print(f'  [SKIP] No CLAM for K={k}, {method}: {dataset_name}')
            continue

        print(f'\n{dataset_name}: K={k}, method={method}')
        fig = plot_pcsp(clam, k, dataset_name, method)
        _save(fig, f'pcsp_{dataset_name}_{method}')
        generated += 1

    print(f'\nPCSP complete. Generated {generated} figure(s).')


if __name__ == '__main__':
    main()
