"""
01_distributions.py - ERICA Statistics: Per-Dataset Distribution Panels

For each dataset (at K*), plots the per-sample ERICA statistic distribution
(max/sum of CLAM row), N = n_samples.

Density histogram with a vertical dashed line at the mean.

Outputs: ../figures/erica_statistics/dist_{dataset}.{pdf,png}
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from style import set_publication_style, save_figure, SINGLE_COL, METRIC_COLORS

import numpy as np
import matplotlib.pyplot as plt
import joblib

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, '..', 'results')
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
FIGURES_DIR = os.path.join(SCRIPT_DIR, '..', 'figures', 'erica_statistics')

DATASETS = [
    'vdx_3gene', 'gauss4c_sigma0p01', 'gauss4c_sigma0p1', 'gauss4c_sigma1p0', 'gauss4c_sigma10p0',
    'well_separated', 'high_dim', 'overlapping', 'moons_2d', 'blobs_2d',
]

STAT_COLORS = {
    'ERICA': '#009E73',
}


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


def plot_distributions(er, config, dataset_name, method='kmeans'):
    k = choose_k(config, er, method)
    clam = er.get('clam_matrices', {}).get((k, method))

    if clam is None:
        print(f"    No CLAM matrix for K={k}, method={method}")
        return None

    erica_vals = erica_stat_from_clam(clam)

    fig, ax = plt.subplots(1, 1, figsize=(SINGLE_COL * 1.2, 3.0))

    ax.hist(erica_vals, bins=30, density=True, alpha=0.7,
            color=STAT_COLORS['ERICA'], edgecolor='white', linewidth=0.5)
    ax.axvline(np.mean(erica_vals), color=STAT_COLORS['ERICA'],
               linestyle='--', linewidth=1.5)
    ax.set_xlabel('ERICA statistic')
    ax.set_ylabel('Density')
    ax.set_title(f'ERICA stat (N={len(erica_vals)} samples)', fontsize=9)

    fig.suptitle(f'{dataset_name}  (K={k}, {method})', fontsize=11, y=1.02)
    fig.tight_layout()
    return fig


def main():
    set_publication_style()
    os.makedirs(FIGURES_DIR, exist_ok=True)

    generated = []
    skipped = []

    for name in DATASETS:
        result_path = os.path.join(RESULTS_DIR, f'{name}.joblib')
        if not os.path.exists(result_path):
            print(f"  Skipping {name}: results file not found.")
            skipped.append(name)
            continue

        data = joblib.load(result_path)
        er = data['erica_results']
        config = data.get('config', {})
        print(f"\nLoaded '{name}'  (K values: {sorted(er['metrics'].keys())})")

        fig = plot_distributions(er, config, name.replace('_', ' '))
        if fig is None:
            skipped.append(name)
            continue

        for fmt in ('pdf', 'png'):
            out_path = os.path.join(FIGURES_DIR, f'dist_{name}.{fmt}')
            fig.savefig(out_path, format=fmt)
            print(f"    Saved: {out_path}")
        plt.close(fig)
        generated.append(f'dist_{name}')

    print(f"\nDone. Generated {len(generated)} figure(s), skipped {len(skipped)} dataset(s).")


if __name__ == '__main__':
    main()
