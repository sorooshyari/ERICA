"""
01_distributions.py — ERICA Statistics: Per-Dataset Distribution Panels

Exploratory analysis extending Parmigiani et al.'s use of ARI/AMI as
partition-comparison measures to the Monte Carlo subsampling framework of
ERICA.  For each dataset (at K*), plots three side-by-side distributions:

  Left   — ERICA statistic (per-sample max/sum of CLAM row), N = n_samples.
  Center — ERICA-ARI (per-iteration Adjusted Rand Index), N = n_iterations.
  Right  — ERICA-AMI (per-iteration Adjusted Mutual Information), N = n_iterations.

Each panel shows a density histogram with a vertical dashed line at the mean.
The difference in N (samples vs. iterations) is noted in the subplot subtitles.

Outputs: ../figures/erica_statistics/dist_{dataset}.{pdf,png}
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from style import set_publication_style, save_figure, DOUBLE_COL, SINGLE_COL, METRIC_COLORS

import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score

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
    'ARI': '#E69F00',
    'AMI': '#CC79A7',
}


def erica_stat_from_clam(clam):
    row_sums = clam.sum(axis=1)
    row_sums[row_sums == 0] = 1
    return clam.max(axis=1) / row_sums


def compute_per_iteration_ari_ami(iteration_labels):
    predicted = iteration_labels["predicted"]
    true_labels = iteration_labels["true"]
    n = len(predicted)
    ari_vals = np.empty(n)
    ami_vals = np.empty(n)
    for i in range(n):
        ari_vals[i] = adjusted_rand_score(true_labels[i], predicted[i])
        ami_vals[i] = adjusted_mutual_info_score(true_labels[i], predicted[i])
    return ari_vals, ami_vals


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
    res_entry = er.get('results', {}).get((k, method), {})
    il = res_entry.get('iteration_labels')

    if clam is None:
        print(f"    No CLAM matrix for K={k}, method={method}")
        return None

    erica_vals = erica_stat_from_clam(clam)

    ari_vals = ami_vals = None
    if il is not None:
        ari_vals, ami_vals = compute_per_iteration_ari_ami(il)

    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL, 3.0))

    axes[0].hist(erica_vals, bins=30, density=True, alpha=0.7,
                 color=STAT_COLORS['ERICA'], edgecolor='white', linewidth=0.5)
    axes[0].axvline(np.mean(erica_vals), color=STAT_COLORS['ERICA'],
                    linestyle='--', linewidth=1.5)
    axes[0].set_xlabel('ERICA statistic')
    axes[0].set_ylabel('Density')
    axes[0].set_title(f'ERICA stat (N={len(erica_vals)} samples)', fontsize=9)

    if ari_vals is not None:
        axes[1].hist(ari_vals, bins=30, density=True, alpha=0.7,
                     color=STAT_COLORS['ARI'], edgecolor='white', linewidth=0.5)
        axes[1].axvline(np.mean(ari_vals), color=STAT_COLORS['ARI'],
                        linestyle='--', linewidth=1.5)
        axes[1].set_title(f'ERICA-ARI (N={len(ari_vals)} iters)', fontsize=9)
    else:
        axes[1].text(0.5, 0.5, 'No iteration labels', ha='center', va='center',
                     transform=axes[1].transAxes, fontsize=9, color='gray')
        axes[1].set_title('ERICA-ARI', fontsize=9)
    axes[1].set_xlabel('ARI')
    axes[1].set_ylabel('Density')

    if ami_vals is not None:
        axes[2].hist(ami_vals, bins=30, density=True, alpha=0.7,
                     color=STAT_COLORS['AMI'], edgecolor='white', linewidth=0.5)
        axes[2].axvline(np.mean(ami_vals), color=STAT_COLORS['AMI'],
                        linestyle='--', linewidth=1.5)
        axes[2].set_title(f'ERICA-AMI (N={len(ami_vals)} iters)', fontsize=9)
    else:
        axes[2].text(0.5, 0.5, 'No iteration labels', ha='center', va='center',
                     transform=axes[2].transAxes, fontsize=9, color='gray')
        axes[2].set_title('ERICA-AMI', fontsize=9)
    axes[2].set_xlabel('AMI')
    axes[2].set_ylabel('Density')

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
