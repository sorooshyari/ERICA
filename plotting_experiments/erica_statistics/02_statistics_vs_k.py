"""
02_statistics_vs_k.py - ERICA Statistics: ERICA stat vs K

Plots the ERICA statistic vs K with +/-1 std shaded band over samples
(from the CLAM matrix). K* is marked with a vertical dashed line.

Outputs: ../figures/erica_statistics/stats_vs_k_{dataset}_{method}.{pdf,png}
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

METHODS = ['kmeans', 'agglomerative_ward']

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


def plot_stats_vs_k(er, config, dataset_name, method):
    metrics = er['metrics']

    k_values = sorted(metrics.keys())
    k_arr = np.array(k_values, dtype=float)
    n_k = len(k_values)

    erica_mean = np.full(n_k, np.nan)
    erica_std = np.full(n_k, np.nan)

    for i, k in enumerate(k_values):
        clam = er.get('clam_matrices', {}).get((k, method))
        if clam is not None:
            vals = erica_stat_from_clam(clam)
            erica_mean[i] = np.mean(vals)
            erica_std[i] = np.std(vals)

    fig, ax = plt.subplots(1, 1, figsize=(SINGLE_COL * 1.4, 3.5))

    valid = ~np.isnan(erica_mean)
    if valid.any():
        kv = k_arr[valid]
        mv = erica_mean[valid]
        sv = erica_std[valid]

        color = STAT_COLORS['ERICA']
        ax.fill_between(kv, mv - sv, mv + sv, color=color, alpha=0.15, linewidth=0)
        ax.plot(kv, mv, color=color, marker='o', markersize=5,
                label='ERICA stat', zorder=3)

    k_star = choose_k(config, er, method)
    if k_star in k_values:
        ax.axvline(k_star, color='gray', linestyle='--', linewidth=1.0, alpha=0.7,
                   label=f'K*={k_star}')

    ax.set_xlabel('K')
    ax.set_ylabel('Statistic value')
    method_label = method.replace('_', ' ').title()
    ax.set_title(f'{dataset_name} - {method_label}')
    ax.set_xticks(k_values)
    ax.legend(frameon=False, fontsize=8)
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

        available_methods = set()
        for k in er['metrics']:
            available_methods.update(er['metrics'][k].keys())

        for method in METHODS:
            if method not in available_methods:
                print(f"  Skipping method '{method}' for {name} (not in results).")
                continue

            print(f"  Generating stats_vs_k_{name}_{method} ...")
            fig = plot_stats_vs_k(er, config, name.replace('_', ' '), method)

            for fmt in ('pdf', 'png'):
                out_path = os.path.join(FIGURES_DIR, f'stats_vs_k_{name}_{method}.{fmt}')
                fig.savefig(out_path, format=fmt)
                print(f"    Saved: {out_path}")
            plt.close(fig)
            generated.append(f'stats_vs_k_{name}_{method}')

    print(f"\nDone. Generated {len(generated)} figure(s), skipped {len(skipped)} dataset(s).")


if __name__ == '__main__':
    main()
