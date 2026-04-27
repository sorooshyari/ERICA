"""
03_method_comparison.py - ERICA Statistics: Method Comparison Grid

Per-dataset 2x1 grid comparing K-Means and Ward on the ERICA statistic.
Each cell shows the statistic vs K with +/-1 std shaded bands.

  Rows:    K-Means, Agglomerative Ward
  Column:  ERICA stat

Outputs: ../figures/erica_statistics/method_compare_{dataset}.{pdf,png}
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
METHOD_LABELS = {'kmeans': 'K-Means', 'agglomerative_ward': 'Ward'}

STAT_COLORS = {
    'ERICA': '#009E73',
}


def erica_stat_from_clam(clam):
    row_sums = clam.sum(axis=1)
    row_sums[row_sums == 0] = 1
    return clam.max(axis=1) / row_sums


def collect_stats_vs_k(er, method):
    """Return (k_values, erica_mean, erica_std) for the ERICA statistic."""
    metrics = er['metrics']

    k_values = sorted(metrics.keys())
    n_k = len(k_values)

    erica_mean = np.full(n_k, np.nan)
    erica_std = np.full(n_k, np.nan)

    for i, k in enumerate(k_values):
        clam = er.get('clam_matrices', {}).get((k, method))
        if clam is not None:
            vals = erica_stat_from_clam(clam)
            erica_mean[i] = np.mean(vals)
            erica_std[i] = np.std(vals)

    return k_values, erica_mean, erica_std


def plot_method_comparison(er, dataset_name):
    available_methods = set()
    for k in er['metrics']:
        available_methods.update(er['metrics'][k].keys())

    methods_present = [m for m in METHODS if m in available_methods]
    if not methods_present:
        return None

    n_rows = len(methods_present)
    fig, axes = plt.subplots(n_rows, 1, figsize=(SINGLE_COL * 1.4, 2.8 * n_rows),
                             squeeze=False)

    for row, method in enumerate(methods_present):
        k_values, erica_mean, erica_std = collect_stats_vs_k(er, method)
        k_arr = np.array(k_values, dtype=float)

        ax = axes[row, 0]
        color = STAT_COLORS['ERICA']

        valid = ~np.isnan(erica_mean)
        if valid.any():
            kv = k_arr[valid]
            mv = erica_mean[valid]
            sv = erica_std[valid]
            ax.fill_between(kv, mv - sv, mv + sv, color=color, alpha=0.15, linewidth=0)
            ax.plot(kv, mv, color=color, marker='o', markersize=4, zorder=3)

        ax.set_xticks(k_values)
        ax.set_xlabel('K')
        ax.set_ylabel(METHOD_LABELS[method])
        if row == 0:
            ax.set_title('ERICA stat', fontsize=10)

    fig.suptitle(dataset_name, fontsize=11, y=1.01)
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
        print(f"\nLoaded '{name}'  (K values: {sorted(er['metrics'].keys())})")

        print(f"  Generating method_compare_{name} ...")
        fig = plot_method_comparison(er, name.replace('_', ' '))
        if fig is None:
            print(f"  Skipping {name}: no methods available.")
            skipped.append(name)
            continue

        for fmt in ('pdf', 'png'):
            out_path = os.path.join(FIGURES_DIR, f'method_compare_{name}.{fmt}')
            fig.savefig(out_path, format=fmt)
            print(f"    Saved: {out_path}")
        plt.close(fig)
        generated.append(f'method_compare_{name}')

    print(f"\nDone. Generated {len(generated)} figure(s), skipped {len(skipped)} dataset(s).")


if __name__ == '__main__':
    main()
