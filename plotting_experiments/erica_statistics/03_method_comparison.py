"""
03_method_comparison.py — ERICA Statistics: Method Comparison Grid

Per-dataset 2x3 grid comparing K-Means and Ward across all three ERICA
statistics (ERICA stat, ERICA-ARI, ERICA-AMI).  Each cell shows the
statistic vs K with ±1 std shaded bands.

  Rows:    K-Means, Agglomerative Ward
  Columns: ERICA stat, ERICA-ARI, ERICA-AMI

Outputs: ../figures/erica_statistics/method_compare_{dataset}.{pdf,png}
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
    'vdx_3gene', 'gauss4c_sigma0p1', 'gauss4c_sigma1p0', 'gauss4c_sigma10p0',
    'well_separated', 'high_dim', 'overlapping', 'moons_2d', 'blobs_2d',
]

METHODS = ['kmeans', 'agglomerative_ward']
METHOD_LABELS = {'kmeans': 'K-Means', 'agglomerative_ward': 'Ward'}

STAT_COLORS = {
    'ERICA': '#009E73',
    'ARI': '#E69F00',
    'AMI': '#CC79A7',
}

STAT_KEYS = ['ERICA', 'ARI', 'AMI']
STAT_LABELS = {'ERICA': 'ERICA stat', 'ARI': 'ERICA-ARI', 'AMI': 'ERICA-AMI'}


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


def collect_stats_vs_k(er, method):
    """Return (k_values, erica_mean, erica_std, ari_mean, ari_std, ami_mean, ami_std)."""
    metrics = er['metrics']
    results = er.get('results', {})

    k_values = sorted(metrics.keys())
    n_k = len(k_values)

    erica_mean = np.full(n_k, np.nan)
    erica_std = np.full(n_k, np.nan)
    ari_mean = np.full(n_k, np.nan)
    ari_std = np.full(n_k, np.nan)
    ami_mean = np.full(n_k, np.nan)
    ami_std = np.full(n_k, np.nan)

    for i, k in enumerate(k_values):
        clam = er.get('clam_matrices', {}).get((k, method))
        if clam is not None:
            vals = erica_stat_from_clam(clam)
            erica_mean[i] = np.mean(vals)
            erica_std[i] = np.std(vals)

        res_entry = results.get((k, method), {})
        il = res_entry.get('iteration_labels')
        if il is not None:
            ari_vals, ami_vals = compute_per_iteration_ari_ami(il)
            ari_mean[i] = np.mean(ari_vals)
            ari_std[i] = np.std(ari_vals)
            ami_mean[i] = np.mean(ami_vals)
            ami_std[i] = np.std(ami_vals)
        else:
            m_dict = metrics[k].get(method, {})
            ari_mean[i] = float(m_dict.get('ARI_mean', np.nan))
            ari_std[i] = float(m_dict.get('ARI_std', np.nan))
            ami_mean[i] = float(m_dict.get('AMI_mean', np.nan))
            ami_std[i] = float(m_dict.get('AMI_std', np.nan))

    stat_data = {
        'ERICA': (erica_mean, erica_std),
        'ARI': (ari_mean, ari_std),
        'AMI': (ami_mean, ami_std),
    }
    return k_values, stat_data


def plot_method_comparison(er, dataset_name):
    available_methods = set()
    for k in er['metrics']:
        available_methods.update(er['metrics'][k].keys())

    methods_present = [m for m in METHODS if m in available_methods]
    if not methods_present:
        return None

    n_rows = len(methods_present)
    n_cols = len(STAT_KEYS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(DOUBLE_COL, 2.8 * n_rows),
                             squeeze=False)

    for row, method in enumerate(methods_present):
        k_values, stat_data = collect_stats_vs_k(er, method)
        k_arr = np.array(k_values, dtype=float)

        for col, stat_key in enumerate(STAT_KEYS):
            ax = axes[row, col]
            mean_arr, std_arr = stat_data[stat_key]
            color = STAT_COLORS[stat_key]

            valid = ~np.isnan(mean_arr)
            if valid.any():
                kv = k_arr[valid]
                mv = mean_arr[valid]
                sv = std_arr[valid]
                ax.fill_between(kv, mv - sv, mv + sv, color=color, alpha=0.15, linewidth=0)
                ax.plot(kv, mv, color=color, marker='o', markersize=4, zorder=3)

            ax.set_xticks(k_values)
            ax.set_xlabel('K')
            if col == 0:
                ax.set_ylabel(METHOD_LABELS[method])
            if row == 0:
                ax.set_title(STAT_LABELS[stat_key], fontsize=10)

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
