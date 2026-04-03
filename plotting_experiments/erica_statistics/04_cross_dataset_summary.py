"""Cross-dataset replicability summary heatmap.

Rows: datasets, Columns: ERICA stat / ERICA-ARI / ERICA-AMI at K* for K-Means.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from style import set_publication_style, DOUBLE_COL, SINGLE_COL, METRIC_COLORS

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, '..', 'results')
FIGURES_DIR = os.path.join(SCRIPT_DIR, '..', 'figures', 'erica_statistics')

DATASETS = [
    'vdx_3gene', 'gauss4c_sigma0p1', 'gauss4c_sigma1p0', 'gauss4c_sigma10p0',
    'well_separated', 'high_dim', 'overlapping', 'moons_2d', 'blobs_2d',
]


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


def main():
    set_publication_style()
    os.makedirs(FIGURES_DIR, exist_ok=True)

    method = 'kmeans'
    col_labels = ['ERICA stat', 'ERICA-ARI', 'ERICA-AMI']

    valid_rows = []
    row_labels = []
    skipped = []

    for dataset in DATASETS:
        path = os.path.join(RESULTS_DIR, f'{dataset}.joblib')
        if not os.path.exists(path):
            skipped.append(dataset)
            print(f'  Skipped (missing): {dataset}')
            continue

        r = joblib.load(path)
        er = r['erica_results']
        config = r['config']
        k = choose_k(config, er, method)

        res_entry = er.get('results', {}).get((k, method), {})
        clam = res_entry.get('clam_matrix')
        if clam is None:
            skipped.append(dataset)
            print(f'  Skipped (no CLAM at K={k}): {dataset}')
            continue

        erica_stat_mean = float(np.mean(erica_stat_from_clam(clam)))

        il = res_entry.get('iteration_labels')
        if il is not None:
            ari_vals, ami_vals = compute_per_iteration_ari_ami(il)
            ari_mean = float(np.mean(ari_vals))
            ami_mean = float(np.mean(ami_vals))
        else:
            m_dict = er.get('metrics', {}).get(k, {}).get(method, {})
            ari_mean = float(m_dict.get('ARI_mean', np.nan))
            ami_mean = float(m_dict.get('AMI_mean', np.nan))

        valid_rows.append([erica_stat_mean, ari_mean, ami_mean])
        row_labels.append(dataset.replace('_', ' '))
        print(f'  {dataset}: ERICA={erica_stat_mean:.3f}  ARI={ari_mean:.3f}  AMI={ami_mean:.3f}  (K={k})')

    if not valid_rows:
        print('No valid data found. Exiting.')
        return

    if skipped:
        print(f'\nSkipped datasets: {", ".join(skipped)}')

    data = np.array(valid_rows)
    n_rows, n_cols = data.shape

    fig, ax = plt.subplots(figsize=(SINGLE_COL * 1.8, max(2.5, n_rows * 0.55)))
    im = ax.imshow(data, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')

    for i in range(n_rows):
        for j in range(n_cols):
            val = data[i, j]
            text_color = 'white' if val < 0.4 else 'black'
            if np.isnan(val):
                ax.text(j, i, 'N/A', ha='center', va='center', fontsize=9, color='gray')
            else:
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                        fontsize=9, fontweight='bold', color=text_color)

    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels(col_labels, fontsize=9)
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(row_labels, fontsize=9)
    ax.set_title('Cross-Dataset Replicability Summary (K-Means, at K*)',
                 fontsize=11, fontweight='bold', pad=10)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=8)

    fig.tight_layout()

    for fmt in ('pdf', 'png'):
        out = os.path.join(FIGURES_DIR, f'cross_dataset_summary.{fmt}')
        fig.savefig(out, format=fmt, bbox_inches='tight', dpi=300)
        print(f'  Saved: {out}')
    plt.close(fig)

    print('\nCross-dataset summary complete.')


if __name__ == '__main__':
    main()
