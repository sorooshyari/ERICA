"""Sigma degradation study for Gaussian 4-center datasets.

Three-line plot showing ERICA stat, ERICA-ARI, and ERICA-AMI vs sigma,
side-by-side for K-Means and Ward.
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

SIGMA_DATASETS = [
    ('gauss4c_sigma0p01', 0.01),
    ('gauss4c_sigma0p1', 0.1),
    ('gauss4c_sigma1p0', 1.0),
    ('gauss4c_sigma10p0', 10.0),
]

K_EVAL = 4

LINE_STYLES = {
    'ERICA stat': {'color': '#009E73', 'label': 'ERICA stat'},
    'ERICA-ARI':  {'color': '#E69F00', 'label': 'ERICA-ARI'},
    'ERICA-AMI':  {'color': '#CC79A7', 'label': 'ERICA-AMI'},
}


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


def collect_metrics(method):
    """Return (sigmas, erica_stat_means, ari_means, ami_means) for the given method."""
    sigmas, estat, aris, amis = [], [], [], []

    for dataset, sigma in SIGMA_DATASETS:
        path = os.path.join(RESULTS_DIR, f'{dataset}.joblib')
        if not os.path.exists(path):
            print(f'  Warning: {path} not found, skipping sigma={sigma}')
            continue

        r = joblib.load(path)
        er = r['erica_results']

        res_entry = er.get('results', {}).get((K_EVAL, method), {})
        clam = res_entry.get('clam_matrix')
        if clam is None:
            print(f'  Warning: no CLAM for {dataset}/{method} at K={K_EVAL}, skipping')
            continue

        sigmas.append(sigma)
        estat.append(float(np.mean(erica_stat_from_clam(clam))))

        il = res_entry.get('iteration_labels')
        if il is not None:
            ari_vals, ami_vals = compute_per_iteration_ari_ami(il)
            aris.append(float(np.mean(ari_vals)))
            amis.append(float(np.mean(ami_vals)))
        else:
            m_dict = er.get('metrics', {}).get(K_EVAL, {}).get(method, {})
            aris.append(float(m_dict.get('ARI_mean', np.nan)))
            amis.append(float(m_dict.get('AMI_mean', np.nan)))

    return np.array(sigmas), np.array(estat), np.array(aris), np.array(amis)


def plot_panel(ax, sigmas, estat, aris, amis, title):
    for key, vals in [('ERICA stat', estat), ('ERICA-ARI', aris), ('ERICA-AMI', amis)]:
        style = LINE_STYLES[key]
        ax.plot(sigmas, vals, 'o-', color=style['color'], label=style['label'], markersize=7)

    ax.set_xscale('log')
    ax.set_xlabel(r'$\sigma$ (cluster standard deviation)')
    ax.set_ylabel('Metric value')
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(title, fontweight='bold', fontsize=11)
    ax.set_xticks(sigmas)
    ax.set_xticklabels([str(s) for s in sigmas])
    ax.legend(frameon=False, fontsize=8)


def main():
    set_publication_style()
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print('Collecting K-Means metrics ...')
    km_sigmas, km_estat, km_aris, km_amis = collect_metrics('kmeans')
    print('Collecting Ward metrics ...')
    wd_sigmas, wd_estat, wd_aris, wd_amis = collect_metrics('agglomerative_ward')

    if len(km_sigmas) == 0 and len(wd_sigmas) == 0:
        print('No data found for either method. Exiting.')
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DOUBLE_COL * 1.2, 3.5), sharey=True)

    if len(km_sigmas) > 0:
        plot_panel(ax1, km_sigmas, km_estat, km_aris, km_amis, 'K-Means')
    else:
        ax1.set_title('K-Means (no data)', fontweight='bold', fontsize=11)

    if len(wd_sigmas) > 0:
        plot_panel(ax2, wd_sigmas, wd_estat, wd_aris, wd_amis, 'Ward')
        ax2.set_ylabel('')
    else:
        ax2.set_title('Ward (no data)', fontweight='bold', fontsize=11)

    fig.suptitle(r'Replicability Degradation — Gaussian 4-center (K=4)',
                 fontsize=12, fontweight='bold', y=1.02)
    fig.tight_layout()

    for fmt in ('pdf', 'png'):
        out = os.path.join(FIGURES_DIR, f'sigma_degradation.{fmt}')
        fig.savefig(out, format=fmt, bbox_inches='tight', dpi=300)
        print(f'  Saved: {out}')
    plt.close(fig)

    print('\nSigma degradation complete.')


if __name__ == '__main__':
    main()
