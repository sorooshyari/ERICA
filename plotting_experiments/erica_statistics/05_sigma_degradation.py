"""Sigma degradation study for Gaussian 4-center datasets.

Single-line plot showing the ERICA statistic vs sigma, side-by-side for
K-Means and Ward.
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
}


def erica_stat_from_clam(clam):
    row_sums = clam.sum(axis=1)
    row_sums[row_sums == 0] = 1
    return clam.max(axis=1) / row_sums


def collect_metrics(method):
    """Return arrays of (sigmas, means, stds) for the ERICA statistic."""
    sigmas = []
    estat_mean, estat_std = [], []

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
        e_vals = erica_stat_from_clam(clam)
        estat_mean.append(float(np.mean(e_vals)))
        estat_std.append(float(np.std(e_vals)))

    return (
        np.array(sigmas),
        np.array(estat_mean), np.array(estat_std),
    )


def plot_panel(ax, sigmas, means_dict, stds_dict, title):
    for key in ['ERICA stat']:
        m_vals = means_dict[key]
        s_vals = stds_dict[key]
        style = LINE_STYLES[key]
        ax.plot(sigmas, m_vals, 'o-', color=style['color'],
                label=style['label'], markersize=7)
        ax.fill_between(
            sigmas,
            np.clip(m_vals - s_vals, -0.05, 1.05),
            np.clip(m_vals + s_vals, -0.05, 1.05),
            color=style['color'], alpha=0.15,
        )

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
    km_s, km_em, km_es = collect_metrics('kmeans')
    print('Collecting Ward metrics ...')
    wd_s, wd_em, wd_es = collect_metrics('agglomerative_ward')

    if len(km_s) == 0 and len(wd_s) == 0:
        print('No data found for either method. Exiting.')
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DOUBLE_COL * 1.2, 3.5), sharey=True)

    if len(km_s) > 0:
        km_means = {'ERICA stat': km_em}
        km_stds = {'ERICA stat': km_es}
        plot_panel(ax1, km_s, km_means, km_stds, 'K-Means')
    else:
        ax1.set_title('K-Means (no data)', fontweight='bold', fontsize=11)

    if len(wd_s) > 0:
        wd_means = {'ERICA stat': wd_em}
        wd_stds = {'ERICA stat': wd_es}
        plot_panel(ax2, wd_s, wd_means, wd_stds, 'Ward')
        ax2.set_ylabel('')
    else:
        ax2.set_title('Ward (no data)', fontweight='bold', fontsize=11)

    fig.suptitle(r'Replicability Degradation - Gaussian 4-center (K=4)',
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
