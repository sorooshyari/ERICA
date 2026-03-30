"""Sigma-sweep curves for the Gaussian mixture study.

Generates four figures:
  A) TWCRI at K=4 vs sigma (log x-axis), one line per K-based method,
     HDBSCAN as discrete markers.
  B) ARI at K=4 vs sigma (same format), AMI as dashed lines.
  C) HDBSCAN noise fraction vs sigma.
  D) K* summary bar chart — grouped bars, one group per sigma, bars per method.

Note: Uses joblib to load locally-generated ERICA results from 02_run_erica.py.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from style import set_publication_style, DOUBLE_COL, SINGLE_COL, METHOD_COLORS, METRIC_COLORS

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, '..', 'results')
FIGURES_DIR = os.path.join(SCRIPT_DIR, '..', 'figures', 'gaussian_study')

SIGMAS_INFO = [
    ('gauss4c_sigma0p01', 0.01),
    ('gauss4c_sigma0p1',  0.1),
    ('gauss4c_sigma1p0',  1.0),
    ('gauss4c_sigma10p0', 10.0),
]

METHODS = ['kmeans', 'agglomerative_ward', 'agglomerative_single']
METHOD_LABELS = {
    'kmeans': 'K-Means',
    'agglomerative_ward': 'Ward',
    'agglomerative_single': 'Single',
    'hdbscan': 'HDBSCAN',
}
K_EVAL = 4  # We evaluate metrics at the true K


def save_fig(fig, name, formats=('pdf', 'png')):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    for fmt in formats:
        path = os.path.join(FIGURES_DIR, f'{name}.{fmt}')
        fig.savefig(path, format=fmt, bbox_inches='tight', dpi=300)
        print(f'  Saved: {path}')
    plt.close(fig)


def load_all_results():
    """Load results for all sigma levels, return list of (sigma, er) tuples."""
    results = []
    for fname, sigma in SIGMAS_INFO:
        path = os.path.join(RESULTS_DIR, f'{fname}.joblib')
        if not os.path.exists(path):
            print(f'  Warning: {path} not found, skipping')
            continue
        r = joblib.load(path)
        results.append((sigma, r['erica_results'], r.get('config', {})))
    return results


def plot_twcri_vs_sigma(results):
    """Figure A: TWCRI at K=4 vs sigma."""
    fig, ax = plt.subplots(figsize=(SINGLE_COL * 1.8, 3.5))

    sigmas = np.array([s for s, _, _ in results])

    for method in METHODS:
        vals = []
        valid_sigmas = []
        nan_sigmas = []
        for sigma, er, _ in results:
            m_dict = er['metrics'].get(K_EVAL, {}).get(method, {})
            twcri = m_dict.get('TWCRI', np.nan)
            try:
                twcri = float(twcri)
            except (TypeError, ValueError):
                twcri = np.nan
            if np.isnan(twcri):
                nan_sigmas.append(sigma)
            else:
                vals.append(twcri)
                valid_sigmas.append(sigma)

        color = METHOD_COLORS[method]
        if valid_sigmas:
            ax.plot(valid_sigmas, vals, 'o-', color=color,
                    label=METHOD_LABELS[method], markersize=7)
        if nan_sigmas:
            ax.plot(nan_sigmas, [0] * len(nan_sigmas), 'x', color=color,
                    markersize=10, markeredgewidth=2)

    # HDBSCAN as discrete markers
    hdb_sigmas = []
    hdb_vals = []
    for sigma, er, _ in results:
        hdb = er.get('auto_k', {}).get('hdbscan', {})
        metrics_modal = hdb.get('metrics_at_modal_k', {})
        twcri = metrics_modal.get('TWCRI', np.nan)
        if not np.isnan(twcri):
            hdb_sigmas.append(sigma)
            hdb_vals.append(twcri)
    if hdb_sigmas:
        ax.plot(hdb_sigmas, hdb_vals, 'D', color=METHOD_COLORS['hdbscan'],
                label='HDBSCAN', markersize=8, markeredgecolor='black',
                markeredgewidth=0.5, linestyle='none')

    ax.set_xscale('log')
    ax.set_xlabel(r'$\sigma$')
    ax.set_ylabel('TWCRI')
    ax.set_title(f'TWCRI at K={K_EVAL} vs Noise Level', fontweight='bold')
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks(sigmas)
    ax.set_xticklabels([str(s) for s in sigmas])
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig


def plot_ari_ami_vs_sigma(results):
    """Figure B: ARI and AMI at K=4 vs sigma."""
    fig, ax = plt.subplots(figsize=(SINGLE_COL * 1.8, 3.5))

    sigmas = np.array([s for s, _, _ in results])

    for method in METHODS:
        color = METHOD_COLORS[method]
        ari_vals, ari_sigmas = [], []
        ami_vals, ami_sigmas = [], []
        nan_sigmas = []

        for sigma, er, _ in results:
            m_dict = er['metrics'].get(K_EVAL, {}).get(method, {})
            ari = float(m_dict.get('ARI_mean', np.nan))
            ami = float(m_dict.get('AMI_mean', np.nan))

            if np.isnan(ari):
                nan_sigmas.append(sigma)
            else:
                ari_vals.append(ari)
                ari_sigmas.append(sigma)

            if not np.isnan(ami):
                ami_vals.append(ami)
                ami_sigmas.append(sigma)

        label_base = METHOD_LABELS[method]
        if ari_sigmas:
            ax.plot(ari_sigmas, ari_vals, 'o-', color=color,
                    label=f'{label_base} ARI', markersize=6)
        if ami_sigmas:
            ax.plot(ami_sigmas, ami_vals, 's--', color=color,
                    label=f'{label_base} AMI', markersize=5, alpha=0.7)
        if nan_sigmas:
            ax.plot(nan_sigmas, [0] * len(nan_sigmas), 'x', color=color,
                    markersize=10, markeredgewidth=2)

    # HDBSCAN discrete markers
    for metric_key, marker, ls_label in [('ARI_mean', 'D', 'ARI'), ('AMI_mean', 'd', 'AMI')]:
        hdb_s, hdb_v = [], []
        for sigma, er, _ in results:
            hdb = er.get('auto_k', {}).get('hdbscan', {})
            val = float(hdb.get(metric_key, np.nan))
            if not np.isnan(val):
                hdb_s.append(sigma)
                hdb_v.append(val)
        if hdb_s:
            ls = '-' if ls_label == 'ARI' else '--'
            ax.plot(hdb_s, hdb_v, marker, color=METHOD_COLORS['hdbscan'],
                    label=f'HDBSCAN {ls_label}', markersize=7,
                    markeredgecolor='black', markeredgewidth=0.5,
                    linestyle='none')

    ax.set_xscale('log')
    ax.set_xlabel(r'$\sigma$')
    ax.set_ylabel('Metric Value')
    ax.set_title(f'ARI / AMI at K={K_EVAL} vs Noise Level', fontweight='bold')
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks(sigmas)
    ax.set_xticklabels([str(s) for s in sigmas])
    ax.legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout()
    return fig


def plot_hdbscan_noise_fraction(results):
    """Figure C: HDBSCAN noise fraction vs sigma."""
    fig, ax = plt.subplots(figsize=(SINGLE_COL * 1.8, 3.0))

    sigmas_plot = []
    fracs = []

    for sigma, er, cfg in results:
        hdb = er.get('auto_k', {}).get('hdbscan', {})
        noise_counts = hdb.get('noise_counts', [])
        if len(noise_counts) == 0:
            continue
        # n_test_samples: each iteration uses a test split; the CLAM matrix
        # rows = n_samples total, but noise_counts are per iteration.
        # Estimate n_test from config or CLAM shape.
        n_samples = cfg.get('n_samples', 400)
        # Default test fraction is 0.5 in ERICA
        n_test = n_samples // 2
        mean_noise = np.mean(noise_counts)
        frac = mean_noise / n_test
        sigmas_plot.append(sigma)
        fracs.append(frac)

    ax.plot(sigmas_plot, fracs, 'o-', color=METHOD_COLORS['hdbscan'],
            markersize=8, linewidth=2)

    ax.set_xscale('log')
    ax.set_xlabel(r'$\sigma$')
    ax.set_ylabel('Noise Fraction')
    ax.set_title('HDBSCAN Noise Fraction vs Noise Level', fontweight='bold')
    ax.set_ylim(-0.02, max(fracs) * 1.15 if fracs else 1.0)
    sigmas_all = [s for s, _, _ in results]
    ax.set_xticks(sigmas_all)
    ax.set_xticklabels([str(s) for s in sigmas_all])
    fig.tight_layout()
    return fig


def plot_kstar_summary(results):
    """Figure D: K* summary bar chart — grouped bars per sigma per method."""
    fig, ax = plt.subplots(figsize=(DOUBLE_COL, 3.5))

    sigma_labels = [f'\u03c3={s}' for s, _, _ in results]
    n_groups = len(results)

    all_methods = METHODS.copy()
    method_labels = [METHOD_LABELS[m] for m in all_methods]

    n_bars = len(all_methods)
    bar_width = 0.7 / n_bars
    x = np.arange(n_groups)

    for i, method in enumerate(all_methods):
        kstar_vals = []
        for sigma, er, _ in results:
            kstar = er.get('k_star', {}).get('TWCRI', {}).get(method, np.nan)
            kstar_vals.append(kstar)

        offset = (i - (n_bars - 1) / 2) * bar_width
        bars = ax.bar(x + offset, kstar_vals, bar_width * 0.9,
                      color=METHOD_COLORS[method],
                      label=METHOD_LABELS[method],
                      edgecolor='white', linewidth=0.5)

        # Add value labels on bars
        for bar, val in zip(bars, kstar_vals):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                        str(int(val)), ha='center', va='bottom', fontsize=8)

    ax.axhline(y=4, color='gray', linestyle='--', linewidth=1, alpha=0.7,
               label='True K=4')
    ax.set_xlabel('Noise Level')
    ax.set_ylabel('K* (TWCRI)')
    ax.set_title('Optimal K* by Method and Noise Level', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(sigma_labels)
    ax.set_ylim(0, 8)
    ax.set_yticks(range(0, 9))
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig


def main():
    set_publication_style()
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print('Loading all sigma results ...')
    results = load_all_results()
    if not results:
        print('No results found. Exiting.')
        return

    print(f'Loaded {len(results)} sigma levels.')

    print('\nFigure A: TWCRI vs sigma')
    fig_a = plot_twcri_vs_sigma(results)
    save_fig(fig_a, 'sigma_twcri_curves')

    print('\nFigure B: ARI/AMI vs sigma')
    fig_b = plot_ari_ami_vs_sigma(results)
    save_fig(fig_b, 'sigma_ari_ami_curves')

    print('\nFigure C: HDBSCAN noise fraction')
    fig_c = plot_hdbscan_noise_fraction(results)
    save_fig(fig_c, 'sigma_hdbscan_noise')

    print('\nFigure D: K* summary')
    fig_d = plot_kstar_summary(results)
    save_fig(fig_d, 'sigma_kstar_summary')

    print('\nSigma curves complete.')


if __name__ == '__main__':
    main()
