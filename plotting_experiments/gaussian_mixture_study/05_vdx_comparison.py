"""VDX vs Gaussian comparison figures.

Side-by-side TWCRI curves: VDX (3-gene) on the left, sigma=1.0 Gaussian
on the right. Shared y-axis scale so the two datasets are directly
comparable.

Note: Uses joblib to load locally-generated ERICA results from 02_run_erica.py.
The VDX file may require a numpy compatibility shim (numpy._core -> numpy.core).
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import joblib

# Numpy version compatibility: the VDX joblib file may have been saved with
# numpy >= 2.0 which uses numpy._core instead of numpy.core.
if 'numpy._core' not in sys.modules:
    sys.modules['numpy._core'] = type(sys)('numpy._core')
    sys.modules['numpy._core'].multiarray = np.core.multiarray
if 'numpy._core.multiarray' not in sys.modules:
    sys.modules['numpy._core.multiarray'] = np.core.multiarray

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from style import set_publication_style, DOUBLE_COL, METHOD_COLORS

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, '..', 'results')
FIGURES_DIR = os.path.join(SCRIPT_DIR, '..', 'figures', 'gaussian_study')

METHODS = ['kmeans', 'agglomerative_ward', 'agglomerative_single']
METHOD_LABELS = {
    'kmeans': 'K-Means',
    'agglomerative_ward': 'Ward',
    'agglomerative_single': 'Single',
    'hdbscan': 'HDBSCAN',
}

MARKERS = {
    'kmeans': 'o',
    'agglomerative_ward': 's',
    'agglomerative_single': '^',
}


def save_fig(fig, name, formats=('pdf', 'png')):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    for fmt in formats:
        path = os.path.join(FIGURES_DIR, f'{name}.{fmt}')
        fig.savefig(path, format=fmt, bbox_inches='tight', dpi=300)
        print(f'  Saved: {path}')
    plt.close(fig)


def extract_metric_curves(er, metric_key, stat_key=None):
    """Extract metric values for all methods across K values.

    Parameters
    ----------
    er : dict
        erica_results dict.
    metric_key : str
        e.g. 'TWCRI'.
    stat_key : str or None
        If provided, also extract this for error bars.

    Returns
    -------
    k_values : list of int
    curves : dict of method -> array of values
    errors : dict of method -> array of std values (or None)
    """
    metrics = er['metrics']
    k_values = sorted(metrics.keys())
    curves = {}
    errors = {}

    for method in METHODS:
        vals = []
        stds = []
        for k in k_values:
            m_dict = metrics[k].get(method, {})
            v = m_dict.get(metric_key, np.nan)
            try:
                v = float(v)
            except (TypeError, ValueError):
                v = np.nan
            vals.append(v)
            if stat_key:
                s = m_dict.get(stat_key, np.nan)
                try:
                    s = float(s)
                except (TypeError, ValueError):
                    s = np.nan
                stds.append(s)
        curves[method] = np.array(vals)
        errors[method] = np.array(stds) if stat_key else None

    return k_values, curves, errors


def plot_comparison(er_vdx, er_gauss):
    """Create 1x2 comparison figure: TWCRI vs K (VDX left, Gauss right)."""
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 3.0), sharey=True)

    datasets = [
        ('VDX 3-Gene', er_vdx),
        (r'Gaussian ($\sigma$=1.0)', er_gauss),
    ]
    metric_label = 'TWCRI'
    metric_key = 'TWCRI'

    for col, (ds_label, er) in enumerate(datasets):
        ax = axes[col]
        k_values, curves, _ = extract_metric_curves(er, metric_key)
        k_arr = np.array(k_values, dtype=float)

        for method in METHODS:
            vals = curves[method]
            color = METHOD_COLORS[method]
            marker = MARKERS[method]
            valid = ~np.isnan(vals)

            if valid.any():
                ax.plot(k_arr[valid], vals[valid], marker=marker,
                        color=color, linestyle='-',
                        label=METHOD_LABELS[method] if col == 0 else None,
                        markersize=6, linewidth=1.5)

        # HDBSCAN marker: place at its modal_k
        hdb = er.get('auto_k', {}).get('hdbscan', {})
        modal_k = hdb.get('modal_k', None)
        if modal_k is not None:
            modal_k = int(modal_k)
            hdb_metrics = hdb.get('metrics_at_modal_k', {})
            hdb_val = hdb_metrics.get('TWCRI', np.nan)

            if not np.isnan(hdb_val):
                ax.plot(modal_k, hdb_val, 'D',
                        color=METHOD_COLORS['hdbscan'],
                        markersize=9, markeredgecolor='black',
                        markeredgewidth=0.5,
                        label='HDBSCAN' if col == 0 else None,
                        zorder=5)

        ax.set_xticks(k_values)
        ax.set_xlim(min(k_values) - 0.3, max(k_values) + 0.3)
        ax.set_xlabel('K')
        if col == 0:
            ax.set_ylabel(metric_label)
        ax.set_title(ds_label, fontsize=11, fontweight='bold')

    # Shared y limits across the two panels
    ymin = min(ax.get_ylim()[0] for ax in axes)
    ymax = max(ax.get_ylim()[1] for ax in axes)
    margin = (ymax - ymin) * 0.05
    for ax in axes:
        ax.set_ylim(max(-0.05, ymin - margin), min(1.05, ymax + margin))

    # Legend in left panel
    axes[0].legend(frameon=False, fontsize=8, loc='best')

    fig.suptitle('VDX vs Gaussian TWCRI Comparison', fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def main():
    set_publication_style()
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # Load VDX
    vdx_path = os.path.join(RESULTS_DIR, 'vdx_3gene.joblib')
    gauss_path = os.path.join(RESULTS_DIR, 'gauss4c_sigma1p0.joblib')

    if not os.path.exists(vdx_path):
        print(f'VDX results not found at {vdx_path}')
        return
    if not os.path.exists(gauss_path):
        print(f'Gaussian results not found at {gauss_path}')
        return

    print('Loading VDX 3-gene results ...')
    vdx = joblib.load(vdx_path)
    er_vdx = vdx['erica_results']

    print('Loading Gaussian sigma=1.0 results ...')
    gauss = joblib.load(gauss_path)
    er_gauss = gauss['erica_results']

    print('\nGenerating VDX vs Gaussian comparison figure ...')
    fig = plot_comparison(er_vdx, er_gauss)
    save_fig(fig, 'vdx_gaussian_comparison')

    print('\nVDX comparison complete.')


if __name__ == '__main__':
    main()
