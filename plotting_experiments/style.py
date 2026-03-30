"""Shared publication style settings for ERICA plotting experiments."""

import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Figure widths for bioinformatics journals (inches)
SINGLE_COL = 3.5
DOUBLE_COL = 7.0

# Colormaps
CMAP_SEQ = 'viridis'
CMAP_DIV = 'RdBu_r'

# Okabe-Ito colorblind-safe palette for method comparisons
METHOD_COLORS = {
    'kmeans': '#E69F00',
    'agglomerative_ward': '#56B4E9',
    'agglomerative_single': '#009E73',
    'agglomerative_complete': '#F0E442',
    'agglomerative_average': '#0072B2',
    'hdbscan': '#D55E00',
}

# Metric-specific colors
METRIC_COLORS = {
    'CRI': '#0072B2',
    'WCRI': '#D55E00',
    'TWCRI': '#009E73',
    'ARI': '#E69F00',
    'AMI': '#CC79A7',
}

# Metric-specific dash patterns for K* lines
METRIC_DASHES = {
    'CRI': (4, 2),
    'WCRI': (1, 1),
    'TWCRI': (8, 2),
}


def set_publication_style():
    """Configure matplotlib rcParams for publication-quality figures."""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans'],
        'font.size': 10,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 9,
        'lines.linewidth': 2,
        'lines.markersize': 6,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.spines.top': False,
        'axes.spines.right': False,
    })


def save_figure(fig, name, formats=('pdf', 'png')):
    """Save figure to figures/ directory in specified formats."""
    fig_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    for fmt in formats:
        path = os.path.join(fig_dir, f'{name}.{fmt}')
        fig.savefig(path, format=fmt)
        print(f'  Saved: {path}')
    plt.close(fig)
