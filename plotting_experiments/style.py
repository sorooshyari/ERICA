"""Backward-compat shim — style constants now live in erica.plotting_mpl."""
from erica.plotting_mpl import (
    SINGLE_COL, DOUBLE_COL, CMAP_SEQ, CMAP_DIV,
    METHOD_COLORS, METRIC_COLORS, METRIC_DASHES,
    set_publication_style,
)

# Keep the old save_figure with the original hardcoded "figures" path behavior
import os
import matplotlib.pyplot as plt


def save_figure(fig, name, formats=('pdf', 'png')):
    """Save figure to figures/ directory in specified formats."""
    fig_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    for fmt in formats:
        path = os.path.join(fig_dir, f'{name}.{fmt}')
        fig.savefig(path, format=fmt)
        print(f'  Saved: {path}')
    plt.close(fig)
