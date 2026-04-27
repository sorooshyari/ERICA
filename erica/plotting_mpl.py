"""Matplotlib-backed plotting module for ERICA.

This module provides publication-quality, matplotlib-based plotting primitives
and statistical plots for working directly with CLAM matrices and ERICA result
dictionaries.

Contents
--------
1. Style constants and helpers (figure widths, colour palettes, rcParams,
   ``save_figure``).
2. Tier-1 primitives: single-axis plots that operate on basic numeric inputs
   (metric curves, CLAM heatmap, cluster size bars).
3. Tier-2 statistical plots that take CLAM matrices directly: Inter-Cluster
   Assignment Heatmap (ICAH), Per-Cluster Scatter Plots (PCSP), and
   replicability metric (CRI/WCRI/TWCRI) curves.
4. ``extract_metric_curves`` convenience extractor for ERICA result dicts.

All single-axis primitives accept ``ax=None`` (creating a new figure on
demand) and return ``(fig, ax)``. Multi-axis plots like PCSP accept ``fig=None``
and return ``fig`` only.
"""

from __future__ import annotations

import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------

# Figure widths for bioinformatics journals (inches)
SINGLE_COL = 3.5
DOUBLE_COL = 7.0

# Colormaps
CMAP_SEQ = "viridis"
CMAP_DIV = "RdBu_r"

# Okabe-Ito colorblind-safe palette for method comparisons
METHOD_COLORS: Dict[str, str] = {
    "kmeans": "#E69F00",
    "agglomerative_ward": "#56B4E9",
    "agglomerative_single": "#009E73",
    "agglomerative_complete": "#F0E442",
    "agglomerative_average": "#0072B2",
    "hdbscan": "#D55E00",
}

# Metric-specific colors
METRIC_COLORS: Dict[str, str] = {
    "CRI": "#0072B2",
    "WCRI": "#D55E00",
    "TWCRI": "#009E73",
}

# Metric-specific dash patterns for K* lines
METRIC_DASHES: Dict[str, Tuple[int, int]] = {
    "CRI": (4, 2),
    "WCRI": (1, 1),
    "TWCRI": (8, 2),
}


def set_publication_style() -> None:
    """Configure matplotlib rcParams for publication-quality figures."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 10,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "lines.linewidth": 2,
        "lines.markersize": 6,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def save_figure(
    fig: plt.Figure,
    name: str,
    output_dir: str,
    formats: Sequence[str] = ("pdf", "png"),
) -> None:
    """Save ``fig`` to ``output_dir`` in each of the given ``formats`` and close it.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save.
    name : str
        Filename stem (no extension).
    output_dir : str
        Directory in which to write the files. Created if it does not exist.
    formats : sequence of str, optional
        File formats to write, e.g. ``("pdf", "png")``.
    """
    os.makedirs(output_dir, exist_ok=True)
    try:
        for fmt in formats:
            path = os.path.join(output_dir, f"{name}.{fmt}")
            fig.savefig(path, format=fmt)
            print(f"  Saved: {path}")
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------------
# Tier-1 primitives
# ---------------------------------------------------------------------------

def _ensure_ax(
    ax: Optional[plt.Axes], figsize: Tuple[float, float] = (SINGLE_COL, SINGLE_COL)
) -> Tuple[plt.Figure, plt.Axes]:
    """Return ``(fig, ax)``. Create a new figure if ``ax`` is None."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    return fig, ax


def plot_metric_vs_k(
    k_values: Sequence[int],
    values: Sequence[float],
    std: Optional[Sequence[float]] = None,
    label: Optional[str] = None,
    color: Optional[str] = None,
    marker: str = "o",
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot a single metric curve vs K with optional ±std band.

    Parameters
    ----------
    k_values : sequence of int
        K values on the x-axis.
    values : sequence of float
        Metric values; same length as ``k_values``. NaNs are dropped from the
        line.
    std : sequence of float, optional
        Per-K standard deviation. If given, a ``fill_between`` band is drawn at
        ``values ± std``.
    label : str, optional
        Legend label for the curve.
    color : str, optional
        Line/marker color.
    marker : str, optional
        Matplotlib marker spec.
    ax : matplotlib.axes.Axes, optional
        Axis to draw on. Created if None.

    Returns
    -------
    (fig, ax) : tuple
    """
    fig, ax = _ensure_ax(ax)

    k_arr = np.asarray(k_values, dtype=float)
    v_arr = np.asarray(values, dtype=float)
    valid = ~np.isnan(v_arr)

    if valid.any():
        ax.plot(
            k_arr[valid], v_arr[valid],
            marker=marker, color=color, label=label,
        )

    if std is not None:
        s_arr = np.asarray(std, dtype=float)
        band_valid = valid & ~np.isnan(s_arr)
        if band_valid.any():
            ax.fill_between(
                k_arr[band_valid],
                v_arr[band_valid] - s_arr[band_valid],
                v_arr[band_valid] + s_arr[band_valid],
                color=color,
                alpha=0.2,
                linewidth=0,
            )

    ax.set_xlabel("K")
    ax.set_ylabel("Metric value")
    if k_arr.size:
        ax.set_xticks(k_arr)
    return fig, ax


def plot_metrics_vs_k(
    k_values: Sequence[int],
    curves: Dict[str, Sequence[float]],
    stds: Optional[Dict[str, Sequence[float]]] = None,
    colors: Optional[Dict[str, str]] = None,
    ax: Optional[plt.Axes] = None,
    ylabel: str = "Metric value",
) -> Tuple[plt.Figure, plt.Axes]:
    """Overlay multiple metric curves on a single axis.

    Parameters
    ----------
    k_values : sequence of int
    curves : dict[str, array-like]
        Mapping ``label -> per-K values``.
    stds : dict[str, array-like], optional
        Mapping ``label -> per-K standard deviations`` for ±std bands.
    colors : dict[str, str], optional
        Mapping ``label -> color``. If None, ``METRIC_COLORS`` is consulted by
        label; falls back to matplotlib defaults.
    ax : matplotlib.axes.Axes, optional
    ylabel : str, optional

    Returns
    -------
    (fig, ax) : tuple
    """
    fig, ax = _ensure_ax(ax, figsize=(DOUBLE_COL, SINGLE_COL))

    if colors is None:
        colors = {}

    k_arr = np.asarray(k_values, dtype=float)

    for label, values in curves.items():
        color = colors.get(label) or METRIC_COLORS.get(label)
        std = None
        if stds is not None and label in stds and stds[label] is not None:
            std = stds[label]
        plot_metric_vs_k(
            k_values=k_values,
            values=values,
            std=std,
            label=label,
            color=color,
            ax=ax,
        )

    ax.set_xlabel("K")
    ax.set_ylabel(ylabel)
    if k_arr.size:
        ax.set_xticks(k_arr)
    ax.legend(frameon=False, fontsize=8)
    return fig, ax


def plot_clam_heatmap(
    clam: np.ndarray,
    sort: bool = True,
    cmap: str = CMAP_SEQ,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Render a CLAM matrix (n_samples, K) as a heatmap with a colorbar.

    Parameters
    ----------
    clam : np.ndarray
        Shape ``(n_samples, K)``.
    sort : bool, optional
        If True, sort rows by primary cluster (argmax along axis=1).
    cmap : str, optional
        Matplotlib colormap name.
    ax : matplotlib.axes.Axes, optional
    title : str, optional

    Returns
    -------
    (fig, ax) : tuple
    """
    fig, ax = _ensure_ax(ax, figsize=(SINGLE_COL, DOUBLE_COL * 0.7))

    clam_arr = np.asarray(clam)
    if sort:
        primary = np.argmax(clam_arr, axis=1)
        order = np.argsort(primary, kind="stable")
        clam_arr = clam_arr[order]

    im = ax.imshow(clam_arr, aspect="auto", cmap=cmap)
    n_samples, k = clam_arr.shape
    ax.set_xticks(np.arange(k))
    ax.set_xticklabels([f"C{i+1}" for i in range(k)])
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Sample")
    if title is not None:
        ax.set_title(title)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return fig, ax


def plot_cluster_sizes(
    clam: np.ndarray,
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Bar chart of cluster sizes (count of samples per primary cluster).

    Primary cluster is ``argmax(clam, axis=1)``.

    Parameters
    ----------
    clam : np.ndarray
        Shape ``(n_samples, K)``.
    ax : matplotlib.axes.Axes, optional

    Returns
    -------
    (fig, ax) : tuple
    """
    fig, ax = _ensure_ax(ax)

    clam_arr = np.asarray(clam)
    k = clam_arr.shape[1]
    primary = np.argmax(clam_arr, axis=1)
    counts = np.bincount(primary, minlength=k)

    labels = [f"C{i+1}" for i in range(k)]
    ax.bar(labels, counts, color="#4477AA")
    for i, c in enumerate(counts):
        ax.text(i, c, str(int(c)), ha="center", va="bottom", fontsize=9)

    ax.set_xlabel("Cluster")
    ax.set_ylabel("Number of samples")
    return fig, ax


# ---------------------------------------------------------------------------
# Tier-2 statistical plots (CLAM-first)
# ---------------------------------------------------------------------------

def compute_icah(clam: np.ndarray, k: int) -> np.ndarray:
    """Compute the Inter-Cluster Assignment Heatmap (ICAH) matrix.

    For each source cluster ``i``, take rows where the primary cluster is
    ``i`` (``primary = argmax(clam, axis=1)``), normalise each row of CLAM to
    sum to 1, and average across rows. Entry ``[i, j]`` is the mean
    normalised assignment from cluster-``i`` samples to cluster-``j``.

    Parameters
    ----------
    clam : np.ndarray
        Shape ``(n_samples, K)``.
    k : int
        Number of clusters.

    Returns
    -------
    np.ndarray
        Shape ``(k, k)``. Rows whose source cluster has no samples are 0.
    """
    clam_arr = np.asarray(clam, dtype=float)
    row_sums = clam_arr.sum(axis=1, keepdims=True)
    safe_sums = np.where(row_sums == 0, 1, row_sums)
    clam_norm = clam_arr / safe_sums
    primary = np.argmax(clam_arr, axis=1)

    icah = np.zeros((k, k))
    for src in range(k):
        mask = primary == src
        if mask.any():
            icah[src, :] = clam_norm[mask, :].mean(axis=0)
    return icah


def plot_icah(
    clam_or_icah: np.ndarray,
    k: Optional[int] = None,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    precomputed: bool = False,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot the Inter-Cluster Assignment Heatmap.

    Accepts either a raw CLAM matrix (computes ICAH internally) or a
    precomputed ICAH matrix. The ``precomputed`` kwarg is the sole switch
    between the two interpretations — there is no auto-detection.

    Parameters
    ----------
    clam_or_icah : np.ndarray
        Either CLAM ``(n_samples, K)`` (when ``precomputed=False``) or ICAH
        ``(K, K)`` (when ``precomputed=True``).
    k : int, optional
        Number of clusters. When computing ICAH from CLAM and the caller wants
        a specific K, pass it here; otherwise the second dim of the CLAM input
        is used. When ``precomputed=True`` and ``k`` is None, uses
        ``icah.shape[0]``.
    ax : matplotlib.axes.Axes, optional
    title : str, optional
    precomputed : bool, optional
        If True, ``clam_or_icah`` is treated as a precomputed ICAH matrix.
        If False (default), it is treated as a CLAM matrix and ICAH is
        computed from it.

    Returns
    -------
    (fig, ax) : tuple
    """
    arr = np.asarray(clam_or_icah)

    if precomputed:
        icah = arr
        if k is None:
            k = icah.shape[0]
    else:
        # arr is a CLAM matrix
        if k is None:
            k = arr.shape[1]
        icah = compute_icah(arr, k)

    k_val = icah.shape[0]

    fig, ax = _ensure_ax(ax, figsize=(SINGLE_COL * 1.3, SINGLE_COL * 1.2))

    im = ax.imshow(icah, cmap="RdYlGn", vmin=0, vmax=1, aspect="equal")

    for i in range(k_val):
        for j in range(k_val):
            val = icah[i, j]
            text_color = "white" if val < 0.4 else "black"
            ax.text(
                j, i, f"{val:.2f}",
                ha="center", va="center",
                fontsize=9, fontweight="bold", color=text_color,
            )

    labels = [f"C{i+1}" for i in range(k_val)]
    ax.set_xticks(np.arange(k_val))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticks(np.arange(k_val))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Target cluster", fontsize=9)
    ax.set_ylabel("Source cluster", fontsize=9)

    if title is None:
        title = f"Inter-Cluster Assignment (K={k_val})"
    ax.set_title(title, fontsize=10)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Mean assignment", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    return fig, ax


def plot_pcsp(
    clam: np.ndarray,
    k: int,
    fig: Optional[plt.Figure] = None,
    title: Optional[str] = None,
) -> plt.Figure:
    """Per-Cluster Scatter Plots (PCSP).

    Creates a grid of scatter subplots — one per source cluster — showing how
    samples primary-assigned to that cluster distribute their normalised
    assignment mass across the other clusters.

    Computation:
      1. Normalize CLAM rows: ``clam_norm = clam / row_sum``.
      2. Find primary cluster per sample: ``primary = argmax(clam, axis=1)``.
      3. For source cluster ``i``, gather samples with ``primary == i``.
      4. For each target cluster ``j != i``, plot those samples'
         ``clam_norm[:, j]`` along the x-axis index.

    Parameters
    ----------
    clam : np.ndarray
        Shape ``(n_samples, K)``.
    k : int
        Number of clusters.
    fig : matplotlib.figure.Figure, optional
        Existing figure to draw into. If None, a new one is created.
        (PCSP creates its own subplot grid; ``ax`` is not applicable.)
    title : str, optional
        Figure suptitle. If None, a default is used.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    clam_arr = np.asarray(clam, dtype=float)
    row_sums = clam_arr.sum(axis=1, keepdims=True)
    safe_sums = np.where(row_sums == 0, 1, row_sums)
    clam_norm = clam_arr / safe_sums
    primary = np.argmax(clam_arr, axis=1)

    ncols = min(k, 4)
    nrows = int(np.ceil(k / ncols))

    if fig is None:
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(4 * ncols, 3.5 * nrows),
            squeeze=False,
        )
    else:
        axes = fig.subplots(nrows, ncols, squeeze=False)

    axes_flat = axes.flatten()

    for idx in range(k, nrows * ncols):
        axes_flat[idx].set_visible(False)

    for src_k in range(k):
        ax = axes_flat[src_k]
        mask = primary == src_k
        if not mask.any():
            ax.set_title(f"C{src_k+1} (empty)", fontsize=10)
            ax.set_ylim(-0.05, 1.05)
            continue

        segments: List[np.ndarray] = []
        boundaries = [0]
        target_labels: List[str] = []
        for tgt_k in range(k):
            if tgt_k == src_k:
                continue
            vals = clam_norm[mask, tgt_k]
            segments.append(vals)
            boundaries.append(boundaries[-1] + len(vals))
            target_labels.append(f"C{tgt_k+1}")

        if not segments:
            ax.set_title(f"C{src_k+1} (n={int(mask.sum())})", fontsize=10)
            ax.set_ylim(-0.05, 1.05)
            continue

        all_vals = np.concatenate(segments)
        x = np.arange(len(all_vals))
        ax.scatter(x, all_vals, s=8, alpha=0.6, c="#4477AA")

        for b in boundaries[1:-1]:
            ax.axvline(b, linestyle=":", color="#888", linewidth=0.8)

        mid_points = [
            (boundaries[i] + boundaries[i + 1]) / 2
            for i in range(len(target_labels))
        ]
        for mid, label in zip(mid_points, target_labels):
            ax.text(
                mid, -0.12, label,
                ha="center", va="top", fontsize=7, color="#555",
            )

        ax.set_title(f"C{src_k+1} (n={int(mask.sum())})", fontsize=10)
        ax.set_xlabel("Index", fontsize=8)
        if src_k % ncols == 0:
            ax.set_ylabel("Assignment to other clusters", fontsize=8)
        ax.set_ylim(-0.05, 1.05)
        ax.tick_params(labelsize=7)

    if title is None:
        title = f"Per-Cluster Scatter Plots (K={k})"
    fig.suptitle(title, fontsize=12, y=1.02)
    fig.tight_layout()
    return fig


def plot_replicability_metrics(
    k_values: Sequence[int],
    cri: Sequence[float],
    wcri: Sequence[float],
    twcri: Sequence[float],
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot CRI, WCRI, TWCRI vs K via :func:`plot_metrics_vs_k`."""
    curves = {
        "CRI": np.asarray(cri, dtype=float),
        "WCRI": np.asarray(wcri, dtype=float),
        "TWCRI": np.asarray(twcri, dtype=float),
    }
    fig, ax = plot_metrics_vs_k(
        k_values=k_values,
        curves=curves,
        ax=ax,
        ylabel="Replicability metric",
    )
    ax.set_title("Replicability Metrics")
    return fig, ax


# ---------------------------------------------------------------------------
# Tier-2 extractor for ERICA result dicts
# ---------------------------------------------------------------------------

def extract_metric_curves(
    er: Dict,
    method: str,
    metric_keys: Iterable[str],
    std_keys: Optional[Iterable[Optional[str]]] = None,
) -> Tuple[List[int], Dict[str, np.ndarray], Optional[Dict[str, np.ndarray]]]:
    """Extract per-K metric curves from an ERICA result dict.

    Pulls values from ``er['metrics'][k][method][metric_key]``. Missing keys or
    non-numeric entries become ``np.nan``.

    Parameters
    ----------
    er : dict
        ERICA result dict (must contain a ``'metrics'`` key whose value is a
        dict keyed by K).
    method : str
        Clustering method name (e.g. ``'kmeans'``).
    metric_keys : iterable of str
        Metric keys to pull, e.g. ``['CRI', 'WCRI', 'TWCRI']``.
    std_keys : iterable of (str or None), optional
        Aligned with ``metric_keys`` — the std-key to fetch for each metric.
        Use ``None`` for entries where no std is available. If ``std_keys`` is
        omitted, the returned ``stds`` dict is None.

    Returns
    -------
    (k_values, curves, stds) : tuple
        ``k_values`` : sorted list of K
        ``curves`` : ``{metric_key: np.ndarray}`` of values per K
        ``stds`` : ``{metric_key: np.ndarray}`` per K (or None if ``std_keys``
        was not provided)
    """
    metrics = er.get("metrics", {})
    k_values = sorted(int(k) for k in metrics.keys())

    metric_keys = list(metric_keys)
    if std_keys is not None:
        std_keys = list(std_keys)
        if len(std_keys) != len(metric_keys):
            raise ValueError(
                "std_keys must have the same length as metric_keys"
            )

    curves: Dict[str, np.ndarray] = {}
    stds: Optional[Dict[str, np.ndarray]] = {} if std_keys is not None else None

    for idx, mkey in enumerate(metric_keys):
        vals: List[float] = []
        s_vals: List[float] = []
        for k in k_values:
            m_dict = metrics.get(k, {}).get(method, {})

            v = m_dict.get(mkey, np.nan)
            try:
                v = float(v)
            except (TypeError, ValueError):
                v = np.nan
            vals.append(v)

            if std_keys is not None:
                skey = std_keys[idx]
                if skey is None:
                    s_vals.append(np.nan)
                else:
                    s = m_dict.get(skey, np.nan)
                    try:
                        s = float(s)
                    except (TypeError, ValueError):
                        s = np.nan
                    s_vals.append(s)

        curves[mkey] = np.asarray(vals, dtype=float)
        if std_keys is not None and stds is not None:
            stds[mkey] = np.asarray(s_vals, dtype=float)

    return k_values, curves, stds
