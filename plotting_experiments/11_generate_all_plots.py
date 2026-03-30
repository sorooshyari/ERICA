"""
11_generate_all_plots.py

Generate the complete plot suite for EVERY dataset that has results in results/.

For each dataset, generates (where applicable):
  - Sorted CLAM heatmap at true_k (or K=3 fallback) for kmeans
  - Stability strips at true_k for kmeans
  - Metrics curves (ERICA + ARI/AMI) for kmeans and ward
  - Method comparison (3-panel CRI/WCRI/TWCRI)
  - K* bar chart
  - 3D CLAM surface at true_k for kmeans
  - TWCRI landscape 2D (K x method heatmap)
  - For 2D datasets (n_features==2): cluster scatter + confidence map

All figures saved to figures/by_dataset/{dataset_name}/.

Note: joblib is used here to load locally-generated ERICA result files produced
by 02_run_erica_pipeline.py, not untrusted external data.
"""

import sys
import os
import glob
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import joblib

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ---------------------------------------------------------------------------
# Import shared style
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from style import (
    set_publication_style,
    CMAP_SEQ,
    DOUBLE_COL,
    SINGLE_COL,
    METHOD_COLORS,
    METRIC_COLORS,
    METRIC_DASHES,
)

RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
FIGURES_ROOT = os.path.join(SCRIPT_DIR, "figures", "by_dataset")

K_BASED_METHODS = ["kmeans", "agglomerative_ward"]
ERICA_METRICS = ["CRI", "WCRI", "TWCRI"]
PARMI_METRICS = ["ARI", "AMI"]

METHOD_LABELS = {
    "kmeans": "K-Means",
    "agglomerative_ward": "Agglo. Ward",
    "hdbscan": "HDBSCAN",
}

CLUSTER_COLORS = [
    "#4477AA", "#EE6677", "#228833", "#CCBB44",
    "#66CCEE", "#AA3377", "#BBBBBB", "#332288",
]

SKIP_PATTERNS = ["sweep", "summary", "workdir"]


# ============================================================================
# Utility
# ============================================================================

def save_fig(fig, out_dir, name, formats=("pdf", "png")):
    """Save figure to out_dir in specified formats, then close."""
    os.makedirs(out_dir, exist_ok=True)
    for fmt in formats:
        path = os.path.join(out_dir, f"{name}.{fmt}")
        fig.savefig(path, format=fmt, bbox_inches="tight", dpi=300)
    plt.close(fig)


def get_methods_in_data(er):
    """Return set of method strings present in the metrics dict."""
    methods = set()
    for k in er["metrics"]:
        methods.update(er["metrics"][k].keys())
    return methods


def dataset_name_from_path(path):
    """Derive dataset name from joblib filename."""
    return os.path.splitext(os.path.basename(path))[0]


def find_data_file(name):
    """Locate the .npz data file for a dataset name.

    Checks data/{name}.npz first, then data/gmm_sweep/{name}.npz.
    """
    p1 = os.path.join(DATA_DIR, f"{name}.npz")
    if os.path.exists(p1):
        return p1
    p2 = os.path.join(DATA_DIR, "gmm_sweep", f"{name}.npz")
    if os.path.exists(p2):
        return p2
    return None


# ============================================================================
# 1. Sorted CLAM heatmap
# ============================================================================

def sort_clam(clam_matrix):
    """Sort a CLAM matrix by primary cluster assignment and strength."""
    clam = np.array(clam_matrix, dtype=float)
    row_sums = clam.sum(axis=1)
    primary = np.argmax(clam, axis=1)
    strength = np.where(row_sums > 0, clam.max(axis=1) / row_sums, 0.0)
    sort_indices = np.lexsort((-strength, primary))
    return clam[sort_indices], sort_indices


def plot_clam_heatmap(clam_matrix, title):
    """Plot a sorted CLAM matrix as a heatmap. Returns fig."""
    clam_sorted, _ = sort_clam(clam_matrix)
    k = clam_sorted.shape[1]

    fig, ax = plt.subplots(figsize=(4, 5))
    im = ax.imshow(clam_sorted, aspect="auto", cmap=CMAP_SEQ, interpolation="nearest")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Co-assignment count")
    ax.set_xticks(range(k))
    ax.set_xticklabels([f"C{i+1}" for i in range(k)], fontsize=9)
    ax.set_ylabel("Sample (sorted)")
    ax.set_xlabel("Cluster")
    ax.set_title(title)
    fig.tight_layout()
    return fig


# ============================================================================
# 2. Stability strips
# ============================================================================

MAX_SAMPLES_STRIP = 100


def normalize_clam(clam_matrix):
    clam = np.array(clam_matrix, dtype=float)
    row_sums = clam.sum(axis=1)
    valid = row_sums > 0
    props = np.zeros_like(clam)
    props[valid] = clam[valid] / row_sums[valid, np.newaxis]
    return props, valid


def shannon_entropy(proportions):
    with np.errstate(divide="ignore", invalid="ignore"):
        log_p = np.where(proportions > 0, np.log2(proportions), 0.0)
    return -np.sum(proportions * log_p, axis=1)


def plot_stability_strips(clam_matrix, title, max_samples=MAX_SAMPLES_STRIP):
    """Horizontal stacked-bar stability strip plot."""
    proportions, _ = normalize_clam(clam_matrix)
    entropy = shannon_entropy(proportions)
    n_samples, k = proportions.shape

    sort_order = np.argsort(entropy)
    if n_samples > max_samples:
        sort_order = sort_order[:max_samples]

    props_sorted = proportions[sort_order]
    ent_sorted = entropy[sort_order]
    n_display = len(sort_order)

    bar_height = 0.6
    fig_height = max(2.0, min(8.0, n_display * bar_height * 0.12 + 1.5))
    fig, ax = plt.subplots(figsize=(SINGLE_COL + 0.5, fig_height))

    y_pos = np.arange(n_display)
    colors = [CLUSTER_COLORS[i % len(CLUSTER_COLORS)] for i in range(k)]

    lefts = np.zeros(n_display)
    for c in range(k):
        widths = props_sorted[:, c]
        ax.barh(y_pos, widths, left=lefts, height=bar_height,
                color=colors[c], label=f"C{c+1}", linewidth=0)
        lefts += widths

    ax.set_xlim(0, 1)
    ax.set_xlabel("Proportion of iterations")
    ax.set_ylabel("Sample (sorted by entropy)")
    ax.set_title(title, fontsize=10)
    ax.set_yticks([0, n_display - 1])
    ax.set_yticklabels(["most stable", "least stable"], fontsize=8)

    ax_right = ax.twinx()
    ax_right.set_ylim(ax.get_ylim())
    ax_right.set_yticks([0, n_display - 1])
    ax_right.set_yticklabels(
        [f"H={ent_sorted[0]:.2f}", f"H={ent_sorted[-1]:.2f}"], fontsize=7)
    ax_right.set_ylabel("Shannon entropy (bits)", fontsize=8)
    ax_right.spines["top"].set_visible(False)

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18),
              ncol=min(k, 4), fontsize=7, frameon=False)

    if n_samples > max_samples:
        ax.text(0.5, 1.02, f"(showing {n_display} of {n_samples} samples)",
                transform=ax.transAxes, ha="center", fontsize=7, color="gray")
    fig.tight_layout()
    return fig


# ============================================================================
# 3. Metrics curves (ERICA + ARI/AMI)
# ============================================================================

def plot_metrics_curves(er, dataset_label, method):
    """Two-panel metrics curves for one dataset/method."""
    metrics = er["metrics"]
    k_star = er["k_star"]
    k_values = sorted(metrics.keys())

    erica_vals = {m: [] for m in ERICA_METRICS}
    ari_means, ari_stds = [], []
    ami_means, ami_stds = [], []

    for k in k_values:
        m_dict = metrics[k].get(method, {})
        for metric in ERICA_METRICS:
            val = m_dict.get(metric, np.nan)
            try:
                val = float(val)
            except (TypeError, ValueError):
                val = np.nan
            erica_vals[metric].append(val)
        ari_means.append(float(m_dict.get("ARI_mean", np.nan)))
        ari_stds.append(float(m_dict.get("ARI_std", np.nan)))
        ami_means.append(float(m_dict.get("AMI_mean", np.nan)))
        ami_stds.append(float(m_dict.get("AMI_std", np.nan)))

    k_arr = np.array(k_values, dtype=float)
    ari_means = np.array(ari_means)
    ari_stds = np.array(ari_stds)
    ami_means = np.array(ami_means)
    ami_stds = np.array(ami_stds)

    disq_mask = np.array([
        any(np.isnan(erica_vals[m][i]) for m in ERICA_METRICS)
        for i in range(len(k_values))
    ])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DOUBLE_COL, 3.5))

    # Panel 1: ERICA
    for metric in ERICA_METRICS:
        vals = np.array(erica_vals[metric])
        color = METRIC_COLORS[metric]
        dashes = METRIC_DASHES[metric]
        valid = ~np.isnan(vals)
        if valid.any():
            ax1.plot(k_arr[valid], vals[valid], color=color, dashes=dashes,
                     marker="o", label=metric, zorder=3)
        disq_nan = disq_mask & np.isnan(vals)
        if disq_nan.any():
            ax1.plot(k_arr[disq_nan], np.zeros(disq_nan.sum()), marker="x",
                     color=color, linestyle="none", markersize=8,
                     markeredgewidth=2, zorder=4)
        k_star_val = k_star.get(metric, {}).get(method)
        if k_star_val is not None:
            ax1.axvline(x=k_star_val, color=color, dashes=dashes,
                        linewidth=1.2, alpha=0.7, zorder=2)

    ax1.set_xlabel("K")
    ax1.set_ylabel("Metric value")
    ax1.set_title("ERICA metrics")
    ax1.set_xticks(k_values)
    ax1.set_ylim(-0.05, 1.05)
    ax1.legend(frameon=False, fontsize=8)

    # Panel 2: Parmigiani
    ax2.errorbar(k_arr, ari_means, yerr=ari_stds, color=METRIC_COLORS["ARI"],
                 marker="o", capsize=3, label="ARI", zorder=3)
    ax2.errorbar(k_arr, ami_means, yerr=ami_stds, color=METRIC_COLORS["AMI"],
                 marker="s", capsize=3, label="AMI", zorder=3)
    ax2.set_xlabel("K")
    ax2.set_ylabel("Metric value")
    ax2.set_title("Parmigiani metrics")
    ax2.set_xticks(k_values)
    ax2.legend(frameon=False, fontsize=8)

    method_label = method.replace("_", " ").title()
    fig.suptitle(f"{dataset_label} - {method_label}", fontsize=11, y=1.01)
    fig.tight_layout()
    return fig


# ============================================================================
# 4. Method comparison (3-panel CRI/WCRI/TWCRI)
# ============================================================================

def plot_method_comparison(er, dataset_label):
    """1x3 subplot: each panel shows one metric across K for each method."""
    metrics = er["metrics"]
    k_values = sorted(metrics.keys())
    k_arr = np.array(k_values, dtype=float)

    hdb = er.get("auto_k", {}).get("hdbscan", {})
    hdb_valid = (isinstance(hdb, dict)
                 and hdb.get("k_agreement_rate", 0) > 0
                 and "metrics_at_modal_k" in hdb)
    hdb_modal_k = hdb.get("modal_k") if hdb_valid else None
    hdb_metric_vals = hdb.get("metrics_at_modal_k", {}) if hdb_valid else {}

    methods_in = get_methods_in_data(er)
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL * 1.3, 3.6), sharey=False)

    for ax, metric in zip(axes, ERICA_METRICS):
        for method in K_BASED_METHODS:
            if method not in methods_in:
                continue
            vals = []
            for k in k_values:
                v = metrics[k].get(method, {}).get(metric, np.nan)
                try:
                    v = float(v)
                except (TypeError, ValueError):
                    v = np.nan
                vals.append(v)
            vals = np.array(vals)
            valid = ~np.isnan(vals)
            if valid.any():
                ax.plot(k_arr[valid], vals[valid],
                        color=METHOD_COLORS.get(method, "gray"),
                        marker="o", markersize=5, linewidth=1.8,
                        label=METHOD_LABELS.get(method, method), zorder=3)

        if hdb_valid and hdb_modal_k is not None:
            hdb_val = hdb_metric_vals.get(metric, np.nan)
            try:
                hdb_val = float(hdb_val)
            except (TypeError, ValueError):
                hdb_val = np.nan
            if not np.isnan(hdb_val):
                ax.plot(float(hdb_modal_k), hdb_val,
                        color=METHOD_COLORS["hdbscan"], marker="*",
                        markersize=12, linestyle="none",
                        label=f"HDBSCAN (K={hdb_modal_k})", zorder=5)

        ax.set_xlabel("K", fontsize=10)
        ax.set_ylabel(metric, fontsize=10)
        ax.set_title(metric, fontsize=11)
        ax.set_xticks(k_values)
        ax.set_ylim(-0.05, 1.05)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
        ax.legend(frameon=False, fontsize=7.5, loc="best")

    fig.suptitle(f"{dataset_label} - Method Comparison", fontsize=11, y=1.02)
    fig.tight_layout()
    return fig


# ============================================================================
# 5. K* bar chart
# ============================================================================

def plot_kstar_bars(er, dataset_label):
    """Grouped bar chart of K* per method per metric."""
    k_star = er.get("k_star", {})
    methods_in = get_methods_in_data(er)

    methods_present = [m for m in K_BASED_METHODS if m in methods_in]
    # Only include methods that have at least one K*
    methods_present = [m for m in methods_present
                       if any(k_star.get(met, {}).get(m) is not None
                              for met in ERICA_METRICS)]
    if not methods_present:
        return None

    n_metrics = len(ERICA_METRICS)
    n_methods = len(methods_present)
    bar_width = 0.25
    group_gap = 0.1
    group_width = n_methods * bar_width + group_gap
    group_centers = np.arange(n_metrics) * group_width
    offsets = np.linspace(-(n_methods - 1) * bar_width / 2,
                          (n_methods - 1) * bar_width / 2, n_methods)

    fig, ax = plt.subplots(figsize=(DOUBLE_COL * 0.8, 3.5))

    for j, method in enumerate(methods_present):
        kstar_vals = []
        for metric in ERICA_METRICS:
            kv = k_star.get(metric, {}).get(method, None)
            kstar_vals.append(kv if kv is not None else 0)
        x_pos = group_centers + offsets[j]
        bars = ax.bar(x_pos, kstar_vals, width=bar_width,
                      color=METHOD_COLORS.get(method, "gray"),
                      label=METHOD_LABELS.get(method, method),
                      edgecolor="white", linewidth=0.5, zorder=3)
        for bar, kv in zip(bars, kstar_vals):
            if kv > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.05, str(kv),
                        ha="center", va="bottom", fontsize=8, fontweight="bold",
                        color=METHOD_COLORS.get(method, "gray"))

    ax.set_xticks(group_centers)
    ax.set_xticklabels(ERICA_METRICS, fontsize=10)
    ax.set_ylabel("K*", fontsize=10)
    ax.set_xlabel("Metric", fontsize=10)

    all_kstar = [k_star.get(m, {}).get(mth, 0)
                 for m in ERICA_METRICS for mth in methods_present
                 if k_star.get(m, {}).get(mth) is not None]
    if all_kstar:
        ax.set_ylim(0, max(all_kstar) + 1.5)
        ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax.set_title(f"{dataset_label} - K* by Method and Metric", fontsize=11)
    fig.tight_layout()
    return fig


# ============================================================================
# 6. 3D CLAM surface
# ============================================================================

def plot_clam_surface(clam_matrix, title):
    """3D surface of a sorted CLAM matrix."""
    clam_sorted, _ = sort_clam(clam_matrix)
    n_samples, k = clam_sorted.shape

    cluster_idx = np.arange(k)
    sample_idx = np.arange(n_samples)
    X_mesh, Y_mesh = np.meshgrid(cluster_idx, sample_idx)

    fig = plt.figure(figsize=(DOUBLE_COL, 5))
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(X_mesh, Y_mesh, clam_sorted, cmap=CMAP_SEQ,
                           linewidth=0, antialiased=False,
                           rcount=min(200, n_samples), ccount=k)
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label="Co-assignment count")
    ax.set_xlabel("Cluster index", labelpad=8)
    ax.set_ylabel("Sample index (sorted)", labelpad=8)
    ax.set_zlabel("Count", labelpad=6)
    ax.set_title(title)
    ax.set_xticks(cluster_idx)
    ax.set_xticklabels([f"C{i+1}" for i in cluster_idx])
    fig.tight_layout()
    return fig


# ============================================================================
# 7. TWCRI landscape 2D (K x method heatmap)
# ============================================================================

def plot_twcri_landscape(er, dataset_label):
    """2D heatmap of TWCRI (methods on y-axis, K on x-axis)."""
    metrics = er["metrics"]
    k_values = sorted(metrics.keys())
    methods_in = get_methods_in_data(er)
    methods = [m for m in K_BASED_METHODS if m in methods_in]
    if not methods:
        return None

    n_methods = len(methods)
    n_k = len(k_values)
    twcri = np.full((n_methods, n_k), np.nan)

    for ki, k in enumerate(k_values):
        for mi, method in enumerate(methods):
            v = metrics[k].get(method, {}).get("TWCRI", np.nan)
            try:
                twcri[mi, ki] = float(v)
            except (TypeError, ValueError):
                twcri[mi, ki] = np.nan

    twcri_masked = np.ma.masked_invalid(twcri)
    cmap = plt.get_cmap(CMAP_SEQ).copy()
    cmap.set_bad(color="#cccccc")

    fig, ax = plt.subplots(figsize=(DOUBLE_COL * 0.8, max(2.0, 1.0 + n_methods * 0.7)))
    im = ax.imshow(twcri_masked, aspect="auto", cmap=cmap, vmin=0, vmax=1,
                   interpolation="nearest")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="TWCRI")

    for mi in range(n_methods):
        for ki in range(n_k):
            v = twcri[mi, ki]
            if np.isnan(v):
                label = "NaN"
                color = "#555555"
            else:
                label = f"{v:.2f}"
                color = "white" if v < 0.55 else "black"
            ax.text(ki, mi, label, ha="center", va="center",
                    fontsize=9, color=color)

    ax.set_xticks(range(n_k))
    ax.set_xticklabels([str(k) for k in k_values])
    ax.set_yticks(range(n_methods))
    ax.set_yticklabels([METHOD_LABELS.get(m, m) for m in methods])
    ax.set_xlabel("K")
    ax.set_ylabel("Method")
    ax.set_title(f"{dataset_label} - TWCRI Landscape")
    fig.tight_layout()
    return fig


# ============================================================================
# 8. Cluster scatter + confidence map (2D datasets only)
# ============================================================================

def get_primary_clusters(clam_matrix):
    primary = np.argmax(clam_matrix, axis=1)
    row_sums = clam_matrix.sum(axis=1).copy()
    row_sums[row_sums == 0] = 1
    confidence = clam_matrix[np.arange(len(clam_matrix)), primary] / row_sums
    return primary, confidence


def plot_cluster_scatter(X, clam_matrix, k, method, dataset_label):
    """Scatter plot colored by primary cluster, size by confidence."""
    primary, confidence = get_primary_clusters(clam_matrix)
    fig, ax = plt.subplots(figsize=(SINGLE_COL, SINGLE_COL))
    for c_idx in range(k):
        mask = primary == c_idx
        if not mask.any():
            continue
        color = CLUSTER_COLORS[c_idx % len(CLUSTER_COLORS)]
        sizes = 10 + 40 * confidence[mask]
        ax.scatter(X[mask, 0], X[mask, 1], s=sizes, c=color,
                   label=f"C{c_idx+1}", alpha=0.7, edgecolors="none")
    ax.set_title(f"{dataset_label} - {method}, K={k}")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.legend(fontsize=7, loc="best", markerscale=1.5)
    ax.set_aspect("equal")
    fig.tight_layout()
    return fig


def plot_confidence_map(X, clam_matrix, k, method, dataset_label):
    """Scatter where color = assignment confidence."""
    _, confidence = get_primary_clusters(clam_matrix)
    fig, ax = plt.subplots(figsize=(SINGLE_COL, SINGLE_COL))
    sc = ax.scatter(X[:, 0], X[:, 1], s=15, c=confidence, cmap="RdYlGn",
                    vmin=0.3, vmax=1.0, edgecolors="none")
    plt.colorbar(sc, ax=ax, label="Confidence")
    ax.set_title(f"{dataset_label} - {method}, K={k}\nConfidence Map")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_aspect("equal")
    fig.tight_layout()
    return fig


# ============================================================================
# Main driver
# ============================================================================

def process_dataset(result_path):
    """Generate all plots for a single dataset result file.

    Returns the number of figures generated.
    """
    name = dataset_name_from_path(result_path)
    label = name.replace("_", " ").title()
    out_dir = os.path.join(FIGURES_ROOT, name)
    os.makedirs(out_dir, exist_ok=True)

    # joblib loads locally-generated ERICA results (not untrusted data)
    data = joblib.load(result_path)
    er = data["erica_results"]
    config = data.get("config", {})
    true_k = config.get("true_k", 3)
    n_features = config.get("n_features", None)
    methods_in = get_methods_in_data(er)
    clam = er["clam_matrices"]
    k_values = sorted(er["metrics"].keys())

    # Choose the K to use for single-K plots
    focus_k = true_k if true_k and true_k in k_values else (3 if 3 in k_values else k_values[0])
    count = 0

    # ---- 1. Sorted CLAM heatmap (kmeans at focus_k) ----
    clam_key = (focus_k, "kmeans")
    if clam_key in clam:
        fig = plot_clam_heatmap(clam[clam_key],
                                f"CLAM (sorted) - {label}, K={focus_k}, K-Means")
        save_fig(fig, out_dir, f"clam_sorted_kmeans_k{focus_k}")
        count += 1

    # ---- 2. Stability strips (kmeans at focus_k) ----
    if clam_key in clam:
        fig = plot_stability_strips(clam[clam_key],
                                    f"Stability - {label}, K={focus_k}, K-Means")
        save_fig(fig, out_dir, f"stability_kmeans_k{focus_k}")
        count += 1

    # ---- 3. Metrics curves (for each method) ----
    for method in K_BASED_METHODS:
        if method not in methods_in:
            continue
        fig = plot_metrics_curves(er, label, method)
        save_fig(fig, out_dir, f"metrics_{method}")
        count += 1

    # ---- 4. Method comparison (3-panel) ----
    fig = plot_method_comparison(er, label)
    save_fig(fig, out_dir, "method_comparison")
    count += 1

    # ---- 5. K* bar chart ----
    fig = plot_kstar_bars(er, label)
    if fig is not None:
        save_fig(fig, out_dir, "kstar_bars")
        count += 1

    # ---- 6. 3D CLAM surface (kmeans at focus_k) ----
    if clam_key in clam:
        fig = plot_clam_surface(clam[clam_key],
                                f"CLAM Surface - {label}, K={focus_k}, K-Means")
        save_fig(fig, out_dir, f"surface_clam_kmeans_k{focus_k}")
        count += 1

    # ---- 7. TWCRI landscape 2D ----
    fig = plot_twcri_landscape(er, label)
    if fig is not None:
        save_fig(fig, out_dir, "twcri_landscape")
        count += 1

    # ---- 8. 2D scatter + confidence (only if n_features == 2) ----
    if n_features == 2:
        data_path = find_data_file(name)
        if data_path is not None:
            try:
                X = np.load(data_path, allow_pickle=True)["X"]
                if X.shape[1] == 2:
                    for method in ["kmeans", "agglomerative_ward"]:
                        ck = (focus_k, method)
                        if ck in clam:
                            fig = plot_cluster_scatter(
                                X, clam[ck], focus_k, method, label)
                            save_fig(fig, out_dir,
                                     f"scatter_{method}_k{focus_k}")
                            count += 1

                            fig = plot_confidence_map(
                                X, clam[ck], focus_k, method, label)
                            save_fig(fig, out_dir,
                                     f"confidence_{method}_k{focus_k}")
                            count += 1
            except Exception as exc:
                print(f"    Warning: could not load data for scatter: {exc}")

    # ---- HDBSCAN CLAM (experimental, if available) ----
    hdb = er.get("auto_k", {}).get("hdbscan", {})
    if (isinstance(hdb, dict)
            and hdb.get("k_agreement_rate", 0) > 0
            and "clam_matrix" in hdb):
        modal_k = hdb.get("modal_k", "?")
        fig = plot_clam_heatmap(
            hdb["clam_matrix"],
            f"CLAM (sorted) - {label}, HDBSCAN (modal K={modal_k}) [experimental]")
        save_fig(fig, out_dir, f"clam_sorted_hdbscan_modal_k{modal_k}")
        count += 1

    return count


def main():
    set_publication_style()

    # Discover all result files
    pattern = os.path.join(RESULTS_DIR, "*.joblib")
    all_files = sorted(glob.glob(pattern))

    # Filter out sweep/summary/workdir files
    result_files = []
    for f in all_files:
        basename = os.path.basename(f).lower()
        if any(skip in basename for skip in SKIP_PATTERNS):
            print(f"  Skipping: {os.path.basename(f)}")
            continue
        result_files.append(f)

    print(f"Found {len(result_files)} dataset result files.\n")

    total_figures = 0
    for rf in result_files:
        name = dataset_name_from_path(rf)
        print(f"Processing {name} ...")
        try:
            n = process_dataset(rf)
            print(f"  -> {n} figures saved to figures/by_dataset/{name}/")
            total_figures += n
        except Exception as exc:
            print(f"  ERROR processing {name}: {exc}")
            import traceback
            traceback.print_exc()

    print(f"\nDone. Generated {total_figures} figures across {len(result_files)} datasets.")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
