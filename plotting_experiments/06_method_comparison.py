"""
06_method_comparison.py

Two figures per dataset:
  1. Main figure — 1x3 subplots (CRI, WCRI, TWCRI)
       - K-based methods (kmeans, agglomerative_ward) as lines
       - HDBSCAN as a single star marker at its modal K (when available)
  2. Secondary figure — grouped bar chart of K* per method per metric
       - Only K-based methods (no HDBSCAN K*)
       - Bar tops annotated with the K* value

Generated for: vdx_3gene, well_separated, overlapping
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import set_publication_style, save_figure, DOUBLE_COL, METHOD_COLORS, METRIC_COLORS

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")

DATASETS = ["vdx_3gene", "well_separated", "overlapping"]
K_BASED_METHODS = ["kmeans", "agglomerative_ward"]
METRICS = ["CRI", "WCRI", "TWCRI"]

METHOD_LABELS = {
    "kmeans": "K-Means",
    "agglomerative_ward": "Agglo. Ward",
    "hdbscan": "HDBSCAN",
}


# ---------------------------------------------------------------------------
# Figure 1: Method comparison — 1x3 metric subplots
# ---------------------------------------------------------------------------

def plot_method_comparison(er, dataset_name):
    """1x3 subplot: each panel shows one metric across K for each method.

    K-based methods are drawn as continuous lines; HDBSCAN (if valid) is
    plotted as a star marker at its modal K.

    Parameters
    ----------
    er : dict
        erica_results dict (from .joblib file).
    dataset_name : str
        Human-readable label for the figure suptitle.

    Returns
    -------
    fig : matplotlib Figure
    """
    metrics = er["metrics"]
    k_values = sorted(metrics.keys())
    k_arr = np.array(k_values, dtype=float)

    # --- HDBSCAN data ---
    hdb = er.get("auto_k", {}).get("hdbscan", {})
    hdb_valid = (
        isinstance(hdb, dict)
        and hdb.get("k_agreement_rate", 0) > 0
        and "metrics_at_modal_k" in hdb
    )
    hdb_modal_k = hdb.get("modal_k") if hdb_valid else None
    hdb_metric_vals = hdb.get("metrics_at_modal_k", {}) if hdb_valid else {}

    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL * 1.3, 3.6), sharey=False)

    for ax, metric in zip(axes, METRICS):
        # --- K-based method lines ---
        for method in K_BASED_METHODS:
            vals = []
            for k in k_values:
                m_dict = metrics[k].get(method, {})
                v = m_dict.get(metric, np.nan)
                try:
                    v = float(v)
                except (TypeError, ValueError):
                    v = np.nan
                vals.append(v)
            vals = np.array(vals)

            valid_mask = ~np.isnan(vals)
            if valid_mask.any():
                ax.plot(
                    k_arr[valid_mask],
                    vals[valid_mask],
                    color=METHOD_COLORS[method],
                    marker="o",
                    markersize=5,
                    linewidth=1.8,
                    label=METHOD_LABELS[method],
                    zorder=3,
                )

        # --- HDBSCAN star marker ---
        if hdb_valid and hdb_modal_k is not None:
            hdb_val = hdb_metric_vals.get(metric, np.nan)
            try:
                hdb_val = float(hdb_val)
            except (TypeError, ValueError):
                hdb_val = np.nan

            if not np.isnan(hdb_val):
                ax.plot(
                    float(hdb_modal_k),
                    hdb_val,
                    color=METHOD_COLORS["hdbscan"],
                    marker="*",
                    markersize=12,
                    linestyle="none",
                    label=f"HDBSCAN (K={hdb_modal_k})",
                    zorder=5,
                )

        ax.set_xlabel("K", fontsize=10)
        ax.set_ylabel(metric, fontsize=10)
        ax.set_title(metric, fontsize=11)
        ax.set_xticks(k_values)
        ax.set_ylim(-0.05, 1.05)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
        ax.legend(frameon=False, fontsize=7.5, loc="best")

    dataset_label = dataset_name.replace("_", " ").title()
    fig.suptitle(f"{dataset_label} — Method Comparison", fontsize=11, y=1.02)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 2: K* grouped bar chart
# ---------------------------------------------------------------------------

def plot_kstar_bars(er, dataset_name):
    """Grouped bar chart of K* per method (K-based only) per metric.

    Parameters
    ----------
    er : dict
        erica_results dict.
    dataset_name : str
        Human-readable label for the figure suptitle.

    Returns
    -------
    fig : matplotlib Figure
    """
    k_star = er.get("k_star", {})

    # Build data: metrics × methods
    methods_present = []
    for method in K_BASED_METHODS:
        # Check if method appears in any metric's k_star
        for metric in METRICS:
            if metric in k_star and method in k_star[metric]:
                if method not in methods_present:
                    methods_present.append(method)
                break

    if not methods_present:
        return None

    n_metrics = len(METRICS)
    n_methods = len(methods_present)

    bar_width = 0.25
    group_gap = 0.1  # extra space between metric groups
    group_width = n_methods * bar_width + group_gap

    # x positions for each metric group center
    group_centers = np.arange(n_metrics) * group_width
    # offsets within each group so bars are centered
    offsets = np.linspace(
        -(n_methods - 1) * bar_width / 2,
        (n_methods - 1) * bar_width / 2,
        n_methods,
    )

    fig, ax = plt.subplots(figsize=(DOUBLE_COL * 0.8, 3.5))

    for j, method in enumerate(methods_present):
        kstar_vals = []
        for metric in METRICS:
            kv = k_star.get(metric, {}).get(method, None)
            kstar_vals.append(kv if kv is not None else 0)

        x_positions = group_centers + offsets[j]
        bars = ax.bar(
            x_positions,
            kstar_vals,
            width=bar_width,
            color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )

        # Annotate bar tops with K* value
        for bar, kv in zip(bars, kstar_vals):
            if kv > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.05,
                    str(kv),
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    fontweight="bold",
                    color=METHOD_COLORS[method],
                )

    ax.set_xticks(group_centers)
    ax.set_xticklabels(METRICS, fontsize=10)
    ax.set_ylabel("K*", fontsize=10)
    ax.set_xlabel("Metric", fontsize=10)

    # Set y-axis to integer ticks
    all_kstar = [
        k_star.get(m, {}).get(mth, 0)
        for m in METRICS
        for mth in methods_present
        if k_star.get(m, {}).get(mth) is not None
    ]
    if all_kstar:
        ymax = max(all_kstar) + 1.5
        ax.set_ylim(0, ymax)
        ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)

    dataset_label = dataset_name.replace("_", " ").title()
    ax.set_title(f"{dataset_label} — K* by Method and Metric", fontsize=11)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    set_publication_style()

    generated = []

    for name in DATASETS:
        result_path = os.path.join(RESULTS_DIR, f"{name}.joblib")
        if not os.path.exists(result_path):
            print(f"  Skipping {name}: results file not found at {result_path}")
            continue

        data = joblib.load(result_path)
        er = data["erica_results"]
        print(f"\nLoaded {name}  (K values: {sorted(er['metrics'].keys())})")

        # Figure 1: method comparison
        print(f"  Generating method_{name} ...")
        fig1 = plot_method_comparison(er, name)
        fname1 = f"method_{name}"
        save_figure(fig1, fname1)
        generated.append(fname1)

        # Figure 2: K* grouped bar chart
        print(f"  Generating kstar_{name} ...")
        fig2 = plot_kstar_bars(er, name)
        if fig2 is not None:
            fname2 = f"kstar_{name}"
            save_figure(fig2, fname2)
            generated.append(fname2)
        else:
            print(f"    No K* data found for {name}, skipping bar chart.")

    print(f"\nDone. Generated {len(generated)} figure(s):")
    for f in generated:
        print(f"  {f}")


if __name__ == "__main__":
    main()
