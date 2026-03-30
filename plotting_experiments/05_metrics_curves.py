"""
05_metrics_curves.py

Two-panel figure showing:
  Panel 1 — ERICA metrics (CRI, WCRI, TWCRI) vs K as lines+markers,
             with vertical K* lines and 'x' markers for disqualified K values.
  Panel 2 — Parmigiani metrics (ARI, AMI) with error bars (mean ± std).

Generated for each combination of dataset × clustering method.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import (
    set_publication_style,
    save_figure,
    DOUBLE_COL,
    METRIC_COLORS,
    METRIC_DASHES,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")

# Datasets and methods to generate figures for
DATASETS = ["vdx_3gene", "well_separated", "overlapping"]
METHODS = ["kmeans", "agglomerative_ward"]

ERICA_METRICS = ["CRI", "WCRI", "TWCRI"]
PARMI_METRICS = ["ARI", "AMI"]


# ---------------------------------------------------------------------------
# Core plotting
# ---------------------------------------------------------------------------

def plot_metrics_curves(er, dataset_name, method):
    """Create two-panel metrics curves figure for a dataset/method combination.

    Parameters
    ----------
    er : dict
        erica_results dict loaded from the .joblib file.
    dataset_name : str
        Human-readable dataset name used in the suptitle.
    method : str
        Clustering method key (e.g. 'kmeans', 'agglomerative_ward').

    Returns
    -------
    fig : matplotlib Figure
    """
    metrics = er["metrics"]
    k_star = er["k_star"]

    k_values = sorted(metrics.keys())

    # Build arrays for each ERICA metric and Parmigiani metric
    # shape: (len(k_values),)
    erica_vals = {m: [] for m in ERICA_METRICS}
    ari_means, ari_stds = [], []
    ami_means, ami_stds = [], []

    for k in k_values:
        m_dict = metrics[k].get(method, {})
        for metric in ERICA_METRICS:
            val = m_dict.get(metric, np.nan)
            # Ensure we treat non-numeric gracefully
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

    # Identify disqualified K indices (NaN in any ERICA metric for this method)
    disq_mask = np.array([
        any(np.isnan(erica_vals[m][i]) for m in ERICA_METRICS)
        for i in range(len(k_values))
    ])

    # ------------------------------------------------------------------
    # Create figure
    # ------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DOUBLE_COL, 3.5))

    # ---- Panel 1: ERICA metrics ----------------------------------------
    for metric in ERICA_METRICS:
        vals = np.array(erica_vals[metric])
        color = METRIC_COLORS[metric]
        dashes = METRIC_DASHES[metric]

        # Plot only valid (non-NaN) points connected by a line
        valid_mask = ~np.isnan(vals)
        if valid_mask.any():
            ax1.plot(
                k_arr[valid_mask],
                vals[valid_mask],
                color=color,
                dashes=dashes,
                marker="o",
                label=metric,
                zorder=3,
            )

        # Mark disqualified K values with 'x' at y=0
        disq_valid = disq_mask & ~np.isnan(vals)
        if disq_valid.any():
            ax1.plot(
                k_arr[disq_valid],
                np.zeros(disq_valid.sum()),
                marker="x",
                color=color,
                linestyle="none",
                markersize=8,
                markeredgewidth=2,
                zorder=4,
            )

        # Also mark NaN positions (disqualified) where value IS NaN
        disq_nan = disq_mask & np.isnan(vals)
        if disq_nan.any():
            ax1.plot(
                k_arr[disq_nan],
                np.zeros(disq_nan.sum()),
                marker="x",
                color=color,
                linestyle="none",
                markersize=8,
                markeredgewidth=2,
                zorder=4,
            )

        # Vertical K* line
        k_star_val = k_star.get(metric, {}).get(method)
        if k_star_val is not None:
            ax1.axvline(
                x=k_star_val,
                color=color,
                dashes=dashes,
                linewidth=1.2,
                alpha=0.7,
                zorder=2,
            )

    ax1.set_xlabel("K")
    ax1.set_ylabel("Metric value")
    ax1.set_title("ERICA metrics")
    ax1.set_xticks(k_values)
    ax1.set_ylim(-0.05, 1.05)
    ax1.legend(frameon=False, fontsize=8)

    # ---- Panel 2: Parmigiani metrics -----------------------------------
    ax2.errorbar(
        k_arr,
        ari_means,
        yerr=ari_stds,
        color=METRIC_COLORS["ARI"],
        marker="o",
        capsize=3,
        label="ARI",
        zorder=3,
    )
    ax2.errorbar(
        k_arr,
        ami_means,
        yerr=ami_stds,
        color=METRIC_COLORS["AMI"],
        marker="s",
        capsize=3,
        label="AMI",
        zorder=3,
    )

    ax2.set_xlabel("K")
    ax2.set_ylabel("Metric value")
    ax2.set_title("Parmigiani metrics")
    ax2.set_xticks(k_values)
    ax2.legend(frameon=False, fontsize=8)

    # ------------------------------------------------------------------
    # Suptitle
    # ------------------------------------------------------------------
    method_label = method.replace("_", " ").title()
    fig.suptitle(f"{dataset_name} — {method_label}", fontsize=11, y=1.01)
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
        methods_in_data = set()
        for k in er["metrics"]:
            methods_in_data.update(er["metrics"][k].keys())

        print(f"\nLoaded {name}  (K values: {sorted(er['metrics'].keys())})")

        for method in METHODS:
            if method not in methods_in_data:
                print(f"  Skipping method '{method}' (not present in results).")
                continue

            dataset_label = name.replace("_", " ")
            print(f"  Generating metrics_{name}_{method} ...")
            fig = plot_metrics_curves(er, dataset_label, method)
            fname = f"metrics_{name}_{method}"
            save_figure(fig, fname)
            generated.append(fname)

    print(f"\nDone. Generated {len(generated)} figure(s):")
    for f in generated:
        print(f"  {f}")


if __name__ == "__main__":
    main()
