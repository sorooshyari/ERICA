"""
01_error_bands.py — Literature Comparison: Error Bands on Metric Curves

Adds confidence bands (±1 std) to the metric-vs-K curves, following the style
of Parmigiani et al. (2023) and the Gap Statistic paper (Tibshirani et al. 2001).

Two-panel layout per figure:
  Left  — ERICA metrics (CRI, WCRI, TWCRI) with shaded ±1 std bands.
           Band width is the std of per-sample assignment consistency
           (max proportion assigned to modal cluster) across all samples.
           All three metrics share the same band width because the
           underlying per-sample distribution is identical; only the
           aggregate (mean) differs between CRI, WCRI, and TWCRI.
  Right — Parmigiani metrics (ARI, AMI) with shaded ±1 std bands computed
           from per-iteration ARI/AMI values (iteration-level variance).

K* is marked with a vertical dashed line per metric.

Outputs: ../figures/literature_comparison/error_bands_{dataset}_{method}.{pdf,png}
"""

import sys
import os

# Resolve style from parent directory
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from style import set_publication_style, save_figure, DOUBLE_COL, METRIC_COLORS

import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "results")
FIGURES_DIR = os.path.join(SCRIPT_DIR, "..", "figures", "literature_comparison")

DATASETS = [
    "vdx_3gene",
    "gauss4c_sigma0p01",
    "gauss4c_sigma0p1",
    "gauss4c_sigma1p0",
    "gauss4c_sigma10p0",
    "moons_2d",
    "blobs_2d",
    "well_separated",
]

METHODS = ["kmeans", "agglomerative_ward"]

ERICA_METRICS = ["CRI", "WCRI", "TWCRI"]
PARMI_METRICS = ["ARI", "AMI"]

# Dash patterns for ERICA metric lines
METRIC_DASHES = {
    "CRI": (4, 2),
    "WCRI": (1, 1),
    "TWCRI": (8, 2),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def per_sample_cri_from_clam(clam_matrix):
    """Compute per-sample CRI from a CLAM matrix.

    CLAM is (n_samples, k) where entry [i, j] = number of iterations
    that assigned sample i to cluster j (among iterations where sample i
    appeared in the test set).

    Per-sample CRI = max fraction of iterations assigning the sample to
    its modal cluster.  This is also TWCRI's building block.

    Returns
    -------
    per_sample_cri : ndarray, shape (n_samples,)
    """
    row_sums = clam_matrix.sum(axis=1, keepdims=True)
    # Avoid division by zero for samples never in test set
    safe_sums = np.where(row_sums == 0, 1, row_sums)
    clam_norm = clam_matrix / safe_sums
    return clam_norm.max(axis=1)


def per_sample_wcri_from_clam(clam_matrix):
    """Compute per-sample WCRI proxy from CLAM.

    WCRI is the within-cluster average of per-sample CRI, weighted by cluster
    size.  For the band we use the full per-sample distribution (no grouping),
    so the std reflects variability across all samples.

    Returns
    -------
    per_sample_wcri : ndarray, shape (n_samples,)
        Per-sample assignment consistency (same as per-sample CRI numerically).
    """
    return per_sample_cri_from_clam(clam_matrix)


def compute_per_iteration_ari_ami(iteration_labels):
    """Compute ARI and AMI for each iteration from stored predicted/true arrays.

    Parameters
    ----------
    iteration_labels : dict with keys 'predicted' and 'true'
        Each is a list of length n_iterations with ndarray label vectors.

    Returns
    -------
    ari_vals : ndarray, shape (n_iterations,)
    ami_vals : ndarray, shape (n_iterations,)
    """
    predicted = iteration_labels["predicted"]
    true_labels = iteration_labels["true"]
    n = len(predicted)

    ari_vals = np.empty(n)
    ami_vals = np.empty(n)
    for i in range(n):
        ari_vals[i] = adjusted_rand_score(true_labels[i], predicted[i])
        ami_vals[i] = adjusted_mutual_info_score(true_labels[i], predicted[i])
    return ari_vals, ami_vals


# ---------------------------------------------------------------------------
# Core plotting
# ---------------------------------------------------------------------------

def plot_error_bands(er, dataset_name, method):
    """Generate a two-panel figure with ±1 std shaded bands.

    Parameters
    ----------
    er : dict
        erica_results dict (value of data['erica_results'] from the joblib).
    dataset_name : str
        Human-readable dataset label for the figure title.
    method : str
        Clustering method key ('kmeans' or 'agglomerative_ward').

    Returns
    -------
    fig : matplotlib Figure
    """
    metrics = er["metrics"]
    k_star = er["k_star"]
    results = er["results"]

    k_values = sorted(metrics.keys())
    k_arr = np.array(k_values, dtype=float)
    n_k = len(k_values)

    # ------------------------------------------------------------------
    # Collect ERICA band data (mean ± std over samples from CLAM)
    #
    # NOTE: The band width (std) is derived from the per-sample
    # assignment consistency distribution, which is the same for all
    # three metrics (CRI, WCRI, TWCRI).  The MEAN values differ because
    # each metric aggregates differently, but the underlying per-sample
    # distribution is identical.  This is intentional: the band shows
    # cross-sample variability in assignment stability.
    # ------------------------------------------------------------------
    erica_mean = {m: np.full(n_k, np.nan) for m in ERICA_METRICS}
    erica_std = {m: np.full(n_k, np.nan) for m in ERICA_METRICS}

    for i, k in enumerate(k_values):
        m_dict = metrics[k].get(method, {})
        if not m_dict or m_dict.get("has_empty_clusters", False):
            continue

        # Retrieve CLAM for this (k, method)
        clam = results.get((k, method), {}).get("clam_matrix")
        if clam is None:
            continue

        per_sample = per_sample_cri_from_clam(clam)

        # CRI: mean/std of per-sample assignment consistency
        erica_mean["CRI"][i] = float(m_dict.get("CRI", np.nan))
        erica_std["CRI"][i] = float(np.std(per_sample))

        # WCRI: stored aggregate; std from per-sample distribution
        wcri_val = m_dict.get("WCRI", np.nan)
        try:
            wcri_val = float(wcri_val)
        except (TypeError, ValueError):
            wcri_val = np.nan
        erica_mean["WCRI"][i] = wcri_val
        erica_std["WCRI"][i] = float(np.std(per_sample))

        # TWCRI: stored aggregate; std from per-sample distribution
        twcri_val = m_dict.get("TWCRI", np.nan)
        try:
            twcri_val = float(twcri_val)
        except (TypeError, ValueError):
            twcri_val = np.nan
        erica_mean["TWCRI"][i] = twcri_val
        erica_std["TWCRI"][i] = float(np.std(per_sample))

    # ------------------------------------------------------------------
    # Collect Parmigiani band data (mean ± std per iteration from labels)
    # ------------------------------------------------------------------
    ari_mean = np.full(n_k, np.nan)
    ari_std = np.full(n_k, np.nan)
    ami_mean = np.full(n_k, np.nan)
    ami_std = np.full(n_k, np.nan)

    for i, k in enumerate(k_values):
        res_entry = results.get((k, method), {})
        il = res_entry.get("iteration_labels")
        if il is None:
            # Fall back to stored aggregates
            m_dict = metrics[k].get(method, {})
            ari_mean[i] = float(m_dict.get("ARI_mean", np.nan))
            ari_std[i] = float(m_dict.get("ARI_std", np.nan))
            ami_mean[i] = float(m_dict.get("AMI_mean", np.nan))
            ami_std[i] = float(m_dict.get("AMI_std", np.nan))
        else:
            ari_vals, ami_vals = compute_per_iteration_ari_ami(il)
            ari_mean[i] = float(np.mean(ari_vals))
            ari_std[i] = float(np.std(ari_vals))
            ami_mean[i] = float(np.mean(ami_vals))
            ami_std[i] = float(np.std(ami_vals))

    # ------------------------------------------------------------------
    # Build figure
    # ------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DOUBLE_COL, 3.5))

    # ---- Panel 1: ERICA metrics with bands ----------------------------
    for metric in ERICA_METRICS:
        color = METRIC_COLORS[metric]
        dashes = METRIC_DASHES[metric]
        mean_vals = erica_mean[metric]
        std_vals = erica_std[metric]

        valid = ~np.isnan(mean_vals)
        if not valid.any():
            continue

        kv = k_arr[valid]
        mv = mean_vals[valid]
        sv = std_vals[valid]

        # Shaded band
        ax1.fill_between(
            kv,
            np.clip(mv - sv, 0, 1),
            np.clip(mv + sv, 0, 1),
            color=color,
            alpha=0.15,
            linewidth=0,
        )
        # Line + markers
        ax1.plot(
            kv,
            mv,
            color=color,
            dashes=dashes,
            marker="o",
            markersize=5,
            label=metric,
            zorder=3,
        )

        # Vertical K* line
        k_star_val = k_star.get(metric, {}).get(method)
        if k_star_val is not None and k_star_val in k_values:
            ax1.axvline(
                x=k_star_val,
                color=color,
                linestyle="--",
                linewidth=1.0,
                alpha=0.75,
                zorder=2,
            )

    ax1.set_xlabel("K")
    ax1.set_ylabel("Metric value")
    ax1.set_title("ERICA metrics (\u00b11 std)")
    ax1.set_xticks(k_values)
    ax1.set_ylim(-0.05, 1.05)
    ax1.legend(frameon=False, fontsize=8)

    # ---- Panel 2: Parmigiani metrics with bands ----------------------
    for metric_name, mean_arr, std_arr, marker in [
        ("ARI", ari_mean, ari_std, "o"),
        ("AMI", ami_mean, ami_std, "s"),
    ]:
        color = METRIC_COLORS[metric_name]
        valid = ~np.isnan(mean_arr)
        if not valid.any():
            continue

        kv = k_arr[valid]
        mv = mean_arr[valid]
        sv = std_arr[valid]

        ax2.fill_between(
            kv,
            mv - sv,
            mv + sv,
            color=color,
            alpha=0.15,
            linewidth=0,
        )
        ax2.plot(
            kv,
            mv,
            color=color,
            marker=marker,
            markersize=5,
            label=metric_name,
            zorder=3,
        )

    ax2.set_xlabel("K")
    ax2.set_ylabel("Metric value")
    ax2.set_title("Parmigiani metrics (\u00b11 std)")
    ax2.set_xticks(k_values)
    ax2.legend(frameon=False, fontsize=8)

    # ------------------------------------------------------------------
    # Title and layout
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
    os.makedirs(FIGURES_DIR, exist_ok=True)

    generated = []
    skipped = []

    for name in DATASETS:
        result_path = os.path.join(RESULTS_DIR, f"{name}.joblib")
        if not os.path.exists(result_path):
            print(f"  Skipping {name}: results file not found.")
            skipped.append(name)
            continue

        data = joblib.load(result_path)
        er = data["erica_results"]

        available_methods = set()
        for k in er["metrics"]:
            available_methods.update(er["metrics"][k].keys())

        print(f"\nLoaded '{name}'  (K values: {sorted(er['metrics'].keys())})")

        for method in METHODS:
            if method not in available_methods:
                print(f"  Skipping method '{method}' (not present in results).")
                continue

            dataset_label = name.replace("_", " ")
            print(f"  Generating error_bands_{name}_{method} ...")
            fig = plot_error_bands(er, dataset_label, method)

            # Save directly into the literature_comparison subfolder
            for fmt in ("pdf", "png"):
                out_path = os.path.join(
                    FIGURES_DIR, f"error_bands_{name}_{method}.{fmt}"
                )
                fig.savefig(out_path, format=fmt)
                print(f"    Saved: {out_path}")
            plt.close(fig)
            generated.append(f"error_bands_{name}_{method}")

    print(f"\nDone. Generated {len(generated)} figure(s), skipped {len(skipped)} dataset(s).")
    for f in generated:
        print(f"  {f}")


if __name__ == "__main__":
    main()
