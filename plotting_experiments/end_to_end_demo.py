"""End-to-end ERICA demo on the VDX 3-gene dataset.

Runs the full pipeline using only the public `erica` library API and produces
every library-supplied plot type. Intended as the simplest possible
"this is what ERICA does" walkthrough for new users.

Inputs : examples/data/VDX_3_SV.csv (344 samples x 3 genes)
Outputs: figures/end_to_end_demo/*.png

Run from the repo root:
    python plotting_experiments/end_to_end_demo.py

First written: 2026-04-26
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

from erica import (
    ERICA,
    set_publication_style,
    plot_replicability_metrics,
    plot_clam_heatmap_mpl,
    plot_cluster_sizes_mpl,
    plot_icah,
    plot_pcsp,
    plot_k_star_bars,
    plot_stability_strips,
    extract_metric_curves,
    plot_metrics_vs_k,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(REPO_ROOT, "examples", "data", "VDX_3_SV.csv")
OUT_DIR = os.path.join(REPO_ROOT, "plotting_experiments", "figures", "end_to_end_demo")


def save(fig, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"{name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {path}")


def main():
    set_publication_style()

    print(f"Loading {CSV_PATH}")
    X = pd.read_csv(CSV_PATH, header=None).to_numpy()
    print(f"  shape: {X.shape}  (n_samples, n_features)")

    print("\nRunning ERICA (kmeans + ward, K=2..6, 50 iterations) ...")
    er = ERICA(
        data=X,
        k_range=[2, 3, 4, 5, 6],
        n_iterations=50,
        method=["kmeans", "agglomerative_ward"],
        transpose=False,
        random_state=42,
        verbose=False,
    )
    results = er.run()

    k_star_twcri = er.get_k_star("TWCRI")
    print(f"  K* (TWCRI): {k_star_twcri}")

    method = "kmeans"
    k_star = k_star_twcri.get(method) or 3
    K_RANGE = [2, 3, 4, 5, 6]

    print(f"\nGenerating aggregate figures over K={K_RANGE} ...")

    # --- 01-02: aggregate across K ---
    k_values, curves, _ = extract_metric_curves(
        results, method, ["CRI", "WCRI", "TWCRI"]
    )
    fig, ax = plot_replicability_metrics(
        k_values, curves["CRI"], curves["WCRI"], curves["TWCRI"]
    )
    ax.axvline(k_star, color="gray", linestyle="--", linewidth=0.8,
               label=f"K*={k_star}")
    ax.legend()
    ax.set_title(f"VDX 3-gene — {method} replicability metrics")
    save(fig, "01_replicability_metrics")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, m in zip(axes, ["kmeans", "agglomerative_ward"]):
        kvals, c, _ = extract_metric_curves(results, m, ["CRI", "WCRI", "TWCRI"])
        plot_metrics_vs_k(kvals, c, ax=ax)
        ax.set_title(m)
    fig.suptitle("Method comparison — CRI / WCRI / TWCRI vs K", fontsize=12)
    fig.tight_layout()
    save(fig, "02_method_comparison")

    k_star_all = {m: er.get_k_star(m) for m in ["CRI", "WCRI", "TWCRI"]}
    fig, _ = plot_k_star_bars(k_star_all)
    save(fig, "03_k_star_bars")

    # --- 04-08: per-K, one figure per K with " (K*)" annotation when matched ---
    for k in K_RANGE:
        clam = results["clam_matrices"].get((k, method))
        if clam is None:
            print(f"  [SKIP] no CLAM at K={k}")
            continue
        suffix = " — K*" if k == k_star else ""
        print(f"\n  K={k}{suffix}")

        fig, _ = plot_clam_heatmap_mpl(
            clam, sort=True,
            title=f"CLAM heatmap (sorted) — K={k}{suffix}",
        )
        save(fig, f"04_clam_heatmap_sorted_K{k}")

        fig, _ = plot_cluster_sizes_mpl(clam)
        save(fig, f"05_cluster_sizes_K{k}")

        fig, _ = plot_icah(clam, k=k,
                           title=f"Inter-Cluster Assignment Heatmap — K={k}{suffix}")
        save(fig, f"06_icah_K{k}")

        fig = plot_pcsp(clam, k=k,
                        title=f"Per-Cluster Scatter Plots — K={k}{suffix}")
        save(fig, f"07_pcsp_K{k}")

        fig, _ = plot_stability_strips(
            clam, max_samples=80,
            title=f"Stability strips — K={k}{suffix}",
        )
        save(fig, f"08_stability_strips_K{k}")

    print(f"\nDone. Figures in {OUT_DIR}")


if __name__ == "__main__":
    main()
