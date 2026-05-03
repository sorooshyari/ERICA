"""End-to-end test of the ERICA library at K=4 on VDX 3-gene and VDX full.

Two datasets, K=4 fixed, both clustering methods. MATLAB-recap figures
(plot_assignment_scatter, plot_assignment_heatmap) are generated first
and prioritised by file ordering, followed by the enriched library
figures (PCSP, ICAH, CLAM heatmap, cluster sizes, stability strips).

Inputs
------
* VDX 3-gene : examples/data/VDX_3_SV.csv (344 samples x 3 genes)
* VDX full   : plotting_experiments/data/vdx_full.npz (344 x 22283)

The full VDX `.npz` is produced from the upstream `vdx_dict.npy` (Parmigiani
et al. clustering replicability repo) via a one-shot conversion. Both raw
files are gitignored. Re-create the `.npz` with the snippet at
``plotting_experiments/README_vdx_full.md`` (download with curl, then
convert the dict-of-DataFrames `.npy` to a DataFrame-free `.npz`).

Outputs (committed)
-------------------
plotting_experiments/figures/e2e_test/<dataset>/*.png
plotting_experiments/figures/e2e_test/<dataset>/clams/clam_<method>_K4.npz

Run from the repo root::

    python plotting_experiments/e2e_test.py

First written: 2026-05-02
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from erica import (
    ERICA,
    set_publication_style,
    plot_assignment_scatter,
    plot_assignment_heatmap,
    plot_clam_heatmap_mpl,
    plot_cluster_sizes_mpl,
    plot_icah,
    plot_pcsp,
    plot_stability_strips,
)


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "plotting_experiments", "data")
OUT_ROOT = os.path.join(REPO_ROOT, "plotting_experiments", "figures", "e2e_test")

VDX_3GENE_CSV = os.path.join(REPO_ROOT, "examples", "data", "VDX_3_SV.csv")
VDX_FULL_NPZ = os.path.join(DATA_DIR, "vdx_full.npz")

K = 4
N_ITERATIONS_3GENE = 50
N_ITERATIONS_FULL = 30   # 22k features -> keep iterations modest
METHODS = ["kmeans", "agglomerative_ward"]


def load_vdx_3gene():
    X = pd.read_csv(VDX_3GENE_CSV, header=None).to_numpy()
    return X, "vdx_3gene"


def load_vdx_full():
    if not os.path.exists(VDX_FULL_NPZ):
        sys.exit(
            f"Missing {VDX_FULL_NPZ}. Run prep_vdx_full.py first (see module docstring)."
        )
    data = np.load(VDX_FULL_NPZ)
    X = data["X_all"]
    return X, "vdx_full"


def save(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {path}")


def run_dataset(X, label, n_iterations):
    print(f"\n=== {label}: shape={X.shape}, K={K}, n_iter={n_iterations} ===")

    er = ERICA(
        data=X,
        k_range=[K],
        n_iterations=n_iterations,
        method=METHODS,
        transpose=False,
        random_state=42,
        verbose=False,
    )
    results = er.run()

    out_dir = os.path.join(OUT_ROOT, label)
    clam_dir = os.path.join(out_dir, "clams")
    os.makedirs(clam_dir, exist_ok=True)

    for method in METHODS:
        clam = results["clam_matrices"].get((K, method))
        if clam is None:
            print(f"  [SKIP] no CLAM for {method}")
            continue

        clam_path = os.path.join(clam_dir, f"clam_{method}_K{K}.npz")
        np.savez_compressed(clam_path, clam=clam, K=K, method=method, dataset=label)
        size_kb = os.path.getsize(clam_path) / 1024
        print(f"  saved CLAM: {clam_path}  ({size_kb:.1f} KB)")

        # MATLAB-recap figures first (prioritised by ordering).
        fig = plot_assignment_scatter(
            clam, method=method, k=K,
            title=f"{label} - {method}, K={K} - assignment scatter",
        )
        save(fig, out_dir, f"01_assignment_scatter_{method}")

        fig, _ = plot_assignment_heatmap(
            clam, k=K,
            title=f"{label} - {method}, K={K} - cross-assignment (%)",
        )
        save(fig, out_dir, f"02_assignment_heatmap_{method}")

        # Enriched library figures.
        fig = plot_pcsp(clam, k=K,
                        title=f"{label} - {method}, K={K} - PCSP")
        save(fig, out_dir, f"03_pcsp_{method}")

        fig, _ = plot_icah(clam, k=K,
                           title=f"{label} - {method}, K={K} - ICAH")
        save(fig, out_dir, f"04_icah_{method}")

        fig, _ = plot_clam_heatmap_mpl(
            clam, sort=True,
            title=f"{label} - {method}, K={K} - CLAM (sorted)",
        )
        save(fig, out_dir, f"05_clam_heatmap_{method}")

        fig, _ = plot_cluster_sizes_mpl(clam)
        save(fig, out_dir, f"06_cluster_sizes_{method}")

        fig, _ = plot_stability_strips(
            clam, max_samples=80,
            title=f"{label} - {method}, K={K} - stability strips",
        )
        save(fig, out_dir, f"07_stability_strips_{method}")


def main():
    set_publication_style()

    X3, label3 = load_vdx_3gene()
    run_dataset(X3, label3, N_ITERATIONS_3GENE)

    Xf, labelf = load_vdx_full()
    run_dataset(Xf, labelf, N_ITERATIONS_FULL)

    print(f"\nDone. Figures + CLAMs in {OUT_ROOT}")


if __name__ == "__main__":
    main()
