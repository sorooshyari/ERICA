"""
02_run_erica_pipeline.py

Run ERICA on all 7 datasets with kmeans, agglomerative_ward, and hdbscan.
Validates the new flattened method API and HDBSCAN support.
Saves results as joblib files for downstream plotting scripts.
"""

import os
import sys
import time
import numpy as np
import joblib
from pathlib import Path

from erica import ERICA

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data"
RESULTS_DIR = SCRIPT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Dataset configurations
# ---------------------------------------------------------------------------
DATASET_CONFIGS = {
    "vdx_3gene": {
        "k_range": [2, 3, 4, 5, 6],
        "n_iterations": 100,
        "hdbscan_params": {"min_cluster_size": 15},
    },
    "well_separated": {
        "k_range": [2, 3, 4, 5],
        "n_iterations": 100,
        "hdbscan_params": {"min_cluster_size": 10},
    },
    "overlapping": {
        "k_range": [2, 3, 4, 5, 6],
        "n_iterations": 100,
        "hdbscan_params": {"min_cluster_size": 10},
    },
    "moons_2d": {
        "k_range": [2, 3, 4],
        "n_iterations": 100,
        "hdbscan_params": {"min_cluster_size": 10},
    },
    "circles_2d": {
        "k_range": [2, 3, 4],
        "n_iterations": 100,
        "hdbscan_params": {"min_cluster_size": 10},
    },
    "blobs_2d": {
        "k_range": [2, 3, 4, 5],
        "n_iterations": 100,
        "hdbscan_params": {"min_cluster_size": 10},
    },
    "high_dim": {
        "k_range": [2, 3, 4, 5],
        "n_iterations": 100,
        "hdbscan_params": {"min_cluster_size": 10},
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_dataset(name):
    """Load a dataset from the data directory."""
    path = DATA_DIR / f"{name}.npz"
    d = np.load(path, allow_pickle=True)
    X = d["X"]
    meta = d["meta"].item()
    return X, meta


def validate_results(erica_obj, name):
    """Validate ERICA results against expected API structure.

    Returns a list of (check_name, passed_bool) tuples.
    """
    checks = []
    results = erica_obj.get_results()

    # Check 1: auto_k contains 'hdbscan' key
    has_hdbscan_key = "hdbscan" in results.get("auto_k", {})
    checks.append(("auto_k has 'hdbscan'", has_hdbscan_key))

    # Check 2: metrics contain ARI_mean for at least one (k, method) combo
    has_ari = False
    for k, methods in results.get("metrics", {}).items():
        for method, metric_dict in methods.items():
            if "ARI_mean" in metric_dict:
                has_ari = True
                break
        if has_ari:
            break
    checks.append(("metrics contain ARI_mean", has_ari))

    # Check 3: get_auto_k_results('hdbscan') returns non-None
    auto_k_hdbscan = erica_obj.get_auto_k_results("hdbscan")
    checks.append(("get_auto_k_results('hdbscan') non-None", auto_k_hdbscan is not None))

    # Check 4: config method is a list
    config_method = results.get("config", {}).get("method")
    checks.append(("config['method'] is list", isinstance(config_method, list)))

    return checks


def fmt_time(seconds):
    """Format elapsed time as mm:ss."""
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def run_all_datasets():
    """Run ERICA on every dataset, validate, and save results."""
    print("=" * 70)
    print("ERICA Pipeline — Running all 7 datasets")
    print("=" * 70)

    global_start = time.time()
    all_validations = {}

    for i, (name, cfg) in enumerate(DATASET_CONFIGS.items(), 1):
        print(f"\n{'='*70}")
        print(f"[{i}/7] Dataset: {name}")
        print(f"{'='*70}")

        X, meta = load_dataset(name)
        print(f"  Shape: {X.shape}  |  true_k: {meta.get('true_k')}")

        ds_start = time.time()

        erica = ERICA(
            data=X,
            k_range=cfg["k_range"],
            n_iterations=cfg["n_iterations"],
            method=["kmeans", "agglomerative_ward", "hdbscan"],
            transpose=False,
            output_dir=os.path.join(str(RESULTS_DIR), "erica_workdir"),
            verbose=True,
            hdbscan_params=cfg["hdbscan_params"],
        )
        erica.run()

        ds_elapsed = time.time() - ds_start

        # Validate
        checks = validate_results(erica, name)
        all_validations[name] = checks

        print(f"\n  Validation for {name}:")
        for check_name, passed in checks:
            status = "PASS" if passed else "FAIL"
            print(f"    [{status}] {check_name}")

        print(f"  Elapsed: {fmt_time(ds_elapsed)}")

        # Save result
        results = erica.get_results()
        save_obj = {
            "erica_results": results,
            "auto_k_results": erica.get_auto_k_results("hdbscan"),
            "config": {
                "dataset_name": name,
                "n_samples": int(X.shape[0]),
                "n_features": int(X.shape[1]),
                "true_k": meta.get("true_k"),
                "transpose": False,
            },
        }
        out_path = RESULTS_DIR / f"{name}.joblib"
        joblib.dump(save_obj, out_path)
        print(f"  Saved: {out_path}")

    # Summary
    total_elapsed = time.time() - global_start
    print(f"\n{'='*70}")
    print(f"VALIDATION SUMMARY  (total elapsed: {fmt_time(total_elapsed)})")
    print(f"{'='*70}")

    all_pass = True
    for name, checks in all_validations.items():
        failed = [c for c, p in checks if not p]
        if failed:
            all_pass = False
            print(f"  {name:20s}  FAIL  ({', '.join(failed)})")
        else:
            print(f"  {name:20s}  ALL PASS")

    if all_pass:
        print("\nAll validations passed.")
    else:
        print("\nSome validations FAILED — see above.")

    return all_pass


# ---------------------------------------------------------------------------
# HDBSCAN parameter sweep on VDX 3-gene
# ---------------------------------------------------------------------------
def run_hdbscan_sweep():
    """Sweep HDBSCAN parameters on the VDX 3-gene dataset."""
    print(f"\n{'='*70}")
    print("HDBSCAN Parameter Sweep — VDX 3-gene")
    print(f"{'='*70}")

    X, meta = load_dataset("vdx_3gene")
    min_cluster_sizes = [5, 10, 15, 20, 30, 50]
    min_samples_list = [None, 3, 5, 10]

    sweep_results = {}
    total = len(min_cluster_sizes) * len(min_samples_list)
    sweep_start = time.time()

    for idx, mcs in enumerate(min_cluster_sizes):
        for ms in min_samples_list:
            combo_num = idx * len(min_samples_list) + min_samples_list.index(ms) + 1
            label = f"mcs={mcs}, ms={ms}"
            print(f"\n  [{combo_num}/{total}] {label}")

            hparams = {"min_cluster_size": mcs}
            if ms is not None:
                hparams["min_samples"] = ms

            combo_start = time.time()
            erica = ERICA(
                data=X,
                k_range=[2],  # dummy for hdbscan-only
                n_iterations=50,
                method=["hdbscan"],
                transpose=False,
                output_dir=os.path.join(str(RESULTS_DIR), "erica_workdir"),
                verbose=False,
                hdbscan_params=hparams,
            )
            erica.run()
            combo_elapsed = time.time() - combo_start

            auto_k = erica.get_auto_k_results("hdbscan")
            if auto_k is not None:
                k_dist = auto_k.get("k_distribution", {})
                modal_k = auto_k.get("modal_k", None)
                k_agreement = auto_k.get("k_agreement_rate", None)
            else:
                k_dist = {}
                modal_k = None
                k_agreement = None

            sweep_results[(mcs, ms)] = {
                "modal_k": modal_k,
                "k_agreement_rate": k_agreement,
                "k_distribution": k_dist,
                "params": hparams,
            }

            print(f"    modal_k={modal_k}  agreement={k_agreement}  elapsed={fmt_time(combo_elapsed)}")

    sweep_elapsed = time.time() - sweep_start
    out_path = RESULTS_DIR / "vdx_hdbscan_sweep.joblib"
    joblib.dump(sweep_results, out_path)
    print(f"\n  Sweep saved: {out_path}  ({total} combos, elapsed: {fmt_time(sweep_elapsed)})")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    all_pass = run_all_datasets()
    run_hdbscan_sweep()

    if all_pass:
        print("\nPipeline complete — all validations passed.")
    else:
        print("\nPipeline complete — some validations FAILED.")
        sys.exit(1)
