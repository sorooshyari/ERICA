"""
01_generate_datasets.py

Generate and save datasets for plotting experiments:
  - VDX 3-gene subset (real breast cancer gene expression data)
  - 6 sklearn synthetic datasets
"""

import numpy as np
from pathlib import Path
from sklearn.datasets import make_blobs, make_moons, make_circles

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

VDX_CSV = SCRIPT_DIR / "../examples/data/VDX_3_SV.csv"


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def save_dataset(name, X, meta):
    out_path = DATA_DIR / f"{name}.npz"
    np.savez(out_path, X=X, meta=meta)
    print(f"  Saved {out_path.name}  shape={X.shape}")


# ---------------------------------------------------------------------------
# 1. VDX 3-gene dataset
# ---------------------------------------------------------------------------
print("Loading VDX 3-gene dataset...")
X_vdx = np.loadtxt(VDX_CSV, delimiter=",")
print(f"  Shape: {X_vdx.shape}")
assert X_vdx.shape == (344, 3), f"Expected (344, 3), got {X_vdx.shape}"

save_dataset(
    "vdx_3gene",
    X_vdx,
    {
        "n_samples": 344,
        "n_features": 3,
        "true_k": None,
        "description": "VDX breast cancer dataset — 3-gene subset (Parmigiani et al.)",
    },
)

# ---------------------------------------------------------------------------
# 2. Synthetic datasets
# ---------------------------------------------------------------------------
print("\nGenerating synthetic datasets...")

synthetic = [
    (
        "well_separated",
        *make_blobs(n_samples=300, centers=3, cluster_std=0.8, n_features=5, random_state=42),
        3,
        "Well-separated blobs (5-D, 3 clusters, std=0.8)",
    ),
    (
        "overlapping",
        *make_blobs(n_samples=300, centers=4, cluster_std=2.0, n_features=5, random_state=42),
        4,
        "Overlapping blobs (5-D, 4 clusters, std=2.0)",
    ),
    (
        "moons_2d",
        *make_moons(n_samples=300, noise=0.05, random_state=42),
        2,
        "Two moons (2-D, noise=0.05)",
    ),
    (
        "circles_2d",
        *make_circles(n_samples=300, noise=0.05, factor=0.5, random_state=42),
        2,
        "Concentric circles (2-D, noise=0.05, factor=0.5)",
    ),
    (
        "blobs_2d",
        *make_blobs(n_samples=300, centers=3, cluster_std=0.5, n_features=2, random_state=42),
        3,
        "Tight blobs (2-D, 3 clusters, std=0.5)",
    ),
    (
        "high_dim",
        *make_blobs(n_samples=200, centers=3, n_features=50, random_state=42),
        3,
        "High-dimensional blobs (50-D, 3 clusters)",
    ),
]

for name, X, y, true_k, description in synthetic:
    save_dataset(
        name,
        X,
        {
            "n_samples": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "true_k": int(true_k),
            "description": description,
        },
    )

print("\nAll datasets generated successfully.")
