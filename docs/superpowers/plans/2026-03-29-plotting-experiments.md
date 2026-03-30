# Plotting Experiments Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a `plotting_experiments/` folder with 7 scripts that generate datasets, run the ERICA pipeline (validating the new flattened method API + HDBSCAN), and produce publication-quality matplotlib figures.

**Architecture:** Three phases: (1) data generation with sklearn + local VDX CSV, (2) ERICA pipeline runs saved via joblib, (3) matplotlib plot scripts loading from saved results. A shared `style.py` module provides publication rcParams. A `Makefile` orchestrates the phases.

**Tech Stack:** Python, numpy, matplotlib, scikit-learn, joblib, erica (pip-installed)

**Spec:** `docs/superpowers/specs/2026-03-29-plotting-experiments-design.md`

**Note on joblib:** This plan uses `joblib.dump`/`joblib.load` for serializing ERICA results because ERICA's `get_results()` returns dicts with tuple keys like `(3, 'kmeans')` and nested structures that `np.savez` cannot handle. joblib is bundled with scikit-learn (already an ERICA dependency) and is standard practice in sklearn workflows. The serialized files are locally generated, not downloaded from external sources.

---

### Task 1: Scaffold Directory Structure

**Files:**
- Create: `plotting_experiments/` directory tree
- Modify: `.gitignore`
- Create: `plotting_experiments/Makefile`
- Create: `plotting_experiments/README.md`

- [ ] **Step 1: Create directories**

```bash
mkdir -p plotting_experiments/{data,results,figures}
```

- [ ] **Step 2: Add gitignore rules**

Append to `.gitignore`:

```
# Plotting experiments artifacts
plotting_experiments/data/
plotting_experiments/results/
plotting_experiments/figures/
```

- [ ] **Step 3: Create Makefile**

Create `plotting_experiments/Makefile`:

```makefile
.PHONY: data pipeline plots all clean

data:
	python 01_generate_datasets.py

pipeline: data
	python 02_run_erica_pipeline.py

plots: pipeline
	python 03_clam_heatmaps.py
	python 04_stability_strips.py
	python 05_metrics_curves.py
	python 06_method_comparison.py
	python 07_surfaces.py

all: plots

clean:
	rm -rf data/ results/ figures/
```

- [ ] **Step 4: Create README.md**

Create `plotting_experiments/README.md`:

```markdown
# Plotting Experiments

Publication-quality matplotlib visualizations of ERICA clustering results.
Also serves as end-to-end validation of the flattened method API + HDBSCAN support.

## Prerequisites

```bash
pip install -e ".[all]"  # from the ERICA repo root
```

## Quick Start

```bash
cd plotting_experiments
make all  # runs data generation, ERICA pipeline, and all plots
```

## Individual Steps

```bash
make data      # 01: generate/copy datasets (~5 seconds)
make pipeline  # 02: run ERICA on all datasets (~15 minutes)
make plots     # 03-07: generate all figures (~30 seconds)
make clean     # remove data/, results/, figures/
```

## Scripts

| # | Script | What it does |
|---|--------|-------------|
| 01 | `01_generate_datasets.py` | Copies VDX 3-gene CSV, generates sklearn synthetics |
| 02 | `02_run_erica_pipeline.py` | Runs ERICA on each dataset, saves results via joblib |
| 03 | `03_clam_heatmaps.py` | Raw + sorted CLAM heatmaps |
| 04 | `04_stability_strips.py` | Per-sample cluster assignment stacked bars |
| 05 | `05_metrics_curves.py` | CRI/WCRI/TWCRI and ARI/AMI vs K curves |
| 06 | `06_method_comparison.py` | Side-by-side method comparison panels |
| 07 | `07_surfaces.py` | 3D surfaces + 2D heatmap fallbacks |

## Output

Figures are saved to `figures/` as both PDF (publication) and PNG (preview).
```

- [ ] **Step 5: Commit**

```bash
git add plotting_experiments/Makefile plotting_experiments/README.md .gitignore
git commit -m "feat: scaffold plotting_experiments directory structure"
```

---

### Task 2: Shared Style Module

**Files:**
- Create: `plotting_experiments/style.py`

- [ ] **Step 1: Create style.py with publication rcParams, colormaps, and save helper**

The file should export: `set_publication_style()`, `save_figure(fig, name)`, `SINGLE_COL` (3.5), `DOUBLE_COL` (7.0), `CMAP_SEQ` ('viridis'), `CMAP_DIV` ('RdBu_r'), `METHOD_COLORS` (Okabe-Ito palette dict), `METRIC_COLORS` (CRI/WCRI/TWCRI/ARI/AMI color dict), `METRIC_DASHES` (dash pattern dict). `save_figure` creates `figures/` dir via `os.makedirs(exist_ok=True)` and saves as both PDF and PNG.

- [ ] **Step 2: Verify import works**

```bash
cd plotting_experiments && python -c "from style import set_publication_style, CMAP_SEQ; set_publication_style(); print('OK')"
```

- [ ] **Step 3: Commit**

```bash
git add plotting_experiments/style.py
git commit -m "feat: add shared publication style module"
```

---

### Task 3: Dataset Generation

**Files:**
- Create: `plotting_experiments/01_generate_datasets.py`

- [ ] **Step 1: Create script that loads VDX_3_SV.csv via `np.loadtxt` (NOT erica.data.load_data) and generates 6 sklearn synthetic datasets**

Datasets: vdx_3gene (344x3), well_separated (300x5, 3 centers), overlapping (300x5, 4 centers), moons_2d (300x2), circles_2d (300x2), blobs_2d (300x2, 3 centers), high_dim (200x50, 3 centers). Each saved as `data/{name}.npz` with keys `X` and `meta`.

- [ ] **Step 2: Run and verify**

```bash
cd plotting_experiments && python 01_generate_datasets.py
python -c "import numpy as np; d=np.load('data/vdx_3gene.npz',allow_pickle=True); print(d['X'].shape)"
```

Expected: `(344, 3)`

- [ ] **Step 3: Commit**

```bash
git add plotting_experiments/01_generate_datasets.py
git commit -m "feat: add dataset generation script"
```

---

### Task 4: ERICA Pipeline

**Files:**
- Create: `plotting_experiments/02_run_erica_pipeline.py`

- [ ] **Step 1: Create pipeline script**

Must: import ERICA via `from erica import ERICA`, use `method=['kmeans', 'agglomerative_ward', 'hdbscan']`, set `transpose=False`, set `output_dir='results/erica_workdir'`, save results via `joblib.dump`. Include validation that checks for `auto_k` key, ARI/AMI in metrics, and calls `get_auto_k_results('hdbscan')`. Also include HDBSCAN parameter sweep (6 min_cluster_size x 4 min_samples = 24 combos, 50 iterations each, `method=['hdbscan']`, dummy `k_range=[2]`).

- [ ] **Step 2: Run pipeline**

```bash
cd plotting_experiments && python 02_run_erica_pipeline.py
```

Expected: ~15 minutes. Validation summaries printed per dataset.

- [ ] **Step 3: Verify results are loadable and complete**

```bash
python -c "
import joblib
r = joblib.load('results/vdx_3gene.joblib')
er = r['erica_results']
print('auto_k hdbscan modal_k:', er['auto_k']['hdbscan']['modal_k'])
print('ARI_mean K=3 kmeans:', er['metrics'][3]['kmeans']['ARI_mean'])
print('Config method:', er['config']['method'])
"
```

- [ ] **Step 4: Commit**

```bash
git add plotting_experiments/02_run_erica_pipeline.py
git commit -m "feat: add ERICA pipeline with API validation and HDBSCAN sweep"
```

---

### Task 5: CLAM Heatmaps

**Files:**
- Create: `plotting_experiments/03_clam_heatmaps.py`

- [ ] **Step 1: Create script with `sort_clam()`, `plot_clam()`, and `plot_clam_multipanel()` functions**

Sorted view: samples sorted by primary cluster (argmax), then by assignment strength (descending) within cluster. Handles zero-sum rows. Access K-based CLAMs via `er['clam_matrices'][(k, method)]` (tuple keys). Access HDBSCAN CLAM via `er['auto_k']['hdbscan']['clam_matrix']`. Default outputs: sorted single (K=3 kmeans), raw single, multipanel (K=2,3,4), HDBSCAN at modal K.

- [ ] **Step 2: Run and verify figures**

```bash
cd plotting_experiments && python 03_clam_heatmaps.py && ls figures/clam_*.pdf
```

- [ ] **Step 3: Commit**

```bash
git add plotting_experiments/03_clam_heatmaps.py
git commit -m "feat: add CLAM heatmap plots (raw + sorted + multipanel)"
```

---

### Task 6: Stability Strips

**Files:**
- Create: `plotting_experiments/04_stability_strips.py`

- [ ] **Step 1: Create script with entropy-sorted horizontal stacked bars**

Normalize CLAM rows to proportions (skip zero-sum rows). Sort by Shannon entropy. Subsample to max 100 samples. Use qualitative colorblind-friendly cluster colors.

- [ ] **Step 2: Run and verify**

```bash
cd plotting_experiments && python 04_stability_strips.py && ls figures/stability_*.pdf
```

- [ ] **Step 3: Commit**

```bash
git add plotting_experiments/04_stability_strips.py
git commit -m "feat: add stability strip plots"
```

---

### Task 7: Metrics Curves

**Files:**
- Create: `plotting_experiments/05_metrics_curves.py`

- [ ] **Step 1: Create script with two-panel figure: ERICA metrics + Parmigiani ARI/AMI**

Panel 1: CRI/WCRI/TWCRI vs K with color-matched K* dashed lines. Mark NaN (disqualified K) with 'x' markers. Panel 2: ARI mean +/- std and AMI mean +/- std errorbar plots. Access metrics via `er['metrics'][k][method]['CRI']` etc. K* via `er['k_star']['TWCRI'][method]`. Generate for vdx_3gene, well_separated, overlapping x kmeans, agglomerative_ward.

- [ ] **Step 2: Run and verify**

```bash
cd plotting_experiments && python 05_metrics_curves.py && ls figures/metrics_*.pdf
```

- [ ] **Step 3: Commit**

```bash
git add plotting_experiments/05_metrics_curves.py
git commit -m "feat: add metrics curves plots (ERICA + Parmigiani)"
```

---

### Task 8: Method Comparison

**Files:**
- Create: `plotting_experiments/06_method_comparison.py`

- [ ] **Step 1: Create script with 3-subplot comparison + K* grouped bars**

Main figure: CRI, WCRI, TWCRI subplots with K-based methods as lines and HDBSCAN as single star marker at modal K (only if `k_agreement_rate > 0`). Secondary: grouped bar chart of K* per method per metric. Use METHOD_COLORS from style.py.

- [ ] **Step 2: Run and verify**

```bash
cd plotting_experiments && python 06_method_comparison.py && ls figures/method_*.pdf figures/kstar_*.pdf
```

- [ ] **Step 3: Commit**

```bash
git add plotting_experiments/06_method_comparison.py
git commit -m "feat: add method comparison plots"
```

---

### Task 9: 3D Surfaces + 2D Fallbacks

**Files:**
- Create: `plotting_experiments/07_surfaces.py`

- [ ] **Step 1: Create script with three surface types, each 2D + 3D**

A) CLAM topography: sorted CLAM as `plot_surface`. B) Metric landscape: K x method x TWCRI as 2D `imshow` (annotated cells) + 3D surface. C) HDBSCAN sensitivity: min_cluster_size x min_samples x k_agreement_rate from sweep results. Mark zero-agreement cells with 'X' in 2D view. Load sweep from `results/vdx_hdbscan_sweep.joblib`.

- [ ] **Step 2: Run and verify**

```bash
cd plotting_experiments && python 07_surfaces.py && ls figures/surface_*.pdf figures/landscape_*.pdf figures/hdbscan_*.pdf
```

- [ ] **Step 3: Commit**

```bash
git add plotting_experiments/07_surfaces.py
git commit -m "feat: add 3D surface and 2D heatmap plots"
```

---

### Task 10: End-to-End Verification

- [ ] **Step 1: Clean and run everything**

```bash
cd plotting_experiments && make clean && make all
```

- [ ] **Step 2: Verify figure count**

```bash
ls figures/*.pdf | wc -l
```

Expected: ~20+ PDFs.

- [ ] **Step 3: Spot-check a figure**

```bash
open figures/clam_sorted_vdx_3gene_kmeans_k3.pdf
```

- [ ] **Step 4: Final commit**

```bash
git add -A plotting_experiments/
git commit -m "feat: complete plotting experiments — all scripts verified end-to-end"
```
