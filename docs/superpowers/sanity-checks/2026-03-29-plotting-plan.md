# Sanity Check Report: Plotting Experiments Plan for ERICA

**Generated:** 2026-03-29
**Mode:** Plan review
**Rounds completed:** 2 / 2
**Convergence status:** Near-converged (Round 2 findings were refinements, not blockers)

---

## Round 1 Findings

### ADVERSARIAL AGENT

**Finding A1 (CRITICAL): `get_results()` return structure vs. .npz serialization mismatch**

The plan says `02_run_erica_pipeline.py` saves results as `.npz` files. However, `get_results()` returns a dict with tuple keys like `(k, 'kmeans')` for `clam_matrices` and nested dicts for `metrics`. `np.savez` cannot serialize tuple keys or nested dicts of dicts. You will hit a `TypeError` or silently lose structure.

- `results['clam_matrices']` has keys like `(3, 'kmeans')` -- not valid `.npz` key names (must be strings).
- `results['metrics']` is `{k: {method: {metric_name: value}}}` -- nested dicts cannot be directly stored in `.npz`.
- `results['auto_k']` contains variable-length lists, Counter objects, etc.

**Recommendation:** Use `joblib.dump` for saving (it handles arbitrary Python objects and is standard in scikit-learn workflows), or manually flatten keys to string format like `"clam_3_kmeans"` and save individual arrays plus a JSON manifest for the metadata. The plan must specify the serialization strategy explicitly, or every plot script will fail to load data.

**Finding A2 (HIGH): Mainz and Transbig datasets may not exist at the planned URLs**

The plan references `mainz_dict.npy` and `transbig_dict.npy` from Lorenzo's repo. However, `check_and_download_example_data()` in `erica/data.py` only knows about `vdx_dict.npy` and `VDX_3_SV.csv`. The DATA_SOURCES dict does not include Mainz or Transbig. Additionally, Lorenzo's repo (`github.com/lorenzomasoero/clustering_replicability`) may or may not host those files at those exact names.

**Recommendation:** Before committing to three real datasets, verify the exact file paths in Lorenzo's repo. The plan should include a verification step or fallback strategy. If only VDX is available, the plan should state that explicitly and note Mainz/Transbig as stretch goals.

**Finding A3 (HIGH): Shape datasets from Lorenzo's repo -- format assumptions are untested**

The plan lists `aggregation.txt`, `compound.txt`, etc. as "from Lorenzo's repo." These files from the Franeti & Sieranoja collection are typically whitespace-separated with columns `x y cluster_id`. But their exact format in Lorenzo's repo may differ. The plan has no data validation or format-checking step.

**Recommendation:** Add a format verification step to `01_fetch_datasets.py` that checks column counts and data types after download.

**Finding A4 (MEDIUM): HDBSCAN parameter sensitivity grid (Plot F, script 07) -- no API support**

The plan includes an "HDBSCAN parameter sensitivity grid" in `07_surfaces.py`. However, the current ERICA API accepts only a single `hdbscan_params` dict. To sweep over `min_cluster_size` and `min_samples`, you would need to instantiate and run ERICA N times with different params. This is not a plotting concern -- it is a pipeline concern that should be addressed in `02_run_erica_pipeline.py`.

**Recommendation:** The plan for script 02 must explicitly include the HDBSCAN parameter sweep loop. Script 07 only loads results, so the sweep must already be computed.

**Finding A5 (MEDIUM): `compute_metrics_for_clam` prints to stdout unconditionally**

Lines 245-260 of `metrics.py` show that `compute_metrics_for_clam` prints a formatted summary to stdout every time it is called. Running the pipeline on multiple datasets, methods, and K values will produce hundreds of print statements. This is messy for batch pipeline scripts.

**Recommendation:** Either suppress output in script 02 (redirect stdout), or note this as a known annoyance. It does not block the plan, but it is worth knowing.

---

### PRAGMATIC AGENT

**Finding P1 (HIGH): Runtime estimate missing -- VDX 22K genes x 344 samples at 200 iterations is expensive**

Running ERICA with `method=['kmeans', 'agglomerative_ward', 'hdbscan']` on VDX (22K features after transpose = 344 samples x 22K features) for K=2..6 at 200 iterations means:
- K-Means: 5 K values x 200 iterations = 1,000 clustering calls
- Agglomerative ward: 5 K values x 200 iterations = 1,000 calls
- HDBSCAN: 200 iterations

Each K-Means iteration on ~275 training samples x 22K features is fast (~0.1s), but agglomerative clustering computes an O(n^2) distance matrix on 22K features. On 344 samples this is manageable, but the plan should include timing estimates and potentially recommend running the pipeline overnight.

**Recommendation:** Add an estimated runtime note. Consider running VDX with 100 iterations first as a smoke test.

**Finding P2 (HIGH): The `transpose` parameter must be set correctly per dataset**

VDX `.npy` loads as `(22283, 344)` -- genomics format (features x samples). So `transpose=True` is correct (the default). But the 2D shape datasets (`aggregation.txt`, etc.) are already in `(n_samples, n_features)` format. And `make_blobs` returns `(n_samples, n_features)`. The plan does not mention that `transpose=False` must be used for shape datasets and synthetic data. Getting this wrong silently produces garbage results.

**Recommendation:** Script 02 must explicitly set `transpose` per dataset type. This is a common source of confusion and should be called out in the plan.

**Finding P3 (MEDIUM): The plan says "each script loads from results/" but does not specify the exact file naming convention**

Scripts 03-07 all load from `results/`. But the plan never specifies the naming convention for saved files. Without a clear contract (e.g., `results/vdx_kmeans_k2-6_200iter.dat`), each plot script will have to guess file names or hardcode them.

**Recommendation:** Define an explicit naming convention in the plan. A manifest file (`results/manifest.json`) listing all result files and their configs would be even better.

**Finding P4 (MEDIUM): 3D matplotlib surfaces (script 07) are notoriously hard to make publication-quality**

`plot_surface` in matplotlib produces acceptable but not great 3D plots. Camera angles, axis labels overlapping, and colorbar placement all require manual tuning. For a publication, 3D plots often get rejected in favor of 2D heatmaps.

**Recommendation:** Consider a 2D heatmap fallback (imshow with K on one axis, method on the other, metric as color) as the primary output, with 3D surface as an optional "fancy" view.

**Finding P5 (LOW): The `.gitignore` does not yet include `plotting_experiments/` patterns**

The plan says `data/`, `results/`, and `figures/` inside `plotting_experiments/` should be gitignored. The current `.gitignore` has no rules for this directory.

**Recommendation:** Add gitignore rules before creating the directory structure.

---

### VISIONARY AGENT

**Finding V1 (HIGH): Missing shared style module -- plan has 5 plot scripts with duplicated rcParams**

The plan says all scripts use `matplotlib.rc('xtick', labelsize=14), rcParams['lines.linewidth'] = 2, 'cool' colormap`. If this is copy-pasted across 5 scripts, any style change requires editing 5 files.

**Recommendation:** Add a `plotting_experiments/style.py` or `_style.py` module that exports a `set_publication_style()` function and a colormap constant. Every plot script imports it. This is a one-file addition that saves significant maintenance cost.

**Finding V2 (MEDIUM): No CLI interface or Makefile for running the pipeline end-to-end**

The plan has 7 scripts that must be run in order (01 through 07). But there is no driver script or Makefile. A new contributor has to read the README to understand the order.

**Recommendation:** Add a `Makefile` or `run_all.sh` with targets: `make fetch`, `make pipeline`, `make plots`, `make all`. This takes 10 minutes to write and saves everyone time.

**Finding V3 (MEDIUM): Plan does not address figure sizing for publication columns**

Publication figures need exact dimensions (e.g., single column = 3.5 inches, double column = 7 inches, at 300 DPI). The plan specifies DPI but not figure widths. Without this, figures may need resizing later.

**Recommendation:** Define target figure sizes in the style module. For a typical bioinformatics journal: single column = 3.5", double = 7.0", with 8pt minimum font size.

**Finding V4 (LOW): Opportunity to make plot scripts reusable as library functions**

The 5 plot scripts each create one class of figure. These could be implemented as functions in a `plotting_experiments/plots.py` module, then called from thin scripts. This makes them reusable from notebooks or other contexts.

**Recommendation:** Consider a two-layer architecture: plot functions in a module, and scripts that call them. Not critical, but good practice.

---

## Round 2 Findings (Delta from Round 1)

### ADVERSARIAL AGENT

**Finding A6 (MEDIUM): CLAM matrix shape inconsistency between K-based and HDBSCAN methods**

For K-based methods, `clam_matrices_` is keyed by `(k, method)` and each matrix is `(n_samples, k)`. For HDBSCAN, `auto_k_results_['hdbscan']['clam_matrix']` is `(n_samples, modal_k)`. The plan's script 03 (CLAM heatmaps) would need to handle both access patterns. The plan does not mention this.

**Recommendation:** Script 03 must branch on method type to extract CLAM matrices from different locations in the result dict. Document this in the plan.

**Finding A7 (LOW): `iteration_labels` structure differs between methods**

For K-based methods, `iteration_labels` is always populated. For HDBSCAN, some iterations have empty arrays (`np.array([])`) when all points are noise. Plot scripts using these labels (e.g., for per-iteration ARI distributions) need to filter empties.

**Recommendation:** Note this edge case in the plan for script 05.

### PRAGMATIC AGENT

**Finding P6 (MEDIUM): ARI/AMI error bars require storing per-iteration scores, not just mean/std**

The plan says script 05 plots "error bars for ARI/AMI." The ERICA results store `ARI_mean` and `ARI_std` but NOT the per-iteration scores. To plot proper error bars, you need the raw list. These are computed in `_compute_all_metrics` but only the aggregated values are stored in the result dict.

The raw scores ARE available in `results[(k, method)]['iteration_labels']` -- you can recompute them. But the plan should acknowledge this recomputation step.

**Recommendation:** Either (a) have script 02 save per-iteration ARI/AMI lists alongside aggregated values, or (b) have script 05 recompute from iteration_labels. Option (a) is cleaner.

**Finding P7 (LOW): The `select_optimal_k` in the result dict is per-metric, per-method**

`k_star_` structure is `{metric_name: {method_name: k_value}}`. Script 05 needs to extract K* for the right metric. The plan says "vertical dashed line at K*" but does not specify which metric's K* to show (CRI? WCRI? TWCRI?). When plotting all three metrics on the same plot, there could be three different K* values.

**Recommendation:** Show K* for each metric with matching colored dashed lines, or only show TWCRI K* with annotation. Specify this in the plan.

### VISIONARY AGENT

**Finding V5 (LOW): No mention of colorblind-friendly palettes**

Publication-quality figures increasingly require colorblind-friendly palettes. The plan uses the 'cool' colormap for heatmaps, which is not colorblind-safe. For lines, no palette is specified.

**Recommendation:** Use 'viridis' or 'cividis' for heatmaps (both are perceptually uniform and colorblind-safe). For categorical lines, use a colorblind-friendly qualitative palette (e.g., Okabe-Ito or Tol's qualitative scheme).

**No further new critical findings in Round 2.** The adversarial, pragmatic, and visionary agents are converging. Remaining items are refinements.

---

## Summary of Action Items (Prioritized)

### Must-Fix Before Implementation

| # | Finding | Priority | Action |
|---|---------|----------|--------|
| A1 | `.npz` cannot serialize tuple keys / nested dicts | CRITICAL | Switch to `joblib.dump` or define explicit flattened key scheme with JSON manifest |
| A2 | Mainz/Transbig URLs unverified | HIGH | Verify file existence in Lorenzo's repo before plan finalization; add fallback |
| P2 | `transpose` must vary per dataset | HIGH | Explicitly specify `transpose=False` for shape/synthetic data in script 02 |
| V1 | Duplicated style across 5 scripts | HIGH | Add `style.py` shared module |
| P1 | No runtime estimate | HIGH | Add timing notes; recommend smoke-test at 100 iterations |

### Should-Fix

| # | Finding | Priority | Action |
|---|---------|----------|--------|
| A3 | Shape dataset format unverified | HIGH | Add format validation in script 01 |
| A4 | HDBSCAN param sweep not in pipeline plan | MEDIUM | Add sweep loop to script 02 spec |
| P3 | No file naming convention | MEDIUM | Define naming scheme + optional manifest |
| P6 | Per-iteration ARI/AMI not stored | MEDIUM | Save raw scores in pipeline step |
| A6 | CLAM access differs K-based vs HDBSCAN | MEDIUM | Document access pattern branching |
| P7 | Ambiguous K* on multi-metric plot | MEDIUM | Specify which K* lines to draw |
| P4 | 3D surfaces hard to make pub-quality | MEDIUM | Add 2D heatmap as primary fallback |
| V2 | No Makefile / driver script | MEDIUM | Add `Makefile` or `run_all.sh` |
| V3 | No figure size specs | MEDIUM | Define sizes in style module |

### Nice-to-Have

| # | Finding | Priority | Action |
|---|---------|----------|--------|
| A5 | `compute_metrics_for_clam` prints to stdout | LOW | Note as known; optionally suppress |
| A7 | HDBSCAN empty iteration_labels | LOW | Document filter requirement |
| P5 | `.gitignore` update needed | LOW | Add rules before creating dirs |
| V4 | Plot scripts could be reusable functions | LOW | Two-layer architecture (optional) |
| V5 | Colorblind-unfriendly palette | LOW | Switch to viridis/cividis |

---

## Verdict

The plan is **structurally sound** -- the phased separation of data acquisition, pipeline execution, and plotting is correct and well-motivated. The seven-script decomposition maps cleanly to distinct plot types.

However, **one critical issue (A1: serialization format)** will cause an immediate `TypeError` at runtime and must be resolved before implementation begins. Several high-priority issues (dataset URL verification, transpose handling, shared style module) should also be addressed to avoid wasted implementation cycles.

The plan is ready for implementation after addressing the Must-Fix items above.
