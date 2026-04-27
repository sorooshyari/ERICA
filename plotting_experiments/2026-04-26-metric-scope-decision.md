# Metric Scope Decision — 2026-04-26

## Decision

ERICA library focus: **CRI, WCRI, TWCRI only**. Drop ARI/AMI (partition comparison metrics, originally from Parmigiani et al.) from the canonical workflow.

## Why

ARI/AMI does not translate cleanly to ERICA's canonical workflow:

- **ERICA workflow** stores CLAM matrices per `(K, method)`. CRI/WCRI/TWCRI derive directly from CLAM — no extra state needed.
- **ARI/AMI workflow** requires per-iteration label pairs (`predicted_labels` from train-fit projected onto test, `true_labels` from fresh fit on test). Not derivable from CLAM alone. Needs extra storage + extra clustering fit per iteration.
- Forcing both into one pipeline made `core.py` carry two parallel computation paths that share only the train/test splits.
- Interpretation also diverges: CRI/WCRI/TWCRI are sample-level assignment stability; ARI/AMI is partition-level transferability. Mixing them on the same plot is a category error for end users.

## Scope changes

### Library (`erica/`)

| Surface | Status |
|---|---|
| `compute_ari`, `compute_ami` | candidate for removal |
| `compute_partition_metrics`, `aggregate_partition_metrics` | candidate for removal (canonical names landed 2026-04-26) |
| `compute_parmigiani_metrics`, `aggregate_parmigiani_metrics` | candidate for removal (deprecated aliases landed 2026-04-26) |
| `ARI_mean`/`ARI_std`/`AMI_mean`/`AMI_std` keys in `er['metrics'][k][method]` | drop from `core.py` pipeline |
| `iteration_labels` storage in clustering results | drop unless needed elsewhere |
| `plot_partition_metrics` in `erica/plotting_mpl.py` | candidate for removal |

### plotting_experiments

- [05_metrics_curves.py](05_metrics_curves.py) — currently a 2-panel figure (ERICA top, ARI/AMI bottom). Rework to ERICA-only single panel or 1×3 (CRI/WCRI/TWCRI side-by-side).
- [gaussian_mixture_study/05_vdx_comparison.py](gaussian_mixture_study/05_vdx_comparison.py) — 2×2 grid currently shows TWCRI top row + ARI bottom row. Drop bottom row or swap for CRI/WCRI.
- Any future galleries — no ARI/AMI panels.

## Rip plan (pending user confirmation)

Option 1 (recommended): hard rip — delete ARI/AMI compute functions, plot function, pipeline integration, `iteration_labels` storage. Breaking change for any user who imported these. Library smaller, single coherent metric family, no deprecation debt.

Option 2: soft deprecate — keep functions, stop computing in `core.py` pipeline. Faster ERICA runs, back-compat preserved.

## Rationale for tracking this here

`plotting_experiments/` is the staging ground where library shape is figured out. Recording metric-scope decisions here before code changes lets the experimental scripts and library evolve in lock-step.

## Rip executed (2026-04-26)

Hard rip of ARI/AMI/Parmigiani references from `plotting_experiments/` scripts (library `erica/` handled in parallel by another agent):

- `05_metrics_curves.py` — collapsed 2-panel layout to single-panel CRI/WCRI/TWCRI; dropped `PARMI_METRICS` and ARI/AMI accumulators.
- `11_generate_all_plots.py` — `plot_metrics_curves` now single-panel; dropped `PARMI_METRICS` and ARI/AMI keys.
- `gaussian_mixture_study/05_vdx_comparison.py` — 2x2 grid reduced to 1x2 (TWCRI only across VDX vs Gaussian); dropped HDBSCAN ARI lookup; updated suptitle.
- `gaussian_mixture_study/04_sigma_curves.py` — removed Figure B (`plot_ari_ami_vs_sigma`); now generates only Figures A, C, D.
- `gaussian_mixture_study/02_run_erica.py` — dropped ARI@4 from per-method summary print.
- `02_run_erica_pipeline.py` — removed the "metrics contain ARI_mean" validation check.
- `literature_comparison/01_error_bands.py` — collapsed 2-panel layout to single-panel ERICA bands; dropped Parmigiani band computation and `compute_per_iteration_ari_ami`.
- `erica_statistics/01_distributions.py` — 3-panel (ERICA stat / ARI / AMI) collapsed to single ERICA stat panel.
- `erica_statistics/02_statistics_vs_k.py` — three-line plot (ERICA / ARI / AMI) reduced to single ERICA stat line.
- `erica_statistics/03_method_comparison.py` — 2x3 grid reduced to 2x1 (ERICA stat per method).
- `erica_statistics/04_cross_dataset_summary.py` — 3-column heatmap reduced to 1-column ERICA stat.
- `erica_statistics/05_sigma_degradation.py` — three-line panel reduced to single ERICA stat line.
- `PLOTTING_GUIDE.md` — updated overview, results table (dropped Parmigiani metrics row), and `05_metrics_curves.py` description.
- `WHAT_WE_DID.md` — annotated with a 2026-04-26 update banner; historical content preserved.

No files were deleted; all scripts still parse and import cleanly.
