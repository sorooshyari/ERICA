# Sanity Check Report: plotting-experiments-design.md (Pass 2)

**Date:** 2026-03-29
**Target:** `docs/superpowers/specs/2026-03-29-plotting-experiments-design.md`
**Rounds:** 2 (converged -- Round 2 findings were marginal)

---

## Round 1 Findings (3 reviewers)

### P0 -- Spec-Breaking (FIXED)

| # | Finding | Fix Applied |
|---|---------|-------------|
| 1 | **Wrong dict key for HDBSCAN results.** Spec used `auto_k_results` but `get_results()` returns `auto_k`. Plot scripts would KeyError. | Fixed all access paths in spec. Added full key map in output format comment block. |
| 2 | **`iteration_labels` duplication was wrong.** Spec proposed a separate `iteration_labels` dict extracted from `erica.results_` but omitted HDBSCAN labels (which live in `auto_k_results_`). | Removed the redundant extraction. Documented that labels are already inside `erica_results['results'][(k,method)]` and `erica_results['auto_k']['hdbscan']`. |
| 3 | **VDX_3_SV.csv loading would lose a sample.** The CSV has NO header row, NO sample_id column (pure 344x3 floats). Using `erica.data.load_data()` would treat row 0 as column headers, yielding 343 samples. | Changed spec to use `np.loadtxt(path, delimiter=',')` with explanation of why. |

### P1 -- Will Cause Runtime Errors or Wrong Results (FIXED)

| # | Finding | Fix Applied |
|---|---------|-------------|
| 4 | **HDBSCAN sweep described as "K=2..6"** but HDBSCAN ignores k_range. Misleading and could cause confusion. | Clarified that k_range is irrelevant for HDBSCAN-only runs; use dummy `k_range=[2]`. |
| 5 | **Plot scripts had no documented access patterns** for the nested metrics dict. Structure is `{k: {method: {metric_name: value}}}` with tuple keys for clam_matrices. | Added explicit access patterns in sections 03 and 05. |
| 6 | **`compute_metrics_for_clam()` prints to stdout** for every call (~100+ times during pipeline). | Added note warning about expected verbose output. |
| 7 | **Prose said "verifies `auto_k_results` in output"** but the actual key is `auto_k`. | Fixed wording. |

### P2 -- Minor / Defensive

| # | Finding | Status |
|---|---------|--------|
| 8 | `make_blobs` for 2D didn't specify `n_features=2` explicitly | Fixed |
| 9 | HDBSCAN param sweep surface plot needs NaN handling for modal_k=0 combos | Added edge case note |
| 10 | Redundant `auto_k_results` shortcut key in saved dict (same data as in erica_results) | Left as-is, documented as convenience shortcut |

---

## Round 2 Findings (convergence check)

### All 3 reviewers found only marginal issues:

| # | Finding | Severity | Action |
|---|---------|----------|--------|
| 11 | `min_samples=None` passed explicitly to sklearn HDBSCAN -- works but slightly unclean | P2 | No change (sklearn handles it) |
| 12 | Makefile plot targets are sequential (could parallelize) | P3 | No change (optimization, not correctness) |
| 13 | No `matplotlib.use('Agg')` for headless environments | P3 | No change (local dev only) |
| 14 | Saved dict has redundant copy of HDBSCAN results (both `auto_k_results` key and inside `erica_results['auto_k']`) | P2 | No change (documented as accessor validation) |

**Verdict: CONVERGED after Round 2.** All P0/P1 issues from Round 1 have been fixed. Round 2 found only P2/P3 items that don't warrant spec changes.

---

## Key Structural Insights for Implementation

1. **`get_results()` return structure** (verified from source):
   ```
   {
     'clam_matrices': {(k, method_str): np.ndarray},     # tuple keys!
     'metrics':       {k_int: {method_str: {metric: val}}},
     'k_star':        {metric_str: {method_str: k_int}},
     'disqualified_k': {method_str: [k_ints]},
     'auto_k':        {method_str: hdbscan_dict},         # NOT 'auto_k_results'
     'config':        {k_range, n_iterations, ...},
     'output_folders': [str],
     'results':       {(k, method_str): raw_result_dict},  # includes iteration_labels
   }
   ```

2. **HDBSCAN result dict keys** (from `hdbscan_clustering()`):
   ```
   k_distribution, modal_k, k_agreement_rate, clam_matrix,
   n_iterations_used, iteration_labels, noise_counts, output_folder
   ```
   Plus after `_compute_all_metrics()` adds: `ARI_mean, ARI_std, AMI_mean, AMI_std, metrics_at_modal_k`

3. **VDX_3_SV.csv**: 344 rows x 3 cols, NO header, NO index column. Pure float CSV.

---

## Files Modified

- `/Users/shawnshirazi/LocalExperiments/ERICA/docs/superpowers/specs/2026-03-29-plotting-experiments-design.md` -- 8 edits applied
