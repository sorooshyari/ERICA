# Sanity Check Report: plotting_experiments/literature_comparison/

**Date:** 2026-03-29
**Target:** `plotting_experiments/literature_comparison/` (3 scripts + README)
**Mode:** code | 2 rounds | 3 agents (Adversarial, Pragmatic, Visionary)

---

## Executive Summary

Two rounds of review found **1 P1 issue** (co-assignment matrix semantics), **1 P1 documentation issue** (author attribution), and **1 P2 observation** (shared band widths). All P1 issues have been fixed. No P0 (data-corrupting) bugs were found. The figures are correct for what they compute, though the co-assignment heatmap computes a different quantity than the Monti consensus matrix it was named after.

---

## Round 1 Findings

### F1. Co-assignment matrix is NOT the Monti consensus matrix [P1 -- FIXED]

**File:** `02_coassignment_heatmap.py`, `compute_coassignment()`

The code computes `normalize(clam, 'l1') @ normalize(clam, 'l1').T`, which yields the inner product of assignment probability vectors:

    coassign[i, j] = sum_c P(i -> c) * P(j -> c)

The **Monti consensus matrix** is:

    M(i, j) = (# iterations where i and j co-cluster) / (# iterations where both appear in test)

These are different quantities. The inner product is a marginal product; the Monti matrix conditions on co-occurrence in the same iteration. Key symptom: the diagonal of the inner-product matrix is < 1.0 for uncertain samples, while the Monti diagonal is always 1.0.

**However:** The inner-product matrix IS useful -- it reveals the same block-diagonal structure for stable clusterings, and the two converge for well-separated clusters. Computing the true Monti matrix requires iteration-level co-occurrence data not available from the aggregated CLAM alone.

**Fix applied:** Updated docstrings, module docstring, colorbar label, figure title, and README to accurately describe the quantity being computed. The computation itself is retained as-is since it is a valid and informative visualization.

### F2. Author attribution: "Parmigiani" should be "Masoero et al." [P1 -- FIXED]

**File:** `README.md`

The README referred to "Parmigiani Fig 4, 6" and "Parmigiani does this with tSNE", but the paper (reference [4]) has first author Masoero. Parmigiani is third author.

**Fix applied:** Changed all informal references from "Parmigiani" to "Masoero et al." in both the prose and the scripts table.

### F3. ERICA metrics share identical band widths [P2 -- documented]

**File:** `01_error_bands.py`

CRI, WCRI, and TWCRI all use the same std (from `per_sample_cri_from_clam`), so all three bands have identical width. Only the center line differs. This is mathematically correct (the per-sample distribution is the same building block for all three metrics), but could confuse readers.

**Fix applied:** Added explicit comments in the code and updated the module docstring to document this design choice.

### F4. Error band center vs. distribution mean mismatch [P2 -- noted]

**File:** `01_error_bands.py`

The band center uses the stored aggregate CRI (mean of per-cluster CRI averages), but the band std comes from the per-sample CRI distribution (no cluster grouping). For unequal cluster sizes, `mean(per_sample_CRI) != stored_CRI`. The band center thus does not equal the mean of the distribution generating the std. This is a minor visual inconsistency, not a data error.

---

## Round 2 Findings

No additional P0 or P1 issues. Verified Round 1 fixes are correct.

Minor observations (no fix needed):
- **Band clipping:** ERICA bands correctly clip to [0, 1]. Parmigiani bands do not clip, which is correct for ARI (range [-1, 1]).
- **Co-assignment distance:** `1.0 - coassign` is always non-negative (Cauchy-Schwarz guarantees coassign in [0, 1]).
- **Entropy edge case (k=1):** Handled correctly (returns 0).
- **CLAM access inconsistency:** Script 01 uses `er['results'][(k,method)]['clam_matrix']`, scripts 02/03 use `er['clam_matrices'][(k,method)]`. Both keys exist in the data. Not a bug, just cosmetic inconsistency. Script 01 needs `er['results']` for `iteration_labels` anyway.
- **PCA scatter ERICA stat:** Correctly computes `max(row) / sum(row)`, matching the per-sample building block of CRI. Verified with edge cases including zero-sum rows.
- **DOI format:** All 5 citations use valid DOI format. All appear to reference the correct papers.

---

## Files Modified

| File | Change |
|------|--------|
| `02_coassignment_heatmap.py` | Updated module docstring, `compute_coassignment()` docstring, colorbar label, figure title to clarify inner-product vs. Monti semantics |
| `01_error_bands.py` | Updated module docstring and added inline comment explaining shared band widths |
| `README.md` | Fixed "Parmigiani" -> "Masoero et al." attribution (4 locations); updated co-assignment description |

---

## Verdict

**PASS with fixes applied.** The scripts produce correct figures for the quantities they compute. The main subtlety (co-assignment inner product vs. Monti consensus) is now properly documented. No silent data-corrupting bugs were found.
