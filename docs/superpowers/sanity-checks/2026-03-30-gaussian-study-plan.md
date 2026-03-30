# Sanity Check Report: Gaussian Mixture Replicability Study

**Target:** `plotting_experiments/gaussian_mixture_study/EXPERIMENT_PLAN.md`
**Mode:** Plan review
**Rounds:** 3 (converged at round 3 -- all remaining findings P2/marginal)
**Agents:** Adversarial, Pragmatic, Visionary

---

## P0 Issues Found and Fixed (3)

### 1. Dimension mismatch: Plan said 20D, actual data is 100D
- **Finding:** The plan stated "4 Gaussian components in 20 dimensions" and "Dimensions: 20", but both existing `.npz` files (`gauss4c_sigma0p01.npz`, `gauss4c_sigma0p1.npz`) contain 100-dimensional data with `n_features=100` in metadata.
- **Fix:** Updated all references from 20D to 100D. Updated covariance from `sigma^2 * I_20` to `sigma^2 * I_100`. Updated center placement from "dimensions 10-19" to "dimensions 10-99".

### 2. K range mismatch: Plan said K=2..7, existing results use K=2..6
- **Finding:** Plan specified `k_range=[2,3,4,5,6,7]` but the existing `gauss4c_sigma0p01.joblib` was run with `k_range=[2,3,4,5,6]`. The existing pipeline script (`02_run_erica_pipeline.py`) also uses K ranges topping at 5 or 6.
- **Fix:** Changed K range to `[2,3,4,5,6]` throughout. Updated grid dimensions from "4-row x 6-column" to "3-row x 5-column". Updated section header from "K=2..7" to "K=2..6".

### 3. hdbscan_params mismatch: Plan said min_cluster_size=20, existing results used 15
- **Finding:** Plan specified `min_cluster_size=20` with justification "following min_K = dim recommendation", but existing results used `min_cluster_size=15`. With 100D data and 80 samples per cluster in the train split, setting min_cluster_size to dim=100 would exceed cluster sizes.
- **Fix:** Changed to `min_cluster_size=15` to match existing runs. Updated justification.

## P1 Issues Found and Fixed (7)

### 4. Files incorrectly marked as "New"
- `gauss4c_sigma0p01.npz` and `gauss4c_sigma0p1.npz` already exist in `data/`. `gauss4c_sigma0p01.joblib` already exists in `results/`. File tree updated to reflect actual status.

### 5. Dataset naming inconsistency for sigma=1.0 and sigma=10.0
- Plan used `gauss4c_sigma1` and `gauss4c_sigma10` but the established naming convention uses `p` for decimal points (e.g., `sigma0p1`). Changed to `gauss4c_sigma1p0` and `gauss4c_sigma10p0` for consistency.

### 6. Separation math / Expected K* wrong for 100D
- In 100D (not 20D), sigma=1.0 gives inter/intra ratio of 0.95 (severe overlap, not "moderate"). Updated Expected K* from "4 (possible) or 3" to "3 or 2". Added explicit separation math section explaining the ratio.

### 7. HDBSCAN grid layout clarified
- HDBSCAN does not produce K-indexed CLAM matrices (it's auto-K). Clarified that HDBSCAN is shown as a separate single-panel below each grid at its modal K, not as a grid row.

### 8. HDBSCAN noise fraction analysis added
- Added section 2b: "HDBSCAN Noise Fraction vs Sigma" plot using `noise_counts` from auto-K results. This is critical for understanding HDBSCAN behavior at high sigma.

### 9. Replicability curve now includes HDBSCAN and AMI
- HDBSCAN shown as discrete markers (not lines) since it doesn't operate at fixed K. AMI included as supplemental panel alongside ARI.

### 10. VDX comparison updated for 100D overlap reality
- At sigma=1.0, overlap is severe (not "moderate"). Updated comparison to use both sigma=0.1 and sigma=1.0 as bracketing reference points for VDX. Added empty cluster handling note.

## P2 Issues Noted (not fixed -- informational)

- **Random seed clarification:** Data generation uses seed=42; ERICA uses seed=123. Plan now documents this explicitly. Both are correct and intentional.
- **Runtime estimate:** 4 datasets x full ERICA pipeline at 200 iterations will take approximately 40-60 minutes total. Not added to plan but worth knowing.
- **Existing `gmm_sweep` data:** The `data/` directory contains a parallel GMM sweep study (`gmm_4c_100d_sigma*.joblib`). These could provide additional reference points but are out of scope for this plan.
- **K* at sigma=10.0:** With complete overlap, K* algorithm will select K=2 because all TWCRI values will be uniformly low and the non-decreasing criterion stops at the smallest K. This is expected behavior.
- **Single linkage empty clusters:** At high sigma with K=5,6, single linkage will likely produce empty clusters, disqualifying those K values. This is now documented in the plan.

## Verification Summary

| Aspect | Status |
|--------|--------|
| Dimensions match actual data | FIXED (100D) |
| K range matches existing results | FIXED ([2..6]) |
| HDBSCAN params match existing results | FIXED (min_cluster_size=15) |
| File naming consistent | FIXED (sigma1p0, sigma10p0) |
| Separation math correct for 100D | FIXED (added explicit calculation) |
| Analysis plan complete | ENHANCED (added noise fraction, AMI, dual VDX comparison) |
| File organization matches existing scripts | VERIFIED (shared data/results dirs, gitignored) |
| ERICA API calls match actual code | VERIFIED (checked against core.py and clustering.py) |
| HDBSCAN return structure handled correctly | FIXED (clarified grid layout, added noise_counts) |
