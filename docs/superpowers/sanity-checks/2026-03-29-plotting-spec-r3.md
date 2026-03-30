# Sanity Check Report: plotting-experiments-design.md

**Date:** 2026-03-29
**Spec:** `docs/superpowers/specs/2026-03-29-plotting-experiments-design.md`
**Round:** 3 (final)
**Prior rounds fixed:** 8 issues across rounds 1-2

---

## Summary

Round 3 ran three reviewer agents (Adversarial, Pragmatic, Visionary) across three sub-rounds. Found **5 P0 issues** and **7 P1 issues** that rounds 1 and 2 missed. All P0 and most P1 issues have been fixed directly in the spec.

---

## Round 1: Data Integrity and Design Contracts

### ADVERSARIAL AGENT

| # | Sev | Finding | Status |
|---|-----|---------|--------|
| 1 | P0 | **`transpose=False` is a no-op for numpy arrays.** `prepare_samples_array()` (data.py:138-140) returns numeric numpy arrays immediately without ever checking the `transpose` parameter. The transpose logic only applies to DataFrames. Since all datasets in the spec are numpy arrays, the `transpose=False` parameter does nothing. This is semantically correct (data is already samples x features) but the spec was silent about this gotcha. | **FIXED** — added explanatory note in per-dataset config section |
| 2 | P1 | **`setup.py` vs `pyproject.toml` disagree on matplotlib.** Spec line 277 said "matplotlib (already a core dep in pyproject.toml)" — but `setup.py` puts matplotlib only in `extras_require['plots']`, not `install_requires`. If `pip install -e .` uses setup.py, matplotlib is missing. | **FIXED** — changed install command to `pip install -e ".[all]"` and documented the discrepancy |

### PRAGMATIC AGENT

| # | Sev | Finding | Status |
|---|-----|---------|--------|
| 3 | P0 | **`from style import ...` fails when scripts are run from outside `plotting_experiments/`.** Python cannot resolve a bare `import style` unless the script's directory is on `sys.path`. The Makefile doesn't enforce cwd. | **FIXED** — added `sys.path.insert(0, os.path.dirname(__file__))` pattern to the shared import block |
| 4 | P1 | **`save_figure()` assumes `figures/` exists.** No script or Makefile target creates it. | **FIXED** — added note that `save_figure()` must call `os.makedirs('figures', exist_ok=True)` internally |

### VISIONARY AGENT

| # | Sev | Finding | Status |
|---|-----|---------|--------|
| 5 | P0 | **ERICA pollutes working directory with `./erica_output/`.** Every ERICA run creates `./erica_output/erica_run_{timestamp}/` with subdirectories and `.npy` files. The spec never mentioned setting `output_dir`. These artifacts would accumulate outside the gitignored `results/` directory. | **FIXED** — added instruction to set `output_dir='results/erica_workdir'` in every ERICA constructor call, and documented this in the gitignore note |

---

## Round 2: Error Handling and Integration

### ADVERSARIAL AGENT

| # | Sev | Finding | Status |
|---|-----|---------|--------|
| 6 | P0 | **Spec said `modal_k=0` for all-noise scenarios. Source code says `modal_k=1`.** In `clustering.py:663`, when `k_counts` is empty, `modal_k` is set to 1 (not 0), `k_agreement_rate=0.0`, and `clam_matrix` is `zeros((n_samples, 1))`. The spec's edge case at line 240 ("Some param combos may yield `modal_k=0`") was wrong. | **FIXED** — corrected to document the actual behavior: `modal_k=1`, `k_agreement_rate=0.0`, empty `k_distribution` |

### PRAGMATIC AGENT

| # | Sev | Finding | Status |
|---|-----|---------|--------|
| 7 | P1 | **HDBSCAN sweep dummy `k_range=[2]` still validated.** `validate_dataset(self.samples_array, min(self.k_range), self.train_percent)` is always called in `__init__`. With 344 samples and `k_range=[2]`, this passes fine, but it's a non-obvious constraint worth documenting. | **FIXED** — added clarifying note |
| 8 | P1 | **Makefile working directory ambiguity.** The Makefile uses bare `python script.py` without `cd` and the `from style import ...` depends on cwd. | **FIXED** — the `sys.path.insert` fix from finding #3 also resolves this; Makefile targets implicitly run from the Makefile's directory |

### VISIONARY AGENT

| # | Sev | Finding | Status |
|---|-----|---------|--------|
| 9 | P2 | **No `__init__.py` needed.** `plotting_experiments/` is not a Python package; scripts import `style` as a module from the same directory. The `sys.path.insert` fix handles this. No `__init__.py` needed. | N/A |

---

## Round 3: Edge Cases in Plot Scripts

### ADVERSARIAL AGENT

| # | Sev | Finding | Status |
|---|-----|---------|--------|
| 10 | P1 | **All-zero CLAM rows cause div-by-zero in stability strips.** If a sample was never in any test set, its CLAM row sums to 0. Normalizing to proportions divides by zero. Unlikely with 100 iterations (each sample is in test ~20 times) but possible. | **FIXED** — added edge case note to stability strips section |
| 11 | P1 | **NaN metrics create silent gaps in matplotlib `errorbar`.** When a K is disqualified, CRI/WCRI/TWCRI are `float('nan')`. Matplotlib skips NaN y-values silently, producing unexplained gaps. | **FIXED** — added note to use explicit 'x' markers or grayed bands for disqualified K |

### PRAGMATIC AGENT

| # | Sev | Finding | Status |
|---|-----|---------|--------|
| 12 | P1 | **Overlapping K* lines when all metrics agree.** If K* is identical for CRI, WCRI, and TWCRI, three dashed vertical lines stack on top of each other. | **FIXED** — added suggestion for x-offset or distinct dash patterns |
| 13 | P1 | **`metrics_at_modal_k` is meaningless when `k_agreement_rate=0.0`.** The code sets `modal_k=1` and computes metrics on a zero CLAM matrix. Plot 06 should guard against this. | **FIXED** — added check `k_agreement_rate > 0` before using HDBSCAN metrics in method comparison |

### VISIONARY AGENT

| # | Sev | Finding | Status |
|---|-----|---------|--------|
| 14 | P2 | **DPI spec says "300 for PDF" but PDF is vector.** DPI is irrelevant for vector PDF output; it only affects rasterized elements within PDF. Technically correct since matplotlib's PDF backend uses DPI for rasterized fallback, but worth noting this is a minor conceptual inaccuracy. | Not fixed (cosmetic) |

---

## Convergence Assessment

After 3 rounds, the remaining unfixed items are P2 cosmetic issues. The spec is now consistent with the actual source code on all material points:

- `get_results()` key structure verified against `core.py:356-365`
- `metrics` nesting `{k: {method: {metric_name: value}}}` verified against `core.py:228-293`
- `k_star` nesting `{metric_name: {method: k_value}}` verified against `core.py:296-306`
- `auto_k` key (not `auto_k_results`) verified against `core.py:361`
- HDBSCAN result dict keys verified against `clustering.py:700-712`
- `metrics_at_modal_k` conditional assignment verified against `core.py:287-291`
- `iteration_labels` structure (`{'predicted': [...], 'true': [...]}`) verified against `clustering.py:336-339, 706-709`
- VDX CSV format (no header, pure float) verified against actual file content
- matplotlib dependency status verified against both `setup.py` and `pyproject.toml`

**VERDICT: CONVERGED.** No further rounds needed.
