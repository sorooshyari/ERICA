# Empty Cluster Handling Fix

## Problem Description

When running ERICA with certain K values, some clusters could end up empty (size = 0). This occurred when the clustering algorithm failed to assign any samples to a particular cluster. The original implementation had two issues:

1. **Incorrect averaging**: CRI was computed by averaging over ALL k clusters, including empty ones (which had CRI = 0.0), leading to artificially low scores
2. **No disqualification**: K values with empty clusters were not automatically disqualified from K* selection, even though they violate the assumptions of Algorithm 2

### Example Case (K=8)

```
CRI_per_cluster: [0.537, 0.639, 0.469, 0.404, 0.520, 0.444, 0.643, 0.0]
cluster_sizes: [73, 49, 44, 16, 62, 51, 49, 0]

OLD BEHAVIOR:
  CRI = sum(all 8 values) / 8 = 0.457  ❌ WRONG
  K=8 could still be selected as K*

CORRECT BEHAVIOR:
  CRI = sum(7 non-empty values) / 7 = 0.522  ✓ (if no empty clusters)
  OR
  CRI = NaN (disqualify this K entirely)  ✓ (preferred solution)
```

## Root Cause

The issue stems from Algorithm 2, Line 4 in the ERICA paper:

```
if NA ∉ {M_K(k)} then  % is violated if ∃ k ≥ 1 : X_k = 0
```

This means: "If there exists any cluster k with size X_k = 0, the metric should be marked as NA (not available), and this K value should be skipped in K* selection."

The original implementation did not check for empty clusters, allowing invalid K values to be considered.

## Solution

Modified `erica/metrics.py::compute_metrics_for_clam()` to:

1. **Detect empty clusters**: Check if any cluster has size 0
2. **Mark as NaN**: If empty clusters exist, set all metrics (CRI, WCRI, TWCRI) to `float('nan')`
3. **Add flag**: Include `has_empty_clusters` boolean in the returned metrics dictionary
4. **Automatic disqualification**: The existing `select_optimal_k()` function already skips NaN values per Algorithm 2, Line 4

### Code Changes

```python
# Check for empty clusters (violates Algorithm 2, Line 4: ∃ k ≥ 1 : X_k = 0)
has_empty_clusters = np.any(cluster_sizes == 0)

# If there are empty clusters, mark all metrics as NaN to disqualify this K
if has_empty_clusters:
    cri_mean = float('nan')
    wcri_mean = float('nan')
    twcri_final = float('nan')
else:
    # Only average non-empty clusters for CRI
    non_empty_mask = cluster_sizes > 0
    cri_mean = float(np.mean(cri_values[non_empty_mask]))
    wcri_mean = float(mean_wcri)
    twcri_final = float(twcri_value)
```

## Testing

Added three comprehensive tests in `tests/test_erica.py`:

1. **`test_empty_cluster_disqualification`**: Verifies that K values with empty clusters are marked as NaN
2. **`test_no_empty_cluster_averaging`**: Ensures normal cases still work correctly
3. **`test_k_star_skips_empty_clusters`**: Confirms K* selection skips NaN values

All tests pass successfully.

## Impact

### Before Fix
- Empty clusters artificially lowered CRI scores
- Invalid K values could be selected as K*
- Results were misleading when clustering produced empty clusters
- No way to identify which K values were problematic

### After Fix
- Empty clusters are automatically detected
- K values with empty clusters are disqualified (marked as NaN)
- K* selection only considers valid K values
- Clear warning message in output: "NaN (DISQUALIFIED - empty cluster detected)"
- **NEW**: Disqualified K values are tracked and accessible via `get_disqualified_k()`
- **NEW**: Disqualified K values shown in K* selection summary
- **NEW**: Included in results dictionary as `disqualified_k`

## Related Files

- `erica/metrics.py`: Main fix implementation (NaN marking)
- `erica/core.py`: Tracking and reporting of disqualified K values
- `tests/test_erica.py`: Added comprehensive tests
- `README.md`: Updated documentation with usage examples

## Algorithm 2 Compliance

This fix ensures full compliance with Algorithm 2, Line 4 from the ERICA paper:

```
Algorithm 2: Cluster number (K*) selection with ERICA
1. Input: Metric for considered K values {M_K : K = 2, ..., K^max}
2. K* ← 2  (initialize)
3. for K = 3 to K^max do
4.     if NA ∉ {M_K(k)} then  % is violated if ∃ k ≥ 1 : X_k = 0  ← THIS LINE
5.         if M_K ≥ M_{K-1} then
6.             K* ← K
7.         end if
8.     end if
9. end for
10. return K*
```

The condition "is violated if ∃ k ≥ 1 : X_k = 0" explicitly states that any K value with an empty cluster should be treated as NA and skipped.

## Verification

To verify the fix works correctly, run:

```bash
# Run specific tests
pytest tests/test_erica.py::test_empty_cluster_disqualification -v
pytest tests/test_erica.py::test_k_star_skips_empty_clusters -v

# Run demonstration
python test_empty_cluster_example.py

# Run full test suite
pytest tests/test_erica.py -v
```

All tests should pass, and the demonstration should show clear NaN marking for empty cluster cases.

