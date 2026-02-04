# K* (Optimal K) Selection Implementation

## Summary

Successfully implemented **Algorithm 2** for automatic optimal K selection in ERICA. This feature provides a principled approach to determining the best number of clusters based on replicability metrics.

## Implementation Date
November 2, 2025

## What Was Implemented

### 1. Core Algorithm (`erica/metrics.py`)

#### `select_optimal_k(metric_dict, k_max=None)`
- Implements Algorithm 2 for K* selection
- Handles NaN values gracefully (for failed clustering runs)
- Supports non-consecutive K values
- Tracks most recent valid metric for comparison
- Returns the largest K where metrics are non-decreasing

**Algorithm Logic:**
```
1. Initialize K_star = min(K_values)
2. For each K from min+1 to K_max:
    a. If M[K] is valid (not NaN):
        i. If M[K] >= M[previous_valid_K], set K_star = K
        ii. Update previous_valid_K = K
3. Return K_star
```

#### `select_optimal_k_by_method(metrics_by_k, metric_name='TWCRI')`
- Wrapper function for multiple clustering methods
- Automatically organizes metrics by method
- Returns K* for each method independently
- Supports all three metrics: CRI, WCRI, TWCRI

### 2. ERICA Core Integration (`erica/core.py`)

#### New Attributes:
- `self.k_star_`: Dictionary storing K* for each metric and method

#### New Methods:
- `_select_optimal_k()`: Computes K* for all metrics after clustering
- `_print_k_star_summary()`: Displays K* results in formatted output
- `get_k_star(metric='TWCRI')`: Public API to retrieve K* values

#### Modified Workflow:
```
[1/3] Iterative clustering subsampling
[2/3] Running clustering analysis
[3/3] Computing metrics (CRI, WCRI, TWCRI)
[4/4] Selecting optimal K* using Algorithm 2  ← NEW STEP
```

### 3. Visualization Functions (`erica/plotting.py`)

#### `plot_k_star_selection(metric_dict, k_star, metric_name, method_name, title)`
- Creates interactive plot showing metric values across K
- Highlights K* with a red star marker
- Adds vertical dashed line at K*
- Displays metric values on hover

#### `plot_k_star_by_method(k_star_by_method, metric_name, title)`
- Bar chart comparing K* across different clustering methods
- Color-coded bars for visual distinction
- Shows K* value on each bar

### 4. Comprehensive Tests (`tests/test_erica.py`)

Added 15 new test cases covering:

**Algorithm Tests:**
- `test_select_optimal_k_basic`: Example from specification
- `test_select_optimal_k_monotonic_increasing`: All increasing metrics
- `test_select_optimal_k_monotonic_decreasing`: All decreasing metrics
- `test_select_optimal_k_with_nan`: Handling NaN values
- `test_select_optimal_k_all_nan`: Edge case with mostly NaN
- `test_select_optimal_k_with_gaps`: Non-consecutive K values
- `test_select_optimal_k_equal_values`: Tied metric values
- `test_select_optimal_k_empty_dict`: Error handling
- `test_select_optimal_k_single_value`: Single K case
- `test_select_optimal_k_with_kmax`: K_max parameter

**Multi-Method Tests:**
- `test_select_optimal_k_by_method`: Single method
- `test_select_optimal_k_by_method_multiple_methods`: Multiple methods
- `test_select_optimal_k_by_method_invalid_metric`: Error handling

**Integration Tests:**
- `test_erica_k_star_integration`: End-to-end with ERICA
- `test_erica_k_star_multiple_methods`: Multiple clustering methods

**Test Results:** ✅ All 23 tests pass (100% success rate)

### 5. Example Script (`examples/k_star_example.py`)

Comprehensive example demonstrating:
- Running ERICA with multiple K values
- Automatic K* computation
- Accessing K* programmatically
- Standalone K* selection
- Detailed metrics comparison
- Optional visualization
- Interpretation guide

### 6. Documentation Updates

#### `docs/METHODOLOGY.md`
- Added K* selection as Step 5 in ERICA process
- Documented Algorithm 2 with pseudocode
- Explained key properties and rationale
- Provided usage examples
- Positioned as recommended Method 1 for K selection

#### `README.md`
- Added K* selection to features list
- Updated quick start example to show K* usage
- Highlighted automatic computation

### 7. API Exports (`erica/__init__.py`)

Added to public API:
- `select_optimal_k`
- `select_optimal_k_by_method`
- `plot_k_star_selection`
- `plot_k_star_by_method`

## Key Design Decisions

### 1. Automatic Computation
K* is computed automatically during `erica.run()` for all three metrics (CRI, WCRI, TWCRI). This ensures users always have access to optimal K recommendations without extra steps.

### 2. Tracking Last Valid Metric
The algorithm tracks the most recent valid (non-NaN) metric value for comparison. This handles:
- NaN values from failed clustering
- Non-consecutive K values (e.g., [2, 4, 6, 8])
- Gaps in the K range

### 3. Non-Decreasing Criterion
Uses `>=` instead of `>` to prefer larger K when metrics are equal. This aligns with the principle of selecting the most granular stable clustering.

### 4. Per-Method Selection
K* is computed independently for each clustering method (kmeans, agglomerative_ward, etc.). Different methods may have different optimal K values.

### 5. Per-Metric Selection
K* is computed for all three metrics (CRI, WCRI, TWCRI). Users can choose which metric to use based on their priorities:
- CRI: Pure replicability
- WCRI: Balanced replicability
- TWCRI: Overall stability (recommended)

## Usage Examples

### Basic Usage
```python
from erica import ERICA

erica = ERICA(data, k_range=[2, 3, 4, 5, 6])
results = erica.run()

# K* is automatically computed and displayed
# Access programmatically:
k_star_twcri = erica.get_k_star('TWCRI')
print(f"Optimal K = {k_star_twcri['kmeans']}")
```

### Standalone Usage
```python
from erica.metrics import select_optimal_k

M = {2: 0.71, 3: 0.75, 4: 0.74, 5: float('nan'), 6: 0.78}
k_star = select_optimal_k(M)
print(f"K* = {k_star}")  # Output: K* = 6
```

### Visualization
```python
from erica.plotting import plot_k_star_selection

fig = plot_k_star_selection(
    metric_dict={2: 0.71, 3: 0.75, 4: 0.74, 6: 0.78},
    k_star=6,
    metric_name='TWCRI',
    method_name='kmeans'
)
fig.show()
```

## Testing Coverage

- **Total Tests:** 23 (15 new, 8 existing)
- **Pass Rate:** 100%
- **Code Coverage:** 60% overall
  - `erica/core.py`: 94%
  - `erica/clustering.py`: 89%
  - `erica/metrics.py`: 66%

## Example Output

```
============================================================
K* SELECTION SUMMARY (Algorithm 2)
============================================================

CRI:
  kmeans                    -> K* = 3  (CRI = 1.000000)

WCRI:
  kmeans                    -> K* = 2  (WCRI = 0.432819)

TWCRI:
  kmeans                    -> K* = 3  (TWCRI = 1.000000)
============================================================
```

## Files Modified

1. `erica/metrics.py` - Added K* selection functions
2. `erica/plotting.py` - Added K* visualization functions
3. `erica/core.py` - Integrated K* into ERICA workflow
4. `erica/__init__.py` - Exported new functions
5. `tests/test_erica.py` - Added comprehensive tests
6. `examples/k_star_example.py` - Created example script
7. `docs/METHODOLOGY.md` - Updated documentation
8. `README.md` - Updated features and examples

## Validation

The implementation was validated using synthetic data with 3 known clusters:
- Algorithm correctly identified K* = 3 for both CRI and TWCRI
- WCRI selected K* = 2 (expected due to size weighting)
- All edge cases handled correctly (NaN, gaps, equal values)

## Recent Enhancements (November 6, 2025)

### 1. CoLab Integration
Added `find_largest_increasing_entry_by_index()` function from the CoLab notebook:
- Finds the last position where a metric value increased
- Provides alternative approach to K* selection
- Validates consistency with Algorithm 2 implementation
- Useful for research and comparative analysis

### 2. Improved NA Handling
Enhanced `select_optimal_k()` to explicitly follow Algorithm 2, Line 4:
- Line 4: "if NA ∉ {M_K(k)} then % is violated if ∃ k ≥ 1 : X_k = 0"
- Added explicit comments mapping code to algorithm steps
- Improved documentation to reference paper algorithm directly
- Better handling of edge cases (NaN at start, multiple NaN, etc.)

### 3. Validation
Created comprehensive test suite validating:
- `find_largest_increasing_entry_by_index` correctness
- NA handling per Algorithm 2, Line 4
- Consistency between both approaches
- 10 test cases covering all edge cases
- All tests pass ✓

## Future Enhancements

Potential improvements for future versions:
1. Add confidence intervals for K* selection
2. Support for custom metric functions
3. Statistical significance testing between K values
4. Ensemble K* selection across multiple metrics
5. Alternative K* selection using `find_largest_increasing_entry_by_index`

## References

- Algorithm 2: ERICA Cluster Number Selection
- Based on non-decreasing metric criterion
- Prefers larger K when stability is maintained

