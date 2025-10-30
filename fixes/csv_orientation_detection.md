# CSV Orientation Detection Fix

## Problem
ERICA was failing to work with CSV files, particularly with the VDX_3_SV.csv file. The issue was related to how the library detected whether samples were in rows or columns.

## Root Cause
The `prepare_samples_array()` function was always transposing data, assuming features were in rows and samples in columns (genomics format). However, many CSV files have samples in rows and features in columns (standard ML format).

## Solution
Added automatic orientation detection with a `transpose` parameter:

### Changes Made

1. **Updated `prepare_samples_array()` in `erica/data.py`**:
   - Added `transpose` parameter with options: `'auto'`, `'yes'`, `'no'`
   - Implemented simple heuristic: if `n_cols > n_rows`, transpose (features in rows)
   - Otherwise, don't transpose (samples in rows)

2. **Updated `ERICA` class in `erica/core.py`**:
   - Added `transpose='auto'` parameter to `__init__()`
   - Passes transpose parameter to `prepare_samples_array()`
   - Added comprehensive documentation

3. **Updated `get_dataset_info()` in `erica/data.py`**:
   - Added `transpose` parameter for consistency

4. **Added comprehensive tests in `tests/test_erica.py`**:
   - `test_csv_loading_samples_in_rows()`: Tests standard format
   - `test_csv_loading_features_in_rows()`: Tests genomics format
   - `test_csv_transpose_parameter()`: Tests explicit control
   - `test_erica_with_csv()`: End-to-end CSV test

5. **Updated documentation in `README.md`**:
   - Added feature bullet point
   - Added Example 1b showing different CSV format handling

## How It Works

### Automatic Detection (`transpose='auto'`)
```python
# Simple rule: if we have more columns than rows, transpose
if n_cols > n_rows:
    should_transpose = True  # Features in rows (genomics format)
else:
    should_transpose = False  # Samples in rows (standard format)
```

### Usage Examples

```python
from erica import ERICA
from erica.data import load_data

# Auto-detect (recommended)
data = load_data('your_data.csv')
erica = ERICA(data=data, transpose='auto')  # Default

# Explicit control
erica = ERICA(data=data, transpose='no')   # Samples in rows
erica = ERICA(data=data, transpose='yes')  # Features in rows
```

## Test Results

### VDX_3_SV.csv
- Input: 344 rows × 4 columns (1 ID column + 3 feature columns)
- After processing: 344 samples × 3 features ✓
- Correctly detected as samples-in-rows format

### All Tests Pass
```
tests/test_erica.py::test_erica_import PASSED
tests/test_erica.py::test_erica_initialization PASSED
tests/test_erica.py::test_erica_run PASSED
tests/test_erica.py::test_erica_metrics PASSED
tests/test_erica.py::test_csv_loading_samples_in_rows PASSED
tests/test_erica.py::test_csv_loading_features_in_rows PASSED
tests/test_erica.py::test_csv_transpose_parameter PASSED
tests/test_erica.py::test_erica_with_csv PASSED
```

## Key Insights

1. **Simple is Better**: Initially tried complex heuristics based on row/column counts and thresholds. The final solution uses a simple rule: more columns than rows = transpose.

2. **pd.read_csv Works Fine**: The issue wasn't with pandas CSV loading, but with the subsequent data orientation detection.

3. **Explicit Control Available**: Users can override auto-detection if needed with `transpose='yes'` or `transpose='no'`.

## Files Modified
- `erica/data.py`: Added transpose parameter and auto-detection logic
- `erica/core.py`: Added transpose parameter to ERICA class
- `tests/test_erica.py`: Added comprehensive CSV tests
- `README.md`: Updated documentation with CSV examples

## Date
October 30, 2025

