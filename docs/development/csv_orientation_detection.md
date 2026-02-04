# CSV Transpose Parameter Fix

## Problem
ERICA was failing to work with CSV files that had samples in rows (standard ML format), like VDX_3_SV.csv. The library was always transposing data to match the genomics format used in `.npy` files.

## Root Cause
The `prepare_samples_array()` function was always transposing data without user control. While this worked for `.npy` files (which bypass this logic), CSV files needed explicit control over orientation.

## Solution
Added explicit `transpose` parameter with helpful error messages instead of auto-detection:

### Changes Made

1. **Updated `prepare_samples_array()` in `erica/data.py`**:
   - Added `transpose` boolean parameter (default: `True` to match existing `.npy` workflow)
   - `True`: Features in rows, samples in columns (genomics format - default)
   - `False`: Samples in rows, features in columns (standard ML format)

2. **Updated `ERICA` class in `erica/core.py`**:
   - Added `transpose=True` parameter to `__init__()` (default matches genomics format)
   - Passes transpose parameter to `prepare_samples_array()`
   - Added comprehensive documentation

3. **Updated `get_dataset_info()` in `erica/data.py`**:
   - Added `transpose=True` parameter for consistency

4. **Added helpful error messages in `validate_dataset()`**:
   - When data has too few samples but many features, suggests trying `transpose=False`
   - Helps users quickly identify orientation issues

5. **Added comprehensive tests in `tests/test_erica.py`**:
   - `test_csv_loading_samples_in_rows()`: Tests standard format
   - `test_csv_loading_features_in_rows()`: Tests genomics format
   - `test_csv_transpose_parameter()`: Tests explicit control
   - `test_erica_with_csv()`: End-to-end CSV test

6. **Updated documentation in `README.md`**:
   - Added feature bullet point
   - Added Example 1b showing different CSV format handling
   - Added note about transpose parameter

## How It Works

### Simple Explicit Control
```python
# Default: genomics format (matches .npy files)
erica = ERICA(data=data, transpose=True)  # Features in rows → transpose

# Standard ML format
erica = ERICA(data=data, transpose=False)  # Samples in rows → don't transpose
```

### Usage Examples

```python
from erica import ERICA
from erica.data import load_data

# For genomics data (features in rows) - DEFAULT
data = load_data('gene_expression.csv')
erica = ERICA(data=data, transpose=True)  # or just erica = ERICA(data=data)

# For standard ML data (samples in rows)
data = load_data('samples.csv')
erica = ERICA(data=data, transpose=False)
```

## Test Results

### VDX_3_SV.csv (Samples in Rows)
- Input: 344 rows × 4 columns (1 ID column + 3 gene columns)
- With `transpose=False`: 344 samples × 3 features ✓
- With `transpose=True`: 3 samples × 344 features (would fail for k>3)

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

1. **Explicit is Better Than Implicit**: After discussion with collaborator, removed auto-detection in favor of explicit `transpose` parameter. Users know their data format better than heuristics can guess.

2. **Default Matches Existing Workflow**: Set `transpose=True` as default to match the genomics format used in existing `.npy` files.

3. **Helpful Error Messages**: When validation fails due to insufficient samples, error message suggests trying `transpose=False` if the data shape indicates possible orientation issue.

4. **`.npy` Files Unaffected**: NumPy arrays bypass all this logic and are used as-is, maintaining backward compatibility.

## Files Modified
- `erica/data.py`: Added transpose parameter and auto-detection logic
- `erica/core.py`: Added transpose parameter to ERICA class
- `tests/test_erica.py`: Added comprehensive CSV tests
- `README.md`: Updated documentation with CSV examples

## Date
October 30, 2025

