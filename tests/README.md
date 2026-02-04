# ERICA Test Suite

This folder contains the test suite for ERICA.

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ -v --cov=erica --cov-report=xml

# Run a specific test
pytest tests/test_erica.py::test_erica_run -v

# Run tests matching a pattern
pytest tests/ -v -k "k_star"
```

## Test Organization

| Test File | Coverage |
|-----------|----------|
| `test_erica.py` | Core ERICA class, metrics, K* selection, empty cluster handling |

## Test Categories

The test suite covers:

1. **Core Functionality**
   - `test_erica_import` - Package imports correctly
   - `test_erica_initialization` - ERICA class initializes properly
   - `test_erica_run` - Full analysis completes

2. **Data Loading**
   - `test_csv_loading_samples_in_rows` - Standard ML format (samples x features)
   - `test_csv_loading_features_in_rows` - Genomics format (features x samples)
   - `test_csv_transpose_parameter` - Transpose parameter works correctly

3. **Metrics Computation**
   - `test_erica_metrics` - CRI, WCRI, TWCRI computed correctly
   - `test_empty_cluster_disqualification` - Empty clusters marked as NaN
   - `test_no_empty_cluster_averaging` - Non-empty clusters averaged correctly

4. **K* Selection (Algorithm 2)**
   - `test_select_optimal_k_basic` - Basic K* selection
   - `test_select_optimal_k_with_nan` - NaN handling
   - `test_k_star_skips_empty_clusters` - Empty clusters skipped

## Adding New Tests

When adding new tests:

1. Follow the naming convention: `test_<feature_name>`
2. Use descriptive test function names
3. Include docstrings explaining what the test verifies
4. Use temporary files for any file I/O tests

Example:
```python
def test_new_feature():
    """Test that new_feature works correctly."""
    # Setup
    data = np.random.rand(50, 10)

    # Execute
    result = new_feature(data)

    # Assert
    assert result is not None
```

## Requirements

```bash
pip install pytest pytest-cov
```
