# ERICA PyPI Library Guide

## Overview

You've successfully transformed your ERICA Gradio application into a well-structured Python library ready for PyPI publication! This guide explains the structure and how to use it.

## What We Built

### Library Structure

```
ERICA_PyPI/
├── erica/                      # Main package directory
│   ├── __init__.py            # Package interface (imports all main functions)
│   ├── core.py                # ERICA main class
│   ├── clustering.py          # All clustering functions
│   ├── metrics.py             # CRI, WCRI, TWCRI metrics
│   ├── data.py                # Data loading and validation
│   ├── plotting.py            # Visualization functions
│   └── utils.py               # Utility functions
├── docs/                       # Documentation
│   ├── API_REFERENCE.md       # Complete API documentation
│   ├── GETTING_STARTED.md     # Tutorial and quick start
│   └── METHODOLOGY.md         # Scientific methodology
├── tests/                      # Unit tests (empty, ready for tests)
├── examples/                   # Usage examples (empty, ready for examples)
├── setup.py                    # Installation configuration
├── pyproject.toml             # Modern Python packaging
├── README.md                   # Main package documentation
├── LICENSE                     # MIT License
├── requirements.txt           # Core dependencies
└── MANIFEST.in                 # Package manifest

```

## How Everything Works Together

### 1. The Library Design

The library follows a **modular design** where:

- **Core functionality** is separated into focused modules
- **Each module** has a specific purpose
- **Functions are well-documented** with docstrings
- **The main class (ERICA)** orchestrates everything

### 2. Module Responsibilities

#### `core.py` - ERICA Class
- **Purpose**: High-level API for complete analysis
- **What it does**: Coordinates all other modules
- **Usage**: `from erica import ERICA`

#### `clustering.py` - Clustering Operations
- **Purpose**: All clustering-related functions
- **Contains**:
  - `iterative_clustering_subsampling()` - Monte Carlo subsampling
  - `kmeans_clustering()` - K-Means with ERICA
  - `agglomerative_clustering()` - Hierarchical clustering
  - Helper functions for alignment and CLAM generation
- **Usage**: `from erica.clustering import kmeans_clustering`

#### `metrics.py` - Replicability Metrics
- **Purpose**: Compute all replicability metrics
- **Contains**:
  - `compute_cri()` - Clustering Replicability Index
  - `compute_wcri()` - Weighted CRI
  - `compute_twcri()` - Total Weighted CRI
  - `select_optimal_k()` - Find K* using Algorithm 2
  - `select_optimal_k_by_method()` - K* selection for multiple methods
- **Usage**: `from erica.metrics import compute_cri`

#### `data.py` - Data Operations
- **Purpose**: Load, validate, and prepare data
- **Contains**:
  - `load_data()` - Load from .npy or .csv
  - `prepare_samples_array()` - Convert to numeric array
  - `validate_dataset()` - Check data meets requirements
  - `save_clam_matrix()` / `load_clam_matrix()` - CLAM I/O
- **Usage**: `from erica.data import load_data`

#### `plotting.py` - Visualization
- **Purpose**: Create interactive plots
- **Contains**:
  - `plot_metrics()` - Plot CRI, WCRI, TWCRI
  - `plot_optimal_k()` - Bar chart of optimal k
  - `plot_clam_heatmap()` - CLAM matrix visualization
  - `create_metrics_plots()` - Comprehensive plots
- **Usage**: `from erica.plotting import plot_metrics`

#### `utils.py` - Utilities
- **Purpose**: Helper functions
- **Contains**:
  - `set_deterministic_mode()` - Reproducibility setup
  - `compute_config_hash()` - Configuration fingerprinting
  - `check_dependencies()` - Verify installations
  - `validate_config()` - Parameter validation
- **Usage**: `from erica.utils import set_deterministic_mode`

## Usage Patterns

### Pattern 1: Simple High-Level API

```python
from erica import ERICA

# One-liner setup and run
erica = ERICA(data=my_data, k_range=[2, 3, 4])
results = erica.run()

# Get everything you need
clam = erica.get_clam_matrix(k=3)
metrics = erica.get_metrics()
fig1, fig2 = erica.plot_metrics()
```

**When to use**: Most common use case, you want ERICA to handle everything.

### Pattern 2: Modular Components

```python
from erica.clustering import iterative_clustering_subsampling, kmeans_clustering
from erica.metrics import compute_metrics_for_clam
from erica.plotting import plot_metrics

# Step by step with full control
subsamples, indices = iterative_clustering_subsampling(...)
result = kmeans_clustering(...)
metrics = compute_metrics_for_clam(result['clam_matrix'], k=3)
fig = plot_metrics(...)
```

**When to use**: You need fine control, custom workflows, or integration with other tools.

### Pattern 3: Data-Focused

```python
from erica import load_data, prepare_samples_array, validate_dataset
from erica.data import get_dataset_info

# Load and inspect
data = load_data('my_data.csv')
info = get_dataset_info(data)
print(f"Loaded {info['n_samples']} samples with {info['n_features']} features")

# Prepare and validate
samples = prepare_samples_array(data)
validate_dataset(samples, min_k=2, train_percent=0.8)
```

**When to use**: Data exploration, validation, or preprocessing.

### Pattern 4: Metrics-Only

```python
from erica import load_clam_matrix
from erica.metrics import compute_metrics_for_clam, select_optimal_k

# Load pre-computed CLAM matrices
clam_k2 = load_clam_matrix('clam_k2.csv')
clam_k3 = load_clam_matrix('clam_k3.csv')

# Compute metrics
metrics_k2 = compute_metrics_for_clam(clam_k2, k=2)
metrics_k3 = compute_metrics_for_clam(clam_k3, k=3)

# Find optimal K using Algorithm 2
twcri_dict = {2: metrics_k2['TWCRI'], 3: metrics_k3['TWCRI']}
optimal_k = select_optimal_k(twcri_dict)
```

**When to use**: You already ran clustering, just need metrics.

## Key Design Decisions

### 1. Why Separate Modules?

- **Maintainability**: Easy to find and fix bugs
- **Testability**: Each module can be tested independently
- **Reusability**: Use only what you need
- **Clarity**: Clear responsibility for each module

### 2. Why Both High-Level and Low-Level APIs?

- **High-level (ERICA class)**: For most users, simple and convenient
- **Low-level (individual functions)**: For advanced users, custom workflows, integration

### 3. Optional Dependencies

```python
# Core (always installed)
import numpy, pandas, sklearn

# Optional (install with [plots])
import plotly, matplotlib

# Optional (install with [gui])
import gradio
```

This lets users install only what they need!

### 4. All Functions are Library Functions

**Important**: All functions in the library are designed to be:
- **Importable**: Can be used from other Python code
- **Well-documented**: Complete docstrings with examples
- **Type-hinted**: Where possible (Python 3.8+ compatible)
- **Tested**: Ready for unit tests
- **Reproducible**: Deterministic when seed is set

## How This Differs from the Gradio App

### Original Gradio App
- **Monolithic**: Everything in one file
- **UI-focused**: Designed for interactive web use
- **Progress tracking**: Global state for UI updates
- **Logging**: Verbose output for user feedback

### PyPI Library
- **Modular**: Separated concerns
- **API-focused**: Designed for programmatic use
- **No global state**: Pure functions where possible
- **Minimal logging**: Only when `verbose=True`

## Next Steps

### 1. Add the Gradio UI (Optional)

You can create `erica/gui.py` that imports from the library:

```python
# erica/gui.py
import gradio as gr
from erica import ERICA
from erica.data import load_data

def launch():
    def run_analysis(files, k_min, k_max, n_iterations):
        data = load_data(files[0])
        erica = ERICA(
            data=data,
            k_range=list(range(k_min, k_max+1)),
            n_iterations=n_iterations
        )
        results = erica.run()
        return str(results)
    
    interface = gr.Interface(
        fn=run_analysis,
        inputs=[
            gr.File(label="Upload Data"),
            gr.Number(label="Min K", value=2),
            gr.Number(label="Max K", value=5),
            gr.Number(label="Iterations", value=200)
        ],
        outputs="text"
    )
    interface.launch()

if __name__ == "__main__":
    launch()
```

### 2. Add Tests

Create tests in `tests/` directory:

```python
# tests/test_clustering.py
import numpy as np
from erica.clustering import iterative_clustering_subsampling

def test_subsampling():
    data = np.random.rand(100, 50)
    _, indices_folder = iterative_clustering_subsampling(
        data, 100, 10, 80, './test_output'
    )
    assert os.path.exists(indices_folder)
```

### 3. Add Examples

Create examples in `examples/` directory:

```python
# examples/basic_usage.py
import numpy as np
from erica import ERICA

data = np.random.rand(100, 50)
erica = ERICA(data=data, k_range=[2, 3, 4])
results = erica.run()
print(results)
```

### 4. Publish to PyPI

Follow the steps in `PYPI_GUIDE.md`:

```bash
# Build the package
python -m build

# Upload to TestPyPI first
python -m twine upload --repository testpypi dist/*

# Test installation
pip install --index-url https://test.pypi.org/simple/ erica-clustering

# If all good, upload to PyPI
python -m twine upload dist/*
```

## Common Questions

### Q: Where did the logging functions go?
**A**: Replaced with simple `verbose` parameter. When `verbose=True`, functions print progress. No global state.

### Q: Can I still use the Gradio interface?
**A**: Yes! You can create a separate GUI module that imports from the library. Keep UI and logic separate.

### Q: Do I need to know the internals to use ERICA?
**A**: No! Just use the `ERICA` class for simple use cases. The modular design is for advanced users and maintainability.

### Q: How do I add new clustering methods?
**A**: Add a new function to `clustering.py` following the same pattern as `kmeans_clustering()` and `agglomerative_clustering()`.

### Q: Can I use only parts of ERICA?
**A**: Yes! All modules are independent. Import only what you need.

## Summary

You now have a **professional, modular, well-documented Python library** that:

✅ Separates concerns (clustering, metrics, data, plotting)
✅ Provides both high-level and low-level APIs  
✅ Has comprehensive documentation
✅ Follows Python best practices
✅ Is ready for PyPI publication
✅ Maintains all functionality from the original app

**All your clustering functions are now library functions** that can be imported, tested, and reused by anyone who installs your package!

---

*Need help? Check the documentation in `docs/` or open an issue on GitHub.*


