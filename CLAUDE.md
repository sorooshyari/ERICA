# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ERICA (Evaluating Replicability via Iterative Clustering Assignments) is a Python library for assessing clustering stability through Monte Carlo subsampling. It provides replicability metrics (CRI, WCRI, TWCRI) to evaluate how consistently samples are assigned to clusters across different subsamples of the data.

## Common Commands

```bash
# Install in development mode
pip install -e .

# Install with all optional dependencies
pip install -e ".[all]"

# Run all tests
pytest tests/ -v

# Run tests with coverage
pytest tests/ -v --cov=erica --cov-report=xml

# Run a single test
pytest tests/test_erica.py::test_erica_run -v

# Code quality (dev dependencies required)
black erica/
flake8 erica/
mypy erica/
```

## Architecture

The package follows a modular architecture with clear separation of concerns:

```
erica/
├── core.py        # ERICA class - orchestrates the complete workflow
├── clustering.py  # Clustering algorithms and CLAM matrix generation
├── metrics.py     # CRI/WCRI/TWCRI computation and K* selection
├── data.py        # Data loading and validation
├── plotting.py    # Visualization (requires plotly)
└── utils.py       # Deterministic mode, hashing utilities
```

### Key Workflow (core.py)

The `ERICA` class coordinates a 4-step process:
1. **Iterative subsampling** - Monte Carlo train/test splits (`iterative_clustering_subsampling`)
2. **Clustering** - Run K-Means or Agglomerative for each K value
3. **Metrics computation** - Calculate CRI/WCRI/TWCRI from CLAM matrices
4. **K* selection** - Select optimal K using Algorithm 2 (non-decreasing metric criterion)

### Data Orientation Convention

ERICA expects data in `(n_samples, n_features)` format. The `transpose` parameter controls input interpretation:
- `transpose=True` (default): Input is genomics format (features × samples)
- `transpose=False`: Input is standard ML format (samples × features)

### Empty Cluster Handling

When clustering produces empty clusters, metrics are set to NaN and that K value is disqualified from K* selection. Track disqualified values via `get_disqualified_k()`.

## Sample Data

Sample breast cancer gene expression data from [Parmigiani et al.'s clustering replicability paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10600961/):

```bash
# Full dataset (22,283 genes × 344 samples)
curl -L -o vdx_dict.npy https://raw.githubusercontent.com/lorenzomasoero/clustering_replicability/master/real_data/Data/vdx_dict.npy

# Small 3-gene subset
curl -L -o VDX_3_SV.csv https://raw.githubusercontent.com/lorenzomasoero/clustering_replicability/master/real_data/Data/VDX_3_SV.csv
```

Source repo: https://github.com/lorenzomasoero/clustering_replicability

## Key Dependencies

- **Core**: numpy, pandas, scikit-learn, pyyaml
- **Plots** (optional): plotly, matplotlib
- **GUI** (optional): gradio

## Testing Patterns

Tests use temporary files for CSV loading tests and verify:
- Correct data orientation handling
- Metric computation accuracy
- K* selection algorithm behavior
- Empty cluster disqualification

## Version Management

Version is defined in `setup.py` and `erica/__init__.py`. Use `scripts/bump_version.py` for version updates.
