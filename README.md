# ERICA: Evaluating Replicability via Iterative Clustering Assignments

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://badge.fury.io/py/erica-clustering.svg)](https://badge.fury.io/py/erica-clustering)

**ERICA** is a Python implementation for assessing clustering replicability using Monte Carlo subsampling (MCSS). Because finding clusters is easy—finding clusters that *actually exist* is the hard part.

The method evaluates whether cluster structures identified in a dataset are stable and reproducible across random subsamples, providing quantitative metrics for clustering validity assessment.

## Installation

```bash
pip install erica-clustering

# With visualization support
pip install erica-clustering[plots]

# With graphical interface (for those who prefer clicking)
pip install erica-clustering[gui]
```

## Basic Usage

```python
from erica import ERICA
from erica.data import load_data

# Load data (samples × features format)
data = load_data('expression_data.csv')

# Initialize and run analysis
erica = ERICA(
    data=data,
    k_range=[2, 3, 4, 5, 6, 7, 8],
    n_iterations=200,
    method='both'
)
results = erica.run()

# Retrieve recommended cluster number
k_star = erica.get_k_star('TWCRI')
print(f"Recommended K: {k_star['kmeans']}")
```

## Documentation

| Document | Description |
|----------|-------------|
| [Documentation Index](docs/README.md) | Complete documentation |
| [Getting Started](docs/GETTING_STARTED.md) | Installation and tutorial |
| [Metrics Guide](docs/METRICS_GUIDE.md) | CRI, WCRI, TWCRI explained |
| [API Reference](docs/API_REFERENCE.md) | Function documentation |
| [Methodology](docs/METHODOLOGY.md) | The science behind the magic |

## Example Scripts

| # | Script | Description |
|---|--------|-------------|
| 1 | [01_basic_usage.py](examples/01_basic_usage.py) | Synthetic data analysis |
| 2 | [02_vdx_analysis.py](examples/02_vdx_analysis.py) | Breast cancer gene expression (VDX dataset) |
| 3 | [03_k_star_selection.py](examples/03_k_star_selection.py) | K* selection via Algorithm 2 |
| 4 | [04_advanced_usage.py](examples/04_advanced_usage.py) | Component-level workflows |

See [examples/README.md](examples/README.md) for details and data acquisition instructions.

## Data Format

ERICA operates on data in **samples × features** format. (Yes, the orientation matters. We've all learned this the hard way.)

| Data Type | Input Format | Parameter |
|-----------|--------------|-----------|
| Genomics (genes × samples) | Features in rows | `transpose=True` (default) |
| Standard (samples × features) | Samples in rows | `transpose=False` |

## The CRI Metric

**CRI (Clustering Replicability Index)** is the core metric of ERICA. It measures how consistently samples are assigned to their primary cluster across Monte Carlo iterations.

| Metric | Role | Definition |
|--------|------|------------|
| **CRI** | Core metric | Proportion of iterations where samples are assigned to their primary cluster |
| WCRI | Derived | CRI weighted by cluster size |
| TWCRI | Aggregate | Sum of WCRI (used for K* selection) |

Values range from 0 to 1, where higher values indicate greater replicability.

| CRI Range | Interpretation |
|-----------|----------------|
| > 0.8 | High replicability (publishable with confidence) |
| 0.6–0.8 | Moderate replicability (proceed with caution) |
| < 0.6 | Low replicability (perhaps try a different K) |

See the [Metrics Guide](docs/METRICS_GUIDE.md) for detailed explanations.

## Sample Datasets

| Dataset | Description | Source |
|---------|-------------|--------|
| VDX | Breast cancer gene expression (22,283 genes × 344 samples) | [Parmigiani et al.](https://github.com/lorenzomasoero/clustering_replicability) |
| VDX_3_SV | Reduced 3-gene subset (344 samples) | Included in `examples/data/` |

## Requirements

**Core:** Python ≥ 3.8, NumPy, Pandas, scikit-learn, PyYAML

**Optional:** Plotly (visualization), Gradio (GUI)

## Citation

If you use ERICA in your research, please cite:

```bibtex
@software{erica2025,
  title = {ERICA: Evaluating Replicability via Iterative Clustering Assignments},
  author = {Sorooshyari, Siamak and Shirazi, Shawn},
  year = {2025},
  url = {https://github.com/sorooshyari/ERICA}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.

## Contact

- **Issues:** [GitHub Issues](https://github.com/sorooshyari/ERICA/issues)
- **Email:** s.shirazi@berkeley.edu, siamak_sorooshyari@yahoo.com
