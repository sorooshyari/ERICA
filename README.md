# ERICA - Evaluating Replicability via Iterative Clustering Assignments

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://badge.fury.io/py/erica-clustering.svg)](https://badge.fury.io/py/erica-clustering)

**ERICA** is a comprehensive Python library for analyzing clustering replicability using Monte Carlo subsampling. It provides robust evaluation of clustering stability across different subsamples of your data.

## Features

- 🎯 **Multiple Clustering Methods**: Support for K-Means and Agglomerative (Hierarchical) Clustering
- 📊 **Replicability Metrics**: Compute CRI, WCRI, and TWCRI metrics for stability assessment
- 🔄 **Iterative Analysis**: Monte Carlo subsampling for robust evaluation
- 📈 **Interactive Visualization**: Create beautiful plots with Plotly
- 🎨 **Optional GUI**: User-friendly Gradio web interface
- 🔬 **Reproducible**: Deterministic mode for scientific reproducibility
- ⚡ **Optimized I/O**: Smart caching for efficient processing

## Quick Start

### Installation

```bash
# Basic installation
pip install erica-clustering

# With plotting support
pip install erica-clustering[plots]

# With GUI support
pip install erica-clustering[gui]

# Full installation
pip install erica-clustering[all]
```

### Basic Usage

```python
import numpy as np
from erica import ERICA

# Load your data (samples × features)
data = np.random.rand(100, 50)

# Run ERICA analysis
erica = ERICA(
    data=data,
    k_range=[2, 3, 4, 5],
    n_iterations=200,
    method='both'
)

results = erica.run()

# Get results
clam_matrix = erica.get_clam_matrix(k=3)
metrics = erica.get_metrics()

print(f"CRI: {metrics[3]['kmeans']['CRI']:.3f}")
print(f"WCRI: {metrics[3]['kmeans']['WCRI']:.3f}")
print(f"TWCRI: {metrics[3]['kmeans']['TWCRI']:.3f}")

# Visualize results
fig1, fig2 = erica.plot_metrics()
fig1.show()
```

## What is ERICA?

ERICA evaluates clustering stability through:

1. **Iterative Subsampling**: Repeatedly split data into train/test sets
2. **Clustering**: Run clustering algorithms on each subsample
3. **Alignment**: Align cluster identities across iterations
4. **CLAM Matrix Generation**: Track cluster assignments across iterations
5. **Metrics Computation**: Calculate replicability scores

### Replicability Metrics

- **CRI (Clustering Replicability Index)**: Measures how consistently samples are assigned to their primary cluster
- **WCRI (Weighted CRI)**: CRI weighted by cluster sizes
- **TWCRI (Total Weighted CRI)**: Sum of weighted CRI values for overall assessment

**Higher values = Better replicability!**

## Advanced Usage

### Using Individual Components

```python
from erica.clustering import kmeans_clustering, iterative_clustering_subsampling
from erica.metrics import compute_metrics_for_clam
from erica.data import prepare_samples_array

# Prepare data
samples = prepare_samples_array(your_data)

# Perform subsampling
train_size = int(len(samples) * 0.8)
_, indices_folder = iterative_clustering_subsampling(
    samples, len(samples), 200, train_size, './output'
)

# Run clustering
result = kmeans_clustering(
    samples, k=3, n_iterations=200,
    indices_folder=indices_folder,
    output_dir='./output'
)

# Compute metrics
metrics = compute_metrics_for_clam(result['clam_matrix'], k=3)
```

### Custom Plotting

```python
from erica.plotting import plot_metrics, plot_clam_heatmap

# Plot metrics
fig = plot_metrics(k_values, cri_values, wcri_values, twcri_values)
fig.show()

# Visualize CLAM matrix
fig = plot_clam_heatmap(clam_matrix, k=3)
fig.show()
```

## Documentation

- 📖 [Getting Started Guide](docs/GETTING_STARTED.md)
- 📚 [API Reference](docs/API_REFERENCE.md)
- 🧬 [Methodology](docs/METHODOLOGY.md)
- 📦 [PyPI Publishing Guide](PYPI_GUIDE.md)

## Project Structure

```
erica/
├── __init__.py          # Main package interface
├── core.py              # ERICA main class
├── clustering.py        # Clustering algorithms
├── metrics.py           # Replicability metrics
├── data.py              # Data loading and preparation
├── plotting.py          # Visualization functions
└── utils.py             # Utility functions
```

## Requirements

**Core Dependencies:**
- Python >= 3.8
- NumPy >= 1.21.0
- Pandas >= 1.3.0
- scikit-learn >= 1.0.0
- PyYAML >= 6.0

**Optional Dependencies:**
- Plotly >= 5.0.0 (for plotting)
- Matplotlib >= 3.5.0 (for additional plots)
- Gradio >= 4.0.0 (for GUI)

## Use Cases

### Bioinformatics
- Gene expression clustering analysis
- Single-cell RNA-seq clustering validation
- Patient stratification assessment

### General Machine Learning
- Evaluating clustering stability before downstream analysis
- Comparing different clustering methods objectively
- Selecting optimal k with stability criterion
- Identifying outliers or ambiguous samples

## Examples

### Example 1: Gene Expression Analysis

```python
import pandas as pd
from erica import ERICA, load_data

# Load gene expression data
data = load_data('gene_expression.csv')

# Run ERICA
erica = ERICA(data=data, k_range=[2, 3, 4, 5, 6], n_iterations=200)
results = erica.run()

# Find optimal k
from erica.metrics import find_optimal_k
optimal_k, _ = find_optimal_k(erica.get_metrics(), metric_name='TWCRI')
print(f"Recommended number of clusters: {optimal_k}")
```

### Example 2: Method Comparison

```python
from erica import ERICA

# Test K-Means
erica_km = ERICA(data=data, k_range=[2, 3, 4], method='kmeans')
results_km = erica_km.run()

# Test Agglomerative
erica_agg = ERICA(data=data, k_range=[2, 3, 4], method='agglomerative')
results_agg = erica_agg.run()

# Compare metrics
metrics_km = erica_km.get_metrics()
metrics_agg = erica_agg.get_metrics()
```

## Performance Tips

### For Large Datasets (n > 10,000)

1. **Reduce iterations**: Use `n_iterations=100-200` instead of 500
2. **Use K-Means**: Faster than Agglomerative clustering
3. **Optimize I/O**: Enabled by default, keeps memory usage low
4. **Feature selection**: Reduce dimensionality if d is large

### For Small Datasets (n < 100)

1. **Increase iterations**: Use `n_iterations=300-500` for stability
2. **Adjust train/test split**: Use `train_percent=0.7` for larger test sets
3. **Test smaller k range**: Avoid k close to n

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use ERICA in your research, please cite:

```bibtex
@software{erica2025,
  title = {ERICA: Evaluating Replicability via Iterative Clustering Assignments},
  author = {Sorooshyari, Siamak and Shirazi, Shawn},
  year = {2025},
  url = {https://github.com/yourusername/erica-clustering}
}
```

## Acknowledgments

- Original Monte Carlo Subsampling for Clustering Replicability (MCSS) methodology
- scikit-learn for clustering algorithms
- Plotly for interactive visualizations
- Gradio for web interface capabilities

## Support

- 📧 Email: your.email@example.com
- 🐛 Issues: [GitHub Issues](https://github.com/yourusername/erica-clustering/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/yourusername/erica-clustering/discussions)

---

**Made with ❤️ for the scientific community**
# ERICA_ClusterReplic_10242025
