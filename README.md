# ERICA - Evaluating Replicability via Iterative Clustering Assignments

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://badge.fury.io/py/erica-clustering.svg)](https://badge.fury.io/py/erica-clustering)

Evaluating replicability via iterative clustering assignments (**ERICA**) is a Python library for assessing and quantifying clustering replicability via Monte Carlo (MC) subsampling and various clustering techniques. It provides robust evaluation of clustering stability across different savmpled versions of your dataset.

## Features

- 🎯 **Multiple Clustering Methods**: Support for K-Means and agglomerative (hierarchical) clustering with Ward or single linkage. This will be extended to additonal linkages (for hierarchical clustering) and more clustering methods.
- 📊 **Replicability Metrics**: Compute CRI, WCRI, and TWCRI metrics for stability assessment
- ⭐ **Optimal K Selection**: Automatic K\* selection via Algorithm 2 which attempts to mitigate the under-clustering phenomenon.
- 🔄 **Iterative Analysis**: Monte Carlo subsampling for robust evaluation.
- 📈 **Interactive Visualization**: Create important plots with Plotly.
- 🎨 **Optional GUI**: User-friendly Gradio web interface.
- 🔬 **Reproducible**: Deterministic mode for scientific reproducibility.
- ⚡ **Optimized I/O**: Smart caching for efficient processing.
- 📁 **Flexible Data Loading**: Automatic detection of CSV orientation (samples in rows/columns).

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

# Get optimal K* (automatically computed)
k_star = erica.get_k_star('TWCRI')
print(f"Optimal K* = {k_star['kmeans']}")

print(f"CRI: {metrics[3]['kmeans']['CRI']:.3f}")
print(f"WCRI: {metrics[3]['kmeans']['WCRI']:.3f}")
print(f"TWCRI: {metrics[3]['kmeans']['TWCRI']:.3f}")

# Visualize results
fig1, fig2 = erica.plot_metrics()
fig1.show()
```

### Input Dataset Orientation

ERICA expects data in **(n_samples, n_features)** format. The `transpose` parameter controls how your input data is interpreted:

**Default (Genomics Format):**
```python
from erica import ERICA
from erica.data import load_data

# For genomics data: features in rows, samples in columns
# Example: 22,283 genes × 344 samples
data = load_data('gene_expression.npy')
erica = ERICA(data=data)  # transpose=True by default
# Result: 344 samples × 22,283 features ✓
```

**Standard ML Format:**
```python
# For standard ML data: samples in rows, features in columns
# Example: 344 samples × 3 features
data = load_data('samples.csv')
erica = ERICA(data=data, transpose=False)
# Result: 344 samples × 3 features ✓
```

**Important Notes:**
- `.npy` files: If your array is already numeric, it's used as-is (no transposition)
- CSV files: Automatically removes ID columns and converts to numeric
- If you get an error about insufficient samples, try toggling `transpose=True/False`

## What is ERICA?

ERICA evaluates clustering stability through:

1. **Iterative Monte Carlo Subsampling**: Repeatedly split data into train/test sets
2. **Clustering**: Run clustering algorithms on each subsample
3. **Alignment**: Align cluster identities across iterations
4. **CLAM Matrix Generation**: Track cluster assignments across iterations
5. **Metrics Computation**: Calculate replicability scores

### Replicability Metrics

- **CRI (Clustering Replicability Index)**: Measures how consistently samples are assigned to their primary cluster
- **WCRI (Weighted CRI)**: CRI weighted by cluster sizes
- **TWCRI (Total Weighted CRI)**: Sum of weighted CRI values for overall assessment

**Higher values = Higher replicability**

### Empty Cluster Handling

ERICA carefully addresses cases where there are empty (null) clusters - i.e. cluster(s) with no datapoints assigned to them:

- **Detection**: Any K value with one or more empty clusters is flagged
- **Disqualification**: Metrics (CRI, WCRI, TWCRI) are marked as `NaN` for that K value
- **K Selection**: NaN values are automatically skipped per Algorithm 2
- **Tracking**: Disqualified K values are tracked and accessible via `get_disqualified_k()`
- **Output**: Clear warning message: "NaN (DISQUALIFIED - empty cluster detected)"

This ensures that only valid clustering configurations are considered for K* selection, maintaining compliance with the ERICA algorithm specification.

```python
# Get disqualified K values
disqualified = erica.get_disqualified_k()
print(f"Disqualified K values: {disqualified}")
# Output: {'kmeans': [8], 'agglomerative_ward': [7, 8]}

# Check if specific K was disqualified
if 8 in erica.get_disqualified_k('kmeans'):
    print("K=8 had empty clusters and was disqualified")
```

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

### Example 1: Gene Expression Analysis (.npy files)

```python
from erica import ERICA
from erica.data import load_data

# Load genomics data (features in rows, samples in columns)
# Example: vdx_dict.npy contains 22,283 genes × 344 samples
data = load_data('vdx_dict.npy')

# Run ERICA (default transpose=True handles genomics format)
erica = ERICA(
    data=data, 
    k_range=[2, 3, 4, 5, 6, 7, 8], 
    n_iterations=200
)
results = erica.run()

# Get optimal K* (automatically computed by ERICA)
k_star = erica.get_k_star('TWCRI')
optimal_k = k_star['kmeans']
print(f"Recommended number of clusters: {optimal_k}")
```

### Example 1b: CSV Files with Different Orientations

```python
from erica import ERICA
from erica.data import load_data

# Case 1: Genomics CSV (features in rows, samples in columns)
# File structure: each row = gene, each column = sample
# Example: 22,283 rows × 345 columns (1 ID + 344 samples)
data = load_data('gene_expression.csv')
erica = ERICA(data=data, transpose=True)  # Default
results = erica.run()
# Result: 344 samples × 22,283 features

# Case 2: Standard ML CSV (samples in rows, features in columns)
# File structure: each row = sample, each column = feature
# Example: VDX_3_SV.csv with 344 rows × 4 columns (1 ID + 3 genes)
data = load_data('VDX_3_SV.csv')
erica = ERICA(data=data, transpose=False)  # Must specify!
results = erica.run()
# Result: 344 samples × 3 features
```

**Troubleshooting:** If you get an error like "Dataset has 3 samples but k=4 clusters requested", your data orientation is likely incorrect. Try toggling `transpose=True/False`.

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

### Forthcoming...
<!--
### For Large Datasets (n > 10,000)

1. **Reduce iterations**: Use `n_iterations=100-200` instead of 500
2. **Use K-Means**: Faster than Agglomerative clustering
3. **Optimize I/O**: Enabled by default, keeps memory usage low
4. **Feature selection**: Reduce dimensionality if d is large

### For Small Datasets (n < 100)

1. **Increase iterations**: Use `n_iterations=300-500` for stability
2. **Adjust train/test split**: Use `train_percent=0.7` for larger test sets
3. **Test smaller k range**: Avoid k close to n
-->

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Patterns and Technology

### Core Technologies
- **Python 3.8+**: Primary programming language
- **NumPy**: Numerical computing and array operations
- **pandas**: Data manipulation and I/O
- **scikit-learn**: Clustering algorithms (KMeans, AgglomerativeClustering)
- **Plotly**: Interactive visualizations
- **Gradio**: Web-based GUI interface

### Design Patterns
- **Object-Oriented Design**: Main `ERICA` class encapsulates analysis workflow
- **Modular Architecture**: Separate modules for clustering, metrics, plotting, and data handling
- **Deterministic Execution**: Reproducibility through controlled random seeds
- **Lazy Computation**: CLAM matrices computed on-demand with caching
- **Strategy Pattern**: Multiple clustering methods via pluggable algorithms
- **Factory Pattern**: Automatic method selection and configuration
- **Iterator Pattern**: Monte Carlo subsampling iterations

### Code Organization
```
erica/
├── core.py           # Main ERICA class and workflow orchestration
├── clustering.py     # Clustering algorithms (KMeans, Agglomerative)
├── metrics.py        # Replicability metrics (CRI, WCRI, TWCRI) and K* selection
├── plotting.py       # Visualization functions
├── data.py           # Data loading and preprocessing
└── utils.py          # Utility functions and helpers
```

### Key Algorithms
1. **Monte Carlo Subsampling (MCSS)**: Iterative train/test splitting for stability evaluation
2. **CLAM Matrix Construction**: Co-occurrence matrix tracking cluster assignments
3. **Algorithm 2 (K Selection)**: Non-decreasing metric approach for optimal K determination
4. **CRI/WCRI/TWCRI Metrics**: Replicability indices for clustering quality

### Development Workflow
- **Version Control**: Git/GitHub
- **Testing**: pytest with comprehensive test suite
- **Code Quality**: black (formatting), flake8 (linting), mypy (type checking)
- **Documentation**: Markdown documentation and inline docstrings
- **Packaging**: setuptools with pyproject.toml

### Recent Enhancements (November 2025)
- ✅ Enhanced NA handling per Algorithm 2
- ✅ Improved K* selection documentation with line-by-line algorithm mapping
- ✅ Gradio demo with professional UI (3-tab design)
- ✅ Load previous runs feature for demo interface
- ✅ Comprehensive testing and validation

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

- 📧 Email: siamak_sorooshyari@yahoo.com
- 🐛 Issues: [GitHub Issues](https://github.com/yourusername/erica-clustering/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/yourusername/erica-clustering/discussions)
