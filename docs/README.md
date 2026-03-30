# ERICA Documentation

Welcome to the ERICA documentation. This guide covers everything you need to assess clustering replicability in your research—because "it looked like clusters to me" is not a valid statistical argument.

## Quick Navigation

| Document | Description | Audience |
|----------|-------------|----------|
| [Getting Started](GETTING_STARTED.md) | Installation and first analysis | New users |
| [Metrics Guide](METRICS_GUIDE.md) | CRI, WCRI, TWCRI explained | Understanding results |
| [Methodology](METHODOLOGY.md) | Scientific background and algorithms | Understanding the theory |
| [API Reference](API_REFERENCE.md) | Complete function documentation | Developers |
| [MCSS Algorithms](MCSS_ALGORITHMS.md) | Detailed algorithm specifications | The mathematically curious |

## Quick Start Guide

### 1. Installation

```bash
pip install erica
```

### 2. Basic Analysis

```python
from erica import ERICA
from erica.data import load_data

# Load your gene expression data
# Expected format: genes in rows, samples in columns
data = load_data('your_data.csv')

# Run ERICA analysis
erica = ERICA(
    data=data,
    k_range=[2, 3, 4, 5, 6, 7, 8],  # Test K=2 through K=8
    n_iterations=200,               # 200 Monte Carlo iterations
    method='both'                   # K-Means + Agglomerative (Single & Ward linkage)
)

results = erica.run()

# Get the recommended number of clusters
k_star = erica.get_k_star('TWCRI')
print(f"Recommended K: {k_star['kmeans']}")
```

### 3. Interpreting Results

**CRI (Clustering Replicability Index)** is the core metric of ERICA. It measures how consistently samples are assigned to their primary cluster across iterations.

| Metric | Role | Description |
|--------|------|-------------|
| **CRI** | Core metric | Consistency of sample-to-cluster assignments (0–1 scale) |
| WCRI | Derived | CRI weighted by cluster size |
| TWCRI | Aggregate | Sum of WCRI (used for K* selection) |

**CRI Interpretation:**

| CRI Range | Assessment |
|-----------|------------|
| > 0.8 | High replicability |
| 0.6–0.8 | Moderate replicability |
| < 0.6 | Low replicability (reviewer 2 will have questions) |

See the [Metrics Guide](METRICS_GUIDE.md) for detailed explanations of all metrics.

## Example Scripts

The [examples/](../examples/) folder contains executable scripts (see [examples/README.md](../examples/README.md) for details):

| # | Script | Description |
|---|--------|-------------|
| 1 | [01_basic_usage.py](../examples/01_basic_usage.py) | Synthetic data analysis |
| 2 | [02_vdx_analysis.py](../examples/02_vdx_analysis.py) | Breast cancer gene expression |
| 3 | [03_k_star_selection.py](../examples/03_k_star_selection.py) | K* selection via Algorithm 2 |
| 4 | [04_advanced_usage.py](../examples/04_advanced_usage.py) | Component-level workflows |

## Sample Data

Example datasets in `examples/data/`:

| File | Description |
|------|-------------|
| `VDX_3_SV.csv` | Reduced 3-gene subset (344 samples) |
| `samples_original_1.csv` | Synthetic mixture of 4 Gaussians |

For the full VDX dataset (22,283 genes × 344 samples):
```bash
curl -L -o vdx_dict.npy https://raw.githubusercontent.com/lorenzomasoero/clustering_replicability/master/real_data/Data/vdx_dict.npy
```

## Frequently Asked Questions

### Data orientation issues

**Q: I'm getting an error about insufficient samples.**

Your data orientation is likely incorrect. If your data has samples in rows (standard ML format), set:
```python
erica = ERICA(data=data, transpose=False, ...)
```

### Evaluating K values

**Q: How do I know if a K value is problematic?**

Check for disqualified K values (those with empty clusters):
```python
disqualified = erica.get_disqualified_k()
if 5 in disqualified.get('kmeans', []):
    print("K=5 produced empty clusters")
```

### Choosing clustering methods

**Q: K-Means or Agglomerative?**

| Method | Characteristics |
|--------|-----------------|
| K-Means | Faster; assumes spherical clusters |
| Agglomerative (Single linkage) | Sensitive to elongated/chained clusters |
| Agglomerative (Ward linkage) | Minimizes within-cluster variance; more flexible cluster shapes |

Use `method='both'` to run all three: K-Means, Agglomerative (Single), and Agglomerative (Ward). For finer control, pass a list: `method=['kmeans', 'agglomerative_ward']`.

## Contact

- **Issues:** [GitHub Issues](https://github.com/PhenomML/ERICA/issues)
- **Email:** s.shirazi@berkeley.edu (Shawn Shirazi), siamak_sorooshyari@yahoo.com (Siamak Sorooshyari)

---

*For internal implementation notes, see [development/](development/)*
