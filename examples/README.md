# ERICA Examples

This folder contains executable example scripts demonstrating ERICA for clustering replicability analysis. Start with Example 1 and progress sequentially—unless you enjoy debugging orientation errors, in which case, proceed in any order.

## Quick Start

```bash
# Install ERICA
pip install erica-clustering

# Run the basic example
cd examples
python 01_basic_usage.py
```

## Examples Overview

| # | Script | Description | Data Required |
|---|--------|-------------|---------------|
| 1 | [01_basic_usage.py](01_basic_usage.py) | Synthetic data analysis | None (generates data) |
| 2 | [02_vdx_analysis.py](02_vdx_analysis.py) | Breast cancer gene expression | Downloads automatically |
| 3 | [03_k_star_selection.py](03_k_star_selection.py) | K* selection (Algorithm 2) | None (generates data) |
| 4 | [04_advanced_usage.py](04_advanced_usage.py) | Component-level workflows | None (generates data) |
| 5 | [05_parmigiani_metrics.py](05_parmigiani_metrics.py) | ARI/AMI partition metrics | None (generates data) |

---

## Example 1: Basic Usage

**File:** `01_basic_usage.py`

The simplest introduction to ERICA. Uses randomly generated data to demonstrate the core workflow without external dependencies.

**Learning objectives:**
- Initialize ERICA with data
- Execute a complete analysis
- Access results and metrics
- Retrieve recommended K* values

**Execute:**
```bash
python 01_basic_usage.py
```

---

## Example 2: VDX Analysis

**File:** `02_vdx_analysis.py`

Demonstrates ERICA with real-world breast cancer gene expression data from the VDX dataset (Parmigiani et al.).

**Learning objectives:**
- Load `.npy` genomics data files
- Analyze high-dimensional data (22,283 genes)
- Interpret results for biological datasets

**Data:** The script prompts to download the VDX dataset (~1.8 MB) if not present.

**Execute:**
```bash
python 02_vdx_analysis.py
```

---

## Example 3: K* Selection

**File:** `03_k_star_selection.py`

Detailed examination of K* (recommended cluster number) selection using Algorithm 2.

**Learning objectives:**
- Understand Algorithm 2 mechanics
- Access K* for different metrics (CRI, WCRI, TWCRI)
- Generate K* selection visualizations
- Distinguish K* from naive metric maximization

**Execute:**
```bash
python 03_k_star_selection.py
```

---

## Example 4: Advanced Usage

**File:** `04_advanced_usage.py`

For users requiring fine-grained control over the ERICA workflow. Not recommended as a starting point unless you enjoy reading documentation.

**Learning objectives:**
- Use individual ERICA components
- Configure deterministic execution for reproducibility
- Implement custom workflows with iterative subsampling
- Export and visualize CLAM matrices

**Execute:**
```bash
python 04_advanced_usage.py
```

---

## Example 5: Parmigiani Metrics (ARI/AMI)

**File:** `05_parmigiani_metrics.py`

Demonstrates the Adjusted Rand Index (ARI) and Adjusted Mutual Information (AMI) metrics from Parmigiani et al. (2023) "Cross-Study Replicability in Cluster Analysis".

**Learning objectives:**
- Understand the Parmigiani Algorithm 1 methodology
- Compute ARI and AMI for single iterations
- Aggregate metrics across Monte Carlo iterations
- Compare ERICA's CLAM-based metrics vs partition comparison metrics

**Reference:**
- Paper: Statistical Science, 38(2): 303-316 (2023), DOI: 10.1214/22-STS871
- Code: https://github.com/lorenzomasoero/clustering_replicability

**Execute:**
```bash
python 05_parmigiani_metrics.py
```

---

## Sample Data

The `data/` folder contains sample datasets:

| File | Description | Format |
|------|-------------|--------|
| `VDX_3_SV.csv` | Reduced 3-gene subset (344 samples) | CSV |
| `samples_original_1.csv` | Synthetic mixture of 4 Gaussians | CSV |

### Full Dataset Acquisition

For the complete VDX breast cancer dataset (22,283 genes × 344 samples):

```bash
# Download to examples/data folder
curl -L -o data/vdx_dict.npy \
  https://raw.githubusercontent.com/lorenzomasoero/clustering_replicability/master/real_data/Data/vdx_dict.npy
```

---

## Gradio Web Interface

For users who prefer graphical interfaces:

- [gradio_app.py](gradio_app.py) — Full-featured web application
- [GRADIO_README.md](GRADIO_README.md) — Setup instructions

**Launch:**
```bash
# Install GUI dependencies
pip install erica-clustering[gui]

# Start web interface
python gradio_app.py
```

---

## Troubleshooting

### "Dataset has X samples but k=Y clusters requested"

Data orientation issue. Set `transpose=False` if your samples are in rows:
```python
erica = ERICA(data=data, transpose=False, ...)
```

### "Module not found: plotly"

Install visualization dependencies:
```bash
pip install erica-clustering[plots]
```

### Slow execution

Reduce iterations for development/testing:
```python
erica = ERICA(..., n_iterations=50)  # Default is 200
```

---

## Resources

- [Documentation](../docs/README.md)
- [API Reference](../docs/API_REFERENCE.md)
- [GitHub Issues](https://github.com/sorooshyari/ERICA/issues)
