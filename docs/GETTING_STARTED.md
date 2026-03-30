# Getting Started with ERICA

## Installation

### Basic Installation

```bash
pip install erica
```

### With Plotting Support

```bash
pip install erica[plots]
```

### With GUI Support

```bash
pip install erica[gui]
```

### Full Installation (All Features)

```bash
pip install erica[all]
```

### Development Installation

```bash
git clone https://github.com/PhenomML/ERICA.git
cd ERICA
pip install -e .[dev]
```

---

## Quick Start

### 5-Minute Tutorial

```python
import numpy as np
from erica import ERICA

# 1. Load or generate your data
data = np.random.rand(100, 50)  # 100 samples, 50 features

# 2. Create ERICA instance
erica = ERICA(
    data=data,
    k_range=[2, 3, 4, 5],  # Test k from 2 to 5
    n_iterations=200,       # Number of Monte Carlo iterations
    method='both'           # Test both K-Means and Agglomerative
)

# 3. Run analysis
results = erica.run()

# 4. Get results
clam_matrix = erica.get_clam_matrix(k=3)
metrics = erica.get_metrics()

print(f"CRI: {metrics[3]['kmeans']['CRI']:.3f}")
print(f"WCRI: {metrics[3]['kmeans']['WCRI']:.3f}")
print(f"TWCRI: {metrics[3]['kmeans']['TWCRI']:.3f}")

# 5. Visualize (requires plotly)
fig1, fig2 = erica.plot_metrics()
fig1.show()
```

---

## Understanding ERICA

### What is ERICA?

ERICA (Evaluating Replicability via Iterative Clustering Assignments) is a method for assessing the stability and replicability of clustering results. It works by:

1. **Iterative Subsampling**: Repeatedly splitting your data into train/test sets
2. **Clustering**: Running clustering algorithms on each subsample
3. **Alignment**: Aligning cluster labels across iterations
4. **CLAM Generation**: Creating a CLuster Assignment Matrix that tracks how often each sample is assigned to each cluster
5. **Metrics Computation**: Calculating replicability scores (CRI, WCRI, TWCRI)

### Key Concepts

#### CLAM Matrix
The CLAM (CLuster Assignment Matrix) is a fundamental output where element (i, j) represents the number of times sample i was assigned to cluster j across all iterations.

#### Replicability Metrics

**CRI (Clustering Replicability Index)** is the core metric of ERICA—the number you'll report in your papers. It measures how consistently samples are assigned to their primary cluster across Monte Carlo iterations.

| Metric | Role | Description |
|--------|------|-------------|
| **CRI** | Core metric | Consistency of sample-to-cluster assignments (0–1 scale) |
| WCRI | Derived | CRI weighted by cluster size |
| TWCRI | Aggregate | Sum of WCRI (used for K* selection) |

Higher values indicate better replicability. CRI > 0.8 is excellent; CRI < 0.6 means reviewer 2 will have questions.

---

## Step-by-Step Tutorial

### Step 1: Prepare Your Data

ERICA accepts data in multiple formats:

```python
import numpy as np
import pandas as pd
from erica import load_data, prepare_samples_array

# Option 1: NumPy array (samples x features)
data = np.random.rand(100, 50)

# Option 2: From CSV file
data = load_data('my_data.csv')

# Option 3: From .npy file
data = load_data('my_data.npy')

# Option 4: Pandas DataFrame
df = pd.DataFrame(np.random.rand(100, 50))
data = prepare_samples_array(df)
```

**Data Format Requirements:**
- Rows should be samples (observations)
- Columns should be features (variables)
- Data should be numeric
- No missing values (NaN) or infinite values

### Step 2: Configure ERICA

```python
from erica import ERICA

erica = ERICA(
    data=data,
    k_range=[2, 3, 4, 5],       # Range of k values to test
    n_iterations=200,            # Number of Monte Carlo iterations
    train_percent=0.8,           # 80% for training, 20% for testing
    method='both',               # Test both methods
    linkages=['single', 'ward'], # Linkage methods for agglomerative
    random_seed=123,             # For reproducibility
    output_dir='./erica_output', # Where to save results
    verbose=True                 # Print progress
)
```

**Parameter Guide:**
- `k_range`: Start with a reasonable range based on domain knowledge
- `n_iterations`: 200 is a good default; increase for more stable estimates
- `train_percent`: 0.8 is standard; adjust based on dataset size
- `method`: Use 'both' to compare methods, or 'kmeans'/'agglomerative' for one
- `random_seed`: Set for reproducibility

### Step 3: Run Analysis

```python
# Run the complete analysis
results = erica.run()

# This performs:
# 1. Iterative subsampling
# 2. Clustering for each k and method
# 3. Cluster identity alignment
# 4. CLAM matrix generation
# 5. Metrics computation
```

### Step 4: Examine Results

```python
# Get all results
results = erica.get_results()

# Get CLAM matrix for specific k and method
clam_kmeans_k3 = erica.get_clam_matrix(k=3, method='kmeans')
clam_agg_k3 = erica.get_clam_matrix(k=3, method='agglomerative_ward')

# Get metrics
all_metrics = erica.get_metrics()
k3_metrics = erica.get_metrics(k=3)

# Print metrics for k=3, K-Means
metrics = k3_metrics['kmeans']
print(f"K=3 K-Means Results:")
print(f"  CRI:   {metrics['CRI']:.4f}")
print(f"  WCRI:  {metrics['WCRI']:.4f}")
print(f"  TWCRI: {metrics['TWCRI']:.4f}")
```

### Step 5: Get Recommended K* (Algorithm 2)

ERICA automatically computes K* using Algorithm 2 during the run. K* represents the **recommended** number of clusters—the largest K where replicability metrics remain non-decreasing. This is a data-driven suggestion, not a definitive answer.

```python
# Get recommended K* for each metric (automatically computed)
k_star_twcri = erica.get_k_star('TWCRI')
k_star_cri = erica.get_k_star('CRI')
k_star_wcri = erica.get_k_star('WCRI')

print(f"Recommended K* (TWCRI): {k_star_twcri['kmeans']}")
print(f"Recommended K* (CRI): {k_star_cri['kmeans']}")
print(f"Recommended K* (WCRI): {k_star_wcri['kmeans']}")

# Or use select_optimal_k directly for custom analysis
from erica.metrics import select_optimal_k

# Extract TWCRI values for all K
twcri_dict = {k: all_metrics[k]['kmeans']['TWCRI'] for k in all_metrics}
recommended_k = select_optimal_k(twcri_dict)
print(f"Recommended K* = {recommended_k}")
```

### Step 6: Visualize Results

```python
# Create interactive plots
fig_metrics, fig_optimal = erica.plot_metrics()

# Show plots in browser
fig_metrics.show()
fig_optimal.show()

# Save plots to HTML
fig_metrics.write_html('erica_metrics.html')

# Or use individual plotting functions
from erica.plotting import plot_clam_heatmap, plot_cluster_sizes

# Visualize CLAM matrix
fig_clam = plot_clam_heatmap(clam_kmeans_k3, k=3)
fig_clam.show()

# Visualize cluster sizes
metrics = k3_metrics['kmeans']
fig_sizes = plot_cluster_sizes(metrics['cluster_sizes'], k=3)
fig_sizes.show()
```

### Step 7: Save Results

```python
# Save complete results
erica.save_results('erica_results.json')

# Save specific CLAM matrices
from erica.data import save_clam_matrix

save_clam_matrix(clam_kmeans_k3, 'clam_k3_kmeans.csv', format='csv')
save_clam_matrix(clam_kmeans_k3, 'clam_k3_kmeans.npy', format='npy')
```

---

## Working with Real Data

### Example: Gene Expression Data

```python
import pandas as pd
from erica import ERICA, load_data, prepare_samples_array

# Load gene expression data (genes x samples)
df = pd.read_csv('gene_expression.csv', index_col=0)

# Prepare data (transposes to samples x features)
data = prepare_samples_array(df)

print(f"Data shape: {data.shape}")  # (n_samples, n_genes)

# Run ERICA with appropriate k range for biological data
erica = ERICA(
    data=data,
    k_range=[2, 3, 4, 5, 6, 7, 8],  # Test broader range
    n_iterations=200,
    method='both',
    random_seed=42,
    verbose=True
)

results = erica.run()

# Get recommended K* (automatically computed by ERICA)
k_star = erica.get_k_star('TWCRI')
recommended_k = k_star['kmeans']
print(f"Recommended number of clusters: {recommended_k}")

# Get cluster assignments for recommended k
clam = erica.get_clam_matrix(k=recommended_k, method='kmeans')
primary_clusters = np.argmax(clam, axis=1)

# Add cluster assignments to original dataframe
df_samples = df.T  # Transpose to samples x genes
df_samples['cluster'] = primary_clusters

# Save annotated data
df_samples.to_csv('clustered_samples.csv')
```

---

## Advanced Usage

### Using Individual Components

For more control, use ERICA's modular components:

```python
from erica.clustering import iterative_clustering_subsampling, kmeans_clustering
from erica.metrics import compute_metrics_for_clam
from erica.data import prepare_samples_array
from erica.utils import set_deterministic_mode

# Set reproducibility
set_deterministic_mode(123)

# Prepare data
data = prepare_samples_array(your_data)
n_samples = len(data)
train_size = int(n_samples * 0.8)

# Step 1: Perform subsampling
subsamples_folder, indices_folder = iterative_clustering_subsampling(
    samples_array=data,
    num_samples=n_samples,
    num_iterations=200,
    subsample_size_train=train_size,
    base_save_folder_str='./output',
    verbose=True
)

# Step 2: Run clustering for specific k
result = kmeans_clustering(
    samples_array=data,
    k=3,
    n_iterations=200,
    indices_folder=indices_folder,
    output_dir='./output',
    verbose=True
)

# Step 3: Compute metrics
clam_matrix = result['clam_matrix']
metrics = compute_metrics_for_clam(clam_matrix, k=3)

print(f"CRI: {metrics['CRI']:.4f}")
print(f"WCRI: {metrics['WCRI']:.4f}")
print(f"TWCRI: {metrics['TWCRI']:.4f}")
```

---

## Tips and Best Practices

### Choosing K Range
- Start with domain knowledge or prior analyses
- Include k values from 2 up to √n (roughly)
- For large datasets, test a wider range

### Number of Iterations
- **Small datasets (< 100 samples)**: 100-200 iterations
- **Medium datasets (100-1000 samples)**: 200-500 iterations
- **Large datasets (> 1000 samples)**: 200-300 iterations is usually sufficient

### Train/Test Split
- **Default (0.8)**: Works well for most cases
- **Small datasets**: Use 0.7-0.75 for larger test sets
- **Large datasets**: Can use 0.85-0.9

### Interpreting Metrics
- **CRI > 0.8**: Excellent replicability
- **CRI 0.6-0.8**: Good replicability
- **CRI < 0.6**: Poor replicability, consider different k or method

### Performance Optimization
- Start with fewer iterations during exploration
- Use `method='kmeans'` if agglomerative is slow
- For very large datasets, consider subsampling features

---

## Troubleshooting

### Common Issues

**Issue: "Dataset has only X samples but k=Y clusters requested"**
- Solution: Reduce k or get more samples

**Issue: "Training subset size is smaller than number of clusters"**
- Solution: Reduce k or increase train_percent

**Issue: "Dataset contains NaN values"**
- Solution: Clean your data before running ERICA
```python
data = np.nan_to_num(data)  # Replace NaN with 0
# Or remove rows/columns with NaN
```

**Issue: Plots not showing**
- Solution: Make sure plotly is installed
```bash
pip install erica[plots]
```

**Issue: Very slow for large datasets**
- Solution: Reduce n_iterations or use fewer k values
- Consider running on a subset first to estimate time

---

## Next Steps

- Read the [API Reference](API_REFERENCE.md) for detailed function documentation
- Check out [Examples](../examples/) for more use cases
- See [Methodology](METHODOLOGY.md) for theoretical background
- Visit the [GitHub repository](https://github.com/PhenomML/ERICA) for issues and contributions


