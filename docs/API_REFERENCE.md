# ERICA API Reference

## Table of Contents

- [Core Module](#core-module)
- [Clustering Module](#clustering-module)
- [Metrics Module](#metrics-module)
- [Data Module](#data-module)
- [Plotting Module](#plotting-module)
- [Utils Module](#utils-module)

---

## Core Module

### `ERICA` Class

Main class for ERICA clustering replicability analysis.

#### Constructor

```python
ERICA(
    data,
    k_range=[2, 3, 4, 5],
    n_iterations=200,
    train_percent=0.8,
    method='both',
    linkages=['single', 'ward'],
    random_seed=123,
    output_dir='./erica_output',
    verbose=True
)
```

**Parameters:**
- `data` (np.ndarray or pd.DataFrame): Input data matrix (n_samples, n_features)
- `k_range` (list of int): Range of cluster numbers to evaluate
- `n_iterations` (int): Number of Monte Carlo iterations (B)
- `train_percent` (float): Proportion of data for training subsample (0.0-1.0)
- `method` (str): Clustering method ('kmeans', 'agglomerative', or 'both')
- `linkages` (list of str): Linkage methods for agglomerative clustering
- `random_seed` (int): Random seed for reproducibility
- `output_dir` (str): Base directory for output files
- `verbose` (bool): Whether to print progress messages

#### Methods

##### `run()`
Run the complete ERICA analysis.

**Returns:** dict containing results

**Example:**
```python
erica = ERICA(data=my_data, k_range=[2, 3, 4])
results = erica.run()
```

##### `get_results()`
Get all analysis results.

**Returns:** dict with keys:
- `'clam_matrices'`: CLAM matrices for each (k, method)
- `'metrics'`: Computed metrics
- `'config'`: Configuration parameters
- `'output_folders'`: Output directory paths

##### `get_clam_matrix(k, method='kmeans')`
Get CLAM matrix for specific k and method.

**Parameters:**
- `k` (int): Number of clusters
- `method` (str): Method name

**Returns:** np.ndarray or None

##### `get_metrics(k=None)`
Get computed metrics.

**Parameters:**
- `k` (int, optional): If specified, return metrics only for this k

**Returns:** dict of metrics

##### `plot_metrics(**kwargs)`
Generate metrics plots.

**Returns:** tuple of (metrics_plot, optimal_k_plot)

**Raises:** ImportError if plotly not installed

##### `save_results(filepath)`
Save results to JSON file.

**Parameters:**
- `filepath` (str): Output file path

---

## Clustering Module

### Functions

#### `iterative_clustering_subsampling()`

Perform Monte Carlo subsampling for ERICA analysis.

```python
iterative_clustering_subsampling(
    samples_array,
    num_samples,
    num_iterations,
    subsample_size_train,
    base_save_folder_str,
    iter_prefix="",
    optimize_io=True,
    verbose=False
)
```

**Parameters:**
- `samples_array` (np.ndarray): Input data (n_samples, n_features)
- `num_samples` (int): Total number of samples
- `num_iterations` (int): Number of iterations (B)
- `subsample_size_train` (int): Training subsample size
- `base_save_folder_str` (str): Base folder for outputs
- `iter_prefix` (str): Prefix for subfolder names
- `optimize_io` (bool): If True, only save indices
- `verbose` (bool): Print progress messages

**Returns:** tuple of (subsamples_folder, indices_folder)

**Example:**
```python
data = np.random.rand(100, 50)
train_size = int(100 * 0.8)
subsamples_folder, indices_folder = iterative_clustering_subsampling(
    data, 100, 200, train_size, './output'
)
```

#### `kmeans_clustering()`

Perform K-Means clustering with ERICA analysis.

```python
kmeans_clustering(
    samples_array,
    k,
    n_iterations,
    indices_folder,
    output_dir,
    random_state=0,
    verbose=False
)
```

**Parameters:**
- `samples_array` (np.ndarray): Input data
- `k` (int): Number of clusters
- `n_iterations` (int): Number of iterations
- `indices_folder` (str): Path to indices folder
- `output_dir` (str): Output directory
- `random_state` (int): Random seed for K-Means
- `verbose` (bool): Print progress

**Returns:** dict with keys:
- `'clam_matrix'`: Final CLAM matrix
- `'global_centroids'`: Sorted global centroids
- `'aligned_predictions'`: Aligned prediction matrix
- `'output_folder'`: Output directory path

#### `agglomerative_clustering()`

Perform Agglomerative clustering with ERICA analysis.

```python
agglomerative_clustering(
    samples_array,
    k,
    linkage,
    n_iterations,
    indices_folder,
    output_dir,
    verbose=False
)
```

**Parameters:**
- `samples_array` (np.ndarray): Input data
- `k` (int): Number of clusters
- `linkage` (str): Linkage method ('single', 'ward', 'complete', 'average')
- `n_iterations` (int): Number of iterations
- `indices_folder` (str): Path to indices folder
- `output_dir` (str): Output directory
- `verbose` (bool): Print progress

**Returns:** dict (same structure as kmeans_clustering)

---

## Metrics Module

### Functions

#### `compute_cri()`

Compute Clustering Replicability Index.

```python
compute_cri(clam_matrix, k)
```

**Parameters:**
- `clam_matrix` (np.ndarray): CLAM matrix (n_samples, k)
- `k` (int): Number of clusters

**Returns:** np.ndarray of CRI values for each cluster

**Example:**
```python
clam = np.array([[50, 10], [45, 15], [5, 55]])
cri = compute_cri(clam, k=2)
```

#### `compute_wcri()`

Compute Weighted CRI.

```python
compute_wcri(clam_matrix, k)
```

**Returns:** tuple of (wcri_per_cluster, mean_wcri)

#### `compute_twcri()`

Compute Total Weighted CRI.

```python
compute_twcri(clam_matrix, k)
```

**Returns:** float (TWCRI value)

#### `compute_metrics_for_clam()`

Compute all metrics for a CLAM matrix.

```python
compute_metrics_for_clam(clam_matrix, k)
```

**Returns:** dict with keys:
- `'CRI'`: Mean CRI
- `'CRI_per_cluster'`: CRI for each cluster
- `'WCRI'`: Mean WCRI
- `'WCRI_per_cluster'`: WCRI for each cluster
- `'TWCRI'`: Total weighted CRI
- `'cluster_sizes'`: Size of each cluster
- `'k'`: Number of clusters

#### `select_optimal_k()`

Select optimal K using ERICA Algorithm 2 (non-decreasing metric selection).

```python
select_optimal_k(metric_dict, k_max=None)
```

**Parameters:**
- `metric_dict` (dict): Dictionary mapping K values to metric scores (e.g., {2: 0.71, 3: 0.75, ...})
- `k_max` (int, optional): Maximum K to consider

**Returns:** int (optimal K value)

**Notes:** Implements Algorithm 2 from the ERICA paper. Prefers the largest K where the metric is non-decreasing. Automatically skips NaN values (e.g., from empty clusters).

#### `select_optimal_k_by_method()`

Select optimal K for each clustering method using Algorithm 2.

```python
select_optimal_k_by_method(metrics_by_k, metric_name='TWCRI')
```

**Parameters:**
- `metrics_by_k` (dict): Nested dictionary {k: {method: metrics_dict}}
- `metric_name` (str): Metric to use ('CRI', 'WCRI', or 'TWCRI')

**Returns:** dict mapping method names to their optimal K values

---

## Data Module

### Functions

#### `load_data()`

Load data from file.

```python
load_data(filepath)
```

**Parameters:**
- `filepath` (str): Path to .npy or .csv file

**Returns:** np.ndarray or pd.DataFrame

#### `prepare_samples_array()`

Convert data to numeric samples array.

```python
prepare_samples_array(data)
```

**Parameters:**
- `data` (np.ndarray or pd.DataFrame): Input data

**Returns:** np.ndarray with shape (n_samples, n_features)

**Example:**
```python
df = pd.DataFrame({
    'gene_id': ['gene1', 'gene2'],
    'sample1': [1.0, 2.0],
    'sample2': [3.0, 4.0]
})
array = prepare_samples_array(df)  # Shape: (2, 2)
```

#### `validate_dataset()`

Validate dataset meets minimum requirements.

```python
validate_dataset(samples_array, min_k, train_percent)
```

**Parameters:**
- `samples_array` (np.ndarray): Data array
- `min_k` (int): Minimum number of clusters
- `train_percent` (float): Training percentage

**Raises:** ValueError if validation fails

#### `save_clam_matrix()`

Save CLAM matrix to file.

```python
save_clam_matrix(clam_matrix, filepath, format='csv')
```

**Parameters:**
- `clam_matrix` (np.ndarray): CLAM matrix
- `filepath` (str): Output path
- `format` (str): 'csv' or 'npy'

#### `load_clam_matrix()`

Load CLAM matrix from file.

```python
load_clam_matrix(filepath)
```

**Returns:** np.ndarray

---

## Plotting Module

### Functions

#### `plot_metrics()`

Create line plot of metrics vs k.

```python
plot_metrics(k_values, cri_values, wcri_values, twcri_values, title=...)
```

**Returns:** plotly Figure

#### `plot_optimal_k()`

Create bar plot of optimal k values.

```python
plot_optimal_k(optimal_k_by_metric, title=...)
```

**Returns:** plotly Figure

#### `plot_clam_heatmap()`

Create heatmap visualization of CLAM matrix.

```python
plot_clam_heatmap(clam_matrix, k, title=None)
```

**Returns:** plotly Figure

#### `create_metrics_plots()`

Create comprehensive metrics plots.

```python
create_metrics_plots(metrics_data, show_optimal=True)
```

**Returns:** tuple of (metrics_plot, optimal_k_plot, summary_text)

---

## Utils Module

### Functions

#### `set_deterministic_mode()`

Set deterministic mode for reproducibility.

```python
set_deterministic_mode(seed)
```

**Parameters:**
- `seed` (int): Random seed

#### `compute_config_hash()`

Compute hash of configuration.

```python
compute_config_hash(config)
```

**Returns:** str (12-character hash)

#### `check_dependencies()`

Check if dependencies are available.

```python
check_dependencies()
```

**Returns:** dict mapping dependency names to availability (bool)

#### `validate_config()`

Validate configuration parameters.

```python
validate_config(config)
```

**Raises:** ValueError if invalid

---

## Examples

### Basic Usage

```python
import numpy as np
from erica import ERICA

# Generate or load data
data = np.random.rand(100, 50)  # 100 samples, 50 features

# Run ERICA analysis
erica = ERICA(
    data=data,
    k_range=[2, 3, 4, 5],
    n_iterations=200,
    method='both'
)

results = erica.run()

# Get CLAM matrix
clam = erica.get_clam_matrix(k=3, method='kmeans')

# Get metrics
metrics = erica.get_metrics()

# Create plots
fig1, fig2 = erica.plot_metrics()
fig1.show()
```

### Using Individual Components

```python
from erica.clustering import kmeans_clustering, iterative_clustering_subsampling
from erica.metrics import compute_metrics_for_clam
from erica.plotting import plot_metrics
from erica.data import load_data, prepare_samples_array

# Load and prepare data
data = load_data('my_data.csv')
samples = prepare_samples_array(data)

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

print(f"CRI: {metrics['CRI']:.3f}")
print(f"WCRI: {metrics['WCRI']:.3f}")
print(f"TWCRI: {metrics['TWCRI']:.3f}")
```


