# ERICA Demo Interface

This folder contains a streamlined Gradio-based web interface for the ERICA (Evaluating Replicability via Iterative Clustering Assignments) library. The demo provides an intuitive way to upload datasets, configure clustering parameters, run analyses, and visualize results through a clean, professional interface.

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Testing the Demo](#testing-the-demo)
- [Features](#features)
- [Usage Guide](#usage-guide)
- [Example Workflow](#example-workflow)
- [Troubleshooting](#troubleshooting)
- [File Structure](#file-structure)
- [Extending the Demo](#extending-the-demo)

## Installation

### Option 1: Install with GUI extras

```bash
pip install erica-clustering[gui]
```

### Option 2: Install Gradio separately

```bash
pip install gradio>=4.0.0
```

### Required Dependencies

The demo requires the following packages:
- `gradio>=4.0.0` - Web interface framework
- `plotly>=5.0.0` - Interactive visualizations
- `numpy` - Numerical operations
- `pandas` - Data handling
- All ERICA core dependencies

## Quick Start

### Running the Demo

From the project root directory:

```bash
python demo/gradio_demo.py
```

The demo will automatically find an available port (starting from 7860) and launch a web interface:

```
============================================================
Starting ERICA Gradio Demo Server...
Open your browser to: http://localhost:7860
============================================================
```

Open the displayed URL in your web browser to access the interface.

## Testing the Demo

Before running the full demo, you can verify that all dependencies are correctly installed:

```bash
python demo/test_demo.py
```

This test script will:
- ✓ Verify all required imports (Gradio, ERICA, plotting utilities)
- ✓ Test basic data processing functionality
- ✓ Confirm that the demo is ready to run

If all tests pass, you'll see:

```
============================================================
✓ All tests passed! You can run the demo with:
  python demo/gradio_demo.py
============================================================
```

## Features

The demo interface is organized into 3 streamlined tabs:

### Tab 1: Load Data
The data loading tab has two sub-tabs for flexibility:

**Upload New Data:**
- **Upload Files**: Support for CSV and NPY formats
- **Dataset Preview**: Detailed preview showing:
  - File information and shape
  - Sample and feature counts
  - Statistical summaries (min, max, mean, std)
  - Data validation (NaN, Inf detection)
  - First 5 and last 3 rows of data
- **Transpose Option**: Toggle between genomics format (features×samples) and ML format (samples×features)
- **Header Detection**: Automatically identifies gene ID or feature name headers

**Load Previous Run:**
- **List Available Runs**: View up to 10 most recent ERICA runs
- **Refresh List**: Update the list of available runs
- **Load Previous Analysis**: Restore a completed analysis without re-running
- **Run Directory**: Enter path to a previous run directory
- **Automatic Summary**: See all parameters, metrics, and K* selections from the loaded run
- **Seamless Integration**: Once loaded, view visualizations in Tab 3 as normal

### Tab 2: Configure & Run Analysis
**Configuration Section:**
- **K Range**: Multiple format support
  - Comma-separated: `2,3,4,5`
  - Range: `2-5` → [2, 3, 4, 5]
  - Range with step: `2-10:2` → [2, 4, 6, 8, 10]
- **Iterations**: Control Monte Carlo subsampling (10-1000 iterations)
- **Training Percentage**: Adjust train/test split ratio (50-95%)
- **Clustering Method**: Choose kmeans, agglomerative, or both
- **Linkage Methods**: Configure hierarchical clustering (single, complete, average, ward)
- **Random Seed**: Set seed for reproducibility
- **Output Directory**: Specify results location
- **Verbose Output**: Toggle detailed progress messages

**Run Section:**
- **Status Indicator**: Real-time status updates
- **Run Button**: Execute analysis with configured parameters
- **Results Summary**: Comprehensive output showing:
  - Elapsed time
  - Data shape and parameters used
  - Metrics for each K value
  - Optimal K* selections
  - Next steps guidance

### Tab 3: Results & Visualizations
**Metrics Summary:**
- **Interactive Table**: All metrics for each K value and method
- **K* Highlighting**: Bold rows indicate optimal selections
- **Metrics Display**:
  - CRI (Clustering Replicability Index)
  - WCRI (Weighted CRI)
  - TWCRI (Training-Weighted CRI)
  - Optimal K* column showing which metrics selected each K
- **Auto-Refresh**: One button updates both table and plot

**Visualizations:**
- **Metrics Line Plot**: Trends across K values (auto-generated with metrics)
- **Detailed Analysis Section**:
  - K and Method selectors
  - CLAM Heatmap: Cluster assignment stability visualization
  - Cluster Size Distribution: Bar chart of cluster sizes
  - Auto-updates when selection changes

## Usage Guide

### Data Format Requirements

**Supported File Types:**
- **CSV files**: Comma-separated values with or without headers
- **NPY files**: NumPy binary arrays

**Data Orientation:**
- **Genomics/Biology format**: Features in rows, samples in columns → **Enable "Transpose Data"**
- **Machine Learning format**: Samples in rows, features in columns → **Disable "Transpose Data"**

### Parameter Selection Guidelines

**K Range:**
- Start with `2-5` or `2-6` for initial exploration
- Extend range if K* values approach upper limit
- Use step format (`2-20:2`) for large ranges to save time

**Iterations:**
- **Quick testing**: 50-100 iterations
- **Standard analysis**: 100-200 iterations
- **Publication quality**: 500-1000 iterations
- More iterations = more stable metrics but slower runtime

**Training Percentage:**
- **Default**: 0.80 (80% training, 20% testing)
- **Small datasets** (<50 samples): Use 0.70-0.75
- **Large datasets** (>200 samples): Can use 0.85-0.90

**Clustering Methods:**
- **kmeans**: Fastest, good for spherical clusters
- **agglomerative**: Better for hierarchical structures, multiple linkage options
- **both**: Compare results across methods (recommended for thorough analysis)

## Example Workflow

Here's a streamlined workflow for analyzing a dataset:

### 1. Load Your Data (or Previous Run)

**Option A: Upload New Data**
- Navigate to **Tab 1: Load Data** → **Upload New Data** sub-tab
- Click **"Data File"** and upload your CSV or NPY file
- Check **"Transpose Data"** if using genomics format (features in rows)
- Click **"Load & Preview Data"** to see:
  - Dataset shape and statistics
  - First 5 and last 3 rows
  - Data validation results
- Verify everything looks correct before proceeding

**Option B: Load Previous Run**
- Navigate to **Tab 1: Load Data** → **Load Previous Run** sub-tab
- Review the list of available runs (most recent first)
- Copy a run directory path from the list
- Paste it into "Previous Run Directory" field
- Click **"Load Previous Run"**
- Review the summary showing all parameters and metrics
- Skip to Step 4 to view visualizations

### 2. Configure & Run Analysis
- Navigate to **Tab 2: Configure & Run Analysis**

**Configure Parameters:**
- Set **K Range**: `2-6` (for initial exploration)
- Set **Iterations**: `100` (for quick testing) or `500` (for publication)
- Set **Training Percentage**: `0.8` (standard) or adjust based on sample size
- Select **Method**: `kmeans` (fastest) or `both` (comprehensive comparison)
- Keep default **Random Seed** (`123`) for reproducibility
- Set **Output Directory**: `./erica_output`
- Enable **Verbose Output** if you want detailed progress

**Run Analysis:**
- Scroll down to the "Run Analysis" section
- Check the status indicator (should say "Ready to run analysis")
- Click **"Run Analysis"**
- Watch the status indicator update
- View comprehensive results showing:
  - Elapsed time
  - Data shape and parameters
  - Metrics for each K value
  - Optimal K* selections
  - Next steps

### 3. View Results & Visualizations
- Navigate to **Tab 3: Results & Visualizations**

**Review Metrics:**
- Click **"Refresh Results"** to update both table and plot
- Review the metrics table:
  - **Bold rows** indicate optimal K* selections
  - Check "Optimal K*" column to see which metrics selected each K
  - Compare CRI, WCRI, and TWCRI values across K values
- View the metrics line plot showing trends across K

**Explore Detailed Analysis:**
- Select a **K value** and **Method** from the dropdowns
- View **CLAM Heatmap**: Shows cluster assignment stability
- View **Cluster Size Distribution**: Check if clusters are balanced
- Plots update automatically when you change selections

### 4. Interpret Results
- **High CRI/WCRI/TWCRI** (>0.7): Stable, replicable clusters
- **Moderate values** (0.5-0.7): Moderately stable clusters
- **Low values** (<0.5): Unstable clusters, consider different K or method
- **K* value**: The optimal number of clusters based on stability metrics
- **Bold rows in table**: These K values show best replicability

## Troubleshooting

### Common Issues and Solutions

**Problem: Import Errors**
```
ImportError: No module named 'gradio'
```
**Solution**: Install Gradio with `pip install gradio>=4.0.0` or `pip install erica-clustering[gui]`

**Problem: Data Loading Errors**
```
Error loading data: Shape mismatch or invalid format
```
**Solution**: 
- Check your data orientation (try toggling "Transpose Data")
- Ensure CSV has numeric data only (no text in data columns)
- Verify NPY files contain 2D arrays

**Problem: Analysis Fails to Start**
```
ERROR: Please upload a data file first.
```
**Solution**: Make sure to load data in Tab 1 before running analysis in Tab 3

**Problem: Memory Errors**
```
MemoryError or system slowdown
```
**Solution**:
- Reduce number of iterations (try 50 instead of 500)
- Reduce K range (try 2-5 instead of 2-20)
- Close other applications to free up memory

**Problem: Plots Not Displaying**
```
Error creating plots or blank plot areas
```
**Solution**:
- Install Plotly: `pip install plotly>=5.0.0`
- Refresh the browser page
- Click "Refresh Options" button in Tab 5

**Problem: Port Already in Use**
```
OSError: Address already in use
```
**Solution**: The demo automatically finds available ports. If issues persist, manually kill the process using the port or specify a different port in the code.

**Problem: Slow Performance**
```
Analysis takes too long to complete
```
**Solution**:
- Reduce iterations (100 instead of 500)
- Use kmeans instead of agglomerative
- Reduce K range
- Consider using a smaller dataset for testing

### Getting Help

If you encounter issues not listed here:
1. Check the main [GETTING_STARTED.md](../docs/GETTING_STARTED.md) documentation
2. Review the [API_REFERENCE.md](../docs/API_REFERENCE.md) for detailed parameter information
3. Ensure all dependencies are installed correctly by running `python demo/test_demo.py`
4. Check that your data format matches the expected input format

## Saved Output Structure

Each ERICA run automatically saves its results to a timestamped directory:

```
./erica_output/erica_run_20251103_182956/
├── kmeans_k2/
│   └── cluster_assignments.csv
├── kmeans_k3/
│   └── cluster_assignments.csv
├── kmeans_k4/
│   └── cluster_assignments.csv
├── erica_instance.pkl          # NEW: Complete ERICA state for loading
└── ... (additional K values and methods)
```

**What Gets Saved:**
- **CSV files**: Cluster assignments for each K and method
- **erica_instance.pkl**: Complete ERICA instance with all results, metrics, and parameters
- **Metrics**: All CRI, WCRI, TWCRI values for each K
- **K* selections**: Optimal K values for each metric
- **Configuration**: All parameters used (K range, iterations, method, etc.)

**Benefits of Saved State:**
- Resume analysis without re-running
- Share results with collaborators (just send the directory)
- Compare multiple runs easily
- Create new visualizations from old runs
- Preserve long-running analyses

## File Structure

```
demo/
├── gradio_demo.py    # Main Gradio web interface (streamlined 3-tab design)
├── test_demo.py      # Test script to verify installation and dependencies
├── README.md         # This documentation file
└── SUMMARY.md        # Technical summary of demo implementation
```

### File Descriptions

- **gradio_demo.py**: Professional Gradio interface with 3 streamlined tabs:
  - Tab 1: Data loading (new data or previous runs)
  - Tab 2: Combined configuration and execution
  - Tab 3: Results with auto-refreshing visualizations
  - Includes helper functions for data parsing, plot generation, and result management
  - Automatic state persistence for loading previous runs
  - ~1000 lines of clean, well-documented code

- **test_demo.py**: Lightweight testing script that verifies imports and basic functionality. Run this first to ensure everything is set up correctly.

- **README.md**: This file. Comprehensive user guide for the demo interface.

- **SUMMARY.md**: Technical implementation details and architecture overview.

## Extending the Demo

The demo interface is designed to be modular and easily extensible. Here are some ways to customize it:

### Adding New Visualizations

To add a new plot type in Tab 3:

1. Create a new plotting function in the global scope:
```python
def create_custom_plot() -> Optional[object]:
    """Create your custom plot."""
    global erica_instance
    if erica_instance is None:
        return None
    # Your plotting logic here
    return fig
```

2. Add the plot component in Tab 3 (Results & Visualizations):
```python
with gr.Tab("3. Results & Visualizations"):
    # Add after existing visualizations
    gr.Markdown("### Custom Analysis")
    custom_plot = gr.Plot(label="My Custom Plot")
    refresh_custom_btn = gr.Button("Generate Custom Plot")
    refresh_custom_btn.click(
        fn=create_custom_plot,
        outputs=[custom_plot]
    )
```

### Adding New Metrics

To display custom metrics:

1. Compute metrics in your analysis
2. Store in the ERICA instance
3. Add display logic in the metrics table generation function

### Customizing Parameters

You can modify default values or add new parameters in Tab 2 by adding new Gradio components (Slider, Dropdown, Textbox, etc.) and passing them to the `run_erica_analysis` function. Remember to update the `run_and_refresh` function to include your new parameters.

### Using with Custom Datasets

The demo works with any numeric dataset. For best results:
- Ensure data is properly normalized/scaled
- Remove missing values or handle them appropriately
- Use appropriate transpose setting for your data format

## Best Practices

### For Exploratory Analysis
1. Load your data and verify the preview (Tab 1)
2. Start with K range 2-5 and 100 iterations (Tab 2)
3. Use kmeans for speed
4. Run analysis and review metrics table (Tab 2)
5. Check visualizations to understand data structure (Tab 3)
6. Iterate with refined parameters as needed

### For Publication-Quality Results
1. Use 500-1000 iterations for stable estimates
2. Test both kmeans and agglomerative methods
3. Compare multiple linkage functions (single, ward, complete)
4. Run with multiple random seeds to verify stability
5. Document all parameter choices in your methods
6. Save comprehensive results from Tab 2 output
7. Export visualizations from Tab 3 for figures

### For Large Datasets (>1000 samples)
- Start with higher training percentage (0.85-0.90)
- Use kmeans (much faster than agglomerative)
- Consider using step format for K range (e.g., `2-20:2`)
- Monitor memory usage

### For Small Datasets (<50 samples)
- Use lower training percentage (0.70-0.75)
- More iterations for stability (200-500)
- Consider using agglomerative clustering
- Be cautious interpreting results with very small samples

## Example Use Cases

### Genomics/Transcriptomics Data
- **Goal**: Identify stable patient subtypes
- **Format**: Gene expression matrix (genes × patients)
- **Settings**: Transpose=True, K=2-6, iterations=500, method=both
- **Interpretation**: High TWCRI indicates reproducible subtypes

### Customer Segmentation
- **Goal**: Find stable customer segments
- **Format**: Customer features matrix (customers × features)
- **Settings**: Transpose=False, K=2-10, iterations=200, method=kmeans
- **Interpretation**: K* suggests optimal segment count

### Image Feature Clustering
- **Goal**: Group similar image features
- **Format**: Feature vectors (images × features)
- **Settings**: Transpose=False, K=5-15:2, iterations=100, method=kmeans
- **Interpretation**: CLAM heatmap shows which clusters are stable

## Additional Resources

- **Main Documentation**: See [GETTING_STARTED.md](../docs/GETTING_STARTED.md) for detailed ERICA usage
- **API Reference**: See [API_REFERENCE.md](../docs/API_REFERENCE.md) for complete API documentation
- **Methodology**: See [METHODOLOGY.md](../docs/METHODOLOGY.md) for algorithmic details
- **Examples**: See [examples/](../examples/) for code examples

## Contributing

To contribute improvements to the demo:

1. Test your changes with `python demo/test_demo.py`
2. Ensure all features work as expected
3. Update this README if you add new features
4. Follow the coding style in existing demo code

---

**Happy Clustering!** 🎯

For questions or issues, please refer to the main project documentation or open an issue on the repository.

