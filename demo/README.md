# 🧬 ERICA Demo Interface

This folder contains a comprehensive Gradio-based web interface for testing and exploring the ERICA (Evaluating Replicability via Iterative Clustering Assignments) library. The demo provides an interactive way to upload datasets, configure clustering parameters, run analyses, and visualize results—all through an intuitive web interface.

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

The demo interface is organized into 5 main tabs:

### Tab 1: Data Loading
- **Upload Files**: Support for CSV and NPY formats
- **Dataset Preview**: View shape, statistics, and data properties
- **Transpose Option**: Toggle between genomics format (features×samples) and ML format (samples×features)
- **Automatic Validation**: Checks for NaN, Inf, and data type issues

### Tab 2: Parameter Configuration
- **K Range**: Multiple format support
  - Comma-separated: `2,3,4,5`
  - Range: `2-5` → [2, 3, 4, 5]
  - Range with step: `2-10:2` → [2, 4, 6, 8, 10]
- **Iterations**: Control Monte Carlo subsampling iterations (10-1000)
- **Training Percentage**: Adjust train/test split ratio (50-95%)
- **Clustering Method**: Choose kmeans, agglomerative, or both
- **Linkage Methods**: Configure hierarchical clustering linkages (single, complete, average, ward)
- **Random Seed**: Set seed for reproducible results
- **Output Directory**: Specify where to save results

### Tab 3: Run Analysis
- **Execute Analysis**: Run ERICA with configured parameters
- **Real-time Progress**: View detailed progress messages
- **Results Summary**: See metrics, K* values, and output locations
- **Automatic Refresh**: Updates visualization options after completion

### Tab 4: View Metrics
- **Interactive Table**: Comprehensive metrics for all K values and methods
- **Metrics Display**:
  - **CRI** (Clustering Replicability Index)
  - **WCRI** (Weighted CRI)
  - **TWCRI** (Training-Weighted CRI)
- **Sortable Results**: Easy comparison across methods

### Tab 5: Visualizations
- **Metrics Plot**: Line charts showing metric trends across K values
- **CLAM Heatmap**: Interactive heatmap of cluster assignment matrices
- **Cluster Sizes**: Bar plots showing cluster size distributions
- **K* Selection Plots**:
  - Bar chart comparing K* across methods
  - Detailed K* selection with elbow detection
- **Dynamic Updates**: Plots update automatically when changing K or method

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

Here's a complete workflow for analyzing a dataset:

### 1. Load Your Data
- Navigate to **Tab 1: Load Data**
- Click **"Data File"** and upload your CSV or NPY file
- Check **"Transpose Data"** if using genomics format
- Click **"Load & Preview Data"** to see dataset information
- Verify the shape and statistics look correct

### 2. Configure Parameters
- Navigate to **Tab 2: Configure Parameters**
- Set **K Range**: `2-6` (for initial exploration)
- Set **Iterations**: `100` (for quick testing)
- Set **Training Percentage**: `0.8`
- Select **Method**: `kmeans` (fastest) or `both` (comprehensive)
- Keep default **Random Seed** for reproducibility
- Set **Output Directory**: `./erica_output`

### 3. Run Analysis
- Navigate to **Tab 3: Run Analysis**
- Click **"Run Analysis"**
- Monitor progress in the output window
- Wait for completion message showing:
  - Data shape
  - Metrics summary for each K
  - K* selection results

### 4. View Metrics
- Navigate to **Tab 4: View Metrics**
- Click **"Refresh Metrics Table"**
- Review CRI, WCRI, and TWCRI values
- Compare metrics across different K values
- Identify trends and patterns

### 5. Explore Visualizations
- Navigate to **Tab 5: Visualizations**
- **Metrics Plot**: Click "Generate Metrics Plot" to see trends
- **CLAM Heatmap**: Select K and method, view cluster stability
- **Cluster Sizes**: Check cluster size distributions
- **K* Selection**: Click "Generate K* Plots" to see optimal K recommendations

### 6. Interpret Results
- **High CRI/WCRI/TWCRI** (>0.7): Stable, replicable clusters
- **Moderate values** (0.5-0.7): Moderately stable clusters
- **Low values** (<0.5): Unstable clusters, consider different K or method
- **K* value**: The optimal number of clusters based on stability metrics

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

## File Structure

```
demo/
├── gradio_demo.py    # Main Gradio web interface with all UI components
├── test_demo.py      # Test script to verify installation and dependencies
├── README.md         # This documentation file
└── SUMMARY.md        # Technical summary of demo implementation
```

### File Descriptions

- **gradio_demo.py**: Complete Gradio interface with 5 tabs (Data Loading, Configuration, Analysis, Metrics, Visualizations). Includes helper functions for data parsing, plot generation, and result management.

- **test_demo.py**: Lightweight testing script that verifies imports and basic functionality. Run this first to ensure everything is set up correctly.

- **README.md**: This file. Comprehensive user guide for the demo interface.

- **SUMMARY.md**: Technical implementation details and architecture overview.

## Extending the Demo

The demo interface is designed to be modular and easily extensible. Here are some ways to customize it:

### Adding New Visualizations

To add a new plot type:

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

2. Add a new plot component in the Visualizations tab:
```python
with gr.Tab("5. Visualizations"):
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

You can modify default values or add new parameters in the Configuration tab by adding new Gradio components (Slider, Dropdown, Textbox, etc.) and passing them to the `run_erica_analysis` function.

### Using with Custom Datasets

The demo works with any numeric dataset. For best results:
- Ensure data is properly normalized/scaled
- Remove missing values or handle them appropriately
- Use appropriate transpose setting for your data format

## Best Practices

### For Exploratory Analysis
- Start with K range 2-5 and 50-100 iterations
- Use kmeans for speed
- Review plots to understand data structure
- Iterate with refined parameters

### For Publication-Quality Results
- Use 500-1000 iterations
- Test both kmeans and agglomerative methods
- Compare multiple linkage functions
- Use multiple random seeds to verify stability
- Document all parameter choices

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

