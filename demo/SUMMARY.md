# Demo Interface Summary

## What Was Created

A comprehensive Gradio demo interface has been created in the `demo/` folder to test and explore the ERICA library.

### Files Created

1. **`demo/gradio_demo.py`** - Main Gradio interface application
2. **`demo/README.md`** - Documentation for using the demo
3. **`demo/test_demo.py`** - Test script to verify imports and basic functionality

## Features

The demo interface provides:

### 1. Data Loading Tab
- File upload (CSV, NPY)
- Data preview with statistics
- Transpose toggle for different data orientations

### 2. Configuration Tab
- K range input (flexible formats)
- Iterations slider
- Training percentage
- Method selection (K-Means, Agglomerative, or both)
- Linkage methods for hierarchical clustering
- Random seed for reproducibility
- Output directory configuration

### 3. Run Analysis Tab
- Execute ERICA analysis
- View detailed results summary
- Automatic refresh of visualization options

### 4. Metrics Tab
- Interactive metrics table
- CRI, WCRI, TWCRI values organized by K and method
- Refresh button for manual updates

### 5. Visualizations Tab
- **Metrics Plot**: Line plots showing metric trends across K values
- **CLAM Heatmap**: Interactive heatmap visualization
- **Cluster Sizes**: Bar plots showing cluster distributions
- **K* Selection**: Plots showing optimal K* selection

## Usage

1. **Install dependencies**:
   ```bash
   pip install gradio>=4.0.0
   ```

2. **Run the demo**:
   ```bash
   python demo/gradio_demo.py
   ```

3. **Access the interface**:
   - Open browser to `http://localhost:7860`
   - Follow the tabs in order: Load Data → Configure → Run → View Metrics → Visualizations

## Testing

Run the test script to verify everything works:

```bash
python demo/test_demo.py
```

## Implementation Details

- Uses global state to store ERICA instance and results
- Supports flexible K range parsing (comma-separated, ranges, step ranges)
- Automatic plot option refresh after analysis
- Error handling with detailed error messages
- Interactive Plotly visualizations embedded in Gradio

## Extending the Demo

The demo can be easily extended by:

1. Adding new visualization functions
2. Testing custom metrics
3. Adding new parameter options
4. Creating comparison views between different runs

## Next Steps

- Test with real datasets
- Add more visualization options
- Extend metrics testing
- Add export functionality for results
- Create comparison workflows

