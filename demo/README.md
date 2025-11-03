# ERICA Demo Interface

This folder contains a comprehensive Gradio interface for testing and exploring the ERICA library.

## Installation

Make sure you have Gradio installed:

```bash
pip install gradio>=4.0.0
```

Or install with the GUI extras:

```bash
pip install erica-clustering[gui]
```

## Running the Demo

From the project root directory:

```bash
python demo/gradio_demo.py
```

This will launch a web interface at `http://localhost:7860`

## Features

The demo interface includes:

### 1. **Data Loading**
- Upload CSV or NPY files
- Preview dataset information
- Toggle data transposition (for genomics vs standard ML format)

### 2. **Parameter Configuration**
- Set K range (comma-separated or range format)
- Configure number of iterations
- Adjust training percentage
- Select clustering method (K-Means, Agglomerative, or both)
- Set linkage methods for hierarchical clustering
- Configure random seed for reproducibility

### 3. **Run Analysis**
- Execute ERICA analysis with configured parameters
- View detailed results summary
- See K* selection results

### 4. **View Metrics**
- Interactive metrics table showing CRI, WCRI, and TWCRI
- Organized by K value and clustering method

### 5. **Visualizations**
- **Metrics Plot**: Line plots showing how metrics change across K values
- **CLAM Heatmap**: Visualize cluster assignment matrices
- **Cluster Sizes**: Bar plots showing cluster size distributions
- **K* Selection**: Plots showing optimal K* selection for each method

## Usage Tips

1. **Data Format**:
   - For genomics data (features in rows, samples in columns): Enable "Transpose Data"
   - For standard ML data (samples in rows, features in columns): Disable "Transpose Data"

2. **K Range Format**:
   - Comma-separated: `2,3,4,5`
   - Range: `2-5` (gives [2,3,4,5])
   - Range with step: `2-10:2` (gives [2,4,6,8,10])

3. **Performance**:
   - Start with fewer iterations (50-100) for quick testing
   - Increase iterations (200-500) for more stable results
   - K-Means is faster than Agglomerative clustering

4. **Testing Different Methods**:
   - Use "kmeans" for faster analysis
   - Use "agglomerative" to test hierarchical clustering
   - Use "both" to compare methods side-by-side

## Example Workflow

1. **Load Data**: Upload a CSV or NPY file and preview the dataset
2. **Configure**: Set your K range, iterations, and method
3. **Run**: Execute the analysis and wait for results
4. **Explore**: View metrics table and generate visualizations
5. **Analyze**: Use K* plots to determine optimal number of clusters

## Troubleshooting

- **Import Errors**: Make sure you're running from the project root directory
- **Data Errors**: Check your data orientation (try toggling transpose)
- **Memory Issues**: Reduce iterations or use smaller K ranges
- **Plot Errors**: Ensure plotly is installed: `pip install plotly>=5.0.0`

## File Structure

```
demo/
├── gradio_demo.py    # Main Gradio interface
└── README.md         # This file
```

## Extending the Demo

The demo interface is designed to be easily extensible. You can:

- Add new visualization functions
- Test custom metrics
- Experiment with different parameter combinations
- Compare results across different datasets

For development, modify `demo/gradio_demo.py` and reload the interface.

