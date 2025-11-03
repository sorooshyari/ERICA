"""
ERICA Gradio Demo Interface

A comprehensive Gradio interface for testing and exploring the ERICA library.
This demo allows you to:
- Load data from files (CSV, NPY)
- Configure clustering parameters
- Run ERICA analysis
- View metrics and visualizations
- Test different clustering methods
"""

import os
import sys
import traceback
from pathlib import Path
from typing import Tuple, Optional, Dict, List
import numpy as np
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import gradio as gr
except ImportError:
    print("ERROR: Gradio not installed. Install with: pip install gradio>=4.0.0")
    sys.exit(1)

from erica import ERICA
from erica.data import load_data, get_dataset_info
from erica.plotting import (
    plot_metrics,
    plot_clam_heatmap,
    plot_k_star_selection,
    plot_k_star_by_method,
    plot_cluster_sizes,
)
from erica.metrics import summarize_metrics


# Global state to store ERICA instance and results
erica_instance = None
erica_results = None


def parse_k_range(k_range_str: str) -> List[int]:
    """Parse k range string into list of integers.
    
    Supports formats:
    - "2,3,4,5" -> [2, 3, 4, 5]
    - "2-5" -> [2, 3, 4, 5]
    - "2-10:2" -> [2, 4, 6, 8, 10]
    """
    if not k_range_str or not k_range_str.strip():
        return [2, 3, 4, 5]
    
    k_range_str = k_range_str.strip()
    
    # Try comma-separated
    if ',' in k_range_str:
        try:
            return [int(k.strip()) for k in k_range_str.split(',')]
        except ValueError:
            pass
    
    # Try range format (e.g., "2-5" or "2-10:2")
    if '-' in k_range_str:
        try:
            parts = k_range_str.replace(':', '-').split('-')
            if len(parts) == 2:
                start, end = int(parts[0]), int(parts[1])
                return list(range(start, end + 1))
            elif len(parts) == 3:
                start, end, step = int(parts[0]), int(parts[1]), int(parts[2])
                return list(range(start, end + 1, step))
        except ValueError:
            pass
    
    # Try single number
    try:
        return [int(k_range_str)]
    except ValueError:
        return [2, 3, 4, 5]


def load_and_preview_data(file, transpose: bool) -> Tuple[str, Optional[Dict]]:
    """Load data file and return preview."""
    if file is None:
        return "No file uploaded", None
    
    try:
        # Load data
        data = load_data(file.name)
        
        # Convert to numpy array if needed
        if isinstance(data, pd.DataFrame):
            data_array = data.values
        else:
            data_array = np.array(data)
        
        # Get dataset info
        info = get_dataset_info(data, transpose=transpose)
        
        # Apply transpose if needed to get the correct orientation for display
        if transpose:
            data_array = data_array.T
        
        # Check for non-numeric data
        has_non_numeric = False
        header_row = False
        if data_array.dtype == object or data_array.dtype.kind in ['U', 'S', 'O']:
            # Check if only first row is non-numeric (likely headers)
            try:
                if data_array.shape[0] > 1:
                    # Try to convert everything except first row to float
                    test_conversion = data_array[1:].astype(float)
                    header_row = True  # First row is headers, rest is numeric
                else:
                    has_non_numeric = True
            except (ValueError, TypeError):
                has_non_numeric = True  # Mixed types throughout dataset
        
        # Create preview
        preview_lines = [
            "=" * 80,
            "DATASET INFORMATION",
            "=" * 80,
            f"File: {Path(file.name).name}",
            f"Original shape: {data.shape if hasattr(data, 'shape') else 'N/A'}",
            f"After processing: {info['shape']}",
            f"  Samples: {info['n_samples']}",
            f"  Features: {info['n_features']}",
            f"  Transpose applied: {transpose}",
            "",
            "Statistics:",
            f"  Data type: {info['dtype']}",
            f"  Min value: {info['min_value']:.6f}",
            f"  Max value: {info['max_value']:.6f}",
            f"  Mean value: {info['mean_value']:.6f}",
            f"  Std value: {info['std_value']:.6f}",
            f"  Has NaN: {info['has_nan']}",
            f"  Has Inf: {info['has_inf']}",
        ]
        
        # Add info/warning messages about data types
        if header_row:
            preview_lines.extend([
                "",
                "INFO: Header row detected (gene IDs or feature names).",
                "  First row contains labels and will be shown in the preview.",
                "  ERICA will use the numeric data rows for clustering analysis.",
            ])
        elif has_non_numeric:
            preview_lines.extend([
                "",
                "WARNING: Non-numeric data detected!",
                "  ERICA requires numeric data for clustering analysis.",
                "  The preview below shows the data, but analysis may fail.",
                "  Consider converting/encoding string values to numeric format.",
            ])
        
        preview_lines.extend([
            "",
            "=" * 80,
            "DATA PREVIEW (First 5 samples, First 10 features)",
            "=" * 80,
        ])
        
        # Show head of data
        n_preview_samples = min(5, data_array.shape[0])
        n_preview_features = min(10, data_array.shape[1])
        
        # Create header
        header = "Sample".ljust(10)
        for j in range(n_preview_features):
            header += f"Feat_{j}".rjust(12)
        if data_array.shape[1] > n_preview_features:
            header += "  ..."
        preview_lines.append(header)
        preview_lines.append("-" * 80)
        
        # Show data rows
        for i in range(n_preview_samples):
            row = f"[{i}]".ljust(10)
            for j in range(n_preview_features):
                value = data_array[i, j]
                # Handle different data types
                try:
                    # Try to format as float
                    row += f"{float(value):12.4f}"
                except (ValueError, TypeError):
                    # If not numeric, show as string (truncated to 12 chars)
                    row += f"{str(value)[:12]:>12}"
            if data_array.shape[1] > n_preview_features:
                row += "  ..."
            preview_lines.append(row)
        
        if data_array.shape[0] > n_preview_samples:
            preview_lines.append(f"... ({data_array.shape[0] - n_preview_samples} more samples)")
        
        preview_lines.append("")
        preview_lines.append("=" * 80)
        preview_lines.append("DATA PREVIEW (Last 3 samples, First 10 features)")
        preview_lines.append("=" * 80)
        
        # Show tail of data
        n_tail_samples = min(3, data_array.shape[0])
        start_idx = max(0, data_array.shape[0] - n_tail_samples)
        
        # Create header
        preview_lines.append(header)
        preview_lines.append("-" * 80)
        
        # Show tail rows
        for i in range(start_idx, data_array.shape[0]):
            row = f"[{i}]".ljust(10)
            for j in range(n_preview_features):
                value = data_array[i, j]
                # Handle different data types
                try:
                    # Try to format as float
                    row += f"{float(value):12.4f}"
                except (ValueError, TypeError):
                    # If not numeric, show as string (truncated to 12 chars)
                    row += f"{str(value)[:12]:>12}"
            if data_array.shape[1] > n_preview_features:
                row += "  ..."
            preview_lines.append(row)
        
        preview_lines.append("=" * 80)
        preview_lines.append("Data loaded successfully. Ready for analysis.")
        preview_lines.append("=" * 80)
        
        return "\n".join(preview_lines), info
        
    except Exception as e:
        error_msg = f"Error loading data:\n{str(e)}\n\n{traceback.format_exc()}"
        return error_msg, None


def run_erica_analysis(
    file,
    transpose: bool,
    k_range_str: str,
    n_iterations: int,
    train_percent: float,
    method: str,
    linkages: str,
    random_seed: int,
    output_dir: str,
    verbose: bool,
) -> Tuple[str, Optional[object]]:
    """Run ERICA analysis with given parameters."""
    global erica_instance, erica_results
    
    if file is None:
        return "ERROR: Please upload a data file first.", None
    
    try:
        import time
        import os
        import pickle
        start_time = time.time()
        
        # Parse k range
        k_range = parse_k_range(k_range_str)
        
        # Parse linkages
        linkage_list = [l.strip() for l in linkages.split(',')] if linkages else ['single', 'ward']
        
        # Load data
        data = load_data(file.name)
        
        # Create ERICA instance
        erica_instance = ERICA(
            data=data,
            k_range=k_range,
            n_iterations=n_iterations,
            train_percent=train_percent,
            method=method,
            linkages=linkage_list,
            random_seed=random_seed,
            output_dir=output_dir,
            transpose=transpose,
            verbose=verbose,
        )
        
        # Run analysis
        print(f"\nStarting ERICA analysis... (K={k_range}, iterations={n_iterations})")
        erica_results = erica_instance.run()
        
        # Save the ERICA instance for later loading
        if erica_instance.output_folders_:
            save_dir = erica_instance.output_folders_[0]
            pickle_path = os.path.join(save_dir, "erica_instance.pkl")
            try:
                with open(pickle_path, 'wb') as f:
                    pickle.dump(erica_instance, f)
                print(f"Saved ERICA instance to: {pickle_path}")
            except Exception as e:
                print(f"Warning: Could not save ERICA instance: {e}")
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        time_str = f"{minutes}m {seconds}s" if minutes > 0 else f"{seconds}s"
        
        # Generate summary
        summary_lines = [
            "=" * 80,
            "ANALYSIS COMPLETE",
            "=" * 80,
            "",
            "Analysis Summary:",
            f"  Elapsed time: {time_str}",
            f"  Data shape: {erica_instance.samples_array.shape}",
            f"  K range: {k_range}",
            f"  Iterations: {n_iterations}",
            f"  Method: {method}",
            f"  Output: {erica_instance.output_folders_[0] if erica_instance.output_folders_ else 'N/A'}",
            "",
            "=" * 80,
            "Results Summary:",
            "=" * 80,
        ]
        
        # Add metrics summary
        metrics = erica_instance.get_metrics()
        for k in sorted(metrics.keys()):
            summary_lines.append(f"\nK = {k}:")
            for method_name, metric_dict in metrics[k].items():
                summary_lines.append(f"  {method_name}:")
                summary_lines.append(f"    CRI:   {metric_dict.get('CRI', 0):.6f}")
                summary_lines.append(f"    WCRI:  {metric_dict.get('WCRI', 0):.6f}")
                summary_lines.append(f"    TWCRI: {metric_dict.get('TWCRI', 0):.6f}")
        
        # Add K* summary
        if erica_instance.k_star_:
            summary_lines.append("\n" + "=" * 80)
            summary_lines.append("Optimal K* Selection:")
            summary_lines.append("=" * 80)
            for metric_name in ['CRI', 'WCRI', 'TWCRI']:
                if metric_name in erica_instance.k_star_:
                    summary_lines.append(f"\n{metric_name}:")
                    for method_name, k_star in erica_instance.k_star_[metric_name].items():
                        summary_lines.append(f"  {method_name}: K* = {k_star}")
        
        # Final completion message
        summary_lines.extend([
            "",
            "=" * 80,
            "Analysis complete. Results are ready for visualization.",
            "=" * 80,
            "",
            "Next steps:",
            "  1. View metrics and visualizations in Tab 3",
            "  2. Check output files in the output directory",
            "",
            "=" * 80,
        ])
        
        return "\n".join(summary_lines), erica_instance
        
    except Exception as e:
        error_msg = f"ERROR RUNNING ANALYSIS:\n\n{str(e)}\n\n{traceback.format_exc()}"
        return error_msg, None


def get_metrics_table() -> Optional[str]:
    """Generate metrics table from results with K* highlighting."""
    global erica_instance
    
    if erica_instance is None:
        return "No analysis results available. Please run analysis first."
    
    try:
        metrics = erica_instance.get_metrics()
        k_star = erica_instance.k_star_ if hasattr(erica_instance, 'k_star_') else {}
        
        # Create K* lookup for easy checking
        k_star_lookup = {}
        if k_star:
            for metric_name in ['CRI', 'WCRI', 'TWCRI']:
                if metric_name in k_star:
                    for method_name, k_val in k_star[metric_name].items():
                        if method_name not in k_star_lookup:
                            k_star_lookup[method_name] = {}
                        k_star_lookup[method_name][metric_name] = k_val
        
        lines = [
            "## Clustering Metrics Summary",
            "",
            "| K | Method | CRI | WCRI | TWCRI | Optimal K* |",
            "|---|--------|-----|------|-------|------------|"
        ]
        
        for k in sorted(metrics.keys()):
            for method_name, metric_dict in metrics[k].items():
                cri = metric_dict.get('CRI', 0)
                wcri = metric_dict.get('WCRI', 0)
                twcri = metric_dict.get('TWCRI', 0)
                
                # Check if this K is selected for any metric
                k_star_markers = []
                if method_name in k_star_lookup:
                    for metric_name, selected_k in k_star_lookup[method_name].items():
                        if selected_k == k:
                            k_star_markers.append(metric_name)
                
                k_star_col = ", ".join(k_star_markers) if k_star_markers else "—"
                
                # Add bold formatting for rows with K* selections
                if k_star_markers:
                    lines.append(f"| **{k}** | **{method_name}** | **{cri:.6f}** | **{wcri:.6f}** | **{twcri:.6f}** | **{k_star_col}** |")
                else:
                    lines.append(f"| {k} | {method_name} | {cri:.6f} | {wcri:.6f} | {twcri:.6f} | {k_star_col} |")
        
        # Add K* summary at the bottom
        if k_star:
            lines.extend([
                "",
                "---",
                "",
                "### Optimal K* Selections",
                ""
            ])
            for metric_name in ['CRI', 'WCRI', 'TWCRI']:
                if metric_name in k_star:
                    lines.append(f"**{metric_name}:**")
                    for method_name, k_val in k_star[metric_name].items():
                        lines.append(f"  - {method_name}: K* = {k_val}")
                    lines.append("")
        
        return "\n".join(lines)
        
    except Exception as e:
        return f"Error generating metrics table: {str(e)}"


def create_metrics_plot() -> Optional[object]:
    """Create metrics plot."""
    global erica_instance
    
    if erica_instance is None:
        return None
    
    try:
        metrics = erica_instance.get_metrics()
        
        # Organize data by method
        plots = []
        
        for method_name in ['kmeans', 'agglomerative_single', 'agglomerative_ward']:
            k_values = []
            cri_values = []
            wcri_values = []
            twcri_values = []
            
            for k in sorted(metrics.keys()):
                if method_name in metrics[k]:
                    k_values.append(k)
                    m = metrics[k][method_name]
                    cri_values.append(m.get('CRI', 0))
                    wcri_values.append(m.get('WCRI', 0))
                    twcri_values.append(m.get('TWCRI', 0))
            
            if k_values:
                fig = plot_metrics(
                    k_values, cri_values, wcri_values, twcri_values,
                    title=f"Metrics for {method_name}"
                )
                plots.append(fig)
        
        return plots[0] if plots else None
        
    except Exception as e:
        print(f"Error creating metrics plot: {str(e)}")
        return None


def create_clam_heatmap(k: int, method: str) -> Optional[object]:
    """Create CLAM heatmap for specified k and method."""
    global erica_instance
    
    if erica_instance is None:
        return None
    
    try:
        clam_matrix = erica_instance.get_clam_matrix(k=k, method=method)
        if clam_matrix is None:
            return None
        
        fig = plot_clam_heatmap(clam_matrix, k=k, title=f"CLAM Matrix: k={k}, method={method}")
        return fig
        
    except Exception as e:
        print(f"Error creating CLAM heatmap: {str(e)}")
        return None


def create_k_star_plots() -> Tuple[Optional[object], Optional[object]]:
    """Create K* selection plots."""
    global erica_instance
    
    if erica_instance is None or not erica_instance.k_star_:
        return None, None
    
    try:
        metrics = erica_instance.get_metrics()
        
        # Create K* selection plot for TWCRI (most common metric)
        if 'TWCRI' in erica_instance.k_star_:
            k_star_by_method = erica_instance.k_star_['TWCRI']
            
            # Create bar plot of K* by method
            fig_bar = plot_k_star_by_method(k_star_by_method, metric_name='TWCRI')
            
            # Create individual K* selection plot for first method
            if k_star_by_method:
                method_name = list(k_star_by_method.keys())[0]
                k_star = k_star_by_method[method_name]
                
                # Extract metric values for this method
                metric_dict = {}
                for k in sorted(metrics.keys()):
                    if method_name in metrics[k]:
                        metric_dict[k] = metrics[k][method_name].get('TWCRI', 0)
                
                if metric_dict:
                    fig_k_star = plot_k_star_selection(
                        metric_dict, k_star, metric_name='TWCRI', method_name=method_name
                    )
                    return fig_bar, fig_k_star
            
            return fig_bar, None
        
        return None, None
        
    except Exception as e:
        print(f"Error creating K* plots: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None


def create_cluster_sizes_plot(k: int, method: str) -> Optional[object]:
    """Create cluster sizes plot."""
    global erica_instance
    
    if erica_instance is None:
        return None
    
    try:
        metrics = erica_instance.get_metrics(k=k)
        if method not in metrics:
            return None
        
        cluster_sizes = metrics[method].get('cluster_sizes', [])
        if not cluster_sizes:
            return None
        
        fig = plot_cluster_sizes(cluster_sizes, k=k, title=f"Cluster Sizes: k={k}, method={method}")
        return fig
        
    except Exception as e:
        print(f"Error creating cluster sizes plot: {str(e)}")
        return None


def get_available_methods_and_k() -> Tuple[List[str], List[int]]:
    """Get available methods and k values from results."""
    global erica_instance
    
    if erica_instance is None:
        return [], []
    
    try:
        metrics = erica_instance.get_metrics()
        methods = set()
        k_values = []
        
        for k in sorted(metrics.keys()):
            k_values.append(k)
            methods.update(metrics[k].keys())
        
        return sorted(list(methods)), k_values
        
    except Exception:
        return [], []


# Create Gradio interface
with gr.Blocks(title="ERICA Clustering Demo", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # ERICA Clustering Replicability Analysis
        
        Evaluate clustering stability through iterative subsampling analysis.
        
        **Features:**
        - Upload and preview datasets (CSV, NPY)
        - Configure clustering parameters
        - Run Monte Carlo subsampling analysis
        - View replicability metrics (CRI, WCRI, TWCRI)
        - Visualize results with interactive plots
        - Identify optimal K* values
        """
    )
    
    # Define plot components early so they can be updated from run analysis
    # These will be used in Tab 5 but need to be accessible from Tab 3
    with gr.Row(visible=False):
        plot_k = gr.Dropdown(choices=[], value=None)
        plot_method = gr.Dropdown(choices=[], value=None)
        plot_k_display = gr.Dropdown(choices=[], value=None)
        plot_method_display = gr.Dropdown(choices=[], value=None)
        plot_k_display_input = gr.Dropdown(choices=[], value=None)
        plot_method_display_input = gr.Dropdown(choices=[], value=None)
    
    with gr.Tabs():
        # Tab 1: Data Loading
        with gr.Tab("1. Load Data"):
            with gr.Tabs():
                # Sub-tab 1: Upload New Data
                with gr.Tab("Upload New Data"):
                    gr.Markdown("### Upload Your Dataset")
                    
                    with gr.Row():
                        data_file = gr.File(
                            label="Data File",
                            file_types=[".csv", ".npy"],
                            type="filepath"
                        )
                        transpose_data = gr.Checkbox(
                            label="Transpose Data",
                            value=True,
                            info="Transpose if features are in rows (genomics format). Uncheck if samples are in rows (standard ML format)."
                        )
                    
                    data_preview = gr.Textbox(
                        label="Dataset Preview",
                        lines=20,
                        interactive=False
                    )
                    
                    load_btn = gr.Button("Load & Preview Data", variant="primary")
                    load_btn.click(
                        fn=load_and_preview_data,
                        inputs=[data_file, transpose_data],
                        outputs=[data_preview]
                    )
                
                # Sub-tab 2: Load Previous Run
                with gr.Tab("Load Previous Run"):
                    gr.Markdown("### Load Results from Previous Analysis")
                    gr.Markdown("*Browse to a previous ERICA output directory to restore results.*")
                    
                    def list_available_runs():
                        """List available ERICA runs in the default output directory."""
                        import os
                        import glob
                        
                        output_dir = "./erica_output"
                        if not os.path.exists(output_dir):
                            return "No previous runs found. Run an analysis first!"
                        
                        run_dirs = glob.glob(os.path.join(output_dir, "erica_run_*"))
                        run_dirs = [d for d in run_dirs if os.path.isdir(d)]
                        run_dirs.sort(reverse=True)  # Most recent first
                        
                        if not run_dirs:
                            return "No previous runs found in ./erica_output/"
                        
                        lines = [
                            "Available runs (most recent first):",
                            ""
                        ]
                        for run_dir in run_dirs[:10]:  # Show last 10 runs
                            run_name = os.path.basename(run_dir)
                            lines.append(f"  {run_dir}")
                        
                        if len(run_dirs) > 10:
                            lines.append(f"\n... and {len(run_dirs) - 10} more")
                        
                        return "\n".join(lines)
                    
                    available_runs = gr.Textbox(
                        label="Available Runs",
                        lines=8,
                        interactive=False,
                        value=list_available_runs()
                    )
                    
                    refresh_runs_btn = gr.Button("Refresh List", size="sm")
                    refresh_runs_btn.click(
                        fn=list_available_runs,
                        outputs=[available_runs]
                    )
                    
                    prev_run_dir = gr.Textbox(
                        label="Previous Run Directory",
                        placeholder="./erica_output/erica_run_YYYYMMDD_HHMMSS",
                        info="Copy a path from the list above or enter manually"
                    )
                    
                    prev_run_preview = gr.Textbox(
                        label="Previous Run Summary",
                        lines=15,
                        interactive=False
                    )
                    
                    load_prev_btn = gr.Button("Load Previous Run", variant="secondary")
                    
                    def load_previous_run(run_dir):
                        """Load a previous ERICA run from directory."""
                        global erica_instance, erica_results
                        
                        if not run_dir or not run_dir.strip():
                            return "Please enter a run directory path."
                        
                        import os
                        run_dir = run_dir.strip()
                        
                        if not os.path.exists(run_dir):
                            return f"Error: Directory not found: {run_dir}"
                        
                        if not os.path.isdir(run_dir):
                            return f"Error: Path is not a directory: {run_dir}"
                        
                        try:
                            # Try to find and load the ERICA instance pickle file
                            pickle_file = os.path.join(run_dir, "erica_instance.pkl")
                            
                            if os.path.exists(pickle_file):
                                import pickle
                                with open(pickle_file, 'rb') as f:
                                    erica_instance = pickle.load(f)
                                    erica_results = erica_instance
                            else:
                                return (
                                    f"Error: No saved ERICA instance found in {run_dir}\n\n"
                                    "Note: This feature requires ERICA to save its state.\n"
                                    "The directory should contain 'erica_instance.pkl'.\n\n"
                                    "Alternatively, you can re-run the analysis with the same parameters."
                                )
                            
                            # Generate summary
                            summary_lines = [
                                "=" * 80,
                                "PREVIOUS RUN LOADED SUCCESSFULLY",
                                "=" * 80,
                                f"Directory: {run_dir}",
                                f"Data shape: {erica_instance.samples_array.shape}",
                                f"K range: {erica_instance.k_range}",
                                f"Iterations: {erica_instance.n_iterations}",
                                f"Method: {erica_instance.method}",
                                "",
                                "=" * 80,
                                "Results Summary:",
                                "=" * 80,
                            ]
                            
                            # Add metrics summary
                            metrics = erica_instance.get_metrics()
                            for k in sorted(metrics.keys()):
                                summary_lines.append(f"\nK = {k}:")
                                for method_name, metric_dict in metrics[k].items():
                                    summary_lines.append(f"  {method_name}:")
                                    summary_lines.append(f"    CRI:   {metric_dict.get('CRI', 0):.6f}")
                                    summary_lines.append(f"    WCRI:  {metric_dict.get('WCRI', 0):.6f}")
                                    summary_lines.append(f"    TWCRI: {metric_dict.get('TWCRI', 0):.6f}")
                            
                            # Add K* summary
                            if hasattr(erica_instance, 'k_star_') and erica_instance.k_star_:
                                summary_lines.append("\n" + "=" * 80)
                                summary_lines.append("Optimal K* Selection:")
                                summary_lines.append("=" * 80)
                                for metric_name in ['CRI', 'WCRI', 'TWCRI']:
                                    if metric_name in erica_instance.k_star_:
                                        summary_lines.append(f"\n{metric_name}:")
                                        for method_name, k_star in erica_instance.k_star_[metric_name].items():
                                            summary_lines.append(f"  {method_name}: K* = {k_star}")
                            
                            summary_lines.extend([
                                "",
                                "=" * 80,
                                "Previous run loaded. You can now view metrics and visualizations in Tab 3.",
                                "=" * 80,
                            ])
                            
                            return "\n".join(summary_lines)
                            
                        except Exception as e:
                            return f"Error loading previous run:\n{str(e)}\n\n{traceback.format_exc()}"
                    
                    load_prev_btn.click(
                        fn=load_previous_run,
                        inputs=[prev_run_dir],
                        outputs=[prev_run_preview]
                    )
        
        # Tab 2: Configuration & Run Analysis
        with gr.Tab("2. Configure & Run Analysis"):
            gr.Markdown("### Analysis Parameters")
            
            with gr.Row():
                k_range = gr.Textbox(
                    label="K Range",
                    value="2,3,4,5",
                    info="Comma-separated (e.g., '2,3,4,5') or range (e.g., '2-5' or '2-10:2')"
                )
                n_iterations = gr.Slider(
                    label="Number of Iterations",
                    minimum=10,
                    maximum=1000,
                    value=100,
                    step=10,
                    info="More iterations = more stable but slower"
                )
            
            with gr.Row():
                train_percent = gr.Slider(
                    label="Training Percentage",
                    minimum=0.5,
                    maximum=0.95,
                    value=0.8,
                    step=0.05,
                    info="Percentage of data used for training"
                )
                random_seed = gr.Number(
                    label="Random Seed",
                    value=123,
                    precision=0,
                    info="For reproducibility"
                )
            
            with gr.Row():
                method = gr.Radio(
                    label="Clustering Method",
                    choices=["kmeans", "agglomerative", "both"],
                    value="kmeans",
                    info="Clustering algorithm to use"
                )
                linkages = gr.Textbox(
                    label="Linkage Methods (for Agglomerative)",
                    value="single,ward",
                    info="Comma-separated: single, complete, average, ward"
                )
            
            with gr.Row():
                output_dir = gr.Textbox(
                    label="Output Directory",
                    value="./erica_output",
                    info="Directory to save results"
                )
                verbose = gr.Checkbox(
                    label="Verbose Output",
                    value=True,
                    info="Print detailed progress messages"
                )
            
            gr.Markdown("---")
            gr.Markdown("### Run Analysis")
            gr.Markdown("*Note: Make sure you've loaded data in Tab 1 before running analysis.*")
            
            # Status indicator
            status_box = gr.Textbox(
                label="Status",
                value="Ready to run analysis",
                lines=1,
                interactive=False,
                elem_id="status_box"
            )
            
            run_btn = gr.Button("Run Analysis", variant="primary", size="lg")
            
            analysis_output = gr.Textbox(
                label="Analysis Results",
                lines=30,
                interactive=False,
                placeholder="Results will appear here after running analysis..."
            )
            
            def run_and_refresh(
                file, transpose, k_str, n_iter, train_pct, meth, linkage_list,
                seed, out_dir, verb
            ):
                """Run analysis and refresh plot options."""
                # Check if data file is loaded
                if file is None:
                    status = "No data loaded - Please upload data in Tab 1"
                    error_msg = (
                        "=" * 80 + "\n"
                        "NO DATA FILE LOADED\n" +
                        "=" * 80 + "\n\n"
                        "Please follow these steps:\n\n"
                        "1. Go to Tab 1: Load Data\n"
                        "2. Click 'Data File' and select your CSV or NPY file\n"
                        "3. Set the 'Transpose Data' checkbox as needed\n"
                        "4. Click 'Load & Preview Data' to verify your data\n"
                        "5. Return to this tab and click 'Run Analysis'\n\n" +
                        "=" * 80
                    )
                    return (
                        status,  # status_box
                        error_msg,  # analysis_output
                        gr.update(),  # plot_method
                        gr.update(),  # plot_k
                        gr.update(),  # plot_method_display
                        gr.update(),  # plot_k_display
                        gr.update(),  # plot_method_display_input
                        gr.update(),  # plot_k_display_input
                    )
                
                # Parse k_range to show in status
                k_range = parse_k_range(k_str)
                
                # Run analysis
                summary, _ = run_erica_analysis(
                    file, transpose, k_str, n_iter, train_pct, meth, linkage_list,
                    seed, out_dir, verb
                )
                
                # Determine status based on result
                if "ANALYSIS COMPLETE" in summary:
                    status = "Analysis completed successfully"
                elif "ERROR" in summary:
                    status = "Analysis failed - check results below"
                else:
                    status = "Analysis finished with warnings"
                
                methods, k_values = get_available_methods_and_k()
                return (
                    status,  # status_box
                    summary,  # analysis_output
                    gr.update(choices=methods, value=methods[0] if methods else None),  # plot_method
                    gr.update(choices=k_values, value=k_values[0] if k_values else None),  # plot_k
                    gr.update(choices=methods, value=methods[0] if methods else None),  # plot_method_display
                    gr.update(choices=k_values, value=k_values[0] if k_values else None),  # plot_k_display
                    gr.update(choices=methods, value=methods[0] if methods else None),  # plot_method_display_input
                    gr.update(choices=k_values, value=k_values[0] if k_values else None),  # plot_k_display_input
                )
            
            run_btn.click(
                fn=run_and_refresh,
                inputs=[
                    data_file,
                    transpose_data,
                    k_range,
                    n_iterations,
                    train_percent,
                    method,
                    linkages,
                    random_seed,
                    output_dir,
                    verbose,
                ],
                outputs=[status_box, analysis_output, plot_method, plot_k, plot_method_display, plot_k_display, plot_method_display_input, plot_k_display_input]
            )
        
        # Tab 3: Results & Visualizations
        with gr.Tab("3. Results & Visualizations"):
            gr.Markdown("### Metrics Summary")
            gr.Markdown("*Bold rows indicate optimal K* selections for each metric.*")
            
            metrics_table = gr.Markdown(
                value="Run analysis first to see metrics."
            )
            
            metrics_line_plot = gr.Plot(label="Metrics Across K Values")
            
            refresh_metrics_btn = gr.Button("Refresh Results")
            
            def refresh_metrics_and_plots():
                """Refresh metrics table and generate plot."""
                table = get_metrics_table()
                plot = create_metrics_plot()
                return table, plot
            
            refresh_metrics_btn.click(
                fn=refresh_metrics_and_plots,
                outputs=[metrics_table, metrics_line_plot]
            )
            
            gr.Markdown("---")
            gr.Markdown("### Detailed Analysis")
            gr.Markdown("Select K and method to view CLAM heatmap and cluster size distributions.")
            
            with gr.Row():
                plot_k_selector = gr.Dropdown(
                    label="Select K",
                    choices=[],
                    value=None
                )
                plot_method_selector = gr.Dropdown(
                    label="Select Method",
                    choices=[],
                    value=None
                )
            
            # Sync from hidden dropdowns when analysis completes
            plot_k_display_input.change(
                lambda x: gr.update(choices=get_available_methods_and_k()[1], value=x),
                inputs=[plot_k_display_input],
                outputs=[plot_k_selector]
            )
            plot_method_display_input.change(
                lambda x: gr.update(choices=get_available_methods_and_k()[0], value=x),
                inputs=[plot_method_display_input],
                outputs=[plot_method_selector]
            )
            
            with gr.Row():
                with gr.Column():
                    clam_heatmap = gr.Plot(label="CLAM Matrix Heatmap")
                with gr.Column():
                    cluster_sizes_plot = gr.Plot(label="Cluster Size Distribution")
            
            def update_detailed_plots(k_val, method_val):
                """Update both CLAM and cluster size plots."""
                if k_val is None or method_val is None:
                    return None, None
                clam = create_clam_heatmap(int(k_val), str(method_val))
                sizes = create_cluster_sizes_plot(int(k_val), str(method_val))
                return clam, sizes
            
            plot_k_selector.change(
                fn=update_detailed_plots,
                inputs=[plot_k_selector, plot_method_selector],
                outputs=[clam_heatmap, cluster_sizes_plot]
            )
            plot_method_selector.change(
                fn=update_detailed_plots,
                inputs=[plot_k_selector, plot_method_selector],
                outputs=[clam_heatmap, cluster_sizes_plot]
            )
    
    # Footer
    gr.Markdown(
        """
        ---
        **ERICA** - Evaluating Replicability via Iterative Clustering Assignments
        
        For more information, visit the [documentation](docs/GETTING_STARTED.md).
        """
    )


if __name__ == "__main__":
    import socket
    
    def find_free_port(start_port=7860, max_attempts=10):
        """Find a free port starting from start_port."""
        for i in range(max_attempts):
            port = start_port + i
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind(('', port))
                    return port
                except OSError:
                    continue
        return start_port  # Fallback to default
    
    # Find free port
    port = find_free_port()
    
    # Print URL before launching
    print("\n" + "="*60)
    print(f"Starting ERICA Gradio Demo Server...")
    print(f"Open your browser to: http://localhost:{port}")
    print("="*60 + "\n")
    
    # Launch the interface
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,
        show_error=True
    )

