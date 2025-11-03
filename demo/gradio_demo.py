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
        
        # Get dataset info
        info = get_dataset_info(data, transpose=transpose)
        
        # Create preview
        preview_lines = [
            "=" * 60,
            "Dataset Information",
            "=" * 60,
            f"Original shape: {data.shape if hasattr(data, 'shape') else 'N/A'}",
            f"After processing: {info['shape']}",
            f"  - Samples: {info['n_samples']}",
            f"  - Features: {info['n_features']}",
            f"  - Transpose: {transpose}",
            "",
            "Statistics:",
            f"  - Min value: {info['min_value']:.4f}",
            f"  - Max value: {info['max_value']:.4f}",
            f"  - Mean value: {info['mean_value']:.4f}",
            f"  - Std value: {info['std_value']:.4f}",
            "",
            f"Data type: {info['dtype']}",
            f"Has NaN: {info['has_nan']}",
            f"Has Inf: {info['has_inf']}",
            "=" * 60,
        ]
        
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
        erica_results = erica_instance.run()
        
        # Generate summary
        summary_lines = [
            "=" * 60,
            "ERICA Analysis Complete!",
            "=" * 60,
            f"Data shape: {erica_instance.samples_array.shape}",
            f"K range: {k_range}",
            f"Iterations: {n_iterations}",
            f"Method: {method}",
            f"Output directory: {erica_instance.output_folders_[0] if erica_instance.output_folders_ else 'N/A'}",
            "",
            "Results Summary:",
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
            summary_lines.append("\n" + "=" * 60)
            summary_lines.append("Optimal K* Selection:")
            summary_lines.append("=" * 60)
            for metric_name in ['CRI', 'WCRI', 'TWCRI']:
                if metric_name in erica_instance.k_star_:
                    summary_lines.append(f"\n{metric_name}:")
                    for method_name, k_star in erica_instance.k_star_[metric_name].items():
                        summary_lines.append(f"  {method_name}: K* = {k_star}")
        
        summary_lines.append("\n" + "=" * 60)
        
        return "\n".join(summary_lines), erica_instance
        
    except Exception as e:
        error_msg = f"Error running ERICA analysis:\n{str(e)}\n\n{traceback.format_exc()}"
        return error_msg, None


def get_metrics_table() -> Optional[str]:
    """Generate metrics table from results."""
    global erica_instance
    
    if erica_instance is None:
        return "No analysis results available. Please run analysis first."
    
    try:
        metrics = erica_instance.get_metrics()
        
        lines = ["| K | Method | CRI | WCRI | TWCRI |"]
        lines.append("|" + "|".join(["---"] * 5) + "|")
        
        for k in sorted(metrics.keys()):
            for method_name, metric_dict in metrics[k].items():
                cri = metric_dict.get('CRI', 0)
                wcri = metric_dict.get('WCRI', 0)
                twcri = metric_dict.get('TWCRI', 0)
                lines.append(f"| {k} | {method_name} | {cri:.6f} | {wcri:.6f} | {twcri:.6f} |")
        
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
        # 🧬 ERICA Clustering Replicability Analysis Demo
        
        Welcome to the ERICA demo interface! This tool allows you to test and explore
        the ERICA library for clustering replicability analysis.
        
        **Features:**
        - 📊 Upload and preview datasets (CSV, NPY)
        - ⚙️ Configure clustering parameters
        - 🔄 Run Monte Carlo subsampling analysis
        - 📈 View metrics (CRI, WCRI, TWCRI)
        - 📉 Visualize results with interactive plots
        - ⭐ Find optimal K* values
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
        
        # Tab 2: Configuration
        with gr.Tab("2. Configure Parameters"):
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
        
        # Tab 3: Run Analysis
        with gr.Tab("3. Run Analysis"):
            gr.Markdown("### Run ERICA Analysis")
            
            run_btn = gr.Button("Run Analysis", variant="primary", size="lg")
            
            analysis_output = gr.Textbox(
                label="Analysis Results",
                lines=30,
                interactive=False
            )
            
            def run_and_refresh(
                file, transpose, k_str, n_iter, train_pct, meth, linkage_list,
                seed, out_dir, verb
            ):
                """Run analysis and refresh plot options."""
                summary, _ = run_erica_analysis(
                    file, transpose, k_str, n_iter, train_pct, meth, linkage_list,
                    seed, out_dir, verb
                )
                methods, k_values = get_available_methods_and_k()
                return (
                    summary,
                    gr.Dropdown.update(choices=methods, value=methods[0] if methods else None),  # plot_method
                    gr.Dropdown.update(choices=k_values, value=k_values[0] if k_values else None),  # plot_k
                    gr.Dropdown.update(choices=methods, value=methods[0] if methods else None),  # plot_method_display
                    gr.Dropdown.update(choices=k_values, value=k_values[0] if k_values else None),  # plot_k_display
                    gr.Dropdown.update(choices=methods, value=methods[0] if methods else None),  # plot_method_display_input
                    gr.Dropdown.update(choices=k_values, value=k_values[0] if k_values else None),  # plot_k_display_input
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
                outputs=[analysis_output, plot_method, plot_k, plot_method_display, plot_k_display, plot_method_display_input, plot_k_display_input]
            )
        
        # Tab 4: Metrics
        with gr.Tab("4. View Metrics"):
            gr.Markdown("### Clustering Replicability Metrics")
            
            metrics_table = gr.Markdown(
                label="Metrics Table",
                value="Run analysis first to see metrics."
            )
            
            refresh_metrics_btn = gr.Button("Refresh Metrics Table")
            refresh_metrics_btn.click(
                fn=get_metrics_table,
                outputs=[metrics_table]
            )
        
        # Tab 5: Visualizations
        with gr.Tab("5. Visualizations"):
            gr.Markdown("### Interactive Plots")
            
            with gr.Row():
                plot_k_display_input_show = gr.Dropdown(
                    label="Select K",
                    choices=[],
                    value=None,
                    info="K value for CLAM heatmap and cluster sizes"
                )
                plot_method_display_input_show = gr.Dropdown(
                    label="Select Method",
                    choices=[],
                    value=None,
                    info="Clustering method"
                )
            
            # Sync hidden to visible
            plot_k_display_input.change(lambda x: gr.Dropdown.update(value=x), inputs=[plot_k_display_input], outputs=[plot_k_display_input_show])
            plot_method_display_input.change(lambda x: gr.Dropdown.update(value=x), inputs=[plot_method_display_input], outputs=[plot_method_display_input_show])
            
            # Sync visible to hidden
            plot_k_display_input_show.change(lambda x: x, inputs=[plot_k_display_input_show], outputs=[plot_k_display_input])
            plot_method_display_input_show.change(lambda x: x, inputs=[plot_method_display_input_show], outputs=[plot_method_display_input])
            
            # Sync hidden components with display components
            def sync_k_to_display(k_val):
                if k_val is not None:
                    return gr.Dropdown.update(value=k_val)
                return gr.Dropdown.update()
            
            def sync_method_to_display(method_val):
                if method_val is not None:
                    return gr.Dropdown.update(value=method_val)
                return gr.Dropdown.update()
            
            plot_k.change(sync_k_to_display, inputs=[plot_k], outputs=[plot_k_display])
            plot_method.change(sync_method_to_display, inputs=[plot_method], outputs=[plot_method_display])
            
            # Also sync back from display to hidden for interactive updates
            def sync_display_to_k(k_val):
                return k_val
            
            def sync_display_to_method(method_val):
                return method_val
            
            plot_k_display_input.change(sync_display_to_k, inputs=[plot_k_display_input], outputs=[plot_k])
            plot_method_display_input.change(sync_display_to_method, inputs=[plot_method_display_input], outputs=[plot_method])
            
            # Sync display components
            plot_k_display.change(lambda x: x, inputs=[plot_k_display], outputs=[plot_k_display_input])
            plot_method_display.change(lambda x: x, inputs=[plot_method_display], outputs=[plot_method_display_input])
            
            # Sync hidden components with display components
            def sync_k_to_display(k_val):
                if k_val is not None:
                    return gr.Dropdown.update(value=k_val)
                return gr.Dropdown.update()
            
            def sync_method_to_display(method_val):
                if method_val is not None:
                    return gr.Dropdown.update(value=method_val)
                return gr.Dropdown.update()
            
            plot_k.change(sync_k_to_display, inputs=[plot_k], outputs=[plot_k_display])
            plot_method.change(sync_method_to_display, inputs=[plot_method], outputs=[plot_method_display])
            
            # Sync display components to input components
            plot_k_display.change(lambda x: gr.Dropdown.update(value=x), inputs=[plot_k_display], outputs=[plot_k_display_input])
            plot_method_display.change(lambda x: gr.Dropdown.update(value=x), inputs=[plot_method_display], outputs=[plot_method_display_input])
            
            # Also sync back from input to hidden for interactive updates
            plot_k_display_input.change(lambda x: x, inputs=[plot_k_display_input], outputs=[plot_k])
            plot_method_display_input.change(lambda x: x, inputs=[plot_method_display_input], outputs=[plot_method])
            
            with gr.Row():
                refresh_plot_options_btn = gr.Button("Refresh Options")
                
                def refresh_plot_options():
                    methods, k_values = get_available_methods_and_k()
                    return (
                        gr.Dropdown.update(choices=methods, value=methods[0] if methods else None),  # plot_method
                        gr.Dropdown.update(choices=k_values, value=k_values[0] if k_values else None),  # plot_k
                        gr.Dropdown.update(choices=methods, value=methods[0] if methods else None),  # plot_method_display
                        gr.Dropdown.update(choices=k_values, value=k_values[0] if k_values else None),  # plot_k_display
                        gr.Dropdown.update(choices=methods, value=methods[0] if methods else None),  # plot_method_display_input
                        gr.Dropdown.update(choices=k_values, value=k_values[0] if k_values else None),  # plot_k_display_input
                    )
                
                refresh_plot_options_btn.click(
                    fn=refresh_plot_options,
                    outputs=[plot_method, plot_k, plot_method_display, plot_k_display, plot_method_display_input, plot_k_display_input]
                )
            
            gr.Markdown("### Metrics Plot")
            metrics_plot = gr.Plot(label="Metrics Across K Values")
            
            refresh_metrics_plot_btn = gr.Button("Generate Metrics Plot")
            refresh_metrics_plot_btn.click(
                fn=create_metrics_plot,
                outputs=[metrics_plot]
            )
            
            gr.Markdown("### CLAM Heatmap")
            clam_heatmap = gr.Plot(label="CLAM Matrix Heatmap")
            
            def update_clam_plot(k_val, method_val):
                if k_val is None or method_val is None:
                    return None
                return create_clam_heatmap(int(k_val), str(method_val))
            
            plot_k_display_input_show.change(
                fn=update_clam_plot,
                inputs=[plot_k_display_input_show, plot_method_display_input_show],
                outputs=[clam_heatmap]
            )
            plot_method_display_input_show.change(
                fn=update_clam_plot,
                inputs=[plot_k_display_input_show, plot_method_display_input_show],
                outputs=[clam_heatmap]
            )
            
            gr.Markdown("### Cluster Sizes")
            cluster_sizes_plot = gr.Plot(label="Cluster Size Distribution")
            
            def update_cluster_sizes(k_val, method_val):
                if k_val is None or method_val is None:
                    return None
                return create_cluster_sizes_plot(int(k_val), str(method_val))
            
            plot_k_display_input_show.change(
                fn=update_cluster_sizes,
                inputs=[plot_k_display_input_show, plot_method_display_input_show],
                outputs=[cluster_sizes_plot]
            )
            plot_method_display_input_show.change(
                fn=update_cluster_sizes,
                inputs=[plot_k_display_input_show, plot_method_display_input_show],
                outputs=[cluster_sizes_plot]
            )
            
            gr.Markdown("### Optimal K* Selection")
            k_star_bar_plot = gr.Plot(label="K* by Method")
            k_star_line_plot = gr.Plot(label="K* Selection Detail")
            
            refresh_k_star_btn = gr.Button("Generate K* Plots")
            refresh_k_star_btn.click(
                fn=create_k_star_plots,
                outputs=[k_star_bar_plot, k_star_line_plot]
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

