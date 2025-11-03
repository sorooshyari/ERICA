"""Plotting functions for ERICA visualization.

This module provides functions for creating interactive plots of:
- Metrics across different k values
- Optimal k selection
- CLAM matrix heatmaps
- Cluster stability visualizations

Requires plotly for interactive plots.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np

try:
    import plotly.graph_objects as go
    import plotly.express as px
    _PLOTLY_AVAILABLE = True
except ImportError:
    _PLOTLY_AVAILABLE = False


def _check_plotly():
    """Check if plotly is available."""
    if not _PLOTLY_AVAILABLE:
        raise ImportError(
            "Plotting requires plotly. Install with: "
            "pip install erica-clustering[plots]"
        )


def plot_metrics(
    k_values: List[int],
    cri_values: List[float],
    wcri_values: List[float],
    twcri_values: List[float],
    title: str = "ERICA Clustering Replicability Metrics"
) -> 'go.Figure':
    """Create line plot of metrics vs k.
    
    Parameters
    ----------
    k_values : list of int
        List of k values
    cri_values : list of float
        CRI values for each k
    wcri_values : list of float
        WCRI values for each k
    twcri_values : list of float
        TWCRI values for each k
    title : str, optional
        Plot title
        
    Returns
    -------
    plotly.graph_objects.Figure
        Interactive plotly figure
        
    Examples
    --------
    >>> fig = plot_metrics([2, 3, 4], [0.85, 0.90, 0.87],
    ...                    [0.82, 0.88, 0.84], [0.88, 0.92, 0.89])
    >>> fig.show()  # doctest: +SKIP
    """
    _check_plotly()
    
    fig = go.Figure()
    
    # Add traces for each metric
    fig.add_trace(go.Scatter(
        x=k_values, y=cri_values,
        mode='lines+markers',
        name='CRI',
        line=dict(color='blue', width=2),
        marker=dict(size=8)
    ))
    
    fig.add_trace(go.Scatter(
        x=k_values, y=wcri_values,
        mode='lines+markers',
        name='WCRI',
        line=dict(color='red', width=2),
        marker=dict(size=8)
    ))
    
    fig.add_trace(go.Scatter(
        x=k_values, y=twcri_values,
        mode='lines+markers',
        name='TWCRI',
        line=dict(color='green', width=2),
        marker=dict(size=8)
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Number of Clusters (k)",
        yaxis_title="Metric Value",
        hovermode='x unified',
        showlegend=True,
        height=500,
        template='plotly_white'
    )
    
    return fig


def plot_optimal_k(
    optimal_k_by_metric: Dict[str, int],
    title: str = "Optimal K Values by Metric"
) -> 'go.Figure':
    """Create bar plot showing optimal k for each metric.
    
    Parameters
    ----------
    optimal_k_by_metric : dict
        Dictionary mapping metric names to optimal k values
    title : str, optional
        Plot title
        
    Returns
    -------
    plotly.graph_objects.Figure
        Interactive plotly figure
        
    Examples
    --------
    >>> optimal_k = {'CRI': 3, 'WCRI': 3, 'TWCRI': 3}
    >>> fig = plot_optimal_k(optimal_k)
    >>> fig.show()  # doctest: +SKIP
    """
    _check_plotly()
    
    metrics = list(optimal_k_by_metric.keys())
    k_values = list(optimal_k_by_metric.values())
    
    colors = {'CRI': 'blue', 'WCRI': 'red', 'TWCRI': 'green'}
    bar_colors = [colors.get(m, 'gray') for m in metrics]
    
    fig = go.Figure(data=[
        go.Bar(
            x=metrics,
            y=k_values,
            marker_color=bar_colors,
            text=k_values,
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title=title,
        xaxis_title="Metric",
        yaxis_title="Optimal K Value",
        showlegend=False,
        height=400,
        template='plotly_white'
    )
    
    return fig


def plot_clam_heatmap(
    clam_matrix: np.ndarray,
    k: int,
    title: Optional[str] = None
) -> 'go.Figure':
    """Create heatmap visualization of CLAM matrix.
    
    Parameters
    ----------
    clam_matrix : np.ndarray
        CLAM matrix (n_samples, k)
    k : int
        Number of clusters
    title : str, optional
        Plot title
        
    Returns
    -------
    plotly.graph_objects.Figure
        Interactive plotly figure
        
    Examples
    --------
    >>> clam = np.array([[50, 10], [45, 15], [5, 55]])
    >>> fig = plot_clam_heatmap(clam, k=2)
    >>> fig.show()  # doctest: +SKIP
    """
    _check_plotly()
    
    if title is None:
        title = f"CLAM Matrix Heatmap (k={k})"
    
    fig = go.Figure(data=go.Heatmap(
        z=clam_matrix,
        x=[f"Cluster {i+1}" for i in range(k)],
        y=[f"Sample {i+1}" for i in range(len(clam_matrix))],
        colorscale='Viridis',
        showscale=True
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Cluster",
        yaxis_title="Sample",
        height=600,
        template='plotly_white'
    )
    
    return fig


def plot_cluster_sizes(
    cluster_sizes: List[int],
    k: int,
    title: Optional[str] = None
) -> 'go.Figure':
    """Plot cluster size distribution.
    
    Parameters
    ----------
    cluster_sizes : list of int
        Size of each cluster
    k : int
        Number of clusters
    title : str, optional
        Plot title
        
    Returns
    -------
    plotly.graph_objects.Figure
        Interactive plotly figure
    """
    _check_plotly()
    
    if title is None:
        title = f"Cluster Size Distribution (k={k})"
    
    fig = go.Figure(data=[
        go.Bar(
            x=[f"Cluster {i+1}" for i in range(k)],
            y=cluster_sizes,
            text=cluster_sizes,
            textposition='auto',
            marker_color='lightblue'
        )
    ])
    
    fig.update_layout(
        title=title,
        xaxis_title="Cluster",
        yaxis_title="Number of Samples",
        showlegend=False,
        height=400,
        template='plotly_white'
    )
    
    return fig


def create_metrics_plots(
    metrics_data: Dict,
    show_optimal: bool = True
) -> Tuple['go.Figure', Optional['go.Figure'], str]:
    """Create comprehensive metrics plots.
    
    This is a convenience function that creates multiple plots from
    metrics data.
    
    Parameters
    ----------
    metrics_data : dict
        Dictionary containing metrics data with keys:
        - 'k_values': list of k values
        - 'CRI_vector': list of CRI values
        - 'WCRI_vector': list of WCRI values
        - 'TWCRI_vector': list of TWCRI values
    show_optimal : bool, optional
        Whether to create optimal k plot, default True
        
    Returns
    -------
    tuple of (Figure, Figure or None, str)
        (metrics_plot, optimal_k_plot, summary_text)
        
    Examples
    --------
    >>> metrics_data = {
    ...     'k_values': [2, 3, 4],
    ...     'CRI_vector': [0.85, 0.90, 0.87],
    ...     'WCRI_vector': [0.82, 0.88, 0.84],
    ...     'TWCRI_vector': [0.88, 0.92, 0.89]
    ... }
    >>> fig1, fig2, summary = create_metrics_plots(metrics_data)
    >>> fig1.show()  # doctest: +SKIP
    """
    _check_plotly()
    
    k_values = metrics_data['k_values']
    cri_values = metrics_data['CRI_vector']
    wcri_values = metrics_data['WCRI_vector']
    twcri_values = metrics_data['TWCRI_vector']
    
    if not k_values:
        return None, None, "No data available for plotting"
    
    # Create main metrics plot
    fig1 = plot_metrics(k_values, cri_values, wcri_values, twcri_values)
    
    # Find optimal k for each metric
    optimal_k = {}
    for name, values in [('CRI', cri_values), ('WCRI', wcri_values), ('TWCRI', twcri_values)]:
        if values:
            max_idx = np.argmax(values)
            optimal_k[name] = k_values[max_idx]
    
    # Create optimal k plot
    fig2 = None
    if show_optimal and optimal_k:
        fig2 = plot_optimal_k(optimal_k)
    
    # Create summary text
    summary_lines = ["📊 **Metrics Summary:**\n"]
    
    for name, k_opt in optimal_k.items():
        idx = k_values.index(k_opt)
        value = None
        if name == 'CRI':
            value = cri_values[idx]
        elif name == 'WCRI':
            value = wcri_values[idx]
        elif name == 'TWCRI':
            value = twcri_values[idx]
        
        if value is not None:
            summary_lines.append(f"- **{name}**: optimal k = {k_opt} (value = {value:.4f})")
    
    summary_lines.append("\n📈 **Detailed Results:**\n")
    summary_lines.append("| k | CRI | WCRI | TWCRI |")
    summary_lines.append("|---|-----|------|-------|")
    
    for i, k in enumerate(k_values):
        cri = cri_values[i] if i < len(cri_values) else 0
        wcri = wcri_values[i] if i < len(wcri_values) else 0
        twcri = twcri_values[i] if i < len(twcri_values) else 0
        summary_lines.append(f"| {k} | {cri:.4f} | {wcri:.4f} | {twcri:.4f} |")
    
    summary_text = "\n".join(summary_lines)
    
    return fig1, fig2, summary_text


def plot_stability_comparison(
    results_dict: Dict[Tuple[int, str], Dict],
    metric_name: str = 'CRI'
) -> 'go.Figure':
    """Compare stability across different methods and k values.
    
    Parameters
    ----------
    results_dict : dict
        Dictionary with (k, method) tuples as keys and results as values
    metric_name : str, optional
        Metric to plot ('CRI', 'WCRI', or 'TWCRI'), default 'CRI'
        
    Returns
    -------
    plotly.graph_objects.Figure
        Interactive plotly figure
    """
    _check_plotly()
    
    # Organize data by method
    methods_data = {}
    for (k, method), result in results_dict.items():
        if method not in methods_data:
            methods_data[method] = {'k_values': [], 'metrics': []}
        
        if 'metrics' in result and metric_name in result['metrics']:
            methods_data[method]['k_values'].append(k)
            methods_data[method]['metrics'].append(result['metrics'][metric_name])
    
    # Create plot
    fig = go.Figure()
    
    for method, data in methods_data.items():
        fig.add_trace(go.Scatter(
            x=data['k_values'],
            y=data['metrics'],
            mode='lines+markers',
            name=method,
            line=dict(width=2),
            marker=dict(size=8)
        ))
    
    fig.update_layout(
        title=f"{metric_name} Comparison Across Methods",
        xaxis_title="Number of Clusters (k)",
        yaxis_title=f"{metric_name} Value",
        hovermode='x unified',
        showlegend=True,
        height=500,
        template='plotly_white'
    )
    
    return fig


def plot_k_star_selection(
    metric_dict: Dict[int, float],
    k_star: int,
    metric_name: str = 'TWCRI',
    method_name: Optional[str] = None,
    title: Optional[str] = None
) -> 'go.Figure':
    """Visualize K_star selection with metric values across K.
    
    This function creates a plot showing how the replicability metric changes
    with K, and highlights the selected K_star value according to Algorithm 2.
    
    Parameters
    ----------
    metric_dict : dict
        Dictionary mapping K values to metric scores
    k_star : int
        Selected optimal K value
    metric_name : str, optional
        Name of the metric being plotted, default 'TWCRI'
    method_name : str, optional
        Name of clustering method, default None
    title : str, optional
        Custom plot title, default None
        
    Returns
    -------
    plotly.graph_objects.Figure
        Interactive plotly figure
        
    Examples
    --------
    >>> M = {2: 0.71, 3: 0.75, 4: 0.74, 5: 0.78}
    >>> k_star = 5
    >>> fig = plot_k_star_selection(M, k_star, 'TWCRI', 'kmeans')
    >>> fig.show()  # doctest: +SKIP
    """
    _check_plotly()
    
    # Sort K values
    k_values = sorted(metric_dict.keys())
    metric_values = [metric_dict[k] for k in k_values]
    
    # Create title
    if title is None:
        if method_name:
            title = f"K* Selection: {metric_name} for {method_name} (K* = {k_star})"
        else:
            title = f"K* Selection: {metric_name} (K* = {k_star})"
    
    # Create figure
    fig = go.Figure()
    
    # Add main line plot
    fig.add_trace(go.Scatter(
        x=k_values,
        y=metric_values,
        mode='lines+markers',
        name=metric_name,
        line=dict(color='blue', width=2),
        marker=dict(size=10, color='blue'),
        hovertemplate='K=%{x}<br>' + metric_name + '=%{y:.4f}<extra></extra>'
    ))
    
    # Highlight K_star with a larger marker
    if k_star in metric_dict:
        fig.add_trace(go.Scatter(
            x=[k_star],
            y=[metric_dict[k_star]],
            mode='markers',
            name=f'K* = {k_star}',
            marker=dict(
                size=20,
                color='red',
                symbol='star',
                line=dict(color='darkred', width=2)
            ),
            hovertemplate=f'K* = {k_star}<br>' + metric_name + f'={metric_dict[k_star]:.4f}<extra></extra>'
        ))
    
    # Add vertical line at K_star
    fig.add_vline(
        x=k_star,
        line_dash="dash",
        line_color="red",
        opacity=0.5,
        annotation_text=f"K* = {k_star}",
        annotation_position="top"
    )
    
    fig.update_layout(
        title=title,
        xaxis_title="Number of Clusters (K)",
        yaxis_title=f"{metric_name} Value",
        hovermode='closest',
        showlegend=True,
        height=500,
        template='plotly_white',
        xaxis=dict(
            tickmode='linear',
            tick0=min(k_values),
            dtick=1
        )
    )
    
    return fig


def plot_k_star_by_method(
    k_star_by_method: Dict[str, int],
    metric_name: str = 'TWCRI',
    title: Optional[str] = None
) -> 'go.Figure':
    """Create bar plot showing K_star for each clustering method.
    
    Parameters
    ----------
    k_star_by_method : dict
        Dictionary mapping method names to their K_star values
    metric_name : str, optional
        Name of metric used for selection, default 'TWCRI'
    title : str, optional
        Custom plot title, default None
        
    Returns
    -------
    plotly.graph_objects.Figure
        Interactive plotly figure
        
    Examples
    --------
    >>> k_star = {'kmeans': 3, 'agglomerative_ward': 4}
    >>> fig = plot_k_star_by_method(k_star, 'TWCRI')
    >>> fig.show()  # doctest: +SKIP
    """
    _check_plotly()
    
    if title is None:
        title = f"Optimal K* Selection by Method (using {metric_name})"
    
    methods = list(k_star_by_method.keys())
    k_values = list(k_star_by_method.values())
    
    # Create color scheme
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    bar_colors = [colors[i % len(colors)] for i in range(len(methods))]
    
    fig = go.Figure(data=[
        go.Bar(
            x=methods,
            y=k_values,
            marker_color=bar_colors,
            text=k_values,
            textposition='auto',
            hovertemplate='%{x}<br>K* = %{y}<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title=title,
        xaxis_title="Clustering Method",
        yaxis_title="Optimal K* Value",
        showlegend=False,
        height=400,
        template='plotly_white',
        yaxis=dict(
            tickmode='linear',
            tick0=0,
            dtick=1
        )
    )
    
    return fig


