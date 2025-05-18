"""
Plotting and visualization utilities for sample size and power calculations.

This module provides functions for creating visualizations to aid in
interpreting power analysis and sample size calculations.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from typing import Callable, Optional, List, Tuple, Dict, Any, Union


def power_curve(sample_sizes: np.ndarray, 
                power_function: Callable[[int], float],
                target_power: float = 0.8,
                title: str = "Power Curve",
                xlabel: str = "Sample Size (per group)",
                ylabel: str = "Power",
                figsize: Tuple[float, float] = (10, 6),
                show_target: bool = True) -> plt.Figure:
    """
    Generate a power curve showing power as a function of sample size.
    
    Parameters
    ----------
    sample_sizes : np.ndarray
        Array of sample sizes to plot
    power_function : callable
        Function that calculates power given a sample size
    target_power : float, optional
        Target power level to highlight, by default 0.8
    title : str, optional
        Plot title, by default "Power Curve"
    xlabel : str, optional
        X-axis label, by default "Sample Size (per group)"
    ylabel : str, optional
        Y-axis label, by default "Power"
    figsize : tuple, optional
        Figure size, by default (10, 6)
    show_target : bool, optional
        Whether to show target power line, by default True
    
    Returns
    -------
    plt.Figure
        The matplotlib figure object
    """
    # Calculate power for each sample size
    powers = np.array([power_function(n) for n in sample_sizes])
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot power curve
    ax.plot(sample_sizes, powers, 'b-', linewidth=2)
    
    # Add target power line if requested
    if show_target:
        ax.axhline(y=target_power, color='r', linestyle='--', alpha=0.7)
        
        # Find closest sample size to target power
        idx_target = np.argmin(np.abs(powers - target_power))
        target_n = sample_sizes[idx_target]
        
        # Add annotation
        ax.annotate(f'n = {target_n}', 
                   xy=(target_n, target_power),
                   xytext=(target_n + 0.1 * (sample_sizes[-1] - sample_sizes[0]), 
                           target_power - 0.05),
                   arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                   fontsize=10)
    
    # Set plot parameters
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Set y-axis limits
    ax.set_ylim([0, 1.05])
    
    # Add horizontal grid lines at 0.8 and 0.9 power
    ax.axhline(y=0.8, color='gray', linestyle=':', alpha=0.5)
    ax.axhline(y=0.9, color='gray', linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    
    return fig


def sample_size_curve(effect_sizes: np.ndarray,
                      sample_size_function: Callable[[float], int],
                      target_effect: Optional[float] = None,
                      title: str = "Required Sample Size by Effect Size",
                      xlabel: str = "Effect Size",
                      ylabel: str = "Required Sample Size (per group)",
                      figsize: Tuple[float, float] = (10, 6),
                      log_scale: bool = False) -> plt.Figure:
    """
    Generate a curve showing required sample size as a function of effect size.
    
    Parameters
    ----------
    effect_sizes : np.ndarray
        Array of effect sizes to plot
    sample_size_function : callable
        Function that calculates sample size given an effect size
    target_effect : float, optional
        Target effect size to highlight, by default None
    title : str, optional
        Plot title, by default "Required Sample Size by Effect Size"
    xlabel : str, optional
        X-axis label, by default "Effect Size"
    ylabel : str, optional
        Y-axis label, by default "Required Sample Size (per group)"
    figsize : tuple, optional
        Figure size, by default (10, 6)
    log_scale : bool, optional
        Whether to use log scale for y-axis, by default False
    
    Returns
    -------
    plt.Figure
        The matplotlib figure object
    """
    # Calculate sample size for each effect size
    sample_sizes = np.array([sample_size_function(es) for es in effect_sizes])
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot sample size curve
    ax.plot(effect_sizes, sample_sizes, 'g-', linewidth=2)
    
    # Add target effect size line if provided
    if target_effect is not None:
        ax.axvline(x=target_effect, color='r', linestyle='--', alpha=0.7)
        
        # Calculate sample size at target effect
        target_n = sample_size_function(target_effect)
        
        # Add annotation
        ax.annotate(f'n = {target_n} @ effect size = {target_effect:.3f}', 
                   xy=(target_effect, target_n),
                   xytext=(target_effect + 0.1 * (effect_sizes[-1] - effect_sizes[0]), 
                           target_n * 0.8),
                   arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                   fontsize=10)
    
    # Set plot parameters
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Apply log scale if requested
    if log_scale:
        ax.set_yscale('log')
    
    plt.tight_layout()
    
    return fig


def power_heatmap(param1_values: np.ndarray,
                 param2_values: np.ndarray,
                 power_function: Callable[[float, float], float],
                 param1_name: str = "Parameter 1",
                 param2_name: str = "Parameter 2",
                 title: str = "Power Analysis Heatmap",
                 cmap: str = "viridis",
                 figsize: Tuple[float, float] = (12, 8),
                 show_contour: bool = True) -> plt.Figure:
    """
    Generate a heatmap showing power as a function of two parameters.
    
    Parameters
    ----------
    param1_values : np.ndarray
        Array of values for the first parameter
    param2_values : np.ndarray
        Array of values for the second parameter
    power_function : callable
        Function that calculates power given both parameters
    param1_name : str, optional
        Name of first parameter, by default "Parameter 1"
    param2_name : str, optional
        Name of second parameter, by default "Parameter 2"
    title : str, optional
        Plot title, by default "Power Analysis Heatmap"
    cmap : str, optional
        Colormap name, by default "viridis"
    figsize : tuple, optional
        Figure size, by default (12, 8)
    show_contour : bool, optional
        Whether to show contour lines, by default True
    
    Returns
    -------
    plt.Figure
        The matplotlib figure object
    """
    # Create meshgrid
    X, Y = np.meshgrid(param1_values, param2_values)
    Z = np.zeros_like(X)
    
    # Calculate power for each parameter combination
    for i, p1 in enumerate(param1_values):
        for j, p2 in enumerate(param2_values):
            Z[j, i] = power_function(p1, p2)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    im = ax.imshow(Z, interpolation='bilinear', origin='lower',
                  extent=[param1_values.min(), param1_values.max(),
                          param2_values.min(), param2_values.max()],
                  aspect='auto', cmap=cmap)
    
    # Add contour lines if requested
    if show_contour:
        contour_levels = [0.5, 0.8, 0.9, 0.95]
        contours = ax.contour(X, Y, Z, levels=contour_levels, colors='black', alpha=0.6)
        ax.clabel(contours, inline=True, fontsize=9)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Power', rotation=270, labelpad=20, fontsize=12)
    
    # Set plot parameters
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(param1_name, fontsize=12)
    ax.set_ylabel(param2_name, fontsize=12)
    
    plt.tight_layout()
    
    return fig


def comparison_plot(x_values: np.ndarray,
                   y_functions: List[Callable[[float], float]],
                   labels: List[str],
                   title: str = "Comparison Plot",
                   xlabel: str = "X",
                   ylabel: str = "Y",
                   figsize: Tuple[float, float] = (10, 6),
                   colors: Optional[List[str]] = None,
                   linestyles: Optional[List[str]] = None,
                   add_legend: bool = True) -> plt.Figure:
    """
    Generate a plot comparing multiple functions over the same domain.
    
    Parameters
    ----------
    x_values : np.ndarray
        Array of x-values to plot
    y_functions : list of callable
        List of functions to compare
    labels : list of str
        Labels for each function
    title : str, optional
        Plot title, by default "Comparison Plot"
    xlabel : str, optional
        X-axis label, by default "X"
    ylabel : str, optional
        Y-axis label, by default "Y"
    figsize : tuple, optional
        Figure size, by default (10, 6)
    colors : list of str, optional
        Colors for each function, by default None
    linestyles : list of str, optional
        Line styles for each function, by default None
    add_legend : bool, optional
        Whether to add a legend, by default True
    
    Returns
    -------
    plt.Figure
        The matplotlib figure object
    """
    # Default colors and linestyles if not provided
    if colors is None:
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    
    if linestyles is None:
        linestyles = ['-', '--', '-.', ':']
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot each function
    for i, (func, label) in enumerate(zip(y_functions, labels)):
        color = colors[i % len(colors)]
        linestyle = linestyles[i % len(linestyles)]
        
        # Calculate y values
        y_values = np.array([func(x) for x in x_values])
        
        # Plot line
        ax.plot(x_values, y_values, color=color, linestyle=linestyle, 
               linewidth=2, label=label)
    
    # Set plot parameters
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Add legend if requested
    if add_legend:
        ax.legend(fontsize=10)
    
    plt.tight_layout()
    
    return fig


def plot_power_by_design(designs: List[str],
                        sample_sizes: List[int],
                        power_values: List[List[float]],
                        target_power: float = 0.8,
                        title: str = "Power Comparison by Study Design",
                        figsize: Tuple[float, float] = (12, 7)) -> plt.Figure:
    """
    Generate a grouped bar chart comparing power across different designs and sample sizes.
    
    Parameters
    ----------
    designs : list of str
        Names of study designs to compare
    sample_sizes : list of int
        Sample sizes to compare
    power_values : list of list of float
        Power values for each design at each sample size
    target_power : float, optional
        Target power level to highlight, by default 0.8
    title : str, optional
        Plot title, by default "Power Comparison by Study Design"
    figsize : tuple, optional
        Figure size, by default (12, 7)
    
    Returns
    -------
    plt.Figure
        The matplotlib figure object
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Number of designs and sample sizes
    n_designs = len(designs)
    n_sizes = len(sample_sizes)
    
    # Bar width and positions
    width = 0.8 / n_sizes
    positions = np.arange(n_designs)
    
    # Plot bars for each sample size
    for i, n in enumerate(sample_sizes):
        offset = (i - n_sizes/2 + 0.5) * width
        bars = ax.bar(positions + offset, [pv[i] for pv in power_values], 
                     width=width, label=f'n = {n}')
    
    # Add target power line
    ax.axhline(y=target_power, color='r', linestyle='--', 
               alpha=0.7, label=f'Target Power = {target_power}')
    
    # Set plot parameters
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Study Design', fontsize=12)
    ax.set_ylabel('Power', fontsize=12)
    ax.set_xticks(positions)
    ax.set_xticklabels(designs, rotation=45, ha='right')
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3)
    
    # Add legend
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    
    return fig
