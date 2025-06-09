"""Cluster RCT component module with backward compatibility.

This module maintains backward compatibility by re-exporting all
functions from the split modules.
"""

# Import all functions to maintain backward compatibility
from .continuous_ui import render_cluster_continuous, render_continuous_advanced_options
from .binary_ui import render_cluster_binary, render_binary_advanced_options  
from .calculations import calculate_cluster_continuous, calculate_cluster_binary
from .cli_generation import generate_cli_code_cluster_continuous, generate_cli_code_cluster_binary

# Export all functions for backward compatibility
__all__ = [
    'render_cluster_continuous',
    'render_continuous_advanced_options', 
    'render_cluster_binary',
    'render_binary_advanced_options',
    'calculate_cluster_continuous',
    'calculate_cluster_binary',
    'generate_cli_code_cluster_continuous',
    'generate_cli_code_cluster_binary'
]