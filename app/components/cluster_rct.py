"""Component module for Cluster RCT designs.

This module provides UI rendering functions and calculation functions for
Cluster Randomized Controlled Trial designs with continuous and binary outcomes.

This is now a compatibility layer that imports from the split modules
to maintain backward compatibility.
"""

# Import all functions from the split modules to maintain backward compatibility
from .cluster_rct.continuous_ui import render_cluster_continuous, render_continuous_advanced_options
from .cluster_rct.binary_ui import render_cluster_binary, render_binary_advanced_options  
from .cluster_rct.calculations import calculate_cluster_continuous, calculate_cluster_binary
from .cluster_rct.cli_generation import generate_cli_code_cluster_continuous, generate_cli_code_cluster_binary

# Re-export all functions for backward compatibility
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