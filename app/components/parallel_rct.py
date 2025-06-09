"""
Backward compatibility layer for Parallel RCT components.

This module maintains backward compatibility by importing all functions
from the split parallel_rct components. The actual implementations are now
organized in separate modules under the parallel_rct/ directory.

All functions are imported and re-exported to maintain existing imports.
"""

# Import all functions from the modular components
from .parallel_rct import *

# Explicit imports for clarity and IDE support
from .parallel_rct import (
    # CLI generation functions
    generate_cli_code_parallel_binary,
    generate_cli_code_parallel_survival,
    generate_cli_code_parallel_continuous,
    
    # UI rendering functions
    render_binary_advanced_options,
    render_parallel_binary,
    render_continuous_advanced_options,
    render_parallel_continuous,
    render_parallel_survival,
    display_survival_results,
    create_survival_visualization,
    
    # Calculation functions
    calculate_parallel_continuous,
    calculate_parallel_binary,
    calculate_parallel_survival
)