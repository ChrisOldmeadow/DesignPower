"""Parallel RCT component module - maintains backward compatibility.

This module exports all functions from the split parallel RCT components
to maintain backward compatibility with existing imports.
"""

# Import CLI generation functions
from .cli_generation import (
    generate_cli_code_parallel_binary,
    generate_cli_code_parallel_survival,
    generate_cli_code_parallel_continuous
)

# Import binary UI functions
from .binary_ui import (
    render_binary_advanced_options,
    render_parallel_binary
)

# Import continuous UI functions
from .continuous_ui import (
    render_continuous_advanced_options,
    render_parallel_continuous
)

# Import survival UI functions
from .survival_ui import (
    render_parallel_survival,
    display_survival_results,
    create_survival_visualization
)

# Import calculation functions
from .calculations import (
    calculate_parallel_continuous,
    calculate_parallel_binary,
    calculate_parallel_survival
)

# Export all functions for backward compatibility
__all__ = [
    # CLI generation
    'generate_cli_code_parallel_binary',
    'generate_cli_code_parallel_survival', 
    'generate_cli_code_parallel_continuous',
    
    # UI components
    'render_binary_advanced_options',
    'render_parallel_binary',
    'render_continuous_advanced_options',
    'render_parallel_continuous',
    'render_parallel_survival',
    'display_survival_results',
    'create_survival_visualization',
    
    # Calculations
    'calculate_parallel_continuous',
    'calculate_parallel_binary',
    'calculate_parallel_survival'
]