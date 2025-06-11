"""
Backward compatibility layer for parallel group analytical methods.

This module maintains backward compatibility by importing all functions
from the split analytical components. The actual implementations are now
organized in separate modules under the analytical/ directory.

All functions are imported and re-exported to maintain existing imports.
"""

# Import all functions from the modular components
from .analytical import *

# Explicit imports for clarity and IDE support
from .analytical import (
    # Continuous outcomes
    sample_size_continuous,
    power_continuous,
    
    # Binary outcomes
    sample_size_binary,
    power_binary,
    
    # Repeated measures
    sample_size_repeated_measures,
    power_repeated_measures,
    min_detectable_effect_repeated_measures,
    
    # Non-inferiority continuous
    sample_size_continuous_non_inferiority,
    power_continuous_non_inferiority,
    min_detectable_non_inferiority_margin,
    
    # Non-inferiority binary
    sample_size_binary_non_inferiority,
    power_binary_non_inferiority,
    min_detectable_binary_non_inferiority_margin
)
