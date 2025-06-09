"""
Backward compatibility layer for simulation_continuous module.

This module maintains backward compatibility by importing all functions
from the split simulation_continuous components. The actual implementations are now
organized in separate modules under the simulation_continuous/ directory.

All functions are imported and re-exported to maintain existing imports.
"""

# Import all functions from the modular components
from .simulation_continuous import *

# Explicit imports for clarity and IDE support
from .simulation_continuous import (
    # Core simulation functions
    simulate_continuous_trial,
    simulate_continuous_non_inferiority,
    
    # Power calculation functions
    power_continuous_sim,
    min_detectable_effect_continuous_sim,
    
    # Sample size calculation functions
    sample_size_continuous_sim,
    
    # Non-inferiority functions
    sample_size_continuous_non_inferiority_sim,
    power_continuous_non_inferiority_sim,
    
    # Utility functions (private)
    _calculate_effective_sd,
    _calculate_welch_satterthwaite_df,
    _simulate_single_continuous_non_inferiority_trial
)