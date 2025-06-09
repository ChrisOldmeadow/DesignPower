"""Simulation methods for continuous outcomes - backward compatibility.

This module exports all functions from the modular simulation_continuous components
to maintain backward compatibility with existing imports.
"""

# Import core simulation functions
from .core_simulation import (
    simulate_continuous_trial,
    simulate_continuous_non_inferiority
)

# Import power calculation functions
from .power_functions import (
    power_continuous_sim,
    min_detectable_effect_continuous_sim
)

# Import sample size calculation functions
from .sample_size_functions import (
    sample_size_continuous_sim
)

# Import non-inferiority functions
from .non_inferiority import (
    sample_size_continuous_non_inferiority_sim,
    power_continuous_non_inferiority_sim
)

# Import utility functions (note: these are private functions with leading underscore)
from .utils import (
    _calculate_effective_sd,
    _calculate_welch_satterthwaite_df,
    _simulate_single_continuous_non_inferiority_trial
)

# Export all functions for backward compatibility
__all__ = [
    # Core simulation
    'simulate_continuous_trial',
    'simulate_continuous_non_inferiority',
    
    # Power functions
    'power_continuous_sim',
    'min_detectable_effect_continuous_sim',
    
    # Sample size functions
    'sample_size_continuous_sim',
    
    # Non-inferiority functions
    'sample_size_continuous_non_inferiority_sim',
    'power_continuous_non_inferiority_sim',
    
    # Utility functions (private)
    '_calculate_effective_sd',
    '_calculate_welch_satterthwaite_df',
    '_simulate_single_continuous_non_inferiority_trial'
]