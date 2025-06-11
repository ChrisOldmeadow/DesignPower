"""
Analytical methods for parallel group RCTs - backward compatibility.

This module exports all functions from the modular analytical components
to maintain backward compatibility with existing imports.
"""

# Import continuous outcome functions
from .continuous_outcomes import (
    sample_size_continuous,
    power_continuous
)

# Import binary outcome functions
from .binary_outcomes import (
    sample_size_binary,
    power_binary
)

# Import repeated measures functions
from .repeated_measures import (
    sample_size_repeated_measures,
    power_repeated_measures,
    min_detectable_effect_repeated_measures
)

# Import non-inferiority functions
from .non_inferiority import (
    # Continuous non-inferiority
    sample_size_continuous_non_inferiority,
    power_continuous_non_inferiority,
    min_detectable_non_inferiority_margin,
    
    # Binary non-inferiority
    sample_size_binary_non_inferiority,
    power_binary_non_inferiority,
    min_detectable_binary_non_inferiority_margin
)

# Export all functions for backward compatibility
__all__ = [
    # Continuous outcomes
    'sample_size_continuous',
    'power_continuous',
    
    # Binary outcomes
    'sample_size_binary',
    'power_binary',
    
    # Repeated measures
    'sample_size_repeated_measures',
    'power_repeated_measures',
    'min_detectable_effect_repeated_measures',
    
    # Non-inferiority continuous
    'sample_size_continuous_non_inferiority',
    'power_continuous_non_inferiority',
    'min_detectable_non_inferiority_margin',
    
    # Non-inferiority binary
    'sample_size_binary_non_inferiority',
    'power_binary_non_inferiority',
    'min_detectable_binary_non_inferiority_margin'
]
