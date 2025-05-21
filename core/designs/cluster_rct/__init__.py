"""Cluster Randomized Controlled Trial (Cluster RCT) design module.

This module provides functions for sample size calculation, power analysis,
and minimum detectable effect estimation for cluster randomized controlled trials.
"""

# Import outcome-specific modules
from . import analytical_continuous
from . import analytical_binary
from . import simulation_continuous
from . import simulation_binary

# Re-export key functions from analytical_continuous
from .analytical_continuous import (
    sample_size_continuous,
    power_continuous,
    min_detectable_effect_continuous
)

# Re-export key functions from analytical_binary
from .analytical_binary import (
    sample_size_binary,
    power_binary,
    min_detectable_effect_binary
)

# Re-export key functions from simulation modules
from .simulation_continuous import (
    sample_size_continuous_sim,
    power_continuous_sim,
    min_detectable_effect_continuous_sim
)

from .simulation_binary import (
    sample_size_binary_sim,
    power_binary_sim,
    min_detectable_effect_binary_sim
)

# Define __all__ to control what's imported with wildcard imports
__all__ = [
    # Modules
    'analytical_continuous',
    'analytical_binary',
    'simulation_continuous',
    'simulation_binary',
    
    # Analytical continuous functions
    'sample_size_continuous',
    'power_continuous',
    'min_detectable_effect_continuous',
    
    # Analytical binary functions
    'sample_size_binary',
    'power_binary',
    'min_detectable_effect_binary',
    
    # Simulation continuous functions
    'sample_size_continuous_sim',
    'power_continuous_sim',
    'min_detectable_effect_continuous_sim',
    
    # Simulation binary functions
    'sample_size_binary_sim',
    'power_binary_sim',
    'min_detectable_effect_binary_sim'
]