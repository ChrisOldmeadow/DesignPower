"""
Parallel group randomized controlled trial modules.

This package contains functions for power analysis and sample size calculation
for parallel group randomized controlled trials with various outcome types.

All functions are organized by outcome type (binary, continuous, survival) and
calculation method (analytical, simulation).
"""

# Import analytical and simulation modules
from . import analytical_binary
from . import simulation_binary
from . import analytical_continuous
from . import simulation_continuous
from . import analytical_survival
from . import simulation_survival
# Import other modules for backward compatibility
from . import continuous
from . import binary

# Re-export key functions for binary outcomes (analytical methods)
from .analytical_binary import (
    sample_size_binary,
    power_binary,
    min_detectable_effect_binary,
    sample_size_binary_non_inferiority,
    power_binary_non_inferiority
)

# Re-export key functions for binary outcomes (simulation methods)
from .simulation_binary import (
    sample_size_binary_sim,
    power_binary_sim,
    min_detectable_effect_binary_sim,
    sample_size_binary_non_inferiority_sim,
    min_detectable_binary_non_inferiority_margin_sim,
    simulate_binary_trial,
    simulate_binary_non_inferiority
)

# Re-export key functions for continuous outcomes (analytical methods)
from .analytical_continuous import (
    sample_size_continuous,
    power_continuous,
    min_detectable_effect_continuous,
    sample_size_continuous_non_inferiority,
    power_continuous_non_inferiority
)

# Re-export key functions for continuous outcomes (simulation methods)
from .simulation_continuous import (
    power_continuous_sim,
    sample_size_continuous_sim,
    min_detectable_effect_continuous_sim,
    sample_size_continuous_non_inferiority_sim,
    simulate_continuous_trial,
    simulate_continuous_non_inferiority
)

# Re-export key functions for survival outcomes (analytical methods)
from .analytical_survival import (
    sample_size_survival,
    power_survival,
    min_detectable_effect_survival,
    sample_size_survival_non_inferiority,
    power_survival_non_inferiority
)

# Re-export key functions for survival outcomes (simulation methods)
from .simulation_survival import (
    sample_size_survival_sim,
    power_survival_sim,
    min_detectable_effect_survival_sim,
    sample_size_survival_non_inferiority_sim,
    power_survival_non_inferiority_sim,
    simulate_survival_trial,
    simulate_survival_non_inferiority
)

# Define what gets imported with "from core.designs.parallel import *"
__all__ = [
    # Submodules by outcome and method type
    'analytical_binary',
    'simulation_binary',
    'analytical_continuous',
    'simulation_continuous',
    'analytical_survival',
    'simulation_survival',
    
    # Legacy submodules for backward compatibility
    'binary',
    'continuous',
    'analytical',
    'simulation',
    
    # Binary outcome functions (analytical)
    'sample_size_binary',
    'power_binary',
    'sample_size_binary_non_inferiority',
    'power_binary_non_inferiority',
    
    # Binary outcome functions (simulation)
    'sample_size_binary_sim',
    'power_binary_sim',
    'min_detectable_effect_binary_sim',
    'sample_size_binary_non_inferiority_sim',
    'min_detectable_binary_non_inferiority_margin_sim',
    'simulate_binary_trial',
    'simulate_binary_non_inferiority',
    
    # Continuous outcome functions (analytical)
    'sample_size_continuous',
    'power_continuous',
    'min_detectable_effect_continuous',
    'sample_size_continuous_non_inferiority',
    'power_continuous_non_inferiority',
    
    # Continuous outcome functions (simulation)
    'power_continuous_sim',
    'sample_size_continuous_sim',
    'min_detectable_effect_continuous_sim',
    'sample_size_continuous_non_inferiority_sim',
    'simulate_continuous_trial',
    'simulate_continuous_non_inferiority'
]
