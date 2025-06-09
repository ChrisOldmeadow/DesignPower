"""
Backward compatibility layer for cluster RCT simulation_continuous module.

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
    
    # Power calculation functions
    power_continuous_sim,
    
    # Sample size calculation functions
    sample_size_continuous_sim,
    
    # Effect size calculation functions
    min_detectable_effect_continuous_sim,
    
    # Analysis utility functions
    _analyze_continuous_trial,
    _ols_cluster_test,
    
    # Bayesian methods (private)
    _get_stan_model,
    _fit_pymc_model,
    _fit_variational_bayes,
    _fit_abc_bayes,
    _STAN_AVAILABLE,
    _PYMC_AVAILABLE,
    _SCIPY_AVAILABLE
)