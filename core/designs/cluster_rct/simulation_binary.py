"""
Backward compatibility layer for cluster RCT simulation_binary module.

This module maintains backward compatibility by importing all functions
from the split simulation_binary components. The actual implementations are now
organized in separate modules under the simulation_binary/ directory.

All functions are imported and re-exported to maintain existing imports.
"""

# Import all functions from the modular components
from .simulation_binary import *

# Explicit imports for clarity and IDE support
from .simulation_binary import (
    # Core simulation functions
    simulate_binary_trial,
    _analyze_binary_deff_ztest,
    _analyze_binary_agg_ttest,
    _analyze_binary_permutation,
    _analyze_binary_glmm,
    _analyze_binary_gee,
    _analyze_binary_bayes,
    
    # Power calculation functions
    power_binary_sim,
    
    # Sample size calculation functions
    sample_size_binary_sim,
    
    # Effect size calculation functions
    min_detectable_effect_binary_sim,
    
    # Bayesian methods (private)
    _get_stan_binary_model,
    _fit_pymc_binary_model,
    _fit_variational_bayes_binary,
    _fit_abc_bayes_binary,
    _STAN_AVAILABLE,
    _PYMC_AVAILABLE,
    _SCIPY_AVAILABLE
)