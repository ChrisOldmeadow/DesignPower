"""
Simulation methods for cluster RCT binary outcomes - backward compatibility.

This module exports all functions from the modular simulation_binary components
to maintain backward compatibility with existing imports.
"""

# Import core simulation functions
from .core_simulation import (
    simulate_binary_trial,
    _analyze_binary_deff_ztest,
    _analyze_binary_agg_ttest,
    _analyze_binary_permutation,
    _analyze_binary_glmm,
    _analyze_binary_gee,
    _analyze_binary_bayes
)

# Import power calculation functions
from .power_functions import (
    power_binary_sim
)

# Import sample size calculation functions
from .sample_size_functions import (
    sample_size_binary_sim
)

# Import minimum detectable effect functions
from .effect_size_functions import (
    min_detectable_effect_binary_sim
)

# Import Bayesian methods (optional)
from .bayesian_methods import (
    _get_stan_binary_model,
    _fit_pymc_binary_model,
    _fit_variational_bayes_binary,
    _fit_abc_bayes_binary,
    _STAN_AVAILABLE,
    _PYMC_AVAILABLE,
    _SCIPY_AVAILABLE
)

# Export all functions for backward compatibility
__all__ = [
    # Core simulation
    'simulate_binary_trial',
    '_analyze_binary_deff_ztest',
    '_analyze_binary_agg_ttest',
    '_analyze_binary_permutation',
    '_analyze_binary_glmm',
    '_analyze_binary_gee',
    '_analyze_binary_bayes',
    
    # Power functions
    'power_binary_sim',
    
    # Sample size functions
    'sample_size_binary_sim',
    
    # Effect size functions
    'min_detectable_effect_binary_sim',
    
    # Bayesian methods (private)
    '_get_stan_binary_model',
    '_fit_pymc_binary_model',
    '_fit_variational_bayes_binary',
    '_fit_abc_bayes_binary',
    '_STAN_AVAILABLE',
    '_PYMC_AVAILABLE',
    '_SCIPY_AVAILABLE'
]