"""
Simulation methods for cluster RCT continuous outcomes - backward compatibility.

This module exports all functions from the modular simulation_continuous components
to maintain backward compatibility with existing imports.
"""

# Import core simulation functions
from .core_simulation import (
    simulate_continuous_trial
)

# Import power calculation functions
from .power_functions import (
    power_continuous_sim
)

# Import sample size calculation functions
from .sample_size_functions import (
    sample_size_continuous_sim
)

# Import minimum detectable effect functions
from .effect_size_functions import (
    min_detectable_effect_continuous_sim
)

# Import analysis utility functions
from .analysis_utils import (
    _analyze_continuous_trial,
    _ols_cluster_test
)

# Import Bayesian methods (optional)
from .bayesian_methods import (
    _get_stan_model,
    _fit_pymc_model,
    _fit_variational_bayes,
    _fit_abc_bayes,
    _STAN_AVAILABLE,
    _PYMC_AVAILABLE,
    _SCIPY_AVAILABLE
)

# Export all functions for backward compatibility
__all__ = [
    # Core simulation
    'simulate_continuous_trial',
    
    # Power functions
    'power_continuous_sim',
    
    # Sample size functions
    'sample_size_continuous_sim',
    
    # Effect size functions
    'min_detectable_effect_continuous_sim',
    
    # Analysis utilities
    '_analyze_continuous_trial',
    '_ols_cluster_test',
    
    # Bayesian methods (private)
    '_get_stan_model',
    '_fit_pymc_model',
    '_fit_variational_bayes',
    '_fit_abc_bayes',
    '_STAN_AVAILABLE',
    '_PYMC_AVAILABLE',
    '_SCIPY_AVAILABLE'
]