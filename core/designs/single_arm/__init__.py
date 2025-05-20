"""
Single-arm (one-sample) study designs.

This module provides functions for power analysis and sample size calculation
for single-arm studies with various outcome types.
"""

# Import submodules
from . import continuous
from . import binary
from . import survival

# Re-export key functions for binary outcomes
from .binary import (
    one_sample_proportion_test_sample_size,
    one_sample_proportion_test_power
)

# Re-export key functions for continuous outcomes
from .continuous import (
    one_sample_t_test_sample_size,
    one_sample_t_test_power,
    min_detectable_effect_one_sample_continuous
)

# Re-export key functions for survival outcomes
from .survival import (
    one_sample_survival_test_sample_size,
    one_sample_survival_test_power,
    min_detectable_effect_one_sample_survival
)

# Define what gets imported with "from core.designs.single_arm import *"
__all__ = [
    # Submodules
    'binary',
    'continuous',
    'survival',
    
    # Binary outcome functions
    'one_sample_proportion_test_sample_size',
    'one_sample_proportion_test_power',
    
    # Continuous outcome functions
    'one_sample_t_test_sample_size',
    'one_sample_t_test_power',
    'min_detectable_effect_one_sample_continuous',
    
    # Survival outcome functions
    'one_sample_survival_test_sample_size',
    'one_sample_survival_test_power',
    'min_detectable_effect_one_sample_survival'
]