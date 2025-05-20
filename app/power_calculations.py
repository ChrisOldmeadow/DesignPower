"""
Power calculation functions for single-arm studies.

BACKWARD COMPATIBILITY MODULE: This module maintains the original interfaces
but delegates to the new structured core modules.

Import functions from core.designs.single_arm.continuous and 
core.designs.single_arm.binary instead of this module for new code.
"""

# Import functions from new locations to maintain backward compatibility
from core.designs.single_arm.continuous import (
    one_sample_t_test_sample_size,
    one_sample_t_test_power,
    min_detectable_effect_one_sample_continuous
)

from core.designs.single_arm.binary import (
    one_sample_proportion_test_sample_size,
    one_sample_proportion_test_power,
    min_detectable_effect_one_sample_binary
)

from core.designs.single_arm.survival import (
    one_sample_survival_test_sample_size,
    one_sample_survival_test_power,
    min_detectable_effect_one_sample_survival
)

# Note: This module now just re-exports the functions from the new module structure.
# All function implementations have been moved to core.designs.single_arm.continuous 
# and core.designs.single_arm.binary.
