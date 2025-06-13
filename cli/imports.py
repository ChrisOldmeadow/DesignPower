"""
Centralized imports for CLI commands.
"""

import typer
from typing import Optional
import json

from core.power import (
    sample_size_difference_in_means,
    power_difference_in_means,
    power_binary_cluster_rct,
    sample_size_binary_cluster_rct,
    min_detectable_effect_binary_cluster_rct
)
from core.designs.single_arm.continuous import (
    one_sample_t_test_sample_size,
    one_sample_t_test_power
)
from core.designs.single_arm.binary import (
    one_sample_proportion_test_sample_size,
    one_sample_proportion_test_power,
    ahern_sample_size,
    ahern_power,
    simons_two_stage_design,
    simons_power
)
from core.designs.single_arm.survival import (
    one_sample_survival_test_sample_size,
    one_sample_survival_test_power
)
from core.designs.parallel.analytical_survival import (
    sample_size_survival,
    power_survival,
    min_detectable_effect_survival,
    sample_size_survival_non_inferiority,
    power_survival_non_inferiority
)
from core.simulation import (
    simulate_parallel_rct,
    simulate_cluster_rct,
    simulate_stepped_wedge,
    simulate_binary_cluster_rct
)

from rich.console import Console
console = Console()

# Import common CLI components
from .common import DesignType, OutcomeType, CalculationType, display_result