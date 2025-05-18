"""
Compatibility layer for backward compatibility with older API.

This module re-exports functions from the new modular structure with their
original names to maintain backward compatibility with existing code.
"""

# Import from new structure - parallel design analytical methods
from core.designs.parallel.analytical import (
    sample_size_continuous as sample_size_difference_in_means,
    power_continuous as power_difference_in_means,
    sample_size_repeated_measures,
    power_repeated_measures,
    min_detectable_effect_repeated_measures,
    sample_size_continuous_non_inferiority,
    power_continuous_non_inferiority,
    min_detectable_non_inferiority_margin,
    sample_size_binary_non_inferiority,
    power_binary_non_inferiority,
    min_detectable_binary_non_inferiority_margin
)

# Import from new structure - cluster design analytical methods
from core.designs.cluster.analytical import (
    power_binary as power_binary_cluster_rct,
    sample_size_binary as sample_size_binary_cluster_rct,
    min_detectable_effect_binary as min_detectable_effect_binary_cluster_rct
)

# Import from new structure - parallel design simulation methods
from core.designs.parallel.simulation import (
    simulate_continuous as simulate_parallel_rct,
    min_detectable_effect_continuous as simulate_min_detectable_effect,
    sample_size_continuous as simulate_sample_size,
    simulate_continuous_non_inferiority,
    sample_size_continuous_non_inferiority as simulate_sample_size_non_inferiority,
    simulate_binary
)

# Import binary simulation functions
from core.designs.parallel.binary_simulation import (
    simulate_binary_non_inferiority,
    sample_size_binary_sim,
    min_detectable_effect_binary_sim,
    sample_size_binary_non_inferiority_sim,
    min_detectable_binary_non_inferiority_margin_sim
)

from core.designs.cluster.simulation import (
    simulate_binary as simulate_binary_cluster_rct,
    simulate_continuous as simulate_cluster_rct
)

from core.designs.stepped_wedge.simulation import (
    simulate_continuous as simulate_stepped_wedge
)

# Re-export all imported functions
__all__ = [
    # Analytical methods
    'sample_size_difference_in_means',
    'power_difference_in_means',
    'power_binary_cluster_rct',
    'sample_size_binary_cluster_rct',
    'min_detectable_effect_binary_cluster_rct',
    'sample_size_repeated_measures',
    'power_repeated_measures',
    'min_detectable_effect_repeated_measures',
    # Non-inferiority testing methods - continuous outcomes
    'sample_size_continuous_non_inferiority',
    'power_continuous_non_inferiority',
    'min_detectable_non_inferiority_margin',
    'simulate_continuous_non_inferiority',
    'simulate_sample_size_non_inferiority',
    # Non-inferiority testing methods - binary outcomes
    'sample_size_binary_non_inferiority',
    'power_binary_non_inferiority',
    'min_detectable_binary_non_inferiority_margin',
    # Binary simulation functions
    'simulate_binary',
    'simulate_binary_non_inferiority',
    'sample_size_binary_sim',
    'min_detectable_effect_binary_sim',
    'sample_size_binary_non_inferiority_sim',
    'min_detectable_binary_non_inferiority_margin_sim',
    'simulate_min_detectable_effect',
    'simulate_sample_size',
    
    # Simulation methods
    'simulate_parallel_rct',
    'simulate_cluster_rct',
    'simulate_binary_cluster_rct',
    'simulate_stepped_wedge'
]
