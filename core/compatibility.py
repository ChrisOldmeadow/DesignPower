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
    min_detectable_effect_repeated_measures
)

# Import from new structure - cluster design analytical methods
from core.designs.cluster.analytical import (
    power_binary as power_binary_cluster_rct,
    sample_size_binary as sample_size_binary_cluster_rct,
    min_detectable_effect_binary as min_detectable_effect_binary_cluster_rct
)

# Import from new structure - simulation methods
from core.designs.parallel.simulation import (
    simulate_continuous as simulate_parallel_rct,
    min_detectable_effect_continuous as simulate_min_detectable_effect
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
    'simulate_min_detectable_effect',
    
    # Simulation methods
    'simulate_parallel_rct',
    'simulate_cluster_rct',
    'simulate_binary_cluster_rct',
    'simulate_stepped_wedge'
]
