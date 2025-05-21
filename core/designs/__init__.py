"""
Study design modules for sample size and power calculation.

This package contains modules for different study designs:
- parallel: Parallel group randomized controlled trials
- single_arm: Single arm (one-sample) designs
- cluster: Cluster randomized controlled trials
- cluster_rct: Cluster randomized controlled trials (updated implementation)
- stepped_wedge: Stepped wedge cluster randomized trials
- interrupted_time_series: Interrupted time series designs
"""

# Import design subpackages
from . import parallel
from . import single_arm
from . import cluster
from . import cluster_rct
from . import stepped_wedge
from . import interrupted_time_series

# Export commonly used functions at the designs level for easier access

# Parallel RCT functions
from .parallel.binary import sample_size_binary, power_binary
from .parallel.continuous import sample_size_continuous, power_continuous

# Single arm functions
from .single_arm.continuous import one_sample_t_test_sample_size
from .single_arm.binary import one_sample_proportion_test_sample_size

__all__ = [
    # Subpackages
    'parallel',
    'single_arm',
    'cluster',
    'cluster_rct',
    'stepped_wedge',
    'interrupted_time_series',
    
    # Common functions
    'sample_size_binary',
    'power_binary',
    'sample_size_continuous',
    'power_continuous',
    'one_sample_t_test_sample_size',
    'one_sample_proportion_test_sample_size'
]
