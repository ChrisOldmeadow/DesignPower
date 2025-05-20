"""
Core package for DesignPower statistical calculations.

This package provides implementations of various study designs and
calculation methods for statistical power, sample size, and 
minimum detectable effect size analysis for different study designs.

The package provides modular components for statistical power analysis
across different study designs and outcome types.
"""

# Import key subpackages
from . import designs
from . import methods
from . import stats

# Version information
__version__ = "1.0.0"

__all__ = ['designs', 'methods', 'stats']
