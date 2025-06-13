"""
Enum definitions for CLI commands.
"""

from enum import Enum


class DesignType(str, Enum):
    """Enum for supported study designs."""
    PARALLEL = "parallel"
    CLUSTER = "cluster"
    STEPPED_WEDGE = "stepped-wedge"
    SINGLE_ARM = "single-arm"


class OutcomeType(str, Enum):
    """Enum for supported outcome types."""
    CONTINUOUS = "continuous"
    BINARY = "binary"
    SURVIVAL = "survival"


class CalculationType(str, Enum):
    """Enum for calculation types."""
    SAMPLE_SIZE = "sample-size"
    POWER = "power"
    MDE = "mde"  # Minimum Detectable Effect