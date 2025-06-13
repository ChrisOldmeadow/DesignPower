"""
Shared CLI utilities and base components.

This module contains shared enums, display functions, and utilities
used across all CLI commands.
"""
import typer
from typing import Optional
from enum import Enum
from rich.console import Console
from rich.table import Table
import json

console = Console()


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


def display_result(result, method_name):
    """Display calculation result in a formatted table."""
    # Display main result
    table = Table(title=f"{method_name} Results")
    table.add_column("Parameter", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")
    
    for key, value in result.items():
        if isinstance(value, (int, float)):
            formatted_value = f"{value:.4f}" if isinstance(value, float) else str(value)
        else:
            formatted_value = str(value)
        table.add_row(key, formatted_value)
    
    console.print(table)
    
    # Also print JSON for programmatic use
    console.print("\nJSON output:")
    console.print(json.dumps(result, indent=2))