"""
Display and formatting utilities for CLI results.
"""

import json
from rich.console import Console
from rich.table import Table
from core.utils import generate_plain_language_summary

console = Console()


def display_result(result, method_name):
    """Display calculation result in a formatted table."""
    # Display main result
    main_table = Table(title="Calculation Result")
    
    # Add columns dynamically based on result keys
    main_table.add_column("Metric", style="cyan")
    main_table.add_column("Value", style="green")
    
    for key, value in result.items():
        if key != "parameters" and not isinstance(value, dict):
            # Format float values
            if isinstance(value, float):
                formatted_value = f"{value:.4f}"
            else:
                formatted_value = str(value)
            
            main_table.add_row(key.replace("_", " ").title(), formatted_value)
    
    console.print(main_table)
    
    # Display parameters
    param_table = Table(title="Input Parameters")
    param_table.add_column("Parameter", style="cyan")
    param_table.add_column("Value", style="green")
    
    for key, value in result.get("parameters", {}).items():
        # Format float values
        if isinstance(value, float):
            formatted_value = f"{value:.4f}"
        else:
            formatted_value = str(value)
        
        param_table.add_row(key.replace("_", " ").title(), formatted_value)
    
    console.print(param_table)
    
    # Display plain language summary
    summary = generate_plain_language_summary(method_name, result)
    console.print("\n[bold]Summary:[/bold]")
    console.print(summary)