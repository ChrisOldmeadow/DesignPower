"""
Formatting utilities for the DesignPower application.

This module provides helper functions for formatting numbers, tables,
and other output to ensure consistent presentation across the application.
"""

def format_number_with_precision(value, precision=2):
    """
    Format a number with the specified precision.
    
    Parameters
    ----------
    value : float or int
        The number to format
    precision : int, optional
        Number of decimal places to include, by default 2
        
    Returns
    -------
    str
        Formatted number as a string
    """
    if value is None:
        return "N/A"
        
    if isinstance(value, int):
        return str(value)
        
    # Format float with specified precision
    format_str = f"{{:.{precision}f}}"
    return format_str.format(value)
    
def format_percentage(value, precision=1):
    """
    Format a proportion as a percentage with the specified precision.
    
    Parameters
    ----------
    value : float
        Proportion value (between 0 and 1)
    precision : int, optional
        Number of decimal places to include, by default 1
        
    Returns
    -------
    str
        Formatted percentage as a string with % symbol
    """
    if value is None:
        return "N/A"
        
    # Convert to percentage and format
    percentage = value * 100
    format_str = f"{{:.{precision}f}}%"
    return format_str.format(percentage)
    
def format_p_value(p_value, threshold=0.001):
    """
    Format a p-value with appropriate precision.
    
    For very small p-values, use scientific notation or "<threshold" format.
    
    Parameters
    ----------
    p_value : float
        The p-value to format
    threshold : float, optional
        Threshold below which to use special formatting, by default 0.001
        
    Returns
    -------
    str
        Formatted p-value as a string
    """
    if p_value is None:
        return "N/A"
        
    if p_value < threshold:
        return f"<{threshold}"
    
    if p_value < 0.01:
        return f"{p_value:.3f}"
    
    return f"{p_value:.2f}"
