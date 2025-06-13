"""
Utility functions for extracting and displaying source code of key functions
in generated reproducible scripts.
"""

import inspect
import textwrap


def get_function_source(func):
    """
    Extract the source code of a function with proper formatting.
    
    Parameters
    ----------
    func : callable
        The function to extract source code from
        
    Returns
    -------
    str
        Formatted source code of the function
    """
    try:
        # Get the source code
        source = inspect.getsource(func)
        
        # Remove any leading indentation to normalize
        source = textwrap.dedent(source)
        
        # Add a comment header
        func_name = func.__name__
        module_name = func.__module__
        
        header = f"""
# ============================================================================
# KEY ALGORITHM: {func_name}
# Source: {module_name}
# ============================================================================
"""
        
        return header + source
        
    except (OSError, TypeError) as e:
        # If we can't get source (e.g., built-in function), return a note
        return f"""
# ============================================================================
# KEY ALGORITHM: {func.__name__}
# Note: Source code not available (built-in or compiled function)
# ============================================================================
"""


def get_multiple_functions_source(functions):
    """
    Extract source code for multiple functions.
    
    Parameters
    ----------
    functions : list
        List of functions to extract source code from
        
    Returns
    -------
    str
        Combined source code of all functions
    """
    sources = []
    
    for func in functions:
        sources.append(get_function_source(func))
    
    return "\n".join(sources)


def format_function_documentation(func):
    """
    Extract and format function documentation.
    
    Parameters
    ----------
    func : callable
        The function to extract documentation from
        
    Returns
    -------
    str
        Formatted documentation
    """
    try:
        doc = inspect.getdoc(func)
        if doc:
            return f"""
# ALGORITHM DOCUMENTATION:
# {doc}
"""
        else:
            return f"""
# ALGORITHM: {func.__name__}
# No documentation available
"""
    except Exception:
        return f"""
# ALGORITHM: {func.__name__}
# Documentation extraction failed
"""