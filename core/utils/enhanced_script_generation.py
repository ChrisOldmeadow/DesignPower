"""
Enhanced script generation utilities that include algorithm source code.
"""

from .source_extraction import get_function_source


def add_algorithm_source_to_script(script_content, function_name, actual_function):
    """
    Add algorithm source code to a generated script.
    
    Parameters
    ----------
    script_content : str
        The existing script content
    function_name : str
        Name of the function being used
    actual_function : callable
        The actual function object to extract source from
        
    Returns
    -------
    str
        Enhanced script with algorithm source code
    """
    # Find where to insert the algorithm source (after imports, before main analysis)
    import_section_end = script_content.find('\n# Calculate')
    if import_section_end == -1:
        # Fallback: insert after the import line
        import_line_end = script_content.find('\n\n', script_content.find('from core.designs'))
        if import_line_end != -1:
            import_section_end = import_line_end
        else:
            # If we can't find a good spot, add at the end of imports section
            import_section_end = script_content.find('def main():') - 1
    
    if import_section_end == -1:
        # Last resort: add after the first function import
        import_section_end = script_content.find('\n', script_content.find('import')) + 1
    
    # Get the algorithm source code
    algorithm_section = f"""
# ============================================================================
# STATISTICAL ALGORITHM SOURCE CODE
# 
# The following shows the exact implementation of the {function_name} function
# used in this analysis. This provides full transparency of the statistical
# methodology and allows verification of the approach.
# ============================================================================

{get_function_source(actual_function)}

# ============================================================================
# MAIN ANALYSIS USING THE ALGORITHM SHOWN ABOVE
# ============================================================================
"""
    
    # Insert the algorithm section
    enhanced_script = (
        script_content[:import_section_end] + 
        algorithm_section + 
        script_content[import_section_end:]
    )
    
    return enhanced_script


def get_function_from_name(module_path, function_name):
    """
    Dynamically import and return a function given its module path and name.
    
    Parameters
    ----------
    module_path : str
        Module path like "core.designs.parallel.analytical_continuous"
    function_name : str
        Function name like "sample_size_continuous"
        
    Returns
    -------
    callable or None
        The function object, or None if import fails
    """
    try:
        # Dynamic import
        module = __import__(module_path, fromlist=[function_name])
        return getattr(module, function_name)
    except (ImportError, AttributeError):
        return None


def enhance_parallel_continuous_script(params, original_script):
    """
    Enhance parallel continuous script with algorithm source code.
    
    Parameters
    ----------
    params : dict
        Parameters used to determine which function is being called
    original_script : str
        The original generated script
        
    Returns
    -------
    str
        Enhanced script with algorithm source code
    """
    # Determine which function is being used based on parameters
    calc_type = params.get('calculation_type', 'Sample Size')
    method = params.get('method', 'analytical')
    hypothesis_type = params.get('hypothesis_type', 'Superiority')
    repeated_measures = params.get('repeated_measures', False)
    
    # Map to the actual function being used
    if calc_type == "Sample Size":
        if repeated_measures:
            if method == "simulation":
                function_name = "sample_size_repeated_measures_sim"
                module_path = "core.designs.parallel.simulation_continuous"
            else:
                function_name = "sample_size_repeated_measures"
                module_path = "core.designs.parallel.analytical_continuous"
        elif hypothesis_type == "Non-Inferiority":
            if method == "simulation":
                function_name = "sample_size_continuous_non_inferiority_sim"
                module_path = "core.designs.parallel.simulation_continuous"
            else:
                function_name = "sample_size_continuous_non_inferiority"
                module_path = "core.designs.parallel.analytical.non_inferiority"
        else:
            # Standard superiority
            if method == "simulation":
                function_name = "sample_size_continuous_sim"
                module_path = "core.designs.parallel.simulation_continuous"
            else:
                function_name = "sample_size_continuous"
                module_path = "core.designs.parallel.analytical_continuous"
    
    # Similar logic for Power and MDE calculations...
    # (abbreviated for now, can be extended)
    
    # Get the actual function
    actual_function = get_function_from_name(module_path, function_name)
    
    if actual_function:
        return add_algorithm_source_to_script(original_script, function_name, actual_function)
    else:
        # If we can't get the function, add a note
        note = f"""
# ============================================================================
# STATISTICAL ALGORITHM: {function_name}
# Module: {module_path}
# 
# Note: Algorithm source code could not be extracted automatically.
# The function {function_name} from {module_path} is being used.
# ============================================================================
"""
        # Add the note after imports
        import_end = original_script.find('\n# Calculate')
        if import_end != -1:
            return original_script[:import_end] + note + original_script[import_end:]
        else:
            return original_script + note