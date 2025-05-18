"""
Input validation utilities for sample size and power calculations.

This module provides functions to validate input parameters
across different study designs and outcome types.
"""
import numpy as np


def validate_numeric(value, name, min_value=None, max_value=None, 
                   allow_none=False, allow_zero=True, allow_negative=False,
                   integer_only=False):
    """
    Validate a numeric parameter value.
    
    Parameters
    ----------
    value : float, int, or None
        Value to validate
    name : str
        Name of parameter (for error messages)
    min_value : float or int, optional
        Minimum allowed value, by default None
    max_value : float or int, optional
        Maximum allowed value, by default None
    allow_none : bool, optional
        Whether None is an allowed value, by default False
    allow_zero : bool, optional
        Whether zero is an allowed value, by default True
    allow_negative : bool, optional
        Whether negative values are allowed, by default False
    integer_only : bool, optional
        Whether only integers are allowed, by default False
    
    Returns
    -------
    float or int or None
        Validated value
    
    Raises
    ------
    ValueError
        If the value does not meet the validation criteria
    TypeError
        If the value is not of the expected type
    """
    # Check for None
    if value is None:
        if allow_none:
            return None
        else:
            raise ValueError(f"Parameter '{name}' cannot be None")
    
    # Check type
    if not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError(f"Parameter '{name}' must be numeric, got {type(value).__name__}")
    
    # Convert numpy types to Python types
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    
    # Check if integer required
    if integer_only and not isinstance(value, int) and not value.is_integer():
        raise ValueError(f"Parameter '{name}' must be an integer, got {value}")
    
    # Check for zero
    if value == 0 and not allow_zero:
        raise ValueError(f"Parameter '{name}' cannot be zero")
    
    # Check for negative
    if value < 0 and not allow_negative:
        raise ValueError(f"Parameter '{name}' cannot be negative, got {value}")
    
    # Check min value
    if min_value is not None and value < min_value:
        raise ValueError(f"Parameter '{name}' must be at least {min_value}, got {value}")
    
    # Check max value
    if max_value is not None and value > max_value:
        raise ValueError(f"Parameter '{name}' must be at most {max_value}, got {value}")
    
    return value


def validate_proportion(value, name, allow_none=False):
    """
    Validate a proportion parameter value (between 0 and 1).
    
    Parameters
    ----------
    value : float or None
        Value to validate
    name : str
        Name of parameter (for error messages)
    allow_none : bool, optional
        Whether None is an allowed value, by default False
    
    Returns
    -------
    float or None
        Validated proportion value
    
    Raises
    ------
    ValueError
        If the value is not between 0 and 1
    """
    # First validate as numeric
    value = validate_numeric(value, name, allow_none=allow_none, allow_zero=True, allow_negative=False)
    
    # Check if returned None
    if value is None:
        return None
    
    # Check range
    if value < 0 or value > 1:
        raise ValueError(f"Parameter '{name}' must be between 0 and 1, got {value}")
    
    return value


def validate_power(power, allow_none=False):
    """
    Validate a power parameter value.
    
    Parameters
    ----------
    power : float or None
        Power value to validate (between 0 and 1)
    allow_none : bool, optional
        Whether None is an allowed value, by default False
    
    Returns
    -------
    float or None
        Validated power value
    
    Raises
    ------
    ValueError
        If the power value is not between 0 and 1
    """
    return validate_proportion(power, "power", allow_none=allow_none)


def validate_alpha(alpha, allow_none=False):
    """
    Validate a significance level (alpha) parameter value.
    
    Parameters
    ----------
    alpha : float or None
        Alpha value to validate (between 0 and 1)
    allow_none : bool, optional
        Whether None is an allowed value, by default False
    
    Returns
    -------
    float or None
        Validated alpha value
    
    Raises
    ------
    ValueError
        If the alpha value is not between 0 and 1
    """
    value = validate_proportion(alpha, "alpha", allow_none=allow_none)
    
    # Additional check for common range
    if value is not None and value > 0.2:
        # Don't raise error, but provide a warning
        import warnings
        warnings.warn(f"Significance level (alpha) of {value} is unusually high")
    
    return value


def validate_sample_size(n, name="sample_size", allow_none=False):
    """
    Validate a sample size parameter value.
    
    Parameters
    ----------
    n : int or None
        Sample size value to validate
    name : str, optional
        Name of parameter (for error messages), by default "sample_size"
    allow_none : bool, optional
        Whether None is an allowed value, by default False
    
    Returns
    -------
    int or None
        Validated sample size value
    
    Raises
    ------
    ValueError
        If the sample size is not a positive integer
    """
    return validate_numeric(n, name, min_value=1, allow_none=allow_none, 
                           allow_zero=False, integer_only=True)


def validate_std_dev(std_dev, name="std_dev", allow_none=False):
    """
    Validate a standard deviation parameter value.
    
    Parameters
    ----------
    std_dev : float or None
        Standard deviation value to validate
    name : str, optional
        Name of parameter (for error messages), by default "std_dev"
    allow_none : bool, optional
        Whether None is an allowed value, by default False
    
    Returns
    -------
    float or None
        Validated standard deviation value
    
    Raises
    ------
    ValueError
        If the standard deviation is not positive
    """
    return validate_numeric(std_dev, name, min_value=0, allow_none=allow_none, 
                           allow_zero=False, allow_negative=False)


def validate_effect_size(effect_size, name="effect_size", allow_none=False, allow_zero=False):
    """
    Validate an effect size parameter value.
    
    Parameters
    ----------
    effect_size : float or None
        Effect size value to validate
    name : str, optional
        Name of parameter (for error messages), by default "effect_size"
    allow_none : bool, optional
        Whether None is an allowed value, by default False
    allow_zero : bool, optional
        Whether zero is an allowed value, by default False
    
    Returns
    -------
    float or None
        Validated effect size value
    
    Raises
    ------
    ValueError
        If the effect size is not valid
    """
    # Effect size can be positive or negative
    return validate_numeric(effect_size, name, allow_none=allow_none, 
                           allow_zero=allow_zero, allow_negative=True)


def validate_allocation_ratio(ratio, allow_none=False):
    """
    Validate an allocation ratio parameter value.
    
    Parameters
    ----------
    ratio : float or None
        Allocation ratio value to validate
    allow_none : bool, optional
        Whether None is an allowed value, by default False
    
    Returns
    -------
    float or None
        Validated allocation ratio value
    
    Raises
    ------
    ValueError
        If the allocation ratio is not positive
    """
    return validate_numeric(ratio, "allocation_ratio", min_value=0, 
                           allow_none=allow_none, allow_zero=False)


def validate_icc(icc, allow_none=False):
    """
    Validate an intraclass correlation coefficient (ICC) parameter value.
    
    Parameters
    ----------
    icc : float or None
        ICC value to validate (between 0 and 1)
    allow_none : bool, optional
        Whether None is an allowed value, by default False
    
    Returns
    -------
    float or None
        Validated ICC value
    
    Raises
    ------
    ValueError
        If the ICC value is not between 0 and 1
    """
    return validate_proportion(icc, "icc", allow_none=allow_none)


def validate_cluster_size(cluster_size, allow_none=False):
    """
    Validate a cluster size parameter value.
    
    Parameters
    ----------
    cluster_size : int or None
        Cluster size value to validate
    allow_none : bool, optional
        Whether None is an allowed value, by default False
    
    Returns
    -------
    int or None
        Validated cluster size value
    
    Raises
    ------
    ValueError
        If the cluster size is not a positive integer
    """
    return validate_sample_size(cluster_size, "cluster_size", allow_none=allow_none)
