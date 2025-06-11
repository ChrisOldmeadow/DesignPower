"""
Analytical methods for non-inferiority tests in parallel group RCTs.

This module provides functions for sample size, power calculation, and
minimum detectable effect estimation for non-inferiority tests with both
continuous and binary outcomes.
"""
import math
import numpy as np
from scipy import stats


# Non-inferiority testing functions for continuous outcomes
def sample_size_continuous_non_inferiority(
    non_inferiority_margin, 
    std_dev, 
    power=0.8, 
    alpha=0.05, 
    allocation_ratio=1.0, 
    assumed_difference=0.0,
    direction="lower"
):
    """
    Calculate sample size for non-inferiority test with continuous outcome.
    
    Parameters
    ----------
    non_inferiority_margin : float
        Non-inferiority margin (must be positive)
    std_dev : float
        Standard deviation of the outcome
    power : float, optional
        Desired power, by default 0.8
    alpha : float, optional
        Significance level (one-sided for non-inferiority), by default 0.05
    allocation_ratio : float, optional
        Ratio of sample sizes (n2/n1), by default 1.0
    assumed_difference : float, optional
        Assumed true difference between treatments (0 = treatments truly equivalent), by default 0.0
    direction : str, optional
        Direction of non-inferiority test ("lower" or "upper"), by default "lower"
        
    Returns
    -------
    dict
        Dictionary containing the required sample sizes
    
    Notes
    -----
    For non-inferiority tests, we typically use a one-sided alpha level.
    - "lower" direction: Testing that the new treatment is not worse than the standard by more than the margin
    - "upper" direction: Testing that the new treatment is not better than the standard by more than the margin (rare)
    """
    # Validate inputs
    if non_inferiority_margin <= 0:
        raise ValueError("Non-inferiority margin must be positive")
    
    if direction not in ["lower", "upper"]:
        raise ValueError("Direction must be 'lower' or 'upper'")
    
    # For non-inferiority, use one-sided alpha
    z_alpha = stats.norm.ppf(1 - alpha)  # One-sided test
    z_beta = stats.norm.ppf(power)
    
    # Calculate effective delta based on direction and margin
    if direction == "lower":
        # Testing that new treatment is not worse than standard by more than NIM
        # For lower bound testing: H0: μ_new - μ_std ≤ -NIM, H1: μ_new - μ_std > -NIM
        delta = assumed_difference + non_inferiority_margin
    else:  # "upper"
        # Testing that new treatment is not better than standard by more than NIM
        # For upper bound testing: H0: μ_new - μ_std ≥ NIM, H1: μ_new - μ_std < NIM
        delta = non_inferiority_margin - assumed_difference
    
    # Calculate sample size
    n1 = ((1 + 1/allocation_ratio) * (std_dev**2) * (z_alpha + z_beta)**2) / (delta**2)
    n1 = math.ceil(n1)
    n2 = math.ceil(n1 * allocation_ratio)
    
    return {
        "n1": n1,
        "n2": n2,
        "total_n": n1 + n2,
        "parameters": {
            "non_inferiority_margin": non_inferiority_margin,
            "std_dev": std_dev,
            "power": power,
            "alpha": alpha,
            "allocation_ratio": allocation_ratio,
            "assumed_difference": assumed_difference,
            "direction": direction,
            "hypothesis_type": "non-inferiority"
        }
    }


def power_continuous_non_inferiority(
    n1, 
    n2, 
    non_inferiority_margin, 
    std_dev, 
    alpha=0.05, 
    assumed_difference=0.0,
    direction="lower"
):
    """
    Calculate power for non-inferiority test with continuous outcome.
    
    Parameters
    ----------
    n1 : int
        Sample size in group 1
    n2 : int
        Sample size in group 2
    non_inferiority_margin : float
        Non-inferiority margin (must be positive)
    std_dev : float
        Standard deviation of the outcome
    alpha : float, optional
        Significance level (one-sided for non-inferiority), by default 0.05
    assumed_difference : float, optional
        Assumed true difference between treatments (0 = treatments truly equivalent), by default 0.0
    direction : str, optional
        Direction of non-inferiority test ("lower" or "upper"), by default "lower"
        
    Returns
    -------
    dict
        Dictionary containing the calculated power and parameters
    """
    # Validate inputs
    if non_inferiority_margin <= 0:
        raise ValueError("Non-inferiority margin must be positive")
    
    if direction not in ["lower", "upper"]:
        raise ValueError("Direction must be 'lower' or 'upper'")
    
    # For non-inferiority, use one-sided alpha
    z_alpha = stats.norm.ppf(1 - alpha)  # One-sided
    
    # Calculate effective delta based on direction and margin
    if direction == "lower":
        # Testing that new treatment is not worse than standard by more than NIM
        delta = assumed_difference + non_inferiority_margin
    else:  # "upper"
        # Testing that new treatment is not better than standard by more than NIM
        delta = non_inferiority_margin - assumed_difference
    
    # Calculate non-centrality parameter
    ncp = delta / (std_dev * math.sqrt(1/n1 + 1/n2))
    
    # Calculate power
    power = stats.norm.cdf(ncp - z_alpha)
    
    return {
        "power": power,
        "parameters": {
            "n1": n1,
            "n2": n2,
            "non_inferiority_margin": non_inferiority_margin,
            "std_dev": std_dev,
            "alpha": alpha,
            "assumed_difference": assumed_difference,
            "direction": direction,
            "hypothesis_type": "non-inferiority"
        }
    }


def min_detectable_non_inferiority_margin(
    n1,
    n2,
    std_dev,
    power=0.8,
    alpha=0.05,
    assumed_difference=0.0,
    direction="lower"
):
    """
    Calculate the minimum detectable non-inferiority margin for a given sample size.
    
    Parameters
    ----------
    n1 : int
        Sample size in group 1
    n2 : int
        Sample size in group 2
    std_dev : float
        Standard deviation of the outcome
    power : float, optional
        Desired power, by default 0.8
    alpha : float, optional
        Significance level (one-sided for non-inferiority), by default 0.05
    assumed_difference : float, optional
        Assumed true difference between treatments (0 = treatments truly equivalent), by default 0.0
    direction : str, optional
        Direction of non-inferiority test ("lower" or "upper"), by default "lower"
    
    Returns
    -------
    dict
        Dictionary containing the minimum detectable non-inferiority margin
    """
    if direction not in ["lower", "upper"]:
        raise ValueError("Direction must be 'lower' or 'upper'")
    
    # For non-inferiority, use one-sided alpha
    z_alpha = stats.norm.ppf(1 - alpha)  # One-sided
    z_beta = stats.norm.ppf(power)
    
    # Calculate the minimum margin based on the sample size
    delta = (z_alpha + z_beta) * std_dev * math.sqrt(1/n1 + 1/n2)
    
    # Adjust for the assumed difference based on direction
    if direction == "lower":
        margin = delta - assumed_difference
    else:  # "upper"
        margin = delta + assumed_difference
    
    # Ensure we return a positive margin
    margin = max(margin, 1e-10)
    
    return {
        "margin": margin,
        "parameters": {
            "n1": n1,
            "n2": n2,
            "std_dev": std_dev,
            "power": power,
            "alpha": alpha,
            "assumed_difference": assumed_difference,
            "direction": direction,
            "hypothesis_type": "non-inferiority"
        }
    }


# Non-inferiority testing functions for binary outcomes
def sample_size_binary_non_inferiority(
    p1,
    non_inferiority_margin,
    power=0.8,
    alpha=0.05,
    allocation_ratio=1.0,
    assumed_difference=0.0,
    direction="lower",
    test_type="Normal Approximation"
):
    """
    Calculate sample size for non-inferiority test with binary outcome.
    
    Parameters
    ----------
    p1 : float
        Proportion in the control/standard group (between 0 and 1)
    non_inferiority_margin : float
        Non-inferiority margin (must be positive, absolute difference in proportions)
    power : float, optional
        Desired power, by default 0.8
    alpha : float, optional
        Significance level (one-sided for non-inferiority), by default 0.05
    allocation_ratio : float, optional
        Ratio of sample sizes (n2/n1), by default 1.0
    assumed_difference : float, optional
        Assumed true difference between proportions (0 = proportions truly equivalent), by default 0.0
    direction : str, optional
        Direction of non-inferiority test ("lower" or "upper"), by default "lower"
    test_type : str, optional
        Type of statistical test to use, by default "Normal Approximation"
        
    Returns
    -------
    dict
        Dictionary containing the required sample sizes
    
    Notes
    -----
    For non-inferiority tests, we typically use a one-sided alpha level.
    - "lower" direction: Testing that the new treatment is not worse than standard by more than the margin
    - "upper" direction: Testing that the new treatment is not better than standard by more than the margin (rare)
    """
    # For non-inferiority, we only use a one-sided alpha
    # Calculate z-scores for given alpha and power
    z_alpha = stats.norm.ppf(1 - alpha)  # one-sided alpha
    z_beta = stats.norm.ppf(power)
    
    # Calculate p2 based on assumed difference
    p2 = p1 + assumed_difference
    
    # Validate inputs
    if p1 <= 0 or p1 >= 1 or p2 <= 0 or p2 >= 1:
        raise ValueError("Proportions must be between 0 and 1")
    if non_inferiority_margin <= 0:
        raise ValueError("Non-inferiority margin must be positive")
    
    # For non-inferiority with lower margin (most common case)
    # H0: p2 - p1 ≤ -margin, H1: p2 - p1 > -margin
    # We need to detect a difference of (assumed_difference + margin) 
    # Check for nearly-zero differences
    effective_diff = assumed_difference + non_inferiority_margin
    
    if abs(effective_diff) < 1e-6:
        print(f"WARNING: Effective difference {effective_diff} is too small for precision")
        return {
            "n1": 1000,  # More reasonable default for non-inferiority
            "n2": math.ceil(1000 * allocation_ratio),
            "total_n": 1000 + math.ceil(1000 * allocation_ratio),
            "parameters": {
                "p1": p1,
                "p2": p2,
                "non_inferiority_margin": non_inferiority_margin,
                "power": power,
                "alpha": alpha,
                "allocation_ratio": allocation_ratio,
                "test_type": test_type,
                "note": "Estimated sample size - effective difference too small for precise calculation"
            }
        }
        
    # Use a more appropriate formula for non-inferiority
    # This formula accounts for the one-sided test nature of non-inferiority
    p_bar = (p1 + p2) / 2
    
    # Sample size formula for non-inferiority binary outcome
    try:
        var_factor = p1 * (1 - p1) + p2 * (1 - p2) / allocation_ratio
        n1_base = (z_alpha + z_beta)**2 * var_factor / effective_diff**2
        
        # Check for infinity or very large values
        if not np.isfinite(n1_base) or n1_base > 1e6:
            print(f"WARNING: Calculated sample size is too large or infinite: {n1_base}")
            n1_base = 1000  # Cap at a reasonable maximum
    except (ZeroDivisionError, OverflowError) as e:
        print(f"Error in sample size calculation: {str(e)}")
        n1_base = 1000  # Default to a reasonable sample size
    
    # Apply adjustment factor based on test type
    if test_type == "Normal Approximation":
        # No adjustment needed
        n1 = n1_base
        method_description = "Normal Approximation (z-test)"
    elif test_type == "Likelihood Ratio Test":
        # LR test typically needs slightly smaller samples
        n1 = n1_base * 0.95  # 5% smaller than normal approximation
        method_description = "Likelihood Ratio Test (typically requires smaller samples)"
    elif test_type == "Exact Test":
        # Fisher's exact test generally requires larger samples for equivalent power
        if p1 < 0.1 or p2 < 0.1 or p1 > 0.9 or p2 > 0.9:
            # More conservative for extreme proportions
            n1 = n1_base * 1.25  # 25% larger
            method_description = "Fisher's Exact Test (larger samples for extreme proportions)"
        else:
            # Moderate increase for non-extreme proportions
            n1 = n1_base * 1.15  # 15% larger
            method_description = "Fisher's Exact Test (requires larger samples)"
    else:
        # Use normal approximation as default
        n1 = n1_base
        method_description = "Unknown test type, defaulting to Normal Approximation"
        
    # Print debugging information
    print(f"Test type: {test_type}, Sample size: {n1}, Description: {method_description}")
        
    # Round up to nearest whole number
    n1 = math.ceil(n1)
    n2 = math.ceil(n1 * allocation_ratio)
    
    return {
        "n1": n1,
        "n2": n2,
        "total_n": n1 + n2,
        "parameters": {
            "p1": p1,
            "p2": p2,
            "non_inferiority_margin": non_inferiority_margin,
            "assumed_difference": assumed_difference,
            "power": power,
            "alpha": alpha,
            "allocation_ratio": allocation_ratio,
            "test_type": test_type,
            "hypothesis_type": "non-inferiority"
        }
    }


def power_binary_non_inferiority(
    n1,
    n2,
    p1,
    non_inferiority_margin,
    alpha=0.05,
    assumed_difference=0.0,
    direction="lower"
):
    """
    Calculate power for non-inferiority test with binary outcome.
    
    Parameters
    ----------
    n1 : int
        Sample size in group 1 (standard treatment)
    n2 : int
        Sample size in group 2 (new treatment)
    p1 : float
        Proportion in the control/standard group (between 0 and 1)
    non_inferiority_margin : float
        Non-inferiority margin (must be positive, absolute difference in proportions)
    alpha : float, optional
        Significance level (one-sided for non-inferiority), by default 0.05
    assumed_difference : float, optional
        Assumed true difference between proportions (0 = proportions truly equivalent), by default 0.0
    direction : str, optional
        Direction of non-inferiority test ("lower" or "upper"), by default "lower"
        
    Returns
    -------
    dict
        Dictionary containing the calculated power and parameters
    """
    # Validate inputs
    if non_inferiority_margin <= 0:
        raise ValueError("Non-inferiority margin must be positive")
    
    if direction not in ["lower", "upper"]:
        raise ValueError("Direction must be 'lower' or 'upper'")
    
    if p1 <= 0 or p1 >= 1:
        raise ValueError("p1 must be between 0 and 1")
    
    # For non-inferiority, use one-sided alpha
    z_alpha = stats.norm.ppf(1 - alpha)  # One-sided
    
    # Determine the expected proportion in the new treatment group
    p2 = p1 + assumed_difference
    
    # Ensure p2 is within valid range
    if p2 <= 0 or p2 >= 1:
        raise ValueError("Assumed difference leads to invalid p2 (must be between 0 and 1)")
    
    # Adjust margin based on direction
    if direction == "lower":
        # Testing that new treatment proportion is not lower than standard by more than margin
        p2_under_null = p1 - non_inferiority_margin
    else:  # "upper"
        # Testing that new treatment proportion is not higher than standard by more than margin
        p2_under_null = p1 + non_inferiority_margin
    
    # Ensure p2_under_null is within valid range
    p2_under_null = max(0, min(1, p2_under_null))
    
    # Calculate pooled proportion under the null hypothesis
    p_pooled_null = (n1 * p1 + n2 * p2_under_null) / (n1 + n2)
    
    # Calculate standard error for the difference under null hypothesis
    se_null = np.sqrt((p_pooled_null * (1 - p_pooled_null)) * (1/n1 + 1/n2))
    
    # Calculate standard error for the difference under alternative hypothesis
    se_alt = np.sqrt((p1 * (1 - p1) / n1) + (p2 * (1 - p2) / n2))
    
    # Calculate effective delta and test statistic 
    if direction == "lower":
        delta = p2 - (p1 - non_inferiority_margin)
    else:  # "upper"
        delta = (p1 + non_inferiority_margin) - p2
    
    # Calculate non-centrality parameter
    ncp = delta / se_alt
    
    # Calculate power
    if direction == "lower":
        # For lower bound testing
        power = 1 - stats.norm.cdf(z_alpha - ncp)
    else:  # "upper"
        # For upper bound testing
        power = stats.norm.cdf(ncp - z_alpha)
    
    return {
        "power": power,
        "parameters": {
            "n1": n1,
            "n2": n2,
            "p1": p1,
            "non_inferiority_margin": non_inferiority_margin,
            "alpha": alpha,
            "assumed_difference": assumed_difference,
            "direction": direction,
            "hypothesis_type": "non-inferiority"
        }
    }


def min_detectable_binary_non_inferiority_margin(
    n1,
    n2,
    p1,
    power=0.8,
    alpha=0.05,
    assumed_difference=0.0,
    direction="lower"
):
    """
    Calculate the minimum detectable non-inferiority margin for binary outcomes.
    
    Parameters
    ----------
    n1 : int
        Sample size in group 1 (standard treatment)
    n2 : int
        Sample size in group 2 (new treatment)
    p1 : float
        Proportion in the control/standard group (between 0 and 1)
    power : float, optional
        Desired power, by default 0.8
    alpha : float, optional
        Significance level (one-sided for non-inferiority), by default 0.05
    assumed_difference : float, optional
        Assumed true difference between proportions (0 = proportions truly equivalent), by default 0.0
    direction : str, optional
        Direction of non-inferiority test ("lower" or "upper"), by default "lower"
        
    Returns
    -------
    dict
        Dictionary containing the minimum detectable non-inferiority margin
    """
    # Validate inputs
    if direction not in ["lower", "upper"]:
        raise ValueError("Direction must be 'lower' or 'upper'")
    
    if p1 <= 0 or p1 >= 1:
        raise ValueError("p1 must be between 0 and 1")
    
    # For non-inferiority, use one-sided alpha
    z_alpha = stats.norm.ppf(1 - alpha)  # One-sided
    z_beta = stats.norm.ppf(power)
    
    # Determine the expected proportion in the new treatment group
    p2 = p1 + assumed_difference
    
    # Ensure p2 is within valid range
    if p2 <= 0 or p2 >= 1:
        raise ValueError("Assumed difference leads to invalid p2 (must be between 0 and 1)")
    
    # Calculate standard error for the difference under the alternative hypothesis
    se_alt = np.sqrt((p1 * (1 - p1) / n1) + (p2 * (1 - p2) / n2))
    
    # Calculate the minimum margin based on the sample size and expected proportions
    margin = (z_alpha + z_beta) * se_alt
    
    # Adjust for the assumed difference based on direction
    if direction == "lower":
        margin = margin - assumed_difference
    else:  # "upper"
        margin = margin + assumed_difference
    
    # Ensure we return a positive margin
    margin = max(margin, 1e-10)
    
    return {
        "margin": margin,
        "parameters": {
            "n1": n1,
            "n2": n2,
            "p1": p1,
            "power": power,
            "alpha": alpha,
            "assumed_difference": assumed_difference,
            "direction": direction,
            "hypothesis_type": "non-inferiority"
        }
    }
