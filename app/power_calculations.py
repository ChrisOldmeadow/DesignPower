"""
Power calculation functions for single-arm studies.

This module provides functions for sample size and power calculations
for single-arm studies with continuous and binary outcomes.
"""
import math
import numpy as np
from scipy import stats


def one_sample_t_test_sample_size(mean_null, mean_alt, std_dev, alpha=0.05, power=0.8, sides=2):
    """
    Calculate sample size for one-sample t-test (continuous outcome).
    
    Parameters
    ----------
    mean_null : float
        Null hypothesis mean
    mean_alt : float
        Alternative hypothesis mean
    std_dev : float
        Standard deviation
    alpha : float, optional
        Significance level, by default 0.05
    power : float, optional
        Desired power (1 - beta), by default 0.8
    sides : int, optional
        One-sided or two-sided test (1 or 2), by default 2
    
    Returns
    -------
    int
        Required sample size (rounded up)
    """
    # Effect size (standardized mean difference)
    effect_size = abs(mean_alt - mean_null) / std_dev
    
    # Calculate t-test critical values based on sides
    if sides == 1:
        t_alpha = stats.t.ppf(1 - alpha, np.inf)
    else:  # sides == 2
        t_alpha = stats.t.ppf(1 - alpha/2, np.inf)
        
    t_beta = stats.t.ppf(power, np.inf)
    
    # Calculate sample size
    n = ((t_alpha + t_beta) / effect_size) ** 2
    
    # Round up to nearest whole number
    return math.ceil(n)


def one_sample_proportion_test_sample_size(p0, p1, alpha=0.05, power=0.8, sides=2):
    """
    Calculate sample size for one-sample proportion test (binary outcome).
    
    Parameters
    ----------
    p0 : float
        Null hypothesis proportion (between 0 and 1)
    p1 : float
        Alternative hypothesis proportion (between 0 and 1)
    alpha : float, optional
        Significance level, by default 0.05
    power : float, optional
        Desired power (1 - beta), by default 0.8
    sides : int, optional
        One-sided or two-sided test (1 or 2), by default 2
    
    Returns
    -------
    int
        Required sample size (rounded up)
    """
    # Calculate z-scores based on sides
    if sides == 1:
        z_alpha = stats.norm.ppf(1 - alpha)
    else:  # sides == 2
        z_alpha = stats.norm.ppf(1 - alpha/2)
    
    z_beta = stats.norm.ppf(power)
    
    # Calculate sample size using normal approximation
    term1 = z_alpha * math.sqrt(p0 * (1 - p0))
    term2 = z_beta * math.sqrt(p1 * (1 - p1))
    denominator = p1 - p0
    
    # Handle small differences to prevent division by zero
    if abs(denominator) < 1e-10:
        return 10000  # Default to large sample size for very small differences
    
    n = ((term1 + term2) / denominator) ** 2
    
    # Round up to nearest whole number
    return math.ceil(n)


def one_sample_proportion_test_power(n, p0, p1, alpha=0.05, sides=2):
    """
    Calculate power for one-sample proportion test.
    
    Parameters
    ----------
    n : int
        Sample size
    p0 : float
        Null hypothesis proportion
    p1 : float
        Alternative hypothesis proportion
    alpha : float, optional
        Significance level, by default 0.05
    sides : int, optional
        One-sided or two-sided test (1 or 2), by default 2
    
    Returns
    -------
    float
        Power (1 - beta)
    """
    # Calculate z-score for alpha
    if sides == 1:
        z_alpha = stats.norm.ppf(1 - alpha)
    else:  # sides == 2
        z_alpha = stats.norm.ppf(1 - alpha/2)
    
    # Calculate non-centrality parameter
    denominator = math.sqrt(p0 * (1 - p0) / n)
    
    # Handle potential division by zero
    if denominator < 1e-10:
        return 0.999  # Default to high power when denominator is very small
    
    ncp = (p1 - p0) / denominator
    
    # Calculate power
    power = 1 - stats.norm.cdf(z_alpha - ncp)
    
    return power


def min_detectable_effect_one_sample_continuous(n, mean_null=0, std_dev=1, power=0.8, alpha=0.05, sides=2):
    """
    Calculate minimum detectable effect for one-sample t-test.
    
    Parameters
    ----------
    n : int
        Sample size
    mean_null : float, optional
        Null hypothesis mean, by default 0
    std_dev : float, optional
        Standard deviation, by default 1
    power : float, optional
        Desired power (1 - beta), by default 0.8
    alpha : float, optional
        Significance level, by default 0.05
    sides : int, optional
        One-sided or two-sided test (1 or 2), by default 2
    
    Returns
    -------
    float
        Minimum detectable effect (standardized mean difference)
    """
    # Calculate critical values
    if sides == 1:
        t_alpha = stats.t.ppf(1 - alpha, n - 1)
    else:  # sides == 2
        t_alpha = stats.t.ppf(1 - alpha/2, n - 1)
    
    t_beta = stats.t.ppf(power, n - 1)
    
    # Calculate MDE
    mde = (t_alpha + t_beta) * std_dev / math.sqrt(n)
    
    # Return MDE as absolute mean difference
    return mde
