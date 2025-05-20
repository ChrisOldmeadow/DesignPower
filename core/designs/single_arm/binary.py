"""
Binary outcome functions for single-arm (one-sample) designs.

This module provides functions for power analysis and sample size calculation
for single-arm studies with binary outcomes.
"""

import math
import numpy as np
from scipy import stats


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


def min_detectable_effect_one_sample_binary(n, p0=0.5, power=0.8, alpha=0.05, sides=2):
    """
    Calculate minimum detectable effect for one-sample proportion test.
    
    Parameters
    ----------
    n : int
        Sample size
    p0 : float, optional
        Null hypothesis proportion, by default 0.5
    power : float, optional
        Desired power (1 - beta), by default 0.8
    alpha : float, optional
        Significance level, by default 0.05
    sides : int, optional
        One-sided or two-sided test (1 or 2), by default 2
    
    Returns
    -------
    float
        Minimum detectable difference in proportions
    """
    # Calculate critical values
    if sides == 1:
        z_alpha = stats.norm.ppf(1 - alpha)
    else:  # sides == 2
        z_alpha = stats.norm.ppf(1 - alpha/2)
    
    z_beta = stats.norm.ppf(power)
    
    # Calculate standard error for null proportion
    se = math.sqrt(p0 * (1 - p0) / n)
    
    # Calculate minimum detectable effect (absolute difference)
    mde = (z_alpha + z_beta) * se
    
    return mde


def simulate_one_sample_binary_trial(n, p0, p1, nsim=1000, alpha=0.05, sides=2, seed=None):
    """
    Simulate a single-arm study with binary outcome.
    
    Parameters
    ----------
    n : int
        Sample size
    p0 : float
        Null hypothesis proportion
    p1 : float
        Alternative hypothesis proportion
    nsim : int, optional
        Number of simulations, by default 1000
    alpha : float, optional
        Significance level, by default 0.05
    sides : int, optional
        One-sided or two-sided test (1 or 2), by default 2
    seed : int, optional
        Random seed for reproducibility, by default None
    
    Returns
    -------
    dict
        Dictionary containing simulation results, including empirical power
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Initialize counters
    significant_results = 0
    
    # Run simulations
    for _ in range(nsim):
        # Generate data
        data = np.random.binomial(1, p1, n)
        p_hat = np.mean(data)
        
        # Calculate standard error under null hypothesis
        se = math.sqrt(p0 * (1 - p0) / n)
        
        # Calculate test statistic
        z = (p_hat - p0) / se
        
        # Calculate p-value based on sides
        if sides == 1:
            # One-sided test
            if p1 > p0:  # Upper-tailed
                p_value = 1 - stats.norm.cdf(z)
            else:  # Lower-tailed
                p_value = stats.norm.cdf(z)
        else:  # sides == 2
            # Two-sided test
            p_value = 2 * (1 - stats.norm.cdf(abs(z)))
        
        # Count significant results
        if p_value < alpha:
            significant_results += 1
    
    # Calculate empirical power
    power = significant_results / nsim
    
    return {
        "power": power,
        "significant_results": significant_results,
        "nsim": nsim,
        "n": n,
        "p0": p0,
        "p1": p1,
        "sides": sides
    }
