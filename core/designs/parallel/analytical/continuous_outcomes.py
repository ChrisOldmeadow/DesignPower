"""
Analytical methods for continuous outcomes in parallel group RCTs.

This module provides functions for sample size, power calculation, and
minimum detectable effect estimation for continuous outcomes.
"""
import math
from scipy import stats
from typing import Union


def sample_size_continuous(delta, std_dev, power=0.8, alpha=0.05, allocation_ratio=1.0, std_dev2=None):
    """
    Calculate sample size required for detecting a difference in means between two groups.
    
    Parameters
    ----------
    delta : float
        Minimum detectable effect (difference between means)
    std_dev : float
        Pooled standard deviation of the outcome
    power : float, optional
        Desired statistical power (1 - beta), by default 0.8
    alpha : float, optional
        Significance level, by default 0.05
    allocation_ratio : float, optional
        Ratio of sample sizes (n2/n1), by default 1.0
    
    Returns
    -------
    dict
        Dictionary containing the required sample sizes for each group and total
    """
    # Calculate z-scores for given alpha and power
    z_alpha = stats.norm.ppf(1 - alpha/2)
    z_beta = stats.norm.ppf(power)
    
    # Calculate sample size for group 1
    if std_dev2 is not None:
        # Unequal variances formula
        n1 = ((std_dev**2 + std_dev2**2 / allocation_ratio) * (z_alpha + z_beta)**2) / (delta**2)
    else:
        # Equal variances formula
        n1 = ((1 + 1/allocation_ratio) * (std_dev**2) * (z_alpha + z_beta)**2) / (delta**2)
    
    n1 = math.ceil(n1)
    
    # Calculate sample size for group 2
    n2 = math.ceil(n1 * allocation_ratio)
    
    return {
        "n1": n1,
        "n2": n2,
        "total_n": n1 + n2,
        "parameters": {
            "delta": delta,
            "std_dev": std_dev,
            "std_dev2": std_dev2,
            "power": power,
            "alpha": alpha,
            "allocation_ratio": allocation_ratio
        }
    }


def power_continuous(n1, n2, delta, std_dev, alpha=0.05, std_dev2=None):
    """
    Calculate statistical power for detecting a difference in means with given sample sizes.
    
    Parameters
    ----------
    n1 : int
        Sample size for group 1
    n2 : int
        Sample size for group 2
    delta : float
        Minimum detectable effect (difference between means)
    std_dev : float
        Pooled standard deviation of the outcome
    alpha : float, optional
        Significance level, by default 0.05
    
    Returns
    -------
    dict
        Dictionary containing the calculated power and input parameters
    """
    # Calculate z-score for given alpha
    z_alpha = stats.norm.ppf(1 - alpha/2)
    
    # Handle unequal standard deviations if specified
    if std_dev2 is not None:
        # Welch-Satterthwaite approximation for unequal variances
        ncp = delta / math.sqrt((std_dev**2 / n1) + (std_dev2**2 / n2))
    else:
        # Equal variances (pooled standard deviation)
        ncp = delta / (std_dev * math.sqrt(1/n1 + 1/n2))
    
    # Calculate power
    power = stats.norm.cdf(ncp - z_alpha)
    
    return {
        "power": power,
        "parameters": {
            "n1": n1,
            "n2": n2,
            "delta": delta,
            "std_dev": std_dev,
            "std_dev2": std_dev2,
            "alpha": alpha
        }
    }
