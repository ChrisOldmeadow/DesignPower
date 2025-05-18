"""
Utilities for continuous outcomes in sample size and power calculation.

This module provides common functions and utilities for working with
continuous outcomes across different study designs.
"""
import numpy as np
from scipy import stats


def standardized_effect_size(mean1, mean2, std_dev):
    """
    Calculate standardized effect size (Cohen's d).
    
    Parameters
    ----------
    mean1 : float
        Mean of first group or time period
    mean2 : float
        Mean of second group or time period
    std_dev : float
        Pooled standard deviation
    
    Returns
    -------
    float
        Cohen's d standardized effect size
    """
    return abs(mean1 - mean2) / std_dev


def confidence_interval(mean, std_dev, n, alpha=0.05):
    """
    Calculate confidence interval for a mean.
    
    Parameters
    ----------
    mean : float
        Sample mean
    std_dev : float
        Sample standard deviation
    n : int
        Sample size
    alpha : float, optional
        Significance level, by default 0.05
    
    Returns
    -------
    tuple
        Lower and upper bounds of the confidence interval
    """
    # Standard error of the mean
    se = std_dev / np.sqrt(n)
    
    # Critical value for the given alpha
    t_crit = stats.t.ppf(1 - alpha/2, n - 1)
    
    # Lower and upper bounds
    lower = mean - t_crit * se
    upper = mean + t_crit * se
    
    return (lower, upper)


def sample_size_precision(std_dev, width, alpha=0.05):
    """
    Calculate sample size required for desired precision of a mean.
    
    Parameters
    ----------
    std_dev : float
        Standard deviation
    width : float
        Desired total width of the confidence interval
    alpha : float, optional
        Significance level, by default 0.05
    
    Returns
    -------
    int
        Required sample size
    """
    # Critical value for the given alpha
    z_crit = stats.norm.ppf(1 - alpha/2)
    
    # Calculate sample size
    n = ((2 * z_crit * std_dev) / width) ** 2
    
    return int(np.ceil(n))


def effect_size_from_required_precision(std_dev, n, alpha=0.05):
    """
    Calculate the minimum effect size that can be detected with a given sample size and precision.
    
    Parameters
    ----------
    std_dev : float
        Standard deviation
    n : int
        Sample size
    alpha : float, optional
        Significance level, by default 0.05
    
    Returns
    -------
    float
        Minimum detectable effect size (half-width of confidence interval)
    """
    # Critical value for the given alpha
    z_crit = stats.norm.ppf(1 - alpha/2)
    
    # Standard error
    se = std_dev / np.sqrt(n)
    
    # Half-width of confidence interval
    half_width = z_crit * se
    
    return half_width
