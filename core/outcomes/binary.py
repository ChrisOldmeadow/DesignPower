"""
Utilities for binary outcomes in sample size and power calculation.

This module provides common functions and utilities for working with
binary outcomes across different study designs.
"""
import numpy as np
import math
from scipy import stats


def effect_size_to_odds_ratio(p1, p2):
    """
    Convert proportions to odds ratio.
    
    Parameters
    ----------
    p1 : float
        Proportion in first group (between 0 and 1)
    p2 : float
        Proportion in second group (between 0 and 1)
    
    Returns
    -------
    float
        Odds ratio
    """
    odds1 = p1 / (1 - p1)
    odds2 = p2 / (1 - p2)
    return odds2 / odds1


def odds_ratio_to_proportions(p1, odds_ratio):
    """
    Convert odds ratio to second proportion, given the first proportion.
    
    Parameters
    ----------
    p1 : float
        Proportion in first group (between 0 and 1)
    odds_ratio : float
        Odds ratio
    
    Returns
    -------
    float
        Proportion in second group
    """
    odds1 = p1 / (1 - p1)
    odds2 = odds1 * odds_ratio
    p2 = odds2 / (1 + odds2)
    return p2


def confidence_interval_proportion(p, n, alpha=0.05, method='wilson'):
    """
    Calculate confidence interval for a proportion.
    
    Parameters
    ----------
    p : float
        Sample proportion
    n : int
        Sample size
    alpha : float, optional
        Significance level, by default 0.05
    method : str, optional
        Method to use. Options are 'wilson', 'wald', 'agresti-coull', by default 'wilson'
    
    Returns
    -------
    tuple
        Lower and upper bounds of the confidence interval
    """
    # Critical value for the given alpha
    z_crit = stats.norm.ppf(1 - alpha/2)
    
    if method == 'wilson':
        # Wilson score interval
        center = (p + z_crit**2 / (2 * n)) / (1 + z_crit**2 / n)
        halfwidth = z_crit * math.sqrt(p * (1 - p) / n + z_crit**2 / (4 * n**2)) / (1 + z_crit**2 / n)
        lower = max(0, center - halfwidth)
        upper = min(1, center + halfwidth)
    elif method == 'wald':
        # Wald interval
        halfwidth = z_crit * math.sqrt(p * (1 - p) / n)
        lower = max(0, p - halfwidth)
        upper = min(1, p + halfwidth)
    elif method == 'agresti-coull':
        # Agresti-Coull interval
        n_tilde = n + z_crit**2
        p_tilde = (n * p + z_crit**2 / 2) / n_tilde
        halfwidth = z_crit * math.sqrt(p_tilde * (1 - p_tilde) / n_tilde)
        lower = max(0, p_tilde - halfwidth)
        upper = min(1, p_tilde + halfwidth)
    else:
        raise ValueError(f"Method {method} not supported")
    
    return (lower, upper)


def sample_size_precision_proportion(p, width, alpha=0.05):
    """
    Calculate sample size required for desired precision of a proportion.
    
    Parameters
    ----------
    p : float
        Expected proportion
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
    n = (4 * z_crit**2 * p * (1 - p)) / width**2
    
    return int(np.ceil(n))


def risk_ratio(p1, p2):
    """
    Calculate risk ratio (relative risk) from two proportions.
    
    Parameters
    ----------
    p1 : float
        Proportion in first group (between 0 and 1)
    p2 : float
        Proportion in second group (between 0 and 1)
    
    Returns
    -------
    float
        Risk ratio (p2/p1)
    """
    return p2 / p1


def risk_difference(p1, p2):
    """
    Calculate risk difference (absolute risk reduction) from two proportions.
    
    Parameters
    ----------
    p1 : float
        Proportion in first group (between 0 and 1)
    p2 : float
        Proportion in second group (between 0 and 1)
    
    Returns
    -------
    float
        Risk difference (p2-p1)
    """
    return p2 - p1


def number_needed_to_treat(p1, p2):
    """
    Calculate number needed to treat (NNT) from two proportions.
    
    Parameters
    ----------
    p1 : float
        Proportion of events in control group (between 0 and 1)
    p2 : float
        Proportion of events in treatment group (between 0 and 1)
    
    Returns
    -------
    float
        Number needed to treat
    """
    # Check if the treatment has a beneficial effect
    if p2 >= p1:
        # Treatment is not beneficial or has no effect
        return float('inf')
    
    # Calculate NNT
    absolute_risk_reduction = p1 - p2
    return 1 / absolute_risk_reduction
