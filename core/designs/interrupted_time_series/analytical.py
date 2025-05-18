"""
Analytical methods for interrupted time series designs.

This module provides functions for sample size, power calculation, and
minimum detectable effect estimation for interrupted time series designs
using analytical formulas.
"""
import math
import numpy as np
from scipy import stats


def power_continuous(n_pre, n_post, mean_change, std_dev, alpha=0.05, autocorr=0.0):
    """
    Calculate power for detecting a change in level in an interrupted time series with continuous outcome.
    
    Parameters
    ----------
    n_pre : int
        Number of time points pre-intervention
    n_post : int
        Number of time points post-intervention
    mean_change : float
        Expected change in level (difference in means)
    std_dev : float
        Standard deviation of the outcome
    alpha : float, optional
        Significance level, by default 0.05
    autocorr : float, optional
        Autocorrelation coefficient, by default 0.0
    
    Returns
    -------
    dict
        Dictionary containing the calculated power and input parameters
    """
    # Adjusting for autocorrelation
    if autocorr != 0:
        # Calculate effective sample size
        n_pre_eff = n_pre * (1 - autocorr) / (1 + autocorr)
        n_post_eff = n_post * (1 - autocorr) / (1 + autocorr)
    else:
        n_pre_eff = n_pre
        n_post_eff = n_post
    
    # Calculate standard error for the difference
    se = std_dev * math.sqrt(1/n_pre_eff + 1/n_post_eff)
    
    # Calculate z-scores for given alpha
    z_alpha = stats.norm.ppf(1 - alpha/2)
    
    # Calculate non-centrality parameter
    ncp = abs(mean_change) / se
    
    # Calculate power
    power = stats.norm.cdf(ncp - z_alpha)
    
    return {
        "power": power,
        "parameters": {
            "n_pre": n_pre,
            "n_post": n_post,
            "mean_change": mean_change,
            "std_dev": std_dev,
            "alpha": alpha,
            "autocorr": autocorr
        }
    }


def sample_size_continuous(mean_change, std_dev, power=0.8, alpha=0.05, autocorr=0.0, ratio=1.0):
    """
    Calculate required number of time points for an interrupted time series with continuous outcome.
    
    Parameters
    ----------
    mean_change : float
        Expected change in level (difference in means)
    std_dev : float
        Standard deviation of the outcome
    power : float, optional
        Desired statistical power (1 - beta), by default 0.8
    alpha : float, optional
        Significance level, by default 0.05
    autocorr : float, optional
        Autocorrelation coefficient, by default 0.0
    ratio : float, optional
        Ratio of post to pre time points (post/pre), by default 1.0
    
    Returns
    -------
    dict
        Dictionary containing the required number of time points and input parameters
    """
    # Calculate z-scores for given alpha and power
    z_alpha = stats.norm.ppf(1 - alpha/2)
    z_beta = stats.norm.ppf(power)
    
    # Adjustment factor for autocorrelation
    if autocorr != 0:
        autocorr_factor = (1 + autocorr) / (1 - autocorr)
    else:
        autocorr_factor = 1.0
    
    # Calculate n_pre
    n_pre = autocorr_factor * (std_dev**2) * (z_alpha + z_beta)**2 * (1 + 1/ratio) / (mean_change**2)
    n_pre = math.ceil(n_pre)
    
    # Calculate n_post
    n_post = math.ceil(n_pre * ratio)
    
    return {
        "n_pre": n_pre,
        "n_post": n_post,
        "total_n": n_pre + n_post,
        "parameters": {
            "mean_change": mean_change,
            "std_dev": std_dev,
            "power": power,
            "alpha": alpha,
            "autocorr": autocorr,
            "ratio": ratio
        }
    }


def power_binary(n_pre, n_post, p_pre, p_post, alpha=0.05, autocorr=0.0):
    """
    Calculate power for detecting a change in proportion in an interrupted time series with binary outcome.
    
    Parameters
    ----------
    n_pre : int
        Number of observations pre-intervention
    n_post : int
        Number of observations post-intervention
    p_pre : float
        Proportion pre-intervention
    p_post : float
        Proportion post-intervention
    alpha : float, optional
        Significance level, by default 0.05
    autocorr : float, optional
        Autocorrelation coefficient, by default 0.0
    
    Returns
    -------
    dict
        Dictionary containing the calculated power and input parameters
    """
    # Adjusting for autocorrelation
    if autocorr != 0:
        # Calculate effective sample size
        n_pre_eff = n_pre * (1 - autocorr) / (1 + autocorr)
        n_post_eff = n_post * (1 - autocorr) / (1 + autocorr)
    else:
        n_pre_eff = n_pre
        n_post_eff = n_post
    
    # Calculate pooled proportion
    p_pooled = (n_pre * p_pre + n_post * p_post) / (n_pre + n_post)
    
    # Calculate standard error for the difference in proportions
    se = math.sqrt(p_pooled * (1 - p_pooled) * (1/n_pre_eff + 1/n_post_eff))
    
    # Calculate z-scores for given alpha
    z_alpha = stats.norm.ppf(1 - alpha/2)
    
    # Calculate non-centrality parameter
    ncp = abs(p_post - p_pre) / se
    
    # Calculate power
    power = stats.norm.cdf(ncp - z_alpha)
    
    return {
        "power": power,
        "parameters": {
            "n_pre": n_pre,
            "n_post": n_post,
            "p_pre": p_pre,
            "p_post": p_post,
            "alpha": alpha,
            "autocorr": autocorr
        }
    }


def sample_size_binary(p_pre, p_post, power=0.8, alpha=0.05, autocorr=0.0, ratio=1.0):
    """
    Calculate required number of observations for an interrupted time series with binary outcome.
    
    Parameters
    ----------
    p_pre : float
        Proportion pre-intervention
    p_post : float
        Proportion post-intervention
    power : float, optional
        Desired statistical power (1 - beta), by default 0.8
    alpha : float, optional
        Significance level, by default 0.05
    autocorr : float, optional
        Autocorrelation coefficient, by default 0.0
    ratio : float, optional
        Ratio of post to pre observations (post/pre), by default 1.0
    
    Returns
    -------
    dict
        Dictionary containing the required number of observations and input parameters
    """
    # Calculate z-scores for given alpha and power
    z_alpha = stats.norm.ppf(1 - alpha/2)
    z_beta = stats.norm.ppf(power)
    
    # Calculate pooled proportion
    p_pooled = (p_pre + p_post) / 2
    
    # Adjustment factor for autocorrelation
    if autocorr != 0:
        autocorr_factor = (1 + autocorr) / (1 - autocorr)
    else:
        autocorr_factor = 1.0
    
    # Calculate n_pre
    n_pre = autocorr_factor * p_pooled * (1 - p_pooled) * (z_alpha + z_beta)**2 * (1 + 1/ratio) / (p_post - p_pre)**2
    n_pre = math.ceil(n_pre)
    
    # Calculate n_post
    n_post = math.ceil(n_pre * ratio)
    
    return {
        "n_pre": n_pre,
        "n_post": n_post,
        "total_n": n_pre + n_post,
        "parameters": {
            "p_pre": p_pre,
            "p_post": p_post,
            "power": power,
            "alpha": alpha,
            "autocorr": autocorr,
            "ratio": ratio
        }
    }
