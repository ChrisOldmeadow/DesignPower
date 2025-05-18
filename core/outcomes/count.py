"""
Utilities for count outcomes in sample size and power calculation.

This module provides common functions and utilities for working with
count outcomes across different study designs.
"""
import numpy as np
import math
from scipy import stats


def confidence_interval_poisson(count, exposure=1.0, alpha=0.05):
    """
    Calculate confidence interval for a Poisson rate.
    
    Parameters
    ----------
    count : int
        Number of events
    exposure : float, optional
        Total exposure (e.g., person-years), by default 1.0
    alpha : float, optional
        Significance level, by default 0.05
    
    Returns
    -------
    tuple
        Lower and upper bounds of the confidence interval (rate per unit exposure)
    """
    # For 0 events, use special handling for lower bound
    if count == 0:
        lower = 0
    else:
        lower = stats.chi2.ppf(alpha/2, 2 * count) / (2 * exposure)
    
    # Upper bound
    upper = stats.chi2.ppf(1 - alpha/2, 2 * (count + 1)) / (2 * exposure)
    
    return (lower, upper)


def sample_size_precision_poisson(rate, width, alpha=0.05):
    """
    Calculate required exposure time for desired precision of a Poisson rate.
    
    Parameters
    ----------
    rate : float
        Expected rate of events per unit time
    width : float
        Desired total width of the confidence interval (in same units as rate)
    alpha : float, optional
        Significance level, by default 0.05
    
    Returns
    -------
    float
        Required exposure (e.g., person-years)
    """
    # Critical value for the given alpha
    z_crit = stats.norm.ppf(1 - alpha/2)
    
    # For Poisson, variance equals the mean (rate * exposure)
    # For required width = 2 * z_crit * sqrt(variance / exposure^2)
    # Solving for exposure:
    exposure = (4 * z_crit**2 * rate) / width**2
    
    return exposure


def rate_ratio(rate1, rate2):
    """
    Calculate rate ratio (incidence rate ratio) from two rates.
    
    Parameters
    ----------
    rate1 : float
        Rate in first group
    rate2 : float
        Rate in second group
    
    Returns
    -------
    float
        Rate ratio (rate2/rate1)
    """
    return rate2 / rate1


def rate_difference(rate1, rate2):
    """
    Calculate rate difference from two rates.
    
    Parameters
    ----------
    rate1 : float
        Rate in first group
    rate2 : float
        Rate in second group
    
    Returns
    -------
    float
        Rate difference (rate2-rate1)
    """
    return rate2 - rate1


def overdispersion_factor(mean, variance):
    """
    Calculate overdispersion factor for a count variable.
    
    Parameters
    ----------
    mean : float
        Mean count
    variance : float
        Variance of counts
    
    Returns
    -------
    float
        Overdispersion factor (variance/mean)
    """
    return variance / mean


def negative_binomial_parameters(mean, overdispersion):
    """
    Convert mean and overdispersion to negative binomial parameters.
    
    Parameters
    ----------
    mean : float
        Mean count
    overdispersion : float
        Overdispersion factor (variance/mean)
    
    Returns
    -------
    tuple
        (r, p) parameters of negative binomial distribution
    """
    # For negative binomial, mean = r * (1-p)/p and variance = r * (1-p)/p^2
    # Overdispersion = variance/mean = 1/p
    p = 1 / overdispersion
    r = mean * p / (1 - p)
    
    return (r, p)


def sample_size_count_comparison(rate1, rate2, exposure1, exposure2=None, 
                               power=0.8, alpha=0.05, overdispersion=1.0):
    """
    Calculate sample size for comparing two count rates.
    
    Parameters
    ----------
    rate1 : float
        Rate in first group
    rate2 : float
        Rate in second group
    exposure1 : float
        Exposure in first group
    exposure2 : float, optional
        Exposure in second group, by default None (equal to exposure1)
    power : float, optional
        Desired power, by default 0.8
    alpha : float, optional
        Significance level, by default 0.05
    overdispersion : float, optional
        Overdispersion factor, by default 1.0
    
    Returns
    -------
    dict
        Dictionary containing the required exposure times and expected counts
    """
    if exposure2 is None:
        exposure2 = exposure1
    
    # Calculate z-scores for given alpha and power
    z_alpha = stats.norm.ppf(1 - alpha/2)
    z_beta = stats.norm.ppf(power)
    
    # For count data, we use the following formula for the required total count
    # (allowing for overdispersion):
    total_count_req = overdispersion * (z_alpha + z_beta)**2 * (rate1 + rate2) / (rate2 - rate1)**2
    
    # Determine required exposure times
    exposure_ratio = exposure2 / exposure1
    total_exposure_req = total_count_req / (rate1 + rate2 * exposure_ratio)
    
    # Calculate required exposure for each group
    exposure1_req = total_exposure_req
    exposure2_req = exposure1_req * exposure_ratio
    
    # Expected counts
    count1_exp = rate1 * exposure1_req
    count2_exp = rate2 * exposure2_req
    
    return {
        "exposure1": exposure1_req,
        "exposure2": exposure2_req,
        "total_exposure": exposure1_req + exposure2_req,
        "count1": count1_exp,
        "count2": count2_exp,
        "total_count": count1_exp + count2_exp,
        "parameters": {
            "rate1": rate1,
            "rate2": rate2,
            "power": power,
            "alpha": alpha,
            "overdispersion": overdispersion,
            "exposure_ratio": exposure_ratio
        }
    }
