"""
Common statistical utilities for sample size and power calculations.

This module provides shared statistical functions that are used across
different study designs and outcome types.
"""
import numpy as np
from scipy import stats
import math


def z_score(alpha, two_sided=True):
    """
    Calculate z-score for a given significance level.
    
    Parameters
    ----------
    alpha : float
        Significance level (e.g., 0.05)
    two_sided : bool, optional
        Whether the test is two-sided, by default True
    
    Returns
    -------
    float
        z-score
    """
    if two_sided:
        return stats.norm.ppf(1 - alpha/2)
    else:
        return stats.norm.ppf(1 - alpha)


def t_score(alpha, df, two_sided=True):
    """
    Calculate t-score for a given significance level and degrees of freedom.
    
    Parameters
    ----------
    alpha : float
        Significance level (e.g., 0.05)
    df : int
        Degrees of freedom
    two_sided : bool, optional
        Whether the test is two-sided, by default True
    
    Returns
    -------
    float
        t-score
    """
    if two_sided:
        return stats.t.ppf(1 - alpha/2, df)
    else:
        return stats.t.ppf(1 - alpha, df)


def pooled_standard_deviation(n1, std1, n2, std2):
    """
    Calculate pooled standard deviation.
    
    Parameters
    ----------
    n1 : int
        Sample size of first group
    std1 : float
        Standard deviation of first group
    n2 : int
        Sample size of second group
    std2 : float
        Standard deviation of second group
    
    Returns
    -------
    float
        Pooled standard deviation
    """
    return math.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))


def cohens_d(mean1, mean2, std_pooled):
    """
    Calculate Cohen's d effect size.
    
    Parameters
    ----------
    mean1 : float
        Mean of first group
    mean2 : float
        Mean of second group
    std_pooled : float
        Pooled standard deviation
    
    Returns
    -------
    float
        Cohen's d
    """
    return abs(mean1 - mean2) / std_pooled


def cohens_d_to_variance_explained(d):
    """
    Convert Cohen's d to variance explained (r-squared).
    
    Parameters
    ----------
    d : float
        Cohen's d effect size
    
    Returns
    -------
    float
        Variance explained (r-squared)
    """
    r = d / math.sqrt(d**2 + 4)
    return r**2


def variance_explained_to_cohens_d(r_squared):
    """
    Convert variance explained (r-squared) to Cohen's d.
    
    Parameters
    ----------
    r_squared : float
        Variance explained
    
    Returns
    -------
    float
        Cohen's d effect size
    """
    r = math.sqrt(r_squared)
    return 2 * r / math.sqrt(1 - r**2)


def intraclass_correlation(between_cluster_variance, within_cluster_variance):
    """
    Calculate intraclass correlation coefficient (ICC).
    
    Parameters
    ----------
    between_cluster_variance : float
        Variance between clusters
    within_cluster_variance : float
        Variance within clusters
    
    Returns
    -------
    float
        Intraclass correlation coefficient
    """
    return between_cluster_variance / (between_cluster_variance + within_cluster_variance)


def design_effect(cluster_size, icc):
    """
    Calculate design effect for cluster randomized trials.
    
    Parameters
    ----------
    cluster_size : int or float
        Average number of observations per cluster
    icc : float
        Intraclass correlation coefficient
    
    Returns
    -------
    float
        Design effect
    """
    return 1 + (cluster_size - 1) * icc


def effective_sample_size(n, design_effect_val):
    """
    Calculate effective sample size after adjusting for design effect.
    
    Parameters
    ----------
    n : int
        Original sample size
    design_effect_val : float
        Design effect
    
    Returns
    -------
    float
        Effective sample size
    """
    return n / design_effect_val


def sample_size_for_power(effect_size, power=0.8, alpha=0.05, two_sided=True):
    """
    Calculate sample size required to achieve desired power for a given effect size.
    
    Parameters
    ----------
    effect_size : float
        Standardized effect size (Cohen's d)
    power : float, optional
        Desired power, by default 0.8
    alpha : float, optional
        Significance level, by default 0.05
    two_sided : bool, optional
        Whether the test is two-sided, by default True
    
    Returns
    -------
    int
        Required sample size per group
    """
    # Calculate z-scores
    z_a = z_score(alpha, two_sided)
    z_b = stats.norm.ppf(power)
    
    # Calculate sample size per group
    n = 2 * ((z_a + z_b) / effect_size)**2
    
    return math.ceil(n)


def power_from_sample_size(effect_size, n, alpha=0.05, two_sided=True):
    """
    Calculate power achieved with a given sample size and effect size.
    
    Parameters
    ----------
    effect_size : float
        Standardized effect size (Cohen's d)
    n : int
        Sample size per group
    alpha : float, optional
        Significance level, by default 0.05
    two_sided : bool, optional
        Whether the test is two-sided, by default True
    
    Returns
    -------
    float
        Achieved power
    """
    # Calculate z-score for alpha
    z_a = z_score(alpha, two_sided)
    
    # Calculate non-centrality parameter
    ncp = effect_size * math.sqrt(n / 2)
    
    # Calculate power
    power = 1 - stats.norm.cdf(z_a - ncp)
    
    if two_sided:
        power += stats.norm.cdf(-z_a - ncp)
    
    return power
