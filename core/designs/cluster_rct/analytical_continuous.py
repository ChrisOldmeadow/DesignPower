"""Analytical methods for cluster randomized controlled trials with continuous outcomes.

This module provides functions for sample size calculation, power analysis,
and minimum detectable effect estimation for cluster randomized controlled trials
with continuous outcomes using analytical formulas.
"""

import math
import numpy as np
from scipy import stats


def power_continuous(n_clusters, cluster_size, icc, mean1, mean2, std_dev, alpha=0.05):
    """
    Calculate power for a cluster randomized controlled trial with continuous outcome.
    
    Parameters
    ----------
    n_clusters : int
        Number of clusters per arm
    cluster_size : int
        Average number of individuals per cluster
    icc : float
        Intracluster correlation coefficient
    mean1 : float
        Mean outcome in control group
    mean2 : float
        Mean outcome in intervention group
    std_dev : float
        Pooled standard deviation of the outcome
    alpha : float, optional
        Significance level, by default 0.05
    
    Returns
    -------
    dict
        Dictionary containing the calculated power and input parameters
    """
    # Design effect
    deff = 1 + (cluster_size - 1) * icc
    
    # Effective sample size per arm
    n_eff = n_clusters * cluster_size / deff
    
    # Calculate effect size (standardized mean difference)
    delta = abs(mean1 - mean2) / std_dev
    
    # Calculate standard error
    se = math.sqrt(2 / n_eff)
    
    # Calculate critical value for two-sided test
    z_alpha = stats.norm.ppf(1 - alpha/2)
    
    # Calculate non-centrality parameter
    ncp = delta / se
    
    # Calculate power
    power = 1 - stats.norm.cdf(z_alpha - ncp) + stats.norm.cdf(-z_alpha - ncp)
    
    # Format results as dictionary
    results = {
        "power": power,
        "n_clusters": n_clusters,
        "cluster_size": cluster_size,
        "total_n": 2 * n_clusters * cluster_size,
        "icc": icc,
        "mean1": mean1,
        "mean2": mean2,
        "difference": abs(mean1 - mean2),
        "std_dev": std_dev,
        "design_effect": deff,
        "effective_n": n_eff,
        "alpha": alpha
    }
    
    return results


def sample_size_continuous(mean1, mean2, std_dev, icc, cluster_size, power=0.8, alpha=0.05):
    """
    Calculate required number of clusters for a cluster randomized controlled trial with continuous outcome.
    
    Parameters
    ----------
    mean1 : float
        Mean outcome in control group
    mean2 : float
        Mean outcome in intervention group
    std_dev : float
        Pooled standard deviation of the outcome
    icc : float
        Intracluster correlation coefficient
    cluster_size : int
        Average number of individuals per cluster
    power : float, optional
        Desired statistical power (1 - beta), by default 0.8
    alpha : float, optional
        Significance level, by default 0.05
    
    Returns
    -------
    dict
        Dictionary containing the required number of clusters and input parameters
    """
    # Design effect
    deff = 1 + (cluster_size - 1) * icc
    
    # Calculate effect size (standardized mean difference)
    delta = abs(mean1 - mean2) / std_dev
    
    # Critical values for significance level and power
    z_alpha = stats.norm.ppf(1 - alpha/2)
    z_beta = stats.norm.ppf(power)
    
    # Calculate required effective sample size per arm
    n_eff = 2 * ((z_alpha + z_beta) / delta)**2
    
    # Calculate required number of clusters per arm (accounting for design effect)
    n_clusters = math.ceil(n_eff * deff / cluster_size)
    
    # Recalculate achieved power with the ceiling function applied
    achieved_power = power_continuous(n_clusters, cluster_size, icc, mean1, mean2, std_dev, alpha)["power"]
    
    # Format results as dictionary
    results = {
        "n_clusters": n_clusters,
        "cluster_size": cluster_size,
        "total_n": 2 * n_clusters * cluster_size,
        "mean1": mean1,
        "mean2": mean2,
        "difference": abs(mean1 - mean2),
        "std_dev": std_dev,
        "icc": icc,
        "design_effect": deff,
        "alpha": alpha,
        "target_power": power,
        "achieved_power": achieved_power
    }
    
    return results


def min_detectable_effect_continuous(n_clusters, cluster_size, icc, std_dev, power=0.8, alpha=0.05):
    """
    Calculate minimum detectable effect for a cluster RCT with continuous outcome.
    
    Parameters
    ----------
    n_clusters : int
        Number of clusters per arm
    cluster_size : int
        Average number of individuals per cluster
    icc : float
        Intracluster correlation coefficient
    std_dev : float
        Pooled standard deviation of the outcome
    power : float, optional
        Desired statistical power (1 - beta), by default 0.8
    alpha : float, optional
        Significance level, by default 0.05
    
    Returns
    -------
    dict
        Dictionary containing the minimum detectable effect and input parameters
    """
    # Design effect
    deff = 1 + (cluster_size - 1) * icc
    
    # Effective sample size per arm
    n_eff = n_clusters * cluster_size / deff
    
    # Critical values for significance level and power
    z_alpha = stats.norm.ppf(1 - alpha/2)
    z_beta = stats.norm.ppf(power)
    
    # Calculate standard error
    se = math.sqrt(2 / n_eff)
    
    # Calculate minimum detectable standardized effect size
    delta = (z_alpha + z_beta) * se
    
    # Convert standardized effect size to raw difference
    min_diff = delta * std_dev
    
    # Format results as dictionary
    results = {
        "mde": min_diff,
        "standardized_mde": delta,
        "n_clusters": n_clusters,
        "cluster_size": cluster_size,
        "total_n": 2 * n_clusters * cluster_size,
        "icc": icc,
        "std_dev": std_dev,
        "design_effect": deff,
        "effective_n": n_eff,
        "alpha": alpha,
        "power": power
    }
    
    return results