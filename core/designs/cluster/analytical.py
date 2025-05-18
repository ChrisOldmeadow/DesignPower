"""
Analytical methods for cluster randomized controlled trials.

This module provides functions for sample size, power calculation, and
minimum detectable effect estimation for cluster randomized controlled trials
using analytical formulas.
"""
import math
import numpy as np
from scipy import stats


def power_binary(n_clusters, cluster_size, icc, p1, p2, alpha=0.05):
    """
    Calculate power for a cluster randomized controlled trial with binary outcome.
    
    Parameters
    ----------
    n_clusters : int
        Number of clusters per arm
    cluster_size : int
        Average number of individuals per cluster
    icc : float
        Intracluster correlation coefficient
    p1 : float
        Proportion in control group
    p2 : float
        Proportion in intervention group
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
    
    # Calculate pooled proportion
    p_pooled = (p1 + p2) / 2
    
    # Calculate standard error under H0
    se = math.sqrt(2 * p_pooled * (1 - p_pooled) / n_eff)
    
    # Calculate z-score for given alpha
    z_alpha = stats.norm.ppf(1 - alpha/2)
    
    # Calculate z-score for effect
    z_effect = abs(p2 - p1) / se
    
    # Calculate power
    power = stats.norm.cdf(z_effect - z_alpha)
    
    return {
        "power": power,
        "parameters": {
            "n_clusters": n_clusters,
            "cluster_size": cluster_size,
            "total_n": n_clusters * 2 * cluster_size,
            "icc": icc,
            "p1": p1,
            "p2": p2,
            "alpha": alpha
        }
    }


def sample_size_binary(p1, p2, icc, cluster_size, power=0.8, alpha=0.05):
    """
    Calculate required number of clusters for a cluster randomized controlled trial with binary outcome.
    
    Parameters
    ----------
    p1 : float
        Proportion in control group
    p2 : float
        Proportion in intervention group
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
    
    # Calculate z-scores for given alpha and power
    z_alpha = stats.norm.ppf(1 - alpha/2)
    z_beta = stats.norm.ppf(power)
    
    # Calculate pooled proportion
    p_pooled = (p1 + p2) / 2
    
    # Calculate required sample size per arm (without design effect)
    n_per_arm = (2 * p_pooled * (1 - p_pooled) * (z_alpha + z_beta)**2) / (p2 - p1)**2
    
    # Apply design effect and convert to number of clusters
    n_clusters_per_arm = math.ceil((n_per_arm * deff) / cluster_size)
    
    return {
        "n_clusters_per_arm": n_clusters_per_arm,
        "total_clusters": n_clusters_per_arm * 2,
        "total_n": n_clusters_per_arm * 2 * cluster_size,
        "parameters": {
            "p1": p1,
            "p2": p2,
            "icc": icc,
            "cluster_size": cluster_size,
            "power": power,
            "alpha": alpha
        }
    }


def min_detectable_effect_binary(n_clusters, cluster_size, icc, p1, power=0.8, alpha=0.05):
    """
    Calculate minimum detectable effect for a cluster RCT with binary outcome.
    
    Parameters
    ----------
    n_clusters : int
        Number of clusters per arm
    cluster_size : int
        Average number of individuals per cluster
    icc : float
        Intracluster correlation coefficient
    p1 : float
        Proportion in control group
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
    
    # Calculate z-scores for given alpha and power
    z_alpha = stats.norm.ppf(1 - alpha/2)
    z_beta = stats.norm.ppf(power)
    
    # Calculate minimum detectable effect
    mde = math.sqrt(2 * p1 * (1 - p1) * (z_alpha + z_beta)**2 / n_eff)
    p2 = p1 + mde
    
    # Ensure p2 is within bounds
    p2 = min(p2, 1.0)
    
    return {
        "p2": p2,
        "mde": p2 - p1,
        "parameters": {
            "n_clusters": n_clusters,
            "cluster_size": cluster_size,
            "total_n": n_clusters * 2 * cluster_size,
            "icc": icc,
            "p1": p1,
            "power": power,
            "alpha": alpha
        }
    }


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
    
    # Calculate standard error
    se = std_dev * math.sqrt(2 / n_eff)
    
    # Calculate z-score for given alpha
    z_alpha = stats.norm.ppf(1 - alpha/2)
    
    # Calculate z-score for effect
    z_effect = abs(mean2 - mean1) / se
    
    # Calculate power
    power = stats.norm.cdf(z_effect - z_alpha)
    
    return {
        "power": power,
        "parameters": {
            "n_clusters": n_clusters,
            "cluster_size": cluster_size,
            "total_n": n_clusters * 2 * cluster_size,
            "icc": icc,
            "mean1": mean1,
            "mean2": mean2,
            "std_dev": std_dev,
            "alpha": alpha
        }
    }


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
    
    # Calculate z-scores for given alpha and power
    z_alpha = stats.norm.ppf(1 - alpha/2)
    z_beta = stats.norm.ppf(power)
    
    # Calculate required sample size per arm (without design effect)
    n_per_arm = (2 * (std_dev**2) * (z_alpha + z_beta)**2) / (mean2 - mean1)**2
    
    # Apply design effect and convert to number of clusters
    n_clusters_per_arm = math.ceil((n_per_arm * deff) / cluster_size)
    
    return {
        "n_clusters_per_arm": n_clusters_per_arm,
        "total_clusters": n_clusters_per_arm * 2,
        "total_n": n_clusters_per_arm * 2 * cluster_size,
        "parameters": {
            "mean1": mean1,
            "mean2": mean2,
            "std_dev": std_dev,
            "icc": icc,
            "cluster_size": cluster_size,
            "power": power,
            "alpha": alpha
        }
    }
