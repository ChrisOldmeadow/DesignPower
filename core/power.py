"""
Power and sample size calculation utilities for various study designs.

This module provides functions to calculate sample size requirements, 
statistical power, and minimum detectable effects for different study designs.
"""
import math
import numpy as np
from scipy import stats


def sample_size_difference_in_means(delta, std_dev, power=0.8, alpha=0.05, allocation_ratio=1.0):
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
            "power": power,
            "alpha": alpha,
            "allocation_ratio": allocation_ratio
        }
    }


def power_difference_in_means(n1, n2, delta, std_dev, alpha=0.05):
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
    
    # Calculate non-centrality parameter
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
            "alpha": alpha
        }
    }


def power_binary_cluster_rct(n_clusters, cluster_size, icc, p1, p2, alpha=0.05):
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


def sample_size_binary_cluster_rct(p1, p2, icc, cluster_size, power=0.8, alpha=0.05):
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


def min_detectable_effect_binary_cluster_rct(n_clusters, cluster_size, icc, p1, power=0.8, alpha=0.05):
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
