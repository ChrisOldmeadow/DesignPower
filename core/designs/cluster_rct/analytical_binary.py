"""Analytical methods for cluster randomized controlled trials with binary outcomes.

This module provides functions for sample size calculation, power analysis,
and minimum detectable effect estimation for cluster randomized controlled trials
with binary outcomes using analytical formulas.
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
    
    # Calculate non-centrality parameter
    effect = abs(p2 - p1)
    ncp = effect / se
    
    # Calculate power
    power = 1 - stats.norm.cdf(z_alpha - ncp) + stats.norm.cdf(-z_alpha - ncp)
    
    # Calculate additional metrics
    risk_ratio = p2 / p1 if p1 > 0 else float('inf')
    odds_ratio = (p2 / (1 - p2)) / (p1 / (1 - p1)) if p1 < 1 and p2 < 1 else float('inf')
    
    # Format results as dictionary
    results = {
        "power": power,
        "n_clusters": n_clusters,
        "cluster_size": cluster_size,
        "total_n": 2 * n_clusters * cluster_size,
        "icc": icc,
        "p1": p1,
        "p2": p2,
        "risk_difference": abs(p2 - p1),
        "risk_ratio": risk_ratio,
        "odds_ratio": odds_ratio,
        "design_effect": deff,
        "effective_n": n_eff,
        "alpha": alpha
    }
    
    return results


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
    
    # Calculate pooled proportion
    p_pooled = (p1 + p2) / 2
    
    # Critical values for significance level and power
    z_alpha = stats.norm.ppf(1 - alpha/2)
    z_beta = stats.norm.ppf(power)
    
    # Calculate required effective sample size per arm
    effect = abs(p2 - p1)
    n_eff = (2 * p_pooled * (1 - p_pooled) * (z_alpha + z_beta)**2) / (effect**2)
    
    # Calculate required number of clusters per arm (accounting for design effect)
    n_clusters = math.ceil(n_eff * deff / cluster_size)
    
    # Recalculate achieved power with the ceiling function applied
    achieved_power = power_binary(n_clusters, cluster_size, icc, p1, p2, alpha)["power"]
    
    # Calculate additional metrics
    risk_ratio = p2 / p1 if p1 > 0 else float('inf')
    odds_ratio = (p2 / (1 - p2)) / (p1 / (1 - p1)) if p1 < 1 and p2 < 1 else float('inf')
    
    # Format results as dictionary
    results = {
        "n_clusters": n_clusters,
        "cluster_size": cluster_size,
        "total_n": 2 * n_clusters * cluster_size,
        "p1": p1,
        "p2": p2,
        "risk_difference": abs(p2 - p1),
        "risk_ratio": risk_ratio,
        "odds_ratio": odds_ratio,
        "icc": icc,
        "design_effect": deff,
        "alpha": alpha,
        "target_power": power,
        "achieved_power": achieved_power
    }
    
    return results


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
    
    # Critical values for significance level and power
    z_alpha = stats.norm.ppf(1 - alpha/2)
    z_beta = stats.norm.ppf(power)
    
    # Calculate minimum detectable effect (risk difference)
    mde = (z_alpha + z_beta) * math.sqrt(2 * p1 * (1 - p1) / n_eff)
    
    # Ensure MDE is within bounds (no proportion > 1)
    p2 = min(p1 + mde, 0.9999)  # Avoid exactly 1.0 for calculation purposes
    
    # Calculate pooled proportion with the derived p2
    p_pooled = (p1 + p2) / 2
    
    # Recalculate MDE with pooled proportion (more accurate)
    mde_refined = (z_alpha + z_beta) * math.sqrt(2 * p_pooled * (1 - p_pooled) / n_eff)
    p2_refined = min(p1 + mde_refined, 0.9999)  # Refine p2
    
    # Calculate additional metrics
    risk_ratio = p2_refined / p1 if p1 > 0 else float('inf')
    odds_ratio = (p2_refined / (1 - p2_refined)) / (p1 / (1 - p1)) if p1 < 1 and p2_refined < 1 else float('inf')
    
    # Format results as dictionary
    results = {
        "mde": mde_refined,
        "p1": p1,
        "p2": p2_refined,
        "risk_difference": mde_refined,
        "risk_ratio": risk_ratio,
        "odds_ratio": odds_ratio,
        "n_clusters": n_clusters,
        "cluster_size": cluster_size,
        "total_n": 2 * n_clusters * cluster_size,
        "icc": icc,
        "design_effect": deff,
        "effective_n": n_eff,
        "alpha": alpha,
        "power": power
    }
    
    return results