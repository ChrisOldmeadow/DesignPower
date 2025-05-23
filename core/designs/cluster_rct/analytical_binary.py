"""Analytical methods for cluster randomized controlled trials with binary outcomes.

This module provides functions for sample size calculation, power analysis,
and minimum detectable effect estimation for cluster randomized controlled trials
with binary outcomes using analytical formulas.

Features include:
- Support for equal and unequal cluster sizes
- Multiple effect size specifications (risk difference, risk ratio, odds ratio)
- ICC conversion between linear and logit scales
- Validation and warnings for small numbers of clusters
"""

import math
import numpy as np
from scipy import stats
from .cluster_utils import (design_effect_equal, design_effect_unequal, 
                           validate_cluster_parameters, convert_effect_measures)


def power_binary(n_clusters, cluster_size, icc, p1, p2, alpha=0.05, cv_cluster_size=0, 
                effect_measure=None, effect_value=None, cluster_sizes=None):
    """
    Calculate power for a cluster randomized controlled trial with binary outcome.
    
    Parameters
    ----------
    n_clusters : int
        Number of clusters per arm
    cluster_size : int or float
        Average number of individuals per cluster
    icc : float
        Intracluster correlation coefficient
    p1 : float
        Proportion in control group
    p2 : float, optional
        Proportion in intervention group. If None, it will be calculated from
        effect_measure and effect_value.
    alpha : float, optional
        Significance level, by default 0.05
    cv_cluster_size : float, optional
        Coefficient of variation of cluster sizes, by default 0
    effect_measure : str, optional
        Type of effect measure: 'risk_difference', 'risk_ratio', or 'odds_ratio'.
        Used with effect_value if p2 is None.
    effect_value : float, optional
        Value of the effect measure. Used with effect_measure if p2 is None.
    cluster_sizes : list or array, optional
        List of cluster sizes. If provided, cv_cluster_size will be calculated.
    
    Returns
    -------
    dict
        Dictionary containing the calculated power and input parameters
    """
    # Calculate p2 from effect measure if needed
    if p2 is None and effect_measure is not None and effect_value is not None:
        effect_info = convert_effect_measures(p1, effect_measure, effect_value)
        p2 = effect_info['p2']
    
    # Calculate CV from cluster sizes if provided
    if cluster_sizes is not None:
        cv_cluster_size = np.std(cluster_sizes) / np.mean(cluster_sizes)
    
    # Design effect (accounting for unequal cluster sizes if CV > 0)
    if cv_cluster_size > 0:
        deff = design_effect_unequal(cluster_size, icc, cv_cluster_size)
    else:
        deff = design_effect_equal(cluster_size, icc)
    
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
    
    # Validate number of clusters and get warnings
    validation = validate_cluster_parameters(2 * n_clusters)
    
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
        "alpha": alpha,
        "cv_cluster_size": cv_cluster_size
    }
    
    # Add validation warnings
    if not validation["valid"]:
        results["warnings"] = validation["warnings"]
    
    return results


def sample_size_binary(p1, p2=None, icc=0.01, cluster_size=50, power=0.8, alpha=0.05,
                      cv_cluster_size=0, effect_measure=None, effect_value=None, cluster_sizes=None):
    """
    Calculate required number of clusters for a cluster randomized controlled trial with binary outcome.
    
    Parameters
    ----------
    p1 : float
        Proportion in control group
    p2 : float, optional
        Proportion in intervention group. If None, it will be calculated from
        effect_measure and effect_value.
    icc : float, optional
        Intracluster correlation coefficient, by default 0.01
    cluster_size : int or float, optional
        Average number of individuals per cluster, by default 50
    power : float, optional
        Desired statistical power (1 - beta), by default 0.8
    alpha : float, optional
        Significance level, by default 0.05
    cv_cluster_size : float, optional
        Coefficient of variation of cluster sizes, by default 0
    effect_measure : str, optional
        Type of effect measure: 'risk_difference', 'risk_ratio', or 'odds_ratio'.
        Used with effect_value if p2 is None.
    effect_value : float, optional
        Value of the effect measure. Used with effect_measure if p2 is None.
    cluster_sizes : list or array, optional
        List of cluster sizes. If provided, cv_cluster_size will be calculated.
    
    Returns
    -------
    dict
        Dictionary containing the required number of clusters and input parameters
    """
    # Calculate p2 from effect measure if needed
    if p2 is None and effect_measure is not None and effect_value is not None:
        effect_info = convert_effect_measures(p1, effect_measure, effect_value)
        p2 = effect_info['p2']
    elif p2 is None:
        raise ValueError("Either p2 or both effect_measure and effect_value must be provided")
    
    # Calculate CV from cluster sizes if provided
    if cluster_sizes is not None:
        cv_cluster_size = np.std(cluster_sizes) / np.mean(cluster_sizes)
    
    # Design effect (accounting for unequal cluster sizes if CV > 0)
    if cv_cluster_size > 0:
        deff = design_effect_unequal(cluster_size, icc, cv_cluster_size)
    else:
        deff = design_effect_equal(cluster_size, icc)
    
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
    achieved_power = power_binary(
        n_clusters, cluster_size, icc, p1, p2, alpha, 
        cv_cluster_size=cv_cluster_size
    )["power"]
    
    # Calculate additional metrics
    risk_ratio = p2 / p1 if p1 > 0 else float('inf')
    odds_ratio = (p2 / (1 - p2)) / (p1 / (1 - p1)) if p1 < 1 and p2 < 1 else float('inf')
    
    # Validate number of clusters and get warnings
    validation = validate_cluster_parameters(2 * n_clusters)
    
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
        "achieved_power": achieved_power,
        "cv_cluster_size": cv_cluster_size
    }
    
    # Add validation warnings
    if not validation["valid"]:
        results["warnings"] = validation["warnings"]
    
    return results


def min_detectable_effect_binary(n_clusters, cluster_size, icc, p1, power=0.8, alpha=0.05,
                              cv_cluster_size=0, cluster_sizes=None, effect_measure='risk_difference'):
    """
    Calculate minimum detectable effect for a cluster RCT with binary outcome.
    
    Parameters
    ----------
    n_clusters : int
        Number of clusters per arm
    cluster_size : int or float
        Average number of individuals per cluster
    icc : float
        Intracluster correlation coefficient
    p1 : float
        Proportion in control group
    power : float, optional
        Desired statistical power (1 - beta), by default 0.8
    alpha : float, optional
        Significance level, by default 0.05
    cv_cluster_size : float, optional
        Coefficient of variation of cluster sizes, by default 0
    cluster_sizes : list or array, optional
        List of cluster sizes. If provided, cv_cluster_size will be calculated.
    effect_measure : str, optional
        Type of effect measure to return: 'risk_difference', 'risk_ratio', or 'odds_ratio'.
        Default is 'risk_difference'
    
    Returns
    -------
    dict
        Dictionary containing the minimum detectable effect and input parameters
    """
    # Calculate CV from cluster sizes if provided
    if cluster_sizes is not None:
        cv_cluster_size = np.std(cluster_sizes) / np.mean(cluster_sizes)
    
    # Design effect (accounting for unequal cluster sizes if CV > 0)
    if cv_cluster_size > 0:
        deff = design_effect_unequal(cluster_size, icc, cv_cluster_size)
    else:
        deff = design_effect_equal(cluster_size, icc)
    
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
    
    # Validate number of clusters and get warnings
    validation = validate_cluster_parameters(2 * n_clusters)
    
    # Determine which effect measure to return as the primary MDE
    if effect_measure == 'risk_difference':
        mde_primary = mde_refined
    elif effect_measure == 'risk_ratio':
        mde_primary = risk_ratio
    elif effect_measure == 'odds_ratio':
        mde_primary = odds_ratio
    else:
        # Default to risk difference
        mde_primary = mde_refined
        effect_measure = 'risk_difference'
    
    # Format results as dictionary
    results = {
        "mde": mde_primary,
        "effect_measure": effect_measure,
        "n_clusters": n_clusters,
        "cluster_size": cluster_size,
        "total_n": 2 * n_clusters * cluster_size,
        "p1": p1,
        "p2": p2_refined,
        "risk_difference": mde_refined,
        "risk_ratio": risk_ratio,
        "odds_ratio": odds_ratio,
        "icc": icc,
        "design_effect": deff,
        "effective_n": n_eff,
        "alpha": alpha,
        "power": power,
        "cv_cluster_size": cv_cluster_size
    }
    
    # Add validation warnings
    if not validation["valid"]:
        results["warnings"] = validation["warnings"]
    
    # Verify that the calculated power matches the requested power
    achieved_power = power_binary(
        n_clusters, cluster_size, icc, p1, p2_refined, alpha, 
        cv_cluster_size=cv_cluster_size
    )["power"]
    results["achieved_power"] = achieved_power
    
    return results