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


def sample_size_continuous(mean1, mean2, std_dev, icc, power=0.8, alpha=0.05, cluster_size=None, n_clusters_fixed=None):
    """
    Calculate required sample size (either number of clusters or cluster size per arm)
    for a cluster randomized controlled trial with continuous outcome.

    Exactly one of 'cluster_size' or 'n_clusters_fixed' must be specified.
    If 'cluster_size' is specified, the function calculates the required 'n_clusters'.
    If 'n_clusters_fixed' is specified, the function calculates the required 'cluster_size'.
    
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
    power : float, optional
        Desired statistical power (1 - beta), by default 0.8
    alpha : float, optional
        Significance level, by default 0.05
    cluster_size : int, optional
        Average number of individuals per cluster. Specify this to calculate n_clusters.
    n_clusters_fixed : int, optional
        Number of clusters per arm. Specify this to calculate cluster_size.
    
    Returns
    -------
    dict
        Dictionary containing the required sample size parameters (n_clusters, cluster_size),
        total sample size, design effect, achieved power, and input parameters.
        May include a 'warning' key if target power is unachievable.
    """

    if not ( (cluster_size is not None and n_clusters_fixed is None) or 
             (cluster_size is None and n_clusters_fixed is not None) ):
        raise ValueError("Exactly one of 'cluster_size' or 'n_clusters_fixed' must be specified.")

    if icc < 0 or icc > 1:
        raise ValueError("ICC must be between 0 and 1.")
    if power <= 0 or power >= 1:
        raise ValueError("Power must be between 0 and 1 (exclusive).")
    if alpha <= 0 or alpha >= 1:
        raise ValueError("Alpha must be between 0 and 1 (exclusive).")
    if std_dev <= 0:
        raise ValueError("Standard deviation must be positive.")

    # Calculate effect size (standardized mean difference)
    delta = abs(mean1 - mean2) / std_dev
    if delta == 0:
        # If there's no difference to detect, sample size is theoretically infinite
        # or minimal if power is not sought for a difference.
        # For practical purposes, returning inf for sample size parameters.
        return {
            "n_clusters": float('inf'),
            "cluster_size": float('inf'),
            "total_n": float('inf'),
            "mean1": mean1,
            "mean2": mean2,
            "difference": 0,
            "std_dev": std_dev,
            "icc": icc,
            "design_effect": float('inf'), # Or undefined
            "alpha": alpha,
            "target_power": power,
            "achieved_power": 0.0, # Or power if no difference means 100% power
            "warning": "Mean outcomes are identical (delta=0), cannot calculate sample size for a difference."
        }

    # Critical values for significance level and power
    z_alpha = stats.norm.ppf(1 - alpha/2)
    z_beta = stats.norm.ppf(power)
    
    # Calculate required effective sample size per arm (n_eff)
    n_eff = 2 * ((z_alpha + z_beta) / delta)**2

    calculated_n_clusters = None
    calculated_cluster_size = None
    warning_message = None
    final_deff = None

    if cluster_size is not None: # Solve for n_clusters
        if not isinstance(cluster_size, int) or cluster_size < 2:
            raise ValueError("Average cluster size (m) must be an integer and at least 2.")
        calculated_cluster_size = cluster_size
        final_deff = 1 + (calculated_cluster_size - 1) * icc
        
        if calculated_cluster_size == 0: # Should be caught by cluster_size < 2
            calculated_n_clusters = float('inf')
        else:
            # Ensure n_eff * final_deff doesn't overflow if huge
            val_for_ceil = (n_eff * final_deff) / calculated_cluster_size
            if val_for_ceil == float('inf'):
                 calculated_n_clusters = float('inf')
            else:
                 calculated_n_clusters = math.ceil(val_for_ceil)
        
        if calculated_n_clusters != float('inf') and calculated_n_clusters < 2:
            calculated_n_clusters = 2 # Minimum 2 clusters per arm

    elif n_clusters_fixed is not None: # Solve for cluster_size
        if not isinstance(n_clusters_fixed, int) or n_clusters_fixed < 2:
            raise ValueError("Number of clusters per arm (k) must be an integer and at least 2.")
        calculated_n_clusters = n_clusters_fixed
        
        denominator = calculated_n_clusters - (n_eff * icc)
        
        if denominator <= 1e-9: # Using a small epsilon to avoid issues with floating point precision near zero
            calculated_cluster_size = float('inf')
            warning_message = (
                f"Cannot achieve target power with {calculated_n_clusters} clusters per arm and ICC={icc}. "
                f"Required cluster size is excessively large or infinite. "
                f"Consider increasing k, alpha, or decreasing power/ICC."
            )
            final_deff = float('inf')
        else:
            cluster_size_float = (n_eff * (1 - icc)) / denominator
            calculated_cluster_size = math.ceil(cluster_size_float)
            if calculated_cluster_size < 2:
                calculated_cluster_size = 2 # Minimum practical cluster size
            final_deff = 1 + (calculated_cluster_size - 1) * icc

    # Recalculate achieved power with the determined integer n_clusters and cluster_size
    achieved_power = 0.0 # Default if cannot be calculated
    if calculated_n_clusters != float('inf') and calculated_cluster_size != float('inf') and \
       calculated_n_clusters > 0 and calculated_cluster_size > 0:
        try:
            # Ensure integer inputs for power_continuous as it might expect them
            final_k = int(calculated_n_clusters)
            final_m = int(calculated_cluster_size)
            
            # power_continuous should handle k < 2 or m < 1 if it has internal checks,
            # but our logic above should ensure final_k >= 2 and final_m >= 2.
            if final_k >=2 and final_m >=1: # power_continuous might need m>=1
                 power_results = power_continuous(
                    n_clusters=final_k, 
                    cluster_size=final_m, 
                    icc=icc, 
                    mean1=mean1, 
                    mean2=mean2, 
                    std_dev=std_dev, 
                    alpha=alpha
                )
                 achieved_power = power_results["power"]
            else:
                # This case should ideally not be reached if prior logic is correct
                achieved_power = 0.0 
                if warning_message is None: warning_message = "Could not calculate achieved power due to invalid k or m."

        except Exception as e:
            # Catch any error during power calculation (e.g. if inputs are still problematic)
            achieved_power = 0.0 # Or np.nan
            if warning_message is None: warning_message = f"Error calculating achieved power: {str(e)}"
    elif warning_message is None and (calculated_n_clusters == float('inf') or calculated_cluster_size == float('inf')):
        warning_message = "Calculated sample size parameter is infinite, target power may not be achievable."


    total_n = float('inf')
    if calculated_n_clusters != float('inf') and calculated_cluster_size != float('inf'):
        total_n = 2 * calculated_n_clusters * calculated_cluster_size
    
    total_clusters = float('inf')
    if calculated_n_clusters != float('inf'):
        total_clusters = 2 * calculated_n_clusters
    
    results = {
        "n_clusters": calculated_n_clusters,
        "cluster_size": calculated_cluster_size,
        "total_n": total_n,
        "total_clusters": total_clusters,  # Total clusters across both arms
        "mean1": mean1,
        "mean2": mean2,
        "difference": abs(mean1 - mean2),
        "std_dev": std_dev,
        "icc": icc,
        "design_effect": final_deff,
        "alpha": alpha,
        "target_power": power,
        "achieved_power": achieved_power
    }
    if warning_message:
        results["warning"] = warning_message
    
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