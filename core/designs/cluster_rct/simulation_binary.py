"""Simulation-based methods for cluster randomized controlled trials with binary outcomes.

This module provides functions for power analysis, sample size calculation,
and minimum detectable effect estimation for cluster randomized controlled trials
with binary outcomes using simulation-based approaches.

Features include:
- Support for equal and unequal cluster sizes
- Multiple effect size specifications (risk difference, risk ratio, odds ratio)
- ICC conversion between linear and logit scales
- Validation and warnings for small numbers of clusters
"""

import numpy as np
from scipy import stats
from tqdm import tqdm
import math
from scipy.special import beta as beta_func
from .cluster_utils import (design_effect_equal, design_effect_unequal, 
                           validate_cluster_parameters, convert_effect_measures)


def simulate_binary_trial(n_clusters, cluster_size, icc, p1, p2, cluster_sizes=None, cv_cluster_size=0):
    """
    Simulate a single cluster RCT with binary outcome using the beta-binomial model.
    
    Parameters
    ----------
    n_clusters : int
        Number of clusters per arm
    cluster_size : int or float
        Average number of individuals per cluster (if cluster_sizes not provided)
    icc : float
        Intracluster correlation coefficient
    p1 : float
        Proportion in control arm
    p2 : float
        Proportion in intervention arm
    cluster_sizes : list or array, optional
        Specific cluster sizes to use. If provided, overrides cluster_size parameter.
    cv_cluster_size : float, optional
        Coefficient of variation for cluster sizes. Used to generate variable cluster
        sizes if cluster_sizes is not provided. Default is 0 (equal cluster sizes).
    
    Returns
    -------
    tuple
        (z_statistic, p_value) for the simulated trial
    """
    # Generate cluster sizes if not provided
    if cluster_sizes is None:
        if cv_cluster_size > 0:
            # Generate variable cluster sizes using gamma distribution
            # (gamma is often used for generating sizes with a specified CV)
            mean_size = cluster_size
            variance = (cv_cluster_size * mean_size) ** 2
            # Calculate shape and scale parameters for gamma distribution
            shape = mean_size ** 2 / variance
            scale = variance / mean_size
            
            # Generate cluster sizes and ensure they're integers at least 1
            control_cluster_sizes = np.maximum(np.round(np.random.gamma(shape, scale, n_clusters)), 1).astype(int)
            intervention_cluster_sizes = np.maximum(np.round(np.random.gamma(shape, scale, n_clusters)), 1).astype(int)
        else:
            # Equal cluster sizes
            control_cluster_sizes = np.full(n_clusters, cluster_size, dtype=int)
            intervention_cluster_sizes = np.full(n_clusters, cluster_size, dtype=int)
    else:
        # Use provided cluster sizes if they match the required number of clusters
        if len(cluster_sizes) >= 2 * n_clusters:
            control_cluster_sizes = np.array(cluster_sizes[:n_clusters], dtype=int)
            intervention_cluster_sizes = np.array(cluster_sizes[n_clusters:2*n_clusters], dtype=int)
        else:
            # If not enough cluster sizes are provided, use them as a distribution to sample from
            control_cluster_sizes = np.random.choice(cluster_sizes, n_clusters)
            intervention_cluster_sizes = np.random.choice(cluster_sizes, n_clusters)
    
    # For the beta-binomial model, we need to convert icc and p to alpha, beta parameters
    # The relationship is: icc = 1 / (1 + alpha + beta)
    # and p = alpha / (alpha + beta)
    
    # Generate cluster-level probabilities
    if icc <= 0:
        # If ICC is zero or negative, use fixed probabilities (no clustering effect)
        control_probs = np.full(n_clusters, p1)
        intervention_probs = np.full(n_clusters, p2)
    else:
        # Handle p1 = 0 or p1 = 1 for control arm
        if p1 == 0.0:
            control_probs = np.zeros(n_clusters)
        elif p1 == 1.0:
            control_probs = np.ones(n_clusters)
        else:
            # Control arm beta parameters
            alpha1 = p1 * (1 - icc) / icc
            beta1 = (1 - p1) * (1 - icc) / icc
            control_probs = np.random.beta(alpha1, beta1, n_clusters)

        # Handle p2 = 0 or p2 = 1 for intervention arm
        if p2 == 0.0:
            intervention_probs = np.zeros(n_clusters)
        elif p2 == 1.0:
            intervention_probs = np.ones(n_clusters)
        else:
            # Intervention arm beta parameters
            alpha2 = p2 * (1 - icc) / icc
            beta2 = (1 - p2) * (1 - icc) / icc
            intervention_probs = np.random.beta(alpha2, beta2, n_clusters)
    
    # Generate binomial data for each cluster with variable cluster sizes
    control_successes = np.zeros(n_clusters)
    intervention_successes = np.zeros(n_clusters)
    
    for i in range(n_clusters):
        control_successes[i] = np.random.binomial(control_cluster_sizes[i], control_probs[i])
        intervention_successes[i] = np.random.binomial(intervention_cluster_sizes[i], intervention_probs[i])
    
    # Calculate means and proportions
    total_control_size = np.sum(control_cluster_sizes)
    total_intervention_size = np.sum(intervention_cluster_sizes)
    
    control_mean = np.sum(control_successes) / total_control_size
    intervention_mean = np.sum(intervention_successes) / total_intervention_size
    
    # Calculate pooled proportion and variance
    total_successes = np.sum(control_successes) + np.sum(intervention_successes)
    total_size = total_control_size + total_intervention_size
    pooled_p = total_successes / total_size
    
    # Calculate effective design effect based on average cluster size
    avg_cluster_size = (total_control_size + total_intervention_size) / (2 * n_clusters)
    deff = 1 + (avg_cluster_size - 1) * icc
    
    # Calculate standard error and test statistic
    pooled_var = pooled_p * (1 - pooled_p) * deff
    se = np.sqrt(pooled_var * (1/total_control_size + 1/total_intervention_size))
    
    # Avoid division by zero
    if se == 0:
        z_statistic = 0.0
        p_value = 1.0
    else:
        z_statistic = abs(intervention_mean - control_mean) / se
        p_value = 2 * (1 - stats.norm.cdf(z_statistic))  # Two-sided p-value
    
    return z_statistic, p_value


def power_binary_sim(n_clusters, cluster_size, icc, p1, p2=None, 
                      nsim=1000, alpha=0.05, seed=None, cv_cluster_size=0,
                      cluster_sizes=None, effect_measure=None, effect_value=None):
    """
    Simulate a cluster RCT with binary outcome and estimate power.
    
    Parameters
    ----------
    n_clusters : int
        Number of clusters per arm
    cluster_size : int or float
        Average number of individuals per cluster
    icc : float
        Intracluster correlation coefficient
    p1 : float
        Proportion in control arm
    p2 : float, optional
        Proportion in intervention arm. If None, it will be calculated from
        effect_measure and effect_value.
    nsim : int, optional
        Number of simulations, by default 1000
    alpha : float, optional
        Significance level, by default 0.05
    seed : int, optional
        Random seed for reproducibility, by default None
    cv_cluster_size : float, optional
        Coefficient of variation of cluster sizes, by default 0
    cluster_sizes : list or array, optional
        List of cluster sizes. If provided, cv_cluster_size will be calculated from these.
    effect_measure : str, optional
        Type of effect measure: 'risk_difference', 'risk_ratio', or 'odds_ratio'.
        Used with effect_value if p2 is None.
    effect_value : float, optional
        Value of the effect measure. Used with effect_measure if p2 is None.
    
    Returns
    -------
    dict
        Dictionary containing the estimated power and simulation details
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
    
    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)
    
    # Initialize counters
    rejections = 0
    z_stats = []
    p_values = []
    
    # Run simulations with progress bar
    for _ in tqdm(range(nsim), desc="Simulating power"):
        # Simulate a single trial with variable cluster sizes
        z_stat, p_value = simulate_binary_trial(
            n_clusters, cluster_size, icc, p1, p2, 
            cluster_sizes=cluster_sizes, cv_cluster_size=cv_cluster_size
        )
        
        # Record results
        z_stats.append(z_stat)
        p_values.append(p_value)
        
        # Count rejections
        if p_value < alpha:
            rejections += 1
    
    # Calculate power as proportion of rejections
    power = rejections / nsim
    
    # Calculate additional metrics
    risk_ratio = p2 / p1 if p1 > 0 else float('inf')
    odds_ratio = (p2 / (1 - p2)) / (p1 / (1 - p1)) if p1 < 1 and p2 < 1 else float('inf')
    
    # Calculate 95% confidence interval for power
    power_se = np.sqrt(power * (1 - power) / nsim)
    power_ci_lower = max(0, power - 1.96 * power_se)
    power_ci_upper = min(1, power + 1.96 * power_se)
    
    # Calculate design effect (accounting for unequal cluster sizes if CV > 0)
    if cv_cluster_size > 0:
        deff = design_effect_unequal(cluster_size, icc, cv_cluster_size)
    else:
        deff = design_effect_equal(cluster_size, icc)
    
    # Validate number of clusters and get warnings
    validation = validate_cluster_parameters(2 * n_clusters)
    
    # Format results
    results = {
        "power": power,
        "power_ci": [power_ci_lower, power_ci_upper],
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
        "cv_cluster_size": cv_cluster_size,
        "nsim": nsim,
        "alpha": alpha,
        "mean_z_stat": np.mean(z_stats),
        "mean_p_value": np.mean(p_values)
    }
    
    # Add validation warnings
    if not validation["valid"]:
        results["warnings"] = validation["warnings"]
    
    return results


def sample_size_binary_sim(p1, p2=None, icc=0.01, cluster_size=50, 
                            power=0.8, alpha=0.05, nsim=1000, 
                            min_n=2, max_n=100, seed=None,
                            cv_cluster_size=0, cluster_sizes=None,
                            effect_measure=None, effect_value=None):
    """
    Find required sample size for a cluster RCT with binary outcome using simulation.
    
    Parameters
    ----------
    p1 : float
        Proportion in control arm
    p2 : float, optional
        Proportion in intervention arm. If None, it will be calculated from
        effect_measure and effect_value, by default None
    icc : float, optional
        Intracluster correlation coefficient, by default 0.01
    cluster_size : int or float, optional
        Average number of individuals per cluster, by default 50
    power : float, optional
        Desired power, by default 0.8
    alpha : float, optional
        Significance level, by default 0.05
    nsim : int, optional
        Number of simulations per sample size, by default 1000
    min_n : int, optional
        Minimum number of clusters to try, by default 2
    max_n : int, optional
        Maximum number of clusters to try, by default 100
    seed : int, optional
        Random seed for reproducibility, by default None
    cv_cluster_size : float, optional
        Coefficient of variation of cluster sizes, by default 0
    cluster_sizes : list or array, optional
        List of cluster sizes. If provided, cv_cluster_size will be calculated from these.
    effect_measure : str, optional
        Type of effect measure: 'risk_difference', 'risk_ratio', or 'odds_ratio'.
        Used with effect_value if p2 is None.
    effect_value : float, optional
        Value of the effect measure. Used with effect_measure if p2 is None.
    
    Returns
    -------
    dict
        Dictionary containing the required number of clusters and simulation details
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
    
    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)
    
    # Calculate design effect (accounting for unequal cluster sizes if CV > 0)
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
    n_clusters_estimate = max(min_n, min(max_n, math.ceil(n_eff * deff / cluster_size)))
    
    # Use binary search to find the optimal number of clusters
    low = min_n
    high = max(n_clusters_estimate * 2, max_n)  # Double the analytical estimate as upper bound
    
    # Track the minimum n_clusters that meets the power requirement
    min_adequate_n = high
    
    print(f"Starting binary search with n_clusters between {low} and {high}")
    
    while low <= high:
        mid = (low + high) // 2
        print(f"Testing n_clusters = {mid}...")
        
        # Run simulation with current n_clusters
        sim_results = power_binary_sim(
            n_clusters=mid, 
            cluster_size=cluster_size,
            icc=icc,
            p1=p1,
            p2=p2,
            nsim=nsim,
            alpha=alpha,
            seed=seed,
            cv_cluster_size=cv_cluster_size,
            cluster_sizes=cluster_sizes
        )
        
        empirical_power = sim_results["power"]
        print(f"Achieved power: {empirical_power:.4f}")
        
        if empirical_power >= power:
            # This n_clusters is sufficient, try smaller
            min_adequate_n = min(min_adequate_n, mid)
            high = mid - 1
        else:
            # This n_clusters is insufficient, try larger
            low = mid + 1
    
    # Get final power for the optimal n_clusters
    final_results = power_binary_sim(
        n_clusters=min_adequate_n, 
        cluster_size=cluster_size,
        icc=icc,
        p1=p1,
        p2=p2,
        nsim=nsim,
        alpha=alpha,
        seed=seed,
        cv_cluster_size=cv_cluster_size,
        cluster_sizes=cluster_sizes
    )
    
    # Calculate additional metrics
    risk_ratio = p2 / p1 if p1 > 0 else float('inf')
    odds_ratio = (p2 / (1 - p2)) / (p1 / (1 - p1)) if p1 < 1 and p2 < 1 else float('inf')
    
    # Calculate design effect (accounting for unequal cluster sizes if CV > 0)
    if cv_cluster_size > 0:
        deff = design_effect_unequal(cluster_size, icc, cv_cluster_size)
    else:
        deff = design_effect_equal(cluster_size, icc)
    
    # Validate number of clusters and get warnings
    validation = validate_cluster_parameters(2 * min_adequate_n)
    
    # Format results as dictionary
    results = {
        "n_clusters": min_adequate_n,
        "cluster_size": cluster_size,
        "total_n": 2 * min_adequate_n * cluster_size,
        "power": final_results["power"],
        "icc": icc,
        "p1": p1,
        "p2": p2,
        "risk_difference": abs(p2 - p1),
        "risk_ratio": risk_ratio,
        "odds_ratio": odds_ratio,
        "design_effect": deff,
        "cv_cluster_size": cv_cluster_size,
        "alpha": alpha,
        "nsim": nsim,
        "sim_details": {
            "method": "simulation",
            "iterations": final_results["nsim"],
            "search_range": [min_n, max_n]
        }
    }
    
    # Add validation warnings
    if not validation["valid"]:
        results["warnings"] = validation["warnings"]
    
    return results


def min_detectable_effect_binary_sim(n_clusters, cluster_size, icc, p1,
                                      power=0.8, alpha=0.05, nsim=1000,
                                      min_effect=0.01, max_effect=0.5,
                                      precision=0.01, max_iterations=10,
                                      seed=None, cv_cluster_size=0, cluster_sizes=None,
                                      effect_measure='risk_difference'):
    """
    Calculate minimum detectable effect for a cluster RCT with binary outcome using simulation.
    
    Parameters
    ----------
    n_clusters : int
        Number of clusters per arm
    cluster_size : int or float
        Average number of individuals per cluster
    icc : float
        Intracluster correlation coefficient
    p1 : float
        Proportion in control arm
    power : float, optional
        Desired power, by default 0.8
    alpha : float, optional
        Significance level, by default 0.05
    nsim : int, optional
        Number of simulations per effect size, by default 1000
    min_effect : float, optional
        Minimum effect size to consider, by default 0.01
    max_effect : float, optional
        Maximum effect size to consider, by default 0.5
    precision : float, optional
        Precision for effect size estimation, by default 0.01
    max_iterations : int, optional
        Maximum number of iterations for binary search, by default 10
    seed : int, optional
        Random seed for reproducibility, by default None
    cv_cluster_size : float, optional
        Coefficient of variation of cluster sizes, by default 0
    cluster_sizes : list or array, optional
        List of cluster sizes. If provided, cv_cluster_size will be calculated from these.
    effect_measure : str, optional
        Type of effect measure to return: 'risk_difference', 'risk_ratio', or 'odds_ratio'.
        Default is 'risk_difference'
    
    Returns
    -------
    dict
        Dictionary containing the minimum detectable effect and simulation details
    """
    # Calculate CV from cluster sizes if provided
    if cluster_sizes is not None:
        cv_cluster_size = np.std(cluster_sizes) / np.mean(cluster_sizes)
    
    # Calculate design effect (accounting for unequal cluster sizes if CV > 0)
    if cv_cluster_size > 0:
        deff = design_effect_unequal(cluster_size, icc, cv_cluster_size)
    else:
        deff = design_effect_equal(cluster_size, icc)
    
    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)
    
    # Get a rough estimate from analytical formula to use as a starting point
    # Design effect
    deff = 1 + (cluster_size - 1) * icc
    
    # Effective sample size per arm
    n_eff = n_clusters * cluster_size / deff
    
    # Critical values for significance level and power
    z_alpha = stats.norm.ppf(1 - alpha/2)
    z_beta = stats.norm.ppf(power)
    
    # Calculate minimum detectable effect (risk difference) using normal approximation
    mde_estimate = (z_alpha + z_beta) * math.sqrt(2 * p1 * (1 - p1) / n_eff)
    
    # Ensure mde_estimate is within bounds
    mde_estimate = max(min_effect, min(max_effect, mde_estimate))
    
    # Calculate p2 from p1 and mde_estimate
    p2_estimate = min(p1 + mde_estimate, 0.9999)
    
    # Refine with pooled proportion
    p_pooled = (p1 + p2_estimate) / 2
    mde_refined = (z_alpha + z_beta) * math.sqrt(2 * p_pooled * (1 - p_pooled) / n_eff)
    mde_estimate = max(min_effect, min(max_effect, mde_refined))
    
    # Use binary search to find the minimum detectable effect
    low = max(min_effect, mde_estimate / 2)
    high = min(max_effect, min(mde_estimate * 2, 1 - p1))  # Ensure high doesn't exceed valid p2 range
    
    print(f"Starting binary search with effect size between {low:.4f} and {high:.4f}")
    
    iteration = 0
    min_adequate_effect = high
    
    while iteration < max_iterations and high - low > precision:
        mid = (low + high) / 2
        print(f"Iteration {iteration + 1}: Testing effect size = {mid:.4f}...")
        
        # Calculate p2 based on effect size
        p2_current = p1 + mid
        
        if p2_current > 0.9999:  # Avoid numerical issues near 1
            p2_current = 0.9999
        
        # Run simulation with current effect size
        sim_results = power_binary_sim(
            n_clusters=n_clusters, 
            cluster_size=cluster_size,
            icc=icc,
            p1=p1,
            p2=p2_current,
            nsim=nsim,
            alpha=alpha,
            seed=seed,
            cv_cluster_size=cv_cluster_size,
            cluster_sizes=cluster_sizes
        )
        
        empirical_power = sim_results["power"]
        print(f"Achieved power: {empirical_power:.4f}")
        
        if empirical_power >= power:
            # This effect size is sufficient, try smaller
            min_adequate_effect = min(min_adequate_effect, mid)
            high = mid
        else:
            # This effect size is insufficient, try larger
            low = mid
        
        iteration += 1
    
    # Use the minimum effect size that meets the power requirement
    final_effect = min_adequate_effect
    final_p2 = min(p1 + final_effect, 0.9999)
    
    # Run final simulation to get accurate power estimate
    final_results = power_binary_sim(
        n_clusters=n_clusters,
        cluster_size=cluster_size,
        icc=icc,
        p1=p1,
        p2=final_p2,
        nsim=nsim,
        alpha=alpha,
        seed=seed
    )
    
    # Extract the empirical power from final simulation
    achieved_power = final_results["power"]
    
    # Calculate p2 based on the minimum adequate effect
    p2_final = p1 + min_adequate_effect
    
    # Calculate effect measures
    risk_difference = min_adequate_effect
    risk_ratio = p2_final / p1 if p1 > 0 else float('inf')
    odds_ratio = (p2_final / (1 - p2_final)) / (p1 / (1 - p1)) if p1 < 1 and p2_final < 1 else float('inf')
    
    # Determine which effect measure to return as the primary MDE
    if effect_measure == 'risk_difference':
        mde_primary = risk_difference
    elif effect_measure == 'risk_ratio':
        mde_primary = risk_ratio
    elif effect_measure == 'odds_ratio':
        mde_primary = odds_ratio
    else:
        # Default to risk difference
        mde_primary = risk_difference
        effect_measure = 'risk_difference'
    
    # Validate number of clusters and get warnings
    validation = validate_cluster_parameters(2 * n_clusters)
    
    # Format results as dictionary
    results = {
        "mde": mde_primary,
        "effect_measure": effect_measure,
        "p1": p1,
        "p2": p2_final,
        "risk_difference": risk_difference,
        "risk_ratio": risk_ratio,
        "odds_ratio": odds_ratio,
        "n_clusters": n_clusters,
        "cluster_size": cluster_size,
        "total_n": 2 * n_clusters * cluster_size,
        "icc": icc,
        "design_effect": deff,
        "cv_cluster_size": cv_cluster_size,
        "alpha": alpha,
        "power": power, # This is the target power
        "achieved_power": achieved_power, # This is the empirical power for the MDE
        "nsim": nsim,
        "iterations": iteration,
        "sim_details": {
            "method": "simulation",
            "search_range": [min_effect, max_effect],
            "precision": precision,
            "iterations": iteration
        }
    }
    
    # Add validation warnings
    if not validation["valid"]:
        results["warnings"] = validation["warnings"]
    
    return results