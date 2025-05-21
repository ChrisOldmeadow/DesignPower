"""Simulation-based methods for cluster randomized controlled trials with binary outcomes.

This module provides functions for power analysis, sample size calculation,
and minimum detectable effect estimation for cluster randomized controlled trials
with binary outcomes using simulation-based approaches.
"""

import numpy as np
from scipy import stats
from tqdm import tqdm
import math
from scipy.special import beta as beta_func


def simulate_binary_trial(n_clusters, cluster_size, icc, p1, p2):
    """
    Simulate a single cluster RCT with binary outcome using the beta-binomial model.
    
    Parameters
    ----------
    n_clusters : int
        Number of clusters per arm
    cluster_size : int
        Number of individuals per cluster
    icc : float
        Intracluster correlation coefficient
    p1 : float
        Proportion in control arm
    p2 : float
        Proportion in intervention arm
    
    Returns
    -------
    tuple
        (z_statistic, p_value) for the simulated trial
    """
    # For the beta-binomial model, we need to convert icc and p to alpha, beta parameters
    # The relationship is: icc = 1 / (1 + alpha + beta)
    # and p = alpha / (alpha + beta)
    
    # Control arm beta parameters
    if icc > 0:
        # Calculate alpha and beta for control arm
        alpha1 = p1 * (1 - icc) / icc
        beta1 = (1 - p1) * (1 - icc) / icc
        
        # Calculate alpha and beta for intervention arm
        alpha2 = p2 * (1 - icc) / icc
        beta2 = (1 - p2) * (1 - icc) / icc
        
        # Generate cluster-level probabilities from beta distributions
        control_probs = np.random.beta(alpha1, beta1, n_clusters)
        intervention_probs = np.random.beta(alpha2, beta2, n_clusters)
    else:
        # If ICC=0, use fixed probabilities (no clustering effect)
        control_probs = np.full(n_clusters, p1)
        intervention_probs = np.full(n_clusters, p2)
    
    # Generate binomial data for each cluster
    control_successes = np.zeros(n_clusters)
    intervention_successes = np.zeros(n_clusters)
    
    for i in range(n_clusters):
        control_successes[i] = np.random.binomial(cluster_size, control_probs[i])
        intervention_successes[i] = np.random.binomial(cluster_size, intervention_probs[i])
    
    # Calculate means and proportions
    control_mean = np.sum(control_successes) / (n_clusters * cluster_size)
    intervention_mean = np.sum(intervention_successes) / (n_clusters * cluster_size)
    
    # Pool the results
    pooled_p = (np.sum(control_successes) + np.sum(intervention_successes)) / (2 * n_clusters * cluster_size)
    pooled_var = pooled_p * (1 - pooled_p) * (1 + (cluster_size - 1) * icc)
    
    # Calculate test statistic and p-value
    se = np.sqrt(2 * pooled_var / (n_clusters * cluster_size))
    z_stat = abs(intervention_mean - control_mean) / se
    p_value = 2 * (1 - stats.norm.cdf(z_stat))  # Two-sided test
    
    return z_stat, p_value


def power_binary_sim(n_clusters, cluster_size, icc, p1, p2, 
                      nsim=1000, alpha=0.05, seed=None):
    """
    Simulate a cluster RCT with binary outcome and estimate power.
    
    Parameters
    ----------
    n_clusters : int
        Number of clusters per arm
    cluster_size : int
        Number of individuals per cluster
    icc : float
        Intracluster correlation coefficient
    p1 : float
        Proportion in control arm
    p2 : float
        Proportion in intervention arm
    nsim : int, optional
        Number of simulations, by default 1000
    alpha : float, optional
        Significance level, by default 0.05
    seed : int, optional
        Random seed for reproducibility, by default None
    
    Returns
    -------
    dict
        Dictionary containing the estimated power and simulation details
    """
    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)
    
    # Initialize counter for significant results
    sig_count = 0
    
    # Store p-values for all simulations
    p_values = []
    
    # Run simulations with progress bar
    for _ in tqdm(range(nsim), desc="Running simulations", disable=False):
        _, p_value = simulate_binary_trial(n_clusters, cluster_size, icc, p1, p2)
        p_values.append(p_value)
        
        # Increment counter if result is significant
        if p_value < alpha:
            sig_count += 1
    
    # Calculate design effect
    deff = 1 + (cluster_size - 1) * icc
    
    # Calculate effective sample size
    n_eff = n_clusters * cluster_size / deff
    
    # Calculate empirical power
    empirical_power = sig_count / nsim
    
    # Calculate additional metrics
    risk_ratio = p2 / p1 if p1 > 0 else float('inf')
    odds_ratio = (p2 / (1 - p2)) / (p1 / (1 - p1)) if p1 < 1 and p2 < 1 else float('inf')
    
    # Format results as dictionary
    results = {
        "power": empirical_power,
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
        "nsim": nsim,
        "significant_sims": sig_count,
        "p_values_mean": np.mean(p_values),
        "p_values_median": np.median(p_values),
        "sim_details": {
            "method": "simulation",
            "sim_type": "cluster_binary"
        }
    }
    
    return results


def sample_size_binary_sim(p1, p2, icc, cluster_size, 
                            power=0.8, alpha=0.05, nsim=1000, 
                            min_n=2, max_n=100, seed=None):
    """
    Find required sample size for a cluster RCT with binary outcome using simulation.
    
    Parameters
    ----------
    p1 : float
        Proportion in control arm
    p2 : float
        Proportion in intervention arm
    icc : float
        Intracluster correlation coefficient
    cluster_size : int
        Number of individuals per cluster
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
    
    Returns
    -------
    dict
        Dictionary containing the required number of clusters and simulation details
    """
    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)
    
    # Get a rough estimate from analytical formula to use as a starting point
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
            seed=seed
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
    
    # Use the minimum n_clusters that meets the power requirement
    final_n_clusters = min_adequate_n
    
    # Run final simulation to get accurate power estimate
    final_results = power_binary_sim(
        n_clusters=final_n_clusters, 
        cluster_size=cluster_size,
        icc=icc,
        p1=p1,
        p2=p2,
        nsim=nsim,
        alpha=alpha,
        seed=seed
    )
    
    # Extract the empirical power from final simulation
    achieved_power = final_results["power"]
    
    # Calculate additional metrics
    risk_ratio = p2 / p1 if p1 > 0 else float('inf')
    odds_ratio = (p2 / (1 - p2)) / (p1 / (1 - p1)) if p1 < 1 and p2 < 1 else float('inf')
    
    # Format results as dictionary
    results = {
        "n_clusters": final_n_clusters,
        "cluster_size": cluster_size,
        "total_n": 2 * final_n_clusters * cluster_size,
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
        "nsim": nsim,
        "p_values_mean": final_results["p_values_mean"],
        "p_values_median": final_results["p_values_median"],
        "sim_details": {
            "method": "simulation",
            "sim_type": "cluster_binary",
            "analytical_estimate": n_clusters_estimate
        }
    }
    
    return results


def min_detectable_effect_binary_sim(n_clusters, cluster_size, icc, p1,
                                     power=0.8, alpha=0.05, nsim=1000,
                                     min_effect=0.01, max_effect=0.5,
                                     precision=0.01, max_iterations=10,
                                     seed=None):
    """
    Calculate minimum detectable effect for a cluster RCT with binary outcome using simulation.
    
    Parameters
    ----------
    n_clusters : int
        Number of clusters per arm
    cluster_size : int
        Number of individuals per cluster
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
    
    Returns
    -------
    dict
        Dictionary containing the minimum detectable effect and simulation details
    """
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
        p2 = p1 + mid
        
        if p2 > 0.9999:  # Avoid numerical issues near 1
            p2 = 0.9999
        
        # Run simulation with current effect size
        sim_results = power_binary_sim(
            n_clusters=n_clusters,
            cluster_size=cluster_size,
            icc=icc,
            p1=p1,
            p2=p2,
            nsim=nsim,
            alpha=alpha,
            seed=seed
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
    
    # Calculate additional metrics
    risk_ratio = final_p2 / p1 if p1 > 0 else float('inf')
    odds_ratio = (final_p2 / (1 - final_p2)) / (p1 / (1 - p1)) if p1 < 1 and final_p2 < 1 else float('inf')
    
    # Format results as dictionary
    results = {
        "mde": final_effect,
        "p1": p1,
        "p2": final_p2,
        "risk_difference": final_effect,
        "risk_ratio": risk_ratio,
        "odds_ratio": odds_ratio,
        "n_clusters": n_clusters,
        "cluster_size": cluster_size,
        "total_n": 2 * n_clusters * cluster_size,
        "icc": icc,
        "design_effect": deff,
        "effective_n": n_eff,
        "alpha": alpha,
        "target_power": power,
        "achieved_power": achieved_power,
        "nsim": nsim,
        "iterations": iteration,
        "sim_details": {
            "method": "simulation",
            "sim_type": "cluster_binary",
            "analytical_estimate": mde_estimate
        }
    }
    
    return results