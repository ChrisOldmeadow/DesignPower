"""Simulation-based methods for cluster randomized controlled trials with continuous outcomes.

This module provides functions for power analysis, sample size calculation,
and minimum detectable effect estimation for cluster randomized controlled trials
with continuous outcomes using simulation-based approaches.
"""

import numpy as np
from scipy import stats
from tqdm import tqdm
import math


def simulate_continuous_trial(n_clusters, cluster_size, icc, mean1, mean2, std_dev):
    """
    Simulate a single cluster RCT with continuous outcome.
    
    Parameters
    ----------
    n_clusters : int
        Number of clusters per arm
    cluster_size : int
        Number of individuals per cluster
    icc : float
        Intracluster correlation coefficient
    mean1 : float
        Mean outcome in control arm
    mean2 : float
        Mean outcome in intervention arm
    std_dev : float
        Total standard deviation of outcome
    
    Returns
    -------
    tuple
        (t_statistic, p_value) for the simulated trial
    """
    # Calculate between-cluster and within-cluster variance
    var_between = icc * std_dev**2
    var_within = (1 - icc) * std_dev**2
    
    # Generate random cluster effects for control arm
    control_cluster_effects = np.random.normal(0, np.sqrt(var_between), n_clusters)
    
    # Generate random cluster effects for intervention arm
    intervention_cluster_effects = np.random.normal(0, np.sqrt(var_between), n_clusters)
    
    # Initialize arrays for all data points
    control_data = np.zeros(n_clusters * cluster_size)
    intervention_data = np.zeros(n_clusters * cluster_size)
    
    # Generate data for each cluster
    for i in range(n_clusters):
        # Control arm: base mean + cluster effect + individual variation
        start_idx = i * cluster_size
        end_idx = start_idx + cluster_size
        control_data[start_idx:end_idx] = mean1 + control_cluster_effects[i] + \
                                         np.random.normal(0, np.sqrt(var_within), cluster_size)
        
        # Intervention arm: base mean + cluster effect + individual variation
        intervention_data[start_idx:end_idx] = mean2 + intervention_cluster_effects[i] + \
                                              np.random.normal(0, np.sqrt(var_within), cluster_size)
    
    # Perform t-test (accounting for cluster design)
    t_stat, p_value = stats.ttest_ind(control_data, intervention_data, equal_var=True)
    
    return t_stat, p_value


def power_continuous_sim(n_clusters, cluster_size, icc, mean1, mean2, std_dev, 
                          nsim=1000, alpha=0.05, seed=None):
    """
    Simulate a cluster RCT with continuous outcome and estimate power.
    
    Parameters
    ----------
    n_clusters : int
        Number of clusters per arm
    cluster_size : int
        Number of individuals per cluster
    icc : float
        Intracluster correlation coefficient
    mean1 : float
        Mean outcome in control arm
    mean2 : float
        Mean outcome in intervention arm
    std_dev : float
        Total standard deviation of outcome
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
        _, p_value = simulate_continuous_trial(n_clusters, cluster_size, icc, mean1, mean2, std_dev)
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
    
    # Format results as dictionary
    results = {
        "power": empirical_power,
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
        "alpha": alpha,
        "nsim": nsim,
        "significant_sims": sig_count,
        "p_values_mean": np.mean(p_values),
        "p_values_median": np.median(p_values),
        "sim_details": {
            "method": "simulation",
            "sim_type": "cluster_continuous"
        }
    }
    
    return results


def sample_size_continuous_sim(mean1, mean2, std_dev, icc, cluster_size, 
                                power=0.8, alpha=0.05, nsim=1000, 
                                min_n=2, max_n=100, seed=None):
    """
    Find required sample size for a cluster RCT with continuous outcome using simulation.
    
    Parameters
    ----------
    mean1 : float
        Mean outcome in control arm
    mean2 : float
        Mean outcome in intervention arm
    std_dev : float
        Total standard deviation of outcome
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
    
    # Calculate effect size (standardized mean difference)
    delta = abs(mean1 - mean2) / std_dev
    
    # Critical values for significance level and power
    z_alpha = stats.norm.ppf(1 - alpha/2)
    z_beta = stats.norm.ppf(power)
    
    # Calculate required effective sample size per arm
    n_eff = 2 * ((z_alpha + z_beta) / delta)**2
    
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
        sim_results = power_continuous_sim(
            n_clusters=mid, 
            cluster_size=cluster_size,
            icc=icc,
            mean1=mean1,
            mean2=mean2,
            std_dev=std_dev,
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
    final_results = power_continuous_sim(
        n_clusters=final_n_clusters, 
        cluster_size=cluster_size,
        icc=icc,
        mean1=mean1,
        mean2=mean2,
        std_dev=std_dev,
        nsim=nsim,
        alpha=alpha,
        seed=seed
    )
    
    # Extract the empirical power from final simulation
    achieved_power = final_results["power"]
    
    # Format results as dictionary
    results = {
        "n_clusters": final_n_clusters,
        "cluster_size": cluster_size,
        "total_n": 2 * final_n_clusters * cluster_size,
        "mean1": mean1,
        "mean2": mean2,
        "difference": abs(mean1 - mean2),
        "std_dev": std_dev,
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
            "sim_type": "cluster_continuous",
            "analytical_estimate": n_clusters_estimate
        }
    }
    
    return results


def min_detectable_effect_continuous_sim(n_clusters, cluster_size, icc, std_dev,
                                         power=0.8, alpha=0.05, nsim=1000,
                                         precision=0.01, max_iterations=10,
                                         seed=None):
    """
    Calculate minimum detectable effect for a cluster RCT with continuous outcome using simulation.
    
    Parameters
    ----------
    n_clusters : int
        Number of clusters per arm
    cluster_size : int
        Number of individuals per cluster
    icc : float
        Intracluster correlation coefficient
    std_dev : float
        Total standard deviation of outcome
    power : float, optional
        Desired power, by default 0.8
    alpha : float, optional
        Significance level, by default 0.05
    nsim : int, optional
        Number of simulations per effect size, by default 1000
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
    
    # Calculate standard error
    se = math.sqrt(2 / n_eff)
    
    # Calculate minimum detectable standardized effect size
    delta_estimate = (z_alpha + z_beta) * se
    
    # Convert standardized effect size to raw difference
    mde_estimate = delta_estimate * std_dev
    
    # Use binary search to find the minimum detectable effect
    low = max(mde_estimate / 2, precision)  # Start with half the analytical estimate
    high = mde_estimate * 2  # Double the analytical estimate as upper bound
    
    print(f"Starting binary search with effect size between {low:.4f} and {high:.4f}")
    
    iteration = 0
    min_adequate_effect = high
    
    while iteration < max_iterations and high - low > precision:
        mid = (low + high) / 2
        print(f"Iteration {iteration + 1}: Testing effect size = {mid:.4f}...")
        
        # Define means for the simulation (arbitrarily set mean1=0 and mean2=effect)
        mean1 = 0
        mean2 = mid
        
        # Run simulation with current effect size
        sim_results = power_continuous_sim(
            n_clusters=n_clusters,
            cluster_size=cluster_size,
            icc=icc,
            mean1=mean1,
            mean2=mean2,
            std_dev=std_dev,
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
    
    # Run final simulation to get accurate power estimate
    final_results = power_continuous_sim(
        n_clusters=n_clusters,
        cluster_size=cluster_size,
        icc=icc,
        mean1=0,
        mean2=final_effect,
        std_dev=std_dev,
        nsim=nsim,
        alpha=alpha,
        seed=seed
    )
    
    # Extract the empirical power from final simulation
    achieved_power = final_results["power"]
    
    # Format results as dictionary
    results = {
        "mde": final_effect,
        "standardized_mde": final_effect / std_dev,
        "n_clusters": n_clusters,
        "cluster_size": cluster_size,
        "total_n": 2 * n_clusters * cluster_size,
        "icc": icc,
        "std_dev": std_dev,
        "design_effect": deff,
        "effective_n": n_eff,
        "alpha": alpha,
        "target_power": power,
        "achieved_power": achieved_power,
        "nsim": nsim,
        "iterations": iteration,
        "sim_details": {
            "method": "simulation",
            "sim_type": "cluster_continuous",
            "analytical_estimate": mde_estimate
        }
    }
    
    return results