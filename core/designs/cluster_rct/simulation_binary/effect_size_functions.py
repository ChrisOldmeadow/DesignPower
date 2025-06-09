"""
Minimum detectable effect calculation functions for cluster randomized controlled trials with binary outcomes.

This module provides simulation-based minimum detectable effect calculation using binary search.
"""

import numpy as np
import math
from scipy import stats
from tqdm import tqdm

from .power_functions import power_binary_sim
from ..cluster_utils import design_effect_equal, design_effect_unequal, validate_cluster_parameters


def min_detectable_effect_binary_sim(n_clusters, cluster_size, icc, p1,
                                      power=0.8, alpha=0.05, nsim=1000,
                                      min_effect=0.01, max_effect=0.5,
                                      precision=0.01, max_iterations=10,
                                      seed=None, cv_cluster_size=0, cluster_sizes=None,
                                      effect_measure='risk_difference',
                                      analysis_method="deff_ztest", bayes_backend="stan", 
                                      bayes_draws=500, bayes_warmup=500, 
                                      bayes_inference_method="credible_interval", progress_callback=None):
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
    analysis_method : str, optional
        Analysis method to use, by default "deff_ztest"
    bayes_backend : str, optional
        Bayesian backend if using Bayesian analysis, by default "stan"
    bayes_draws : int, optional
        Number of Bayesian draws, by default 500
    bayes_warmup : int, optional
        Number of Bayesian warmup iterations, by default 500
    bayes_inference_method : str, optional
        Bayesian inference method, by default "credible_interval"
    progress_callback : function, optional
        A function to call with progress updates during simulation.
    
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
    
    # Create progress bar for binary search
    pbar = tqdm(total=max_iterations, desc="Binary search for MDE (binary sim)", disable=max_iterations < 5)
    
    while iteration < max_iterations and high - low > precision:
        pbar.update(1)
        mid = (low + high) / 2
        print(f"Iteration {iteration + 1}: Testing effect size = {mid:.4f}...")
        
        # Calculate p2 based on effect size
        p2_current = p1 + mid
        
        if p2_current > 0.9999:  # Avoid numerical issues near 1
            p2_current = 0.9999
        
        # Run simulation with current effect size
        sim_results = power_binary_sim(
            analysis_method=analysis_method, 
            n_clusters=n_clusters, 
            cluster_size=cluster_size,
            icc=icc,
            p1=p1,
            p2=p2_current,
            nsim=nsim,
            alpha=alpha,
            seed=seed,
            cv_cluster_size=cv_cluster_size,
            cluster_sizes=cluster_sizes,
            bayes_backend=bayes_backend,
            bayes_draws=bayes_draws,
            bayes_warmup=bayes_warmup,
            bayes_inference_method=bayes_inference_method,
            progress_callback=progress_callback
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
    
    pbar.close()
    
    # Use the minimum effect size that meets the power requirement
    final_effect = min_adequate_effect
    final_p2 = min(p1 + final_effect, 0.9999)
    
    # Run final simulation to get accurate power estimate
    final_results = power_binary_sim(
        analysis_method=analysis_method, 
        n_clusters=n_clusters,
        cluster_size=cluster_size,
        icc=icc,
        p1=p1,
        p2=final_p2,
        nsim=nsim,
        alpha=alpha,
        seed=seed,
        cv_cluster_size=cv_cluster_size,
        cluster_sizes=cluster_sizes,
        bayes_backend=bayes_backend,
        bayes_draws=bayes_draws,
        bayes_warmup=bayes_warmup,
        bayes_inference_method=bayes_inference_method,
        progress_callback=progress_callback
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
        "analysis_method": analysis_method,
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