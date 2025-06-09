"""
Sample size calculation functions for cluster randomized controlled trials with binary outcomes.

This module provides simulation-based sample size calculation functions using binary search.
"""

import numpy as np
import math
from scipy import stats
from tqdm import tqdm

from .power_functions import power_binary_sim
from ..cluster_utils import design_effect_equal, design_effect_unequal, validate_cluster_parameters, convert_effect_measures


def sample_size_binary_sim(p1, p2=None, icc=0.01, cluster_size=50, 
                            power=0.8, alpha=0.05, nsim=1000, 
                            min_n=2, max_n=100, seed=None,
                            cv_cluster_size=0, cluster_sizes=None,
                            effect_measure=None, effect_value=None,
                            analysis_method="deff_ztest", bayes_backend="stan", 
                            bayes_draws=500, bayes_warmup=500, 
                            bayes_inference_method="credible_interval", progress_callback=None):
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
        Dictionary containing the required number of clusters and simulation details
    """
    # Determine p2 and populate effect_info dictionary
    if p2 is None: # If p2 is not provided, calculate it from effect measure
        if effect_measure is not None and effect_value is not None:
            effect_info = convert_effect_measures(p1=p1, measure_type=effect_measure, measure_value=effect_value)
            p2 = effect_info.get('p2') # p2 is now defined from effect_info
            if p2 is None: # Should not happen if convert_effect_measures is robust
                 raise ValueError("p2 could not be derived from effect measure and value.")
        else:
            # p2 is None and no way to calculate it
            raise ValueError("Either p2 or both effect_measure and effect_value must be provided if p2 is None")
    else: # p2 was provided directly
        # Calculate other effect measures based on the given p1 and p2
        risk_difference_val = p2 - p1
        risk_ratio_val = p2 / p1 if p1 != 0 else float('inf')
        # Handle cases for odds_ratio to prevent division by zero or issues with p1/p2 at 0 or 1
        if p1 == 0 or p1 == 1 or p2 == 0 or p2 == 1:
            if p1 == p2: # e.g. p1=0, p2=0 or p1=1, p2=1
                odds_ratio_val = 1.0 # Or handle as undefined/NaN if preferred
            else: # e.g. p1=0, p2=0.1 or p1=0.9, p2=1
                odds_ratio_val = float('inf') # Or handle as undefined/NaN
        else:
            odds_ratio_val = (p2 / (1 - p2)) / (p1 / (1 - p1))
        
        effect_info = {
            'p1': p1,
            'p2': p2,
            'risk_difference': risk_difference_val,
            'risk_ratio': risk_ratio_val,
            'odds_ratio': odds_ratio_val
        }
    
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
    
    # Create progress bar for binary search
    search_iterations = int(np.log2(high - low + 1)) + 1  # Approximate max iterations for binary search
    pbar = tqdm(total=search_iterations, desc="Binary search for sample size (binary sim)", disable=search_iterations < 5)
    
    while low <= high:
        pbar.update(1)
        mid = (low + high) // 2
        print(f"Testing n_clusters = {mid}...")
        
        # Run simulation with current n_clusters
        sim_results = power_binary_sim(
            analysis_method=analysis_method, 
            n_clusters=mid, 
            cluster_size=cluster_size,
            icc=icc,
            p1=p1,
            p2=p2,
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
            # This n_clusters is sufficient, try smaller
            min_adequate_n = min(min_adequate_n, mid)
            high = mid - 1
        else:
            # This n_clusters is insufficient, try larger
            low = mid + 1
    
    pbar.close()
    
    # Get final power for the optimal n_clusters
    final_results = power_binary_sim(
        analysis_method=analysis_method, 
        n_clusters=min_adequate_n, 
        cluster_size=cluster_size,
        icc=icc,
        p1=p1,
        p2=p2,
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
        "risk_difference": effect_info.get('risk_difference', abs(p2 - p1)), # Fallback for risk_difference
        "risk_ratio": effect_info.get('risk_ratio'),
        "odds_ratio": effect_info.get('odds_ratio'),
        "design_effect": deff,
        "cv_cluster_size": cv_cluster_size,
        "alpha": alpha,
        "nsim": nsim,
        "analysis_method": analysis_method,
        "sim_details": {
            "method": "simulation",
            "iterations": final_results["nsim_run"],
            "search_range": [min_n, max_n]
        }
    }
    
    # Add validation warnings
    if not validation["valid"]:
        results["warnings"] = validation["warnings"]
    
    return results