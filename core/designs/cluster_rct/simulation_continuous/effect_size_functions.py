"""
Minimum detectable effect calculation functions for cluster randomized controlled trials with continuous outcomes.

This module provides simulation-based minimum detectable effect calculation using binary search.
"""

import numpy as np
import math
from scipy import stats
from tqdm import tqdm

from .power_functions import power_continuous_sim
from .bayesian_methods import _STAN_AVAILABLE


def min_detectable_effect_continuous_sim(
    n_clusters,
    cluster_size,
    icc,
    std_dev,
    power=0.8,
    alpha=0.05,
    nsim=1000,
    precision=0.01,
    max_iterations=10,
    seed=None,
    *,
    analysis_model: str = "ttest",
    use_satterthwaite: bool = False,
    use_bias_correction: bool = False,
    bayes_draws: int = 500,
    bayes_warmup: int = 500,
    bayes_inference_method: str = "credible_interval",
    bayes_backend: str = "stan",
    lmm_method: str = "auto",
    lmm_reml: bool = True,
):
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
        Standard deviation of the outcome
    power : float, optional
        Desired statistical power, by default 0.8
    alpha : float, optional
        Significance level, by default 0.05
    nsim : int, optional
        Number of simulations to run, by default 1000
    precision : float, optional
        Precision for binary search convergence, by default 0.01
    max_iterations : int, optional
        Maximum iterations for binary search, by default 10
    seed : int, optional
        Random seed for reproducibility, by default None
    analysis_model : str, optional
        Analysis method to use, by default "ttest"
    use_satterthwaite : bool, optional
        Use Satterthwaite degrees of freedom for mixedlm, by default False
    use_bias_correction : bool, optional
        Use bias correction for GEE, by default False
    bayes_draws : int, optional
        Number of MCMC draws for Bayesian analysis, by default 500
    bayes_warmup : int, optional
        Number of warmup iterations for Bayesian analysis, by default 500
    bayes_inference_method : str, optional
        Bayesian inference method, by default "credible_interval"
    bayes_backend : str, optional
        Bayesian backend to use, by default "stan"
    lmm_method : str, optional
        Optimizer for MixedLM, by default "auto"
    lmm_reml : bool, optional
        Use REML for MixedLM, by default True
        
    Returns
    -------
    dict
        Dictionary containing the minimum detectable effect and related statistics
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
    
    # Create progress bar for binary search
    pbar = tqdm(total=max_iterations, desc="Binary search for MDE (continuous sim)", disable=max_iterations < 5)
    
    while iteration < max_iterations and high - low > precision:
        pbar.update(1)
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
            seed=seed,
            analysis_model=analysis_model,
            use_satterthwaite=use_satterthwaite,
            use_bias_correction=use_bias_correction,
            bayes_draws=bayes_draws,
            bayes_warmup=bayes_warmup,
            bayes_inference_method=bayes_inference_method,
            bayes_backend=bayes_backend,
            lmm_method=lmm_method,
            lmm_reml=lmm_reml,
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
        seed=seed,
        analysis_model=analysis_model,
        use_satterthwaite=use_satterthwaite,
        use_bias_correction=use_bias_correction,
        bayes_draws=bayes_draws,
        bayes_warmup=bayes_warmup,
        bayes_inference_method=bayes_inference_method,
        bayes_backend=bayes_backend,
        lmm_method=lmm_method,
        lmm_reml=lmm_reml,
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
        # Expose model choices directly for easier reporting
        "analysis_model": analysis_model,
        "use_satterthwaite": use_satterthwaite if analysis_model == "mixedlm" else None,
        "use_bias_correction": use_bias_correction if analysis_model == "gee" else None,
        "lmm_method": lmm_method if analysis_model == "mixedlm" else None,
        "lmm_reml": lmm_reml if analysis_model == "mixedlm" else None,
        "bayes_draws": bayes_draws if analysis_model == "bayes" else None,
        "bayes_warmup": bayes_warmup if analysis_model == "bayes" else None,
        "sim_details": {
            "method": "simulation",
            "sim_type": "cluster_continuous",
            "analysis_model": analysis_model,
            "use_satterthwaite": use_satterthwaite,
            "use_bias_correction": use_bias_correction,
            "bayes_draws": bayes_draws if analysis_model == "bayes" else None,
            "bayes_backend": bayes_backend if analysis_model == "bayes" else None,
            "lmm_method": lmm_method if analysis_model == "mixedlm" else None,
            "lmm_reml": lmm_reml if analysis_model == "mixedlm" else None,
            "stan_installed": _STAN_AVAILABLE,
        },
    }
    
    return results