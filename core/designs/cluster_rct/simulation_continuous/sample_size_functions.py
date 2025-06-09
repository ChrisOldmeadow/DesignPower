"""
Sample size calculation functions for cluster randomized controlled trials with continuous outcomes.

This module provides simulation-based sample size calculation functions using binary search.
"""

import numpy as np
import math
from scipy import stats
from tqdm import tqdm

from .power_functions import power_continuous_sim


def sample_size_continuous_sim(
    mean1,
    mean2,
    std_dev,
    icc,
    power=0.8,
    alpha=0.05,
    nsim=1000,
    seed=None,
    cluster_size=None, 
    n_clusters_fixed=None, 
    min_n_clusters=2, 
    max_n_clusters=100, 
    min_cluster_size=2, 
    max_cluster_size=500, 
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
    lmm_cov_penalty_weight: float = 0.0,
    lmm_fit_kwargs: dict = None,
    progress_callback=None,
):
    """
    Find required sample size (number of clusters or cluster size) for a cluster RCT 
    with continuous outcome using simulation.

    Exactly one of 'cluster_size' (to solve for number of clusters) or 
    'n_clusters_fixed' (to solve for cluster size) must be specified.

    Parameters
    ----------
    mean1 : float
        Mean outcome in control group.
    mean2 : float
        Mean outcome in intervention group.
    std_dev : float
        Pooled standard deviation of the outcome.
    icc : float
        Intracluster correlation coefficient.
    power : float, optional
        Desired statistical power (1 - beta), by default 0.8.
    alpha : float, optional
        Significance level, by default 0.05.
    nsim : int, optional
        Number of simulations to run for each power estimation, by default 1000.
    seed : int, optional
        Random seed for reproducibility, by default None.
    cluster_size : int, optional
        Average number of individuals per cluster. Provide to calculate n_clusters.
    n_clusters_fixed : int, optional
        Number of clusters per arm. Provide to calculate cluster_size.
    min_n_clusters : int, optional
        Minimum number of clusters per arm for search range, by default 2.
    max_n_clusters : int, optional
        Maximum number of clusters per arm for search range, by default 100.
    min_cluster_size : int, optional
        Minimum average cluster size for search range, by default 2.
    max_cluster_size : int, optional
        Maximum average cluster size for search range, by default 500.
    analysis_model : str, optional
        The statistical model to use for analysis in simulations ('ttest', 'mixedlm', 'gee', 'bayes').
    use_satterthwaite : bool, optional
        Whether to use Satterthwaite approximation for mixed models, by default False.
    use_bias_correction : bool, optional
        Whether to use bias correction for GEE models, by default False.
    bayes_draws : int, optional
        Number of posterior draws for Bayesian models, by default 500.
    bayes_warmup : int, optional
        Number of warmup/burn-in draws for Bayesian models, by default 500.
    bayes_inference_method : str, optional
        Bayesian inference method, by default "credible_interval"
    bayes_backend : str, optional
        Bayesian backend to use, by default "stan"
    lmm_method : str, optional
        Optimization method for LMM, by default "auto".
    lmm_reml : bool, optional
        Whether to use REML for LMM, by default True.
    lmm_cov_penalty_weight : float, optional
        Weight for L2 penalty on covariance structure for MixedLM. Default 0.0 (no penalty).
    lmm_fit_kwargs : dict, optional
        Additional keyword arguments to pass to MixedLM.fit() (e.g., {'gtol': 1e-8}). Default None.
    progress_callback : function, optional
        A function to call with progress updates during simulation.

    Returns
    -------
    dict
        Dictionary containing the determined sample size parameters (n_clusters, cluster_size),
        total sample size, achieved power, design effect, and simulation details.
        May include a 'warning' key.
    """

    if not ( (cluster_size is not None and n_clusters_fixed is None) or \
             (cluster_size is None and n_clusters_fixed is not None) ):
        raise ValueError("Exactly one of 'cluster_size' (to solve for n_clusters) or 'n_clusters_fixed' (to solve for cluster_size) must be specified.")

    if icc < 0 or icc > 1: raise ValueError("ICC must be between 0 and 1.")
    if power <= 0 or power >= 1: raise ValueError("Power must be between 0 and 1 (exclusive).")
    if alpha <= 0 or alpha >= 1: raise ValueError("Alpha must be between 0 and 1 (exclusive).")
    if std_dev <= 0: raise ValueError("Standard deviation must be positive.")
    if nsim < 1: raise ValueError("Number of simulations (nsim) must be at least 1.")

    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)
    
    # Calculate effect size (standardized mean difference)
    delta = abs(mean1 - mean2) / std_dev
    if delta == 0:
        return {
            "n_clusters": float('inf'), "cluster_size": float('inf'), "total_n": float('inf'),
            "mean1": mean1, "mean2": mean2, "difference": 0, "std_dev": std_dev, "icc": icc,
            "design_effect": float('inf'), "alpha": alpha, "target_power": power, "achieved_power": 0.0,
            "nsim": nsim, "analysis_model": analysis_model,
            "warning": "Mean outcomes are identical (delta=0), cannot simulate sample size for a difference."
        }

    # Critical values for significance level and power (used for analytical estimate if any)
    z_alpha = stats.norm.ppf(1 - alpha/2)
    z_beta = stats.norm.ppf(power)
    n_eff_analytical = 2 * ((z_alpha + z_beta) / delta)**2 # Effective sample size per arm from analytical formula

    final_n_clusters_result = None
    final_cluster_size_result = None
    warning_message = None

    # --- Scenario 1: Solve for Number of Clusters (k) given Cluster Size (m) --- 
    if cluster_size is not None:
        if not isinstance(cluster_size, int) or cluster_size < min_cluster_size:
            raise ValueError(f"Input cluster_size must be an integer >= {min_cluster_size}.")
        final_cluster_size_result = cluster_size

        # Analytical estimate for n_clusters to guide search range
        deff_analytical = 1 + (final_cluster_size_result - 1) * icc
        n_clusters_estimate_analytical = max(min_n_clusters, math.ceil(n_eff_analytical * deff_analytical / final_cluster_size_result))
        n_clusters_if_icc0 = math.ceil(n_eff_analytical / final_cluster_size_result) if final_cluster_size_result > 0 else min_n_clusters
        low_k = max(min_n_clusters, n_clusters_if_icc0)
        # Adjust high_k based on analytical estimate, but ensure it's within max_n_clusters
        high_k = min(max(n_clusters_estimate_analytical * 2, low_k + 10), max_n_clusters) 
        if high_k < low_k: high_k = low_k # Ensure high is not less than low

        min_adequate_k = high_k + 1 # Initialize to a value outside search range, indicating not found

        print(f"Simulating to find n_clusters (k) with fixed m={final_cluster_size_result}. Search range k: [{low_k}-{high_k}]")
        current_low_k, current_high_k = low_k, high_k
        iterations_k = 0
        max_iterations_k = 30 # Max iterations for k search
        
        # Create progress bar for binary search
        pbar_k = tqdm(total=max_iterations_k, desc="Binary search for cluster count (continuous sim)", disable=max_iterations_k < 5)
        
        while current_low_k <= current_high_k:
            if iterations_k >= max_iterations_k:
                warning_message = (
                    f"Binary search for n_clusters (k) exceeded maximum iterations ({max_iterations_k}). "
                    f"Using k={min_adequate_k if min_adequate_k <= max_n_clusters else max_n_clusters} based on current findings."
                )
                print(f"Warning: {warning_message}")
                break
            iterations_k += 1
            pbar_k.update(1)
            mid_k = (current_low_k + current_high_k) // 2
            if mid_k < min_n_clusters: mid_k = min_n_clusters # Ensure mid_k is not below min
            if mid_k == 0: # Avoid n_clusters = 0
                current_low_k = 1
                continue

            print(f"  Testing k = {mid_k}, m = {final_cluster_size_result}...")
            sim_results = power_continuous_sim(
                n_clusters=mid_k, cluster_size=final_cluster_size_result, icc=icc,
                mean1=mean1, mean2=mean2, std_dev=std_dev, nsim=nsim, alpha=alpha, seed=seed,
                analysis_model=analysis_model, use_satterthwaite=use_satterthwaite,
                use_bias_correction=use_bias_correction, bayes_draws=bayes_draws,
                bayes_warmup=bayes_warmup, bayes_inference_method=bayes_inference_method,
                bayes_backend=bayes_backend,
                lmm_method=lmm_method, lmm_reml=lmm_reml,
                lmm_cov_penalty_weight=lmm_cov_penalty_weight, lmm_fit_kwargs=lmm_fit_kwargs,
                progress_callback=progress_callback
            )
            empirical_power = sim_results["power"]
            print(f"  Achieved power for k={mid_k}, m={final_cluster_size_result}: {empirical_power:.4f} (Target: {power:.4f})")
            
            if empirical_power >= power:
                min_adequate_k = min(min_adequate_k, mid_k)
                current_high_k = mid_k - 1
            else:
                current_low_k = mid_k + 1
        
        pbar_k.close()
        
        if min_adequate_k > max_n_clusters and min_adequate_k == high_k + 1: # Initial value not updated
             # This means power was not achieved even at high_k. Use high_k for final sim to report achieved power.
            final_n_clusters_result = max_n_clusters 
            warning_message = f"Target power {power:.2f} may not be achievable with m={final_cluster_size_result} within k=[{min_n_clusters}-{max_n_clusters}]. Reporting for k={final_n_clusters_result}."
        elif min_adequate_k == high_k + 1: # Still initial value, but high_k might have been small
            final_n_clusters_result = max_n_clusters # Default to max if nothing found
            warning_message = f"Target power {power:.2f} not achieved with m={final_cluster_size_result} up to k={max_n_clusters}. Consider increasing max_n_clusters or check parameters."
        else:
            final_n_clusters_result = min_adequate_k

    # --- Scenario 2: Solve for Cluster Size (m) given Number of Clusters (k) --- 
    elif n_clusters_fixed is not None:
        if not isinstance(n_clusters_fixed, int) or n_clusters_fixed < min_n_clusters:
            raise ValueError(f"Input n_clusters_fixed must be an integer >= {min_n_clusters}.")
        final_n_clusters_result = n_clusters_fixed
        m_if_icc0 = math.ceil(n_eff_analytical / final_n_clusters_result) if final_n_clusters_result > 0 else min_cluster_size
        low_m = max(min_cluster_size, m_if_icc0)
        high_m = max_cluster_size
        min_adequate_m = high_m + 1 # Initialize to a value outside search range

        print(f"Simulating to find cluster_size (m) with fixed k={final_n_clusters_result}. Search range m: [{low_m}-{high_m}]")
        current_low_m, current_high_m = low_m, high_m
        iterations_m = 0
        max_iterations_m = 30 # Max iterations for m search
        
        # Create progress bar for binary search
        pbar_m = tqdm(total=max_iterations_m, desc="Binary search for cluster size (continuous sim)", disable=max_iterations_m < 5)
        
        while current_low_m <= current_high_m:
            if iterations_m >= max_iterations_m:
                warning_message = (
                    f"Binary search for cluster_size (m) exceeded maximum iterations ({max_iterations_m}). "
                    f"Using m={min_adequate_m if min_adequate_m <= max_cluster_size else max_cluster_size} based on current findings."
                )
                print(f"Warning: {warning_message}")
                break
            iterations_m += 1
            pbar_m.update(1)
            mid_m = (current_low_m + current_high_m) // 2
            if mid_m < min_cluster_size: mid_m = min_cluster_size
            if mid_m == 0: # Avoid cluster_size = 0
                current_low_m = 1
                continue

            print(f"  Testing k = {final_n_clusters_result}, m = {mid_m}...")
            sim_results = power_continuous_sim(
                n_clusters=final_n_clusters_result, cluster_size=mid_m, icc=icc,
                mean1=mean1, mean2=mean2, std_dev=std_dev, nsim=nsim, alpha=alpha, seed=seed,
                analysis_model=analysis_model, use_satterthwaite=use_satterthwaite,
                use_bias_correction=use_bias_correction, bayes_draws=bayes_draws,
                bayes_warmup=bayes_warmup, bayes_inference_method=bayes_inference_method,
                bayes_backend=bayes_backend,
                lmm_method=lmm_method, lmm_reml=lmm_reml,
                lmm_cov_penalty_weight=lmm_cov_penalty_weight, lmm_fit_kwargs=lmm_fit_kwargs,
                progress_callback=progress_callback
            )
            empirical_power = sim_results["power"]
            print(f"  Achieved power for k={final_n_clusters_result}, m={mid_m}: {empirical_power:.4f} (Target: {power:.4f})")

            if empirical_power >= power:
                min_adequate_m = min(min_adequate_m, mid_m)
                current_high_m = mid_m - 1
            else:
                current_low_m = mid_m + 1
        
        pbar_m.close()
        
        if min_adequate_m > max_cluster_size and min_adequate_m == high_m + 1:
            final_cluster_size_result = max_cluster_size
            warning_message = f"Target power {power:.2f} may not be achievable with k={final_n_clusters_result} within m=[{min_cluster_size}-{max_cluster_size}]. Reporting for m={final_cluster_size_result}."
        elif min_adequate_m == high_m + 1:
            final_cluster_size_result = max_cluster_size
            warning_message = f"Target power {power:.2f} not achieved with k={final_n_clusters_result} up to m={max_cluster_size}. Consider increasing max_cluster_size or check parameters."
        else:
            final_cluster_size_result = min_adequate_m
    
    # Run final simulation with the determined parameters for accurate power and other metrics
    print(f"Running final simulation with k={final_n_clusters_result}, m={final_cluster_size_result}...")
    final_sim_output = power_continuous_sim(
        n_clusters=final_n_clusters_result, 
        cluster_size=final_cluster_size_result,
        icc=icc, mean1=mean1, mean2=mean2, std_dev=std_dev, nsim=nsim, alpha=alpha, seed=seed,
        analysis_model=analysis_model, use_satterthwaite=use_satterthwaite,
        use_bias_correction=use_bias_correction, bayes_draws=bayes_draws,
        bayes_warmup=bayes_warmup, bayes_inference_method=bayes_inference_method,
        bayes_backend=bayes_backend,
        lmm_method=lmm_method, lmm_reml=lmm_reml,
        lmm_cov_penalty_weight=lmm_cov_penalty_weight, lmm_fit_kwargs=lmm_fit_kwargs,
        progress_callback=progress_callback
    )
    
    achieved_power_final = final_sim_output["power"]
    final_deff = 1 + (final_cluster_size_result - 1) * icc if final_cluster_size_result > 0 else 1

    # Assemble results dictionary
    results = {
        "n_clusters": final_n_clusters_result,
        "cluster_size": final_cluster_size_result,
        "total_n": 2 * final_n_clusters_result * final_cluster_size_result,
        "icc": icc,
        "mean1": mean1,
        "mean2": mean2,
        "difference": abs(mean1 - mean2),
        "std_dev": std_dev,
        "design_effect": final_deff,
        "alpha": alpha,
        "target_power": power,
        "achieved_power": achieved_power_final,
        "nsim": nsim,
        # Include details from final_sim_output that might be useful
        "p_values_mean": final_sim_output.get("p_values_mean"),
        "p_values_median": final_sim_output.get("p_values_median"),
        "analysis_model": analysis_model,
        "use_satterthwaite": use_satterthwaite if analysis_model == "mixedlm" else None,
        "use_bias_correction": use_bias_correction if analysis_model == "gee" else None,
        "lmm_method": lmm_method if analysis_model == "mixedlm" else None,
        "lmm_reml": lmm_reml if analysis_model == "mixedlm" else None,
        "bayes_draws": bayes_draws if analysis_model == "bayes" else None,
        "bayes_warmup": bayes_warmup if analysis_model == "bayes" else None,
        "sim_details": final_sim_output.get("sim_details", {})
    }
    if warning_message:
        results["warning"] = warning_message
    
    return results