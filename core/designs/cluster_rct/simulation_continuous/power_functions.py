"""
Power calculation functions for cluster randomized controlled trials with continuous outcomes.

This module provides simulation-based power calculation functions using various analysis methods.
"""

import numpy as np
import warnings
from tqdm import tqdm

from .core_simulation import simulate_continuous_trial
from .bayesian_methods import _STAN_AVAILABLE, _PYMC_AVAILABLE


def power_continuous_sim(
    n_clusters,
    cluster_size,
    icc,
    mean1,
    mean2,
    std_dev,
    nsim=1000,
    alpha=0.05,
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
    lmm_cov_penalty_weight: float = 0.0,
    lmm_fit_kwargs: dict = None,
    progress_callback=None,
):
    """
    Simulate a cluster RCT with continuous outcome and estimate power using the
    specified analysis model.
    
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
        Number of simulations to run, by default 1000
    alpha : float, optional
        Significance level, by default 0.05
    seed : int, optional
        Random seed for reproducibility, by default None
    analysis_model : str, optional
        Analysis method ("ttest", "mixedlm", "gee", "bayes"), by default "ttest"
    use_satterthwaite : bool, optional
        Use Satterthwaite degrees of freedom correction for mixedlm, by default False
    use_bias_correction : bool, optional
        Use bias-reduced sandwich covariance for GEE, by default False
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
    lmm_cov_penalty_weight : float, optional
        L2 penalty weight for MixedLM covariance, by default 0.0
    lmm_fit_kwargs : dict, optional
        Additional kwargs for MixedLM.fit(), by default None
    progress_callback : callable, optional
        Progress callback function, by default None
        
    Returns
    -------
    dict
        Dictionary containing power estimate and simulation details
    """
    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)
    
    # Initialize counters
    sig_count = 0
    p_values = []
    
    # LMM specific counters
    lmm_success_count = 0
    lmm_convergence_warnings_count = 0
    lmm_success_boundary_ols_fallback_count = 0 # Boundary condition, used OLS fallback
    lmm_ols_fallbacks_count = 0 # Covers fit_error_ols_fallback and fit_error_ols_fallback_invalid_stats
    lmm_ttest_fallbacks_outer_count = 0
    lmm_total_considered_for_power = 0

    if progress_callback is None:
        iterator = tqdm(range(nsim), desc="Running simulations", disable=False)
    else:
        iterator = range(nsim)

    # Suppress RuntimeWarnings from this module during simulations
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning, module=simulate_continuous_trial.__module__)
        for i in iterator:
            if analysis_model == "mixedlm":
                _, p_value, details = simulate_continuous_trial(
                    n_clusters, cluster_size, icc, mean1, mean2, std_dev,
                    analysis_model=analysis_model, use_satterthwaite=use_satterthwaite,
                    use_bias_correction=use_bias_correction, bayes_draws=bayes_draws,
                    bayes_warmup=bayes_warmup, bayes_inference_method=bayes_inference_method,
                    bayes_backend=bayes_backend,
                    lmm_method=lmm_method, lmm_reml=lmm_reml,
                    lmm_cov_penalty_weight=lmm_cov_penalty_weight, lmm_fit_kwargs=lmm_fit_kwargs,
                    return_details=True
                )
                status = details.get("status", "unknown")
                
                # Increment status counters
                if status == "success":
                    lmm_success_count += 1
                elif status == "success_convergence_warning":
                    lmm_convergence_warnings_count += 1
                elif status == "success_boundary_ols_fallback":
                    lmm_success_boundary_ols_fallback_count += 1
                elif status in ["fit_error_ols_fallback", "fit_error_ols_fallback_invalid_stats"]:
                    lmm_ols_fallbacks_count += 1
                elif status == "fit_error_ttest_fallback_outer":
                    lmm_ttest_fallbacks_outer_count += 1
                
                # Collect p-value and count significance if the result is considered usable for power
                # Exclude 'fit_error_ttest_fallback_outer' from primary power calculation for LMM
                if status not in ["fit_error_ttest_fallback_outer", "unknown"] and not np.isnan(p_value):
                    p_values.append(p_value)
                    lmm_total_considered_for_power += 1
                    if p_value < alpha:
                        sig_count += 1
                elif np.isnan(p_value) and status not in ["fit_error_ttest_fallback_outer", "unknown"]:
                     # Still count it as considered if it was supposed to be, but p_value was NaN from a fallback
                     lmm_total_considered_for_power += 1 

            else: # For 'ttest', 'gee', 'bayes'
                # These models have simpler or self-contained fallback logic in simulate_continuous_trial
                _, p_value, details = simulate_continuous_trial(
                    n_clusters, cluster_size, icc, mean1, mean2, std_dev,
                    analysis_model=analysis_model, use_satterthwaite=use_satterthwaite,
                    use_bias_correction=use_bias_correction, bayes_draws=bayes_draws,
                    bayes_warmup=bayes_warmup, bayes_inference_method=bayes_inference_method,
                    bayes_backend=bayes_backend,
                    lmm_method=lmm_method, lmm_reml=lmm_reml,
                    lmm_cov_penalty_weight=lmm_cov_penalty_weight, lmm_fit_kwargs=lmm_fit_kwargs,
                    return_details=True # Get details for GEE/Bayes fallbacks if any
                )
                # For non-LMM, if p_value is valid, count it.
                # The 'status' for GEE/Bayes might indicate fallback to t-test, which is fine.
                if not np.isnan(p_value):
                    p_values.append(p_value)
                    if p_value < alpha:
                        sig_count += 1
            
            if progress_callback and ((i + 1) % max(1, nsim // 100) == 0 or (i + 1) == nsim):
                progress_callback(i + 1, nsim)

    # Calculate design effect
    deff = 1 + (cluster_size - 1) * icc if cluster_size > 0 else 1.0
    
    # Calculate effective sample size
    n_eff = n_clusters * cluster_size / deff if deff > 0 else 0
    
    # Calculate power
    # For LMM, power is based on simulations that were successfully processed through LMM path or valid fallbacks.
    # For other models, it's based on total simulations (nsim).
    if analysis_model == "mixedlm":
        power = sig_count / lmm_total_considered_for_power if lmm_total_considered_for_power > 0 else 0
    elif nsim > 0:
        # This branch covers 'ttest', 'gee', 'bayes'. 
        # For these, simulate_continuous_trial handles its own fallbacks (usually to ttest), 
        # and all nsim are generally considered unless p_value is NaN.
        # The current p_values list for non-LMM models includes all non-NaN p-values.
        # If a non-LMM model consistently produced NaNs, len(p_values) would be < nsim.
        # Power should be based on valid p-values obtained.
        power = sig_count / len(p_values) if len(p_values) > 0 else 0
    else:
        power = 0

    # Base results dictionary
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
        "alpha": alpha,
        "nsim": nsim,
        "significant_sims": sig_count,
        "p_values_mean": np.mean(p_values) if p_values else np.nan,
        "p_values_median": np.median(p_values) if p_values else np.nan,
        "analysis_model": analysis_model,
        "use_satterthwaite": use_satterthwaite if analysis_model == "mixedlm" else None,
        "use_bias_correction": use_bias_correction if analysis_model == "gee" else None,
        "lmm_method": lmm_method if analysis_model == "mixedlm" else None,
        "lmm_reml": lmm_reml if analysis_model == "mixedlm" else None,
        "bayes_draws": bayes_draws if analysis_model == "bayes" else None,
        "bayes_warmup": bayes_warmup if analysis_model == "bayes" else None,
    }

    # Add LMM specific details if applicable
    if analysis_model == "mixedlm":
        # Calculate total fits that did not contribute to power due to LMM path issues (e.g. ttest_fallback_outer or other unhandled NaN p-values)
        # This is essentially nsim minus those that *were* considered.
        total_not_converged_or_failed_lmm_path = nsim - lmm_total_considered_for_power

        results["lmm_fit_stats"] = {
            "successful_fits": lmm_success_count,
            "convergence_warnings": lmm_convergence_warnings_count,
            "success_boundary_ols_fallbacks": lmm_success_boundary_ols_fallback_count, # Boundary, used OLS fallback
            "ols_fallbacks_errors": lmm_ols_fallbacks_count, # covers fit_error_ols_fallback and fit_error_ols_fallback_invalid_stats
            "ttest_fallbacks_outer_errors": lmm_ttest_fallbacks_outer_count,
            "total_considered_for_power": lmm_total_considered_for_power,
            "total_not_converged_or_failed_lmm_path": total_not_converged_or_failed_lmm_path
        }# For backward compatibility / simplicity in other reports, provide a general converged_sims
        results["converged_sims"] = lmm_total_considered_for_power
        results["failed_sims"] = nsim - lmm_total_considered_for_power
    else:
        # For other models, assume all simulations that produced a p_value are 'converged'
        results["converged_sims"] = len(p_values)
        results["failed_sims"] = nsim - len(p_values)

    # Sim details sub-dictionary (can be expanded)
    results["sim_details"] = {
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
        "pymc_installed": _PYMC_AVAILABLE,
    }
    if analysis_model == "mixedlm":
        results["sim_details"]["lmm_fit_stats"] = results["lmm_fit_stats"]

    return results