"""
Power calculation functions for cluster randomized controlled trials with binary outcomes.

This module provides simulation-based power calculation functions using various analysis methods.
"""

import numpy as np
import warnings
from tqdm import tqdm
from collections import Counter

from .core_simulation import (
    simulate_binary_trial, _analyze_binary_deff_ztest, _analyze_binary_agg_ttest,
    _analyze_binary_permutation, _analyze_binary_glmm, _analyze_binary_gee, _analyze_binary_bayes
)
from ..cluster_utils import convert_effect_measures


def power_binary_sim(n_clusters, cluster_size, icc, p1, p2=None, nsim=1000, alpha=0.05, seed=None, 
                     cv_cluster_size=0, cluster_sizes=None, effect_measure=None, effect_value=None, 
                     analysis_method="deff_ztest", bayes_backend="stan", bayes_draws=500, 
                     bayes_warmup=500, bayes_inference_method="credible_interval", progress_callback=None):
    """
    Simulate a cluster RCT with binary outcome and estimate power.
    
    Parameters
    ----------
    n_clusters : int
        Number of clusters per arm
    cluster_size : int or float
        Average number of individuals per cluster
    icc : float
        Intracluster correlation coefficient. Must be >= 0 for DEFF calculation.
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
    analysis_method : str, optional
        The statistical method to use for analysis in each simulation.
        Options: "deff_ztest", "aggregate_ttest", "permutation", "glmm", "gee", "bayes". 
        Default is "deff_ztest".
    bayes_backend : str, optional
        Bayesian backend to use when analysis_method="bayes".
        Options: "stan", "pymc", "variational", "abc". Default is "stan".
    bayes_draws : int, optional
        Number of posterior draws for Bayesian analysis. Default is 500.
    bayes_warmup : int, optional
        Number of warmup iterations for Bayesian analysis. Default is 500.
    bayes_inference_method : str, optional
        Method for Bayesian significance testing. Options: "credible_interval", 
        "posterior_probability", "rope". Default is "credible_interval".
    progress_callback : function, optional
        A function to call with progress updates during simulation.
    
    Returns
    -------
    dict
        Dictionary containing the estimated power, simulation details, and fit statistics.
    """
    if p2 is None and effect_measure is not None and effect_value is not None:
        p1_float = float(p1)
        effect_info = convert_effect_measures(p1_float, effect_measure, effect_value, icc_on_logit_scale=False)
        p2 = effect_info['p2']
    elif p2 is None:
        raise ValueError("Either p2 or both effect_measure and effect_value must be provided")

    if cluster_sizes is not None and len(cluster_sizes) > 0:
        mean_cs = np.mean(cluster_sizes)
        if mean_cs > 0: cv_cluster_size = np.std(cluster_sizes) / mean_cs
    
    if seed is not None:
        np.random.seed(seed)

    if icc < 0:
        warnings.warn(f"ICC value {icc} is negative. For DEFF calculation, ICC will be treated as 0. Data generation will treat it as no clustering.", UserWarning)
        icc_for_deff = 0.0
    else:
        icc_for_deff = icc

    significant_results = 0
    p_values_list = [] 
    fit_statuses = Counter()

    if n_clusters < 2 :
        warnings.warn("Number of clusters per arm is less than 2. Results may be unreliable.", UserWarning)

    if progress_callback is None:
        iterator = tqdm(range(nsim), desc=f"Simulating trials ({analysis_method}) for power (binary)")
    else:
        iterator = range(nsim)
    
    for i in iterator:
        df_trial = simulate_binary_trial(
            n_clusters=n_clusters, 
            cluster_size=cluster_size, 
            icc=icc, 
            p1=p1, 
            p2=p2, 
            cluster_sizes=cluster_sizes, 
            cv_cluster_size=cv_cluster_size
        )

        analysis_result = {'p_value': 1.0, 'fit_status': 'not_analyzed'}
        if analysis_method == "deff_ztest":
            analysis_result = _analyze_binary_deff_ztest(df_trial, icc_for_deff)
        elif analysis_method == "aggregate_ttest":
            analysis_result = _analyze_binary_agg_ttest(df_trial)
        elif analysis_method == "permutation":
            analysis_result = _analyze_binary_permutation(df_trial)
        elif analysis_method == "glmm":
            analysis_result = _analyze_binary_glmm(df_trial)
        elif analysis_method == "gee":
            analysis_result = _analyze_binary_gee(df_trial)
        elif analysis_method == "bayes":
            analysis_result = _analyze_binary_bayes(
                df_trial, 
                bayes_backend=bayes_backend,
                bayes_draws=bayes_draws,
                bayes_warmup=bayes_warmup,
                bayes_inference_method=bayes_inference_method
            )
        else:
            # This case should ideally not be reached if UI and calculation mapping are correct
            # but serves as a fallback for unknown methods.
            fit_statuses['error_unknown_method'] += 1
            warnings.warn(f"Unknown analysis_method '{analysis_method}' encountered in power_binary_sim. Skipping trial.", UserWarning)
            continue 

        p_val = analysis_result['p_value']
        fit_status = analysis_result['fit_status']
        
        p_values_list.append(p_val)
        fit_statuses[fit_status] += 1

        if analysis_method == "deff_ztest":
            acceptable_statuses_for_power = ['success', 'success_no_variance_in_pooled_outcome', 'success_zero_se_after_deff']
        elif analysis_method == "aggregate_ttest":
            acceptable_statuses_for_power = ['success', 'success_novar_means_equal', 'success_novar_means_diff']
        elif analysis_method == "glmm":
            acceptable_statuses_for_power = ['success', 'success_cluster_robust']
        elif analysis_method == "gee":
            acceptable_statuses_for_power = ['success', 'success_cluster_robust_fallback']
        elif analysis_method == "bayes":
            acceptable_statuses_for_power = ['success']  # Simplified for modular version
        elif analysis_method == "permutation":
            acceptable_statuses_for_power = ['success']
        else: # Should not be reached if method is validated
            acceptable_statuses_for_power = ['success'] # Fallback
        
        if fit_status in acceptable_statuses_for_power and p_val < alpha:
            significant_results += 1
        
        # Update progress if callback provided
        if progress_callback and ((i + 1) % max(1, nsim // 100) == 0 or (i + 1) == nsim):
            progress_callback(i + 1, nsim)
            
    num_valid_sims_for_power = sum(count for status, count in fit_statuses.items() if status in acceptable_statuses_for_power)

    power = significant_results / num_valid_sims_for_power if num_valid_sims_for_power > 0 else 0.0
    
    return {
        "power": power,
        "p1": p1,
        "p2": p2,
        "n_clusters_per_arm": n_clusters,
        "cluster_size_avg_input": cluster_size, 
        "cv_cluster_size_input": cv_cluster_size, 
        "icc_input": icc, 
        "alpha": alpha,
        "nsim_run": nsim,
        "analysis_method": analysis_method,
        "significant_results_count": significant_results,
        "p_values_sample": p_values_list[:min(10, len(p_values_list))], 
        "fit_statuses": dict(fit_statuses),
        "num_valid_sims_for_power": num_valid_sims_for_power
    }