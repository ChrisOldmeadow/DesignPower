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
import pandas as pd
from scipy import stats
from tqdm import tqdm
import math
from scipy.special import beta as beta_func
from .cluster_utils import (design_effect_equal, design_effect_unequal, 
                           validate_cluster_parameters, convert_effect_measures)
import warnings
from collections import Counter # For fit_statuses
# statsmodels imports will be added later when GLMM/GEE are implemented


def simulate_binary_trial(n_clusters, cluster_size, icc, p1, p2, cluster_sizes=None, cv_cluster_size=0):
    """
    Simulate individual-level data for a single cluster RCT with binary outcome 
    using the beta-binomial model.
    
    Parameters
    ----------
    n_clusters : int
        Number of clusters per arm
    cluster_size : int or float
        Average number of individuals per cluster (if cluster_sizes not provided).
        If cv_cluster_size is 0, this will be treated as int.
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
    pd.DataFrame
        DataFrame with columns: 'outcome' (0/1), 'treatment' (0/1), 'cluster_id'
    """
    # Generate cluster sizes if not provided
    if cluster_sizes is None:
        if cv_cluster_size > 0 and cluster_size > 0: # Ensure mean_size > 0 for gamma
            mean_size = float(cluster_size) # Ensure float for calculations
            variance = (cv_cluster_size * mean_size) ** 2
            if variance == 0: # Avoid division by zero if mean_size or cv is zero leading to zero variance
                 control_cluster_sizes = np.full(n_clusters, int(mean_size), dtype=int)
                 intervention_cluster_sizes = np.full(n_clusters, int(mean_size), dtype=int)
            else:
                shape = mean_size ** 2 / variance
                scale = variance / mean_size
                control_cluster_sizes = np.maximum(np.round(np.random.gamma(shape, scale, n_clusters)), 1).astype(int)
                intervention_cluster_sizes = np.maximum(np.round(np.random.gamma(shape, scale, n_clusters)), 1).astype(int)
        else:
            # Equal cluster sizes or cv_cluster_size is 0 or cluster_size is 0
            control_cluster_sizes = np.full(n_clusters, int(cluster_size), dtype=int)
            intervention_cluster_sizes = np.full(n_clusters, int(cluster_size), dtype=int)
    else:
        # Use provided cluster sizes
        num_provided = len(cluster_sizes)
        if num_provided >= 2 * n_clusters:
            control_cluster_sizes = np.array(cluster_sizes[:n_clusters], dtype=int)
            intervention_cluster_sizes = np.array(cluster_sizes[n_clusters:2*n_clusters], dtype=int)
        elif num_provided > 0 : # Sample if some are provided but not enough for all
            control_cluster_sizes = np.random.choice(cluster_sizes, n_clusters).astype(int)
            intervention_cluster_sizes = np.random.choice(cluster_sizes, n_clusters).astype(int)
        else: # Fallback if cluster_sizes is empty list
            control_cluster_sizes = np.full(n_clusters, int(cluster_size), dtype=int)
            intervention_cluster_sizes = np.full(n_clusters, int(cluster_size), dtype=int)

    # Ensure cluster sizes are at least 1
    control_cluster_sizes = np.maximum(control_cluster_sizes, 1)
    intervention_cluster_sizes = np.maximum(intervention_cluster_sizes, 1)

    # Generate cluster-level probabilities
    if icc <= 0: # No clustering effect
        control_probs = np.full(n_clusters, p1)
        intervention_probs = np.full(n_clusters, p2)
    else:
        # Control arm beta parameters
        if p1 == 0.0: control_probs = np.zeros(n_clusters)
        elif p1 == 1.0: control_probs = np.ones(n_clusters)
        else:
            alpha1 = p1 * (1.0 - icc) / icc
            beta1 = (1.0 - p1) * (1.0 - icc) / icc
            control_probs = np.random.beta(alpha1, beta1, n_clusters)
        
        # Intervention arm beta parameters
        if p2 == 0.0: intervention_probs = np.zeros(n_clusters)
        elif p2 == 1.0: intervention_probs = np.ones(n_clusters)
        else:
            alpha2 = p2 * (1.0 - icc) / icc
            beta2 = (1.0 - p2) * (1.0 - icc) / icc
            intervention_probs = np.random.beta(alpha2, beta2, n_clusters)

    # Generate individual-level data
    data_list = []
    
    # Control group (treatment=0)
    for i in range(n_clusters):
        cluster_id_val = i # Cluster IDs 0 to n_clusters-1
        num_subjects = control_cluster_sizes[i]
        prob_success = control_probs[i]
        outcomes = np.random.binomial(1, prob_success, num_subjects)
        for outcome_val in outcomes:
            data_list.append({'outcome': outcome_val, 'treatment': 0, 'cluster_id': cluster_id_val})
    
    # Intervention group (treatment=1)
    for i in range(n_clusters):
        cluster_id_val = n_clusters + i # Cluster IDs n_clusters to 2*n_clusters-1
        num_subjects = intervention_cluster_sizes[i]
        prob_success = intervention_probs[i]
        outcomes = np.random.binomial(1, prob_success, num_subjects)
        for outcome_val in outcomes:
            data_list.append({'outcome': outcome_val, 'treatment': 1, 'cluster_id': cluster_id_val})
            
    return pd.DataFrame(data_list)


def _analyze_binary_deff_ztest(df, icc):
    """
    Analyzes a single simulated binary trial using a z-test for two proportions,
    adjusted for clustering using a design effect (DEFF).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns: 'outcome' (0/1), 'treatment' (0/1), 'cluster_id'.
    icc : float
        Intracluster correlation coefficient.

    Returns
    -------
    dict
        {'p_value': float, 'fit_status': str}
    """
    try:
        control_df = df[df['treatment'] == 0]
        intervention_df = df[df['treatment'] == 1]

        if control_df.empty or intervention_df.empty:
            return {'p_value': 1.0, 'fit_status': 'data_error_empty_arm'}

        total_control_size = len(control_df)
        total_intervention_size = len(intervention_df)

        if total_control_size == 0 or total_intervention_size == 0:
             return {'p_value': 1.0, 'fit_status': 'data_error_zero_size_arm'}

        control_successes = control_df['outcome'].sum()
        intervention_successes = intervention_df['outcome'].sum()

        control_mean = control_successes / total_control_size
        intervention_mean = intervention_successes / total_intervention_size
        
        total_successes = control_successes + intervention_successes
        total_size = total_control_size + total_intervention_size
        
        if total_size == 0: # Should be caught by individual arm checks, but as a safeguard
            return {'p_value': 1.0, 'fit_status': 'data_error_zero_total_size'}
            
        pooled_p = total_successes / total_size

        n_total_clusters = df['cluster_id'].nunique()
        if n_total_clusters == 0: # Should not happen if arms are not empty
             return {'p_value': 1.0, 'fit_status': 'data_error_zero_clusters'}

        avg_cluster_size_actual = total_size / n_total_clusters # Actual average size in this sim
        
        deff_calc_term = avg_cluster_size_actual - 1
        if deff_calc_term < 0: deff_calc_term = 0 

        deff = 1 + deff_calc_term * icc
        if deff <= 0: deff = 1 

        pooled_var_numerator = pooled_p * (1 - pooled_p)
        
        if pooled_var_numerator == 0:
            p_value = 0.0 if control_mean != intervention_mean else 1.0
            return {'p_value': p_value, 'fit_status': 'success_no_variance_in_pooled_outcome'}

        term_in_sqrt = pooled_var_numerator * (1.0/total_control_size + 1.0/total_intervention_size) * deff
        
        if term_in_sqrt <= 0: 
            p_value = 0.0 if control_mean != intervention_mean else 1.0
            return {'p_value': p_value, 'fit_status': 'success_zero_se_after_deff'}

        se = np.sqrt(term_in_sqrt)
        
        z_statistic = abs(intervention_mean - control_mean) / se
        p_value = 2 * (1 - stats.norm.cdf(z_statistic))
        
        return {'p_value': p_value, 'fit_status': 'success'}

    except Exception as e:
        return {'p_value': 1.0, 'fit_status': f'error_deff_ztest_{type(e).__name__}'}


def _analyze_binary_agg_ttest(df):
    """
    Analyzes a single simulated binary trial using an independent samples t-test
    on cluster-level proportions.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns: 'outcome' (0/1), 'treatment' (0/1), 'cluster_id'.

    Returns
    -------
    dict
        {'p_value': float, 'fit_status': str}
    """
    try:
        if df.empty or not all(col in df.columns for col in ['treatment', 'cluster_id', 'outcome']):
            return {'p_value': 1.0, 'fit_status': 'data_error_malformed_input_df'}

        # Aggregate data to cluster level: sum of outcomes and count of subjects per cluster
        cluster_data = df.groupby(['treatment', 'cluster_id'], observed=True).agg(
            sum_outcomes=('outcome', 'sum'),
            n_subjects=('outcome', 'count')
        ).reset_index()

        # Check for empty clusters (n_subjects == 0)
        if (cluster_data['n_subjects'] == 0).any():
            # This case might indicate an issue upstream or a very sparse cluster.
            # For a t-test on proportions, such clusters can't contribute meaningfully.
            return {'p_value': 1.0, 'fit_status': 'data_error_empty_cluster_found'}
        
        cluster_data['proportion'] = cluster_data['sum_outcomes'] / cluster_data['n_subjects']
        
        control_props = cluster_data[cluster_data['treatment'] == 0]['proportion']
        intervention_props = cluster_data[cluster_data['treatment'] == 1]['proportion']

        if control_props.empty or intervention_props.empty:
            return {'p_value': 1.0, 'fit_status': 'data_error_empty_arm'}

        # For t-test, require at least 2 clusters per arm for variance calculation and meaningful test
        if len(control_props) < 2 or len(intervention_props) < 2:
            return {'p_value': 1.0, 'fit_status': 'data_error_too_few_clusters_for_ttest'}

        # Check for variance within each group's proportions
        var_control = control_props.var(ddof=1) # ddof=1 for sample variance
        var_intervention = intervention_props.var(ddof=1)

        # Case 1: Both groups have zero variance in proportions
        if np.isclose(var_control, 0) and np.isclose(var_intervention, 0):
            mean_control = control_props.mean()
            mean_intervention = intervention_props.mean()
            if np.isclose(mean_control, mean_intervention):
                # Proportions are constant and equal in both arms
                return {'p_value': 1.0, 'fit_status': 'success_novar_means_equal'}
            else:
                # Proportions are constant but different; perfect separation
                return {'p_value': 0.0, 'fit_status': 'success_novar_means_diff'}
        
        # Perform Welch's t-test (equal_var=False) by default as cluster-level variances might differ
        ttest_result = stats.ttest_ind(control_props, intervention_props, equal_var=False, nan_policy='raise')
        
        if np.isnan(ttest_result.pvalue):
            # This can happen if, e.g., one group has variance and the other doesn't, 
            # and means are identical, or other edge cases leading to NaN in t-statistic components.
            return {'p_value': 1.0, 'fit_status': 'error_ttest_nan_pvalue_from_stats'}

        return {'p_value': ttest_result.pvalue, 'fit_status': 'success'}

    except Exception as e:
        # Catch any other unexpected errors during the t-test process
        return {'p_value': 1.0, 'fit_status': f'error_agg_ttest_{type(e).__name__}'}


def power_binary_sim(n_clusters, cluster_size, icc, p1, p2=None, 
                      nsim=1000, alpha=0.05, seed=None, cv_cluster_size=0,
                      cluster_sizes=None, effect_measure=None, effect_value=None,
                      analysis_method="deff_ztest"):
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
        Options: "deff_ztest", "aggregate_ttest". Default is "deff_ztest".
        Future options: "glmm", "gee".
    
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

    for _ in tqdm(range(nsim), desc=f"Simulating trials ({analysis_method}) for power (binary)"):
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
        # Future analysis methods will be added here
        # elif analysis_method == "glmm":
        #     analysis_result = _analyze_binary_glmm(df_trial) 
        # elif analysis_method == "gee":
        #     analysis_result = _analyze_binary_gee(df_trial)
        else:
            fit_statuses['error_unknown_method'] += 1
            continue 

        p_val = analysis_result['p_value']
        fit_status = analysis_result['fit_status']
        
        p_values_list.append(p_val)
        fit_statuses[fit_status] += 1

        if analysis_method == "deff_ztest":
            acceptable_statuses_for_power = ['success', 'success_no_variance_in_pooled_outcome', 'success_zero_se_after_deff']
        elif analysis_method == "aggregate_ttest":
            acceptable_statuses_for_power = ['success', 'success_novar_means_equal', 'success_novar_means_diff']
        else: # Should not be reached if method is validated
            acceptable_statuses_for_power = ['success'] # Fallback
        
        if fit_status in acceptable_statuses_for_power and p_val < alpha:
            significant_results += 1
            
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
        "risk_difference": effect_info.get('risk_difference', abs(p2 - p1)), # Fallback for risk_difference
        "risk_ratio": effect_info.get('risk_ratio'),
        "odds_ratio": effect_info.get('odds_ratio'),
        "design_effect": deff,
        "cv_cluster_size": cv_cluster_size,
        "alpha": alpha,
        "nsim": nsim,
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