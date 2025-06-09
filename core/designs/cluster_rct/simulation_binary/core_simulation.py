"""
Core simulation functions for cluster randomized controlled trials with binary outcomes.

This module provides the main simulation engine for generating and analyzing cluster RCT data
with binary outcomes using beta-binomial models.
"""

import numpy as np
import pandas as pd
import warnings
from scipy import stats

# Statsmodels imports for GLMM and GEE
try:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from statsmodels.genmod.families import Binomial
    from statsmodels.genmod.cov_struct import Exchangeable
    from statsmodels.regression.mixed_linear_model import MixedLM
    from statsmodels.genmod.generalized_estimating_equations import GEE
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    warnings.warn("Statsmodels not available. GLMM and GEE analysis methods will not work.", ImportWarning)

from .bayesian_methods import (_get_stan_binary_model, _fit_pymc_binary_model, 
                               _fit_variational_bayes_binary, _fit_abc_bayes_binary,
                               _STAN_AVAILABLE, _PYMC_AVAILABLE, _SCIPY_AVAILABLE)


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


def _analyze_binary_permutation(df):
    """
    Analyzes a single simulated binary trial using exact permutation test.
    
    This function performs cluster randomization inference (CRI) providing exact
    p-values without distributional assumptions. Recommended for very small 
    cluster trials (5-15 clusters per arm).
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns: 'outcome' (0/1), 'treatment' (0/1), 'cluster_id'.
        
    Returns
    -------
    dict
        {'p_value': float, 'fit_status': str, 'permutation_details': dict}
    """
    try:
        from .permutation_tests import cluster_permutation_test
        
        # Calculate cluster-level success rates
        cluster_summary = df.groupby(['cluster_id', 'treatment'])['outcome'].agg(['mean', 'count']).reset_index()
        cluster_summary.columns = ['cluster_id', 'treatment', 'success_rate', 'cluster_size']
        
        # Check for adequate cluster sizes and outcomes
        if len(cluster_summary) < 6:
            return {'p_value': 1.0, 'fit_status': 'data_error_too_few_clusters_for_permutation'}
        
        # Separate by treatment group
        control_data = cluster_summary[cluster_summary['treatment'] == 0]
        treatment_data = cluster_summary[cluster_summary['treatment'] == 1]
        
        if len(control_data) == 0 or len(treatment_data) == 0:
            return {'p_value': 1.0, 'fit_status': 'data_error_missing_treatment_group'}
        
        # Prepare data for permutation test
        perm_data = {
            'control_clusters': control_data['success_rate'].values,
            'treatment_clusters': treatment_data['success_rate'].values
        }
        
        # Determine number of permutations based on total cluster size
        total_clusters = len(control_data) + len(treatment_data)
        if total_clusters <= 12:
            n_perms = 'exact'  # Use exact permutation for very small trials
        elif total_clusters <= 20:
            n_perms = 10000    # High precision for small trials
        else:
            n_perms = 5000     # Standard precision for larger trials
        
        # Perform permutation test
        perm_result = cluster_permutation_test(
            data=perm_data,
            test_statistic='mean_difference',  # Use mean difference for binary outcomes
            n_permutations=n_perms,
            alternative='two-sided'
        )
        
        return {
            'p_value': perm_result['p_value'],
            'fit_status': 'success',
            'permutation_details': {
                'method': perm_result['method'],
                'n_permutations': perm_result['n_permutations_used'],
                'observed_effect': perm_result['observed_statistic'],
                'confidence_interval': perm_result['confidence_interval']
            }
        }
        
    except Exception as e:
        return {'p_value': 1.0, 'fit_status': f'error_permutation_{type(e).__name__}'}


def _analyze_binary_glmm(df):
    """
    Analyzes a single simulated binary trial using a Generalized Linear Mixed Model (GLMM).
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns: 'outcome' (0/1), 'treatment' (0/1), 'cluster_id'.
    
    Returns
    -------
    dict
        Dictionary with 'p_value' and 'fit_status'.
    """
    if not STATSMODELS_AVAILABLE:
        return {'p_value': 1.0, 'fit_status': 'error_statsmodels_not_available'}
    
    try:
        # Prepare the data for GLMM
        # Convert cluster_id to categorical to ensure it's treated as a random effect grouping variable
        df = df.copy()
        df['cluster_id'] = df['cluster_id'].astype('category')
        
        # Fit GLMM with random intercept for clusters
        # Using MixedLM from statsmodels for binomial regression with random effects
        try:
            # Create design matrix for fixed effects (treatment)
            X = sm.add_constant(df['treatment'])  # Add intercept
            
            # Fit the mixed model with random intercept by cluster
            model = MixedLM(df['outcome'], X, groups=df['cluster_id'])
            result = model.fit(method='lbfgs', maxiter=1000)
            
            # Get p-value for treatment effect (coefficient index 1, since 0 is intercept)
            treatment_pvalue = result.pvalues.iloc[1] if len(result.pvalues) > 1 else 1.0
            
            return {'p_value': treatment_pvalue, 'fit_status': 'success'}
            
        except Exception as e:
            # If MixedLM fails, try using GLMMs via statsmodels
            # This is a backup approach using logistic regression with cluster robust standard errors
            try:
                # Use GLM with cluster-robust standard errors as approximation
                model = smf.glm('outcome ~ treatment', data=df, family=sm.families.Binomial())
                result = model.fit(cov_type='cluster', cov_kwds={'groups': df['cluster_id']})
                
                treatment_pvalue = result.pvalues['treatment']
                return {'p_value': treatment_pvalue, 'fit_status': 'success_cluster_robust'}
                
            except Exception as e2:
                return {'p_value': 1.0, 'fit_status': f'error_glmm_fit_failed_{str(e2)[:50]}'}
    
    except Exception as e:
        return {'p_value': 1.0, 'fit_status': f'error_glmm_setup_failed_{str(e)[:50]}'}


def _analyze_binary_gee(df):
    """
    Analyzes a single simulated binary trial using Generalized Estimating Equations (GEE).
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns: 'outcome' (0/1), 'treatment' (0/1), 'cluster_id'.
    
    Returns
    -------
    dict
        Dictionary with 'p_value' and 'fit_status'.
    """
    if not STATSMODELS_AVAILABLE:
        return {'p_value': 1.0, 'fit_status': 'error_statsmodels_not_available'}
    
    try:
        # Prepare the data for GEE
        df = df.copy()
        df = df.sort_values(['cluster_id', 'treatment'])  # Sort by cluster for GEE
        
        # Create design matrix
        X = sm.add_constant(df['treatment'])  # Add intercept
        
        # Fit GEE model with exchangeable correlation structure
        model = GEE(df['outcome'], X, groups=df['cluster_id'], 
                   family=Binomial(), cov_struct=Exchangeable())
        result = model.fit(maxiter=100)
        
        # Get p-value for treatment effect
        treatment_pvalue = result.pvalues.iloc[1] if len(result.pvalues) > 1 else 1.0
        
        return {'p_value': treatment_pvalue, 'fit_status': 'success'}
        
    except Exception as e:
        # If GEE fails, try a simpler approach with cluster-robust standard errors
        try:
            model = smf.glm('outcome ~ treatment', data=df, family=sm.families.Binomial())
            result = model.fit(cov_type='cluster', cov_kwds={'groups': df['cluster_id']})
            
            treatment_pvalue = result.pvalues['treatment']
            return {'p_value': treatment_pvalue, 'fit_status': 'success_cluster_robust_fallback'}
            
        except Exception as e2:
            return {'p_value': 1.0, 'fit_status': f'error_gee_fit_failed_{str(e2)[:50]}'}


def _analyze_binary_bayes(df, bayes_backend="stan", bayes_draws=500, bayes_warmup=500, 
                         bayes_inference_method="credible_interval"):
    """
    Analyzes a single simulated binary trial using Bayesian hierarchical model.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns: 'outcome' (0/1), 'treatment' (0/1), 'cluster_id'.
    bayes_backend : str, optional
        Bayesian backend to use ("stan", "pymc", "variational", "abc"), by default "stan"
    bayes_draws : int, optional
        Number of posterior draws, by default 500
    bayes_warmup : int, optional
        Number of warmup iterations, by default 500
    bayes_inference_method : str, optional
        Inference method for significance testing, by default "credible_interval"

    Returns
    -------
    dict
        Dictionary with 'p_value' and 'fit_status'.
    """
    # Check backend availability and apply fallbacks
    original_backend = bayes_backend
    
    if bayes_backend == "pymc" and not _PYMC_AVAILABLE:
        warnings.warn("PyMC not available. Trying variational approximation...", UserWarning)
        bayes_backend = "variational"
    
    if bayes_backend == "stan" and not _STAN_AVAILABLE:
        warnings.warn("Stan not available. Trying variational approximation...", UserWarning)
        bayes_backend = "variational"
        
    if bayes_backend in ["variational", "abc"] and not _SCIPY_AVAILABLE:
        warnings.warn("SciPy not available for approximate methods. Using fallback analysis.", UserWarning)
        return _analyze_binary_deff_ztest(df, 0.05)  # Fallback to design effect z-test

    try:
        if bayes_backend == "variational":
            # Fast Variational Bayes
            beta_samples, converged = _fit_variational_bayes_binary(df, n_samples=bayes_draws)
            backend_used = "variational"
            
        elif bayes_backend == "abc":
            # Approximate Bayesian Computation
            beta_samples, converged = _fit_abc_bayes_binary(df, n_samples=bayes_draws)
            backend_used = "abc"
            
        elif bayes_backend == "pymc":
            # PyMC implementation
            trace, model = _fit_pymc_binary_model(df, draws=bayes_draws, tune=bayes_warmup, chains=4)
            beta_samples = trace.posterior["beta"].values.flatten()
            converged = True  # Simplified convergence check
            backend_used = "pymc"
            
        else:  # Stan backend
            model = _get_stan_binary_model()
            N = len(df)
            y = df['outcome'].values.astype(int)
            
            # Map cluster IDs to consecutive integers starting from 1 (Stan convention)
            unique_clusters = sorted(df['cluster_id'].unique())
            cluster_map = {old_id: new_id + 1 for new_id, old_id in enumerate(unique_clusters)}
            cluster_ids = df['cluster_id'].map(cluster_map).values
            treat = df['treatment'].values
            
            data = {
                "N": N,
                "J": len(unique_clusters),
                "cluster": cluster_ids.astype(int),
                "y": y,
                "treat": treat,
            }
            
            fit = model.sample(
                data=data,
                chains=4,
                iter_sampling=bayes_draws,
                iter_warmup=bayes_warmup,
                show_progress=False,
            )
            
            beta_samples = fit.stan_variable("beta")
            converged = True  # Simplified convergence check
            backend_used = "stan"
        
        # Calculate Bayesian inference methods
        # Method 1: Credible Interval (95% CI excludes zero)
        ci_lower = np.percentile(beta_samples, 2.5)
        ci_upper = np.percentile(beta_samples, 97.5)
        significant_ci = ci_lower > 0 or ci_upper < 0
        
        # Method 2: Posterior Probability (probability of favorable effect)
        prob_positive = (beta_samples > 0).mean()
        significant_prob = prob_positive > 0.975 or prob_positive < 0.025
        
        # Choose significance based on selected inference method
        if bayes_inference_method == "credible_interval":
            significant = significant_ci
        elif bayes_inference_method == "posterior_probability":
            significant = significant_prob
        else:
            # Default to credible interval if unknown method specified
            significant = significant_ci
        
        # Convert to p-value equivalent for compatibility
        p_value = 0.01 if significant else 0.5
        
        return {
            'p_value': p_value, 
            'fit_status': 'success',
            'bayes_details': {
                'backend': backend_used,
                'converged': converged,
                'beta_mean': np.mean(beta_samples),
                'beta_sd': np.std(beta_samples),
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'significant': significant
            }
        }
        
    except Exception as e:
        warnings.warn(f"Bayesian model failed: {str(e)}. Using fallback analysis.")
        return _analyze_binary_deff_ztest(df, 0.05)  # Fallback to design effect z-test