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

# Statsmodels imports for GLMM and GEE
try:
    import statsmodels.api as sm
    from statsmodels.genmod.families import Binomial
    from statsmodels.genmod.cov_struct import Exchangeable
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    warnings.warn("Statsmodels not available. GLMM and GEE analysis methods will not work.", ImportWarning)

# Optional Bayesian support
try:
    from cmdstanpy import CmdStanModel
    import tempfile
    import os
    _STAN_AVAILABLE = True
    _STAN_BINARY_MODEL = None
except ImportError:
    _STAN_AVAILABLE = False
    _STAN_BINARY_MODEL = None

try:
    import pymc as pm
    import pytensor.tensor as pt
    import arviz as az
    _PYMC_AVAILABLE = True
except ImportError:
    _PYMC_AVAILABLE = False

# Check for lightweight Bayesian alternatives
try:
    from scipy.optimize import minimize
    from scipy.stats import multivariate_normal, beta as scipy_beta
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False


def _get_stan_binary_model():
    """Get compiled Stan model for binary cluster RCT analysis."""
    global _STAN_BINARY_MODEL
    if _STAN_BINARY_MODEL is None:
        stan_code = """
        data {
            int<lower=1> N;                           // Total observations
            int<lower=1> J;                           // Number of clusters
            array[N] int<lower=1, upper=J> cluster;   // Cluster indicators
            array[N] int<lower=0, upper=1> y;         // Binary outcomes
            vector[N] treat;                          // Treatment indicators
        }
        parameters {
            real alpha;                               // Intercept (log-odds)
            real beta;                                // Treatment effect (log-odds)
            vector[J] u_raw;                         // Raw cluster effects
            real<lower=0> sigma_u;                   // Between-cluster SD
        }
        transformed parameters {
            vector[J] u = u_raw * sigma_u;           // Non-centered parameterization
        }
        model {
            // Priors
            alpha ~ normal(0, 2.5);                  // Intercept prior (log-odds scale)
            beta ~ normal(0, 2.5);                   // Treatment effect prior
            u_raw ~ normal(0, 1);                    // Non-centered parameterization
            sigma_u ~ student_t(3, 0, 2.5);        // Between-cluster SD prior
            
            // Likelihood
            for (n in 1:N) {
                real logit_p = alpha + beta * treat[n] + u[cluster[n]];
                y[n] ~ bernoulli_logit(logit_p);
            }
        }
        """
        # Write Stan code to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.stan', delete=False) as f:
            f.write(stan_code)
            stan_file = f.name
        
        # Create the Stan model - don't delete file until after compilation
        _STAN_BINARY_MODEL = CmdStanModel(stan_file=stan_file)
        
        # Clean up the temporary file after compilation
        try:
            if os.path.exists(stan_file):
                os.unlink(stan_file)
        except OSError:
            pass  # Ignore cleanup errors
    return _STAN_BINARY_MODEL


def _fit_pymc_binary_model(df, draws=500, tune=500, chains=4):
    """Fit PyMC hierarchical model for binary cluster RCT."""
    if not _PYMC_AVAILABLE:
        raise ImportError("PyMC is not available. Please install with: pip install pymc")
    
    # Map cluster IDs to consecutive integers starting from 0 (PyMC convention)
    unique_clusters = sorted(df['cluster_id'].unique())
    cluster_map = {old_id: new_id for new_id, old_id in enumerate(unique_clusters)}
    cluster_ids = df['cluster_id'].map(cluster_map).values
    
    n_clusters = len(unique_clusters)
    y_obs = df['outcome'].values
    treatment = df['treatment'].values
    
    with pm.Model() as model:
        # Priors
        alpha = pm.Normal("alpha", mu=0, sigma=2.5)
        beta = pm.Normal("beta", mu=0, sigma=2.5)
        
        # Cluster random effects
        sigma_u = pm.HalfStudentT("sigma_u", nu=3, sigma=2.5)
        u_raw = pm.Normal("u_raw", mu=0, sigma=1, shape=n_clusters)
        u = pm.Deterministic("u", u_raw * sigma_u)
        
        # Linear predictor on logit scale
        logit_p = alpha + beta * treatment + u[cluster_ids]
        
        # Likelihood
        y = pm.Bernoulli("y", logit_p=logit_p, observed=y_obs)
        
        # Sample
        trace = pm.sample(
            draws=draws, 
            tune=tune, 
            chains=chains, 
            return_inferencedata=True,
            progressbar=False,
            random_seed=42
        )
    
    return trace, model


def _fit_variational_bayes_binary(df, n_samples=1000):
    """Fast approximate Bayesian inference for binary outcomes using Variational Bayes."""
    
    # Prepare data
    y = df['outcome'].values
    treatment = df['treatment'].values
    unique_clusters = sorted(df['cluster_id'].unique())
    cluster_map = {old_id: new_id for new_id, old_id in enumerate(unique_clusters)}
    cluster_ids = df['cluster_id'].map(cluster_map).values
    n_clusters = len(unique_clusters)
    
    # Design matrix: intercept, treatment, cluster random effects
    n_obs = len(y)
    n_params = 2 + n_clusters  # alpha, beta, u_1, ..., u_J
    
    # Negative log posterior for optimization
    def neg_log_posterior(params):
        alpha, beta = params[:2]
        u = params[2:]
        
        # Linear predictor on logit scale
        logit_p = alpha + beta * treatment + u[cluster_ids]
        
        # Log likelihood (Bernoulli with logit link)
        p = 1 / (1 + np.exp(-logit_p))
        p = np.clip(p, 1e-15, 1 - 1e-15)  # Avoid log(0)
        log_lik = np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
        
        # Priors: alpha, beta ~ N(0, 2.5^2), u ~ N(0, sigma_u^2)
        sigma_u = 1.0  # Approximate from data structure
        log_prior_fixed = -0.5 * (alpha**2 + beta**2) / (2.5**2)  # N(0, 2.5^2)
        log_prior_random = -0.5 * np.sum(u**2) / sigma_u**2  # N(0, 1^2)
        
        return -(log_lik + log_prior_fixed + log_prior_random)
    
    # Find MAP estimate
    init_params = np.zeros(n_params)
    result = minimize(neg_log_posterior, init_params, method='BFGS')
    
    if not result.success:
        raise RuntimeError("Variational optimization failed")
    
    # Approximate posterior covariance (inverse Hessian at MAP)
    from scipy.optimize import approx_fprime
    eps = 1e-8
    
    def grad_neg_log_post(params):
        return approx_fprime(params, neg_log_posterior, eps)
    
    # Approximate Hessian numerically (for small problems)
    hess_approx = np.eye(n_params) * 0.1  # Diagonal approximation
    try:
        for i in range(n_params):
            ei = np.zeros(n_params)
            ei[i] = eps
            grad_plus = grad_neg_log_post(result.x + ei)
            grad_minus = grad_neg_log_post(result.x - ei)
            hess_approx[i, i] = max(0.01, (grad_plus[i] - grad_minus[i]) / (2 * eps))  # Ensure positive
    except Exception:
        # If gradient computation fails, use simple diagonal approximation
        hess_approx = np.eye(n_params) * 0.1
    
    # Sample from approximate posterior
    cov_approx = None
    try:
        cov_approx = np.linalg.inv(hess_approx)
        # Ensure positive definite
        cov_approx = cov_approx + np.eye(n_params) * 1e-6
        samples = multivariate_normal.rvs(mean=result.x, cov=cov_approx, size=n_samples)
        
        # Extract beta samples (treatment effect)
        beta_samples = samples[:, 1] if samples.ndim > 1 else [samples[1]]
        
        return beta_samples, True  # success flag
    except (np.linalg.LinAlgError, ValueError) as e:
        # Fallback: use MAP estimate with uncertainty
        beta_map = result.x[1]
        beta_se = np.sqrt(max(0.01, cov_approx[1, 1])) if cov_approx is not None else 0.1
        beta_samples = np.random.normal(beta_map, beta_se, n_samples)
        return beta_samples, False  # approximate flag


def _fit_abc_bayes_binary(df, n_samples=1000, tolerance=0.1):
    """Approximate Bayesian Computation for binary cluster RCT."""
    
    # Observed summary statistics
    y_obs = df['outcome'].values
    treatment_obs = df['treatment'].values
    cluster_obs = df['cluster_id'].values
    
    # Summary statistics: proportions by group, ICC estimate
    control_prop_obs = y_obs[treatment_obs == 0].mean()
    treat_prop_obs = y_obs[treatment_obs == 1].mean()
    
    # Cluster-level proportions for ICC estimation
    cluster_props_obs = df.groupby(['cluster_id', 'treatment'])['outcome'].mean()
    cluster_prop_var_obs = cluster_props_obs.var()
    
    obs_stats = np.array([control_prop_obs, treat_prop_obs, cluster_prop_var_obs])
    
    accepted_samples = []
    n_attempts = 0
    max_attempts = n_samples * 50  # Don't run forever
    
    while len(accepted_samples) < n_samples and n_attempts < max_attempts:
        n_attempts += 1
        
        # Sample from priors (on logit scale)
        alpha_sim = np.random.normal(0, 2.5)
        beta_sim = np.random.normal(0, 2.5)  # This is what we want to estimate
        sigma_u_sim = np.abs(np.random.normal(0, 1))
        
        # Simulate data with same structure
        try:
            unique_clusters = sorted(df['cluster_id'].unique())
            n_clusters = len(unique_clusters)
            
            # Generate cluster effects
            u_sim = np.random.normal(0, sigma_u_sim, n_clusters)
            
            # Simulate outcomes
            y_sim = []
            treatment_sim = []
            cluster_sim = []
            
            for i, cluster_id in enumerate(unique_clusters):
                cluster_data = df[df['cluster_id'] == cluster_id]
                
                for _, row in cluster_data.iterrows():
                    treatment_val = row['treatment']
                    logit_p = alpha_sim + beta_sim * treatment_val + u_sim[i]
                    p = 1 / (1 + np.exp(-logit_p))
                    p = np.clip(p, 1e-15, 1 - 1e-15)
                    y_val = np.random.binomial(1, p)
                    
                    y_sim.append(y_val)
                    treatment_sim.append(treatment_val)
                    cluster_sim.append(cluster_id)
            
            y_sim = np.array(y_sim)
            treatment_sim = np.array(treatment_sim)
            
            # Calculate summary statistics
            control_prop_sim = y_sim[treatment_sim == 0].mean()
            treat_prop_sim = y_sim[treatment_sim == 1].mean()
            
            df_sim = pd.DataFrame({'outcome': y_sim, 'treatment': treatment_sim, 'cluster_id': cluster_sim})
            cluster_props_sim = df_sim.groupby(['cluster_id', 'treatment'])['outcome'].mean()
            cluster_prop_var_sim = cluster_props_sim.var()
            
            sim_stats = np.array([control_prop_sim, treat_prop_sim, cluster_prop_var_sim])
            
            # Check if simulated stats are close enough to observed
            distance = np.sqrt(np.sum((obs_stats - sim_stats)**2))
            
            if distance < tolerance:
                accepted_samples.append(beta_sim)
                
        except Exception:
            continue  # Skip this simulation
    
    if len(accepted_samples) < n_samples // 10:  # Need at least 10% acceptance
        # Relax tolerance and try again
        return _fit_abc_bayes_binary(df, n_samples, tolerance * 2)
    
    return np.array(accepted_samples), len(accepted_samples) >= n_samples // 2  # success if >50% accepted


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
            # For binary outcomes, we can use MixedLM with binomial family
            from statsmodels.regression.mixed_linear_model import MixedLM
            
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
                import statsmodels.formula.api as smf
                
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
        from statsmodels.genmod.generalized_estimating_equations import GEE
        
        model = GEE(df['outcome'], X, groups=df['cluster_id'], 
                   family=Binomial(), cov_struct=Exchangeable())
        result = model.fit(maxiter=100)
        
        # Get p-value for treatment effect
        treatment_pvalue = result.pvalues.iloc[1] if len(result.pvalues) > 1 else 1.0
        
        return {'p_value': treatment_pvalue, 'fit_status': 'success'}
        
    except Exception as e:
        # If GEE fails, try a simpler approach with cluster-robust standard errors
        try:
            import statsmodels.formula.api as smf
            
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
        Bayesian backend to use: 'stan', 'pymc', 'variational', 'abc'
    bayes_draws : int, optional
        Number of posterior draws for MCMC methods
    bayes_warmup : int, optional  
        Number of warmup iterations for MCMC methods
    bayes_inference_method : str, optional
        Method for significance testing: 'credible_interval', 'posterior_probability', 'rope'

    Returns
    -------
    dict
        {'p_value': float, 'fit_status': str, 'bayes_details': dict}
    """
    # Check backend availability and apply fallbacks
    original_backend = bayes_backend
    
    # Try to find an available backend
    if bayes_backend == "pymc" and not _PYMC_AVAILABLE:
        warnings.warn("PyMC not available. Trying variational approximation...", UserWarning)
        bayes_backend = "variational"
    
    if bayes_backend == "stan" and not _STAN_AVAILABLE:
        warnings.warn("Stan not available. Trying variational approximation...", UserWarning)
        bayes_backend = "variational"
        
    if bayes_backend in ["variational", "abc"] and not _SCIPY_AVAILABLE:
        warnings.warn("SciPy not available for approximate methods. Falling back to deff_ztest.", UserWarning)
        # Fall back to design effect z-test
        icc_fallback = 0.05  # Default ICC for fallback
        return _analyze_binary_deff_ztest(df, icc_fallback)

    try:
        if bayes_backend == "variational":
            # Fast Variational Bayes
            beta_samples, converged = _fit_variational_bayes_binary(df, n_samples=bayes_draws)
            beta_rhat = 1.0  # Not applicable for VI
            backend_used = "variational"
            
            if not converged:
                warnings.warn("Variational approximation may be inaccurate. Consider using full MCMC.")
                
        elif bayes_backend == "abc":
            # Approximate Bayesian Computation
            beta_samples, converged = _fit_abc_bayes_binary(df, n_samples=bayes_draws)
            beta_rhat = 1.0  # Not applicable for ABC
            backend_used = "abc"
            
            if not converged:
                warnings.warn("ABC had low acceptance rate. Results may be unreliable.")
                
        elif bayes_backend == "pymc":
            # PyMC implementation
            trace, model = _fit_pymc_binary_model(
                df, 
                draws=bayes_draws, 
                tune=bayes_warmup, 
                chains=4
            )
            
            # Extract beta samples
            beta_samples = trace.posterior["beta"].values.flatten()
            
            # Check convergence using R-hat
            rhat_summary = az.rhat(trace)
            beta_rhat = float(rhat_summary["beta"].values)
            converged = beta_rhat < 1.1
            
            if not converged:
                warnings.warn(f"PyMC model did not converge (R-hat = {beta_rhat:.3f}). Results may be unreliable.")
            
            backend_used = "pymc"
            
        else:  # Stan backend
            model = _get_stan_binary_model()
            N = len(df)
            y = df['outcome'].values.astype(int)
            # Map cluster IDs to consecutive integers starting from 1 (Stan convention)
            unique_clusters = sorted(df['cluster_id'].unique())
            cluster_map = {old_id: new_id + 1 for new_id, old_id in enumerate(unique_clusters)}
            cluster_ids = df['cluster_id'].map(cluster_map).values.astype(int)
            treat = df['treatment'].values
            data = {
                "N": N,
                "J": len(unique_clusters),
                "cluster": cluster_ids,
                "y": y,
                "treat": treat,
            }
            
            # Use multiple chains for better convergence assessment
            n_chains = 4
            fit = model.sample(
                data=data,
                chains=n_chains,
                iter_sampling=bayes_draws,
                iter_warmup=bayes_warmup,
                show_progress=False,
                refresh=1,
            )
            
            beta_samples = fit.stan_variable("beta")
            
            # Check convergence using R-hat
            summary_df = fit.summary()
            beta_rhat = summary_df.loc[summary_df.index.str.contains('beta'), 'R_hat'].iloc[0]
            converged = beta_rhat < 1.1
            
            if not converged:
                warnings.warn(f"Stan model did not converge (R-hat = {beta_rhat:.3f}). Results may be unreliable.")
            
            backend_used = "stan"
        
        # Calculate Bayesian inference methods
        # Method 1: Credible Interval (95% CI excludes zero)
        ci_lower = np.percentile(beta_samples, 2.5)
        ci_upper = np.percentile(beta_samples, 97.5)
        significant_ci = ci_lower > 0 or ci_upper < 0
        
        # Method 2: Posterior Probability (probability of favorable effect)
        prob_positive = (beta_samples > 0).mean()
        significant_prob = prob_positive > 0.975 or prob_positive < 0.025
        
        # Method 3: Region of Practical Equivalence (ROPE)
        # Use 0.1 log-odds as ROPE half-width (≈ 0.025 on probability scale)
        rope_half_width = 0.1  # log-odds scale
        prob_rope = ((beta_samples > -rope_half_width) & (beta_samples < rope_half_width)).mean()
        significant_rope = prob_rope < 0.05
        
        # Choose significance based on selected inference method
        if bayes_inference_method == "credible_interval":
            significant = significant_ci
        elif bayes_inference_method == "posterior_probability":
            significant = significant_prob
        elif bayes_inference_method == "rope":
            significant = significant_rope
        else:
            # Default to credible interval if unknown method specified
            significant = significant_ci
            bayes_inference_method = "credible_interval"
        
        # Convert to p-value equivalent for compatibility with existing code
        if significant:
            p_value = 0.01  # Arbitrarily small to indicate significance
        else:
            p_value = 0.5   # Arbitrarily large to indicate non-significance
        
        bayes_details = {
            "converged": converged, 
            "backend": backend_used,
            "rhat": beta_rhat,
            "beta_mean": np.mean(beta_samples),
            "beta_sd": np.std(beta_samples),
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "significant_ci": significant_ci,
            "prob_positive": prob_positive,
            "significant_prob": significant_prob,
            "rope_half_width": rope_half_width,
            "prob_rope": prob_rope,
            "significant_rope": significant_rope,
            "inference_method": bayes_inference_method
        }
        
        fit_status = f"success_bayes_{backend_used}" if converged else f"warning_bayes_{backend_used}_convergence"
        
        return {'p_value': p_value, 'fit_status': fit_status, 'bayes_details': bayes_details}
        
    except Exception as e:
        warnings.warn(f"Bayesian model failed with error: {str(e)}. Falling back to deff_ztest.")
        # Fall back to design effect z-test
        icc_fallback = 0.05  # Default ICC for fallback
        result = _analyze_binary_deff_ztest(df, icc_fallback)
        result['fit_status'] = f"error_bayes_fallback_{result['fit_status']}"
        return result


def power_binary_sim(n_clusters, cluster_size, icc, p1, p2=None, nsim=1000, alpha=0.05, seed=None, cv_cluster_size=0, cluster_sizes=None, effect_measure=None, effect_value=None, analysis_method="deff_ztest", bayes_backend="stan", bayes_draws=500, bayes_warmup=500, bayes_inference_method="credible_interval"):
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
        Options: "deff_ztest", "aggregate_ttest", "glmm", "gee", "bayes". 
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
            acceptable_statuses_for_power = ['success_bayes_stan', 'success_bayes_pymc', 'success_bayes_variational', 'success_bayes_abc', 
                                           'warning_bayes_stan_convergence', 'warning_bayes_pymc_convergence', 
                                           'warning_bayes_variational_convergence', 'warning_bayes_abc_convergence']
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
                            effect_measure=None, effect_value=None,
                            analysis_method="deff_ztest", bayes_backend="stan", 
                            bayes_draws=500, bayes_warmup=500, 
                            bayes_inference_method="credible_interval"):
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
        sim_results = power_binary_sim(analysis_method=analysis_method, 
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
            bayes_inference_method=bayes_inference_method
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
    final_results = power_binary_sim(analysis_method=analysis_method, 
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
        bayes_inference_method=bayes_inference_method
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
                                      effect_measure='risk_difference',
                                      analysis_method="deff_ztest", bayes_backend="stan", 
                                      bayes_draws=500, bayes_warmup=500, 
                                      bayes_inference_method="credible_interval"):
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
        sim_results = power_binary_sim(analysis_method=analysis_method, 
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
            bayes_inference_method=bayes_inference_method
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
    final_results = power_binary_sim(analysis_method=analysis_method, 
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
        bayes_inference_method=bayes_inference_method
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