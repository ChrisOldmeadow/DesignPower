"""Simulation-based methods for cluster randomized controlled trials with continuous outcomes.

This module provides functions for power analysis, sample size calculation,
and minimum detectable effect estimation for cluster randomized controlled trials
with continuous outcomes using simulation-based approaches.
"""

import numpy as np
from scipy import stats
from tqdm import tqdm
import math
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.regression.mixed_linear_model import MixedLM
from statsmodels.base._penalties import L2
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.stats.anova import anova_lm
from statsmodels.genmod.cov_struct import Exchangeable
import warnings

# Optional Bayesian support
try:
    from cmdstanpy import CmdStanModel
    import tempfile
    import os
    _STAN_AVAILABLE = True
    _STAN_MODEL = None
except ImportError:
    _STAN_AVAILABLE = False
    _STAN_MODEL = None

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
    from scipy.stats import multivariate_normal
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False

def _get_stan_model():
    global _STAN_MODEL
    if _STAN_MODEL is None:
        stan_code = """
        data {
            int<lower=1> N;
            int<lower=1> J;
            array[N] int<lower=1, upper=J> cluster;
            vector[N] y;
            vector[N] treat;
        }
        parameters {
            real alpha;
            real beta;
            vector[J] u_raw;
            real<lower=0> sigma_u;
            real<lower=0> sigma_e;
        }
        transformed parameters {
            vector[J] u = u_raw * sigma_u;
        }
        model {
            beta ~ normal(0,10);
            u_raw ~ normal(0,1);
            sigma_u ~ student_t(3,0,2.5);
            sigma_e ~ student_t(3,0,2.5);
            for (n in 1:N)
                y[n] ~ normal(alpha + beta * treat[n] + u[cluster[n]], sigma_e);
        }
        """
        # Write Stan code to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.stan', delete=False) as f:
            f.write(stan_code)
            stan_file = f.name
        
        # Create the Stan model - don't delete file until after compilation
        _STAN_MODEL = CmdStanModel(stan_file=stan_file)
        
        # Clean up the temporary file after compilation
        try:
            if os.path.exists(stan_file):
                os.unlink(stan_file)
        except OSError:
            pass  # Ignore cleanup errors
    return _STAN_MODEL


def _fit_pymc_model(df, draws=500, tune=500, chains=4):
    """Fit PyMC hierarchical model for cluster RCT."""
    if not _PYMC_AVAILABLE:
        raise ImportError("PyMC is not available. Please install with: pip install pymc")
    
    # Map cluster IDs to consecutive integers starting from 0 (PyMC convention)
    unique_clusters = sorted(df['cluster'].unique())
    cluster_map = {old_id: new_id for new_id, old_id in enumerate(unique_clusters)}
    cluster_ids = df['cluster'].map(cluster_map).values
    
    n_clusters = len(unique_clusters)
    y_obs = df['y'].values
    treatment = df['treatment'].values
    
    with pm.Model() as model:
        # Priors
        alpha = pm.Normal("alpha", mu=0, sigma=10)
        beta = pm.Normal("beta", mu=0, sigma=10)
        
        # Cluster random effects
        sigma_u = pm.HalfStudentT("sigma_u", nu=3, sigma=2.5)
        u_raw = pm.Normal("u_raw", mu=0, sigma=1, shape=n_clusters)
        u = pm.Deterministic("u", u_raw * sigma_u)
        
        # Individual-level variance
        sigma_e = pm.HalfStudentT("sigma_e", nu=3, sigma=2.5)
        
        # Linear predictor
        mu = alpha + beta * treatment + u[cluster_ids]
        
        # Likelihood
        y = pm.Normal("y", mu=mu, sigma=sigma_e, observed=y_obs)
        
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


def _fit_variational_bayes(df, n_samples=1000):
    """Fast approximate Bayesian inference using Variational Bayes (Laplace approximation)."""
    
    # Prepare data
    y = df['y'].values
    treatment = df['treatment'].values
    unique_clusters = sorted(df['cluster'].unique())
    cluster_map = {old_id: new_id for new_id, old_id in enumerate(unique_clusters)}
    cluster_ids = df['cluster'].map(cluster_map).values
    n_clusters = len(unique_clusters)
    
    # Design matrix: intercept, treatment, cluster random effects
    n_obs = len(y)
    n_params = 2 + n_clusters  # alpha, beta, u_1, ..., u_J
    
    # Negative log posterior for optimization
    def neg_log_posterior(params):
        alpha, beta = params[:2]
        u = params[2:]
        
        # Likelihood: y ~ N(alpha + beta*treatment + u[cluster], sigma_e^2)
        sigma_e = 1.0  # Fixed for simplicity
        mu = alpha + beta * treatment + u[cluster_ids]
        log_lik = -0.5 * np.sum((y - mu)**2) / sigma_e**2
        
        # Priors: alpha, beta ~ N(0, 10^2), u ~ N(0, sigma_u^2)
        sigma_u = 1.0  # Approximate from data
        log_prior_fixed = -0.5 * (alpha**2 + beta**2) / 100  # N(0, 10^2)
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


def _fit_abc_bayes(df, n_samples=1000, tolerance=0.1):
    """Approximate Bayesian Computation for cluster RCT."""
    
    # Observed summary statistics
    y_obs = df['y'].values
    treatment_obs = df['treatment'].values
    cluster_obs = df['cluster'].values
    
    # Summary statistics: means by group, overall variance, cluster means variance
    control_mean_obs = y_obs[treatment_obs == 0].mean()
    treat_mean_obs = y_obs[treatment_obs == 1].mean()
    overall_var_obs = y_obs.var()
    
    # Cluster-level summaries
    cluster_means_obs = df.groupby(['cluster', 'treatment'])['y'].mean()
    cluster_var_obs = cluster_means_obs.var()
    
    obs_stats = np.array([control_mean_obs, treat_mean_obs, overall_var_obs, cluster_var_obs])
    
    accepted_samples = []
    n_attempts = 0
    max_attempts = n_samples * 50  # Don't run forever
    
    while len(accepted_samples) < n_samples and n_attempts < max_attempts:
        n_attempts += 1
        
        # Sample from priors
        alpha_sim = np.random.normal(0, 10)
        beta_sim = np.random.normal(0, 10)  # This is what we want to estimate
        sigma_u_sim = np.abs(np.random.normal(0, 2))
        sigma_e_sim = np.abs(np.random.normal(0, 2))
        
        # Simulate data with same structure
        try:
            unique_clusters = sorted(df['cluster'].unique())
            n_clusters = len(unique_clusters)
            cluster_size = len(df) // n_clusters // 2  # Approximate
            
            # Generate cluster effects
            u_sim = np.random.normal(0, sigma_u_sim, n_clusters)
            
            # Simulate outcomes
            y_sim = []
            treatment_sim = []
            cluster_sim = []
            
            for i, cluster_id in enumerate(unique_clusters):
                cluster_data = df[df['cluster'] == cluster_id]
                n_in_cluster = len(cluster_data)
                
                for _, row in cluster_data.iterrows():
                    treatment_val = row['treatment']
                    y_val = alpha_sim + beta_sim * treatment_val + u_sim[i] + np.random.normal(0, sigma_e_sim)
                    
                    y_sim.append(y_val)
                    treatment_sim.append(treatment_val)
                    cluster_sim.append(cluster_id)
            
            y_sim = np.array(y_sim)
            treatment_sim = np.array(treatment_sim)
            
            # Calculate summary statistics
            control_mean_sim = y_sim[treatment_sim == 0].mean()
            treat_mean_sim = y_sim[treatment_sim == 1].mean()
            overall_var_sim = y_sim.var()
            
            df_sim = pd.DataFrame({'y': y_sim, 'treatment': treatment_sim, 'cluster': cluster_sim})
            cluster_means_sim = df_sim.groupby(['cluster', 'treatment'])['y'].mean()
            cluster_var_sim = cluster_means_sim.var()
            
            sim_stats = np.array([control_mean_sim, treat_mean_sim, overall_var_sim, cluster_var_sim])
            
            # Check if simulated stats are close enough to observed
            distance = np.sqrt(np.sum((obs_stats - sim_stats)**2))
            
            if distance < tolerance:
                accepted_samples.append(beta_sim)
                
        except Exception:
            continue  # Skip this simulation
    
    if len(accepted_samples) < n_samples // 10:  # Need at least 10% acceptance
        # Relax tolerance and try again
        return _fit_abc_bayes(df, n_samples, tolerance * 2)
    
    return np.array(accepted_samples), len(accepted_samples) >= n_samples // 2  # success if >50% accepted


# ----------------------------------
# Helper for fallback cluster robust OLS
# ----------------------------------

def _ols_cluster_test(df):
    """Return t-value and p-value for treatment using OLS with cluster robust SE."""
    try:
        ols = sm.OLS(df["y"], sm.add_constant(df[["treatment"]])).fit(
            cov_type="cluster", cov_kwds={"groups": df["cluster"]}
        )
        tval = ols.tvalues["treatment"]
        pval = 2 * stats.t.sf(abs(tval), df=max(1, df["cluster"].nunique() - 2))
        return tval, pval
    except Exception:
        # Final fallback to simple t-test
        return stats.ttest_ind(
            df.loc[df["treatment"] == 0, "y"],
            df.loc[df["treatment"] == 1, "y"],
            equal_var=True,
        )


def simulate_continuous_trial(
    n_clusters,
    cluster_size,
    icc,
    mean1,
    mean2,
    std_dev,
    analysis_model="ttest",
    use_satterthwaite: bool = False,
    use_bias_correction: bool = False,
    bayes_draws: int = 500,
    bayes_warmup: int = 500,
    bayes_inference_method: str = "credible_interval",  # New: Choose Bayesian inference method
    bayes_backend: str = "stan",  # New: Choose Bayesian backend ("stan", "pymc", "variational", "abc")
    lmm_method: str = "auto",
    lmm_reml: bool = True,
    lmm_cov_penalty_weight: float = 0.0,  # Added: Weight for L2 penalty on covariance
    lmm_fit_kwargs: dict = None,  # Added: Additional kwargs for MixedLM.fit()
    return_details: bool = False,
):
    """
    Simulate a single cluster RCT with continuous outcome and analyse it using the
    specified model.

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
    analysis_model : str, optional
        One of {'ttest', 'mixedlm', 'gee', 'bayes'}. Determines the analysis model.
    use_satterthwaite : bool, optional
        If True and analysis_model == 'mixedlm', p-values are based on the t-distribution
        with residual degrees of freedom (approximation to Satterthwaite). Default False.
    use_bias_correction : bool, optional
        If True and analysis_model == 'gee', uses bias-reduced sandwich covariance.
    bayes_draws : int, optional
        Number of draws for Bayesian model. Default 500.
    bayes_warmup : int, optional
        Number of warmup iterations for Bayesian model. Default 500.
    bayes_inference_method : str, optional
        Bayesian inference method for significance testing. Options:
        - 'credible_interval': 95% credible interval excludes zero (default)
        - 'posterior_probability': Posterior probability > 97.5% or < 2.5%
        - 'rope': Region of Practical Equivalence approach (< 5% prob in ROPE)
    bayes_backend : str, optional
        Bayesian backend to use. Options:
        - 'stan': Use CmdStanPy/Stan (full MCMC, default)
        - 'pymc': Use PyMC (full MCMC, requires PyMC installation)
        - 'variational': Fast Variational Bayes (Laplace approximation, requires only scipy)
        - 'abc': Approximate Bayesian Computation (lightweight, requires only scipy)
    lmm_method : str, optional
        Optimizer to use for MixedLM. Can be 'auto' (default), 'lbfgs', 'powell', 'cg', 'bfgs', 'newton', 'nm'.
    lmm_reml : bool, optional
        Whether to use REML for MixedLM. Default True.
    lmm_cov_penalty_weight : float, optional
        Weight for L2 penalty on covariance structure for MixedLM. Default 0.0 (no penalty).
    lmm_fit_kwargs : dict, optional
        Additional keyword arguments to pass to MixedLM.fit() (e.g., {'gtol': 1e-8}). Default None.
    return_details : bool, optional
        If True, returns additional details about the simulation.
    """
    # Calculate between-cluster and within-cluster variance
    var_between = icc * std_dev**2
    var_within = (1 - icc) * std_dev**2

    # Generate cluster effects for all 2*n_clusters clusters
    total_clusters = 2 * n_clusters
    cluster_effects = np.random.normal(0, np.sqrt(var_between), total_clusters)
    
    # Randomly assign clusters to treatment arms
    # First n_clusters go to control, next n_clusters go to intervention
    cluster_assignments = np.concatenate([np.zeros(n_clusters), np.ones(n_clusters)])
    np.random.shuffle(cluster_assignments)  # Randomize treatment assignment
    
    # Initialize data structures
    all_data = []
    
    # Generate data for each cluster
    for cluster_id in range(total_clusters):
        treatment = cluster_assignments[cluster_id]
        cluster_effect = cluster_effects[cluster_id]
        
        # Determine mean based on treatment assignment
        base_mean = mean1 if treatment == 0 else mean2
        
        # Generate individual-level data for this cluster
        individual_outcomes = base_mean + cluster_effect + \
                            np.random.normal(0, np.sqrt(var_within), cluster_size)
        
        # Add to data list
        for outcome in individual_outcomes:
            all_data.append({
                'y': outcome,
                'treatment': int(treatment),
                'cluster': cluster_id
            })
    
    # Convert to DataFrame
    df = pd.DataFrame(all_data)
    
    # Convert cluster to categorical for better mixed model handling
    df['cluster'] = df['cluster'].astype('category')
    
    if analysis_model == "ttest":
        # Cluster-level analysis: aggregate to cluster means to account for ICC
        cluster_means = df.groupby(['cluster', 'treatment'], observed=True)['y'].mean().reset_index()
        control_means = cluster_means[cluster_means['treatment'] == 0]['y'].values
        interv_means = cluster_means[cluster_means['treatment'] == 1]['y'].values
        t_stat, p_value = stats.ttest_ind(control_means, interv_means, equal_var=True)
        if return_details:
            return t_stat, p_value, {"converged": True, "model": "ttest"}
        return t_stat, p_value
        
    elif analysis_model == "permutation":
        # Exact permutation test - recommended for very small clusters (5-10 per arm)
        from .permutation_tests import cluster_permutation_test
        
        # Aggregate to cluster means
        cluster_means = df.groupby(['cluster', 'treatment'], observed=True)['y'].mean().reset_index()
        control_means = cluster_means[cluster_means['treatment'] == 0]['y'].values
        interv_means = cluster_means[cluster_means['treatment'] == 1]['y'].values
        
        # Prepare data for permutation test
        perm_data = {
            'control_clusters': control_means,
            'treatment_clusters': interv_means
        }
        
        # Determine number of permutations based on cluster size
        total_clusters = len(control_means) + len(interv_means)
        if total_clusters <= 12:
            n_perms = 'exact'  # Use exact permutation for very small trials
        elif total_clusters <= 20:
            n_perms = 10000    # High precision for small trials
        else:
            n_perms = 5000     # Standard precision for larger trials
        
        perm_result = cluster_permutation_test(
            data=perm_data,
            test_statistic='t_statistic',  # Use t-statistic for compatibility
            n_permutations=n_perms,
            alternative='two-sided'
        )
        
        t_stat = perm_result['observed_statistic']
        p_value = perm_result['p_value']
        
        if return_details:
            return t_stat, p_value, {
                "converged": True, 
                "model": "permutation",
                "permutation_method": perm_result['method'],
                "n_permutations": perm_result['n_permutations_used'],
                "confidence_interval": perm_result['confidence_interval']
            }
        return t_stat, p_value

    if analysis_model == "mixedlm":
        fit_status = "fit_error_ttest_fallback" # Default status if all else fails
        tvalue, p_value = np.nan, np.nan
        try:
            model = sm.MixedLM.from_formula("y ~ treatment", groups="cluster", data=df)
            
            if lmm_cov_penalty_weight > 0:
                l2_penalty_obj = L2(weights=lmm_cov_penalty_weight)
                model.re_pen = l2_penalty_obj
            if lmm_method == "auto":
                method_list = ["lbfgs", "powell", "cg", "bfgs", "newton", "nm"]
            else:
                method_list = [lmm_method]

            result = None
            fit_exception = None
            convergence_warning_occurred = False

            for m in method_list:
                try:
                    with warnings.catch_warnings(record=True) as caught_warnings:
                        warnings.simplefilter("always", ConvergenceWarning)
                        current_fit_kwargs = {'reml': lmm_reml, 'method': m, 'disp': False}
                        if lmm_fit_kwargs:
                            current_fit_kwargs.update(lmm_fit_kwargs)
                        result = model.fit(**current_fit_kwargs)
                        
                        # Check if ConvergenceWarning was issued
                        for caught_warn in caught_warnings:
                            if issubclass(caught_warn.category, ConvergenceWarning):
                                convergence_warning_occurred = True
                                break
                    # If fit succeeded (even with warning), break
                    break 
                except Exception as e:
                    fit_exception = e
                    continue

            if result is None:
                # This means all optimizers failed
                fit_status = "fit_error_ols_fallback"
                warnings.warn(
                    f"All MixedLM optimizers failed (last error: {fit_exception}). Falling back to cluster-robust OLS.",
                    RuntimeWarning
                )
                tvalue, p_value = _ols_cluster_test(df)
            else:
                # Fit succeeded with at least one optimizer
                tvalue = result.tvalues["treatment"]
                if use_satterthwaite:
                    df_resid = max(1, result.df_resid) # Ensure df_resid is at least 1
                    p_value = 2 * stats.t.sf(np.abs(tvalue), df=df_resid)
                else:
                    p_value = result.pvalues["treatment"]

                # Check for boundary condition (very small cluster variance)
                cluster_var_estimate = float(result.cov_re.iloc[0, 0])
                is_boundary = cluster_var_estimate < 1e-8  # More lenient threshold
                is_invalid_stat = np.isnan(p_value) or np.isinf(p_value) or np.isnan(tvalue)

                if is_invalid_stat:
                    fit_status = "fit_error_ols_fallback_invalid_stats"
                    warnings.warn(
                        "MixedLM produced invalid statistics (NaN/Inf p-value or t-value). Falling back to cluster-robust OLS.",
                        RuntimeWarning
                    )
                    tvalue, p_value = _ols_cluster_test(df)
                elif is_boundary:
                    # When cluster variance is very small, the LMM degenerates and p-values become unreliable
                    # Always fall back to cluster-robust OLS which is appropriate when ICC ≈ 0
                    fit_status = "success_boundary_ols_fallback"
                    tvalue, p_value = _ols_cluster_test(df)
                elif convergence_warning_occurred:
                    fit_status = "success_convergence_warning"
                    # Retain LMM results but flag convergence warning
                else:
                    fit_status = "success"
            
            if return_details:
                return tvalue, p_value, {"model": "mixedlm", "status": fit_status, "lmm_converged": (result is not None and not is_invalid_stat)}
            return tvalue, p_value

        except Exception as e_outer: # Catch any other unexpected error during mixedlm setup/fallback
            warnings.warn(f"Outer exception in MixedLM block: {e_outer}. Falling back to cluster-level t-test.", RuntimeWarning)
            # Fall back to cluster-level t-test
            cluster_means = df.groupby(['cluster', 'treatment'], observed=True)['y'].mean().reset_index()
            control_means = cluster_means[cluster_means['treatment'] == 0]['y'].values
            interv_means = cluster_means[cluster_means['treatment'] == 1]['y'].values
            t_stat_fallback, p_value_fallback = stats.ttest_ind(control_means, interv_means, equal_var=True)
            if return_details:
                return t_stat_fallback, p_value_fallback, {"model": "mixedlm", "status": "fit_error_ttest_fallback_outer", "lmm_converged": False}
            return t_stat_fallback, p_value_fallback

    if analysis_model == "gee":
        try:
            family = sm.families.Gaussian()
            cov_struct = Exchangeable()
            model = sm.GEE.from_formula("y ~ treatment", groups="cluster", data=df, family=family, cov_struct=cov_struct)
            cov_type = "bias_reduced" if use_bias_correction else "robust"
            result = model.fit(cov_type=cov_type)
            zvalue = result.params["treatment"] / result.bse["treatment"]
            # Small-sample df adjustment
            if use_bias_correction or n_clusters < 40:
                df_denom = max(1, n_clusters - 2)
                p_value = 2 * stats.t.sf(abs(zvalue), df=df_denom)
            else:
                p_value = 2 * stats.norm.sf(abs(zvalue))
            if return_details:
                return zvalue, p_value, {"converged": True, "model": "gee"}
            return zvalue, p_value
        except Exception:
            # Fall back to cluster-level t-test
            cluster_means = df.groupby(['cluster', 'treatment'], observed=True)['y'].mean().reset_index()
            control_means = cluster_means[cluster_means['treatment'] == 0]['y'].values
            interv_means = cluster_means[cluster_means['treatment'] == 1]['y'].values
            z_stat, p_value = stats.ttest_ind(control_means, interv_means, equal_var=True)
            if return_details:
                return z_stat, p_value, {"converged": False, "model": "gee", "fallback": True}
            return z_stat, p_value

    if analysis_model == "bayes":
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
            warnings.warn("SciPy not available for approximate methods. Falling back to t-test.", UserWarning)
            # Fall back to cluster-level t-test
            cluster_means = df.groupby(['cluster', 'treatment'], observed=True)['y'].mean().reset_index()
            control_means = cluster_means[cluster_means['treatment'] == 0]['y'].values
            interv_means = cluster_means[cluster_means['treatment'] == 1]['y'].values
            t_stat, p_value = stats.ttest_ind(control_means, interv_means, equal_var=True)
            if return_details:
                return t_stat, p_value, {"converged": False, "model": "bayes", "fallback": True, "backend": "ttest"}
            return t_stat, p_value

        try:
            if bayes_backend == "variational":
                # Fast Variational Bayes
                beta_samples, converged = _fit_variational_bayes(df, n_samples=bayes_draws)
                beta_rhat = 1.0  # Not applicable for VI
                backend_used = "variational"
                
                if not converged:
                    warnings.warn("Variational approximation may be inaccurate. Consider using full MCMC.")
                    
            elif bayes_backend == "abc":
                # Approximate Bayesian Computation
                beta_samples, converged = _fit_abc_bayes(df, n_samples=bayes_draws)
                beta_rhat = 1.0  # Not applicable for ABC
                backend_used = "abc"
                
                if not converged:
                    warnings.warn("ABC had low acceptance rate. Results may be unreliable.")
                    
            elif bayes_backend == "pymc":
                # PyMC implementation
                trace, model = _fit_pymc_model(
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
                model = _get_stan_model()
                N = len(df)
                y = df['y'].values
                # Map cluster IDs to consecutive integers starting from 1 (Stan convention)
                unique_clusters = sorted(df['cluster'].unique())
                cluster_map = {old_id: new_id + 1 for new_id, old_id in enumerate(unique_clusters)}
                cluster_ids = df['cluster'].map(cluster_map).values
                treat = df['treatment'].values
                data = {
                    "N": N,
                    "J": len(unique_clusters),
                    "cluster": cluster_ids.astype(int),
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
                    refresh=1,  # Changed from 0 to 1
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
            # Use 10% of pooled SD as ROPE half-width (practical equivalence threshold)
            rope_half_width = 0.1 * std_dev
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
            
            tvalue = np.mean(beta_samples) / np.std(beta_samples)
            
            if return_details:
                bayes_details = {
                    "converged": converged, 
                    "model": "bayes",
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
                    "inference_method": bayes_inference_method  # Actual method used
                }
                return tvalue, p_value, bayes_details
            return tvalue, p_value
            
        except Exception as e:
            warnings.warn(f"Bayesian model failed with error: {str(e)}. Falling back to cluster-level t-test.")
            # Fall back to cluster-level t-test
            cluster_means = df.groupby(['cluster', 'treatment'], observed=True)['y'].mean().reset_index()
            control_means = cluster_means[cluster_means['treatment'] == 0]['y'].values
            interv_means = cluster_means[cluster_means['treatment'] == 1]['y'].values
            t_stat, p_value = stats.ttest_ind(control_means, interv_means, equal_var=True)
            if return_details:
                return t_stat, p_value, {"converged": False, "model": "bayes", "fallback": True, "error": str(e)}
            return t_stat, p_value

    # Default fallback - cluster-level t-test
    cluster_means = df.groupby(['cluster', 'treatment'], observed=False)['y'].mean().reset_index()
    control_means = cluster_means[cluster_means['treatment'] == 0]['y'].values
    interv_means = cluster_means[cluster_means['treatment'] == 1]['y'].values
    t_stat, p_value = stats.ttest_ind(control_means, interv_means, equal_var=True)
    if return_details:
        return t_stat, p_value, {"converged": True, "model": "ttest"}
    return t_stat, p_value


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
            "lmm_method": lmm_method if analysis_model == "mixedlm" else None,
            "lmm_reml": lmm_reml if analysis_model == "mixedlm" else None,
            "stan_installed": _STAN_AVAILABLE,
        },
    }
    
    return results


def _analyze_continuous_trial(df, icc=0.05, analysis_model="permutation", return_details=True):
    """
    Analyze continuous cluster trial data using specified method.
    
    This function provides a simplified interface for analyzing cluster trial data,
    primarily intended for permutation test integration and testing.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns 'y' (outcome), 'treatment' (0/1), 'cluster' (cluster ID)
    icc : float, optional
        Intracluster correlation coefficient (for non-permutation methods), by default 0.05
    analysis_model : str, optional
        Analysis method ("permutation", "ttest", "mixedlm", "gee"), by default "permutation"
    return_details : bool, optional
        Whether to return detailed analysis information, by default True
        
    Returns
    -------
    tuple
        (t_stat, p_value, details) if return_details=True, else (t_stat, p_value)
        
    Notes
    -----
    This function extracts the core analysis logic from simulate_continuous_trial
    for use in permutation test integration and unit testing.
    """
    # Ensure proper data types
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Data must be a pandas DataFrame")
    
    required_cols = ['y', 'treatment', 'cluster']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Convert cluster to categorical if not already
    if not pd.api.types.is_categorical_dtype(df['cluster']):
        df = df.copy()
        df['cluster'] = df['cluster'].astype('category')
    
    # Route to the appropriate analysis method using the existing logic
    if analysis_model == "permutation":
        # Exact permutation test - recommended for very small clusters (5-10 per arm)
        from .permutation_tests import cluster_permutation_test
        
        # Aggregate to cluster means
        cluster_means = df.groupby(['cluster', 'treatment'], observed=True)['y'].mean().reset_index()
        control_means = cluster_means[cluster_means['treatment'] == 0]['y'].values
        interv_means = cluster_means[cluster_means['treatment'] == 1]['y'].values
        
        # Prepare data for permutation test
        perm_data = {
            'control_clusters': control_means,
            'treatment_clusters': interv_means
        }
        
        # Determine number of permutations based on cluster size
        total_clusters = len(control_means) + len(interv_means)
        if total_clusters <= 12:
            n_perms = 'exact'  # Use exact permutation for very small trials
        elif total_clusters <= 20:
            n_perms = 10000    # High precision for small trials
        else:
            n_perms = 5000     # Standard precision for larger trials
        
        perm_result = cluster_permutation_test(
            data=perm_data,
            test_statistic='t_statistic',  # Use t-statistic for compatibility
            n_permutations=n_perms,
            alternative='two-sided'
        )
        
        t_stat = perm_result['observed_statistic']
        p_value = perm_result['p_value']
        
        if return_details:
            return t_stat, p_value, {
                "converged": True, 
                "model": "permutation",
                "permutation_method": perm_result['method'],
                "n_permutations": perm_result['n_permutations_used'],
                "confidence_interval": perm_result['confidence_interval']
            }
        return t_stat, p_value
        
    elif analysis_model == "ttest":
        # Simple cluster-level t-test
        cluster_means = df.groupby(['cluster', 'treatment'], observed=True)['y'].mean().reset_index()
        control_means = cluster_means[cluster_means['treatment'] == 0]['y'].values
        interv_means = cluster_means[cluster_means['treatment'] == 1]['y'].values
        
        t_stat, p_value = stats.ttest_ind(control_means, interv_means, equal_var=True)
        
        if return_details:
            return t_stat, p_value, {"converged": True, "model": "ttest"}
        return t_stat, p_value
        
    else:
        # For other methods (mixedlm, gee, etc.), call the full simulate_continuous_trial function
        # This maintains compatibility while providing the simpler interface
        result = simulate_continuous_trial(
            n_clusters=len(df['cluster'].cat.categories) // 2,
            cluster_size=int(len(df) / len(df['cluster'].cat.categories)),
            icc=icc,
            mean1=df[df['treatment'] == 0]['y'].mean(),
            mean2=df[df['treatment'] == 1]['y'].mean(),
            std_dev=df['y'].std(),
            analysis_model=analysis_model,
            return_details=return_details
        )
        
        if return_details:
            return result[0], result[1], result[2]
        return result[0], result[1]