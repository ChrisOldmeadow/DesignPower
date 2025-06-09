"""
Core simulation functions for cluster randomized controlled trials with continuous outcomes.

This module provides the main simulation engine for generating and analyzing cluster RCT data.
"""

import numpy as np
import pandas as pd
import warnings
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.regression.mixed_linear_model import MixedLM
from statsmodels.base._penalties import L2
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.genmod.cov_struct import Exchangeable
from scipy import stats

from .bayesian_methods import _get_stan_model, _fit_pymc_model, _fit_variational_bayes, _fit_abc_bayes, _PYMC_AVAILABLE, _STAN_AVAILABLE, _SCIPY_AVAILABLE
from .analysis_utils import _ols_cluster_test


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
                tvalue, p_value, _ = _ols_cluster_test(df)
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
                    tvalue, p_value, _ = _ols_cluster_test(df)
                elif is_boundary:
                    # When cluster variance is very small, the LMM degenerates and p-values become unreliable
                    # Always fall back to cluster-robust OLS which is appropriate when ICC ≈ 0
                    fit_status = "success_boundary_ols_fallback"
                    tvalue, p_value, _ = _ols_cluster_test(df)
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
                import arviz as az
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