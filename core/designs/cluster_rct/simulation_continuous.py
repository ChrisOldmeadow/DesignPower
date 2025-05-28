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

# Optional Stan support
try:
    from cmdstanpy import CmdStanModel
    _STAN_AVAILABLE = True
    _STAN_MODEL = None

    def _get_stan_model():
        global _STAN_MODEL
        if _STAN_MODEL is None:
            stan_code = """
            data {
                int<lower=1> N;
                int<lower=1> J;
                int<lower=1,upper=J> cluster[N];
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
            _STAN_MODEL = CmdStanModel(stan_code=stan_code)
        return _STAN_MODEL
except ImportError:
    _STAN_AVAILABLE = False
    _STAN_MODEL = None


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

    # Generate random cluster effects for control arm
    control_cluster_effects = np.random.normal(0, np.sqrt(var_between), n_clusters)
    
    # Generate random cluster effects for intervention arm
    intervention_cluster_effects = np.random.normal(0, np.sqrt(var_between), n_clusters)
    
    # Initialize arrays for all data points
    control_data = np.zeros(n_clusters * cluster_size)
    intervention_data = np.zeros(n_clusters * cluster_size)
    
    # Generate data for each cluster
    for i in range(n_clusters):
        # Control arm: base mean + cluster effect + individual variation
        start_idx = i * cluster_size
        end_idx = start_idx + cluster_size
        control_data[start_idx:end_idx] = mean1 + control_cluster_effects[i] + \
                                         np.random.normal(0, np.sqrt(var_within), cluster_size)
        
        # Intervention arm: base mean + cluster effect + individual variation
        intervention_data[start_idx:end_idx] = mean2 + intervention_cluster_effects[i] + \
                                              np.random.normal(0, np.sqrt(var_within), cluster_size)
    
    if analysis_model == "ttest":
        # Cluster-level analysis: aggregate to cluster means to account for ICC
        control_means = control_data.reshape(n_clusters, cluster_size).mean(axis=1)
        interv_means = intervention_data.reshape(n_clusters, cluster_size).mean(axis=1)
        t_stat, p_value = stats.ttest_ind(control_means, interv_means, equal_var=True)
        if return_details:
            return t_stat, p_value, {"converged": True, "model": "ttest"}
        return t_stat, p_value

    # Build dataframe for model-based analyses
    df_control = pd.DataFrame(
        {
            "y": control_data,
            "cluster": np.repeat(np.arange(n_clusters), cluster_size),
            "treatment": 0,
        }
    )
    df_interv = pd.DataFrame(
        {
            "y": intervention_data,
            "cluster": np.repeat(np.arange(n_clusters, 2 * n_clusters), cluster_size),
            "treatment": 1,
        }
    )
    df = pd.concat([df_control, df_interv], ignore_index=True)

    if analysis_model == "mixedlm":
        fit_status = "fit_error_ttest_fallback" # Default status if all else fails
        tvalue, p_value = np.nan, np.nan
        try:
            model = sm.MixedLM.from_formula("y ~ treatment", groups="cluster", data=df)  # Create model first
            if lmm_cov_penalty_weight > 0:
                # L2.__init__ takes 'weights' (plural); a single value scales all penalized params equally if applicable here.
                l2_penalty_obj = L2(weights=lmm_cov_penalty_weight)
                model.re_pen = l2_penalty_obj  # Assign to re_pen attribute
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

                is_boundary = float(result.cov_re.iloc[0, 0]) < 1e-6
                is_invalid_stat = np.isnan(p_value) or np.isinf(p_value) or np.isnan(tvalue)

                if is_invalid_stat:
                    fit_status = "fit_error_ols_fallback_invalid_stats"
                    warnings.warn(
                        "MixedLM produced invalid statistics (NaN/Inf p-value or t-value). Falling back to cluster-robust OLS.",
                        RuntimeWarning
                    )
                    tvalue, p_value = _ols_cluster_test(df)
                elif is_boundary:
                    fit_status = "success_boundary_condition"
                    # Retain LMM results but flag as boundary
                elif convergence_warning_occurred:
                    fit_status = "success_convergence_warning"
                    # Retain LMM results but flag convergence warning
                else:
                    fit_status = "success"
            
            if return_details:
                return tvalue, p_value, {"model": "mixedlm", "status": fit_status, "lmm_converged": (result is not None and not is_invalid_stat)}
            return tvalue, p_value

        except Exception as e_outer: # Catch any other unexpected error during mixedlm setup/fallback
            warnings.warn(f"Outer exception in MixedLM block: {e_outer}. Falling back to basic t-test.", RuntimeWarning)
            t_stat_fallback, p_value_fallback = stats.ttest_ind(control_data, intervention_data, equal_var=True)
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
            z_stat, p_value = stats.ttest_ind(control_data, intervention_data, equal_var=True)
            if return_details:
                return z_stat, p_value, {"converged": False, "model": "gee", "fallback": True}
            return z_stat, p_value

    if analysis_model == "bayes":
        if not _STAN_AVAILABLE:
            warnings.warn("cmdstanpy not available – Bayesian model not run, falling back to t-test.")
            t_stat, p_value = stats.ttest_ind(control_data, intervention_data, equal_var=True)
            if return_details:
                return t_stat, p_value, {"converged": False, "model": "bayes", "fallback": True}
            return t_stat, p_value

        try:
            model = _get_stan_model()
            N = len(control_data) + len(intervention_data)
            y = np.concatenate([control_data, intervention_data])
            cluster_ids = np.concatenate([
                np.repeat(np.arange(1, n_clusters + 1), cluster_size),
                np.repeat(np.arange(n_clusters + 1, 2 * n_clusters + 1), cluster_size),
            ])
            treat = np.concatenate([np.zeros(n_clusters * cluster_size), np.ones(n_clusters * cluster_size)])
            data = {
                "N": N,
                "J": 2 * n_clusters,
                "cluster": cluster_ids.astype(int),
                "y": y,
                "treat": treat,
            }
            fit = model.sample(
                data=data,
                chains=1,
                iter_sampling=bayes_draws,
                iter_warmup=bayes_warmup,
                show_progress=False,
                refresh=0,
            )
            beta_samples = fit.stan_variable("beta")
            tvalue = np.mean(beta_samples) / np.std(beta_samples)
            p_right = (beta_samples > 0).mean()
            p_left = (beta_samples < 0).mean()
            p_value = 2 * min(p_left, p_right)
            if return_details:
                return tvalue, p_value, {"converged": True, "model": "bayes"}
            return tvalue, p_value
        except Exception:
            t_stat, p_value = stats.ttest_ind(control_data, intervention_data, equal_var=True)
            if return_details:
                return t_stat, p_value, {"converged": False, "model": "bayes", "fallback": True}
            return t_stat, p_value

    # Default fallback
    t_stat, p_value = stats.ttest_ind(control_data, intervention_data, equal_var=True)
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
    lmm_boundary_conditions_count = 0
    lmm_ols_fallbacks_count = 0 # Covers fit_error_ols_fallback and fit_error_ols_fallback_invalid_stats
    lmm_ttest_fallbacks_outer_count = 0
    lmm_total_considered_for_power = 0

    if progress_callback is None:
        iterator = tqdm(range(nsim), desc="Running simulations", disable=False)
    else:
        iterator = range(nsim)

    for i in iterator:
        if analysis_model == "mixedlm":
            _, p_value, details = simulate_continuous_trial(
                n_clusters, cluster_size, icc, mean1, mean2, std_dev,
                analysis_model=analysis_model, use_satterthwaite=use_satterthwaite,
                use_bias_correction=use_bias_correction, bayes_draws=bayes_draws,
                bayes_warmup=bayes_warmup, lmm_method=lmm_method, lmm_reml=lmm_reml,
                lmm_cov_penalty_weight=lmm_cov_penalty_weight, lmm_fit_kwargs=lmm_fit_kwargs,
                return_details=True
            )
            status = details.get("status", "unknown")
            
            # Increment status counters
            if status == "success":
                lmm_success_count += 1
            elif status == "success_convergence_warning":
                lmm_convergence_warnings_count += 1
            elif status == "success_boundary_condition":
                lmm_boundary_conditions_count += 1
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
                bayes_warmup=bayes_warmup, lmm_method=lmm_method, lmm_reml=lmm_reml,
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
    
    # Calculate empirical power
    if analysis_model == "mixedlm":
        denom = lmm_total_considered_for_power
    else:
        # For ttest, gee, bayes, power is based on all simulations that yielded a p-value
        denom = len(p_values) 
    empirical_power = sig_count / denom if denom > 0 else 0.0
    
    # Base results dictionary
    results = {
        "power": empirical_power,
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
        results["lmm_fit_stats"] = {
            "successful_fits": lmm_success_count,
            "convergence_warnings": lmm_convergence_warnings_count,
            "boundary_conditions": lmm_boundary_conditions_count,
            "ols_fallbacks": lmm_ols_fallbacks_count,
            "ttest_fallbacks_outer": lmm_ttest_fallbacks_outer_count,
            "total_considered_for_power": lmm_total_considered_for_power,
            "total_not_converged_or_failed_lmm_path": nsim - lmm_total_considered_for_power - lmm_ttest_fallbacks_outer_count
        }
        # For backward compatibility / simplicity in other reports, provide a general converged_sims
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
        "lmm_method": lmm_method if analysis_model == "mixedlm" else None,
        "lmm_reml": lmm_reml if analysis_model == "mixedlm" else None,
        "stan_installed": _STAN_AVAILABLE,
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
        while current_low_k <= current_high_k:
            if iterations_k >= max_iterations_k:
                warning_message = (
                    f"Binary search for n_clusters (k) exceeded maximum iterations ({max_iterations_k}). "
                    f"Using k={min_adequate_k if min_adequate_k <= max_n_clusters else max_n_clusters} based on current findings."
                )
                print(f"Warning: {warning_message}")
                break
            iterations_k += 1
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
                bayes_warmup=bayes_warmup, lmm_method=lmm_method, lmm_reml=lmm_reml,
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
        while current_low_m <= current_high_m:
            if iterations_m >= max_iterations_m:
                warning_message = (
                    f"Binary search for cluster_size (m) exceeded maximum iterations ({max_iterations_m}). "
                    f"Using m={min_adequate_m if min_adequate_m <= max_cluster_size else max_cluster_size} based on current findings."
                )
                print(f"Warning: {warning_message}")
                break
            iterations_m += 1
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
                bayes_warmup=bayes_warmup, lmm_method=lmm_method, lmm_reml=lmm_reml,
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
        bayes_warmup=bayes_warmup, lmm_method=lmm_method, lmm_reml=lmm_reml,
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
    
    while iteration < max_iterations and high - low > precision:
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