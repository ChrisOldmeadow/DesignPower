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
import statsmodels.api as sm
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
        try:
            model = sm.MixedLM.from_formula("y ~ treatment", groups="cluster", data=df)
            # Determine optimizer list
            if lmm_method == "auto":
                method_list = [
                    "lbfgs",
                    "powell",
                    "cg",
                    "bfgs",
                    "newton",
                    "nm",
                ]
            else:
                method_list = [lmm_method]

            result = None
            for m in method_list:
                try:
                    result = model.fit(reml=lmm_reml, method=m, disp=False)
                    break
                except Exception:
                    continue

            if result is None:
                raise RuntimeError("All optimizers failed for MixedLM")

            tvalue = result.tvalues["treatment"]
            if use_satterthwaite:
                df_resid = max(1, result.df_resid)
                p_value = 2 * stats.t.sf(np.abs(tvalue), df=df_resid)
            else:
                p_value = result.pvalues["treatment"]
            
            converged = not (
                np.isnan(p_value)
                or np.isinf(p_value)
                or np.isnan(tvalue)
                or float(result.cov_re.iloc[0, 0]) < 1e-6
            )
            if not converged:
                warnings.warn(
                    "MixedLM returned invalid statistics; falling back to cluster-robust OLS.",
                    RuntimeWarning,
                )
                tvalue, p_value = _ols_cluster_test(df)
                if return_details:
                    return tvalue, p_value, {"converged": converged, "model": "mixedlm", "fallback": (not converged)}
                return tvalue, p_value
            else:
                if return_details:
                    return tvalue, p_value, {"converged": converged, "model": "mixedlm"}
                return tvalue, p_value
        except Exception as e:
            # Fallback to t-test if model fails
            t_stat, p_value = stats.ttest_ind(control_data, intervention_data, equal_var=True)
            if return_details:
                return t_stat, p_value, {"converged": False, "model": "mixedlm", "fallback": True}
            return t_stat, p_value

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
    progress_callback=None,
):
    """
    Simulate a cluster RCT with continuous outcome and estimate power using the
    specified analysis model.
    """
    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)
    
    # Initialize counter for significant results
    sig_count = 0
    
    # Initialize counter for converged results
    converged_count = 0
    
    # Store p-values for all simulations
    p_values = []
    
    # Run simulations with progress bar
    if progress_callback is None:
        iterator = tqdm(range(nsim), desc="Running simulations", disable=False)
    else:
        iterator = range(nsim)

    for i in iterator:
        if analysis_model == "mixedlm":
            _, p_value, details = simulate_continuous_trial(
                n_clusters,
                cluster_size,
                icc,
                mean1,
                mean2,
                std_dev,
                analysis_model=analysis_model,
                use_satterthwaite=use_satterthwaite,
                use_bias_correction=use_bias_correction,
                bayes_draws=bayes_draws,
                bayes_warmup=bayes_warmup,
                lmm_method=lmm_method,
                lmm_reml=lmm_reml,
                return_details=True,
            )
            if details["converged"]:
                converged_count += 1
                p_values.append(p_value)
                if p_value < alpha:
                    sig_count += 1
        else:
            _, p_value = simulate_continuous_trial(
                n_clusters,
                cluster_size,
                icc,
                mean1,
                mean2,
                std_dev,
                analysis_model=analysis_model,
                use_satterthwaite=use_satterthwaite,
                use_bias_correction=use_bias_correction,
                bayes_draws=bayes_draws,
                bayes_warmup=bayes_warmup,
                lmm_method=lmm_method,
                lmm_reml=lmm_reml,
            )
            p_values.append(p_value)
            if p_value < alpha:
                sig_count += 1

        if progress_callback and ((i + 1) % max(1, nsim // 100) == 0 or (i + 1) == nsim):
            progress_callback(i + 1, nsim)

    # Calculate design effect
    deff = 1 + (cluster_size - 1) * icc
    
    # Calculate effective sample size
    n_eff = n_clusters * cluster_size / deff
    
    # Calculate empirical power
    denom = converged_count if analysis_model == "mixedlm" else nsim
    empirical_power = sig_count / denom if denom > 0 else 0.0
    
    # Format results as dictionary
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
        "converged_sims": converged_count if analysis_model == "mixedlm" else nsim,
        "failed_sims": nsim - converged_count if analysis_model == "mixedlm" else 0,
        "p_values_mean": np.mean(p_values),
        "p_values_median": np.median(p_values),
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


def sample_size_continuous_sim(
    mean1,
    mean2,
    std_dev,
    icc,
    cluster_size,
    power=0.8,
    alpha=0.05,
    nsim=1000,
    min_n=2,
    max_n=100,
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
    Find required sample size for a cluster RCT with continuous outcome using simulation.
    """
    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)
    
    # Get a rough estimate from analytical formula to use as a starting point
    # Design effect
    deff = 1 + (cluster_size - 1) * icc
    
    # Calculate effect size (standardized mean difference)
    delta = abs(mean1 - mean2) / std_dev
    
    # Critical values for significance level and power
    z_alpha = stats.norm.ppf(1 - alpha/2)
    z_beta = stats.norm.ppf(power)
    
    # Calculate required effective sample size per arm
    n_eff = 2 * ((z_alpha + z_beta) / delta)**2
    
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
        sim_results = power_continuous_sim(
            n_clusters=mid, 
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
            # This n_clusters is sufficient, try smaller
            min_adequate_n = min(min_adequate_n, mid)
            high = mid - 1
        else:
            # This n_clusters is insufficient, try larger
            low = mid + 1
    
    # Use the minimum n_clusters that meets the power requirement
    final_n_clusters = min_adequate_n
    
    # Run final simulation to get accurate power estimate
    final_results = power_continuous_sim(
        n_clusters=final_n_clusters, 
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
    
    # Extract the empirical power from final simulation
    achieved_power = final_results["power"]
    
    # Format results as dictionary
    results = {
        "n_clusters": final_n_clusters,
        "cluster_size": cluster_size,
        "total_n": 2 * final_n_clusters * cluster_size,
        "icc": icc,
        "mean1": mean1,
        "mean2": mean2,
        "difference": abs(mean1 - mean2),
        "std_dev": std_dev,
        "design_effect": deff,
        "alpha": alpha,
        "target_power": power,
        "achieved_power": achieved_power,
        "nsim": nsim,
        "p_values_mean": final_results["p_values_mean"],
        "p_values_median": final_results["p_values_median"],
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