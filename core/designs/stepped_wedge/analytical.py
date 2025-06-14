"""
Analytical methods for stepped wedge cluster randomized trials.

This module provides functions for power analysis and sample size calculation
for stepped wedge designs using analytical formulas from Hussey & Hughes (2007).

Reference:
Hussey MA, Hughes JP. Design and analysis of stepped wedge cluster randomized trials. 
Contemporary Clinical Trials 2007; 28: 182-191.
"""
import numpy as np
from scipy import stats
from scipy.stats import norm


def hussey_hughes_power_continuous(clusters, steps, individuals_per_cluster, 
                                 icc, cluster_autocorr, treatment_effect, 
                                 std_dev, alpha=0.05):
    """
    Calculate power for stepped wedge design using Hussey & Hughes analytical method.
    
    Parameters
    ----------
    clusters : int
        Number of clusters (K)
    steps : int
        Number of time steps including baseline (T)
    individuals_per_cluster : int
        Number of individuals per cluster per time step (m)
    icc : float
        Intracluster correlation coefficient (ρ)
    cluster_autocorr : float
        Cluster autocorrelation coefficient (CAC/ρ_c)
    treatment_effect : float
        Effect size of the intervention (δ)
    std_dev : float
        Standard deviation of outcome (σ)
    alpha : float, optional
        Significance level, by default 0.05
    
    Returns
    -------
    dict
        Dictionary containing power and calculation details
    """
    # Number of intervention steps (excluding baseline)
    intervention_steps = steps - 1
    
    # Total observations
    total_n = clusters * steps * individuals_per_cluster
    
    # Design matrix construction following Hussey & Hughes
    # Each cluster gets assigned to one intervention step
    clusters_per_step = clusters // intervention_steps
    remaining_clusters = clusters % intervention_steps
    
    # Create exposure matrix X (clusters × steps)
    X = np.zeros((clusters, steps))
    
    cluster_idx = 0
    for step in range(1, steps):  # Steps 1 to T-1 (excluding baseline step 0)
        n_clusters_this_step = clusters_per_step
        if step <= remaining_clusters:
            n_clusters_this_step += 1
        
        for _ in range(n_clusters_this_step):
            if cluster_idx < clusters:
                # This cluster starts intervention at this step
                X[cluster_idx, step:] = 1
                cluster_idx += 1
    
    # Calculate design parameters following Hussey & Hughes formulation
    # Variance components
    sigma2_e = std_dev**2 * (1 - icc)  # Individual-level variance
    sigma2_c = std_dev**2 * icc * (1 - cluster_autocorr)  # Cluster-period variance  
    sigma2_s = std_dev**2 * icc * cluster_autocorr  # Cluster-level variance
    
    # Design effect calculations
    # Individual correlation matrix structure
    n_per_cluster_period = individuals_per_cluster
    
    # Variance of cluster-period means
    var_cluster_period_mean = (sigma2_e / n_per_cluster_period) + sigma2_c + sigma2_s
    
    # Correlation between cluster-period means
    # Same cluster, different periods
    rho_between_periods = sigma2_s / var_cluster_period_mean
    
    # Create variance-covariance matrix for cluster-period means
    # This follows the Hussey & Hughes approach for the variance of the treatment effect estimator
    
    # Number of cluster-periods under control and intervention
    total_cluster_periods = clusters * steps
    n_control_periods = np.sum(X == 0)
    n_intervention_periods = np.sum(X == 1)
    
    # Effective sample sizes accounting for correlation structure
    # Following Hussey & Hughes equation for variance of treatment effect estimator
    
    # Calculate the variance multiplier due to correlation structure
    # This accounts for within-cluster correlation across time periods
    if cluster_autocorr > 0 and steps > 1:
        # Correlation adjustment for multiple periods per cluster
        correlation_adjustment = 1 + (steps - 1) * rho_between_periods
    else:
        correlation_adjustment = 1
    
    # Effective variance for the difference in means
    var_treatment_effect = var_cluster_period_mean * (
        (1 / n_control_periods) + (1 / n_intervention_periods)
    ) * correlation_adjustment
    
    # Standard error of treatment effect
    se_treatment_effect = np.sqrt(var_treatment_effect)
    
    # Calculate power using normal approximation
    z_alpha = norm.ppf(1 - alpha/2)  # Two-sided test
    z_beta = (abs(treatment_effect) / se_treatment_effect) - z_alpha
    power = norm.cdf(z_beta)
    
    return {
        "power": power,
        "se_treatment_effect": se_treatment_effect,
        "var_treatment_effect": var_treatment_effect,
        "correlation_adjustment": correlation_adjustment,
        "n_control_periods": n_control_periods,
        "n_intervention_periods": n_intervention_periods,
        "parameters": {
            "clusters": clusters,
            "steps": steps,
            "individuals_per_cluster": individuals_per_cluster,
            "total_n": total_n,
            "icc": icc,
            "cluster_autocorr": cluster_autocorr,
            "treatment_effect": treatment_effect,
            "std_dev": std_dev,
            "alpha": alpha,
            "method": "Hussey & Hughes Analytical"
        }
    }


def hussey_hughes_sample_size_continuous(target_power, treatment_effect, std_dev,
                                       icc, cluster_autocorr, steps, 
                                       individuals_per_cluster, alpha=0.05):
    """
    Calculate required number of clusters for stepped wedge design using Hussey & Hughes method.
    
    Parameters
    ----------
    target_power : float
        Desired statistical power (e.g., 0.80)
    treatment_effect : float
        Effect size of the intervention
    std_dev : float
        Standard deviation of outcome
    icc : float
        Intracluster correlation coefficient
    cluster_autocorr : float
        Cluster autocorrelation coefficient
    steps : int
        Number of time steps including baseline
    individuals_per_cluster : int
        Number of individuals per cluster per time step
    alpha : float, optional
        Significance level, by default 0.05
    
    Returns
    -------
    dict
        Dictionary containing required number of clusters
    """
    # Binary search for required number of clusters
    min_clusters = 2
    max_clusters = 1000
    
    # Ensure we can distribute clusters across intervention steps
    intervention_steps = steps - 1
    if min_clusters < intervention_steps:
        min_clusters = intervention_steps
    
    while max_clusters - min_clusters > 1:
        test_clusters = (min_clusters + max_clusters) // 2
        
        # Ensure clusters can be distributed across intervention steps
        if test_clusters < intervention_steps:
            min_clusters = intervention_steps
            continue
            
        result = hussey_hughes_power_continuous(
            clusters=test_clusters,
            steps=steps,
            individuals_per_cluster=individuals_per_cluster,
            icc=icc,
            cluster_autocorr=cluster_autocorr,
            treatment_effect=treatment_effect,
            std_dev=std_dev,
            alpha=alpha
        )
        
        if result["power"] >= target_power:
            max_clusters = test_clusters
        else:
            min_clusters = test_clusters
    
    # Final calculation with the determined number of clusters
    final_result = hussey_hughes_power_continuous(
        clusters=max_clusters,
        steps=steps,
        individuals_per_cluster=individuals_per_cluster,
        icc=icc,
        cluster_autocorr=cluster_autocorr,
        treatment_effect=treatment_effect,
        std_dev=std_dev,
        alpha=alpha
    )
    
    return {
        "clusters": max_clusters,
        "total_n": max_clusters * steps * individuals_per_cluster,
        "achieved_power": final_result["power"],
        "target_power": target_power,
        "parameters": final_result["parameters"]
    }


def hussey_hughes_power_binary(clusters, steps, individuals_per_cluster,
                             icc, cluster_autocorr, p_control, p_intervention,
                             alpha=0.05):
    """
    Calculate power for stepped wedge design with binary outcome using Hussey & Hughes method.
    
    Parameters
    ----------
    clusters : int
        Number of clusters
    steps : int
        Number of time steps including baseline
    individuals_per_cluster : int
        Number of individuals per cluster per time step
    icc : float
        Intracluster correlation coefficient
    cluster_autocorr : float
        Cluster autocorrelation coefficient
    p_control : float
        Proportion in control condition
    p_intervention : float
        Proportion in intervention condition
    alpha : float, optional
        Significance level, by default 0.05
    
    Returns
    -------
    dict
        Dictionary containing power and calculation details
    """
    # Convert to continuous scale using arcsine transformation
    # This is a common approach for binary outcomes in stepped wedge designs
    
    # Arcsine transformation
    theta_control = np.arcsin(np.sqrt(p_control))
    theta_intervention = np.arcsin(np.sqrt(p_intervention))
    treatment_effect = theta_intervention - theta_control
    
    # Variance of arcsine transformation
    # Approximate variance is 1/(4*n) for arcsine(sqrt(p))
    var_arcsine = 1 / (4 * individuals_per_cluster)
    std_dev = np.sqrt(var_arcsine)
    
    # Use the continuous method with transformed parameters
    result = hussey_hughes_power_continuous(
        clusters=clusters,
        steps=steps,
        individuals_per_cluster=individuals_per_cluster,
        icc=icc,
        cluster_autocorr=cluster_autocorr,
        treatment_effect=treatment_effect,
        std_dev=std_dev,
        alpha=alpha
    )
    
    # Update outcome type in parameters
    result["parameters"]["p_control"] = p_control
    result["parameters"]["p_intervention"] = p_intervention
    result["parameters"]["arcsine_transformation"] = True
    
    return result


def hussey_hughes_sample_size_binary(target_power, p_control, p_intervention,
                                   icc, cluster_autocorr, steps,
                                   individuals_per_cluster, alpha=0.05):
    """
    Calculate required number of clusters for stepped wedge design with binary outcome.
    
    Parameters
    ----------
    target_power : float
        Desired statistical power
    p_control : float
        Proportion in control condition
    p_intervention : float
        Proportion in intervention condition
    icc : float
        Intracluster correlation coefficient
    cluster_autocorr : float
        Cluster autocorrelation coefficient
    steps : int
        Number of time steps including baseline
    individuals_per_cluster : int
        Number of individuals per cluster per time step
    alpha : float, optional
        Significance level, by default 0.05
    
    Returns
    -------
    dict
        Dictionary containing required number of clusters
    """
    # Convert to continuous scale using arcsine transformation
    theta_control = np.arcsin(np.sqrt(p_control))
    theta_intervention = np.arcsin(np.sqrt(p_intervention))
    treatment_effect = theta_intervention - theta_control
    
    # Variance of arcsine transformation
    var_arcsine = 1 / (4 * individuals_per_cluster)
    std_dev = np.sqrt(var_arcsine)
    
    # Use the continuous sample size method
    result = hussey_hughes_sample_size_continuous(
        target_power=target_power,
        treatment_effect=treatment_effect,
        std_dev=std_dev,
        icc=icc,
        cluster_autocorr=cluster_autocorr,
        steps=steps,
        individuals_per_cluster=individuals_per_cluster,
        alpha=alpha
    )
    
    # Update parameters to include binary outcome information
    result["parameters"]["p_control"] = p_control
    result["parameters"]["p_intervention"] = p_intervention
    result["parameters"]["arcsine_transformation"] = True
    
    return result