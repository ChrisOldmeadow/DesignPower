"""Analytical methods for cluster randomized controlled trials with binary outcomes.

This module provides functions for sample size calculation, power analysis,
and minimum detectable effect estimation for cluster randomized controlled trials
with binary outcomes using analytical formulas.

Features include:
- Support for equal and unequal cluster sizes
- Multiple effect size specifications (risk difference, risk ratio, odds ratio)
- ICC conversion between linear and logit scales
- Validation and warnings for small numbers of clusters
"""

import math
import numpy as np
from scipy import stats
from tqdm import tqdm
from .cluster_utils import (design_effect_equal, design_effect_unequal, 
                           validate_cluster_parameters, convert_effect_measures)


def power_binary(n_clusters, cluster_size, icc, p1, p2, alpha=0.05, cv_cluster_size=0, 
                effect_measure=None, effect_value=None, cluster_sizes=None):
    """
    Calculate power for a cluster randomized controlled trial with binary outcome.
    
    Parameters
    ----------
    n_clusters : int
        Number of clusters per arm
    cluster_size : int or float
        Average number of individuals per cluster
    icc : float
        Intracluster correlation coefficient
    p1 : float
        Proportion in control group
    p2 : float, optional
        Proportion in intervention group. If None, it will be calculated from
        effect_measure and effect_value.
    alpha : float, optional
        Significance level, by default 0.05
    cv_cluster_size : float, optional
        Coefficient of variation of cluster sizes, by default 0
    effect_measure : str, optional
        Type of effect measure: 'risk_difference', 'risk_ratio', or 'odds_ratio'.
        Used with effect_value if p2 is None.
    effect_value : float, optional
        Value of the effect measure. Used with effect_measure if p2 is None.
    cluster_sizes : list or array, optional
        List of cluster sizes. If provided, cv_cluster_size will be calculated.
    
    Returns
    -------
    dict
        Dictionary containing the calculated power and input parameters
    """
    # Calculate p2 from effect measure if needed
    if p2 is None and effect_measure is not None and effect_value is not None:
        effect_info = convert_effect_measures(p1, effect_measure, effect_value)
        p2 = effect_info['p2']
    
    # Calculate CV from cluster sizes if provided
    if cluster_sizes is not None:
        cv_cluster_size = np.std(cluster_sizes) / np.mean(cluster_sizes)
    
    # Design effect (accounting for unequal cluster sizes if CV > 0)
    if cv_cluster_size > 0:
        deff = design_effect_unequal(cluster_size, icc, cv_cluster_size)
    else:
        deff = design_effect_equal(cluster_size, icc)
    
    # Effective sample size per arm
    n_eff = n_clusters * cluster_size / deff
    
    # Calculate pooled proportion
    p_pooled = (p1 + p2) / 2
    
    # Calculate standard error under H0
    se = math.sqrt(2 * p_pooled * (1 - p_pooled) / n_eff)
    
    # Calculate z-score for given alpha
    z_alpha = stats.norm.ppf(1 - alpha/2)
    
    # Calculate non-centrality parameter
    effect = abs(p2 - p1)
    ncp = effect / se
    
    # Calculate power
    power = 1 - stats.norm.cdf(z_alpha - ncp) + stats.norm.cdf(-z_alpha - ncp)
    
    # Calculate additional metrics
    risk_ratio = p2 / p1 if p1 > 0 else float('inf')
    odds_ratio = (p2 / (1 - p2)) / (p1 / (1 - p1)) if p1 < 1 and p2 < 1 else float('inf')
    
    # Validate number of clusters and get warnings
    validation = validate_cluster_parameters(2 * n_clusters)
    
    # Format results as dictionary
    results = {
        "power": power,
        "n_clusters": n_clusters,
        "cluster_size": cluster_size,
        "total_n": 2 * n_clusters * cluster_size,
        "icc": icc,
        "p1": p1,
        "p2": p2,
        "risk_difference": abs(p2 - p1),
        "risk_ratio": risk_ratio,
        "odds_ratio": odds_ratio,
        "design_effect": deff,
        "effective_n": n_eff,
        "alpha": alpha,
        "cv_cluster_size": cv_cluster_size
    }
    
    # Add validation warnings
    if not validation["valid"]:
        results["warnings"] = validation["warnings"]
    
    return results


def sample_size_binary(p1, p2=None, icc=0.01, cluster_size=None, n_clusters_fixed=None, power=0.8, alpha=0.05,
                      cv_cluster_size=0, effect_measure=None, effect_value=None, cluster_sizes=None):
    """
    Calculate required sample size (either number of clusters or cluster size per arm)
    for a cluster randomized controlled trial with binary outcome.

    Exactly one of 'cluster_size' or 'n_clusters_fixed' must be specified.
    If 'cluster_size' is specified, the function calculates the required 'n_clusters'.
    If 'n_clusters_fixed' is specified, the function calculates the required 'cluster_size'.
    
    Parameters
    ----------
    p1 : float
        Proportion in control group
    p2 : float, optional
        Proportion in intervention group. If None, it will be calculated from
        effect_measure and effect_value.
    icc : float, optional
        Intracluster correlation coefficient, by default 0.01
    cluster_size : int or float, optional
        Average number of individuals per cluster. Specify this to calculate n_clusters.
    n_clusters_fixed : int, optional
        Number of clusters per arm. Specify this to calculate cluster_size.
    power : float, optional
        Desired statistical power (1 - beta), by default 0.8
    alpha : float, optional
        Significance level, by default 0.05
    cv_cluster_size : float, optional
        Coefficient of variation of cluster sizes, by default 0
    effect_measure : str, optional
        Type of effect measure: 'risk_difference', 'risk_ratio', or 'odds_ratio'.
        Used with effect_value if p2 is None.
    effect_value : float, optional
        Value of the effect measure. Used with effect_measure if p2 is None.
    cluster_sizes : list or array, optional
        List of cluster sizes. If provided, cv_cluster_size will be calculated.
    
    Returns
    -------
    dict
        Dictionary containing the required sample size parameters and input parameters
    """
    # Calculate p2 from effect measure if needed
    if p2 is None and effect_measure is not None and effect_value is not None:
        effect_info = convert_effect_measures(p1, effect_measure, effect_value)
        p2 = effect_info['p2']
    elif p2 is None:
        raise ValueError("Either p2 or both effect_measure and effect_value must be provided")

    # Validate that exactly one of cluster_size or n_clusters_fixed is specified
    if not ( (cluster_size is not None and n_clusters_fixed is None) or 
             (cluster_size is None and n_clusters_fixed is not None) ):
        raise ValueError("Exactly one of 'cluster_size' or 'n_clusters_fixed' must be specified.")
    
    # Calculate CV from cluster sizes if provided
    if cluster_sizes is not None:
        cv_cluster_size = np.std(cluster_sizes) / np.mean(cluster_sizes)
    
    # Critical values for significance level and power
    z_alpha = stats.norm.ppf(1 - alpha/2)
    z_beta = stats.norm.ppf(power)
    
    # Calculate required effective sample size per arm for individual RCT
    # Following Donner & Klar methodology: use control group variance for sample size
    # This is the standard approach in many cluster trial references
    effect = abs(p2 - p1)
    
    # Use null (control group) variance - matches Donner & Klar approach
    var_null = p1 * (1 - p1)
    n_eff = (2 * var_null * (z_alpha + z_beta)**2) / (effect**2)
    
    calculated_n_clusters = None
    calculated_cluster_size = None
    warning_message = None
    final_deff = None
    
    if cluster_size is not None: # Solve for n_clusters
        # Design effect (accounting for unequal cluster sizes if CV > 0)
        if cv_cluster_size > 0:
            final_deff = design_effect_unequal(cluster_size, icc, cv_cluster_size)
        else:
            final_deff = design_effect_equal(cluster_size, icc)
        
        # Calculate required number of clusters per arm (accounting for design effect)
        calculated_n_clusters = math.ceil(n_eff * final_deff / cluster_size)
        calculated_cluster_size = cluster_size
        
    elif n_clusters_fixed is not None: # Solve for cluster_size
        if not isinstance(n_clusters_fixed, int) or n_clusters_fixed < 2:
            raise ValueError("Number of clusters per arm (k) must be an integer and at least 2.")
        calculated_n_clusters = n_clusters_fixed
        
        # Method 2: Direct algebraic solution for fixed k
        # From: k×m = n_eff × [1 + (m-1)×ICC]  
        # Solve: m = n_eff × (1 - ICC) / (k - n_eff × ICC)
        denominator = calculated_n_clusters - (n_eff * icc)
        
        if denominator <= 1e-9:
            # When algebraic solution fails, try Method 1: Iterative approach
            warning_message = (
                f"Direct algebraic solution failed for {calculated_n_clusters} clusters per arm and ICC={icc}. "
                f"Attempting iterative solution..."
            )
            
            # Method 1: Iterative calculation starting with reasonable initial guess
            m_guess = max(10, n_eff / calculated_n_clusters)  # Initial guess
            converged = False
            max_iterations = 50
            tolerance = 0.01
            max_cluster_size = 10000  # Practical upper limit
            
            for iteration in range(max_iterations):
                # Calculate design effect with current guess (assuming equal cluster sizes)
                deff_guess = 1 + (m_guess - 1) * icc
                
                # Calculate total N needed per arm
                total_n_needed = n_eff * deff_guess
                
                # Update cluster size estimate
                m_new = total_n_needed / calculated_n_clusters
                
                # Check for divergence or impractical cluster sizes
                if m_new > max_cluster_size:
                    converged = False
                    break
                
                # Check convergence
                if abs(m_new - m_guess) < tolerance:
                    converged = True
                    break
                
                # Check for oscillation (if change is very small, consider converged)
                if iteration > 5 and abs(m_new - m_guess) / max(m_new, m_guess) < tolerance:
                    converged = True
                    break
                    
                m_guess = m_new
            
            if converged:
                calculated_cluster_size = math.ceil(m_guess)
                final_deff = 1 + (calculated_cluster_size - 1) * icc
                warning_message = (
                    f"Iterative solution converged after {iteration + 1} iterations. "
                    f"Note: With only {calculated_n_clusters} clusters per arm and ICC={icc}, "
                    f"large cluster sizes ({calculated_cluster_size}) are required."
                )
            else:
                calculated_cluster_size = float('inf')
                # Provide more specific guidance based on the fundamental limitation
                max_feasible_icc = calculated_n_clusters / n_eff
                warning_message = (
                    f"Cannot achieve target power with {calculated_n_clusters} clusters per arm and ICC={icc}. "
                    f"For {calculated_n_clusters} clusters, maximum feasible ICC ≈ {max_feasible_icc:.3f}. "
                    f"Consider: (1) increasing clusters (k ≥ {math.ceil(n_eff * icc) + 2}), "
                    f"(2) decreasing ICC (< {max_feasible_icc:.3f}), or (3) relaxing power requirements."
                )
                final_deff = float('inf')
        else:
            # Method 2: Direct algebraic solution works
            cluster_size_float = (n_eff * (1 - icc)) / denominator
            calculated_cluster_size = math.ceil(cluster_size_float)
            if calculated_cluster_size < 2:
                calculated_cluster_size = 2 # Minimum practical cluster size
            final_deff = 1 + (calculated_cluster_size - 1) * icc
    
    # Recalculate achieved power with the determined parameters
    achieved_power = 0.0
    if calculated_n_clusters != float('inf') and calculated_cluster_size != float('inf'):
        achieved_power = power_binary(
            calculated_n_clusters, calculated_cluster_size, icc, p1, p2, alpha, 
            cv_cluster_size=cv_cluster_size
        )["power"]
    
    # Calculate additional metrics
    risk_ratio = p2 / p1 if p1 > 0 else float('inf')
    odds_ratio = (p2 / (1 - p2)) / (p1 / (1 - p1)) if p1 < 1 and p2 < 1 else float('inf')
    
    # Calculate total sample sizes
    total_n = float('inf')
    total_clusters = float('inf')
    if calculated_n_clusters != float('inf') and calculated_cluster_size != float('inf'):
        total_n = 2 * calculated_n_clusters * calculated_cluster_size
        total_clusters = 2 * calculated_n_clusters
    
    # Validate number of clusters and get warnings
    validation = validate_cluster_parameters(total_clusters if total_clusters != float('inf') else 0)
    
    # Format results as dictionary
    results = {
        "n_clusters": calculated_n_clusters,
        "cluster_size": calculated_cluster_size,
        "total_n": total_n,
        "total_clusters": total_clusters,  # Total clusters across both arms
        "p1": p1,
        "p2": p2,
        "risk_difference": abs(p2 - p1),
        "risk_ratio": risk_ratio,
        "odds_ratio": odds_ratio,
        "icc": icc,
        "design_effect": final_deff,
        "alpha": alpha,
        "target_power": power,
        "achieved_power": achieved_power,
        "cv_cluster_size": cv_cluster_size
    }
    
    # Add validation warnings
    if not validation["valid"]:
        results["warnings"] = validation["warnings"]
    
    # Add any calculation warnings
    if warning_message:
        if "warnings" in results:
            results["warnings"].append(warning_message)
        else:
            results["warnings"] = [warning_message]
    
    return results


def min_detectable_effect_binary(n_clusters, cluster_size, icc, p1, power=0.8, alpha=0.05,
                              cv_cluster_size=0, cluster_sizes=None, effect_measure='risk_difference'):
    """
    Calculate minimum detectable effect for a cluster RCT with binary outcome.
    
    Parameters
    ----------
    n_clusters : int
        Number of clusters per arm
    cluster_size : int or float
        Average number of individuals per cluster
    icc : float
        Intracluster correlation coefficient
    p1 : float
        Proportion in control group
    power : float, optional
        Desired statistical power (1 - beta), by default 0.8
    alpha : float, optional
        Significance level, by default 0.05
    cv_cluster_size : float, optional
        Coefficient of variation of cluster sizes, by default 0
    cluster_sizes : list or array, optional
        List of cluster sizes. If provided, cv_cluster_size will be calculated.
    effect_measure : str, optional
        Type of effect measure to return: 'risk_difference', 'risk_ratio', or 'odds_ratio'.
        Default is 'risk_difference'
    
    Returns
    -------
    dict
        Dictionary containing the minimum detectable effect and input parameters
    """
    # Calculate CV from cluster sizes if provided
    if cluster_sizes is not None:
        cv_cluster_size = np.std(cluster_sizes) / np.mean(cluster_sizes)
    
    # Design effect (accounting for unequal cluster sizes if CV > 0)
    if cv_cluster_size > 0:
        deff = design_effect_unequal(cluster_size, icc, cv_cluster_size)
    else:
        deff = design_effect_equal(cluster_size, icc)
    
    # Effective sample size per arm
    n_eff = n_clusters * cluster_size / deff
    
    # Critical values for significance level and power
    z_alpha = stats.norm.ppf(1 - alpha/2)
    z_beta = stats.norm.ppf(power)
    
    # Calculate minimum detectable effect (risk difference)
    mde = (z_alpha + z_beta) * math.sqrt(2 * p1 * (1 - p1) / n_eff)
    
    # Ensure MDE is within bounds (no proportion > 1)
    p2 = min(p1 + mde, 0.9999)  # Avoid exactly 1.0 for calculation purposes
    
    # Calculate pooled proportion with the derived p2
    p_pooled = (p1 + p2) / 2
    
    # Recalculate MDE with pooled proportion (more accurate)
    mde_refined = (z_alpha + z_beta) * math.sqrt(2 * p_pooled * (1 - p_pooled) / n_eff)
    p2_refined = min(p1 + mde_refined, 0.9999)  # Refine p2
    
    # Calculate additional metrics
    risk_ratio = p2_refined / p1 if p1 > 0 else float('inf')
    odds_ratio = (p2_refined / (1 - p2_refined)) / (p1 / (1 - p1)) if p1 < 1 and p2_refined < 1 else float('inf')
    
    # Validate number of clusters and get warnings
    validation = validate_cluster_parameters(2 * n_clusters)
    
    # Determine which effect measure to return as the primary MDE
    if effect_measure == 'risk_difference':
        mde_primary = mde_refined
    elif effect_measure == 'risk_ratio':
        mde_primary = risk_ratio
    elif effect_measure == 'odds_ratio':
        mde_primary = odds_ratio
    else:
        # Default to risk difference
        mde_primary = mde_refined
        effect_measure = 'risk_difference'
    
    # Format results as dictionary
    results = {
        "mde": mde_primary,
        "effect_measure": effect_measure,
        "n_clusters": n_clusters,
        "cluster_size": cluster_size,
        "total_n": 2 * n_clusters * cluster_size,
        "p1": p1,
        "p2": p2_refined,
        "risk_difference": mde_refined,
        "risk_ratio": risk_ratio,
        "odds_ratio": odds_ratio,
        "icc": icc,
        "design_effect": deff,
        "effective_n": n_eff,
        "alpha": alpha,
        "power": power,
        "cv_cluster_size": cv_cluster_size
    }
    
    # Add validation warnings
    if not validation["valid"]:
        results["warnings"] = validation["warnings"]
    
    # Verify that the calculated power matches the requested power
    achieved_power = power_binary(
        n_clusters, cluster_size, icc, p1, p2_refined, alpha, 
        cv_cluster_size=cv_cluster_size
    )["power"]
    results["achieved_power"] = achieved_power
    
    return results


# ===========================
# Permutation Test Functions  
# ===========================

def power_binary_permutation(n_clusters, cluster_size, icc, p1, p2, alpha=0.05, cv_cluster_size=0.0, n_simulations=1000, progress_callback=None):
    """
    Calculate power for a cluster RCT using permutation tests (exact inference).
    
    This provides exact p-values without distributional assumptions, ideal for 
    small cluster trials (5-15 clusters per arm).
    
    Parameters
    ----------
    n_clusters : int
        Number of clusters per arm
    cluster_size : int
        Average number of individuals per cluster
    icc : float
        Intracluster correlation coefficient
    p1 : float
        Proportion in control group
    p2 : float
        Proportion in intervention group
    alpha : float, optional
        Significance level, by default 0.05
    cv_cluster_size : float, optional
        Coefficient of variation for cluster sizes, by default 0.0
    n_simulations : int, optional
        Number of simulations to estimate power, by default 1000
    progress_callback : function, optional
        A function to call with progress updates during simulation.
        
    Returns
    -------
    dict
        Dictionary containing power estimate and study details
    """
    from .permutation_tests import cluster_permutation_test_binary
    
    # Calculate design effect
    deff = 1 + (cluster_size - 1) * icc
    
    np.random.seed(42)  # For reproducibility
    
    significant_count = 0
    
    if progress_callback is None:
        iterator = tqdm(range(n_simulations), desc="Running permutation power simulations (binary)", disable=n_simulations < 100)
    else:
        iterator = range(n_simulations)
    
    for i in iterator:
        # Generate cluster-level proportions for beta-binomial model
        if icc <= 0:
            # No clustering effect
            control_props = np.full(n_clusters, p1)
            treatment_props = np.full(n_clusters, p2)
        else:
            # Control arm
            if p1 == 0.0:
                control_props = np.zeros(n_clusters)
            elif p1 == 1.0:
                control_props = np.ones(n_clusters)
            else:
                alpha1 = p1 * (1.0 - icc) / icc
                beta1 = (1.0 - p1) * (1.0 - icc) / icc
                control_props = np.random.beta(alpha1, beta1, n_clusters)
            
            # Treatment arm
            if p2 == 0.0:
                treatment_props = np.zeros(n_clusters)
            elif p2 == 1.0:
                treatment_props = np.ones(n_clusters)
            else:
                alpha2 = p2 * (1.0 - icc) / icc
                beta2 = (1.0 - p2) * (1.0 - icc) / icc
                treatment_props = np.random.beta(alpha2, beta2, n_clusters)
        
        # Perform permutation test
        result = cluster_permutation_test_binary(
            control_clusters=control_props,
            treatment_clusters=treatment_props,
            n_permutations='auto',  # Automatically choose exact or Monte Carlo
            confidence_level=1-alpha
        )
        
        if result['p_value'] < alpha:
            significant_count += 1
        
        # Update progress if callback provided
        if progress_callback and ((i + 1) % max(1, n_simulations // 100) == 0 or (i + 1) == n_simulations):
            progress_callback(i + 1, n_simulations)
    
    power = significant_count / n_simulations
    
    # Format results
    results = {
        "power": power,
        "n_clusters": n_clusters,
        "cluster_size": cluster_size,
        "total_n": 2 * n_clusters * cluster_size,
        "icc": icc,
        "p1": p1,
        "p2": p2,
        "risk_difference": abs(p2 - p1),
        "risk_ratio": p2 / p1 if p1 > 0 else float('inf'),
        "odds_ratio": (p2 / (1 - p2)) / (p1 / (1 - p1)) if p1 < 1 and p2 < 1 else float('inf'),
        "design_effect": deff,
        "effective_n": n_clusters * cluster_size / deff,
        "alpha": alpha,
        "cv_cluster_size": cv_cluster_size,
        "method": "permutation_test",
        "n_simulations": n_simulations
    }
    
    return results


def sample_size_binary_permutation(p1, p2, icc, cluster_size, power=0.8, alpha=0.05, cv_cluster_size=0.0, max_clusters=100):
    """
    Calculate required sample size for binary cluster RCT using permutation test power estimation.
    
    Parameters
    ----------
    p1 : float
        Proportion in control group
    p2 : float
        Proportion in intervention group
    icc : float
        Intracluster correlation coefficient
    cluster_size : int
        Average cluster size
    power : float, optional
        Target power, by default 0.8
    alpha : float, optional
        Significance level, by default 0.05
    cv_cluster_size : float, optional
        Coefficient of variation for cluster sizes, by default 0.0
    max_clusters : int, optional
        Maximum clusters to search, by default 100
        
    Returns
    -------
    dict
        Dictionary containing required sample size and study details
    """
    # Solve for number of clusters
    for n_clusters in tqdm(range(2, max_clusters + 1), desc="Searching optimal cluster count (permutation)", disable=max_clusters < 20):
        result = power_binary_permutation(
            n_clusters=n_clusters,
            cluster_size=cluster_size,
            icc=icc,
            p1=p1,
            p2=p2,
            alpha=alpha,
            cv_cluster_size=cv_cluster_size,
            n_simulations=500  # Reduced for sample size search
        )
        
        if result["power"] >= power:
            result.update({
                "target_power": power,
                "achieved_power": result["power"],
                "method": "permutation_test_sample_size"
            })
            return result
    
    # If no solution found within max_clusters
    return {
        "n_clusters": max_clusters,
        "cluster_size": cluster_size,
        "total_n": 2 * max_clusters * cluster_size,
        "power": result["power"],
        "target_power": power,
        "warning": f"Target power not achieved within {max_clusters} clusters per arm",
        "method": "permutation_test_sample_size"
    }


def min_detectable_effect_binary_permutation(n_clusters, cluster_size, icc, p1, power=0.8, alpha=0.05, 
                                           cv_cluster_size=0.0, effect_measure='risk_difference',
                                           precision=0.01, max_iterations=20):
    """
    Calculate minimum detectable effect for binary cluster RCT using permutation tests.
    
    Parameters
    ----------
    n_clusters : int
        Number of clusters per arm
    cluster_size : int
        Average number of individuals per cluster
    icc : float
        Intracluster correlation coefficient
    p1 : float
        Proportion in control group
    power : float, optional
        Target power, by default 0.8
    alpha : float, optional
        Significance level, by default 0.05
    cv_cluster_size : float, optional
        Coefficient of variation for cluster sizes, by default 0.0
    effect_measure : str, optional
        Type of effect measure to return ('risk_difference', 'risk_ratio', 'odds_ratio'), by default 'risk_difference'
    precision : float, optional
        Precision for effect size search, by default 0.01
    max_iterations : int, optional
        Maximum search iterations, by default 20
        
    Returns
    -------
    dict
        Dictionary containing minimum detectable effect and study details
    """
    # Get analytical estimate as starting point
    analytical_result = min_detectable_effect_binary(
        n_clusters=n_clusters,
        cluster_size=cluster_size,
        icc=icc,
        p1=p1,
        power=power,
        alpha=alpha,
        cv_cluster_size=cv_cluster_size,
        effect_measure=effect_measure
    )
    
    # Binary search for minimum detectable effect (in risk difference)
    low = 0.01  # Very small effect
    high = min(analytical_result.get("risk_difference", 0.2) * 2, 1.0 - p1)  # Upper bound
    
    iteration = 0
    best_effect = high
    
    # Create progress bar for binary search
    pbar = tqdm(total=max_iterations, desc="Binary search for MDE (permutation)", disable=max_iterations < 5)
    
    while iteration < max_iterations and (high - low) > precision:
        pbar.update(1)
        mid = (low + high) / 2
        p2_test = p1 + mid  # Test this risk difference
        
        if p2_test > 0.99:  # Avoid boundary
            high = mid
            continue
        
        # Test power with this effect size
        result = power_binary_permutation(
            n_clusters=n_clusters,
            cluster_size=cluster_size,
            icc=icc,
            p1=p1,
            p2=p2_test,
            alpha=alpha,
            cv_cluster_size=cv_cluster_size,
            n_simulations=500  # Reduced for search
        )
        
        if result["power"] >= power:
            # Effect is large enough, try smaller
            best_effect = mid
            high = mid
        else:
            # Effect too small, try larger
            low = mid
        
        iteration += 1
    
    pbar.close()
    
    # Get final power estimate with best effect
    p2_final = p1 + best_effect
    final_result = power_binary_permutation(
        n_clusters=n_clusters,
        cluster_size=cluster_size,
        icc=icc,
        p1=p1,
        p2=p2_final,
        alpha=alpha,
        cv_cluster_size=cv_cluster_size,
        n_simulations=1000  # Full precision for final result
    )
    
    # Calculate effect measures
    risk_difference = best_effect
    risk_ratio = p2_final / p1 if p1 > 0 else float('inf')
    odds_ratio = (p2_final / (1 - p2_final)) / (p1 / (1 - p1)) if p1 < 1 and p2_final < 1 else float('inf')
    
    # Determine primary MDE based on effect_measure
    if effect_measure == 'risk_difference':
        mde_primary = risk_difference
    elif effect_measure == 'risk_ratio':
        mde_primary = risk_ratio
    elif effect_measure == 'odds_ratio':
        mde_primary = odds_ratio
    else:
        mde_primary = risk_difference
        effect_measure = 'risk_difference'
    
    # Format results
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
        "design_effect": 1 + (cluster_size - 1) * icc,
        "cv_cluster_size": cv_cluster_size,
        "alpha": alpha,
        "target_power": power,
        "achieved_power": final_result["power"],
        "method": "permutation_test_mde",
        "iterations": iteration
    }
    
    return results