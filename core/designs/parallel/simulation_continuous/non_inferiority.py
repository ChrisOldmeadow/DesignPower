"""
Non-inferiority functions for continuous outcomes in parallel group RCTs.

This module provides simulation-based non-inferiority test functions.
"""

import numpy as np
import math
from scipy import stats
from .core_simulation import simulate_continuous_non_inferiority
from .utils import _calculate_effective_sd, _calculate_welch_satterthwaite_df, _simulate_single_continuous_non_inferiority_trial


def sample_size_continuous_non_inferiority_sim(non_inferiority_margin, std_dev, power=0.8, alpha=0.05, 
                                            allocation_ratio=1.0, nsim=1000, min_n=10, max_n=1000, step=10,
                                            assumed_difference=0.0, direction="lower", repeated_measures=False,
                                            correlation=0.5, method="change_score", seed=None):
    """
    Calculate sample size for non-inferiority test with continuous outcome using simulation.
    
    Parameters
    ----------
    non_inferiority_margin : float
        Non-inferiority margin (must be positive)
    std_dev : float
        Standard deviation of the outcome. If `repeated_measures` is True,
        this is treated as the standard deviation of the raw outcome for both groups
        before calculating the effective standard deviation based on `correlation` and `method`.
    power : float, optional
        Desired power, by default 0.8
    alpha : float, optional
        Significance level (one-sided for non-inferiority), by default 0.05
    allocation_ratio : float, optional
        Ratio of sample sizes (n2/n1), by default 1.0
    nsim : int, optional
        Number of simulations per sample size, by default 1000
    min_n : int, optional
        Minimum sample size to try for group 1, by default 10
    max_n : int, optional
        Maximum sample size to try for group 1, by default 1000
    step : int, optional
        Step size for incrementing sample size, by default 10
    assumed_difference : float, optional
        Assumed true difference between treatments (0 = treatments truly equivalent), by default 0.0
    direction : str, optional
        Direction of non-inferiority test ("lower" or "upper"), by default "lower"
    repeated_measures : bool, optional
        Whether to simulate repeated measures design, by default False
    correlation : float, optional
        Correlation between baseline and follow-up measurements, by default 0.5
    method : str, optional
        Analysis method for repeated measures: "change_score" or "ancova", by default "change_score"
    seed : int, optional
        Random seed for reproducibility, by default None
    
    Returns
    -------
    dict
        Dictionary containing the required sample sizes (`sample_size_1`, `sample_size_2`, `n_total`),
        the `achieved_power` for these sample sizes, and echoes input parameters including
        `repeated_measures`, `correlation`, and `analysis_method` (if `repeated_measures` is True).
    """
    # Validate inputs
    if std_dev <= 0:
        raise ValueError("Standard deviation must be positive")
    
    if non_inferiority_margin <= 0:
        raise ValueError("Non-inferiority margin must be positive")
    
    if allocation_ratio <= 0:
        raise ValueError("Allocation ratio must be positive")
    
    # Initialize search parameters
    n1 = min_n
    achieved_power = 0.0
    iterations = 0
    
    # Perform search to find required sample size
    while n1 <= max_n and achieved_power < power:
        iterations += 1
        # Calculate n2 based on allocation ratio
        n2 = math.ceil(n1 * allocation_ratio)

        # Simulate trials with current sample sizes
        # The 'std_dev' parameter for sample_size_... is the sd_outcome for repeated measures
        sim_result = simulate_continuous_non_inferiority(
            n1=n1,
            n2=n2,
            non_inferiority_margin=non_inferiority_margin,
            std_dev=std_dev,
            alpha=alpha,
            assumed_difference=assumed_difference,
            direction=direction,
            nsim=nsim,
            seed=seed,
            repeated_measures=repeated_measures,
            correlation=correlation,
            method=method
        )
        achieved_power = sim_result["empirical_power"]
        
        # If power is sufficient, break the loop
        if achieved_power >= power:
            break
        
        # Otherwise, increase sample size
        n1 += step
    
    # Calculate total sample size
    n_total = n1 + n2
    
    # Check if we reached the maximum without achieving desired power
    if n1 > max_n:
        # Use the maximum sample size and report the achieved power
        n1 = max_n
        n2 = math.ceil(n1 * allocation_ratio)
        n_total = n1 + n2
        sim_result = simulate_continuous_non_inferiority(
            n1=n1,
            n2=n2,
            non_inferiority_margin=non_inferiority_margin,
            std_dev=std_dev,
            alpha=alpha,
            assumed_difference=assumed_difference,
            direction=direction,
            nsim=nsim,
            seed=seed,
            repeated_measures=repeated_measures,
            correlation=correlation,
            method=method
        )
        achieved_power = sim_result["empirical_power"]
    
    # Return result dictionary
    return {
        "sample_size_1": n1,
        "sample_size_2": n2,
        "total_sample_size": n_total,
        "achieved_power": achieved_power,
        "target_power": power,
        "non_inferiority_margin": non_inferiority_margin,
        "std_dev": std_dev,
        "assumed_difference": assumed_difference,
        "direction": direction,
        "alpha": alpha,
        "simulations": nsim,
        "method": "simulation",
        "iterations": iterations,
        "repeated_measures": repeated_measures,
        "correlation": correlation if repeated_measures else None,
        "analysis_method": method if repeated_measures else None
    }


def power_continuous_non_inferiority_sim(n1, n2, mean1_control, non_inferiority_margin, 
                                         sd1, sd2=None, alpha=0.05, 
                                         assumed_difference=0.0, direction="lower", 
                                         nsim=1000, seed=None,
                                         repeated_measures=False, correlation=0.5,
                                         analysis_method="change_score"):
    """
    Calculate power for non-inferiority test with continuous outcome using simulation.

    Parameters
    ----------
    n1 : int
        Sample size in group 1 (control/standard group).
    n2 : int
        Sample size in group 2 (treatment group).
    mean1_control : float
        Mean of the control/standard group.
    non_inferiority_margin : float
        Non-inferiority margin (must be positive).
    sd1 : float
        Standard deviation of group 1. If `repeated_measures` is True,
        this is treated as the standard deviation of the raw outcome for group 1.
    sd2 : float, optional
        Standard deviation of group 2. If `repeated_measures` is True,
        this is treated as the standard deviation of the raw outcome for group 2.
        If None, assumes equal to sd1.
    alpha : float, optional
        Significance level (one-sided for non-inferiority), by default 0.05.
    assumed_difference : float, optional
        Assumed true difference (mean2 - mean1_control), by default 0.0.
    direction : str, optional
        Direction of non-inferiority test ("lower" or "upper"), by default "lower".
        "lower": H1 is mean2 - mean1 > -margin (treatment is not worse).
        "upper": H1 is mean2 - mean1 <  margin (treatment is not better).
    nsim : int, optional
        Number of simulations, by default 1000.
    seed : int, optional
        Random seed for reproducibility, by default None.
    repeated_measures : bool, optional
        Whether to simulate a repeated measures design, by default False.
        If True, `sd1` and `sd2` are treated as outcome SDs, and `correlation`
        and `analysis_method` are used to calculate effective SDs for the simulation.
    correlation : float, optional
        Correlation between baseline and follow-up for repeated measures, by default 0.5.
        Only used if `repeated_measures` is True.
    analysis_method : str, optional
        Analysis method for repeated measures ('change_score' or 'ancova'), by default "change_score".
        Only used if `repeated_measures` is True.

    Returns
    -------
    dict
        Dictionary containing the estimated power and simulation details.
        If `repeated_measures` is True, also includes `effective_sd1` and `effective_sd2`.
    """
    # Validate inputs
    if not isinstance(n1, int) or n1 <= 0:
        raise ValueError("n1 must be a positive integer.")
    if not isinstance(n2, int) or n2 <= 0:
        raise ValueError("n2 must be a positive integer.")
    if not (isinstance(sd1, (int, float)) and sd1 > 0):
        raise ValueError("sd1 must be positive.")
    if sd2 is not None and not (isinstance(sd2, (int, float)) and sd2 > 0):
        raise ValueError("sd2 must be positive if provided.")
    if not (isinstance(non_inferiority_margin, (int, float)) and non_inferiority_margin > 0):
        raise ValueError("Non-inferiority margin must be positive.")
    if not (isinstance(alpha, float) and 0 < alpha < 1):
        raise ValueError("Alpha must be between 0 and 1.")
    if not isinstance(nsim, int) or nsim <= 0:
        raise ValueError("Number of simulations (nsim) must be a positive integer.")
    if direction not in ["lower", "upper"]:
        raise ValueError("Direction must be 'lower' or 'upper'.")

    # Determine standard deviations for simulation
    sim_sd1 = sd1
    sim_sd2 = sd2 if sd2 is not None else sd1

    if repeated_measures:
        if not (isinstance(correlation, float) and 0 <= correlation <= 1):
            raise ValueError("Correlation must be between 0 and 1 for repeated measures.")
        if analysis_method not in ["change_score", "ancova"]:
            raise ValueError("Invalid analysis_method for repeated measures. Choose 'change_score' or 'ancova'.")
        
        # sd1 and sd2 are treated as sd_outcome1 and sd_outcome2
        sd_outcome1 = sd1
        sd_outcome2 = sd2 if sd2 is not None else sd1

        sim_sd1 = _calculate_effective_sd(sd_outcome1, correlation, analysis_method)
        sim_sd2 = _calculate_effective_sd(sd_outcome2, correlation, analysis_method)
    else:
        # Ensure sd1 and sd2 are positive if not repeated measures (already validated for repeated measures inside _calculate_effective_sd)
        if not (isinstance(sim_sd1, (int, float)) and sim_sd1 > 0):
             raise ValueError("sd1 must be positive.")
        if not (isinstance(sim_sd2, (int, float)) and sim_sd2 > 0):
             raise ValueError("sd2 must be positive.")

    true_mean1 = mean1_control
    true_mean2 = mean1_control + assumed_difference
    rng = np.random.default_rng(seed)
    
    non_inferior_count = 0
    for _ in range(nsim):
        if _simulate_single_continuous_non_inferiority_trial(
            n1, n2, true_mean1, true_mean2, 
            sim_sd1, sim_sd2, non_inferiority_margin, 
            alpha, direction, rng):
            non_inferior_count += 1
            
    empirical_power = non_inferior_count / nsim

    return {
        "power": empirical_power,
        "n1": n1,
        "n2": n2,
        "mean1_control": mean1_control,
        "assumed_difference": assumed_difference,
        "true_mean2": true_mean2,
        "sd1": sd1, # Original sd1 input (outcome sd1 if repeated_measures)
        "sd2": sd2, # Original sd2 input (outcome sd2 if repeated_measures)
        "sim_sd1_effective": sim_sd1 if repeated_measures else sd1, # Effective SD used in simulation for group 1
        "sim_sd2_effective": sim_sd2 if repeated_measures else (sd2 if sd2 is not None else sd1), # Effective SD used in simulation for group 2
        "non_inferiority_margin": non_inferiority_margin,
        "alpha": alpha,
        "direction": direction,
        "simulations": nsim,
        "seed": seed,
        "repeated_measures": repeated_measures,
        "correlation": correlation if repeated_measures else None,
        "analysis_method": analysis_method if repeated_measures else None
    }