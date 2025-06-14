"""
Simulation methods for binary outcomes in parallel group randomized controlled trials.

This module provides simulation-based functions for power analysis and
sample size calculation for parallel group RCTs with binary outcomes.
"""

import numpy as np
import math
from scipy import stats
from scipy.stats import fisher_exact

# Import analytical test functions for use in simulations
from .analytical_binary import (
    normal_approximation_test,
    likelihood_ratio_test,
    fishers_exact_test,
    perform_binary_test
)

# ===== Simulation Core Functions =====

def simulate_binary_trial(n1, n2, p1, p2, nsim=1000, alpha=0.05, test_type="normal_approximation", seed=None):
    """
    Simulate a parallel group RCT with binary outcome.
    
    Parameters
    ----------
    n1 : int
        Sample size in group 1
    n2 : int
        Sample size in group 2
    p1 : float
        Proportion in group 1 (between 0 and 1)
    p2 : float
        Proportion in group 2 (between 0 and 1)
    nsim : int, optional
        Number of simulations, by default 1000
    alpha : float, optional
        Significance level, by default 0.05
    test_type : str, optional
        Type of test to use, by default "normal_approximation"
    seed : int, optional
        Random seed for reproducibility, by default None
    
    Returns
    -------
    dict
        Dictionary containing simulation results, including empirical power
    """
    # Validate inputs
    if not 0 <= p1 <= 1 or not 0 <= p2 <= 1:
        raise ValueError("Proportions must be between 0 and 1")
    
    # Set random seed for reproducibility
    if seed is not None:
        np.random.seed(seed)
    
    # Initialize counters
    significant_tests = 0
    all_p_values = []
    
    # Run simulations
    for _ in range(nsim):
        # Generate binary outcomes
        group1 = np.random.binomial(1, p1, n1)
        group2 = np.random.binomial(1, p2, n2)
        
        # Count successes
        s1 = np.sum(group1)
        s2 = np.sum(group2)
        
        # Perform statistical test
        p_value = perform_binary_test(n1, n2, s1, s2, test_type)
        all_p_values.append(p_value)
        
        # Check if significant
        if p_value <= alpha:
            significant_tests += 1
    
    # Calculate empirical power
    power = significant_tests / nsim
    
    # Convert p-values to numpy array for statistics
    all_p_values = np.array(all_p_values)
    
    # Return results
    return {
        "empirical_power": power,
        "p1": p1,
        "p2": p2,
        "n1": n1,
        "n2": n2,
        "alpha": alpha,
        "test_type": test_type,
        "simulations": nsim,
        "mean_p_value": np.mean(all_p_values),
        "median_p_value": np.median(all_p_values),
        "method": "simulation"
    }

def simulate_binary_non_inferiority(n1, n2, p1, non_inferiority_margin, nsim=1000, alpha=0.05, 
                                  seed=None, assumed_difference=0.0, direction="lower"):
    """
    Simulate a parallel group RCT for non-inferiority hypothesis with binary outcome.
    
    Parameters
    ----------
    n1 : int
        Sample size in group 1 (standard treatment)
    n2 : int
        Sample size in group 2 (new treatment)
    p1 : float
        Proportion in the control/standard group (between 0 and 1)
    non_inferiority_margin : float
        Non-inferiority margin (must be positive, absolute difference in proportions)
    nsim : int, optional
        Number of simulations, by default 1000
    alpha : float, optional
        Significance level (one-sided for non-inferiority), by default 0.05
    seed : int, optional
        Random seed for reproducibility, by default None
    assumed_difference : float, optional
        Assumed true difference between proportions (0 = proportions truly equivalent), by default 0.0
    direction : str, optional
        Direction of non-inferiority test ("lower" or "upper"), by default "lower"
        
    Returns
    -------
    dict
        Dictionary containing simulation results, including empirical power
    """
    # Validate inputs
    if not 0 <= p1 <= 1:
        raise ValueError("Proportion must be between 0 and 1")
    
    if non_inferiority_margin <= 0:
        raise ValueError("Non-inferiority margin must be positive")
    
    # Calculate p2 from assumed difference
    p2 = p1 + assumed_difference
    
    # Ensure p2 is valid
    if not 0 <= p2 <= 1:
        raise ValueError("The resulting p2 based on assumed difference is not between 0 and 1")
    
    # Set random seed for reproducibility
    if seed is not None:
        np.random.seed(seed)
    
    # Initialize counters
    non_inferior_count = 0
    all_test_stats = []
    all_differences = []
    
    # Define critical value based on direction
    z_alpha = stats.norm.ppf(1 - alpha)
    
    # Run simulations
    for _ in range(nsim):
        # Generate binary outcomes
        group1 = np.random.binomial(1, p1, n1)
        group2 = np.random.binomial(1, p2, n2)
        
        # Calculate proportions
        prop1 = np.mean(group1)
        prop2 = np.mean(group2)
        diff = prop2 - prop1
        all_differences.append(diff)
        
        # Calculate standard error for the difference
        se = math.sqrt(prop1 * (1 - prop1) / n1 + prop2 * (1 - prop2) / n2)
        
        # Handle edge case where standard error is 0
        if se == 0:
            continue
        
        # Calculate test statistic based on direction
        if direction == "lower":
            # For lower non-inferiority, we're testing p2 >= p1 - margin
            # Null: p2 <= p1 - margin, Alt: p2 > p1 - margin
            test_stat = (diff + non_inferiority_margin) / se
            is_non_inferior = test_stat >= z_alpha
        else:  # upper
            # For upper non-inferiority, we're testing p2 <= p1 + margin
            # Null: p2 >= p1 + margin, Alt: p2 < p1 + margin
            test_stat = (non_inferiority_margin - diff) / se
            is_non_inferior = test_stat >= z_alpha
        
        all_test_stats.append(test_stat)
        
        # Check if non-inferior
        if is_non_inferior:
            non_inferior_count += 1
    
    # Calculate empirical power
    power = non_inferior_count / nsim
    
    # Convert lists to numpy arrays for statistics
    all_test_stats = np.array(all_test_stats)
    all_differences = np.array(all_differences)
    
    # Return results
    return {
        "empirical_power": power,
        "p1": p1,
        "p2": p2,
        "n1": n1,
        "n2": n2,
        "non_inferiority_margin": non_inferiority_margin,
        "assumed_difference": assumed_difference,
        "direction": direction,
        "alpha": alpha,
        "simulations": nsim,
        "mean_difference": np.mean(all_differences),
        "method": "simulation"
    }

# ===== Power and Sample Size Functions =====

def power_binary_sim(n1, n2, p1, p2, alpha=0.05, nsim=1000, test_type="Normal Approximation", seed=None):
    """
    Calculate power for binary outcome in parallel design using simulation.
    
    Parameters
    ----------
    n1 : int
        Sample size for group 1
    n2 : int
        Sample size for group 2
    p1 : float
        Proportion in group 1
    p2 : float
        Proportion in group 2
    alpha : float, optional
        Significance level, by default 0.05
    nsim : int, optional
        Number of simulations, by default 1000
    test_type : str, optional
        Type of test to use, by default "Normal Approximation"
    seed : int, optional
        Random seed for reproducibility, by default None
    
    Returns
    -------
    dict
        Dictionary containing power estimate and parameters
    """
    # Convert test_type to format expected by perform_binary_test
    test_type_normalized = test_type.lower().replace(" ", "_")
    
    # Run simulation
    sim_result = simulate_binary_trial(
        n1=n1,
        n2=n2,
        p1=p1,
        p2=p2,
        nsim=nsim,
        alpha=alpha,
        test_type=test_type_normalized,
        seed=seed
    )
    
    # Return results with consistent key naming
    return {
        "power": sim_result["empirical_power"],
        "p1": p1,
        "p2": p2,
        "n1": n1,
        "n2": n2,
        "alpha": alpha,
        "test_type": test_type,
        "simulations": nsim,
        "method": "simulation"
    }

def sample_size_binary_sim(p1, p2, power=0.8, alpha=0.05, allocation_ratio=1.0, nsim=1000, 
                        min_n=10, max_n=1000, step=10, test_type="Normal Approximation", seed=None):
    """
    Calculate sample size required for detecting a difference in proportions using simulation.
    
    Parameters
    ----------
    p1 : float
        Proportion in control group (between 0 and 1)
    p2 : float
        Proportion in intervention group (between 0 and 1)
    power : float, optional
        Desired statistical power (1 - beta), by default 0.8
    alpha : float, optional
        Significance level, by default 0.05
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
    test_type : str, optional
        Type of statistical test to use, by default "Normal Approximation"
    seed : int, optional
        Random seed for reproducibility, by default None
    
    Returns
    -------
    dict
        Dictionary containing the required sample sizes for each group and total
    """
    # Validate inputs
    if not 0 <= p1 <= 1 or not 0 <= p2 <= 1:
        raise ValueError("Proportions must be between 0 and 1")
    
    if allocation_ratio <= 0:
        raise ValueError("Allocation ratio must be positive")
    
    # Convert test_type to format expected by perform_binary_test
    test_type_normalized = test_type.lower().replace(" ", "_")
    
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
        sim_result = simulate_binary_trial(
            n1=n1,
            n2=n2,
            p1=p1,
            p2=p2,
            nsim=nsim,
            alpha=alpha,
            test_type=test_type_normalized,
            seed=seed
        )
        
        # Extract achieved power
        achieved_power = sim_result["empirical_power"]
        
        # If power is sufficient, break the loop
        if achieved_power >= power:
            break
        
        # Otherwise, increase sample size
        n1 += step
    
    # Calculate total sample size
    n_total = n1 + n2
    
    # Return result dictionary
    return {
        "sample_size_1": n1,
        "sample_size_2": n2,
        "total_sample_size": n_total,
        "achieved_power": achieved_power,
        "target_power": power,
        "p1": p1,
        "p2": p2,
        "alpha": alpha,
        "test_type": test_type,
        "simulations": nsim,
        "method": "simulation",
        "iterations": iterations
    }

def min_detectable_effect_binary_sim(n1, n2, p1, power=0.8, nsim=1000, alpha=0.05, 
                                  precision=0.01, test_type="Normal Approximation", seed=None):
    """
    Calculate minimum detectable effect for binary outcomes using simulation-based approach.
    
    Parameters
    ----------
    n1 : int
        Sample size for group 1
    n2 : int
        Sample size for group 2
    p1 : float
        Proportion in control group (between 0 and 1)
    power : float, optional
        Desired statistical power (1 - beta), by default 0.8
    nsim : int, optional
        Number of simulations, by default 1000
    alpha : float, optional
        Significance level, by default 0.05
    precision : float, optional
        Desired precision for the effect size, by default 0.01
    test_type : str, optional
        Type of statistical test to use, by default "Normal Approximation"
    seed : int, optional
        Random seed for reproducibility, by default None
    
    Returns
    -------
    dict
        Dictionary containing the minimum detectable difference in proportions
    """
    # Validate inputs
    if not 0 <= p1 <= 1:
        raise ValueError("Proportion must be between 0 and 1")
    
    # Convert test_type to format expected by perform_binary_test
    test_type_normalized = test_type.lower().replace(" ", "_")
    
    # Initialize binary search for MDE
    low_p2 = max(0, p1 - 0.5)    # Lower bound for p2
    high_p2 = min(1, p1 + 0.5)   # Upper bound for p2
    iterations = 0
    max_iterations = 20  # Prevent infinite loops
    
    # Start binary search
    while (high_p2 - low_p2) > precision and iterations < max_iterations:
        iterations += 1
        # Try midpoint
        p2 = (low_p2 + high_p2) / 2
        
        # Simulate with current p2
        sim_result = simulate_binary_trial(
            n1=n1,
            n2=n2,
            p1=p1,
            p2=p2,
            nsim=nsim,
            alpha=alpha,
            test_type=test_type_normalized,
            seed=seed
        )
        
        # Check if achieved power meets target
        achieved_power = sim_result["empirical_power"]
        
        # Update search space
        if achieved_power < power:
            # Need larger effect size, move further from p1
            if p2 < p1:
                high_p2 = p2  # Move lower bound up if below p1
            else:
                low_p2 = p2   # Move upper bound down if above p1
        else:
            # Power is sufficient, try smaller effect size
            if p2 < p1:
                low_p2 = p2   # Move lower bound down if below p1
            else:
                high_p2 = p2  # Move upper bound up if above p1
    
    # Final choice of p2 and verification simulation
    p2 = (low_p2 + high_p2) / 2
    final_sim = simulate_binary_trial(
        n1=n1,
        n2=n2,
        p1=p1,
        p2=p2,
        nsim=nsim,
        alpha=alpha,
        test_type=test_type_normalized,
        seed=seed
    )
    
    # Calculate minimum detectable effect
    mde = p2 - p1
    
    # Return results
    return {
        "minimum_detectable_effect": mde,
        "absolute_mde": abs(mde),
        "p1": p1,
        "p2": p2,
        "n1": n1,
        "n2": n2,
        "achieved_power": final_sim["empirical_power"],
        "target_power": power,
        "alpha": alpha,
        "test_type": test_type,
        "simulations": nsim,
        "method": "simulation",
        "iterations": iterations
    }

def sample_size_binary_non_inferiority_sim(p1, non_inferiority_margin, power=0.8, alpha=0.05, 
                                        allocation_ratio=1.0, nsim=1000, min_n=10, max_n=1000, 
                                        step=10, assumed_difference=0.0, direction="lower", seed=None):
    """
    Calculate sample size for non-inferiority test with binary outcome using simulation.
    
    Parameters
    ----------
    p1 : float
        Proportion in the control/standard group (between 0 and 1)
    non_inferiority_margin : float
        Non-inferiority margin (must be positive, absolute difference in proportions)
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
        Assumed true difference between proportions (0 = proportions truly equivalent), by default 0.0
    direction : str, optional
        Direction of non-inferiority test ("lower" or "upper"), by default "lower"
    seed : int, optional
        Random seed for reproducibility, by default None
    
    Returns
    -------
    dict
        Dictionary containing the required sample sizes
    """
    # Initialize search parameters
    n1 = min_n
    achieved_power = 0.0
    
    # Calculate the expected p2 based on assumed difference from p1
    p2 = p1 + assumed_difference
    
    # Perform binary search to find required sample size
    while n1 <= max_n and achieved_power < power:
        # Calculate n2 based on allocation ratio
        n2 = math.ceil(n1 * allocation_ratio)
        
        # Simulate trials with current sample sizes
        sim_result = simulate_binary_non_inferiority(
            n1=n1, 
            n2=n2, 
            p1=p1, 
            non_inferiority_margin=non_inferiority_margin, 
            nsim=nsim, 
            alpha=alpha, 
            assumed_difference=assumed_difference,
            direction=direction,
            seed=seed
        )
        
        # Extract achieved power
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
        sim_result = simulate_binary_non_inferiority(
            n1=n1, 
            n2=n2, 
            p1=p1, 
            non_inferiority_margin=non_inferiority_margin, 
            nsim=nsim, 
            alpha=alpha, 
            assumed_difference=assumed_difference,
            direction=direction,
            seed=seed
        )
        achieved_power = sim_result["empirical_power"]
    
    # Return result dictionary
    return {
        "sample_size_1": n1,
        "sample_size_2": n2,
        "total_sample_size": n_total,
        "simulations": nsim,
        "achieved_power": achieved_power,
        "p1": p1,
        "p2": p2,
        "non_inferiority_margin": non_inferiority_margin,
        "assumed_difference": assumed_difference,
        "direction": direction,
        "alpha": alpha,
        "target_power": power,
        "allocation_ratio": allocation_ratio,
        "method": "simulation"
    }

def min_detectable_binary_non_inferiority_margin_sim(n1, n2, p1, power=0.8, alpha=0.05, nsim=1000, 
                                                  precision=0.01, assumed_difference=0.0, direction="lower", seed=None):
    """
    Calculate the minimum detectable non-inferiority margin for binary outcomes using simulation.
    
    Parameters
    ----------
    n1 : int
        Sample size in group 1 (standard treatment)
    n2 : int
        Sample size in group 2 (new treatment)
    p1 : float
        Proportion in the control/standard group (between 0 and 1)
    power : float, optional
        Desired power, by default 0.8
    alpha : float, optional
        Significance level (one-sided for non-inferiority), by default 0.05
    nsim : int, optional
        Number of simulations, by default 1000
    precision : float, optional
        Desired precision for the margin, by default 0.01
    assumed_difference : float, optional
        Assumed true difference between proportions (0 = proportions truly equivalent), by default 0.0
    direction : str, optional
        Direction of non-inferiority test ("lower" or "upper"), by default "lower"
    seed : int, optional
        Random seed for reproducibility, by default None
    
    Returns
    -------
    dict
        Dictionary containing the minimum detectable non-inferiority margin
    """
    # Calculate p2 based on assumed difference
    p2 = p1 + assumed_difference
    
    # Set up search range for margin
    # The maximum possible margin depends on the direction of non-inferiority
    if direction == "lower":
        max_margin = p1  # Maximum possible margin is p1 itself
    else:  # upper
        max_margin = 1.0 - p1  # Maximum possible margin is distance from p1 to 1.0
    
    # Initialize binary search for margin
    low_margin = precision  # Minimum meaningful margin (based on precision)
    high_margin = max_margin
    current_margin = (low_margin + high_margin) / 2
    iterations = 0
    max_iterations = 20  # Prevent infinite loops
    
    while (high_margin - low_margin) > precision and iterations < max_iterations:
        # Simulate trials with current margin
        sim_result = simulate_binary_non_inferiority(
            n1=n1, 
            n2=n2, 
            p1=p1, 
            non_inferiority_margin=current_margin,
            nsim=nsim, 
            alpha=alpha, 
            assumed_difference=assumed_difference,
            direction=direction,
            seed=seed
        )
        
        # Extract achieved power
        achieved_power = sim_result["empirical_power"]
        
        # Adjust the margin based on achieved power
        if achieved_power < power:
            # Power is too low. Current margin is too stringent (too small).
            # Need to try a LARGER margin.
            low_margin = current_margin
        else: # achieved_power >= power
            # Power is sufficient (or too high). Current margin is achievable (or too lenient/large).
            # Try a SMALLER margin to see if it's still achievable.
            high_margin = current_margin
        
        # Set new midpoint for next iteration
        current_margin = (low_margin + high_margin) / 2
        iterations += 1
    
    # Final margin and verification simulation
    final_sim = simulate_binary_non_inferiority(
        n1=n1, 
        n2=n2, 
        p1=p1, 
        non_inferiority_margin=current_margin,
        nsim=nsim, 
        alpha=alpha, 
        assumed_difference=assumed_difference,
        direction=direction,
        seed=seed
    )
    
    # Return results
    return {
        "minimum_detectable_margin": current_margin,
        "p1": p1,
        "p2": p2,
        "n1": n1,
        "n2": n2,
        "achieved_power": final_sim["empirical_power"],
        "target_power": power,
        "alpha": alpha,
        "assumed_difference": assumed_difference,
        "direction": direction,
        "simulations": nsim,
        "method": "simulation",
        "iterations": iterations
    }
