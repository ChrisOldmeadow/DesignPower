"""
Power calculation functions for continuous outcomes in parallel group RCTs.

This module provides simulation-based power calculation functions.
"""

import numpy as np
import math
from scipy import stats
from .core_simulation import simulate_continuous_trial


def power_continuous_sim(n1, n2, mean1, mean2, sd1, sd2=None, alpha=0.05, nsim=1000, test="t-test", seed=None):
    """
    Calculate power for continuous outcome in parallel design using simulation.
    
    Parameters
    ----------
    n1 : int
        Sample size of group 1
    n2 : int
        Sample size of group 2
    mean1 : float
        Mean of group 1
    mean2 : float
        Mean of group 2
    sd1 : float
        Standard deviation of group 1
    sd2 : float, optional
        Standard deviation of group 2. If None, assumes equal to sd1.
    alpha : float, optional
        Significance level, by default 0.05
    nsim : int, optional
        Number of simulations, by default 1000
    test : str, optional
        Type of test to use, by default "t-test"
    seed : int, optional
        Random seed for reproducibility, by default None
    
    Returns
    -------
    dict
        Dictionary containing power estimate and parameters
    """
    # Run simulation
    sim_result = simulate_continuous_trial(
        n1=n1,
        n2=n2,
        mean1=mean1,
        mean2=mean2,
        sd1=sd1,
        sd2=sd2,
        nsim=nsim,
        alpha=alpha,
        test=test,
        seed=seed
    )
    
    # Return results with consistent key naming
    return {
        "power": sim_result["empirical_power"],
        "mean1": mean1,
        "mean2": mean2,
        "sd1": sd1,
        "sd2": sd2 if sd2 is not None else sd1,
        "n1": n1,
        "n2": n2,
        "alpha": alpha,
        "test": test,
        "simulations": nsim,
        "method": "simulation"
    }


def min_detectable_effect_continuous_sim(n1, n2, std_dev, power=0.8, nsim=1000, alpha=0.05, precision=0.01, 
                                      repeated_measures=False, correlation=0.5, method="change_score"):
    """
    Calculate minimum detectable effect size using simulation-based approach and optimization.
    
    Parameters
    ----------
    n1 : int
        Sample size for group 1
    n2 : int
        Sample size for group 2
    std_dev : float
        Standard deviation of the outcome
    power : float, optional
        Desired statistical power (1 - beta), by default 0.8
    nsim : int, optional
        Number of simulations, by default 1000
    alpha : float, optional
        Significance level, by default 0.05
    precision : float, optional
        Desired precision for the effect size, by default 0.01
    repeated_measures : bool, optional
        Whether to simulate repeated measures design with baseline and follow-up measurements,
        by default False
    correlation : float, optional
        Correlation between baseline and follow-up measurements, by default 0.5
    method : str, optional
        Analysis method for repeated measures: "change_score" or "ancova", by default "change_score"
    
    Returns
    -------
    dict
        Dictionary containing the minimum detectable effect and input parameters
    """
    # Validate inputs
    if std_dev <= 0:
        raise ValueError("Standard deviation must be positive")
    
    # Set baseline mean (arbitrary)
    mean1 = 0
    
    # Define the power function that we want to find the root of
    def power_minus_target(delta):
        if not repeated_measures:
            # Simple parallel design
            sim_result = simulate_continuous_trial(
                n1=n1,
                n2=n2,
                mean1=mean1,
                mean2=mean1 + delta,
                sd1=std_dev,
                sd2=std_dev,
                nsim=nsim,
                alpha=alpha
            )
            return sim_result["empirical_power"] - power
        else:
            # Repeated measures design
            # For repeated measures, we need to simulate differently
            variance = std_dev ** 2
            # Create covariance matrix for bivariate normal
            cov_matrix = np.array([[variance, correlation * variance], 
                                [correlation * variance, variance]])
            
            # Initialize counters for simulation
            significant_tests = 0
            
            # Set random seed if provided
            if isinstance(nsim, int) and nsim > 0 and nsim <= 10000:
                np.random.seed(None)  # Reset seed for each simulation set
                
                # Run simulations
                for _ in range(nsim):
                    # Generate baseline and follow-up data for both groups
                    baseline1, followup1 = np.random.multivariate_normal([0, 0], cov_matrix, n1).T
                    baseline2, followup2 = np.random.multivariate_normal([0, delta], cov_matrix, n2).T
                    
                    # Analyze based on specified method
                    if method == "change_score":
                        # Change score analysis
                        group1 = followup1 - baseline1
                        group2 = followup2 - baseline2
                    else:  # ANCOVA
                        # Simplified ANCOVA simulation
                        group1 = followup1 - correlation * baseline1
                        group2 = followup2 - correlation * baseline2
                    
                    # Perform t-test
                    t_stat, p_value = stats.ttest_ind(group1, group2)
                    
                    # Check if significant
                    if p_value <= alpha:
                        significant_tests += 1
                
                # Calculate empirical power
                achieved_power = significant_tests / nsim
                return achieved_power - power
            else:
                # Invalid simulation parameter
                raise ValueError("Number of simulations must be a positive integer <= 10000")
    
    # Initial guess for minimum detectable effect
    # Use an analytical formula as starting point
    z_alpha = stats.norm.ppf(1 - alpha/2)
    z_beta = stats.norm.ppf(power)
    initial_delta = (z_alpha + z_beta) * std_dev * np.sqrt(1/n1 + 1/n2)
    
    # Set up binary search for delta
    low_delta = 0
    high_delta = 2 * initial_delta
    iterations = 0
    max_iterations = 20  # Prevent infinite loops
    delta = initial_delta
    
    # Binary search for the minimum detectable effect
    while (high_delta - low_delta) > precision and iterations < max_iterations:
        iterations += 1
        
        # Try current delta
        current_power_diff = power_minus_target(delta)
        
        # Adjust the search range
        if current_power_diff < 0:  # Power too low, need larger delta
            low_delta = delta
        else:  # Power sufficient, try smaller delta
            high_delta = delta
        
        # Update delta for next iteration
        delta = (low_delta + high_delta) / 2
    
    # Calculate final standardized effect size (Cohen's d)
    cohen_d = delta / std_dev
    
    # Return results
    return {
        "minimum_detectable_effect": delta,
        "standardized_effect": cohen_d,
        "std_dev": std_dev,
        "n1": n1,
        "n2": n2,
        "target_power": power,
        "achieved_power": power + power_minus_target(delta),  # Actual achieved power with final delta
        "alpha": alpha,
        "simulations": nsim,
        "method": "simulation",
        "iterations": iterations,
        "repeated_measures": repeated_measures,
        "correlation": correlation if repeated_measures else None,
        "analysis_method": method if repeated_measures else None
    }