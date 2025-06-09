"""
Sample size calculation functions for continuous outcomes in parallel group RCTs.

This module provides simulation-based sample size calculation functions.
"""

import numpy as np
import math
from scipy import stats
from .core_simulation import simulate_continuous_trial


def sample_size_continuous_sim(delta, std_dev, power=0.8, alpha=0.05, allocation_ratio=1.0, nsim=1000, 
                            min_n=10, max_n=1000, step=10, repeated_measures=False, correlation=0.5, 
                            method="change_score", seed=None):
    """
    Calculate sample size required for detecting a difference in means using simulation.
    
    Parameters
    ----------
    delta : float
        Minimum detectable effect (difference between means)
    std_dev : float
        Standard deviation of the outcome (assumed equal in both groups)
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
    repeated_measures : bool, optional
        Whether to simulate repeated measures design with baseline and follow-up measurements,
        by default False
    correlation : float, optional
        Correlation between baseline and follow-up measurements, by default 0.5
    method : str, optional
        Analysis method for repeated measures: "change_score" or "ancova", by default "change_score"
    seed : int, optional
        Random seed for reproducibility, by default None
    
    Returns
    -------
    dict
        Dictionary containing the required sample sizes for each group and total
    """
    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)
    
    # Validate inputs
    if std_dev <= 0:
        raise ValueError("Standard deviation must be positive")
    
    if allocation_ratio <= 0:
        raise ValueError("Allocation ratio must be positive")
    
    # Set means for specified effect size delta
    mean1 = 0  # Arbitrary baseline
    mean2 = mean1 + delta
    
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
        if not repeated_measures:
            # Simple parallel design
            sim_result = simulate_continuous_trial(
                n1=n1,
                n2=n2,
                mean1=mean1,
                mean2=mean2,
                sd1=std_dev,
                sd2=std_dev,
                nsim=nsim,
                alpha=alpha,
                seed=seed
            )
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
                sim_result = {"empirical_power": achieved_power}
            else:
                # Invalid simulation parameter
                raise ValueError("Number of simulations must be a positive integer <= 10000")
        
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
        "delta": delta,
        "std_dev": std_dev,
        "alpha": alpha,
        "simulations": nsim,
        "method": "simulation",
        "iterations": iterations,
        "repeated_measures": repeated_measures,
        "correlation": correlation if repeated_measures else None,
        "analysis_method": method if repeated_measures else None
    }