"""
Core simulation functions for continuous outcomes in parallel group RCTs.

This module provides the fundamental simulation functions that are used
by the higher-level power and sample size calculation functions.
"""

import numpy as np
from scipy import stats


def simulate_continuous_trial(n1, n2, mean1, mean2, sd1, sd2=None, nsim=1000, alpha=0.05, test="t-test", seed=None):
    """
    Simulate a parallel group RCT with continuous outcome.
    
    Parameters
    ----------
    n1 : int
        Sample size in group 1
    n2 : int
        Sample size in group 2
    mean1 : float
        Mean in group 1
    mean2 : float
        Mean in group 2
    sd1 : float
        Standard deviation in group 1
    sd2 : float, optional
        Standard deviation in group 2, by default None (uses sd1)
    nsim : int, optional
        Number of simulations, by default 1000
    alpha : float, optional
        Significance level, by default 0.05
    test : str, optional
        Type of test to use, by default "t-test"
    seed : int, optional
        Random seed for reproducibility, by default None
    
    Returns
    -------
    dict
        Dictionary containing simulation results, including empirical power
    """
    # Validate inputs
    if sd1 <= 0:
        raise ValueError("Standard deviation must be positive")
    
    # If sd2 not provided, assume equal to sd1
    if sd2 is None:
        sd2 = sd1
    
    # Set random seed for reproducibility
    if seed is not None:
        np.random.seed(seed)
    
    # Initialize counters
    significant_tests = 0
    all_p_values = []
    all_test_stats = []
    
    # Run simulations
    for _ in range(nsim):
        # Generate continuous outcomes
        group1 = np.random.normal(mean1, sd1, n1)
        group2 = np.random.normal(mean2, sd2, n2)
        
        # Perform statistical test
        if test.lower() == "t-test":
            test_stat, p_value = stats.ttest_ind(group1, group2, equal_var=(sd1 == sd2))
            p_value = p_value / 2 if mean1 != mean2 else p_value  # One-sided if directional
        else:
            # Default to t-test if unrecognized test
            test_stat, p_value = stats.ttest_ind(group1, group2, equal_var=(sd1 == sd2))
            p_value = p_value / 2 if mean1 != mean2 else p_value  # One-sided if directional
        
        all_p_values.append(p_value)
        all_test_stats.append(test_stat)
        
        # Check if significant
        if p_value <= alpha:
            significant_tests += 1
    
    # Calculate empirical power
    power = significant_tests / nsim
    
    # Convert lists to numpy arrays for statistics
    all_p_values = np.array(all_p_values)
    all_test_stats = np.array(all_test_stats)
    
    # Return results
    return {
        "empirical_power": power,
        "mean1": mean1,
        "mean2": mean2,
        "sd1": sd1,
        "sd2": sd2,
        "n1": n1,
        "n2": n2,
        "alpha": alpha,
        "test": test,
        "simulations": nsim,
        "mean_p_value": np.mean(all_p_values),
        "median_p_value": np.median(all_p_values),
        "method": "simulation"
    }


def simulate_continuous_non_inferiority(n1, n2, non_inferiority_margin, std_dev, nsim=1000, alpha=0.05, 
                                     seed=None, assumed_difference=0.0, direction="lower",
                                     repeated_measures=False, correlation=0.5, method="change_score"):
    """
    Simulate a parallel group RCT for non-inferiority hypothesis with continuous outcome.
    
    Parameters
    ----------
    n1 : int
        Sample size in group 1 (standard treatment)
    n2 : int
        Sample size in group 2 (new treatment)
    non_inferiority_margin : float
        Non-inferiority margin (must be positive)
    std_dev : float
        Standard deviation of the outcome (assumed equal in both groups)
    nsim : int, optional
        Number of simulations, by default 1000
    alpha : float, optional
        Significance level (one-sided for non-inferiority), by default 0.05
    seed : int, optional
        Random seed for reproducibility, by default None
    assumed_difference : float, optional
        Assumed true difference between treatments (0 = treatments truly equivalent), by default 0.0
    direction : str, optional
        Direction of non-inferiority test ("lower" or "upper"), by default "lower"
    repeated_measures : bool, optional
        Whether to simulate repeated measures design with baseline and follow-up, by default False
    correlation : float, optional
        Correlation between baseline and follow-up measurements, by default 0.5
    method : str, optional
        Analysis method for repeated measures: "change_score" or "ancova", by default "change_score"
    
    Returns
    -------
    dict
        Dictionary containing the estimated power and simulation details
    """
    # Validate inputs
    if std_dev <= 0:
        raise ValueError("Standard deviation must be positive")
    
    if non_inferiority_margin <= 0:
        raise ValueError("Non-inferiority margin must be positive")
    
    # Set random seed for reproducibility
    if seed is not None:
        np.random.seed(seed)
    
    # Initialize counters
    non_inferior_count = 0
    all_test_stats = []
    all_differences = []
    
    # Define critical values for test
    z_alpha = stats.norm.ppf(1 - alpha)
    
    # Run simulations
    for _ in range(nsim):
        # Generate data based on whether it's repeated measures or not
        if not repeated_measures:
            # Simple parallel groups
            group1 = np.random.normal(0, std_dev, n1)  # Standard treatment
            group2 = np.random.normal(assumed_difference, std_dev, n2)  # New treatment with assumed difference
            
            # Calculate mean difference directly
            mean1 = np.mean(group1)
            mean2 = np.mean(group2)
            diff = mean2 - mean1
            
            # Calculate standard error
            se = np.sqrt(std_dev**2 / n1 + std_dev**2 / n2)
        else:
            # Repeated measures design with baseline and follow-up
            # Simulate correlated baseline and follow-up measurements
            rho = correlation
            variance = std_dev**2
            
            # Create covariance matrix for bivariate normal distribution
            cov_matrix = np.array([[variance, rho * variance], [rho * variance, variance]])
            
            # Generate baseline and follow-up data for both groups
            baseline1, followup1 = np.random.multivariate_normal([0, 0], cov_matrix, n1).T
            baseline2, followup2 = np.random.multivariate_normal([0, assumed_difference], cov_matrix, n2).T
            
            # Analyze based on specified method
            if method == "change_score":
                # Change score analysis (follow-up minus baseline)
                group1 = followup1 - baseline1
                group2 = followup2 - baseline2
                
                # Calculate means and difference
                mean1 = np.mean(group1)
                mean2 = np.mean(group2)
                diff = mean2 - mean1
                
                # Standard error for change scores
                change_var = 2 * variance * (1 - rho)
                se = np.sqrt(change_var / n1 + change_var / n2)
            else:  # ANCOVA
                # For ANCOVA, we need to adjust for baseline
                # Simulate as if we performed ANCOVA (simplified)
                group1 = followup1 - rho * baseline1
                group2 = followup2 - rho * baseline2
                
                # Calculate means and difference
                mean1 = np.mean(group1)
                mean2 = np.mean(group2)
                diff = mean2 - mean1
                
                # Standard error for ANCOVA
                ancova_var = variance * (1 - rho**2)
                se = np.sqrt(ancova_var / n1 + ancova_var / n2)
        
        all_differences.append(diff)
        
        # Calculate test statistic based on direction
        if direction == "lower":
            # For lower non-inferiority, we're testing mean2 >= mean1 - margin
            # Null: mean2 <= mean1 - margin, Alt: mean2 > mean1 - margin
            test_stat = (diff + non_inferiority_margin) / se
            is_non_inferior = test_stat >= z_alpha
        else:  # upper
            # For upper non-inferiority, we're testing mean2 <= mean1 + margin
            # Null: mean2 >= mean1 + margin, Alt: mean2 < mean1 + margin
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
        "non_inferiority_margin": non_inferiority_margin,
        "std_dev": std_dev,
        "n1": n1,
        "n2": n2,
        "assumed_difference": assumed_difference,
        "direction": direction,
        "alpha": alpha,
        "simulations": nsim,
        "mean_difference": np.mean(all_differences),
        "method": "simulation",
        "repeated_measures": repeated_measures,
        "correlation": correlation if repeated_measures else None,
        "analysis_method": method if repeated_measures else None
    }