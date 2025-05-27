"""
Simulation methods for continuous outcomes in parallel group randomized controlled trials.

This module provides simulation-based functions for power analysis and
sample size calculation for parallel group RCTs with continuous outcomes.
"""

import numpy as np
import math
from scipy import stats
from scipy import optimize

# ===== Simulation Core Functions =====

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

def simulate_continuous_non_inferiority(
    n1, n2, mean1_control, non_inferiority_margin, 
    sd1, sd2=None, alpha=0.05, 
    assumed_difference=0.0, direction="lower", 
    nsim=1000, seed=None,
    repeated_measures=False, correlation=0.5,
    analysis_method="change_score"
):
    """
    Simulate multiple non-inferiority trials for continuous outcomes.

    This function repeatedly calls _simulate_single_continuous_non_inferiority_trial
    to generate a list of outcomes (whether non-inferiority was established)
    for each simulated trial.

    Parameters
    ----------
    n1 : int
        Sample size in group 1 (control/standard group).
    n2 : int
        Sample size in group 2 (treatment group).
    mean1_control : float
        Mean of the control/standard group.
    non_inferiority_margin : float
        Non-inferiority margin.
    sd1 : float
        Standard deviation of group 1. If `repeated_measures` is True,
        this is treated as the standard deviation of the raw outcome for group 1.
    sd2 : float, optional
        Standard deviation of group 2. If `repeated_measures` is True,
        this is treated as the standard deviation of the raw outcome for group 2.
        If None, assumes equal to sd1.
    alpha : float, optional
        Significance level (one-sided), by default 0.05.
    assumed_difference : float, optional
        Assumed true difference (mean2 - mean1_control), by default 0.0.
        This defines the mean of group 2 as mean1_control + assumed_difference.
    direction : str, optional
        Direction of non-inferiority: "lower" or "upper", by default "lower".
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
        Dictionary containing:
        - 'rejections': A list of booleans (True if non-inferiority established).
        - 'power': The proportion of trials where non-inferiority was established.
        - Plus all input parameters, including effective SDs if repeated_measures is True.
    """
    # Validate inputs
    if not isinstance(n1, int) or n1 <= 0:
        raise ValueError("n1 must be a positive integer.")
    if not isinstance(n2, int) or n2 <= 0:
        raise ValueError("n2 must be a positive integer.")
    # sd1 and sd2 positivity is checked later, after potential effective_sd calculation
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
             raise ValueError("sd2 must be positive if provided, or sd1 must be positive if sd2 is None.")

    true_mean1 = mean1_control
    true_mean2 = mean1_control + assumed_difference

    rng = np.random.default_rng(seed)
    rejections = []
    for _ in range(nsim):
        rejections.append(
            _simulate_single_continuous_non_inferiority_trial(
                n1, n2, true_mean1, true_mean2, 
                sim_sd1, sim_sd2, non_inferiority_margin, 
                alpha, direction, rng
            )
        )
            
    power = sum(rejections) / nsim

    return {
        'rejections': rejections,
        'power': power,
        'n1': n1,
        'n2': n2,
        'mean1_control': mean1_control,
        'assumed_difference': assumed_difference,
        'true_mean2': true_mean2,
        'sd1': sd1, # Original sd1 input (outcome sd1 if repeated_measures)
        'sd2': sd2, # Original sd2 input (outcome sd2 if repeated_measures)
        'sim_sd1_effective': sim_sd1 if repeated_measures else sd1,
        'sim_sd2_effective': sim_sd2 if repeated_measures else (sd2 if sd2 is not None else sd1),
        'non_inferiority_margin': non_inferiority_margin,
        'alpha': alpha,
        'direction': direction,
        'nsim': nsim,
        'seed': seed,
        'repeated_measures': repeated_measures,
        'correlation': correlation if repeated_measures else None,
        'analysis_method': analysis_method if repeated_measures else None
    }


# ===== Power and Sample Size Functions =====

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


def sample_size_continuous_sim(delta, std_dev, power=0.8, alpha=0.05, allocation_ratio=1.0, nsim=1000, 
                            min_n=10, max_n=1000, step=10, repeated_measures=False, correlation=0.5, 
                            method="change_score"):
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
    
    Returns
    -------
    dict
        Dictionary containing the required sample sizes for each group and total
    """
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
                alpha=alpha
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


def sample_size_continuous_non_inferiority_sim(non_inferiority_margin, std_dev, power=0.8, alpha=0.05, 
                                            allocation_ratio=1.0, nsim=1000, min_n=10, max_n=1000, step=10,
                                            assumed_difference=0.0, direction="lower", repeated_measures=False,
                                            correlation=0.5, method="change_score"):
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
            mean1_control=0, # Reference mean for control, effect captured by assumed_difference
            non_inferiority_margin=non_inferiority_margin,
            sd1=std_dev,     # This is sd_outcome1 for repeated measures
            sd2=std_dev,     # This is sd_outcome2 for repeated measures
            alpha=alpha,
            assumed_difference=assumed_difference,
            direction=direction,
            nsim=nsim,
            seed=None, # The sample size function doesn't currently manage a seed to pass here
            repeated_measures=repeated_measures,
            correlation=correlation,
            analysis_method=method
        )
        achieved_power = sim_result["power"]
        
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
            mean1_control=0, # Reference mean for control, effect captured by assumed_difference
            non_inferiority_margin=non_inferiority_margin,
            sd1=std_dev,     # This is sd_outcome1 for repeated measures
            sd2=std_dev,     # This is sd_outcome2 for repeated measures
            alpha=alpha,
            assumed_difference=assumed_difference,
            direction=direction,
            nsim=nsim,
            seed=None, # The sample size function doesn't currently manage a seed to pass here
            repeated_measures=repeated_measures,
            correlation=correlation,
            analysis_method=method
        )
        achieved_power = sim_result["power"]
    
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


# ===== Helper for Repeated Measures Standard Deviation =====

def _calculate_effective_sd(sd_outcome, correlation, analysis_method):
    """
    Calculate the effective standard deviation for repeated measures designs.

    Parameters
    ----------
    sd_outcome : float
        Standard deviation of the raw outcome measure.
    correlation : float
        Correlation between baseline and follow-up measurements.
    analysis_method : str
        The analysis method ('change_score' or 'ancova').

    Returns
    -------
    float
        The effective standard deviation.

    Raises
    ------
    ValueError
        If an invalid analysis_method is provided.
    """
    if not (0 <= correlation <= 1):
        raise ValueError("Correlation must be between 0 and 1.")
    if sd_outcome <= 0:
        raise ValueError("sd_outcome must be positive.")

    if analysis_method == "change_score":
        # For change score, SD_change = SD_outcome * sqrt(2 * (1 - correlation))
        # Assumes SD is same at baseline and follow-up for this calculation.
        effective_sd = sd_outcome * math.sqrt(2 * (1 - correlation))
    elif analysis_method == "ancova":
        # For ANCOVA, SD_adjusted = SD_outcome * sqrt(1 - correlation^2)
        effective_sd = sd_outcome * math.sqrt(1 - correlation**2)
    else:
        raise ValueError("Invalid analysis_method. Choose 'change_score' or 'ancova'.")
    
    # Effective SD cannot be zero unless sd_outcome is zero (handled) or correlation is 1 (for change_score)
    # or correlation is 1 (for ancova, if sd_outcome > 0 then effective_sd > 0 unless correlation is exactly 1)
    # A very small positive value is better than zero if correlation is exactly 1 for ANCOVA to avoid division by zero issues later.
    # However, the formulas naturally handle this. If effective_sd is 0, it implies perfect correlation and no variability in change/adjusted scores.
    return max(effective_sd, 1e-9) # Ensure effective_sd is not exactly zero to avoid potential division by zero if used as a divisor.


# ===== Helper for Welch-Satterthwaite Degrees of Freedom =====
def _calculate_welch_satterthwaite_df(v1, n1, v2, n2):
    """Calculate Welch-Satterthwaite degrees of freedom."""
    if n1 < 2 or n2 < 2: # Variances are undefined or df calculation breaks
        return max(1, n1 + n2 - 2) # Fallback, though NI typically has n >= 2
    
    numerator = (v1 / n1 + v2 / n2)**2
    denominator = ((v1 / n1)**2 / (n1 - 1)) + ((v2 / n2)**2 / (n2 - 1))
    if denominator == 0:
        return n1 + n2 - 2 # Avoid division by zero, fallback to pooled df
    return numerator / denominator

# ===== Simulation Core for Non-Inferiority Power =====
def _simulate_single_continuous_non_inferiority_trial(n1, n2, true_mean1, true_mean2, 
                                                      sd1, sd2, non_inferiority_margin, 
                                                      alpha_one_sided, direction, rng):
    """Simulates a single non-inferiority trial for continuous outcomes."""
    sample1 = rng.normal(loc=true_mean1, scale=sd1, size=n1)
    sample2 = rng.normal(loc=true_mean2, scale=sd2, size=n2)

    if n1 < 2 or n2 < 2: # Need at least 2 observations for variance
        return False

    obs_mean1 = np.mean(sample1)
    obs_var1 = np.var(sample1, ddof=1)
    obs_mean2 = np.mean(sample2)
    obs_var2 = np.var(sample2, ddof=1)

    obs_diff = obs_mean2 - obs_mean1
    
    # Handle cases where variance might be zero (e.g., all sample values are identical)
    if obs_var1 < 1e-9: obs_var1 = 0 # Treat extremely small variance as zero
    if obs_var2 < 1e-9: obs_var2 = 0
    
    # If both variances are zero
    if obs_var1 == 0 and obs_var2 == 0:
        # If means are identical, se_diff is 0. If different, se_diff is 0 but diff is not.
        # This scenario implies no variability, outcome is deterministic.
        if direction == "lower":
            return obs_diff > -non_inferiority_margin
        else: # upper
            return obs_diff < non_inferiority_margin

    se_diff = np.sqrt(obs_var1 / n1 + obs_var2 / n2)
    if se_diff == 0: # Should be caught by var checks, but as a safeguard
         if direction == "lower":
            return obs_diff > -non_inferiority_margin
         else: # upper
            return obs_diff < non_inferiority_margin

    df = _calculate_welch_satterthwaite_df(obs_var1, n1, obs_var2, n2)

    p_value = -1 # Initialize
    if direction == "lower": # H1: mean2 - mean1 > -margin
        # We want to show treatment is not worse than control by more than margin
        # Test statistic for (obs_diff - (-non_inferiority_margin)) / se_diff
        t_stat = (obs_diff - (-non_inferiority_margin)) / se_diff
        p_value = stats.t.sf(t_stat, df) # Survival function (1 - cdf)
    elif direction == "upper": # H1: mean2 - mean1 < margin
        # We want to show treatment is not better than control by more than margin
        # Test statistic for (obs_diff - non_inferiority_margin) / se_diff
        t_stat = (obs_diff - non_inferiority_margin) / se_diff
        p_value = stats.t.cdf(t_stat, df)
    else:
        raise ValueError(f"Invalid direction: {direction}. Must be 'lower' or 'upper'.")

    return p_value < alpha_one_sided
