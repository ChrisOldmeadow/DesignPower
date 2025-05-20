"""
Simulation methods for survival outcomes in parallel group randomized controlled trials.

This module provides simulation-based functions for power analysis and
sample size calculation for parallel group RCTs with survival (time-to-event) outcomes.
"""

import numpy as np
import math
from scipy import stats
from scipy import optimize
from scipy.stats import expon

# ===== Simulation Core Functions =====

def simulate_survival_trial(n1, n2, median1, median2, enrollment_period=1.0, follow_up_period=1.0, 
                         dropout_rate=0.1, nsim=1000, alpha=0.05, seed=None, sides=2):
    """
    Simulate a parallel group RCT with survival (time-to-event) outcome.
    
    Parameters
    ----------
    n1 : int
        Sample size in group 1 (control)
    n2 : int
        Sample size in group 2 (intervention)
    median1 : float
        Median survival time in group 1 (control)
    median2 : float
        Median survival time in group 2 (intervention)
    enrollment_period : float, optional
        Duration of enrollment period, by default 1.0
    follow_up_period : float, optional
        Duration of follow-up period, by default 1.0
    dropout_rate : float, optional
        Expected dropout rate, by default 0.1
    nsim : int, optional
        Number of simulations, by default 1000
    alpha : float, optional
        Significance level, by default 0.05
    seed : int, optional
        Random seed for reproducibility, by default None
    sides : int, optional
        One-sided or two-sided test (1 or 2), by default 2
    
    Returns
    -------
    dict
        Dictionary containing simulation results, including empirical power
    """
    # Validate inputs
    if median1 <= 0 or median2 <= 0:
        raise ValueError("Median survival times must be positive")
    
    if dropout_rate < 0 or dropout_rate >= 1:
        raise ValueError("Dropout rate must be between 0 and 1")
    
    if enrollment_period <= 0 or follow_up_period <= 0:
        raise ValueError("Enrollment and follow-up periods must be positive")
    
    if n1 <= 0 or n2 <= 0:
        raise ValueError("Sample sizes must be positive")
    
    # Set random seed for reproducibility
    if seed is not None:
        np.random.seed(seed)
    
    # Calculate hazard rates from median survival times
    # For exponential survival: hazard = ln(2) / median
    ln2 = math.log(2)
    hazard1 = ln2 / median1
    hazard2 = ln2 / median2
    
    # Calculate true hazard ratio
    hazard_ratio = hazard2 / hazard1
    
    # Calculate total study duration
    total_duration = enrollment_period + follow_up_period
    
    # Initialize counters
    significant_tests = 0
    all_log_hr = []
    all_p_values = []
    all_events = []
    
    # Run simulations
    for _ in range(nsim):
        # Generate enrollment times (uniform over enrollment period)
        enrollment_times1 = np.random.uniform(0, enrollment_period, n1)
        enrollment_times2 = np.random.uniform(0, enrollment_period, n2)
        
        # Generate survival times (exponential with respective hazards)
        survival_times1 = np.random.exponential(scale=1/hazard1, size=n1)
        survival_times2 = np.random.exponential(scale=1/hazard2, size=n2)
        
        # Generate censoring times due to dropouts
        dropout_times1 = np.random.exponential(scale=1/(dropout_rate/(1-dropout_rate)*hazard1), size=n1)
        dropout_times2 = np.random.exponential(scale=1/(dropout_rate/(1-dropout_rate)*hazard2), size=n2)
        
        # Determine observed times and event indicators for group 1
        observed_times1 = np.minimum(survival_times1, dropout_times1)
        event_indicators1 = (survival_times1 <= dropout_times1).astype(int)
        
        # Determine observed times and event indicators for group 2
        observed_times2 = np.minimum(survival_times2, dropout_times2)
        event_indicators2 = (survival_times2 <= dropout_times2).astype(int)
        
        # Apply administrative censoring at end of study
        # Subjects are followed for (total_duration - enrollment_time)
        follow_up_duration1 = total_duration - enrollment_times1
        follow_up_duration2 = total_duration - enrollment_times2
        
        # If observed_time > follow_up_duration, subject is censored at follow_up_duration
        censored1 = observed_times1 > follow_up_duration1
        censored2 = observed_times2 > follow_up_duration2
        
        observed_times1[censored1] = follow_up_duration1[censored1]
        event_indicators1[censored1] = 0
        
        observed_times2[censored2] = follow_up_duration2[censored2]
        event_indicators2[censored2] = 0
        
        # Count events in each group
        events1 = np.sum(event_indicators1)
        events2 = np.sum(event_indicators2)
        total_events = events1 + events2
        all_events.append(total_events)
        
        # Skip trials with too few events for reliable analysis
        if events1 < 5 or events2 < 5:
            continue
        
        # Calculate log hazard ratio and its standard error
        # Using simple approximation based on event counts
        log_hr = math.log((events2 / np.sum(observed_times2)) / (events1 / np.sum(observed_times1)))
        se_log_hr = math.sqrt(1/events1 + 1/events2)
        
        # Store log hazard ratio
        all_log_hr.append(log_hr)
        
        # Calculate z-statistic and p-value
        z_stat = log_hr / se_log_hr
        
        if sides == 1:
            # One-sided p-value (testing HR > 1, i.e., higher hazard in group 2)
            p_value = 1 - stats.norm.cdf(z_stat)
        else:
            # Two-sided p-value
            p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        
        all_p_values.append(p_value)
        
        # Check if significant
        if p_value <= alpha:
            significant_tests += 1
    
    # Calculate empirical power
    valid_sims = len(all_p_values)
    power = significant_tests / valid_sims if valid_sims > 0 else 0
    
    # Convert lists to numpy arrays for statistics
    all_log_hr = np.array(all_log_hr)
    all_p_values = np.array(all_p_values)
    all_events = np.array(all_events)
    
    # Return results
    return {
        "empirical_power": power,
        "median1": median1,
        "median2": median2,
        "hazard1": hazard1,
        "hazard2": hazard2,
        "hazard_ratio": hazard_ratio,
        "n1": n1,
        "n2": n2,
        "enrollment_period": enrollment_period,
        "follow_up_period": follow_up_period,
        "dropout_rate": dropout_rate,
        "alpha": alpha,
        "sides": sides,
        "simulations": nsim,
        "valid_simulations": valid_sims,
        "mean_events": np.mean(all_events) if len(all_events) > 0 else 0,
        "mean_log_hr": np.mean(all_log_hr) if len(all_log_hr) > 0 else 0,
        "method": "simulation"
    }

def simulate_survival_non_inferiority(n1, n2, median1, non_inferiority_margin, enrollment_period=1.0, 
                                   follow_up_period=1.0, dropout_rate=0.1, nsim=1000, alpha=0.05, 
                                   seed=None, assumed_hazard_ratio=1.0):
    """
    Simulate a parallel group RCT for non-inferiority hypothesis with survival outcome.
    
    Parameters
    ----------
    n1 : int
        Sample size in group 1 (control/standard)
    n2 : int
        Sample size in group 2 (intervention/new)
    median1 : float
        Median survival time in group 1 (control/standard)
    non_inferiority_margin : float
        Non-inferiority margin as a hazard ratio (must be greater than 1)
    enrollment_period : float, optional
        Duration of enrollment period, by default 1.0
    follow_up_period : float, optional
        Duration of follow-up period, by default 1.0
    dropout_rate : float, optional
        Expected dropout rate, by default 0.1
    nsim : int, optional
        Number of simulations, by default 1000
    alpha : float, optional
        Significance level (one-sided for non-inferiority), by default 0.05
    seed : int, optional
        Random seed for reproducibility, by default None
    assumed_hazard_ratio : float, optional
        Assumed true hazard ratio (1.0 = treatments truly equivalent), by default 1.0
    
    Returns
    -------
    dict
        Dictionary containing simulation results, including empirical power
    """
    # Validate inputs
    if median1 <= 0:
        raise ValueError("Median survival time must be positive")
    
    if non_inferiority_margin <= 1:
        raise ValueError("Non-inferiority margin must be greater than 1 (as a hazard ratio)")
    
    if dropout_rate < 0 or dropout_rate >= 1:
        raise ValueError("Dropout rate must be between 0 and 1")
    
    if enrollment_period <= 0 or follow_up_period <= 0:
        raise ValueError("Enrollment and follow-up periods must be positive")
    
    if n1 <= 0 or n2 <= 0:
        raise ValueError("Sample sizes must be positive")
    
    if assumed_hazard_ratio <= 0:
        raise ValueError("Assumed hazard ratio must be positive")
    
    # Set random seed for reproducibility
    if seed is not None:
        np.random.seed(seed)
    
    # Calculate hazard rates from median survival times
    ln2 = math.log(2)
    hazard1 = ln2 / median1
    
    # Calculate hazard rate for group 2 based on assumed hazard ratio
    hazard2 = hazard1 * assumed_hazard_ratio
    median2 = ln2 / hazard2
    
    # Calculate total study duration
    total_duration = enrollment_period + follow_up_period
    
    # Initialize counters
    non_inferior_count = 0
    all_log_hr = []
    all_test_stats = []
    all_events = []
    
    # Run simulations
    for _ in range(nsim):
        # Generate enrollment times (uniform over enrollment period)
        enrollment_times1 = np.random.uniform(0, enrollment_period, n1)
        enrollment_times2 = np.random.uniform(0, enrollment_period, n2)
        
        # Generate survival times (exponential with respective hazards)
        survival_times1 = np.random.exponential(scale=1/hazard1, size=n1)
        survival_times2 = np.random.exponential(scale=1/hazard2, size=n2)
        
        # Generate censoring times due to dropouts
        dropout_times1 = np.random.exponential(scale=1/(dropout_rate/(1-dropout_rate)*hazard1), size=n1)
        dropout_times2 = np.random.exponential(scale=1/(dropout_rate/(1-dropout_rate)*hazard2), size=n2)
        
        # Determine observed times and event indicators for group 1
        observed_times1 = np.minimum(survival_times1, dropout_times1)
        event_indicators1 = (survival_times1 <= dropout_times1).astype(int)
        
        # Determine observed times and event indicators for group 2
        observed_times2 = np.minimum(survival_times2, dropout_times2)
        event_indicators2 = (survival_times2 <= dropout_times2).astype(int)
        
        # Apply administrative censoring at end of study
        follow_up_duration1 = total_duration - enrollment_times1
        follow_up_duration2 = total_duration - enrollment_times2
        
        censored1 = observed_times1 > follow_up_duration1
        censored2 = observed_times2 > follow_up_duration2
        
        observed_times1[censored1] = follow_up_duration1[censored1]
        event_indicators1[censored1] = 0
        
        observed_times2[censored2] = follow_up_duration2[censored2]
        event_indicators2[censored2] = 0
        
        # Count events in each group
        events1 = np.sum(event_indicators1)
        events2 = np.sum(event_indicators2)
        total_events = events1 + events2
        all_events.append(total_events)
        
        # Skip trials with too few events for reliable analysis
        if events1 < 5 or events2 < 5:
            continue
        
        # Calculate hazard rates
        rate1 = events1 / np.sum(observed_times1)
        rate2 = events2 / np.sum(observed_times2)
        
        # Calculate log hazard ratio and its standard error
        log_hr = math.log(rate2 / rate1)
        se_log_hr = math.sqrt(1/events1 + 1/events2)
        
        # Store log hazard ratio
        all_log_hr.append(log_hr)
        
        # Calculate test statistic for non-inferiority
        # H0: HR ≥ margin, H1: HR < margin
        log_non_inferiority_margin = math.log(non_inferiority_margin)
        test_stat = (log_non_inferiority_margin - log_hr) / se_log_hr
        all_test_stats.append(test_stat)
        
        # Calculate one-sided p-value for non-inferiority
        p_value = stats.norm.cdf(test_stat)
        
        # Check if non-inferior
        if p_value <= alpha:
            non_inferior_count += 1
    
    # Calculate empirical power
    valid_sims = len(all_log_hr)
    power = non_inferior_count / valid_sims if valid_sims > 0 else 0
    
    # Convert lists to numpy arrays for statistics
    all_log_hr = np.array(all_log_hr) if len(all_log_hr) > 0 else np.array([])
    all_test_stats = np.array(all_test_stats) if len(all_test_stats) > 0 else np.array([])
    all_events = np.array(all_events) if len(all_events) > 0 else np.array([])
    
    # Return results
    return {
        "empirical_power": power,
        "median1": median1,
        "median2": median2,
        "hazard1": hazard1,
        "hazard2": hazard2,
        "assumed_hazard_ratio": assumed_hazard_ratio,
        "non_inferiority_margin": non_inferiority_margin,
        "n1": n1,
        "n2": n2,
        "enrollment_period": enrollment_period,
        "follow_up_period": follow_up_period,
        "dropout_rate": dropout_rate,
        "alpha": alpha,
        "simulations": nsim,
        "valid_simulations": valid_sims,
        "mean_events": np.mean(all_events) if len(all_events) > 0 else 0,
        "mean_log_hr": np.mean(all_log_hr) if len(all_log_hr) > 0 else 0,
        "method": "simulation"
    }


# ===== Power and Sample Size Functions =====

def power_survival_sim(n1, n2, median1, median2, enrollment_period=1.0, follow_up_period=1.0, 
                     dropout_rate=0.1, nsim=1000, alpha=0.05, seed=None, sides=2):
    """
    Calculate power for survival outcome in parallel design using simulation.
    
    Parameters
    ----------
    n1 : int
        Sample size in group 1 (control)
    n2 : int
        Sample size in group 2 (intervention)
    median1 : float
        Median survival time in group 1 (control)
    median2 : float
        Median survival time in group 2 (intervention)
    enrollment_period : float, optional
        Duration of enrollment period, by default 1.0
    follow_up_period : float, optional
        Duration of follow-up period, by default 1.0
    dropout_rate : float, optional
        Expected dropout rate, by default 0.1
    nsim : int, optional
        Number of simulations, by default 1000
    alpha : float, optional
        Significance level, by default 0.05
    seed : int, optional
        Random seed for reproducibility, by default None
    sides : int, optional
        One-sided or two-sided test (1 or 2), by default 2
    
    Returns
    -------
    dict
        Dictionary containing power estimate and parameters
    """
    # Run simulation
    sim_result = simulate_survival_trial(
        n1=n1,
        n2=n2,
        median1=median1,
        median2=median2,
        enrollment_period=enrollment_period,
        follow_up_period=follow_up_period,
        dropout_rate=dropout_rate,
        nsim=nsim,
        alpha=alpha,
        seed=seed,
        sides=sides
    )
    
    # Return results with consistent key naming
    return {
        "power": sim_result["empirical_power"],
        "median1": median1,
        "median2": median2,
        "hazard1": sim_result["hazard1"],
        "hazard2": sim_result["hazard2"],
        "hazard_ratio": sim_result["hazard_ratio"],
        "n1": n1,
        "n2": n2,
        "enrollment_period": enrollment_period,
        "follow_up_period": follow_up_period,
        "dropout_rate": dropout_rate,
        "alpha": alpha,
        "sides": sides,
        "simulations": nsim,
        "valid_simulations": sim_result["valid_simulations"],
        "mean_events": sim_result["mean_events"],
        "method": "simulation"
    }


def sample_size_survival_sim(median1, median2, enrollment_period=1.0, follow_up_period=1.0, 
                          dropout_rate=0.1, power=0.8, alpha=0.05, allocation_ratio=1.0, 
                          nsim=1000, min_n=10, max_n=1000, step=10, sides=2):
    """
    Calculate sample size for survival outcome in parallel design using simulation.
    
    Parameters
    ----------
    median1 : float
        Median survival time in group 1 (control)
    median2 : float
        Median survival time in group 2 (intervention)
    enrollment_period : float, optional
        Duration of enrollment period, by default 1.0
    follow_up_period : float, optional
        Duration of follow-up period, by default 1.0
    dropout_rate : float, optional
        Expected dropout rate, by default 0.1
    power : float, optional
        Desired statistical power, by default 0.8
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
    sides : int, optional
        One-sided or two-sided test (1 or 2), by default 2
    
    Returns
    -------
    dict
        Dictionary containing the required sample sizes for each group and total
    """
    # Validate inputs
    if median1 <= 0 or median2 <= 0:
        raise ValueError("Median survival times must be positive")
    
    if dropout_rate < 0 or dropout_rate >= 1:
        raise ValueError("Dropout rate must be between 0 and 1")
    
    if enrollment_period <= 0 or follow_up_period <= 0:
        raise ValueError("Enrollment and follow-up periods must be positive")
    
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
        sim_result = simulate_survival_trial(
            n1=n1,
            n2=n2,
            median1=median1,
            median2=median2,
            enrollment_period=enrollment_period,
            follow_up_period=follow_up_period,
            dropout_rate=dropout_rate,
            nsim=nsim,
            alpha=alpha,
            sides=sides
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
    
    # Calculate hazard rates from median survival times
    ln2 = math.log(2)
    hazard1 = ln2 / median1
    hazard2 = ln2 / median2
    hazard_ratio = hazard2 / hazard1
    
    # Calculate expected number of events
    # Simplified calculation based on exponential survival
    total_duration = enrollment_period + follow_up_period
    
    # Calculate expected proportion of events during study
    p_event1 = 1 - math.exp(-hazard1 * total_duration)
    p_event2 = 1 - math.exp(-hazard2 * total_duration)
    
    # Adjust for dropouts
    p_event1 *= (1 - dropout_rate)
    p_event2 *= (1 - dropout_rate)
    
    # Calculate expected events
    events1 = n1 * p_event1
    events2 = n2 * p_event2
    total_events = events1 + events2
    
    # Return result dictionary
    return {
        "sample_size_1": n1,
        "sample_size_2": n2,
        "total_sample_size": n_total,
        "achieved_power": achieved_power,
        "target_power": power,
        "median1": median1,
        "median2": median2,
        "hazard1": hazard1,
        "hazard2": hazard2,
        "hazard_ratio": hazard_ratio,
        "expected_events_1": events1,
        "expected_events_2": events2,
        "total_events": total_events,
        "enrollment_period": enrollment_period,
        "follow_up_period": follow_up_period,
        "dropout_rate": dropout_rate,
        "alpha": alpha,
        "sides": sides,
        "simulations": nsim,
        "method": "simulation",
        "iterations": iterations
    }


def min_detectable_effect_survival_sim(n1, n2, median1, enrollment_period=1.0, follow_up_period=1.0, 
                                    dropout_rate=0.1, power=0.8, nsim=1000, alpha=0.05, 
                                    precision=0.1, sides=2):
    """
    Calculate minimum detectable effect for survival outcome using simulation.
    
    Parameters
    ----------
    n1 : int
        Sample size in group 1 (control)
    n2 : int
        Sample size in group 2 (intervention)
    median1 : float
        Median survival time in group 1 (control)
    enrollment_period : float, optional
        Duration of enrollment period, by default 1.0
    follow_up_period : float, optional
        Duration of follow-up period, by default 1.0
    dropout_rate : float, optional
        Expected dropout rate, by default 0.1
    power : float, optional
        Desired statistical power, by default 0.8
    nsim : int, optional
        Number of simulations, by default 1000
    alpha : float, optional
        Significance level, by default 0.05
    precision : float, optional
        Desired precision for the effect size in terms of median survival, by default 0.1
    sides : int, optional
        One-sided or two-sided test (1 or 2), by default 2
    
    Returns
    -------
    dict
        Dictionary containing the minimum detectable effect and parameters
    """
    # Validate inputs
    if median1 <= 0:
        raise ValueError("Median survival time must be positive")
    
    if dropout_rate < 0 or dropout_rate >= 1:
        raise ValueError("Dropout rate must be between 0 and 1")
    
    if enrollment_period <= 0 or follow_up_period <= 0:
        raise ValueError("Enrollment and follow-up periods must be positive")
    
    if n1 <= 0 or n2 <= 0:
        raise ValueError("Sample sizes must be positive")
    
    # Use binary search to find the minimum detectable effect
    # For survival outcomes, we'll search for minimum detectable median2
    # Lower median2 means worse survival, higher median2 means better survival
    
    # Initialize search boundaries
    low_median2 = median1 * 0.5  # Worst survival (higher hazard)
    high_median2 = median1 * 2.0  # Best survival (lower hazard)
    iterations = 0
    max_iterations = 20  # Prevent infinite loops
    
    # Start binary search for median2
    while (high_median2 - low_median2) > precision and iterations < max_iterations:
        iterations += 1
        
        # Try midpoint median2
        median2 = (low_median2 + high_median2) / 2
        
        # Simulate with current median2
        sim_result = simulate_survival_trial(
            n1=n1,
            n2=n2,
            median1=median1,
            median2=median2,
            enrollment_period=enrollment_period,
            follow_up_period=follow_up_period,
            dropout_rate=dropout_rate,
            nsim=nsim,
            alpha=alpha,
            sides=sides
        )
        
        # Extract achieved power
        achieved_power = sim_result["empirical_power"]
        
        # Adjust search space based on power
        if achieved_power < power:
            # Power too low, need larger effect (more different from median1)
            if median2 > median1:  # Higher median2 = better survival
                low_median2 = median2  # Need even higher median2
            else:  # Lower median2 = worse survival
                high_median2 = median2  # Need even lower median2
        else:
            # Power sufficient, try smaller effect (closer to median1)
            if median2 > median1:  # Higher median2 = better survival
                high_median2 = median2  # Try lower median2 (closer to median1)
            else:  # Lower median2 = worse survival
                low_median2 = median2  # Try higher median2 (closer to median1)
    
    # Final median2 and verification simulation
    median2 = (low_median2 + high_median2) / 2
    final_sim = simulate_survival_trial(
        n1=n1,
        n2=n2,
        median1=median1,
        median2=median2,
        enrollment_period=enrollment_period,
        follow_up_period=follow_up_period,
        dropout_rate=dropout_rate,
        nsim=nsim,
        alpha=alpha,
        sides=sides
    )
    
    # Calculate hazard rates and hazard ratio
    ln2 = math.log(2)
    hazard1 = ln2 / median1
    hazard2 = ln2 / median2
    hazard_ratio = hazard2 / hazard1
    
    # Return results
    return {
        "minimum_detectable_median": median2,
        "minimum_detectable_hazard_ratio": hazard_ratio,
        "median1": median1,
        "hazard1": hazard1,
        "hazard2": hazard2,
        "n1": n1,
        "n2": n2,
        "enrollment_period": enrollment_period,
        "follow_up_period": follow_up_period,
        "dropout_rate": dropout_rate,
        "achieved_power": final_sim["empirical_power"],
        "target_power": power,
        "alpha": alpha,
        "sides": sides,
        "simulations": nsim,
        "method": "simulation",
        "iterations": iterations
    }


def power_survival_non_inferiority_sim(n1, n2, median1, non_inferiority_margin, enrollment_period=1.0,
                                    follow_up_period=1.0, dropout_rate=0.1, alpha=0.05, 
                                    nsim=1000, assumed_hazard_ratio=1.0):
    """
    Calculate power for non-inferiority test with survival outcome using simulation.
    
    Parameters
    ----------
    n1 : int
        Sample size in group 1 (control/standard)
    n2 : int
        Sample size in group 2 (intervention/new)
    median1 : float
        Median survival time in group 1 (control/standard)
    non_inferiority_margin : float
        Non-inferiority margin as a hazard ratio (must be greater than 1)
    enrollment_period : float, optional
        Duration of enrollment period, by default 1.0
    follow_up_period : float, optional
        Duration of follow-up period, by default 1.0
    dropout_rate : float, optional
        Expected dropout rate, by default 0.1
    alpha : float, optional
        Significance level (one-sided for non-inferiority), by default 0.05
    nsim : int, optional
        Number of simulations, by default 1000
    assumed_hazard_ratio : float, optional
        Assumed true hazard ratio (1.0 = treatments truly equivalent), by default 1.0
    
    Returns
    -------
    dict
        Dictionary containing the power estimate and parameters
    """
    # Run simulation
    sim_result = simulate_survival_non_inferiority(
        n1=n1,
        n2=n2,
        median1=median1,
        non_inferiority_margin=non_inferiority_margin,
        enrollment_period=enrollment_period,
        follow_up_period=follow_up_period,
        dropout_rate=dropout_rate,
        nsim=nsim,
        alpha=alpha,
        assumed_hazard_ratio=assumed_hazard_ratio
    )
    
    # Return results with consistent key naming
    return {
        "power": sim_result["empirical_power"],
        "median1": median1,
        "median2": sim_result["median2"],
        "hazard1": sim_result["hazard1"],
        "hazard2": sim_result["hazard2"],
        "assumed_hazard_ratio": assumed_hazard_ratio,
        "non_inferiority_margin": non_inferiority_margin,
        "n1": n1,
        "n2": n2,
        "enrollment_period": enrollment_period,
        "follow_up_period": follow_up_period,
        "dropout_rate": dropout_rate,
        "alpha": alpha,
        "simulations": nsim,
        "valid_simulations": sim_result["valid_simulations"],
        "mean_events": sim_result["mean_events"],
        "method": "simulation"
    }


def sample_size_survival_non_inferiority_sim(median1, non_inferiority_margin, enrollment_period=1.0, 
                                          follow_up_period=1.0, dropout_rate=0.1, power=0.8, alpha=0.05, 
                                          allocation_ratio=1.0, nsim=1000, min_n=10, max_n=1000, step=10,
                                          assumed_hazard_ratio=1.0):
    """
    Calculate sample size for non-inferiority test with survival outcome using simulation.
    
    Parameters
    ----------
    median1 : float
        Median survival time in group 1 (control/standard)
    non_inferiority_margin : float
        Non-inferiority margin as a hazard ratio (must be greater than 1)
    enrollment_period : float, optional
        Duration of enrollment period, by default 1.0
    follow_up_period : float, optional
        Duration of follow-up period, by default 1.0
    dropout_rate : float, optional
        Expected dropout rate, by default 0.1
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
    assumed_hazard_ratio : float, optional
        Assumed true hazard ratio (1.0 = treatments truly equivalent), by default 1.0
    
    Returns
    -------
    dict
        Dictionary containing the required sample sizes
    """
    # Validate inputs
    if median1 <= 0:
        raise ValueError("Median survival time must be positive")
    
    if non_inferiority_margin <= 1:
        raise ValueError("Non-inferiority margin must be greater than 1 (as a hazard ratio)")
    
    if dropout_rate < 0 or dropout_rate >= 1:
        raise ValueError("Dropout rate must be between 0 and 1")
    
    if enrollment_period <= 0 or follow_up_period <= 0:
        raise ValueError("Enrollment and follow-up periods must be positive")
    
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
        sim_result = simulate_survival_non_inferiority(
            n1=n1,
            n2=n2,
            median1=median1,
            non_inferiority_margin=non_inferiority_margin,
            enrollment_period=enrollment_period,
            follow_up_period=follow_up_period,
            dropout_rate=dropout_rate,
            nsim=nsim,
            alpha=alpha,
            assumed_hazard_ratio=assumed_hazard_ratio
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
    
    # Calculate hazard rates and median survival for group 2
    ln2 = math.log(2)
    hazard1 = ln2 / median1
    hazard2 = hazard1 * assumed_hazard_ratio
    median2 = ln2 / hazard2
    
    # Calculate expected number of events
    # Simplified calculation based on exponential survival
    total_duration = enrollment_period + follow_up_period
    
    # Calculate expected proportion of events during study
    p_event1 = 1 - math.exp(-hazard1 * total_duration)
    p_event2 = 1 - math.exp(-hazard2 * total_duration)
    
    # Adjust for dropouts
    p_event1 *= (1 - dropout_rate)
    p_event2 *= (1 - dropout_rate)
    
    # Calculate expected events
    events1 = n1 * p_event1
    events2 = n2 * p_event2
    total_events = events1 + events2
    
    # Return result dictionary
    return {
        "sample_size_1": n1,
        "sample_size_2": n2,
        "total_sample_size": n_total,
        "achieved_power": achieved_power,
        "target_power": power,
        "median1": median1,
        "median2": median2,
        "hazard1": hazard1,
        "hazard2": hazard2,
        "assumed_hazard_ratio": assumed_hazard_ratio,
        "non_inferiority_margin": non_inferiority_margin,
        "expected_events_1": events1,
        "expected_events_2": events2,
        "total_events": total_events,
        "enrollment_period": enrollment_period,
        "follow_up_period": follow_up_period,
        "dropout_rate": dropout_rate,
        "alpha": alpha,
        "simulations": nsim,
        "method": "simulation",
        "iterations": iterations
    }
