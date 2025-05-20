"""
Single-arm survival analysis functions.

This module contains functions for sample size calculation, power analysis,
and minimum detectable effect estimation for single-arm survival studies.
"""
import math
import numpy as np
from scipy import stats

def one_sample_survival_test_sample_size(
    median_null, 
    median_alt, 
    enrollment_period, 
    follow_up_period, 
    dropout_rate=0.1,
    alpha=0.05, 
    power=0.8, 
    sides=2
):
    """
    Calculate sample size for one-sample survival analysis.
    
    Parameters
    ----------
    median_null : float
        Median survival time under null hypothesis (in time units)
    median_alt : float
        Median survival time under alternative hypothesis (in time units)
    enrollment_period : float
        Length of enrollment period (in time units)
    follow_up_period : float
        Minimum follow-up period after end of enrollment (in time units)
    dropout_rate : float, optional
        Expected dropout rate (proportion), by default 0.1
    alpha : float, optional
        Significance level, by default 0.05
    power : float, optional
        Desired statistical power, by default 0.8
    sides : int, optional
        One or two-sided test (1 or 2), by default 2
        
    Returns
    -------
    dict
        Dictionary containing the required sample size and expected events
    """
    # Convert medians to hazard rates (assuming exponential distribution)
    hazard_null = math.log(2) / median_null
    hazard_alt = math.log(2) / median_alt
    
    # Calculate hazard ratio
    hazard_ratio = hazard_alt / hazard_null
    
    # Calculate critical values based on significance level and sides
    if sides == 1:
        z_alpha = stats.norm.ppf(1 - alpha)
    else:
        z_alpha = stats.norm.ppf(1 - alpha/2)
    
    z_beta = stats.norm.ppf(power)
    
    # Calculate required number of events
    events = (z_alpha + z_beta)**2 / (math.log(hazard_ratio))**2
    events = math.ceil(events)
    
    # Calculate average follow-up time
    # For exponential accrual over [0, enrollment_period] and additional follow_up_period
    avg_followup = enrollment_period/2 + follow_up_period
    
    # Calculate event rate accounting for hazard rate and dropout
    effective_hazard = (hazard_null + hazard_alt) / 2  # Average hazard rate
    event_rate = effective_hazard * (1 - dropout_rate)  # Adjust for dropout
    
    # Calculate required sample size to observe the required number of events
    n = events / (event_rate * avg_followup)
    n = math.ceil(n)
    
    return {
        "sample_size": n,
        "events": events,
        "median_null": median_null,
        "median_alt": median_alt,
        "hazard_ratio": hazard_ratio,
        "enrollment_period": enrollment_period,
        "follow_up_period": follow_up_period,
        "dropout_rate": dropout_rate,
        "alpha": alpha,
        "power": power,
        "sides": sides
    }

def one_sample_survival_test_power(
    n,
    median_null, 
    median_alt, 
    enrollment_period, 
    follow_up_period, 
    dropout_rate=0.1,
    alpha=0.05, 
    sides=2
):
    """
    Calculate power for one-sample survival analysis with given sample size.
    
    Parameters
    ----------
    n : int
        Sample size
    median_null : float
        Median survival time under null hypothesis (in time units)
    median_alt : float
        Median survival time under alternative hypothesis (in time units)
    enrollment_period : float
        Length of enrollment period (in time units)
    follow_up_period : float
        Minimum follow-up period after end of enrollment (in time units)
    dropout_rate : float, optional
        Expected dropout rate (proportion), by default 0.1
    alpha : float, optional
        Significance level, by default 0.05
    sides : int, optional
        One or two-sided test (1 or 2), by default 2
        
    Returns
    -------
    dict
        Dictionary containing the calculated power and related parameters
    """
    # Convert medians to hazard rates (assuming exponential distribution)
    hazard_null = math.log(2) / median_null
    hazard_alt = math.log(2) / median_alt
    
    # Calculate hazard ratio
    hazard_ratio = hazard_alt / hazard_null
    
    # Calculate critical values based on significance level and sides
    if sides == 1:
        z_alpha = stats.norm.ppf(1 - alpha)
    else:
        z_alpha = stats.norm.ppf(1 - alpha/2)
    
    # Calculate average follow-up time
    avg_followup = enrollment_period/2 + follow_up_period
    
    # Calculate event rate accounting for hazard rate and dropout
    effective_hazard = (hazard_null + hazard_alt) / 2  # Average hazard rate
    event_rate = effective_hazard * (1 - dropout_rate)  # Adjust for dropout
    
    # Calculate expected number of events
    expected_events = n * event_rate * avg_followup
    
    # Calculate power
    log_hazard_ratio = math.log(hazard_ratio)
    term = z_alpha - log_hazard_ratio * math.sqrt(expected_events) / abs(log_hazard_ratio)
    power = 1 - stats.norm.cdf(term)
    
    return {
        "power": power,
        "sample_size": n,
        "expected_events": expected_events,
        "median_null": median_null,
        "median_alt": median_alt,
        "hazard_ratio": hazard_ratio,
        "enrollment_period": enrollment_period,
        "follow_up_period": follow_up_period,
        "dropout_rate": dropout_rate,
        "alpha": alpha,
        "sides": sides
    }

def min_detectable_effect_one_sample_survival(
    n,
    median_null, 
    enrollment_period, 
    follow_up_period, 
    dropout_rate=0.1,
    alpha=0.05, 
    power=0.8,
    sides=2
):
    """
    Calculate minimum detectable effect for one-sample survival analysis.
    
    Parameters
    ----------
    n : int
        Sample size
    median_null : float
        Median survival time under null hypothesis (in time units)
    enrollment_period : float
        Length of enrollment period (in time units)
    follow_up_period : float
        Minimum follow-up period after end of enrollment (in time units)
    dropout_rate : float, optional
        Expected dropout rate (proportion), by default 0.1
    alpha : float, optional
        Significance level, by default 0.05
    power : float, optional
        Desired statistical power, by default 0.8
    sides : int, optional
        One or two-sided test (1 or 2), by default 2
        
    Returns
    -------
    dict
        Dictionary containing the minimum detectable effect and related parameters
    """
    # Convert median to hazard rate (assuming exponential distribution)
    hazard_null = math.log(2) / median_null
    
    # Calculate critical values based on significance level and sides
    if sides == 1:
        z_alpha = stats.norm.ppf(1 - alpha)
    else:
        z_alpha = stats.norm.ppf(1 - alpha/2)
    
    z_beta = stats.norm.ppf(power)
    
    # Calculate average follow-up time
    avg_followup = enrollment_period/2 + follow_up_period
    
    # Calculate expected number of events under null hypothesis
    event_rate = hazard_null * (1 - dropout_rate)  # Adjust for dropout
    expected_events = n * event_rate * avg_followup
    
    # Calculate minimum detectable hazard ratio
    log_hazard_ratio = (z_alpha + z_beta) / math.sqrt(expected_events)
    hazard_ratio = math.exp(log_hazard_ratio)
    
    # Convert back to median survival time
    hazard_alt = hazard_null * hazard_ratio
    median_alt = math.log(2) / hazard_alt
    
    return {
        "median_alt": median_alt,
        "hazard_ratio": hazard_ratio,
        "sample_size": n,
        "expected_events": expected_events,
        "median_null": median_null,
        "enrollment_period": enrollment_period,
        "follow_up_period": follow_up_period,
        "dropout_rate": dropout_rate,
        "alpha": alpha,
        "power": power,
        "sides": sides
    }
