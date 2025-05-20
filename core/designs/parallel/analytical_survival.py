"""
Analytical methods for survival outcomes in parallel group randomized controlled trials.

This module provides analytical (closed-form) functions for power analysis and
sample size calculation for parallel group RCTs with survival (time-to-event) outcomes.
"""

import numpy as np
import math
from scipy import stats

# ===== Main Functions =====

def sample_size_survival(median1, median2, enrollment_period=1.0, follow_up_period=1.0, 
                        dropout_rate=0.1, power=0.8, alpha=0.05, allocation_ratio=1.0, sides=2):
    """
    Calculate sample size for survival outcome in parallel design.
    
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
    sides : int, optional
        One-sided or two-sided test (1 or 2), by default 2
    
    Returns
    -------
    dict
        Dictionary containing sample sizes and parameters
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
    
    if sides not in [1, 2]:
        raise ValueError("Sides must be 1 or 2")
    
    # Calculate hazard rates from median survival times
    # For exponential survival: hazard = ln(2) / median
    ln2 = math.log(2)
    hazard1 = ln2 / median1
    hazard2 = ln2 / median2
    
    # Calculate hazard ratio
    hazard_ratio = hazard2 / hazard1
    
    # Critical value based on one- or two-sided test
    z_alpha = stats.norm.ppf(1 - alpha / sides)
    z_beta = stats.norm.ppf(power)
    
    # Calculate total study duration
    total_duration = enrollment_period + follow_up_period
    
    # Calculate expected proportion of events during study
    p_event1 = 1 - math.exp(-hazard1 * total_duration)
    p_event2 = 1 - math.exp(-hazard2 * total_duration)
    
    # Adjust for dropouts
    p_event1 *= (1 - dropout_rate)
    p_event2 *= (1 - dropout_rate)
    
    # Weighted average of event probabilities
    p_event_avg = (p_event1 + allocation_ratio * p_event2) / (1 + allocation_ratio)
    
    # Calculate required number of events
    # Formula from Schoenfeld's method
    events_required = (z_alpha + z_beta)**2 / (math.log(hazard_ratio))**2 * (1 + allocation_ratio)**2 / allocation_ratio
    
    # Calculate total sample size based on required events and event probability
    n_total = math.ceil(events_required / p_event_avg)
    
    # Calculate group sizes based on allocation ratio
    n1 = math.ceil(n_total / (1 + allocation_ratio))
    n2 = math.ceil(n1 * allocation_ratio)
    
    # Recalculate total based on rounded group sizes
    n_total = n1 + n2
    
    # Calculate expected number of events
    events1 = n1 * p_event1
    events2 = n2 * p_event2
    total_events = events1 + events2
    
    # Return results
    return {
        "sample_size_1": n1,
        "sample_size_2": n2,
        "total_sample_size": n_total,
        "expected_events_1": events1,
        "expected_events_2": events2,
        "total_events": total_events,
        "median1": median1,
        "median2": median2,
        "hazard1": hazard1,
        "hazard2": hazard2,
        "hazard_ratio": hazard_ratio,
        "enrollment_period": enrollment_period,
        "follow_up_period": follow_up_period,
        "dropout_rate": dropout_rate,
        "power": power,
        "alpha": alpha,
        "sides": sides,
        "allocation_ratio": allocation_ratio,
        "method": "analytical"
    }

def power_survival(n1, n2, median1, median2, enrollment_period=1.0, follow_up_period=1.0, 
                 dropout_rate=0.1, alpha=0.05, sides=2):
    """
    Calculate power for survival outcome in parallel design.
    
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
    alpha : float, optional
        Significance level, by default 0.05
    sides : int, optional
        One-sided or two-sided test (1 or 2), by default 2
    
    Returns
    -------
    dict
        Dictionary containing power and parameters
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
    
    if sides not in [1, 2]:
        raise ValueError("Sides must be 1 or 2")
    
    # Calculate hazard rates from median survival times
    ln2 = math.log(2)
    hazard1 = ln2 / median1
    hazard2 = ln2 / median2
    
    # Calculate hazard ratio
    hazard_ratio = hazard2 / hazard1
    
    # Calculate total study duration
    total_duration = enrollment_period + follow_up_period
    
    # Calculate expected proportion of events during study
    p_event1 = 1 - math.exp(-hazard1 * total_duration)
    p_event2 = 1 - math.exp(-hazard2 * total_duration)
    
    # Adjust for dropouts
    p_event1 *= (1 - dropout_rate)
    p_event2 *= (1 - dropout_rate)
    
    # Calculate expected number of events
    events1 = n1 * p_event1
    events2 = n2 * p_event2
    total_events = events1 + events2
    
    # Critical value for alpha
    z_alpha = stats.norm.ppf(1 - alpha / sides)
    
    # Calculate non-centrality parameter (log hazard ratio / SE)
    allocation_ratio = n2 / n1
    se_log_hr = math.sqrt((1 + allocation_ratio)**2 / (allocation_ratio * total_events))
    ncp = abs(math.log(hazard_ratio)) / se_log_hr
    
    # Calculate power
    power = 1 - stats.norm.cdf(z_alpha - ncp)
    
    if sides == 2:
        # For two-sided test, add the probability of the other tail
        power += stats.norm.cdf(-z_alpha - ncp)
    
    # Return results
    return {
        "power": power,
        "n1": n1,
        "n2": n2,
        "total_sample_size": n1 + n2,
        "expected_events_1": events1,
        "expected_events_2": events2,
        "total_events": total_events,
        "median1": median1,
        "median2": median2,
        "hazard1": hazard1,
        "hazard2": hazard2,
        "hazard_ratio": hazard_ratio,
        "enrollment_period": enrollment_period,
        "follow_up_period": follow_up_period,
        "dropout_rate": dropout_rate,
        "alpha": alpha,
        "sides": sides,
        "method": "analytical"
    }

def min_detectable_effect_survival(n1, n2, median1, enrollment_period=1.0, follow_up_period=1.0, 
                                 dropout_rate=0.1, power=0.8, alpha=0.05, sides=2):
    """
    Calculate minimum detectable effect for survival outcome in parallel design.
    
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
    alpha : float, optional
        Significance level, by default 0.05
    sides : int, optional
        One-sided or two-sided test (1 or 2), by default 2
    
    Returns
    -------
    dict
        Dictionary containing minimum detectable effect and parameters
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
    
    if sides not in [1, 2]:
        raise ValueError("Sides must be 1 or 2")
    
    # Calculate hazard rate from median survival time
    ln2 = math.log(2)
    hazard1 = ln2 / median1
    
    # Calculate total study duration
    total_duration = enrollment_period + follow_up_period
    
    # Calculate expected proportion of events during study (for group 1)
    p_event1 = 1 - math.exp(-hazard1 * total_duration)
    
    # Adjust for dropouts
    p_event1 *= (1 - dropout_rate)
    
    # Calculate expected number of events in group 1
    events1 = n1 * p_event1
    
    # Critical values
    z_alpha = stats.norm.ppf(1 - alpha / sides)
    z_beta = stats.norm.ppf(power)
    
    # Calculate expected number of events (assuming similar event rate in group 2)
    # This is an approximation since actual event rate in group 2 depends on the effect size
    # we're trying to calculate
    allocation_ratio = n2 / n1
    expected_events = events1 * (1 + allocation_ratio)  # Starting estimate
    
    # Calculate detectable hazard ratio
    se_log_hr = math.sqrt((1 + allocation_ratio)**2 / (allocation_ratio * expected_events))
    detectable_log_hr = (z_alpha + z_beta) * se_log_hr
    detectable_hazard_ratio = math.exp(detectable_log_hr)
    
    # Calculate median2 from hazard ratio
    hazard2 = hazard1 * detectable_hazard_ratio
    median2 = ln2 / hazard2
    
    # Calculate expected proportion of events in group 2
    p_event2 = 1 - math.exp(-hazard2 * total_duration)
    p_event2 *= (1 - dropout_rate)
    
    # Calculate expected events in group 2
    events2 = n2 * p_event2
    total_events = events1 + events2
    
    # Return results
    return {
        "minimum_detectable_median": median2,
        "minimum_detectable_hazard_ratio": detectable_hazard_ratio,
        "median1": median1,
        "hazard1": hazard1,
        "hazard2": hazard2,
        "n1": n1,
        "n2": n2,
        "total_sample_size": n1 + n2,
        "expected_events_1": events1,
        "expected_events_2": events2,
        "total_events": total_events,
        "enrollment_period": enrollment_period,
        "follow_up_period": follow_up_period,
        "dropout_rate": dropout_rate,
        "power": power,
        "alpha": alpha,
        "sides": sides,
        "method": "analytical"
    }


def sample_size_survival_non_inferiority(median1, non_inferiority_margin, enrollment_period=1.0, 
                                       follow_up_period=1.0, dropout_rate=0.1, power=0.8, alpha=0.05, 
                                       allocation_ratio=1.0, assumed_hazard_ratio=1.0):
    """
    Calculate sample size for non-inferiority test with survival outcome.
    
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
        Desired statistical power, by default 0.8
    alpha : float, optional
        Significance level (one-sided for non-inferiority), by default 0.05
    allocation_ratio : float, optional
        Ratio of sample sizes (n2/n1), by default 1.0
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
    
    if assumed_hazard_ratio <= 0:
        raise ValueError("Assumed hazard ratio must be positive")
    
    # Calculate hazard rate from median survival time
    ln2 = math.log(2)
    hazard1 = ln2 / median1
    
    # Calculate hazard rate for group 2 based on assumed hazard ratio
    hazard2 = hazard1 * assumed_hazard_ratio
    median2 = ln2 / hazard2
    
    # Calculate total study duration
    total_duration = enrollment_period + follow_up_period
    
    # Calculate expected proportion of events during study
    p_event1 = 1 - math.exp(-hazard1 * total_duration)
    p_event2 = 1 - math.exp(-hazard2 * total_duration)
    
    # Adjust for dropouts
    p_event1 *= (1 - dropout_rate)
    p_event2 *= (1 - dropout_rate)
    
    # Weighted average of event probabilities
    p_event_avg = (p_event1 + allocation_ratio * p_event2) / (1 + allocation_ratio)
    
    # Calculate effect size for test (log of hazard ratio difference from margin)
    effect_size = math.log(non_inferiority_margin / assumed_hazard_ratio)
    
    # Critical values (one-sided for non-inferiority)
    z_alpha = stats.norm.ppf(1 - alpha)
    z_beta = stats.norm.ppf(power)
    
    # Calculate required number of events
    # Formula based on Schoenfeld's method
    events_required = (z_alpha + z_beta)**2 / effect_size**2 * (1 + allocation_ratio)**2 / allocation_ratio
    
    # Calculate total sample size based on required events and event probability
    n_total = math.ceil(events_required / p_event_avg)
    
    # Calculate group sizes based on allocation ratio
    n1 = math.ceil(n_total / (1 + allocation_ratio))
    n2 = math.ceil(n1 * allocation_ratio)
    
    # Recalculate total based on rounded group sizes
    n_total = n1 + n2
    
    # Calculate expected number of events
    events1 = n1 * p_event1
    events2 = n2 * p_event2
    total_events = events1 + events2
    
    # Return results
    return {
        "sample_size_1": n1,
        "sample_size_2": n2,
        "total_sample_size": n_total,
        "expected_events_1": events1,
        "expected_events_2": events2,
        "total_events": total_events,
        "median1": median1,
        "median2": median2,
        "hazard1": hazard1,
        "hazard2": hazard2,
        "assumed_hazard_ratio": assumed_hazard_ratio,
        "non_inferiority_margin": non_inferiority_margin,
        "enrollment_period": enrollment_period,
        "follow_up_period": follow_up_period,
        "dropout_rate": dropout_rate,
        "power": power,
        "alpha": alpha,
        "allocation_ratio": allocation_ratio,
        "method": "analytical"
    }

def power_survival_non_inferiority(n1, n2, median1, non_inferiority_margin, enrollment_period=1.0,
                                 follow_up_period=1.0, dropout_rate=0.1, alpha=0.05, 
                                 assumed_hazard_ratio=1.0):
    """
    Calculate power for non-inferiority test with survival outcome.
    
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
    assumed_hazard_ratio : float, optional
        Assumed true hazard ratio (1.0 = treatments truly equivalent), by default 1.0
    
    Returns
    -------
    dict
        Dictionary containing the power estimate and parameters
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
    
    # Calculate hazard rate from median survival time
    ln2 = math.log(2)
    hazard1 = ln2 / median1
    
    # Calculate hazard rate for group 2 based on assumed hazard ratio
    hazard2 = hazard1 * assumed_hazard_ratio
    median2 = ln2 / hazard2
    
    # Calculate total study duration
    total_duration = enrollment_period + follow_up_period
    
    # Calculate expected proportion of events during study
    p_event1 = 1 - math.exp(-hazard1 * total_duration)
    p_event2 = 1 - math.exp(-hazard2 * total_duration)
    
    # Adjust for dropouts
    p_event1 *= (1 - dropout_rate)
    p_event2 *= (1 - dropout_rate)
    
    # Calculate expected number of events
    events1 = n1 * p_event1
    events2 = n2 * p_event2
    total_events = events1 + events2
    
    # Calculate effect size for test (log of hazard ratio difference from margin)
    effect_size = math.log(non_inferiority_margin / assumed_hazard_ratio)
    
    # Calculate standard error of log hazard ratio
    allocation_ratio = n2 / n1
    se_log_hr = math.sqrt((1 + allocation_ratio)**2 / (allocation_ratio * total_events))
    
    # Calculate non-centrality parameter
    ncp = effect_size / se_log_hr
    
    # Critical value (one-sided for non-inferiority)
    z_alpha = stats.norm.ppf(1 - alpha)
    
    # Calculate power
    power = 1 - stats.norm.cdf(z_alpha - ncp)
    
    # Return results
    return {
        "power": power,
        "n1": n1,
        "n2": n2,
        "total_sample_size": n1 + n2,
        "expected_events_1": events1,
        "expected_events_2": events2,
        "total_events": total_events,
        "median1": median1,
        "median2": median2,
        "hazard1": hazard1,
        "hazard2": hazard2,
        "assumed_hazard_ratio": assumed_hazard_ratio,
        "non_inferiority_margin": non_inferiority_margin,
        "enrollment_period": enrollment_period,
        "follow_up_period": follow_up_period,
        "dropout_rate": dropout_rate,
        "alpha": alpha,
        "method": "analytical"
    }
