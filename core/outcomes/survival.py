"""
Utilities for survival outcomes in sample size and power calculation.

This module provides common functions and utilities for working with
survival (time-to-event) outcomes across different study designs.
"""
import numpy as np
import math
from scipy import stats


def hazard_ratio_to_median_survival(median_control, hazard_ratio):
    """
    Calculate median survival in the treatment group given the median survival
    in the control group and the hazard ratio.
    
    Parameters
    ----------
    median_control : float
        Median survival time in control group
    hazard_ratio : float
        Hazard ratio (treatment vs control)
    
    Returns
    -------
    float
        Median survival time in treatment group
    """
    return median_control * (math.log(2) / (hazard_ratio * math.log(2)))


def median_survival_to_hazard_ratio(median_control, median_treatment):
    """
    Calculate hazard ratio given median survival times in control and treatment groups.
    
    Parameters
    ----------
    median_control : float
        Median survival time in control group
    median_treatment : float
        Median survival time in treatment group
    
    Returns
    -------
    float
        Hazard ratio (treatment vs control)
    """
    return math.log(2) / median_treatment * (median_control / math.log(2))


def sample_size_survival(hazard_ratio, control_rate, treatment_rate=None, 
                       power=0.8, alpha=0.05, allocation_ratio=1.0, 
                       dropout_control=0.0, dropout_treatment=None):
    """
    Calculate sample size for comparing two survival curves.
    
    Parameters
    ----------
    hazard_ratio : float
        Expected hazard ratio (treatment vs control)
    control_rate : float
        Event rate in the control group
    treatment_rate : float, optional
        Event rate in the treatment group, by default None (calculated from hazard_ratio)
    power : float, optional
        Desired power, by default 0.8
    alpha : float, optional
        Significance level, by default 0.05
    allocation_ratio : float, optional
        Ratio of participants in treatment group to control group, by default 1.0
    dropout_control : float, optional
        Proportion of participants expected to drop out in control group, by default 0.0
    dropout_treatment : float, optional
        Proportion of participants expected to drop out in treatment group, by default None (equal to dropout_control)
    
    Returns
    -------
    dict
        Dictionary containing the required number of participants and events
    """
    # If treatment_rate not provided, calculate from hazard ratio
    if treatment_rate is None:
        treatment_rate = control_rate * hazard_ratio
    
    # If dropout_treatment not provided, set equal to dropout_control
    if dropout_treatment is None:
        dropout_treatment = dropout_control
    
    # Calculate z-scores for given alpha and power
    z_alpha = stats.norm.ppf(1 - alpha/2)
    z_beta = stats.norm.ppf(power)
    
    # Calculate required number of events
    events_required = (z_alpha + z_beta)**2 / (math.log(hazard_ratio)**2) * (1 + allocation_ratio) / allocation_ratio
    
    # Adjust for dropouts
    effective_control_rate = control_rate * (1 - dropout_control)
    effective_treatment_rate = treatment_rate * (1 - dropout_treatment)
    
    # Calculate weighted average event rate
    p1 = 1 / (1 + allocation_ratio)  # Proportion in control
    p2 = allocation_ratio / (1 + allocation_ratio)  # Proportion in treatment
    avg_event_rate = p1 * effective_control_rate + p2 * effective_treatment_rate
    
    # Calculate total sample size
    n_total = events_required / avg_event_rate
    
    # Calculate sample size per group
    n_control = n_total / (1 + allocation_ratio)
    n_treatment = n_total - n_control
    
    return {
        "n_control": math.ceil(n_control),
        "n_treatment": math.ceil(n_treatment),
        "n_total": math.ceil(n_total),
        "events_required": math.ceil(events_required),
        "parameters": {
            "hazard_ratio": hazard_ratio,
            "control_rate": control_rate,
            "treatment_rate": treatment_rate,
            "power": power,
            "alpha": alpha,
            "allocation_ratio": allocation_ratio,
            "dropout_control": dropout_control,
            "dropout_treatment": dropout_treatment
        }
    }


def power_survival(n_control, n_treatment, hazard_ratio, control_rate, 
                 treatment_rate=None, alpha=0.05, dropout_control=0.0, dropout_treatment=None):
    """
    Calculate power for comparing two survival curves.
    
    Parameters
    ----------
    n_control : int
        Number of participants in control group
    n_treatment : int
        Number of participants in treatment group
    hazard_ratio : float
        Expected hazard ratio (treatment vs control)
    control_rate : float
        Event rate in the control group
    treatment_rate : float, optional
        Event rate in the treatment group, by default None (calculated from hazard_ratio)
    alpha : float, optional
        Significance level, by default 0.05
    dropout_control : float, optional
        Proportion of participants expected to drop out in control group, by default 0.0
    dropout_treatment : float, optional
        Proportion of participants expected to drop out in treatment group, by default None (equal to dropout_control)
    
    Returns
    -------
    dict
        Dictionary containing the calculated power and parameters
    """
    # If treatment_rate not provided, calculate from hazard ratio
    if treatment_rate is None:
        treatment_rate = control_rate * hazard_ratio
    
    # If dropout_treatment not provided, set equal to dropout_control
    if dropout_treatment is None:
        dropout_treatment = dropout_control
    
    # Calculate z-score for given alpha
    z_alpha = stats.norm.ppf(1 - alpha/2)
    
    # Adjust for dropouts
    effective_control_rate = control_rate * (1 - dropout_control)
    effective_treatment_rate = treatment_rate * (1 - dropout_treatment)
    
    # Calculate expected number of events
    events_control = n_control * effective_control_rate
    events_treatment = n_treatment * effective_treatment_rate
    
    # Calculate allocation ratio
    allocation_ratio = n_treatment / n_control
    
    # Calculate non-centrality parameter
    ncp = math.log(hazard_ratio) * math.sqrt(events_control * events_treatment / (events_control + events_treatment))
    
    # Calculate power
    power = stats.norm.cdf(ncp - z_alpha) + stats.norm.cdf(-ncp - z_alpha)
    
    return {
        "power": power,
        "expected_events": events_control + events_treatment,
        "parameters": {
            "n_control": n_control,
            "n_treatment": n_treatment,
            "hazard_ratio": hazard_ratio,
            "control_rate": control_rate,
            "treatment_rate": treatment_rate,
            "alpha": alpha,
            "dropout_control": dropout_control,
            "dropout_treatment": dropout_treatment
        }
    }


def min_detectable_hazard_ratio(n_control, n_treatment, control_rate, 
                              power=0.8, alpha=0.05, dropout_control=0.0, dropout_treatment=None):
    """
    Calculate minimum detectable hazard ratio.
    
    Parameters
    ----------
    n_control : int
        Number of participants in control group
    n_treatment : int
        Number of participants in treatment group
    control_rate : float
        Event rate in the control group
    power : float, optional
        Desired power, by default 0.8
    alpha : float, optional
        Significance level, by default 0.05
    dropout_control : float, optional
        Proportion of participants expected to drop out in control group, by default 0.0
    dropout_treatment : float, optional
        Proportion of participants expected to drop out in treatment group, by default None (equal to dropout_control)
    
    Returns
    -------
    dict
        Dictionary containing the minimum detectable hazard ratio and parameters
    """
    # If dropout_treatment not provided, set equal to dropout_control
    if dropout_treatment is None:
        dropout_treatment = dropout_control
    
    # Calculate z-scores for given alpha and power
    z_alpha = stats.norm.ppf(1 - alpha/2)
    z_beta = stats.norm.ppf(power)
    
    # Adjust for dropouts
    effective_control_rate = control_rate * (1 - dropout_control)
    
    # Calculate expected number of events in control group
    events_control = n_control * effective_control_rate
    
    # Calculate allocation ratio
    allocation_ratio = n_treatment / n_control
    
    # Calculate effective event rate in treatment group based on control rate
    effective_treatment_rate = effective_control_rate  # Initial guess
    
    # Iteratively solve for hazard ratio
    # (This is an approximation; a more accurate approach would use numerical optimization)
    hazard_ratio = math.exp(
        (z_alpha + z_beta) * 
        math.sqrt((1 + allocation_ratio) / (allocation_ratio * events_control))
    )
    
    # Calculate treatment event rate from hazard ratio
    treatment_rate = control_rate * hazard_ratio
    effective_treatment_rate = treatment_rate * (1 - dropout_treatment)
    
    return {
        "hazard_ratio": hazard_ratio,
        "treatment_rate": treatment_rate,
        "parameters": {
            "n_control": n_control,
            "n_treatment": n_treatment,
            "control_rate": control_rate,
            "power": power,
            "alpha": alpha,
            "dropout_control": dropout_control,
            "dropout_treatment": dropout_treatment
        }
    }
