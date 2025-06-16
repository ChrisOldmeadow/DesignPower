"""
Advanced survival power calculation methods for parallel group RCTs.

This module implements established methods for survival analysis power calculations:
- Schoenfeld (1983): Standard log-rank test formula
- Freedman (1982): Alternative approach with different assumptions  
- Lakatos (1988): Complex accrual and follow-up patterns

References
----------
Schoenfeld, D. (1983). Sample-size formula for the proportional-hazards regression model. 
    Biometrics, 39(2), 499-503.
Freedman, L. S. (1982). Tables of the number of patients required in clinical trials 
    using the logrank test. Statistics in Medicine, 1(2), 121-129.
Lakatos, E. (1988). Sample sizes based on the log-rank statistic in complex clinical trials. 
    Biometrics, 44(1), 229-241.
"""

import numpy as np
import math
from scipy import stats
from scipy.integrate import quad
from typing import Dict, Optional, Union, List


# ===== Method Selection Guidance =====

METHODS_GUIDANCE = {
    "schoenfeld": {
        "name": "Schoenfeld (1983)",
        "best_for": [
            "Standard log-rank test power calculations",
            "Simple trial designs with uniform accrual",
            "When proportional hazards assumption holds",
            "Quick initial sample size estimates"
        ],
        "assumptions": [
            "Exponential survival distributions",
            "Uniform patient accrual over enrollment period",
            "Proportional hazards (constant hazard ratio)",
            "Independent censoring"
        ],
        "limitations": [
            "Does not account for complex accrual patterns",
            "Assumes simple study design (accrual + follow-up)",
            "May be inaccurate for non-uniform enrollment"
        ],
        "when_to_use": "First choice for standard clinical trials with uniform accrual"
    },
    
    "freedman": {
        "name": "Freedman (1982)", 
        "best_for": [
            "Trials with different assumptions about censoring",
            "When Schoenfeld assumptions may be violated",
            "Historical comparison with published tables",
            "Alternative validation of sample size estimates"
        ],
        "assumptions": [
            "Log-rank test statistic",
            "Different approach to handling censoring",
            "Exponential survival distributions",
            "Independent censoring"
        ],
        "limitations": [
            "Less commonly used than Schoenfeld",
            "May give different results from Schoenfeld",
            "Not as well validated in modern software"
        ],
        "when_to_use": "Secondary method for validation or when Schoenfeld seems inappropriate"
    },
    
    "lakatos": {
        "name": "Lakatos (1988)",
        "best_for": [
            "Complex accrual patterns (non-uniform enrollment)",
            "Trials with varying follow-up periods",
            "Multi-stage or adaptive trial designs", 
            "Maximum precision when trial design is complex"
        ],
        "assumptions": [
            "Flexible accrual patterns",
            "Accounts for complex study timelines",
            "Proportional hazards",
            "Can handle piecewise enrollment rates"
        ],
        "limitations": [
            "More complex to implement and understand",
            "Requires more detailed specification of trial design",
            "Computational complexity higher"
        ],
        "when_to_use": "When accrual is non-uniform or trial design is complex"
    }
}


# ===== Core Mathematical Functions =====

def _calculate_survival_probabilities(hazard_rate: float, times: np.ndarray) -> np.ndarray:
    """Calculate survival probabilities for exponential distribution."""
    return np.exp(-hazard_rate * times)


def _calculate_event_probabilities(hazard_rate: float, times: np.ndarray) -> np.ndarray:
    """Calculate event probabilities for exponential distribution."""
    return 1 - np.exp(-hazard_rate * times)


def _hypergeometric_variance(n1: int, n2: int, d1: int, d2: int) -> float:
    """
    Calculate hypergeometric variance for log-rank test.
    
    Parameters
    ----------
    n1, n2 : int
        Number at risk in groups 1 and 2
    d1, d2 : int  
        Number of events in groups 1 and 2
    """
    n = n1 + n2
    d = d1 + d2
    if n <= 1 or d == 0:
        return 0.0
    return (n1 * n2 * d * (n - d)) / (n * n * (n - 1))


# ===== Schoenfeld Method (1983) =====

def schoenfeld_sample_size(hazard_ratio: float, power: float = 0.8, alpha: float = 0.05, 
                          allocation_ratio: float = 1.0, sides: int = 2,
                          enrollment_period: float = 12.0, follow_up_period: float = 12.0,
                          median_control: Optional[float] = None, 
                          hazard_control: Optional[float] = None,
                          dropout_rate: float = 0.0) -> Dict[str, Union[float, int]]:
    """
    Calculate sample size using Schoenfeld (1983) method.
    
    This is the standard method for log-rank test sample size calculations,
    widely implemented in statistical software.
    
    Parameters
    ----------
    hazard_ratio : float
        Hazard ratio (treatment/control)
    power : float, default 0.8
        Desired statistical power
    alpha : float, default 0.05
        Type I error rate
    allocation_ratio : float, default 1.0
        Ratio of treatment to control sample sizes (n_treatment/n_control)
    sides : int, default 2
        1 for one-sided test, 2 for two-sided test
    enrollment_period : float, default 12.0
        Duration of patient enrollment (months)
    follow_up_period : float, default 12.0 
        Additional follow-up after enrollment ends (months)
    median_control : float, optional
        Median survival in control group (months)
    hazard_control : float, optional
        Hazard rate in control group (per month)
    dropout_rate : float, default 0.0
        Proportion of patients lost to follow-up
        
    Returns
    -------
    dict
        Sample size calculation results including:
        - n_control, n_treatment, n_total: Sample sizes
        - events_required: Number of events needed
        - prob_event_control, prob_event_treatment: Event probabilities
        - method: "schoenfeld"
    """
    # Input validation
    if hazard_ratio <= 0:
        raise ValueError("Hazard ratio must be positive")
    if not 0 < power < 1:
        raise ValueError("Power must be between 0 and 1")
    if not 0 < alpha < 1:
        raise ValueError("Alpha must be between 0 and 1")
    if allocation_ratio <= 0:
        raise ValueError("Allocation ratio must be positive")
    if sides not in [1, 2]:
        raise ValueError("Sides must be 1 or 2")
    if enrollment_period <= 0 or follow_up_period <= 0:
        raise ValueError("Enrollment and follow-up periods must be positive")
        
    # Determine hazard rate
    if hazard_control is not None:
        lambda_control = hazard_control
    elif median_control is not None:
        lambda_control = math.log(2) / median_control
    else:
        raise ValueError("Must specify either median_control or hazard_control")
        
    lambda_treatment = lambda_control * hazard_ratio
    
    # Critical values
    z_alpha = stats.norm.ppf(1 - alpha / sides)
    z_beta = stats.norm.ppf(power)
    
    # Core Schoenfeld formula for required events
    # E = (z_α + z_β)² / (log(HR))² × (r+1)²/(4r)
    # where r = allocation_ratio
    log_hr = math.log(hazard_ratio)
    events_required = ((z_alpha + z_beta) ** 2 / log_hr ** 2 * 
                      (allocation_ratio + 1) ** 2 / (4 * allocation_ratio))
    
    # Calculate study duration and event probabilities
    total_duration = enrollment_period + follow_up_period
    
    # For uniform accrual, average follow-up time is:
    # (enrollment_period/2) + follow_up_period
    avg_followup = enrollment_period / 2 + follow_up_period
    
    # Event probabilities accounting for dropout
    prob_event_control = (1 - math.exp(-lambda_control * avg_followup)) * (1 - dropout_rate)
    prob_event_treatment = (1 - math.exp(-lambda_treatment * avg_followup)) * (1 - dropout_rate)
    
    # Weighted average event probability
    prob_event_avg = (prob_event_control + allocation_ratio * prob_event_treatment) / (1 + allocation_ratio)
    
    # Total sample size
    n_total = math.ceil(events_required / prob_event_avg)
    
    # Individual group sizes
    n_control = math.ceil(n_total / (1 + allocation_ratio))
    n_treatment = math.ceil(n_control * allocation_ratio)
    n_total_actual = n_control + n_treatment
    
    return {
        "method": "schoenfeld",
        "n_control": n_control,
        "n_treatment": n_treatment, 
        "n_total": n_total_actual,
        "events_required": events_required,
        "prob_event_control": prob_event_control,
        "prob_event_treatment": prob_event_treatment,
        "prob_event_avg": prob_event_avg,
        "hazard_control": lambda_control,
        "hazard_treatment": lambda_treatment,
        "hazard_ratio": hazard_ratio,
        "avg_followup": avg_followup,
        "total_duration": total_duration,
        "power": power,
        "alpha": alpha,
        "sides": sides,
        "allocation_ratio": allocation_ratio
    }


def schoenfeld_power(n_control: int, n_treatment: int, hazard_ratio: float, 
                    alpha: float = 0.05, sides: int = 2,
                    enrollment_period: float = 12.0, follow_up_period: float = 12.0,
                    median_control: Optional[float] = None,
                    hazard_control: Optional[float] = None,
                    dropout_rate: float = 0.0) -> Dict[str, Union[float, int]]:
    """
    Calculate power using Schoenfeld (1983) method.
    
    Parameters
    ----------
    n_control : int
        Sample size in control group
    n_treatment : int
        Sample size in treatment group
    hazard_ratio : float
        Hazard ratio (treatment/control)
    alpha : float, default 0.05
        Type I error rate
    sides : int, default 2
        1 for one-sided test, 2 for two-sided test
    enrollment_period : float, default 12.0
        Duration of patient enrollment (months)
    follow_up_period : float, default 12.0
        Additional follow-up after enrollment ends (months)
    median_control : float, optional
        Median survival in control group (months)
    hazard_control : float, optional
        Hazard rate in control group (per month)
    dropout_rate : float, default 0.0
        Proportion of patients lost to follow-up
        
    Returns
    -------
    dict
        Power calculation results
    """
    # Input validation
    if n_control <= 0 or n_treatment <= 0:
        raise ValueError("Sample sizes must be positive")
    if hazard_ratio <= 0:
        raise ValueError("Hazard ratio must be positive")
    if not 0 < alpha < 1:
        raise ValueError("Alpha must be between 0 and 1")
        
    # Determine hazard rate
    if hazard_control is not None:
        lambda_control = hazard_control
    elif median_control is not None:
        lambda_control = math.log(2) / median_control
    else:
        raise ValueError("Must specify either median_control or hazard_control")
        
    lambda_treatment = lambda_control * hazard_ratio
    allocation_ratio = n_treatment / n_control
    
    # Calculate event probabilities
    avg_followup = enrollment_period / 2 + follow_up_period
    prob_event_control = (1 - math.exp(-lambda_control * avg_followup)) * (1 - dropout_rate)
    prob_event_treatment = (1 - math.exp(-lambda_treatment * avg_followup)) * (1 - dropout_rate)
    
    # Expected number of events
    expected_events_control = n_control * prob_event_control
    expected_events_treatment = n_treatment * prob_event_treatment
    total_events = expected_events_control + expected_events_treatment
    
    # Critical value
    z_alpha = stats.norm.ppf(1 - alpha / sides)
    
    # Calculate power using Schoenfeld formula
    log_hr = math.log(hazard_ratio)
    test_statistic = abs(log_hr) * math.sqrt(total_events * allocation_ratio / (allocation_ratio + 1) ** 2)
    power = 1 - stats.norm.cdf(z_alpha - test_statistic)
    
    return {
        "method": "schoenfeld",
        "power": power,
        "n_control": n_control,
        "n_treatment": n_treatment,
        "n_total": n_control + n_treatment,
        "expected_events": total_events,
        "expected_events_control": expected_events_control,
        "expected_events_treatment": expected_events_treatment,
        "prob_event_control": prob_event_control,
        "prob_event_treatment": prob_event_treatment,
        "hazard_control": lambda_control,
        "hazard_treatment": lambda_treatment,
        "hazard_ratio": hazard_ratio,
        "avg_followup": avg_followup,
        "alpha": alpha,
        "sides": sides,
        "allocation_ratio": allocation_ratio
    }


# ===== Freedman Method (1982) =====

def freedman_sample_size(hazard_ratio: float, power: float = 0.8, alpha: float = 0.05,
                        allocation_ratio: float = 1.0, sides: int = 2,
                        enrollment_period: float = 12.0, follow_up_period: float = 12.0,
                        median_control: Optional[float] = None,
                        hazard_control: Optional[float] = None,
                        dropout_rate: float = 0.0) -> Dict[str, Union[float, int]]:
    """
    Calculate sample size using Freedman (1982) method.
    
    Alternative approach to survival sample size calculation with different
    handling of censoring and event probability calculations.
    
    Parameters
    ----------
    Same as schoenfeld_sample_size
        
    Returns
    -------
    dict
        Sample size calculation results
    """
    # Input validation (same as Schoenfeld)
    if hazard_ratio <= 0:
        raise ValueError("Hazard ratio must be positive")
    if not 0 < power < 1:
        raise ValueError("Power must be between 0 and 1")
    if not 0 < alpha < 1:
        raise ValueError("Alpha must be between 0 and 1")
        
    # Determine hazard rate
    if hazard_control is not None:
        lambda_control = hazard_control
    elif median_control is not None:
        lambda_control = math.log(2) / median_control
    else:
        raise ValueError("Must specify either median_control or hazard_control")
        
    lambda_treatment = lambda_control * hazard_ratio
    
    # Critical values
    z_alpha = stats.norm.ppf(1 - alpha / sides)
    z_beta = stats.norm.ppf(power)
    
    # Freedman's approach: different calculation of effective sample size
    # Accounts for censoring differently than Schoenfeld
    
    # Calculate event probabilities using Freedman's approach
    total_duration = enrollment_period + follow_up_period
    
    # Freedman uses a different weighting for event probabilities
    # More conservative approach to censoring
    t_min = follow_up_period  # Minimum follow-up time
    t_max = total_duration     # Maximum follow-up time
    
    # Average survival probabilities over the range of follow-up times
    def integrand_control(t):
        return math.exp(-lambda_control * t)
    
    def integrand_treatment(t):
        return math.exp(-lambda_treatment * t)
    
    # Integration over follow-up period
    survival_integral_control, _ = quad(integrand_control, t_min, t_max)
    survival_integral_treatment, _ = quad(integrand_treatment, t_min, t_max)
    
    # Average survival probabilities
    avg_survival_control = survival_integral_control / enrollment_period
    avg_survival_treatment = survival_integral_treatment / enrollment_period
    
    # Event probabilities accounting for dropout
    prob_event_control = (1 - avg_survival_control) * (1 - dropout_rate)
    prob_event_treatment = (1 - avg_survival_treatment) * (1 - dropout_rate)
    
    # Freedman's formula for required events (similar core formula)
    log_hr = math.log(hazard_ratio)
    events_required = ((z_alpha + z_beta) ** 2 / log_hr ** 2 * 
                      (allocation_ratio + 1) ** 2 / (4 * allocation_ratio))
    
    # Weighted average event probability (Freedman approach)
    prob_event_avg = (prob_event_control + allocation_ratio * prob_event_treatment) / (1 + allocation_ratio)
    
    # Total sample size
    n_total = math.ceil(events_required / prob_event_avg)
    
    # Individual group sizes
    n_control = math.ceil(n_total / (1 + allocation_ratio))
    n_treatment = math.ceil(n_control * allocation_ratio)
    n_total_actual = n_control + n_treatment
    
    return {
        "method": "freedman",
        "n_control": n_control,
        "n_treatment": n_treatment,
        "n_total": n_total_actual,
        "events_required": events_required,
        "prob_event_control": prob_event_control,
        "prob_event_treatment": prob_event_treatment,
        "prob_event_avg": prob_event_avg,
        "avg_survival_control": avg_survival_control,
        "avg_survival_treatment": avg_survival_treatment,
        "hazard_control": lambda_control,
        "hazard_treatment": lambda_treatment,
        "hazard_ratio": hazard_ratio,
        "total_duration": total_duration,
        "power": power,
        "alpha": alpha,
        "sides": sides,
        "allocation_ratio": allocation_ratio
    }


def freedman_power(n_control: int, n_treatment: int, hazard_ratio: float,
                  alpha: float = 0.05, sides: int = 2,
                  enrollment_period: float = 12.0, follow_up_period: float = 12.0,
                  median_control: Optional[float] = None,
                  hazard_control: Optional[float] = None,
                  dropout_rate: float = 0.0) -> Dict[str, Union[float, int]]:
    """
    Calculate power using Freedman (1982) method.
    
    Parameters
    ----------
    Same as schoenfeld_power
        
    Returns
    -------
    dict
        Power calculation results
    """
    # Input validation
    if n_control <= 0 or n_treatment <= 0:
        raise ValueError("Sample sizes must be positive")
    if hazard_ratio <= 0:
        raise ValueError("Hazard ratio must be positive")
        
    # Determine hazard rate
    if hazard_control is not None:
        lambda_control = hazard_control
    elif median_control is not None:
        lambda_control = math.log(2) / median_control
    else:
        raise ValueError("Must specify either median_control or hazard_control")
        
    lambda_treatment = lambda_control * hazard_ratio
    allocation_ratio = n_treatment / n_control
    
    # Calculate event probabilities using Freedman approach
    total_duration = enrollment_period + follow_up_period
    t_min = follow_up_period
    t_max = total_duration
    
    # Integration approach for more accurate event probabilities
    def integrand_control(t):
        return math.exp(-lambda_control * t)
    
    def integrand_treatment(t):
        return math.exp(-lambda_treatment * t)
    
    survival_integral_control, _ = quad(integrand_control, t_min, t_max)
    survival_integral_treatment, _ = quad(integrand_treatment, t_min, t_max)
    
    avg_survival_control = survival_integral_control / enrollment_period
    avg_survival_treatment = survival_integral_treatment / enrollment_period
    
    prob_event_control = (1 - avg_survival_control) * (1 - dropout_rate)
    prob_event_treatment = (1 - avg_survival_treatment) * (1 - dropout_rate)
    
    # Expected number of events
    expected_events_control = n_control * prob_event_control
    expected_events_treatment = n_treatment * prob_event_treatment
    total_events = expected_events_control + expected_events_treatment
    
    # Critical value
    z_alpha = stats.norm.ppf(1 - alpha / sides)
    
    # Calculate power
    log_hr = math.log(hazard_ratio)
    test_statistic = abs(log_hr) * math.sqrt(total_events * allocation_ratio / (allocation_ratio + 1) ** 2)
    power = 1 - stats.norm.cdf(z_alpha - test_statistic)
    
    return {
        "method": "freedman",
        "power": power,
        "n_control": n_control,
        "n_treatment": n_treatment,
        "n_total": n_control + n_treatment,
        "expected_events": total_events,
        "expected_events_control": expected_events_control,
        "expected_events_treatment": expected_events_treatment,
        "prob_event_control": prob_event_control,
        "prob_event_treatment": prob_event_treatment,
        "avg_survival_control": avg_survival_control,
        "avg_survival_treatment": avg_survival_treatment,
        "hazard_control": lambda_control,
        "hazard_treatment": lambda_treatment,
        "hazard_ratio": hazard_ratio,
        "alpha": alpha,
        "sides": sides,
        "allocation_ratio": allocation_ratio
    }


# ===== Lakatos Method (1988) =====

def lakatos_sample_size(hazard_ratio: float, power: float = 0.8, alpha: float = 0.05,
                       allocation_ratio: float = 1.0, sides: int = 2,
                       accrual_pattern: str = "uniform", accrual_parameters: Optional[Dict] = None,
                       median_control: Optional[float] = None,
                       hazard_control: Optional[float] = None,
                       dropout_rate: float = 0.0,
                       time_periods: Optional[List[float]] = None) -> Dict[str, Union[float, int]]:
    """
    Calculate sample size using Lakatos (1988) method.
    
    Advanced method that accounts for complex accrual patterns and varying 
    follow-up periods. Most accurate for non-uniform enrollment.
    
    Parameters
    ----------
    hazard_ratio : float
        Hazard ratio (treatment/control)
    power : float, default 0.8
        Desired statistical power
    alpha : float, default 0.05
        Type I error rate
    allocation_ratio : float, default 1.0
        Ratio of treatment to control sample sizes
    sides : int, default 2
        1 for one-sided test, 2 for two-sided test
    accrual_pattern : str, default "uniform"
        Type of accrual pattern: "uniform", "ramp_up", "exponential", "piecewise"
    accrual_parameters : dict, optional
        Parameters specific to accrual pattern
    median_control : float, optional
        Median survival in control group (months)
    hazard_control : float, optional
        Hazard rate in control group (per month)
    dropout_rate : float, default 0.0
        Proportion of patients lost to follow-up
    time_periods : list of float, optional
        Time points for piecewise calculations (months)
        
    Returns
    -------
    dict
        Sample size calculation results
    """
    # Input validation
    if hazard_ratio <= 0:
        raise ValueError("Hazard ratio must be positive")
    if not 0 < power < 1:
        raise ValueError("Power must be between 0 and 1")
    if not 0 < alpha < 1:
        raise ValueError("Alpha must be between 0 and 1")
        
    # Determine hazard rate
    if hazard_control is not None:
        lambda_control = hazard_control
    elif median_control is not None:
        lambda_control = math.log(2) / median_control
    else:
        raise ValueError("Must specify either median_control or hazard_control")
        
    lambda_treatment = lambda_control * hazard_ratio
    
    # Set default parameters if not provided
    if accrual_parameters is None:
        accrual_parameters = {"enrollment_period": 12.0, "follow_up_period": 12.0}
    
    # Extract periods first
    enrollment_period = accrual_parameters.get("enrollment_period", 12.0)
    follow_up_period = accrual_parameters.get("follow_up_period", 12.0)
    total_duration = enrollment_period + follow_up_period
    
    if time_periods is None:
        # Default to quarterly intervals over study duration
        time_periods = list(np.linspace(1, total_duration, max(4, int(total_duration/3))))
    
    # Critical values
    z_alpha = stats.norm.ppf(1 - alpha / sides)
    z_beta = stats.norm.ppf(power)
    
    # Core formula for required events (same as other methods)
    log_hr = math.log(hazard_ratio)
    events_required = ((z_alpha + z_beta) ** 2 / log_hr ** 2 * 
                      (allocation_ratio + 1) ** 2 / (4 * allocation_ratio))
    
    # Calculate average follow-up time (used by both uniform and non-uniform)
    avg_followup = enrollment_period / 2 + follow_up_period
    
    # For uniform accrual, use analytical formula (should match Schoenfeld closely)
    if accrual_pattern == "uniform":
        
        # Event probabilities
        prob_event_control = (1 - math.exp(-lambda_control * avg_followup)) * (1 - dropout_rate)
        prob_event_treatment = (1 - math.exp(-lambda_treatment * avg_followup)) * (1 - dropout_rate)
        
        # Weighted average event probability
        prob_event_avg = (prob_event_control + allocation_ratio * prob_event_treatment) / (1 + allocation_ratio)
        
        # Total sample size
        n_total = math.ceil(events_required / prob_event_avg)
        
        # For debugging/validation
        cumulative_events = n_total * prob_event_avg
        variance_correction = 1.0
        accrual_rates = [1.0]  # Uniform accrual represented as single rate
        
    else:
        # Complex accrual patterns - use piecewise calculation
        # This is where Lakatos method really differs from others
        
        # Calculate accrual rates for each period
        n_periods = max(4, int(enrollment_period))  # At least 4 periods, typically monthly
        period_length = enrollment_period / n_periods
        
        accrual_rates = _calculate_accrual_rates(accrual_pattern, enrollment_period, 
                                               list(range(1, n_periods + 1)), accrual_parameters)
        
        # Calculate expected events for unit total sample size
        total_events_per_unit = 0
        
        for i in range(n_periods):
            # Fraction of total sample enrolled in this period
            period_fraction = accrual_rates[i]
            
            # Enrollment time for this period (middle of period)
            enrollment_time = (i + 0.5) * period_length
            
            # Follow-up time for patients enrolled in this period
            followup_time = total_duration - enrollment_time
            
            # Event probabilities for this period
            prob_event_control = (1 - math.exp(-lambda_control * followup_time)) * (1 - dropout_rate)
            prob_event_treatment = (1 - math.exp(-lambda_treatment * followup_time)) * (1 - dropout_rate)
            
            # Contribution to total events (per unit sample size)
            period_events_control = period_fraction / (1 + allocation_ratio) * prob_event_control
            period_events_treatment = period_fraction * allocation_ratio / (1 + allocation_ratio) * prob_event_treatment
            
            total_events_per_unit += period_events_control + period_events_treatment
        
        # Scale to get required sample size
        if total_events_per_unit > 0:
            n_total = math.ceil(events_required / total_events_per_unit)
        else:
            # Fallback
            avg_followup = enrollment_period / 2 + follow_up_period
            prob_event_control = (1 - math.exp(-lambda_control * avg_followup)) * (1 - dropout_rate)
            prob_event_treatment = (1 - math.exp(-lambda_treatment * avg_followup)) * (1 - dropout_rate)
            avg_prob_event = (prob_event_control + allocation_ratio * prob_event_treatment) / (1 + allocation_ratio)
            n_total = math.ceil(events_required / avg_prob_event)
        
        cumulative_events = n_total * total_events_per_unit
        variance_correction = 1.0  # Simplified for now
    
    # Individual group sizes
    n_control = math.ceil(n_total / (1 + allocation_ratio))
    n_treatment = math.ceil(n_control * allocation_ratio)
    n_total_actual = n_control + n_treatment
    
    return {
        "method": "lakatos",
        "n_control": n_control,
        "n_treatment": n_treatment,
        "n_total": n_total_actual,
        "events_required": events_required,
        "cumulative_events": cumulative_events,
        "variance_correction": variance_correction,
        "accrual_pattern": accrual_pattern,
        "accrual_rates": accrual_rates,
        "time_periods": time_periods,
        "hazard_control": lambda_control,
        "hazard_treatment": lambda_treatment,
        "hazard_ratio": hazard_ratio,
        "total_duration": total_duration,
        "avg_followup": avg_followup,
        "power": power,
        "alpha": alpha,
        "sides": sides,
        "allocation_ratio": allocation_ratio
    }


def _calculate_accrual_rates(pattern: str, enrollment_period: float, 
                           time_periods: List[float], parameters: Dict) -> List[float]:
    """Calculate accrual rates for different patterns."""
    n_periods = len([t for t in time_periods if t <= enrollment_period])
    
    if pattern == "uniform":
        # Equal enrollment each period
        rate_per_period = 1.0 / n_periods
        return [rate_per_period] * n_periods
    
    elif pattern == "ramp_up":
        # Linear increase in enrollment rate
        ramp_factor = parameters.get("ramp_factor", 2.0)
        rates = []
        for i in range(n_periods):
            # Linear ramp from 0.5 to ramp_factor
            rate = 0.5 + (ramp_factor - 0.5) * i / (n_periods - 1)
            rates.append(rate)
        # Normalize to sum to 1
        total = sum(rates)
        return [r / total for r in rates]
    
    elif pattern == "exponential":
        # Exponential growth in enrollment
        growth_rate = parameters.get("growth_rate", 0.1)
        rates = []
        for i in range(n_periods):
            rate = math.exp(growth_rate * i)
            rates.append(rate)
        # Normalize to sum to 1
        total = sum(rates)
        return [r / total for r in rates]
    
    elif pattern == "piecewise":
        # User-defined piecewise rates
        custom_rates = parameters.get("rates", [1.0] * n_periods)
        total = sum(custom_rates[:n_periods])
        return [r / total for r in custom_rates[:n_periods]]
    
    else:
        # Default to uniform
        rate_per_period = 1.0 / n_periods
        return [rate_per_period] * n_periods


def _lakatos_variance_contribution(events_control: float, events_treatment: float,
                                 n_control: float, n_treatment: float) -> float:
    """Calculate variance contribution for Lakatos method."""
    if events_control <= 0 or events_treatment <= 0:
        return 0.0
    
    # Hypergeometric-style variance for this time period
    total_events = events_control + events_treatment
    total_n = n_control + n_treatment
    
    if total_n <= 1 or total_events <= 0:
        return 0.0
    
    return (n_control * n_treatment * total_events * (total_n - total_events)) / (total_n ** 2 * (total_n - 1))


# ===== Method Comparison and Selection =====

def compare_methods(hazard_ratio: float, power: float = 0.8, alpha: float = 0.05,
                   allocation_ratio: float = 1.0, sides: int = 2,
                   enrollment_period: float = 12.0, follow_up_period: float = 12.0,
                   median_control: Optional[float] = None,
                   hazard_control: Optional[float] = None,
                   dropout_rate: float = 0.0,
                   accrual_pattern: str = "uniform") -> Dict[str, Dict]:
    """
    Compare sample size calculations across all three methods.
    
    Parameters
    ----------
    Same as individual method functions
    
    Returns
    -------
    dict
        Results from all three methods with comparison summary
    """
    # Calculate using all three methods
    try:
        schoenfeld_result = schoenfeld_sample_size(
            hazard_ratio=hazard_ratio, power=power, alpha=alpha,
            allocation_ratio=allocation_ratio, sides=sides,
            enrollment_period=enrollment_period, follow_up_period=follow_up_period,
            median_control=median_control, hazard_control=hazard_control,
            dropout_rate=dropout_rate
        )
    except Exception as e:
        schoenfeld_result = {"error": str(e)}
    
    try:
        freedman_result = freedman_sample_size(
            hazard_ratio=hazard_ratio, power=power, alpha=alpha,
            allocation_ratio=allocation_ratio, sides=sides,
            enrollment_period=enrollment_period, follow_up_period=follow_up_period,
            median_control=median_control, hazard_control=hazard_control,
            dropout_rate=dropout_rate
        )
    except Exception as e:
        freedman_result = {"error": str(e)}
    
    try:
        lakatos_result = lakatos_sample_size(
            hazard_ratio=hazard_ratio, power=power, alpha=alpha,
            allocation_ratio=allocation_ratio, sides=sides,
            accrual_pattern=accrual_pattern,
            accrual_parameters={"enrollment_period": enrollment_period, 
                              "follow_up_period": follow_up_period},
            median_control=median_control, hazard_control=hazard_control,
            dropout_rate=dropout_rate
        )
    except Exception as e:
        lakatos_result = {"error": str(e)}
    
    # Create comparison summary
    methods_results = {
        "schoenfeld": schoenfeld_result,
        "freedman": freedman_result,
        "lakatos": lakatos_result
    }
    
    # Extract sample sizes for comparison
    sample_sizes = {}
    for method, result in methods_results.items():
        if "error" not in result:
            sample_sizes[method] = result["n_total"]
    
    # Calculate differences
    if len(sample_sizes) > 1:
        sizes = list(sample_sizes.values())
        min_size = min(sizes)
        max_size = max(sizes)
        max_difference = max_size - min_size
        percent_difference = (max_difference / min_size) * 100 if min_size > 0 else 0
    else:
        max_difference = 0
        percent_difference = 0
    
    comparison_summary = {
        "sample_sizes": sample_sizes,
        "max_absolute_difference": max_difference,
        "max_percent_difference": percent_difference,
        "recommended_method": _recommend_method(accrual_pattern, enrollment_period, follow_up_period)
    }
    
    return {
        "methods": methods_results,
        "comparison": comparison_summary,
        "guidance": METHODS_GUIDANCE
    }


def _recommend_method(accrual_pattern: str, enrollment_period: float, follow_up_period: float) -> str:
    """Recommend which method to use based on study design."""
    if accrual_pattern != "uniform":
        return "lakatos"
    elif enrollment_period / follow_up_period > 3 or enrollment_period / follow_up_period < 0.5:
        return "lakatos"
    else:
        return "schoenfeld"


# ===== Convenience Functions =====

def get_method_guidance(method: str) -> Dict:
    """Get guidance for a specific method."""
    return METHODS_GUIDANCE.get(method, {})


def list_available_methods() -> List[str]:
    """List all available methods."""
    return list(METHODS_GUIDANCE.keys())