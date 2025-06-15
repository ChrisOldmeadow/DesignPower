"""
Survival Parameter Conversion Utilities.

This module provides comprehensive functions for converting between different
survival analysis parameters including median survival, hazard rates, survival
fractions, event rates, and hazard ratios.

All conversions assume exponential survival distribution unless otherwise specified.
"""

import math
import numpy as np
from typing import Dict, Union, Optional, List
from dataclasses import dataclass


@dataclass
class SurvivalParameters:
    """Container for all survival parameters with automatic conversion."""
    
    # Primary parameters (user provides one or more)
    median_survival: Optional[float] = None
    hazard_rate: Optional[float] = None 
    survival_fraction: Optional[float] = None
    event_rate: Optional[float] = None
    
    # Time parameters
    time_point: float = 12  # months, for survival fraction and event rate
    
    # Computed parameters (filled automatically)
    _computed: bool = False
    
    def __post_init__(self):
        """Automatically compute all parameters from provided inputs."""
        if not self._computed:
            self._compute_all_parameters()
            self._computed = True
    
    def _compute_all_parameters(self):
        """Compute all survival parameters from the provided input."""
        
        # Count how many parameters are provided
        provided_params = sum([
            self.median_survival is not None,
            self.hazard_rate is not None,
            self.survival_fraction is not None,
            self.event_rate is not None
        ])
        
        if provided_params == 0:
            raise ValueError("At least one survival parameter must be provided")
        
        if provided_params > 1:
            # Validate consistency if multiple parameters provided
            self._validate_consistency()
        
        # Compute missing parameters
        if self.hazard_rate is None:
            self.hazard_rate = self._compute_hazard_rate()
        
        if self.median_survival is None:
            self.median_survival = hazard_to_median(self.hazard_rate)
        
        if self.survival_fraction is None:
            self.survival_fraction = hazard_to_survival_fraction(
                self.hazard_rate, self.time_point
            )
        
        if self.event_rate is None:
            self.event_rate = 1 - self.survival_fraction
    
    def _compute_hazard_rate(self) -> float:
        """Compute hazard rate from available parameters."""
        if self.median_survival is not None:
            return median_to_hazard(self.median_survival)
        elif self.survival_fraction is not None:
            return survival_fraction_to_hazard(self.survival_fraction, self.time_point)
        elif self.event_rate is not None:
            return event_rate_to_hazard(self.event_rate, self.time_point)
        else:
            raise ValueError("Cannot compute hazard rate from provided parameters")
    
    def _validate_consistency(self):
        """Validate that multiple provided parameters are consistent."""
        tolerance = 0.01  # 1% tolerance for floating point errors
        
        # Check median vs hazard
        if self.median_survival is not None and self.hazard_rate is not None:
            expected_median = hazard_to_median(self.hazard_rate)
            if abs(self.median_survival - expected_median) / expected_median > tolerance:
                raise ValueError(
                    f"Inconsistent median survival ({self.median_survival}) "
                    f"and hazard rate ({self.hazard_rate}). "
                    f"Expected median: {expected_median:.3f}"
                )
        
        # Check survival fraction vs event rate
        if self.survival_fraction is not None and self.event_rate is not None:
            if abs(self.survival_fraction + self.event_rate - 1.0) > tolerance:
                raise ValueError(
                    f"Survival fraction ({self.survival_fraction}) and "
                    f"event rate ({self.event_rate}) must sum to 1.0"
                )


# =============================================================================
# CORE CONVERSION FUNCTIONS
# =============================================================================

def median_to_hazard(median_survival: float) -> float:
    """
    Convert median survival time to instantaneous hazard rate.
    
    For exponential distribution: λ = ln(2) / median
    
    Parameters
    ----------
    median_survival : float
        Median survival time (any time unit)
        
    Returns
    -------
    float
        Instantaneous hazard rate (per same time unit)
    """
    if median_survival <= 0:
        raise ValueError("Median survival must be positive")
    
    return math.log(2) / median_survival


def hazard_to_median(hazard_rate: float) -> float:
    """
    Convert instantaneous hazard rate to median survival time.
    
    For exponential distribution: median = ln(2) / λ
    
    Parameters
    ----------
    hazard_rate : float
        Instantaneous hazard rate (per time unit)
        
    Returns
    -------
    float
        Median survival time (same time unit)
    """
    if hazard_rate <= 0:
        raise ValueError("Hazard rate must be positive")
    
    return math.log(2) / hazard_rate


def hazard_to_survival_fraction(hazard_rate: float, time_point: float) -> float:
    """
    Convert hazard rate to survival fraction at specific time.
    
    For exponential distribution: S(t) = exp(-λt)
    
    Parameters
    ----------
    hazard_rate : float
        Instantaneous hazard rate
    time_point : float
        Time at which to calculate survival fraction
        
    Returns
    -------
    float
        Fraction surviving at time_point (between 0 and 1)
    """
    if hazard_rate <= 0:
        raise ValueError("Hazard rate must be positive")
    if time_point < 0:
        raise ValueError("Time point must be non-negative")
    
    return math.exp(-hazard_rate * time_point)


def survival_fraction_to_hazard(survival_fraction: float, time_point: float) -> float:
    """
    Convert survival fraction at specific time to hazard rate.
    
    For exponential distribution: λ = -ln(S(t)) / t
    
    Parameters
    ----------
    survival_fraction : float
        Fraction surviving at time_point (between 0 and 1)
    time_point : float
        Time at which survival fraction is measured
        
    Returns
    -------
    float
        Instantaneous hazard rate
    """
    if not 0 < survival_fraction <= 1:
        raise ValueError("Survival fraction must be between 0 and 1")
    if time_point <= 0:
        raise ValueError("Time point must be positive")
    
    return -math.log(survival_fraction) / time_point


def event_rate_to_hazard(event_rate: float, time_point: float) -> float:
    """
    Convert event rate to hazard rate.
    
    Event rate = 1 - survival fraction, so:
    λ = -ln(1 - event_rate) / t
    
    Parameters
    ----------
    event_rate : float
        Fraction experiencing event by time_point (between 0 and 1)
    time_point : float
        Time at which event rate is measured
        
    Returns
    -------
    float
        Instantaneous hazard rate
    """
    if not 0 <= event_rate < 1:
        raise ValueError("Event rate must be between 0 and 1 (exclusive)")
    if time_point <= 0:
        raise ValueError("Time point must be positive")
    
    survival_fraction = 1 - event_rate
    return survival_fraction_to_hazard(survival_fraction, time_point)


def hazard_to_event_rate(hazard_rate: float, time_point: float) -> float:
    """
    Convert hazard rate to event rate at specific time.
    
    Event rate = 1 - S(t) = 1 - exp(-λt)
    
    Parameters
    ----------
    hazard_rate : float
        Instantaneous hazard rate
    time_point : float
        Time at which to calculate event rate
        
    Returns
    -------
    float
        Fraction experiencing event by time_point (between 0 and 1)
    """
    survival_fraction = hazard_to_survival_fraction(hazard_rate, time_point)
    return 1 - survival_fraction


# =============================================================================
# HAZARD RATIO CONVERSIONS
# =============================================================================

def median_survival_to_hazard_ratio(median_control: float, median_treatment: float) -> float:
    """
    Calculate hazard ratio from median survival times.
    
    HR = λ_treatment / λ_control = median_control / median_treatment
    
    Parameters
    ----------
    median_control : float
        Median survival in control group
    median_treatment : float
        Median survival in treatment group
        
    Returns
    -------
    float
        Hazard ratio (treatment vs control)
    """
    if median_control <= 0 or median_treatment <= 0:
        raise ValueError("Median survival times must be positive")
    
    return median_control / median_treatment


def hazard_ratio_to_median_treatment(hazard_ratio: float, median_control: float) -> float:
    """
    Calculate treatment group median from hazard ratio and control median.
    
    median_treatment = median_control / HR
    
    Parameters
    ----------
    hazard_ratio : float
        Hazard ratio (treatment vs control)
    median_control : float
        Median survival in control group
        
    Returns
    -------
    float
        Median survival in treatment group
    """
    if hazard_ratio <= 0:
        raise ValueError("Hazard ratio must be positive")
    if median_control <= 0:
        raise ValueError("Control median survival must be positive")
    
    return median_control / hazard_ratio


def hazard_ratio_to_median_control(hazard_ratio: float, median_treatment: float) -> float:
    """
    Calculate control group median from hazard ratio and treatment median.
    
    median_control = HR × median_treatment
    
    Parameters
    ----------
    hazard_ratio : float
        Hazard ratio (treatment vs control)
    median_treatment : float
        Median survival in treatment group
        
    Returns
    -------
    float
        Median survival in control group
    """
    if hazard_ratio <= 0:
        raise ValueError("Hazard ratio must be positive")
    if median_treatment <= 0:
        raise ValueError("Treatment median survival must be positive")
    
    return hazard_ratio * median_treatment


# =============================================================================
# COMPREHENSIVE CONVERSION FUNCTIONS
# =============================================================================

def convert_survival_parameters(
    median_survival: Optional[float] = None,
    hazard_rate: Optional[float] = None,
    survival_fraction: Optional[float] = None,
    event_rate: Optional[float] = None,
    time_point: float = 12
) -> Dict[str, float]:
    """
    Convert between any survival parameters.
    
    Provide any one (or more for validation) of the survival parameters,
    and get all equivalent parameters returned.
    
    Parameters
    ----------
    median_survival : float, optional
        Median survival time
    hazard_rate : float, optional
        Instantaneous hazard rate
    survival_fraction : float, optional
        Fraction surviving at time_point
    event_rate : float, optional
        Fraction experiencing event by time_point
    time_point : float, default 12
        Time point for survival fraction and event rate calculations
        
    Returns
    -------
    dict
        Dictionary containing all survival parameters:
        - median_survival
        - hazard_rate
        - survival_fraction
        - event_rate
        - time_point
        
    Examples
    --------
    >>> # Convert from median survival
    >>> convert_survival_parameters(median_survival=12)
    {'median_survival': 12, 'hazard_rate': 0.0578, 'survival_fraction': 0.5, 'event_rate': 0.5, 'time_point': 12}
    
    >>> # Convert from 70% survival at 24 months
    >>> convert_survival_parameters(survival_fraction=0.7, time_point=24)
    {'median_survival': 46.63, 'hazard_rate': 0.0149, 'survival_fraction': 0.7, 'event_rate': 0.3, 'time_point': 24}
    """
    params = SurvivalParameters(
        median_survival=median_survival,
        hazard_rate=hazard_rate,
        survival_fraction=survival_fraction,
        event_rate=event_rate,
        time_point=time_point
    )
    
    return {
        'median_survival': params.median_survival,
        'hazard_rate': params.hazard_rate,
        'survival_fraction': params.survival_fraction,
        'event_rate': params.event_rate,
        'time_point': params.time_point
    }


def convert_hazard_ratio_scenario(
    hazard_ratio: float,
    control_median: Optional[float] = None,
    treatment_median: Optional[float] = None,
    control_hazard: Optional[float] = None,
    treatment_hazard: Optional[float] = None,
    time_point: float = 12
) -> Dict[str, Dict[str, float]]:
    """
    Convert hazard ratio scenario to complete parameter set for both groups.
    
    Provide hazard ratio and any one parameter from either group to get
    complete parameter sets for both groups.
    
    Parameters
    ----------
    hazard_ratio : float
        Hazard ratio (treatment vs control)
    control_median : float, optional
        Control group median survival
    treatment_median : float, optional
        Treatment group median survival
    control_hazard : float, optional
        Control group hazard rate
    treatment_hazard : float, optional
        Treatment group hazard rate
    time_point : float, default 12
        Time point for survival fraction calculations
        
    Returns
    -------
    dict
        Dictionary with 'control' and 'treatment' keys, each containing
        complete parameter sets
        
    Examples
    --------
    >>> # HR=0.7 with control median=12 months
    >>> convert_hazard_ratio_scenario(hazard_ratio=0.7, control_median=12)
    {
        'control': {'median_survival': 12, 'hazard_rate': 0.0578, ...},
        'treatment': {'median_survival': 17.14, 'hazard_rate': 0.0405, ...},
        'hazard_ratio': 0.7
    }
    """
    # Determine which parameters are provided
    control_params_count = sum([
        control_median is not None,
        control_hazard is not None
    ])
    
    treatment_params_count = sum([
        treatment_median is not None,
        treatment_hazard is not None
    ])
    
    if control_params_count + treatment_params_count != 1:
        raise ValueError(
            "Provide exactly one parameter from either control or treatment group"
        )
    
    # Calculate missing medians/hazards using hazard ratio
    if control_median is not None:
        treatment_median = hazard_ratio_to_median_treatment(hazard_ratio, control_median)
    elif treatment_median is not None:
        control_median = hazard_ratio_to_median_control(hazard_ratio, treatment_median)
    elif control_hazard is not None:
        control_median = hazard_to_median(control_hazard)
        treatment_median = hazard_ratio_to_median_treatment(hazard_ratio, control_median)
    elif treatment_hazard is not None:
        treatment_median = hazard_to_median(treatment_hazard)
        control_median = hazard_ratio_to_median_control(hazard_ratio, treatment_median)
    
    # Get complete parameter sets for both groups
    control_params = convert_survival_parameters(
        median_survival=control_median, time_point=time_point
    )
    
    treatment_params = convert_survival_parameters(
        median_survival=treatment_median, time_point=time_point
    )
    
    # Verify hazard ratio calculation
    calculated_hr = treatment_params['hazard_rate'] / control_params['hazard_rate']
    
    return {
        'control': control_params,
        'treatment': treatment_params,
        'hazard_ratio': hazard_ratio,
        'calculated_hazard_ratio': calculated_hr
    }


# =============================================================================
# TIME UNIT CONVERSIONS
# =============================================================================

def convert_time_units(
    value: float,
    from_unit: str,
    to_unit: str
) -> float:
    """
    Convert survival parameters between different time units.
    
    Parameters
    ----------
    value : float
        Value to convert
    from_unit : str
        Original time unit ('days', 'weeks', 'months', 'years')
    to_unit : str
        Target time unit ('days', 'weeks', 'months', 'years')
        
    Returns
    -------
    float
        Converted value
        
    Examples
    --------
    >>> # Convert 12 months to years
    >>> convert_time_units(12, 'months', 'years')
    1.0
    
    >>> # Convert hazard rate from per-month to per-year
    >>> convert_time_units(0.058, 'months', 'years')  # for hazard rates
    0.696
    """
    # Conversion factors to days
    unit_to_days = {
        'days': 1,
        'weeks': 7,
        'months': 30.44,  # Average month length
        'years': 365.25   # Including leap years
    }
    
    if from_unit not in unit_to_days:
        raise ValueError(f"Unknown time unit: {from_unit}")
    if to_unit not in unit_to_days:
        raise ValueError(f"Unknown time unit: {to_unit}")
    
    # Convert to days, then to target unit
    value_in_days = value * unit_to_days[from_unit]
    converted_value = value_in_days / unit_to_days[to_unit]
    
    return converted_value


def convert_survival_parameters_with_units(
    value: float,
    parameter_type: str,
    from_unit: str,
    to_unit: str,
    time_point: float = 12,
    time_point_unit: str = 'months'
) -> Dict[str, float]:
    """
    Convert survival parameters with automatic time unit conversion.
    
    Parameters
    ----------
    value : float
        Parameter value to convert
    parameter_type : str
        Type of parameter ('median', 'hazard', 'survival_fraction', 'event_rate')
    from_unit : str
        Original time unit
    to_unit : str
        Target time unit
    time_point : float, default 12
        Time point for fraction calculations
    time_point_unit : str, default 'months'
        Unit for time_point
        
    Returns
    -------
    dict
        Complete parameter set in target time units
        
    Examples
    --------
    >>> # Convert 1 year median to months-based parameters
    >>> convert_survival_parameters_with_units(1, 'median', 'years', 'months')
    {'median_survival': 12, 'hazard_rate': 0.0578, ...}
    """
    # Convert time point to target units
    converted_time_point = convert_time_units(time_point, time_point_unit, to_unit)
    
    # Handle different parameter types
    if parameter_type == 'median':
        converted_value = convert_time_units(value, from_unit, to_unit)
        return convert_survival_parameters(
            median_survival=converted_value,
            time_point=converted_time_point
        )
    
    elif parameter_type == 'hazard':
        # Hazard rates are per time unit, so conversion is inverse
        converted_value = convert_time_units(value, to_unit, from_unit)
        return convert_survival_parameters(
            hazard_rate=converted_value,
            time_point=converted_time_point
        )
    
    elif parameter_type in ['survival_fraction', 'event_rate']:
        # Fractions are unitless, but need to adjust time point
        kwargs = {parameter_type: value, 'time_point': converted_time_point}
        return convert_survival_parameters(**kwargs)
    
    else:
        raise ValueError(f"Unknown parameter type: {parameter_type}")


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def validate_survival_parameters(**kwargs) -> bool:
    """
    Validate that survival parameters are internally consistent.
    
    Returns
    -------
    bool
        True if parameters are consistent within tolerance
    """
    try:
        SurvivalParameters(**kwargs)
        return True
    except ValueError:
        return False


def survival_summary_table(
    scenarios: List[Dict],
    time_points: List[float] = [6, 12, 24, 36, 60]
) -> str:
    """
    Create a formatted summary table of survival scenarios.
    
    Parameters
    ----------
    scenarios : list
        List of scenario dictionaries with survival parameters
    time_points : list, default [6, 12, 24, 36, 60]
        Time points for survival fraction calculations
        
    Returns
    -------
    str
        Formatted table string
    """
    # Implementation would create formatted table
    # This is a placeholder for the full implementation
    return "Survival Summary Table (implementation pending)"


if __name__ == "__main__":
    # Example usage and testing
    print("=== Survival Parameter Converter Examples ===")
    
    # Example 1: Convert from median survival
    print("\n1. Convert from median survival (12 months):")
    result = convert_survival_parameters(median_survival=12)
    for key, value in result.items():
        print(f"   {key}: {value:.4f}")
    
    # Example 2: Convert from hazard ratio scenario
    print("\n2. Hazard ratio scenario (HR=0.7, control median=12 months):")
    hr_result = convert_hazard_ratio_scenario(hazard_ratio=0.7, control_median=12)
    print(f"   Control median: {hr_result['control']['median_survival']:.2f} months")
    print(f"   Treatment median: {hr_result['treatment']['median_survival']:.2f} months")
    print(f"   Hazard ratio: {hr_result['hazard_ratio']:.3f}")
    
    # Example 3: Convert from survival fraction
    print("\n3. Convert from 70% survival at 24 months:")
    sf_result = convert_survival_parameters(survival_fraction=0.7, time_point=24)
    print(f"   Median survival: {sf_result['median_survival']:.2f} months")
    print(f"   Event rate at 24 months: {sf_result['event_rate']:.1%}")