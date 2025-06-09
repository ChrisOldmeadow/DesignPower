"""
Utility functions for continuous outcome simulations in parallel group RCTs.

This module provides helper functions used by the simulation modules.
"""

import numpy as np
import math
from scipy import stats


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


def _calculate_welch_satterthwaite_df(v1, n1, v2, n2):
    """Calculate Welch-Satterthwaite degrees of freedom."""
    if n1 < 2 or n2 < 2: # Variances are undefined or df calculation breaks
        return max(1, n1 + n2 - 2) # Fallback, though NI typically has n >= 2
    
    numerator = (v1 / n1 + v2 / n2)**2
    denominator = ((v1 / n1)**2 / (n1 - 1)) + ((v2 / n2)**2 / (n2 - 1))
    if denominator == 0:
        return n1 + n2 - 2 # Avoid division by zero, fallback to pooled df
    return numerator / denominator


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