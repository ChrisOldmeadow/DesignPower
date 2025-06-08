"""
Analytical methods for continuous outcomes in parallel group randomized controlled trials.

This module provides analytical (closed-form) functions for power analysis and
sample size calculation for parallel group RCTs with continuous outcomes.
Includes standard parallel group and repeated measures designs.
"""

import numpy as np
import math
from scipy import stats

# ===== Main Functions =====

def sample_size_continuous(mean1, mean2, sd1, sd2=None, power=0.8, alpha=0.05, 
                          allocation_ratio=1.0, test="t-test"):
    """
    Calculate sample size for continuous outcome in parallel design.
    
    Parameters
    ----------
    mean1 : float
        Mean of group 1
    mean2 : float
        Mean of group 2
    sd1 : float
        Standard deviation of group 1
    sd2 : float, optional
        Standard deviation of group 2. If None, assumes equal to sd1.
    power : float, optional
        Desired statistical power, by default 0.8
    alpha : float, optional
        Significance level, by default 0.05
    allocation_ratio : float, optional
        Ratio of sample sizes (n2/n1), by default 1.0
    test : str, optional
        Type of test to use, by default "t-test"
        
    Returns
    -------
    dict
        Dictionary containing sample sizes and parameters
    """
    # Validate inputs
    if sd1 <= 0:
        raise ValueError("Standard deviation must be positive")
    
    # If sd2 not provided, assume equal to sd1
    if sd2 is None:
        sd2 = sd1
    
    if allocation_ratio <= 0:
        raise ValueError("Allocation ratio must be positive")
    
    # Calculate effect size (Cohen's d)
    effect_size = abs(mean1 - mean2) / np.sqrt((sd1**2 + sd2**2) / 2)
    
    # Use t-distribution approach with one-step refinement for more accurate results
    # when variance is unknown (which is always the case in practice)
    z_beta = stats.norm.ppf(power)  # Power calculation still uses normal
    z_alpha = stats.norm.ppf(1 - alpha/2)
    
    # Handle case where denominator is zero
    denominator = (mean1 - mean2)**2
    if denominator == 0:
        raise ValueError("Cannot calculate sample size when means are equal")
    
    # Step 1: Initial estimate using normal approximation
    numerator_normal = (z_alpha + z_beta)**2 * (sd1**2 + sd2**2 / allocation_ratio)
    n_normal = numerator_normal / denominator
    
    # Step 2: Estimate degrees of freedom and refine with t-distribution
    # For two-sample t-test: df = n1 + n2 - 2 = n1(1 + allocation_ratio) - 2
    df_estimate = n_normal * (1 + allocation_ratio) - 2
    
    # Get t critical value with estimated df
    t_alpha = stats.t.ppf(1 - alpha/2, df_estimate)
    
    # Recalculate sample size using t critical value
    numerator_t = (t_alpha + z_beta)**2 * (sd1**2 + sd2**2 / allocation_ratio)
    n_estimate = numerator_t / denominator
    
    n1 = math.ceil(n_estimate)
    n2 = math.ceil(n1 * allocation_ratio)
    
    # Return results
    return {
        "sample_size_1": n1,
        "sample_size_2": n2,
        "total_sample_size": n1 + n2,
        "mean1": mean1,
        "mean2": mean2,
        "sd1": sd1,
        "sd2": sd2,
        "effect_size": effect_size,
        "power": power,
        "alpha": alpha,
        "allocation_ratio": allocation_ratio,
        "test": test,
        "method": "analytical"
    }

def power_continuous(n1, n2, mean1, mean2, sd1, sd2=None, alpha=0.05, test="t-test"):
    """
    Calculate power for continuous outcome in parallel design.
    
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
    test : str, optional
        Type of test to use, by default "t-test"
        
    Returns
    -------
    dict
        Dictionary containing power and parameters
    """
    # Validate inputs
    if sd1 <= 0:
        raise ValueError("Standard deviation must be positive")
    
    # If sd2 not provided, assume equal to sd1
    if sd2 is None:
        sd2 = sd1
    
    # Calculate the standardized effect (Cohen's d)
    effect_size = abs(mean1 - mean2) / np.sqrt((sd1**2 + sd2**2) / 2)
    
    # Calculate standard error of the difference
    se = np.sqrt(sd1**2 / n1 + sd2**2 / n2)
    
    # Calculate non-centrality parameter
    ncp = abs(mean1 - mean2) / se
    
    # Critical value
    cv = stats.norm.ppf(1 - alpha/2)
    
    # Calculate power
    power = 1 - stats.norm.cdf(cv - ncp) + stats.norm.cdf(-cv - ncp)
    
    # Return results
    return {
        "power": power,
        "mean1": mean1,
        "mean2": mean2,
        "sd1": sd1,
        "sd2": sd2,
        "n1": n1,
        "n2": n2,
        "effect_size": effect_size,
        "alpha": alpha,
        "test": test,
        "method": "analytical"
    }

def min_detectable_effect_continuous(n1, n2, sd1, sd2=None, power=0.8, alpha=0.05):
    """
    Calculate minimum detectable effect for continuous outcome in parallel design.
    
    Parameters
    ----------
    n1 : int
        Sample size of group 1
    n2 : int
        Sample size of group 2
    sd1 : float
        Standard deviation of group 1
    sd2 : float, optional
        Standard deviation of group 2. If None, assumes equal to sd1.
    power : float, optional
        Desired statistical power, by default 0.8
    alpha : float, optional
        Significance level, by default 0.05
        
    Returns
    -------
    dict
        Dictionary containing minimum detectable effect and parameters
    """
    # Validate inputs
    if sd1 <= 0:
        raise ValueError("Standard deviation must be positive")
    
    # If sd2 not provided, assume equal to sd1
    if sd2 is None:
        sd2 = sd1
    
    # Get critical values
    z_alpha = stats.norm.ppf(1 - alpha/2)
    z_beta = stats.norm.ppf(power)
    
    # Calculate standard error of the difference
    se = np.sqrt(sd1**2 / n1 + sd2**2 / n2)
    
    # Calculate minimum detectable difference
    mde = (z_alpha + z_beta) * se
    
    # Calculate standardized effect size (Cohen's d)
    cohen_d = mde / np.sqrt((sd1**2 + sd2**2) / 2)
    
    # Return results
    return {
        "minimum_detectable_effect": mde,
        "standardized_effect": cohen_d,
        "sd1": sd1,
        "sd2": sd2,
        "n1": n1,
        "n2": n2,
        "power": power,
        "alpha": alpha,
        "method": "analytical"
    }

def sample_size_continuous_non_inferiority(mean1, non_inferiority_margin, sd1, sd2=None, 
                                          power=0.8, alpha=0.05, allocation_ratio=1.0, 
                                          assumed_difference=0.0, direction="lower"):
    """
    Calculate sample size for non-inferiority test with continuous outcome.
    
    Parameters
    ----------
    mean1 : float
        Mean of the control/standard group
    non_inferiority_margin : float
        Non-inferiority margin (must be positive)
    sd1 : float
        Standard deviation of group 1
    sd2 : float, optional
        Standard deviation of group 2. If None, assumes equal to sd1.
    power : float, optional
        Desired power, by default 0.8
    alpha : float, optional
        Significance level (one-sided for non-inferiority), by default 0.05
    allocation_ratio : float, optional
        Ratio of sample sizes (n2/n1), by default 1.0
    assumed_difference : float, optional
        Assumed true difference between treatments (0 = treatments truly equivalent), by default 0.0
    direction : str, optional
        Direction of non-inferiority test ("lower" or "upper"), by default "lower"
    
    Returns
    -------
    dict
        Dictionary containing the required sample sizes
    """
    # Validate inputs
    if sd1 <= 0:
        raise ValueError("Standard deviation must be positive")
    
    if non_inferiority_margin <= 0:
        raise ValueError("Non-inferiority margin must be positive")
    
    # If sd2 not provided, assume equal to sd1
    if sd2 is None:
        sd2 = sd1
    
    # Calculate mean2 based on assumed difference
    mean2 = mean1 + assumed_difference
    
    # Get critical values (one-sided alpha for non-inferiority)
    z_alpha = stats.norm.ppf(1 - alpha)
    z_beta = stats.norm.ppf(power)
    
    # Calculate effect for testing based on direction
    if direction == "lower":
        # For lower non-inferiority, we're testing mean2 >= mean1 - margin
        effect = (mean2 - (mean1 - non_inferiority_margin))
    else:  # upper
        # For upper non-inferiority, we're testing mean2 <= mean1 + margin
        effect = ((mean1 + non_inferiority_margin) - mean2)
    
    # Calculate sample size
    numerator = (z_alpha + z_beta)**2 * (sd1**2 + sd2**2 / allocation_ratio)
    denominator = effect**2
    
    # Handle case where denominator is zero
    if denominator == 0:
        raise ValueError("Cannot calculate sample size when effect is zero")
    
    n1 = math.ceil(numerator / denominator)
    n2 = math.ceil(n1 * allocation_ratio)
    
    # Return results
    return {
        "sample_size_1": n1,
        "sample_size_2": n2,
        "total_sample_size": n1 + n2,
        "mean1": mean1,
        "mean2": mean2,
        "sd1": sd1,
        "sd2": sd2,
        "non_inferiority_margin": non_inferiority_margin,
        "assumed_difference": assumed_difference,
        "direction": direction,
        "power": power,
        "alpha": alpha,
        "allocation_ratio": allocation_ratio,
        "method": "analytical"
    }

def power_continuous_non_inferiority(n1, n2, mean1, non_inferiority_margin, sd1, sd2=None, 
                                    alpha=0.05, assumed_difference=0.0, direction="lower"):
    """
    Calculate power for non-inferiority test with continuous outcome.
    
    Parameters
    ----------
    n1 : int
        Sample size in group 1 (standard treatment)
    n2 : int
        Sample size in group 2 (new treatment)
    mean1 : float
        Mean of the control/standard group
    non_inferiority_margin : float
        Non-inferiority margin (must be positive)
    sd1 : float
        Standard deviation of group 1
    sd2 : float, optional
        Standard deviation of group 2. If None, assumes equal to sd1.
    alpha : float, optional
        Significance level (one-sided for non-inferiority), by default 0.05
    assumed_difference : float, optional
        Assumed true difference between treatments (0 = treatments truly equivalent), by default 0.0
    direction : str, optional
        Direction of non-inferiority test ("lower" or "upper"), by default "lower"
    
    Returns
    -------
    dict
        Dictionary containing the power estimate and parameters
    """
    # Validate inputs
    if sd1 <= 0:
        raise ValueError("Standard deviation must be positive")
    
    if non_inferiority_margin <= 0:
        raise ValueError("Non-inferiority margin must be positive")
    
    # If sd2 not provided, assume equal to sd1
    if sd2 is None:
        sd2 = sd1
    
    # Calculate mean2 from assumed difference
    mean2 = mean1 + assumed_difference
    
    # Calculate standard error of the difference
    se = np.sqrt(sd1**2 / n1 + sd2**2 / n2)
    
    # Calculate critical value (one-sided alpha for non-inferiority)
    z_alpha = stats.norm.ppf(1 - alpha)
    
    # Calculate test statistic and power based on direction
    if direction == "lower":
        # For lower non-inferiority, we're testing mean2 >= mean1 - margin
        # Null: mean2 <= mean1 - margin, Alt: mean2 > mean1 - margin
        z_stat = (mean2 - (mean1 - non_inferiority_margin)) / se
        power = 1 - stats.norm.cdf(z_alpha - z_stat)
    else:  # upper
        # For upper non-inferiority, we're testing mean2 <= mean1 + margin
        # Null: mean2 >= mean1 + margin, Alt: mean2 < mean1 + margin
        z_stat = ((mean1 + non_inferiority_margin) - mean2) / se
        power = 1 - stats.norm.cdf(z_alpha - z_stat)
    
    # Return results
    return {
        "power": power,
        "mean1": mean1,
        "mean2": mean2,
        "sd1": sd1,
        "sd2": sd2,
        "n1": n1,
        "n2": n2,
        "non_inferiority_margin": non_inferiority_margin,
        "assumed_difference": assumed_difference,
        "direction": direction,
        "alpha": alpha,
        "method": "analytical"
    }


# ===== Repeated Measures Analytical Functions =====

def sample_size_repeated_measures(mean1, mean2, sd1, correlation, sd2=None, power=0.8, alpha=0.05, 
                                allocation_ratio=1.0, method="change_score"):
    """
    Calculate sample size for detecting a difference in means with repeated measures.
    
    Parameters
    ----------
    mean1 : float
        Mean of group 1
    mean2 : float
        Mean of group 2
    sd1 : float
        Standard deviation of group 1
    correlation : float
        Correlation between baseline and follow-up measurements
    sd2 : float, optional
        Standard deviation of group 2. If None, assumes equal to sd1.
    power : float, optional
        Desired statistical power (1 - beta), by default 0.8
    alpha : float, optional
        Significance level, by default 0.05
    allocation_ratio : float, optional
        Ratio of sample sizes (n2/n1), by default 1.0
    method : str, optional
        Analysis method: "change_score" or "ancova", by default "change_score"
    
    Returns
    -------
    dict
        Dictionary containing the required sample sizes for each group and total
    """
    # Calculate the delta (difference in means)
    delta = abs(mean2 - mean1)
    
    # If sd2 not provided, assume equal to sd1
    if sd2 is None:
        sd2 = sd1
    
    # Calculate z-scores for given alpha and power
    z_alpha = stats.norm.ppf(1 - alpha/2)
    z_beta = stats.norm.ppf(power)
    
    # Calculate effective standard deviation based on method
    if method == "change_score":
        # For change score analysis
        std_dev_eff = sd1 * math.sqrt(2 * (1 - correlation))
    else:  # ancova
        # For ANCOVA (more efficient than change score when correlation > 0.5)
        std_dev_eff = sd1 * math.sqrt(1 - correlation**2)
    
    # Calculate sample size for group 1
    n1 = ((1 + 1/allocation_ratio) * (std_dev_eff**2) * (z_alpha + z_beta)**2) / (delta**2)
    n1 = math.ceil(n1)
    
    # Calculate sample size for group 2
    n2 = math.ceil(n1 * allocation_ratio)
    
    return {
        "n1": n1,
        "n2": n2,
        "total_n": n1 + n2,
        "parameters": {
            "delta": delta,
            "mean1": mean1,
            "mean2": mean2,
            "sd1": sd1,
            "sd2": sd2,
            "correlation": correlation,
            "power": power,
            "alpha": alpha,
            "allocation_ratio": allocation_ratio,
            "method": method
        }
    }


def power_repeated_measures(n1, n2, mean1, mean2, sd1, correlation, sd2=None, alpha=0.05, method="change_score"):
    """
    Calculate power for detecting a difference in means with repeated measures.
    
    Parameters
    ----------
    n1 : int
        Sample size for group 1
    n2 : int
        Sample size for group 2
    mean1 : float
        Mean of group 1
    mean2 : float
        Mean of group 2
    sd1 : float
        Standard deviation of group 1
    correlation : float
        Correlation between baseline and follow-up measurements
    sd2 : float, optional
        Standard deviation of group 2. If None, assumes equal to sd1.
    alpha : float, optional
        Significance level, by default 0.05
    method : str, optional
        Analysis method: "change_score" or "ancova", by default "change_score"
    
    Returns
    -------
    dict
        Dictionary containing the calculated power and input parameters
    """
    # Calculate the delta (difference in means)
    delta = abs(mean2 - mean1)
    
    # If sd2 not provided, assume equal to sd1
    if sd2 is None:
        sd2 = sd1
    
    # Calculate z-score for given alpha
    z_alpha = stats.norm.ppf(1 - alpha/2)
    
    # Calculate effective standard deviation based on method
    if method == "change_score":
        # For change score analysis
        std_dev_eff = sd1 * math.sqrt(2 * (1 - correlation))
    else:  # ancova
        # For ANCOVA
        std_dev_eff = sd1 * math.sqrt(1 - correlation**2)
    
    # Calculate non-centrality parameter
    ncp = delta / (std_dev_eff * math.sqrt(1/n1 + 1/n2))
    
    # Calculate power
    power = stats.norm.cdf(ncp - z_alpha) + stats.norm.cdf(-ncp - z_alpha)
    
    return {
        "power": power,
        "parameters": {
            "n1": n1,
            "n2": n2,
            "delta": delta,
            "mean1": mean1,
            "mean2": mean2,
            "sd1": sd1,
            "sd2": sd2,
            "correlation": correlation,
            "alpha": alpha,
            "method": method
        }
    }


def min_detectable_effect_repeated_measures(n1, n2, sd1, correlation, sd2=None, power=0.8, 
                                         alpha=0.05, method="change_score"):
    """
    Calculate minimum detectable effect for repeated measures design.
    
    Parameters
    ----------
    n1 : int
        Sample size for group 1
    n2 : int
        Sample size for group 2
    sd1 : float
        Standard deviation of group 1
    correlation : float
        Correlation between baseline and follow-up measurements
    sd2 : float, optional
        Standard deviation of group 2. If None, assumes equal to sd1.
    power : float, optional
        Desired statistical power (1 - beta), by default 0.8
    alpha : float, optional
        Significance level, by default 0.05
    method : str, optional
        Analysis method: "change_score" or "ancova", by default "change_score"
    
    Returns
    -------
    dict
        Dictionary containing the minimum detectable effect and input parameters
    """
    # If sd2 not provided, assume equal to sd1
    if sd2 is None:
        sd2 = sd1
    
    # Calculate z-scores for given alpha and power
    z_alpha = stats.norm.ppf(1 - alpha/2)
    z_beta = stats.norm.ppf(power)
    
    # Calculate effective standard deviation based on method
    if method == "change_score":
        # For change score analysis
        std_dev_eff = sd1 * math.sqrt(2 * (1 - correlation))
    else:  # ancova
        # For ANCOVA
        std_dev_eff = sd1 * math.sqrt(1 - correlation**2)
    
    # Calculate minimum detectable effect
    delta = (z_alpha + z_beta) * std_dev_eff * math.sqrt(1/n1 + 1/n2)
    
    return {
        "delta": delta,
        "parameters": {
            "n1": n1,
            "n2": n2,
            "sd1": sd1,
            "sd2": sd2,
            "correlation": correlation,
            "power": power,
            "alpha": alpha,
            "method": method
        }
    }
