"""
Analytical methods for repeated measures designs in parallel group RCTs.

This module provides functions for sample size, power calculation, and
minimum detectable effect estimation for repeated measures designs.
"""
import math
from scipy import stats


def sample_size_repeated_measures(delta, std_dev, correlation, power=0.8, alpha=0.05, 
                               allocation_ratio=1.0, method="change_score"):
    """
    Calculate sample size for detecting a difference in means with repeated measures.
    
    Parameters
    ----------
    delta : float
        Minimum detectable effect (difference between means)
    std_dev : float
        Standard deviation of the outcome
    correlation : float
        Correlation between baseline and follow-up measurements
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
    # Calculate z-scores for given alpha and power
    z_alpha = stats.norm.ppf(1 - alpha/2)
    z_beta = stats.norm.ppf(power)
    
    # Calculate effective standard deviation based on method
    if method == "change_score":
        # For change score analysis
        std_dev_eff = std_dev * math.sqrt(2 * (1 - correlation))
    else:  # ancova
        # For ANCOVA (more efficient than change score when correlation > 0.5)
        std_dev_eff = std_dev * math.sqrt(1 - correlation**2)
    
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
            "std_dev": std_dev,
            "correlation": correlation,
            "power": power,
            "alpha": alpha,
            "allocation_ratio": allocation_ratio,
            "method": method
        }
    }


def power_repeated_measures(n1, n2, delta, std_dev, correlation, alpha=0.05, method="change_score"):
    """
    Calculate power for detecting a difference in means with repeated measures.
    
    Parameters
    ----------
    n1 : int
        Sample size for group 1
    n2 : int
        Sample size for group 2
    delta : float
        Minimum detectable effect (difference between means)
    std_dev : float
        Standard deviation of the outcome
    correlation : float
        Correlation between baseline and follow-up measurements
    alpha : float, optional
        Significance level, by default 0.05
    method : str, optional
        Analysis method: "change_score" or "ancova", by default "change_score"
    
    Returns
    -------
    dict
        Dictionary containing the calculated power and input parameters
    """
    # Calculate z-score for given alpha
    z_alpha = stats.norm.ppf(1 - alpha/2)
    
    # Calculate effective standard deviation based on method
    if method == "change_score":
        # For change score analysis
        std_dev_eff = std_dev * math.sqrt(2 * (1 - correlation))
    else:  # ancova
        # For ANCOVA
        std_dev_eff = std_dev * math.sqrt(1 - correlation**2)
    
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
            "std_dev": std_dev,
            "correlation": correlation,
            "alpha": alpha,
            "method": method
        }
    }


def min_detectable_effect_repeated_measures(n1, n2, std_dev, correlation, power=0.8, 
                                        alpha=0.05, method="change_score"):
    """
    Calculate minimum detectable effect for repeated measures design.
    
    Parameters
    ----------
    n1 : int
        Sample size for group 1
    n2 : int
        Sample size for group 2
    std_dev : float
        Standard deviation of the outcome
    correlation : float
        Correlation between baseline and follow-up measurements
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
    # Calculate z-scores for given alpha and power
    z_alpha = stats.norm.ppf(1 - alpha/2)
    z_beta = stats.norm.ppf(power)
    
    # Calculate effective standard deviation based on method
    if method == "change_score":
        # For change score analysis
        std_dev_eff = std_dev * math.sqrt(2 * (1 - correlation))
    else:  # ancova
        # For ANCOVA
        std_dev_eff = std_dev * math.sqrt(1 - correlation**2)
    
    # Calculate minimum detectable effect
    delta = (z_alpha + z_beta) * std_dev_eff * math.sqrt(1/n1 + 1/n2)
    
    return {
        "delta": delta,
        "parameters": {
            "n1": n1,
            "n2": n2,
            "std_dev": std_dev,
            "correlation": correlation,
            "power": power,
            "alpha": alpha,
            "method": method
        }
    }
