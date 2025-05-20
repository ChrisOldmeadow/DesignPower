"""
Continuous outcome functions for single-arm (one-sample) designs.

This module provides functions for power analysis and sample size calculation
for single-arm studies with continuous outcomes.
"""

import math
import numpy as np
from scipy import stats


def one_sample_t_test_sample_size(mean_null, mean_alt, std_dev, alpha=0.05, power=0.8, sides=2):
    """
    Calculate sample size for one-sample t-test (continuous outcome).
    
    Parameters
    ----------
    mean_null : float
        Null hypothesis mean
    mean_alt : float
        Alternative hypothesis mean
    std_dev : float
        Standard deviation
    alpha : float, optional
        Significance level, by default 0.05
    power : float, optional
        Desired power (1 - beta), by default 0.8
    sides : int, optional
        One-sided or two-sided test (1 or 2), by default 2
    
    Returns
    -------
    int
        Required sample size (rounded up)
    """
    # Effect size (standardized mean difference)
    effect_size = abs(mean_alt - mean_null) / std_dev
    
    # Calculate t-test critical values based on sides
    if sides == 1:
        t_alpha = stats.t.ppf(1 - alpha, np.inf)
    else:  # sides == 2
        t_alpha = stats.t.ppf(1 - alpha/2, np.inf)
        
    t_beta = stats.t.ppf(power, np.inf)
    
    # Calculate sample size
    n = ((t_alpha + t_beta) / effect_size) ** 2
    
    # Round up to nearest whole number
    return math.ceil(n)


def one_sample_t_test_power(n, mean_null, mean_alt, std_dev, alpha=0.05, sides=2):
    """
    Calculate power for one-sample t-test.
    
    Parameters
    ----------
    n : int
        Sample size
    mean_null : float
        Null hypothesis mean
    mean_alt : float
        Alternative hypothesis mean
    std_dev : float
        Standard deviation
    alpha : float, optional
        Significance level, by default 0.05
    sides : int, optional
        One-sided or two-sided test (1 or 2), by default 2
    
    Returns
    -------
    float
        Power (1 - beta)
    """
    # Effect size (standardized mean difference)
    effect_size = abs(mean_alt - mean_null) / std_dev
    
    # Standard error
    se = std_dev / math.sqrt(n)
    
    # Non-centrality parameter
    ncp = effect_size * math.sqrt(n)
    
    # Critical values
    if sides == 1:
        t_alpha = stats.t.ppf(1 - alpha, n - 1)
    else:  # sides == 2
        t_alpha = stats.t.ppf(1 - alpha/2, n - 1)
    
    # Calculate power
    power = 1 - stats.nct.cdf(t_alpha, n - 1, ncp)
    
    return power


def min_detectable_effect_one_sample_continuous(n, mean_null=0, std_dev=1, power=0.8, alpha=0.05, sides=2):
    """
    Calculate minimum detectable effect for one-sample t-test.
    
    Parameters
    ----------
    n : int
        Sample size
    mean_null : float, optional
        Null hypothesis mean, by default 0
    std_dev : float, optional
        Standard deviation, by default 1
    power : float, optional
        Desired power (1 - beta), by default 0.8
    alpha : float, optional
        Significance level, by default 0.05
    sides : int, optional
        One-sided or two-sided test (1 or 2), by default 2
    
    Returns
    -------
    float
        Minimum detectable effect (standardized mean difference)
    """
    # Calculate critical values
    if sides == 1:
        t_alpha = stats.t.ppf(1 - alpha, n - 1)
    else:  # sides == 2
        t_alpha = stats.t.ppf(1 - alpha/2, n - 1)
    
    t_beta = stats.t.ppf(power, n - 1)
    
    # Calculate MDE
    mde = (t_alpha + t_beta) * std_dev / math.sqrt(n)
    
    # Return MDE as absolute mean difference
    return mde


def simulate_one_sample_continuous_trial(n, mean_null, mean_alt, std_dev, nsim=1000, alpha=0.05, sides=2, seed=None):
    """
    Simulate a single-arm study with continuous outcome.
    
    Parameters
    ----------
    n : int
        Sample size
    mean_null : float
        Null hypothesis mean
    mean_alt : float
        Alternative hypothesis mean (true mean)
    std_dev : float
        Standard deviation
    nsim : int, optional
        Number of simulations, by default 1000
    alpha : float, optional
        Significance level, by default 0.05
    sides : int, optional
        One-sided or two-sided test (1 or 2), by default 2
    seed : int, optional
        Random seed for reproducibility, by default None
    
    Returns
    -------
    dict
        Dictionary containing simulation results, including empirical power
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Initialize counters
    significant_results = 0
    t_values = []
    p_values = []
    
    # Run simulations
    for _ in range(nsim):
        # Generate data from alternative distribution
        data = np.random.normal(mean_alt, std_dev, n)
        
        # Perform one-sample t-test against null hypothesis
        t_stat, p_value = stats.ttest_1samp(data, mean_null)
        t_values.append(t_stat)
        
        # For one-sided tests, convert two-sided p-value
        if sides == 1:
            # Determine direction of test
            if mean_alt > mean_null:  # Upper-tailed
                p_value = p_value / 2 if t_stat > 0 else 1 - p_value / 2
            else:  # Lower-tailed
                p_value = p_value / 2 if t_stat < 0 else 1 - p_value / 2
        
        p_values.append(p_value)
        
        # Count significant results
        if p_value < alpha:
            significant_results += 1
    
    # Calculate empirical power
    power = significant_results / nsim
    
    return {
        "power": power,
        "significant_results": significant_results,
        "nsim": nsim,
        "n": n,
        "mean_null": mean_null,
        "mean_alt": mean_alt,
        "std_dev": std_dev,
        "sides": sides,
        "average_t": np.mean(t_values),
        "average_p": np.mean(p_values)
    }
