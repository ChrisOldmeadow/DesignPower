"""
Binary outcome functions for single-arm (one-sample) designs.

This module provides functions for power analysis and sample size calculation
for single-arm studies with binary outcomes.
"""

import math
import numpy as np
from scipy import stats


def one_sample_proportion_test_sample_size(p0, p1, alpha=0.05, power=0.8, sides=2):
    """
    Calculate sample size for one-sample proportion test (binary outcome).
    
    Parameters
    ----------
    p0 : float
        Null hypothesis proportion (between 0 and 1)
    p1 : float
        Alternative hypothesis proportion (between 0 and 1)
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
    # Calculate z-scores based on sides
    if sides == 1:
        z_alpha = stats.norm.ppf(1 - alpha)
    else:  # sides == 2
        z_alpha = stats.norm.ppf(1 - alpha/2)
    
    z_beta = stats.norm.ppf(power)
    
    # Calculate sample size using normal approximation
    term1 = z_alpha * math.sqrt(p0 * (1 - p0))
    term2 = z_beta * math.sqrt(p1 * (1 - p1))
    denominator = p1 - p0
    
    # Handle small differences to prevent division by zero
    if abs(denominator) < 1e-10:
        return 10000  # Default to large sample size for very small differences
    
    n = ((term1 + term2) / denominator) ** 2
    
    # Round up to nearest whole number
    return math.ceil(n)


def one_sample_proportion_test_power(n, p0, p1, alpha=0.05, sides=2):
    """
    Calculate power for one-sample proportion test.
    
    Parameters
    ----------
    n : int
        Sample size
    p0 : float
        Null hypothesis proportion
    p1 : float
        Alternative hypothesis proportion
    alpha : float, optional
        Significance level, by default 0.05
    sides : int, optional
        One-sided or two-sided test (1 or 2), by default 2
    
    Returns
    -------
    float
        Power (1 - beta)
    """
    # Calculate z-score for alpha
    if sides == 1:
        z_alpha = stats.norm.ppf(1 - alpha)
    else:  # sides == 2
        z_alpha = stats.norm.ppf(1 - alpha/2)
    
    # Calculate non-centrality parameter
    denominator = math.sqrt(p0 * (1 - p0) / n)
    
    # Handle potential division by zero
    if denominator < 1e-10:
        return 0.999  # Default to high power when denominator is very small
    
    ncp = (p1 - p0) / denominator
    
    # Calculate power
    power = 1 - stats.norm.cdf(z_alpha - ncp)
    
    return power


def min_detectable_effect_one_sample_binary(n, p0=0.5, power=0.8, alpha=0.05, sides=2):
    """
    Calculate minimum detectable effect for one-sample proportion test.
    
    Parameters
    ----------
    n : int
        Sample size
    p0 : float, optional
        Null hypothesis proportion, by default 0.5
    power : float, optional
        Desired power (1 - beta), by default 0.8
    alpha : float, optional
        Significance level, by default 0.05
    sides : int, optional
        One-sided or two-sided test (1 or 2), by default 2
    
    Returns
    -------
    float
        Minimum detectable difference in proportions
    """
    # Calculate critical values
    if sides == 1:
        z_alpha = stats.norm.ppf(1 - alpha)
    else:  # sides == 2
        z_alpha = stats.norm.ppf(1 - alpha/2)
    
    z_beta = stats.norm.ppf(power)
    
    # Calculate standard error for null proportion
    se = math.sqrt(p0 * (1 - p0) / n)
    
    # Calculate minimum detectable effect (absolute difference)
    mde = (z_alpha + z_beta) * se
    
    return mde


def simulate_one_sample_binary_trial(n, p0, p1, nsim=1000, alpha=0.05, sides=2, seed=None):
    """
    Simulate a single-arm study with binary outcome.
    
    Parameters
    ----------
    n : int
        Sample size
    p0 : float
        Null hypothesis proportion
    p1 : float
        Alternative hypothesis proportion
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
    
    # Run simulations
    for _ in range(nsim):
        # Generate data
        data = np.random.binomial(1, p1, n)
        p_hat = np.mean(data)
        
        # Calculate standard error under null hypothesis
        se = math.sqrt(p0 * (1 - p0) / n)
        
        # Calculate test statistic
        z = (p_hat - p0) / se
        
        # Calculate p-value based on sides
        if sides == 1:
            # One-sided test
            if p1 > p0:  # Upper-tailed
                p_value = 1 - stats.norm.cdf(z)
            else:  # Lower-tailed
                p_value = stats.norm.cdf(z)
        else:  # sides == 2
            # Two-sided test
            p_value = 2 * (1 - stats.norm.cdf(abs(z)))
        
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
        "p0": p0,
        "p1": p1,
        "sides": sides
    }


def ahern_sample_size(p0, p1, alpha=0.05, beta=0.2):
    """
    Calculate the sample size and rejection threshold for A'Hern's design.
    
    A'Hern's design is based on exact binomial probabilities rather than
    normal approximations, making it more suitable for small sample sizes
    typical in phase II trials.
    
    Reference: A'Hern, R. P. (2001). Sample size tables for exact single-stage phase II designs.
    Statistics in Medicine, 20(6), 859-866.
    
    Parameters
    ----------
    p0 : float
        Probability of response under the null hypothesis (unacceptable response rate)
    p1 : float
        Probability of response under the alternative hypothesis (desirable response rate)
    alpha : float, optional
        Type I error rate (probability of falsely rejecting H0), by default 0.05
    beta : float, optional
        Type II error rate (probability of falsely accepting H0), by default 0.2
        Note: power = 1 - beta
    
    Returns
    -------
    dict
        A dictionary containing:
        - n: required sample size
        - r: minimum number of responses to reject the null hypothesis
        - p0: null hypothesis response rate
        - p1: alternative hypothesis response rate
        - alpha: type I error rate
        - beta: type II error rate
        - power: power of the test (1 - beta)
        - actual_alpha: actual type I error rate (may differ from requested alpha)
        - actual_beta: actual type II error rate (may differ from requested beta)
    """
    # Validate inputs
    if not (0 < p0 < 1):
        raise ValueError("p0 must be between 0 and 1")
    if not (0 < p1 < 1):
        raise ValueError("p1 must be between 0 and 1")
    if p0 >= p1:
        raise ValueError("p1 must be greater than p0 for this design")
    if not (0 < alpha < 1):
        raise ValueError("alpha must be between 0 and 1")
    if not (0 < beta < 1):
        raise ValueError("beta must be between 0 and 1")
    
    # Initial search parameters
    power = 1 - beta
    max_n = 200  # Maximum sample size to consider
    
    best_n = None
    best_r = None
    best_alpha_diff = float('inf')
    best_actual_alpha = None
    best_actual_beta = None
    
    # Search for the smallest sample size that satisfies both alpha and beta constraints
    for n in range(5, max_n + 1):
        for r in range(0, n + 1):
            # Calculate actual type I error rate - probability of r or more successes under H0
            actual_alpha = 1 - stats.binom.cdf(r - 1, n, p0)
            
            # Calculate actual type II error rate - probability of fewer than r successes under H1
            actual_beta = stats.binom.cdf(r - 1, n, p1)
            
            # Check if this combination satisfies our constraints
            if actual_alpha <= alpha and actual_beta <= beta:
                alpha_diff = abs(actual_alpha - alpha)
                
                # Update best result if this is better than what we've found so far
                if best_n is None or n < best_n or (n == best_n and alpha_diff < best_alpha_diff):
                    best_n = n
                    best_r = r
                    best_alpha_diff = alpha_diff
                    best_actual_alpha = actual_alpha
                    best_actual_beta = actual_beta
                    
                # Break inner loop once we've found a valid r for this n
                break
    
    if best_n is None:
        raise ValueError(f"No solution found within sample size limit of {max_n}. Try relaxing alpha or beta.")
    
    return {
        "n": best_n,
        "r": best_r,
        "p0": p0,
        "p1": p1,
        "alpha": alpha,
        "beta": beta,
        "power": power,
        "actual_alpha": best_actual_alpha,
        "actual_beta": best_actual_beta,
        "actual_power": 1 - best_actual_beta
    }


def ahern_power(n, r, p0, p1):
    """
    Calculate power for A'Hern's design with given parameters.
    
    Parameters
    ----------
    n : int
        Sample size
    r : int
        Minimum number of responses required to reject null hypothesis
    p0 : float
        Probability of response under the null hypothesis
    p1 : float
        Probability of response under the alternative hypothesis
    
    Returns
    -------
    dict
        A dictionary containing:
        - power: power of the test
        - actual_alpha: actual type I error rate
        - actual_beta: actual type II error rate
    """
    # Validate inputs
    if n <= 0 or not isinstance(n, int):
        raise ValueError("n must be a positive integer")
    if r < 0 or r > n or not isinstance(r, int):
        raise ValueError("r must be a non-negative integer less than or equal to n")
    if not (0 < p0 < 1):
        raise ValueError("p0 must be between 0 and 1")
    if not (0 < p1 < 1):
        raise ValueError("p1 must be between 0 and 1")
    
    # Calculate actual type I error rate - probability of r or more successes under H0
    actual_alpha = 1 - stats.binom.cdf(r - 1, n, p0)
    
    # Calculate actual type II error rate - probability of fewer than r successes under H1
    actual_beta = stats.binom.cdf(r - 1, n, p1)
    
    # Calculate power
    actual_power = 1 - actual_beta
    
    return {
        "power": actual_power,
        "actual_alpha": actual_alpha,
        "actual_beta": actual_beta
    }
