"""
Binary outcome functions for parallel group randomized controlled trials.

This module provides comprehensive functions for power analysis and
sample size calculation for parallel group RCTs with binary outcomes.
Includes both analytical and simulation-based approaches.
"""

import numpy as np
import math
from scipy import stats
from scipy.stats import fisher_exact

# Import from other modules for backwards compatibility during transition
from .analytical import sample_size_binary as analytical_sample_size_binary
from .analytical import power_binary as analytical_power_binary
from .analytical import sample_size_binary_non_inferiority as analytical_sample_size_binary_non_inferiority

# ===== Statistical Test Functions =====

def normal_approximation_test(n1, n2, s1, s2):
    """
    Perform a normal approximation test (z-test) for difference in proportions.
    
    Parameters
    ----------
    n1 : int
        Sample size for group 1
    n2 : int
        Sample size for group 2
    s1 : int
        Number of successes in group 1
    s2 : int
        Number of successes in group 2
    
    Returns
    -------
    float
        Two-sided p-value
    """
    # Calculate proportions
    p1 = s1 / n1 if n1 > 0 else 0
    p2 = s2 / n2 if n2 > 0 else 0
    
    # Calculate pooled proportion
    p_pooled = (s1 + s2) / (n1 + n2) if (n1 + n2) > 0 else 0
    
    # Calculate standard error
    se = math.sqrt(p_pooled * (1 - p_pooled) * (1/n1 + 1/n2))
    
    # Handle edge case where standard error is 0
    if se == 0:
        return 1.0  # No difference detected
    
    # Calculate z-statistic
    z = (p1 - p2) / se
    
    # Calculate two-sided p-value
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    
    return p_value


def likelihood_ratio_test(n1, n2, s1, s2):
    """
    Perform a likelihood ratio test for difference in proportions.
    
    Parameters
    ----------
    n1 : int
        Sample size for group 1
    n2 : int
        Sample size for group 2
    s1 : int
        Number of successes in group 1
    s2 : int
        Number of successes in group 2
    
    Returns
    -------
    float
        Two-sided p-value
    """
    # Calculate proportions
    p1 = s1 / n1 if n1 > 0 else 0
    p2 = s2 / n2 if n2 > 0 else 0
    
    # Calculate pooled proportion (under null hypothesis)
    p_pooled = (s1 + s2) / (n1 + n2) if (n1 + n2) > 0 else 0
    
    # Log-likelihood under null hypothesis
    if p_pooled == 0 or p_pooled == 1:
        # Handle edge case to avoid log(0)
        ll_null = 0
    else:
        ll_null = (s1 + s2) * math.log(p_pooled) + (n1 + n2 - s1 - s2) * math.log(1 - p_pooled)
    
    # Log-likelihood under alternative hypothesis
    ll_alt = 0
    if p1 > 0 and p1 < 1:
        ll_alt += s1 * math.log(p1) + (n1 - s1) * math.log(1 - p1)
    if p2 > 0 and p2 < 1:
        ll_alt += s2 * math.log(p2) + (n2 - s2) * math.log(1 - p2)
    
    # Likelihood ratio test statistic
    lr = 2 * (ll_alt - ll_null)
    
    # LR test statistic follows chi-squared with 1 df
    p_value = 1 - stats.chi2.cdf(lr, 1)
    
    return p_value


def fishers_exact_test(n1, n2, s1, s2):
    """
    Perform Fisher's exact test for difference in proportions.
    
    Parameters
    ----------
    n1 : int
        Sample size for group 1
    n2 : int
        Sample size for group 2
    s1 : int
        Number of successes in group 1
    s2 : int
        Number of successes in group 2
    
    Returns
    -------
    float
        Two-sided p-value
    """
    # Create contingency table
    table = np.array([[s1, n1 - s1], [s2, n2 - s2]])
    
    # Perform Fisher's exact test
    _, p_value = fisher_exact(table)
    
    return p_value


def perform_binary_test(n1, n2, s1, s2, test_type="normal_approximation"):
    """
    Perform the appropriate statistical test for binary outcomes based on test_type.
    
    Parameters
    ----------
    n1 : int
        Sample size for group 1
    n2 : int
        Sample size for group 2
    s1 : int
        Number of successes in group 1
    s2 : int
        Number of successes in group 2
    test_type : str, optional
        Type of test to use, by default "normal_approximation"
        Options: "normal_approximation", "likelihood_ratio", "fishers_exact"
    
    Returns
    -------
    float
        Two-sided p-value from the specified test
    """
    test_mapping = {
        "normal_approximation": normal_approximation_test,
        "likelihood_ratio": likelihood_ratio_test,
        "fishers_exact": fishers_exact_test
    }
    
    # Default to normal approximation if the test_type is not recognized
    test_function = test_mapping.get(test_type.lower(), normal_approximation_test)
    
    return test_function(n1, n2, s1, s2)


# ===== Power and Sample Size Functions =====

def power_binary(n1, n2, p1, p2, alpha=0.05, test_type="normal_approximation"):
    """
    Calculate power for binary outcomes using analytical or simulation approaches.
    
    Parameters
    ----------
    n1 : int
        Sample size for group 1
    n2 : int
        Sample size for group 2
    p1 : float
        Proportion in group 1
    p2 : float
        Proportion in group 2
    alpha : float, optional
        Significance level, by default 0.05
    test_type : str, optional
        Type of test to use, by default "normal_approximation"
    
    Returns
    -------
    dict
        Dictionary containing power estimate and parameters
    """
    # For now, delegate to the imported function
    # In the future, this could include simulation approaches as well
    return analytical_power_binary(n1, n2, p1, p2, alpha, test_type)


def sample_size_binary(p1, p2, power=0.8, alpha=0.05, allocation_ratio=1.0, test_type="Normal Approximation"):
    """
    Calculate sample size for binary outcome in parallel design.
    
    Parameters
    ----------
    p1 : float
        Proportion in group 1 (between 0 and 1)
    p2 : float
        Proportion in group 2 (between 0 and 1)
    power : float, optional
        Desired statistical power, by default 0.8
    alpha : float, optional
        Significance level, by default 0.05
    allocation_ratio : float, optional
        Ratio of sample sizes (n2/n1), by default 1.0
    test_type : str, optional
        Type of test to use, by default "Normal Approximation"
        
    Returns
    -------
    dict
        Dictionary containing sample sizes and parameters
    """
    # For now, delegate to the imported function
    return analytical_sample_size_binary(p1, p2, power, alpha, allocation_ratio, test_type)


def sample_size_binary_non_inferiority(p1, non_inferiority_margin, power=0.8, alpha=0.05, 
                                     allocation_ratio=1.0, assumed_difference=0.0, direction="lower"):
    """
    Calculate sample size for non-inferiority test with binary outcome.
    
    Parameters
    ----------
    p1 : float
        Proportion in the control/standard group (between 0 and 1)
    non_inferiority_margin : float
        Non-inferiority margin (must be positive, absolute difference in proportions)
    power : float, optional
        Desired power, by default 0.8
    alpha : float, optional
        Significance level (one-sided for non-inferiority), by default 0.05
    allocation_ratio : float, optional
        Ratio of sample sizes (n2/n1), by default 1.0
    assumed_difference : float, optional
        Assumed true difference between proportions (0 = proportions truly equivalent), by default 0.0
    direction : str, optional
        Direction of non-inferiority test ("lower" or "upper"), by default "lower"
    
    Returns
    -------
    dict
        Dictionary containing the required sample sizes
    """
    # For now, delegate to the imported function
    return analytical_sample_size_binary_non_inferiority(
        p1, non_inferiority_margin, power, alpha, allocation_ratio, assumed_difference, direction
    )


def power_binary_non_inferiority(n1, n2, p1, non_inferiority_margin, alpha=0.05, 
                               assumed_difference=0.0, direction="lower"):
    """
    Calculate power for non-inferiority test with binary outcome.
    
    Parameters
    ----------
    n1 : int
        Sample size in group 1 (standard treatment)
    n2 : int
        Sample size in group 2 (new treatment)
    p1 : float
        Proportion in the control/standard group (between 0 and 1)
    non_inferiority_margin : float
        Non-inferiority margin (must be positive, absolute difference in proportions)
    alpha : float, optional
        Significance level (one-sided for non-inferiority), by default 0.05
    assumed_difference : float, optional
        Assumed true difference between proportions (0 = proportions truly equivalent), by default 0.0
    direction : str, optional
        Direction of non-inferiority test ("lower" or "upper"), by default "lower"
    
    Returns
    -------
    dict
        Dictionary containing the power estimate and parameters
    """
    # Implementation will be added in future updates
    # For now, we'll calculate this using the analytical approach
    p2 = p1 + assumed_difference
    
    # One-sided alpha for non-inferiority test
    z_alpha = stats.norm.ppf(1 - alpha)
    
    # Calculate standard error
    se = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    
    # Calculate power based on direction
    if direction == "lower":
        # Lower bound non-inferiority (new treatment not worse than reference by margin)
        z_beta = ((p2 - p1) + non_inferiority_margin) / se
    else:  # direction == "upper"
        # Upper bound non-inferiority (new treatment not better than reference by margin)
        z_beta = ((p1 - p2) + non_inferiority_margin) / se
        
    # Calculate power
    power = stats.norm.cdf(z_beta - z_alpha)
    
    return {
        "power": power,
        "n1": n1,
        "n2": n2,
        "p1": p1,
        "assumed_p2": p2,
        "non_inferiority_margin": non_inferiority_margin,
        "alpha": alpha,
        "direction": direction
    }


# ===== Simulation-based Functions =====

def simulate_binary_non_inferiority(n1, n2, p1, non_inferiority_margin, nsim=1000, alpha=0.05, seed=None, assumed_difference=0.0, direction="lower"):
    """
    Simulate a parallel group RCT for non-inferiority hypothesis with binary outcome.
    
    Parameters
    ----------
    n1 : int
        Sample size in group 1 (standard treatment)
    n2 : int
        Sample size in group 2 (new treatment)
    p1 : float
        Proportion in the control/standard group (between 0 and 1)
    non_inferiority_margin : float
        Non-inferiority margin (must be positive, absolute difference in proportions)
    nsim : int, optional
        Number of simulations, by default 1000
    alpha : float, optional
        Significance level (one-sided for non-inferiority), by default 0.05
    seed : int, optional
        Random seed for reproducibility, by default None
    assumed_difference : float, optional
        Assumed true difference between proportions (0 = proportions truly equivalent), by default 0.0
    direction : str, optional
        Direction of non-inferiority test ("lower" or "upper"), by default "lower"
        
    Returns
    -------
    dict
        Dictionary containing simulation results, including empirical power
    """
    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)
    
    # Calculate p2 based on assumed difference
    p2 = p1 + assumed_difference
    
    # Initialize results
    reject_count = 0
    p_values = []
    
    for _ in range(nsim):
        # Generate random samples
        x1 = np.random.binomial(1, p1, size=n1)
        x2 = np.random.binomial(1, p2, size=n2)
        
        # Calculate sample proportions
        p1_hat = np.mean(x1)
        p2_hat = np.mean(x2)
        
        # Calculate difference
        diff = p2_hat - p1_hat
        
        # Calculate standard error of the difference
        # Pooled estimate for variance (under null)
        p_pooled = (sum(x1) + sum(x2)) / (n1 + n2)
        se_diff = np.sqrt(p_pooled * (1 - p_pooled) * (1/n1 + 1/n2))
        
        # Handle case of zero successes or failures
        if se_diff == 0:
            se_diff = np.sqrt((p1_hat * (1 - p1_hat) / n1) + (p2_hat * (1 - p2_hat) / n2))
            if se_diff == 0:  # Still zero
                se_diff = 0.0001  # Small non-zero value to avoid division by zero
        
        # Perform non-inferiority test
        if direction == "lower":
            # H0: p2 - p1 <= -margin (new treatment is inferior)
            # H1: p2 - p1 > -margin (new treatment is non-inferior)
            z_score = (diff + non_inferiority_margin) / se_diff
            p_value = 1 - stats.norm.cdf(z_score)
        else:  # upper
            # H0: p2 - p1 >= margin (new treatment is superior by too much)
            # H1: p2 - p1 < margin (new treatment is non-superior by too much)
            z_score = (diff - non_inferiority_margin) / se_diff
            p_value = stats.norm.cdf(z_score)
        
        p_values.append(p_value)
        if p_value < alpha:
            reject_count += 1
    
    # Calculate empirical power
    empirical_power = reject_count / nsim
    
    # Return results
    return {
        "empirical_power": empirical_power,
        "mean_p_value": np.mean(p_values),
        "n1": n1,
        "n2": n2,
        "p1": p1,
        "p2": p2,
        "test_type": test_type,
        "nsim": nsim,
        "alpha": alpha
    }


def sample_size_binary_sim(p1, p2, power=0.8, alpha=0.05, allocation_ratio=1.0, nsim=1000, 
                        min_n=10, max_n=1000, step=10, test_type="Normal Approximation"):
    """
    Calculate sample size required for detecting a difference in proportions using simulation.
    
    Parameters
    ----------
    p1 : float
        Proportion in control group (between 0 and 1)
    p2 : float
        Proportion in intervention group (between 0 and 1)
    power : float, optional
        Desired statistical power (1 - beta), by default 0.8
    alpha : float, optional
        Significance level, by default 0.05
    allocation_ratio : float, optional
        Ratio of sample sizes (n2/n1), by default 1.0
    nsim : int, optional
        Number of simulations per sample size, by default 1000
    min_n : int, optional
        Minimum sample size to try for group 1, by default 10
    max_n : int, optional
        Maximum sample size to try for group 1, by default 1000
    step : int, optional
        Step size for incrementing sample size, by default 10
    test_type : str, optional
        Type of statistical test to use, by default "Normal Approximation"
    
    Returns
    -------
    dict
        Dictionary containing the required sample sizes for each group and total
    """
    # Handle test_type format for compatibility
    if test_type.lower() == "normal approximation":
        test = "normal_approximation"
    elif test_type.lower() == "likelihood ratio test":
        test = "likelihood_ratio"
    elif test_type.lower() == "exact test":
        test = "fishers_exact"
    else:
        test = test_type
    
    # Initialize search parameters
    n1 = min_n
    achieved_power = 0.0
    
    # Perform binary search to find required sample size
    while n1 <= max_n and achieved_power < power:
        # Calculate n2 based on allocation ratio
        n2 = math.ceil(n1 * allocation_ratio)
        
        # Simulate trials with current sample sizes
        sim_result = simulate_binary_trial(n1, n2, p1, p2, nsim=nsim, alpha=alpha, test_type=test)
        
        # Extract achieved power
        achieved_power = sim_result["empirical_power"]
        
        # If power is sufficient, break the loop
        if achieved_power >= power:
            break
        
        # Otherwise, increase sample size
        n1 += step
    
    # Calculate total sample size
    n_total = n1 + n2
    
    # Check if we reached the maximum without achieving desired power
    if n1 > max_n:
        # Use the maximum sample size and report the achieved power
        n1 = max_n
        n2 = math.ceil(n1 * allocation_ratio)
        n_total = n1 + n2
        sim_result = simulate_binary_trial(n1, n2, p1, p2, nsim=nsim, alpha=alpha, test_type=test)
        achieved_power = sim_result["empirical_power"]
    
    # Return result dictionary
    return {
        "sample_size_1": n1,
        "sample_size_2": n2,
        "total_sample_size": n_total,
        "simulations": nsim,
        "achieved_power": achieved_power,
        "p1": p1,
        "p2": p2,
        "alpha": alpha,
        "target_power": power,
        "allocation_ratio": allocation_ratio,
        "test_type": test_type,
        "absolute_effect": abs(p2 - p1),
        "method": "simulation"
    }


def min_detectable_effect_binary_sim(n1, n2, p1, power=0.8, nsim=1000, alpha=0.05, precision=0.01, test_type="Normal Approximation"):
    """
    Calculate minimum detectable effect for binary outcomes using simulation-based approach.
    
    Parameters
    ----------
    n1 : int
        Sample size for group 1
    n2 : int
        Sample size for group 2
    p1 : float
        Proportion in control group (between 0 and 1)
    power : float, optional
        Desired statistical power (1 - beta), by default 0.8
    nsim : int, optional
        Number of simulations, by default 1000
    alpha : float, optional
        Significance level, by default 0.05
    precision : float, optional
        Desired precision for the effect size, by default 0.01
    test_type : str, optional
        Type of statistical test to use, by default "Normal Approximation"
    
    Returns
    -------
    dict
        Dictionary containing the minimum detectable difference in proportions
    """
    # Handle test_type format for compatibility
    if test_type.lower() == "normal approximation":
        test = "normal_approximation"
    elif test_type.lower() == "likelihood ratio test":
        test = "likelihood_ratio"
    elif test_type.lower() == "exact test":
        test = "fishers_exact"
    else:
        test = test_type
    
    # Determine search range based on p1
    # If p1 is small, look for an increase (positive difference)
    # If p1 is large, look for a decrease (negative difference)
    if p1 <= 0.5:
        min_diff = precision  # Smallest positive difference
        max_diff = 1.0 - p1 - precision  # Largest possible increase
        direction = 1  # Positive direction (searching for an increase)
    else:
        min_diff = precision  # Smallest positive difference
        max_diff = p1 - precision  # Largest possible decrease
        direction = -1  # Negative direction (searching for a decrease)
    
    # Initialize search
    low = min_diff
    high = max_diff
    current_diff = (low + high) / 2
    iterations = 0
    max_iterations = 20  # Prevent infinite loops
    
    while (high - low) > precision and iterations < max_iterations:
        # Calculate p2 based on current difference
        p2 = p1 + direction * current_diff
        
        # Ensure p2 is within valid range [0,1]
        p2 = max(0, min(1, p2))
        
        # Simulate trials with current effect size
        sim_result = simulate_binary_trial(n1, n2, p1, p2, nsim=nsim, alpha=alpha, test_type=test)
        achieved_power = sim_result["empirical_power"]
        
        # Binary search: adjust the effect size based on achieved power
        if achieved_power < power:
            # Effect too small, increase it
            low = current_diff
        else:
            # Effect sufficient, try to decrease it
            high = current_diff
        
        # Set new midpoint for next iteration
        current_diff = (low + high) / 2
        iterations += 1
    
    # Final effect size
    mde = direction * current_diff
    p2 = p1 + mde
    
    # One final simulation to verify the result
    final_sim = simulate_binary_trial(n1, n2, p1, p2, nsim=nsim, alpha=alpha, test_type=test)
    
    # Return results
    return {
        "minimum_detectable_effect": mde,
        "absolute_mde": abs(mde),
        "p1": p1,
        "p2": p2,
        "n1": n1,
        "n2": n2,
        "achieved_power": final_sim["empirical_power"],
        "target_power": power,
        "alpha": alpha,
        "test_type": test_type,
        "simulations": nsim,
        "method": "simulation",
        "iterations": iterations
    }


def sample_size_binary_non_inferiority_sim(p1, non_inferiority_margin, power=0.8, alpha=0.05, 
                                        allocation_ratio=1.0, nsim=1000, min_n=10, max_n=1000, 
                                        step=10, assumed_difference=0.0, direction="lower"):
    """
    Calculate sample size for non-inferiority test with binary outcome using simulation.
    
    Parameters
    ----------
    p1 : float
        Proportion in the control/standard group (between 0 and 1)
    non_inferiority_margin : float
        Non-inferiority margin (must be positive, absolute difference in proportions)
    power : float, optional
        Desired power, by default 0.8
    alpha : float, optional
        Significance level (one-sided for non-inferiority), by default 0.05
    allocation_ratio : float, optional
        Ratio of sample sizes (n2/n1), by default 1.0
    nsim : int, optional
        Number of simulations per sample size, by default 1000
    min_n : int, optional
        Minimum sample size to try for group 1, by default 10
    max_n : int, optional
        Maximum sample size to try for group 1, by default 1000
    step : int, optional
        Step size for incrementing sample size, by default 10
    assumed_difference : float, optional
        Assumed true difference between proportions (0 = proportions truly equivalent), by default 0.0
    direction : str, optional
        Direction of non-inferiority test ("lower" or "upper"), by default "lower"
    
    Returns
    -------
    dict
        Dictionary containing the required sample sizes
    """
    # Initialize search parameters
    n1 = min_n
    achieved_power = 0.0
    
    # Calculate the expected p2 based on assumed difference from p1
    p2 = p1 + assumed_difference
    
    # Perform binary search to find required sample size
    while n1 <= max_n and achieved_power < power:
        # Calculate n2 based on allocation ratio
        n2 = math.ceil(n1 * allocation_ratio)
        
        # Simulate trials with current sample sizes
        sim_result = simulate_binary_non_inferiority(n1, n2, p1, non_inferiority_margin, 
                                                  nsim=nsim, alpha=alpha, 
                                                  assumed_difference=assumed_difference,
                                                  direction=direction)
        
        # Extract achieved power
        achieved_power = sim_result["empirical_power"]
        
        # If power is sufficient, break the loop
        if achieved_power >= power:
            break
        
        # Otherwise, increase sample size
        n1 += step
    
    # Calculate total sample size
    n_total = n1 + n2
    
    # Check if we reached the maximum without achieving desired power
    if n1 > max_n:
        # Use the maximum sample size and report the achieved power
        n1 = max_n
        n2 = math.ceil(n1 * allocation_ratio)
        n_total = n1 + n2
        sim_result = simulate_binary_non_inferiority(n1, n2, p1, non_inferiority_margin, 
                                                   nsim=nsim, alpha=alpha, 
                                                   assumed_difference=assumed_difference,
                                                   direction=direction)
        achieved_power = sim_result["empirical_power"]
    
    # Return result dictionary
    return {
        "sample_size_1": n1,
        "sample_size_2": n2,
        "total_sample_size": n_total,
        "simulations": nsim,
        "achieved_power": achieved_power,
        "p1": p1,
        "p2": p2,
        "non_inferiority_margin": non_inferiority_margin,
        "assumed_difference": assumed_difference,
        "direction": direction,
        "alpha": alpha,
        "target_power": power,
        "allocation_ratio": allocation_ratio,
        "method": "simulation"
    }


def min_detectable_binary_non_inferiority_margin_sim(n1, n2, p1, power=0.8, alpha=0.05, nsim=1000, 
                                                  precision=0.01, assumed_difference=0.0, direction="lower"):
    """
    Calculate the minimum detectable non-inferiority margin for binary outcomes using simulation.
    
    Parameters
    ----------
    n1 : int
        Sample size in group 1 (standard treatment)
    n2 : int
        Sample size in group 2 (new treatment)
    p1 : float
        Proportion in the control/standard group (between 0 and 1)
    power : float, optional
        Desired power, by default 0.8
    alpha : float, optional
        Significance level (one-sided for non-inferiority), by default 0.05
    nsim : int, optional
        Number of simulations, by default 1000
    precision : float, optional
        Desired precision for the margin, by default 0.01
    assumed_difference : float, optional
        Assumed true difference between proportions (0 = proportions truly equivalent), by default 0.0
    direction : str, optional
        Direction of non-inferiority test ("lower" or "upper"), by default "lower"
    
    Returns
    -------
    dict
        Dictionary containing the minimum detectable non-inferiority margin
    """
    # Calculate p2 based on assumed difference
    p2 = p1 + assumed_difference
    
    # Set up search range for margin
    # The maximum possible margin depends on the direction of non-inferiority
    if direction == "lower":
        max_margin = p1  # Maximum possible margin is p1 itself
    else:  # upper
        max_margin = 1.0 - p1  # Maximum possible margin is distance from p1 to 1.0
    
    # Initialize binary search for margin
    low_margin = precision  # Minimum meaningful margin (based on precision)
    high_margin = max_margin
    current_margin = (low_margin + high_margin) / 2
    iterations = 0
    max_iterations = 20  # Prevent infinite loops
    
    while (high_margin - low_margin) > precision and iterations < max_iterations:
        # Simulate trials with current margin
        sim_result = simulate_binary_non_inferiority(
            n1=n1, 
            n2=n2, 
            p1=p1, 
            non_inferiority_margin=current_margin,
            nsim=nsim, 
            alpha=alpha, 
            assumed_difference=assumed_difference,
            direction=direction
        )
        
        # Extract achieved power
        achieved_power = sim_result["empirical_power"]
        
        # Adjust the margin based on achieved power
        if achieved_power < power:
            # Margin too large (making it harder to demonstrate non-inferiority)
            # Decrease the margin
            high_margin = current_margin
        else:
            # Power sufficient, try to increase margin (making the test more stringent)
            low_margin = current_margin
        
        # Set new midpoint for next iteration
        current_margin = (low_margin + high_margin) / 2
        iterations += 1
    
    # Final margin and verification simulation
    final_sim = simulate_binary_non_inferiority(
        n1=n1, 
        n2=n2, 
        p1=p1, 
        non_inferiority_margin=current_margin,
        nsim=nsim, 
        alpha=alpha, 
        assumed_difference=assumed_difference,
        direction=direction
    )
    
    # Return results
    return {
        "minimum_detectable_margin": current_margin,
        "p1": p1,
        "p2": p2,
        "n1": n1,
        "n2": n2,
        "achieved_power": final_sim["empirical_power"],
        "target_power": power,
        "alpha": alpha,
        "assumed_difference": assumed_difference,
        "direction": direction,
        "simulations": nsim,
        "method": "simulation",
        "iterations": iterations
    }


def simulate_binary_trial(n1, n2, p1, p2, nsim=1000, alpha=0.05, test_type="normal_approximation", seed=None):
    """
    Simulate a parallel group RCT with binary outcome.
    
    Parameters
    ----------
    n1 : int
        Sample size in group 1
    n2 : int
        Sample size in group 2
    p1 : float
        Proportion in group 1 (between 0 and 1)
    p2 : float
        Proportion in group 2 (between 0 and 1)
    nsim : int, optional
        Number of simulations, by default 1000
    alpha : float, optional
        Significance level, by default 0.05
    test_type : str, optional
        Type of test to use, by default "normal_approximation"
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
        # Generate data for both groups
        group1 = np.random.binomial(1, p1, n1)
        group2 = np.random.binomial(1, p2, n2)
        
        # Count successes
        s1 = np.sum(group1)
        s2 = np.sum(group2)
        
        # Perform statistical test
        p_value = perform_binary_test(n1, n2, s1, s2, test_type)
        
        # Count significant results
        if p_value < alpha:
            significant_results += 1
    
    # Calculate empirical power
    power = significant_results / nsim
    
    return {
        "power": power,
        "significant_results": significant_results,
        "nsim": nsim,
        "n1": n1,
        "n2": n2,
        "p1": p1,
        "p2": p2,
        "test_type": test_type
    }
