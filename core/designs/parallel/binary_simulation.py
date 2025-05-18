"""
Simulation-based methods for binary outcomes in parallel group randomized controlled trials.

This module provides functions for power analysis and sample size calculation
for parallel group RCTs with binary outcomes using simulation-based approaches.
"""
import numpy as np
import math
from scipy import stats


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
        Dictionary containing the estimated power and simulation details
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Calculate the proportion in the new treatment group based on the assumed difference
    p2 = p1 + assumed_difference
    
    # Validate inputs
    if p1 <= 0 or p1 >= 1:
        raise ValueError("p1 must be between 0 and 1")
    if p2 <= 0 or p2 >= 1:
        raise ValueError("p2 (calculated as p1 + assumed_difference) must be between 0 and 1")
    if non_inferiority_margin <= 0:
        raise ValueError("Non-inferiority margin must be positive")
    if direction not in ["lower", "upper"]:
        raise ValueError("Direction must be 'lower' or 'upper'")
    
    # Store p-values from each simulation
    p_values = []
    
    for _ in range(nsim):
        # Generate binary outcomes for each group
        group1 = np.random.binomial(1, p1, n1)  # Standard treatment
        group2 = np.random.binomial(1, p2, n2)  # New treatment
        
        # Calculate proportions in each group
        prop1 = np.mean(group1)
        prop2 = np.mean(group2)
        
        # Calculate the difference (new - standard)
        diff = prop2 - prop1
        
        # Calculate the standard error of the difference
        # using the pooled estimate for the variance
        p_pooled = (sum(group1) + sum(group2)) / (n1 + n2)
        se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n1 + 1/n2))
        
        # Calculate the test statistic based on direction
        if direction == "lower":
            # Testing that new treatment is not worse than standard by more than margin
            # H0: p2 - p1 <= -margin, H1: p2 - p1 > -margin
            test_statistic = (diff + non_inferiority_margin) / se
        else:  # "upper"
            # Testing that new treatment is not better than standard by more than margin
            # H0: p2 - p1 >= margin, H1: p2 - p1 < margin
            test_statistic = (non_inferiority_margin - diff) / se
        
        # Calculate one-sided p-value
        p_value = 1 - stats.norm.cdf(test_statistic)
        p_values.append(p_value)
    
    # Calculate power as the proportion of simulations with p-value < alpha
    significant_tests = sum(p < alpha for p in p_values)
    power_estimate = significant_tests / nsim
    
    return {
        "power": power_estimate,
        "significant_tests": significant_tests,
        "parameters": {
            "n1": n1,
            "n2": n2,
            "p1": p1,
            "p2": p2,
            "non_inferiority_margin": non_inferiority_margin,
            "assumed_difference": assumed_difference,
            "direction": direction,
            "nsim": nsim,
            "alpha": alpha,
            "hypothesis_type": "non-inferiority"
        }
    }


def sample_size_binary_sim(p1, p2, power=0.8, alpha=0.05, allocation_ratio=1.0, nsim=1000, min_n=10, max_n=1000, step=10):
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
    
    Returns
    -------
    dict
        Dictionary containing the required sample sizes for each group and total
    """
    # Validate inputs
    if p1 <= 0 or p1 >= 1 or p2 <= 0 or p2 >= 1:
        raise ValueError("Proportions must be between 0 and 1")
    if allocation_ratio <= 0:
        raise ValueError("Allocation ratio must be positive")
    if min_n < 2 or max_n < min_n:
        raise ValueError("Invalid min_n or max_n values")
    
    # Import here to avoid circular imports
    from core.designs.parallel.simulation import simulate_binary
    
    # Try different sample sizes until we reach desired power
    for n1 in range(min_n, max_n + 1, step):
        n2 = math.ceil(n1 * allocation_ratio)
        
        # Run simulation for this sample size
        sim_result = simulate_binary(n1, n2, p1, p2, nsim, alpha)
        
        # Check if we've reached the desired power
        if sim_result["power"] >= power:
            return {
                "n1": n1,
                "n2": n2,
                "total_n": n1 + n2,
                "achieved_power": sim_result["power"],
                "parameters": {
                    "p1": p1,
                    "p2": p2,
                    "power": power,
                    "alpha": alpha,
                    "allocation_ratio": allocation_ratio,
                    "nsim": nsim
                }
            }
    
    # If we reach here, we couldn't find a sufficient sample size
    raise ValueError(f"Could not achieve desired power with max_n={max_n}. Try increasing max_n.")


def min_detectable_effect_binary_sim(n1, n2, p1, power=0.8, nsim=1000, alpha=0.05, precision=0.01):
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
    
    Returns
    -------
    dict
        Dictionary containing the minimum detectable difference in proportions
    """
    # Validate inputs
    if p1 <= 0 or p1 >= 1:
        raise ValueError("p1 must be between 0 and 1")
    
    # Import here to avoid circular imports
    from core.designs.parallel.simulation import simulate_binary
    
    # Define the objective function: difference between achieved and target power
    def objective(p2):
        # Don't allow p2 to go outside valid range
        if p2 <= 0 or p2 >= 1:
            return float('inf')
        
        sim_result = simulate_binary(n1, n2, p1, p2, nsim, alpha)
        return sim_result["power"] - power
    
    # Set search bounds based on p1
    if p1 < 0.5:
        # If p1 is low, look for increase in p2
        lower_bound = p1 + 0.01
        upper_bound = min(0.99, p1 + 0.5)  # Don't exceed 0.99
    else:
        # If p1 is high, look for decrease in p2
        lower_bound = max(0.01, p1 - 0.5)  # Don't go below 0.01
        upper_bound = p1 - 0.01
    
    # Use binary search to find the minimum detectable p2
    while upper_bound - lower_bound > precision:
        mid = (lower_bound + upper_bound) / 2
        if objective(mid) >= 0:  # Power is at or above target
            if p1 < 0.5:
                upper_bound = mid  # We can go lower (closer to p1)
            else:
                lower_bound = mid  # We can go higher (closer to p1)
        else:  # Power is below target
            if p1 < 0.5:
                lower_bound = mid  # We need to go higher (further from p1)
            else:
                upper_bound = mid  # We need to go lower (further from p1)
    
    # Take the midpoint of the final range
    p2 = (lower_bound + upper_bound) / 2
    
    return {
        "p2": p2,
        "difference": abs(p2 - p1),
        "parameters": {
            "n1": n1,
            "n2": n2,
            "p1": p1,
            "power": power,
            "alpha": alpha,
            "nsim": nsim
        }
    }


def sample_size_binary_non_inferiority_sim(p1, non_inferiority_margin, power=0.8, alpha=0.05, allocation_ratio=1.0, nsim=1000, min_n=10, max_n=1000, step=10, assumed_difference=0.0, direction="lower"):
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
    # Validate inputs
    if p1 <= 0 or p1 >= 1:
        raise ValueError("p1 must be between 0 and 1")
    p2 = p1 + assumed_difference
    if p2 <= 0 or p2 >= 1:
        raise ValueError("p2 (calculated as p1 + assumed_difference) must be between 0 and 1")
    if non_inferiority_margin <= 0:
        raise ValueError("Non-inferiority margin must be positive")
    if direction not in ["lower", "upper"]:
        raise ValueError("Direction must be 'lower' or 'upper'")
    if allocation_ratio <= 0:
        raise ValueError("Allocation ratio must be positive")
    
    # Try different sample sizes until we reach desired power
    for n1 in range(min_n, max_n + 1, step):
        n2 = math.ceil(n1 * allocation_ratio)
        
        # Run simulation for this sample size
        sim_result = simulate_binary_non_inferiority(
            n1, n2, p1, non_inferiority_margin, 
            nsim, alpha, None, assumed_difference, direction
        )
        
        # Check if we've reached the desired power
        if sim_result["power"] >= power:
            return {
                "n1": n1,
                "n2": n2,
                "total_n": n1 + n2,
                "achieved_power": sim_result["power"],
                "parameters": {
                    "p1": p1,
                    "non_inferiority_margin": non_inferiority_margin,
                    "power": power,
                    "alpha": alpha,
                    "allocation_ratio": allocation_ratio,
                    "assumed_difference": assumed_difference,
                    "direction": direction,
                    "nsim": nsim,
                    "hypothesis_type": "non-inferiority"
                }
            }
    
    # If we reach here, we couldn't find a sufficient sample size
    raise ValueError(f"Could not achieve desired power with max_n={max_n}. Try increasing max_n.")


def min_detectable_binary_non_inferiority_margin_sim(n1, n2, p1, power=0.8, alpha=0.05, nsim=1000, precision=0.01, assumed_difference=0.0, direction="lower"):
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
    # Validate inputs
    if p1 <= 0 or p1 >= 1:
        raise ValueError("p1 must be between 0 and 1")
    p2 = p1 + assumed_difference
    if p2 <= 0 or p2 >= 1:
        raise ValueError("p2 (calculated as p1 + assumed_difference) must be between 0 and 1")
    if direction not in ["lower", "upper"]:
        raise ValueError("Direction must be 'lower' or 'upper'")
    
    # Define the objective function: difference between achieved and target power
    def objective(margin):
        if margin <= 0:
            return float('inf')  # Invalid margin
        
        sim_result = simulate_binary_non_inferiority(
            n1, n2, p1, margin, nsim, alpha, None, assumed_difference, direction
        )
        return sim_result["power"] - power
    
    # Use binary search to find the minimum detectable margin
    # Start with a very small margin and a reasonably large one
    lower_bound = 0.01
    upper_bound = 0.5  # Half the possible range of proportions
    
    # First, ensure our bounds bracket the true value
    while objective(upper_bound) < 0:  # Power still below target at upper bound
        upper_bound *= 2
        if upper_bound > 1:
            upper_bound = 1  # Can't exceed 1 for proportions
            if objective(upper_bound) < 0:
                raise ValueError("Could not achieve target power even with maximum margin.")
            break
    
    # Binary search within bounds
    while upper_bound - lower_bound > precision:
        mid = (lower_bound + upper_bound) / 2
        if objective(mid) >= 0:  # Power is at or above target
            upper_bound = mid  # We can try a smaller margin
        else:  # Power is below target
            lower_bound = mid  # We need a larger margin
    
    # Take the upper bound as our conservative estimate
    margin = upper_bound
    
    return {
        "margin": margin,
        "parameters": {
            "n1": n1,
            "n2": n2,
            "p1": p1,
            "power": power,
            "alpha": alpha,
            "assumed_difference": assumed_difference,
            "direction": direction,
            "nsim": nsim,
            "hypothesis_type": "non-inferiority"
        }
    }
