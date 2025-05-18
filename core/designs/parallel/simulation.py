"""
Simulation-based methods for parallel group randomized controlled trials.

This module provides functions for power analysis and sample size calculation
for parallel group RCTs using simulation-based approaches.
"""
import numpy as np
from scipy import stats
from scipy import optimize


def simulate_continuous(n1, n2, mean1, mean2, std_dev, nsim=1000, alpha=0.05):
    """
    Simulate a parallel RCT with continuous outcome and estimate power.
    
    Parameters
    ----------
    n1 : int
        Sample size for group 1
    n2 : int
        Sample size for group 2
    mean1 : float
        Mean outcome in group 1
    mean2 : float
        Mean outcome in group 2
    std_dev : float
        Standard deviation of outcome (assumed equal in both groups)
    nsim : int, optional
        Number of simulations, by default 1000
    alpha : float, optional
        Significance level, by default 0.05
    
    Returns
    -------
    dict
        Dictionary containing the estimated power and simulation details
    """
    # Initialize counter for significant results
    sig_count = 0
    
    # Store p-values for all simulations
    p_values = []
    
    # Run simulations
    for _ in range(nsim):
        # Generate data for both groups
        group1 = np.random.normal(mean1, std_dev, n1)
        group2 = np.random.normal(mean2, std_dev, n2)
        
        # Perform t-test
        t_stat, p_val = stats.ttest_ind(group1, group2, equal_var=True)
        
        # Store p-value
        p_values.append(p_val)
        
        # Check if result is significant
        if p_val < alpha:
            sig_count += 1
    
    # Calculate power
    power = sig_count / nsim
    
    return {
        "power": power,
        "mean_p_value": np.mean(p_values),
        "median_p_value": np.median(p_values),
        "nsim": nsim,
        "parameters": {
            "n1": n1,
            "n2": n2,
            "mean1": mean1,
            "mean2": mean2,
            "delta": mean2 - mean1,
            "std_dev": std_dev,
            "alpha": alpha
        }
    }


def min_detectable_effect_continuous(n1, n2, std_dev, power=0.8, nsim=1000, alpha=0.05, precision=0.01):
    """
    Calculate minimum detectable effect size using simulation-based approach and optimization.
    
    Parameters
    ----------
    n1 : int
        Sample size for group 1
    n2 : int
        Sample size for group 2
    std_dev : float
        Standard deviation of the outcome
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
        Dictionary containing the minimum detectable effect and input parameters
    """
    # Define the function to optimize
    # This function returns the difference between simulated power and target power
    def power_difference(delta):
        # Always set mean1 to 0 and mean2 to delta for simplicity
        result = simulate_continuous(n1=n1, n2=n2, mean1=0, mean2=delta[0], std_dev=std_dev, nsim=nsim, alpha=alpha)
        return abs(result["power"] - power)
    
    # Initial guess - use analytical formula as starting point
    z_alpha = stats.norm.ppf(1 - alpha/2)
    z_beta = stats.norm.ppf(power)
    initial_delta = (z_alpha + z_beta) * std_dev * np.sqrt(1/n1 + 1/n2)
    
    # Run optimization
    result = optimize.minimize(power_difference, [initial_delta], method='Nelder-Mead', 
                              tol=precision, options={'maxiter': 100})
    
    # Extract optimized delta
    delta = float(result.x[0])
    
    return {
        "delta": delta,
        "parameters": {
            "n1": n1,
            "n2": n2,
            "std_dev": std_dev,
            "power": power,
            "nsim": nsim,
            "alpha": alpha,
            "optimization_success": result.success
        }
    }


def simulate_binary(n1, n2, p1, p2, nsim=1000, alpha=0.05):
    """
    Simulate a parallel RCT with binary outcome and estimate power.
    
    Parameters
    ----------
    n1 : int
        Sample size for group 1
    n2 : int
        Sample size for group 2
    p1 : float
        Proportion in group 1 (between 0 and 1)
    p2 : float
        Proportion in group 2 (between 0 and 1)
    nsim : int, optional
        Number of simulations, by default 1000
    alpha : float, optional
        Significance level, by default 0.05
    
    Returns
    -------
    dict
        Dictionary containing the estimated power and simulation details
    """
    # Initialize counter for significant results
    sig_count = 0
    
    # Store p-values for all simulations
    p_values = []
    
    # Run simulations
    for _ in range(nsim):
        # Generate data for both groups
        group1 = np.random.binomial(1, p1, n1)
        group2 = np.random.binomial(1, p2, n2)
        
        # Calculate proportions
        prop1 = np.mean(group1)
        prop2 = np.mean(group2)
        
        # Calculate standard error under null hypothesis
        p_pooled = (sum(group1) + sum(group2)) / (n1 + n2)
        se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n1 + 1/n2))
        
        # Calculate z-statistic
        if se > 0:
            z_stat = (prop2 - prop1) / se
            # Calculate p-value
            p_val = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        else:
            # If standard error is 0 (e.g., both groups have same outcome), set p-value to 1
            p_val = 1.0
        
        # Store p-value
        p_values.append(p_val)
        
        # Check if result is significant
        if p_val < alpha:
            sig_count += 1
    
    # Calculate power
    power = sig_count / nsim
    
    return {
        "power": power,
        "mean_p_value": np.mean(p_values),
        "median_p_value": np.median(p_values),
        "nsim": nsim,
        "parameters": {
            "n1": n1,
            "n2": n2,
            "p1": p1,
            "p2": p2,
            "alpha": alpha
        }
    }
