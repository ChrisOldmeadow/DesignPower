"""
Simulation-based methods for interrupted time series designs.

This module provides functions for power analysis and sample size calculation
for interrupted time series designs using simulation-based approaches.
"""
import numpy as np
from scipy import stats


def simulate_continuous(n_pre, n_post, mean_pre, mean_post, std_dev, 
                       nsim=1000, alpha=0.05, autocorr=0.0):
    """
    Simulate an interrupted time series with continuous outcome and estimate power.
    
    Parameters
    ----------
    n_pre : int
        Number of time points pre-intervention
    n_post : int
        Number of time points post-intervention
    mean_pre : float
        Mean outcome pre-intervention
    mean_post : float
        Mean outcome post-intervention
    std_dev : float
        Standard deviation of the outcome
    nsim : int, optional
        Number of simulations, by default 1000
    alpha : float, optional
        Significance level, by default 0.05
    autocorr : float, optional
        Autocorrelation coefficient, by default 0.0
    
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
        # Generate time series data with autocorrelation
        pre_data = []
        post_data = []
        
        # Generate pre-intervention data
        prev_value = mean_pre + np.random.normal(0, std_dev)
        pre_data.append(prev_value)
        
        for i in range(1, n_pre):
            # New value depends on previous value (autocorrelation) plus random noise
            new_value = mean_pre + autocorr * (prev_value - mean_pre) + np.random.normal(0, std_dev * np.sqrt(1 - autocorr**2))
            pre_data.append(new_value)
            prev_value = new_value
        
        # Generate post-intervention data (continuing the autocorrelation)
        for i in range(n_post):
            # New value depends on previous value (autocorrelation) plus random noise
            new_value = mean_post + autocorr * (prev_value - mean_post) + np.random.normal(0, std_dev * np.sqrt(1 - autocorr**2))
            post_data.append(new_value)
            prev_value = new_value
        
        # Perform t-test
        t_stat, p_val = stats.ttest_ind(pre_data, post_data, equal_var=True)
        
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
            "n_pre": n_pre,
            "n_post": n_post,
            "mean_pre": mean_pre,
            "mean_post": mean_post,
            "std_dev": std_dev,
            "alpha": alpha,
            "autocorr": autocorr
        }
    }


def simulate_binary(n_pre, n_post, p_pre, p_post, nsim=1000, alpha=0.05, autocorr=0.0):
    """
    Simulate an interrupted time series with binary outcome and estimate power.
    
    Parameters
    ----------
    n_pre : int
        Number of observations per time point pre-intervention
    n_post : int
        Number of observations per time point post-intervention
    p_pre : float
        Probability of success pre-intervention
    p_post : float
        Probability of success post-intervention
    nsim : int, optional
        Number of simulations, by default 1000
    alpha : float, optional
        Significance level, by default 0.05
    autocorr : float, optional
        Autocorrelation coefficient, by default 0.0
    
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
        # Generate proportions with autocorrelation
        pre_props = []
        post_props = []
        
        # Generate pre-intervention proportions
        prev_logit = np.log(p_pre / (1 - p_pre))  # Convert to logit scale
        
        for i in range(n_pre):
            # Add autocorrelation on logit scale
            if i == 0:
                logit = prev_logit + np.random.normal(0, 1)
            else:
                logit = prev_logit + autocorr * (logit - prev_logit) + np.random.normal(0, np.sqrt(1 - autocorr**2))
            
            # Convert back to probability
            p = 1 / (1 + np.exp(-logit))
            
            # Generate binomial outcome
            successes = np.random.binomial(100, p)  # Assume 100 observations per time point
            pre_props.append(successes / 100)
        
        # Generate post-intervention proportions
        prev_logit = np.log(p_post / (1 - p_post))  # Convert to logit scale
        
        for i in range(n_post):
            # Add autocorrelation on logit scale
            if i == 0:
                logit = prev_logit + np.random.normal(0, 1)
            else:
                logit = prev_logit + autocorr * (logit - prev_logit) + np.random.normal(0, np.sqrt(1 - autocorr**2))
            
            # Convert back to probability
            p = 1 / (1 + np.exp(-logit))
            
            # Generate binomial outcome
            successes = np.random.binomial(100, p)  # Assume 100 observations per time point
            post_props.append(successes / 100)
        
        # Perform t-test (simplified approach)
        t_stat, p_val = stats.ttest_ind(pre_props, post_props, equal_var=True)
        
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
            "n_pre": n_pre,
            "n_post": n_post,
            "p_pre": p_pre,
            "p_post": p_post,
            "alpha": alpha,
            "autocorr": autocorr
        }
    }


def simulate_count(n_pre, n_post, lambda_pre, lambda_post, 
                  nsim=1000, alpha=0.05, autocorr=0.0, overdispersion=1.0):
    """
    Simulate an interrupted time series with count outcome and estimate power.
    
    Parameters
    ----------
    n_pre : int
        Number of time points pre-intervention
    n_post : int
        Number of time points post-intervention
    lambda_pre : float
        Mean count pre-intervention
    lambda_post : float
        Mean count post-intervention
    nsim : int, optional
        Number of simulations, by default 1000
    alpha : float, optional
        Significance level, by default 0.05
    autocorr : float, optional
        Autocorrelation coefficient, by default 0.0
    overdispersion : float, optional
        Overdispersion parameter (variance/mean), by default 1.0
    
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
        # Generate time series data with autocorrelation
        pre_data = []
        post_data = []
        
        # Generate pre-intervention data
        prev_log_lambda = np.log(lambda_pre)  # Work on log scale for counts
        
        for i in range(n_pre):
            # Add autocorrelation on log scale
            if i == 0:
                log_lambda = prev_log_lambda + np.random.normal(0, 0.1)
            else:
                log_lambda = prev_log_lambda + autocorr * (log_lambda - prev_log_lambda) + np.random.normal(0, 0.1 * np.sqrt(1 - autocorr**2))
            
            # Convert back to count rate
            lambda_i = np.exp(log_lambda)
            
            # Generate counts with overdispersion (using negative binomial)
            if overdispersion > 1.0:
                # For negative binomial, we need size parameter r
                r = lambda_i / (overdispersion - 1)
                p = 1 / overdispersion
                count = np.random.negative_binomial(r, p)
            else:
                # Regular Poisson
                count = np.random.poisson(lambda_i)
            
            pre_data.append(count)
        
        # Generate post-intervention data (continuing the autocorrelation)
        prev_log_lambda = np.log(lambda_post)
        
        for i in range(n_post):
            # Add autocorrelation on log scale
            if i == 0:
                log_lambda = prev_log_lambda + np.random.normal(0, 0.1)
            else:
                log_lambda = prev_log_lambda + autocorr * (log_lambda - prev_log_lambda) + np.random.normal(0, 0.1 * np.sqrt(1 - autocorr**2))
            
            # Convert back to count rate
            lambda_i = np.exp(log_lambda)
            
            # Generate counts with overdispersion (using negative binomial)
            if overdispersion > 1.0:
                # For negative binomial, we need size parameter r
                r = lambda_i / (overdispersion - 1)
                p = 1 / overdispersion
                count = np.random.negative_binomial(r, p)
            else:
                # Regular Poisson
                count = np.random.poisson(lambda_i)
            
            post_data.append(count)
        
        # Perform t-test (this is a simplification; in practice, a more appropriate count model would be used)
        t_stat, p_val = stats.ttest_ind(pre_data, post_data, equal_var=True)
        
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
            "n_pre": n_pre,
            "n_post": n_post,
            "lambda_pre": lambda_pre,
            "lambda_post": lambda_post,
            "alpha": alpha,
            "autocorr": autocorr,
            "overdispersion": overdispersion
        }
    }
