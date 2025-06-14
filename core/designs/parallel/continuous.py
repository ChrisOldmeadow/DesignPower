"""
Continuous outcome functions for parallel group randomized controlled trials.

This module provides comprehensive functions for power analysis and
sample size calculation for parallel group RCTs with continuous outcomes.
Includes both analytical and simulation-based approaches.
"""

import numpy as np
import math
from scipy import stats

# Import from specialized modules
from .analytical_continuous import (
    sample_size_continuous as analytical_sample_size_continuous,
    power_continuous as analytical_power_continuous,
    min_detectable_effect_continuous as analytical_min_detectable_effect_continuous
)
from .simulation_continuous import (
    simulate_continuous_trial,
    sample_size_continuous_sim,
    power_continuous_sim,
    min_detectable_effect_continuous_sim
)


# ===== Main Functions =====

def sample_size_continuous(mean1, mean2, sd1, sd2=None, power=0.8, alpha=0.05, 
                          allocation_ratio=1.0, test="t-test", method="analytical",
                          nsim=1000, min_n=10, max_n=1000, precision=0.01, seed=None):
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
    method : str, optional
        Calculation method, either "analytical" or "simulation", by default "analytical"
    nsim : int, optional
        Number of simulations (only used for simulation method), by default 1000
    min_n : int, optional
        Minimum sample size to consider (simulation only), by default 10
    max_n : int, optional
        Maximum sample size to consider (simulation only), by default 1000
    precision : float, optional
        Desired precision for power in simulations, by default 0.01
    seed : int, optional
        Random seed for reproducibility, by default None
        
    Returns
    -------
    dict
        Dictionary containing sample sizes and parameters
    """
    if method == "analytical":
        # Use analytical method
        return analytical_sample_size_continuous(
            mean1=mean1,
            mean2=mean2,
            sd1=sd1,
            sd2=sd2,
            power=power, 
            alpha=alpha, 
            allocation_ratio=allocation_ratio,
            test=test
        )
    else:
        # Use simulation method
        return sample_size_continuous_sim(
            mean1=mean1,
            mean2=mean2,
            sd1=sd1,
            sd2=sd2,
            power=power,
            alpha=alpha,
            allocation_ratio=allocation_ratio,
            test=test,
            nsim=nsim,
            min_n=min_n,
            max_n=max_n,
            precision=precision,
            seed=seed
        )


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
    # For now, delegate to the imported function
    return analytical_power_continuous(n1, n2, mean1, mean2, sd1, sd2, alpha, test)


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
    # Use the same formula as in the original analytical module
    if sd2 is None:
        sd2 = sd1
    
    # Calculate z-scores for given alpha and power
    z_alpha = stats.norm.ppf(1 - alpha/2)
    z_beta = stats.norm.ppf(power)
    
    # Calculate pooled variance term
    variance_term = sd1**2/n1 + sd2**2/n2
    
    # Calculate minimum detectable effect
    mde = (z_alpha + z_beta) * math.sqrt(variance_term)
    
    return {
        "minimum_detectable_effect": mde,
        "n1": n1,
        "n2": n2,
        "sd1": sd1,
        "sd2": sd2,
        "power": power,
        "alpha": alpha
    }


# ===== Simulation-based Functions (imported from simulation_continuous module) =====

# simulate_continuous_trial is imported from simulation_continuous module above
