"""
Core statistics utility functions for power calculations.
"""
import math
import scipy.stats as stats

def power_calculations(alpha=0.05, beta=0.2, sides=2):
    """
    Basic power calculation utility functions.
    
    Args:
        alpha: Type I error rate
        beta: Type II error rate (1 - power)
        sides: One or two-sided test
        
    Returns:
        Dictionary with z-critical values
    """
    power = 1 - beta
    z_alpha = stats.norm.ppf(1 - alpha/(sides))
    z_beta = stats.norm.ppf(power)
    
    return {
        "z_alpha": z_alpha,
        "z_beta": z_beta,
        "critical_values": (z_alpha, z_beta)
    }
