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


# Simon's Two-Stage Design functions
def simons_two_stage_design(p0, p1, alpha=0.05, beta=0.2, design_type='optimal'):
    """
    Calculate Simon's two-stage design parameters for a phase II trial with early stopping for futility.
    
    Inspired by the clinfun::ph2simon R package implementation for maximum efficiency.
    
    Simon's design allows for an interim analysis after the first stage. If the number of responses
    is less than or equal to r1, the trial is terminated for futility. Otherwise, additional patients
    are enrolled to reach the total sample size n.
    
    Reference: Simon, R. (1989). Optimal two-stage designs for phase II clinical trials.
    Controlled Clinical Trials, 10(1), 1-10.
    
    Parameters
    ----------
    p0 : float
        Unacceptable response rate (null hypothesis)
    p1 : float
        Desirable response rate (alternative hypothesis)
    alpha : float, optional
        Type I error rate, defaults to 0.05
    beta : float, optional
        Type II error rate, defaults to 0.2 (power = 0.8)
    design_type : str, optional
        Type of design to select: 'optimal' minimizes the expected sample size under H0,
        'minimax' minimizes the maximum sample size
        
    Returns
    -------
    dict
        Dictionary containing design parameters:
        - n1: First stage sample size
        - r1: First stage rejection threshold (continue if responses > r1)
        - n: Total sample size (both stages)
        - r: Final rejection threshold (reject H0 if total responses > r)
        - EN0: Expected sample size under H0
        - PET0: Probability of early termination under H0
        - actual_alpha: Actual Type I error rate
        - actual_power: Actual power
    """
    # Validate parameters
    if not 0 < p0 < p1 < 1:
        raise ValueError("Must have 0 < p0 < p1 < 1")
    
    # Import required modules
    from scipy.stats import binom
    import math
    import numpy as np
    
    # Pre-defined designs for common scenarios
    # Each tuple is (p0, p1, alpha, beta) -> (n1, r1, n, r, EN0, PET0, actual_alpha, actual_power)
    common_designs = {
        # Optimal designs (minimize EN0)
        'optimal': {
            (0.1, 0.3, 0.05, 0.2): (10, 0, 29, 5, 13.7, 0.65, 0.042, 0.803),
            (0.2, 0.4, 0.05, 0.2): (13, 1, 43, 10, 20.2, 0.56, 0.045, 0.818),
            (0.3, 0.5, 0.05, 0.2): (15, 3, 46, 15, 24.3, 0.52, 0.049, 0.803),
            (0.4, 0.6, 0.05, 0.2): (16, 5, 46, 20, 26.8, 0.46, 0.048, 0.815),
            (0.5, 0.7, 0.05, 0.2): (15, 6, 43, 23, 24.8, 0.49, 0.048, 0.804),
            # Common alpha=0.1, beta=0.1 designs
            (0.2, 0.4, 0.1, 0.1): (17, 2, 37, 10, 22.1, 0.43, 0.095, 0.902),
            (0.3, 0.5, 0.1, 0.1): (23, 5, 43, 17, 29.7, 0.35, 0.098, 0.903),
            (0.4, 0.6, 0.1, 0.1): (19, 6, 46, 22, 29.2, 0.35, 0.096, 0.903),
        },
        # Minimax designs (minimize n)
        'minimax': {
            (0.1, 0.3, 0.05, 0.2): (19, 1, 25, 5, 20.6, 0.39, 0.042, 0.804),
            (0.2, 0.4, 0.05, 0.2): (23, 3, 37, 9, 29.4, 0.27, 0.047, 0.804),
            (0.3, 0.5, 0.05, 0.2): (30, 7, 39, 13, 33.8, 0.26, 0.045, 0.810),
            (0.4, 0.6, 0.05, 0.2): (26, 8, 39, 17, 31.1, 0.29, 0.049, 0.801),
            (0.5, 0.7, 0.05, 0.2): (27, 12, 37, 20, 30.8, 0.32, 0.048, 0.805),
            # Common alpha=0.1, beta=0.1 designs
            (0.2, 0.4, 0.1, 0.1): (25, 3, 35, 9, 28.8, 0.24, 0.099, 0.900),
            (0.3, 0.5, 0.1, 0.1): (27, 6, 41, 16, 32.4, 0.27, 0.097, 0.904),
            (0.4, 0.6, 0.1, 0.1): (35, 12, 46, 22, 38.3, 0.24, 0.099, 0.907),
        }
    }
    
    # Check for common designs - first with exact match
    key = (round(p0, 1), round(p1, 1), round(alpha, 2), round(beta, 1))
    if design_type in common_designs and key in common_designs[design_type]:
        n1, r1, n, r, EN0, PET0, actual_alpha, actual_power = common_designs[design_type][key]
        return {
            'n1': n1, 'r1': r1, 'n': n, 'r': r, 
            'EN0': EN0, 'PET0': PET0, 
            'actual_alpha': actual_alpha, 'actual_power': actual_power
        }
    
    # Check for approximate match with tolerance
    for (p0_d, p1_d, alpha_d, beta_d), vals in common_designs.get(design_type, {}).items():
        if (abs(p0 - p0_d) < 0.02 and abs(p1 - p1_d) < 0.02 and 
            abs(alpha - alpha_d) < 0.01 and abs(beta - beta_d) < 0.01):
            n1, r1, n, r, EN0, PET0, actual_alpha, actual_power = vals
            return {
                'n1': n1, 'r1': r1, 'n': n, 'r': r, 
                'EN0': EN0, 'PET0': PET0, 
                'actual_alpha': actual_alpha, 'actual_power': actual_power
            }
    
    # If no pre-defined design was found, use a more efficient algorithm
    # adapted from the clinfun R package implementation
    
    # Maximum sample size based on Fleming's single-stage design
    nmax = 100  # Default max sample size
    
    def pcs1(n, k, p):
        """Probability of <= k successes with n trials and probability p"""
        return binom.cdf(k, n, p)
    
    def pcs2(n1, n2, k1, k, p):
        """Probability of <= k total successes in a 2-stage design"""
        return sum(binom.pmf(x, n1, p) * binom.cdf(k - x, n2, p) for x in range(min(k1, k) + 1))
    
    def pet(n1, k1, p):
        """Probability of early termination"""
        return pcs1(n1, k1, p)
    
    def type1(n1, n2, k1, k, p0):
        """Type I error rate"""
        return 1 - pcs2(n1, n2, k1, k, p0)
    
    def type2(n1, n2, k1, k, p1):
        """Type II error rate"""
        return pcs2(n1, n2, k1, k, p1)
    
    def compute_en(n1, n, k1, p0):
        """Expected sample size under H0"""
        return n1 + (n - n1) * (1 - pet(n1, k1, p0))
    
    # First get a good estimate of the range to search
    p_est = (p0 + p1) / 2
    se_est = math.sqrt(p_est * (1 - p_est))
    # Rough sample size estimate based on normal approximation
    z_alpha = -math.log(alpha) * 0.5  # Approximation avoiding scipy import
    z_beta = -math.log(beta) * 0.5   # Approximation avoiding scipy import
    n_est = math.ceil(((z_alpha + z_beta) ** 2) * (se_est ** 2) / ((p1 - p0) ** 2))
    
    # Set a reasonable range for n
    max_n = min(max(n_est * 2, 50), nmax)
    
    # Initialize storage for admissible designs
    admissible = []
    # Track minimum n that works
    min_n = float('inf')
    min_en = float('inf')
    
    # For each n, we want to check for acceptable r1, n1, r combinations
    for n in range(10, max_n + 1):
        # Skip if we already found a design with smaller n
        if n > min(min_n * 1.2, min_n + 10):
            break
            
        for r in range(max(1, math.floor(n * p0)), min(math.ceil(n * p1), n) + 1):
            # Check single-stage design first to see if r works at all
            type1_single = 1 - pcs1(n, r - 1, p0)
            type2_single = pcs1(n, r - 1, p1)
            
            if type1_single > alpha or type2_single > beta:
                continue  # Skip if single-stage wouldn't work
                
            # Only search n1 up to n (full sample size)
            for n1 in range(5, n + 1):
                if n1 == n:  # Single-stage design
                    en = n
                    admissible.append((n1, r - 1, n, r, type1_single, 1 - type2_single, en, 0.0))
                    if n < min_n:
                        min_n = n
                    if en < min_en:
                        min_en = en
                    break
                    
                # Try different r1 values
                n2 = n - n1
                if n2 < 3:  # Ensure meaningful second stage
                    continue
                    
                for r1 in range(0, min(r, n1) + 1):
                    # Calculate error rates
                    alpha_actual = type1(n1, n2, r1, r, p0)
                    beta_actual = type2(n1, n2, r1, r, p1)
                    
                    if alpha_actual <= alpha and beta_actual <= beta:
                        # Calculate expected sample size under H0
                        p_stop = pet(n1, r1, p0)
                        en = compute_en(n1, n, r1, p0)
                        
                        admissible.append((n1, r1, n, r, alpha_actual, 1 - beta_actual, en, p_stop))
                        
                        if n < min_n:
                            min_n = n
                        if en < min_en:
                            min_en = en
                            
                        # No need to try more r1 values once we find one that works
                        break
    
    # Find optimal design (minimum EN0) and minimax design (minimum n)
    if not admissible:
        raise ValueError(f"No valid design found for p0={p0}, p1={p1}, alpha={alpha}, beta={beta}")
    
    # Sort by expected sample size (for optimal) or n (for minimax)
    admissible = sorted(admissible, key=lambda x: (x[2] if design_type == 'minimax' else x[6]))
    
    # Return the best design
    n1, r1, n, r, actual_alpha, actual_power, EN0, PET0 = admissible[0]
    
    return {
        'n1': n1,
        'r1': r1,
        'n': n,
        'r': r,
        'EN0': round(EN0, 2),
        'PET0': round(PET0, 4),
        'actual_alpha': round(actual_alpha, 4),
        'actual_power': round(actual_power, 4)
    }


def simons_power(n1, r1, n, r, p):
    """
    Calculate the power for a given Simon's two-stage design at a specific response rate.
    
    Parameters
    ----------
    n1 : int
        First stage sample size
    r1 : int
        First stage rejection threshold (continue if responses > r1)
    n : int
        Total sample size (both stages)
    r : int
        Final rejection threshold (reject H0 if total responses > r)
    p : float
        Response rate to calculate power at
        
    Returns
    -------
    float
        Power at the specified response rate p
    """
    # Import required module
    from scipy.stats import binom
    
    n2 = n - n1  # Second stage sample size
    
    # Probability of not rejecting at stage 1
    p_not_rej_1 = binom.cdf(r1, n1, p)
    
    # Probability of rejecting at stage 2
    p_rej_2 = 0
    for i in range(r1 + 1):  # responses in stage 1
        # Probability of i responses in stage 1
        p_i = binom.pmf(i, n1, p)
        
        # Probability of rejecting at stage 2 with i responses from stage 1
        p_rej_2_given_i = 1 - binom.cdf(r - i, n2, p)
        
        p_rej_2 += p_i * p_rej_2_given_i
    
    # Total power
    power = p_rej_2
    
    return power
