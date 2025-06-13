"""
Analytical methods for binary outcomes in parallel group randomized controlled trials.

This module provides analytical (closed-form) functions for power analysis and
sample size calculation for parallel group RCTs with binary outcomes.
"""

import numpy as np
import math
from scipy import stats

def fishers_exact_computational_guidance(n1, n2):
    """
    Provide guidance on computational complexity for Fisher's exact test power calculation.
    
    Parameters
    ----------
    n1, n2 : int
        Sample sizes for the two groups
        
    Returns
    -------
    dict
        Dictionary with computational complexity information and recommendations
    """
    max_computations = (n1 + 1) * (n2 + 1)
    
    if max_computations <= 10000:
        complexity = "fast"
        recommendation = "Use exact calculation"
        estimated_time = "< 1 second"
    elif max_computations <= 100000:
        complexity = "moderate" 
        recommendation = "Use exact calculation (may take a few seconds)"
        estimated_time = "1-10 seconds"
    else:
        complexity = "slow"
        recommendation = "Use normal approximation or reduce sample sizes"
        estimated_time = f"> 30 seconds ({max_computations:,} computations)"
    
    return {
        "n1": n1,
        "n2": n2,
        "max_computations": max_computations,
        "complexity": complexity,
        "recommendation": recommendation,
        "estimated_time": estimated_time
    }

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
        return 1.0  # No difference can be detected
    
    # Calculate test statistic
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
    p_pooled = (s1 + s2) / (n1 + n2) if (n1 + n2) > 0 else 0
    
    # Calculate log-likelihoods
    ll_null = 0
    if p_pooled > 0 and p_pooled < 1:
        ll_null += s1 * math.log(p_pooled) + (n1 - s1) * math.log(1 - p_pooled)
        ll_null += s2 * math.log(p_pooled) + (n2 - s2) * math.log(1 - p_pooled)
    
    ll_alt = 0
    if p1 > 0 and p1 < 1:
        ll_alt += s1 * math.log(p1) + (n1 - s1) * math.log(1 - p1)
    if p2 > 0 and p2 < 1:
        ll_alt += s2 * math.log(p2) + (n2 - s2) * math.log(1 - p2)
    
    # Calculate test statistic (2 * log likelihood ratio)
    lrt = 2 * (ll_alt - ll_null)
    
    # Calculate p-value (chi-squared distribution with 1 df)
    p_value = 1 - stats.chi2.cdf(lrt, 1)
    
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
    from scipy.stats import fisher_exact
    
    # Create contingency table
    # Standard 2x2 format: rows are groups, columns are success/failure
    # [[group1_success, group1_failure],
    #  [group2_success, group2_failure]]
    table = [[s1, n1 - s1], [s2, n2 - s2]]
    
    # Calculate p-value
    odds_ratio, p_value = fisher_exact(table)
    
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
    # Normalize test_type to lowercase for case-insensitive comparison
    test_type = test_type.lower().replace(" ", "_")
    
    if test_type == "normal_approximation":
        return normal_approximation_test(n1, n2, s1, s2)
    elif test_type == "likelihood_ratio":
        return likelihood_ratio_test(n1, n2, s1, s2)
    elif test_type in ["fishers_exact", "exact_test"]:
        return fishers_exact_test(n1, n2, s1, s2)
    else:
        raise ValueError(f"Unknown test type: {test_type}")

# ===== Power and Sample Size Functions =====

def power_binary(n1, n2, p1, p2, alpha=0.05, test_type="normal approximation", correction=False):
    """
    Calculate power for binary outcomes using analytical approaches with support for different test types.
    
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
        Type of test to use: "normal approximation", "fishers exact", or "likelihood ratio".
        Default is "normal approximation".
        
        For "fishers exact":
        - Uses exact calculation for small/moderate samples (≤ 316 per group)
        - Falls back to conservative normal approximation for large samples
        - Returns additional fields: calculation_method, computations (if exact)
        
    correction : bool, optional
        Whether to apply continuity correction, by default False
    
    Returns
    -------
    dict
        Dictionary containing power estimate and parameters.
        
        For Fisher's exact test, additional fields:
        - calculation_method: "exact" or "normal_approximation"
        - computations: number of computations (if exact method used)
        
    Notes
    -----
    Fisher's exact test computational complexity:
    - Fast (< 1s): n1, n2 ≤ 100 each
    - Moderate (1-10s): n1, n2 ≤ 316 each  
    - Slow (> 30s): larger samples (uses normal approximation)
    
    Use fishers_exact_computational_guidance(n1, n2) for detailed guidance.
    """
    # Validate inputs
    if not 0 <= p1 <= 1 or not 0 <= p2 <= 1:
        raise ValueError("Proportions must be between 0 and 1")
    
    if n1 <= 0 or n2 <= 0:
        raise ValueError("Sample sizes must be positive")
    
    # Normalize test_type to lowercase for case-insensitive comparison
    test_type = test_type.lower()
    
    # Calculate effect size
    effect_size = abs(p1 - p2)
    
    # Power calculation based on test type
    if test_type == "fishers exact":
        # Fisher's exact test power calculation
        # Use exact calculation for small samples, approximation for large samples
        
        n_total = n1 + n2
        max_computations = (n1 + 1) * (n2 + 1)
        
        # Computational complexity guidelines:
        # - Fast: < 10,000 computations (n1, n2 ≤ 100)
        # - Moderate: < 100,000 computations (n1, n2 ≤ 316) 
        # - Slow: > 100,000 computations
        
        if max_computations <= 100000:  # Use exact calculation
            from scipy.stats import binom, fisher_exact
            
            power = 0.0
            for s1 in range(n1 + 1):
                for s2 in range(n2 + 1):
                    # Probability of this outcome under alternative hypothesis
                    prob = binom.pmf(s1, n1, p1) * binom.pmf(s2, n2, p2)
                    
                    # Perform Fisher's exact test
                    table = [[s1, n1 - s1], [s2, n2 - s2]]
                    _, p_value = fisher_exact(table)
                    
                    # Add to power if significant
                    if p_value < alpha:
                        power += prob
            
            # Store calculation method info
            calculation_method = "exact"
            
        else:  # Use normal approximation for large samples
            # Calculate pooled proportion
            p_pooled = (p1 * n1 + p2 * n2) / (n1 + n2)
            
            # Calculate standard error
            se = math.sqrt(p_pooled * (1 - p_pooled) * (1/n1 + 1/n2))
            
            # Calculate critical value for two-sided test
            z_alpha_sided = stats.norm.ppf(1 - alpha/2)
            
            # Calculate standard power for two-sided test
            es_on_se = effect_size / se
            power = stats.norm.cdf(es_on_se - z_alpha_sided) + stats.norm.cdf(-es_on_se - z_alpha_sided)
            
            # Note: For large samples, normal approximation approaches Fisher's exact test
            # No arbitrary adjustment factors are applied - use the mathematical result as-is
            
            calculation_method = "normal_approximation"
        
    elif test_type == "likelihood_ratio":
        # For likelihood ratio test, use standard asymptotic approximation
        # Note: Exact likelihood ratio power calculation requires more complex methods
        
        # Calculate pooled proportion
        p_pooled = (p1 * n1 + p2 * n2) / (n1 + n2)
        
        # Calculate standard error
        se = math.sqrt(p_pooled * (1 - p_pooled) * (1/n1 + 1/n2))
        
        # Calculate critical value
        z_alpha = stats.norm.ppf(1 - alpha/2)
        
        # Calculate power using standard normal approximation
        z_beta = (effect_size / se - z_alpha)
        power = stats.norm.cdf(z_beta)
        
    else:  # Default to normal approximation
        # Calculate pooled proportion
        p_pooled = (p1 * n1 + p2 * n2) / (n1 + n2)
        
        # Calculate standard error
        se = math.sqrt(p_pooled * (1 - p_pooled) * (1/n1 + 1/n2))
        
        # Apply continuity correction if requested
        if correction:
            # Adjust effect size by continuity correction
            correction_value = 0.5 * (1/n1 + 1/n2)
            adjusted_effect_size = max(0, effect_size - correction_value)
            
            # Calculate critical value for two-sided test
            z_alpha_sided = stats.norm.ppf(1 - alpha/2)
            
            # Calculate power with correction for two-sided test
            adj_es_on_se = adjusted_effect_size / se
            power = stats.norm.cdf(adj_es_on_se - z_alpha_sided) + stats.norm.cdf(-adj_es_on_se - z_alpha_sided)
        else:
            # Calculate critical value for two-sided test
            z_alpha_sided = stats.norm.ppf(1 - alpha/2)
            
            # Calculate standard power for two-sided test
            es_on_se = effect_size / se
            power = stats.norm.cdf(es_on_se - z_alpha_sided) + stats.norm.cdf(-es_on_se - z_alpha_sided)
    
    # Ensure power is between 0 and 1
    power = max(0, min(1, power))
    
    # Build result dictionary
    result = {
        "power": power,
        "n1": n1,
        "n2": n2,
        "p1": p1,
        "p2": p2,
        "effect_size": effect_size,
        "alpha": alpha,
        "test_type": test_type,
        "correction": correction,
        "method": "analytical"
    }
    
    # Add calculation method info for Fisher's exact test
    if test_type == "fishers exact":
        result["calculation_method"] = calculation_method
        if calculation_method == "exact":
            result["computations"] = max_computations
        
    return result

def sample_size_binary(p1, p2, power=0.8, alpha=0.05, allocation_ratio=1.0, test_type="normal approximation", correction=False):
    """
    Calculate sample size for binary outcome in parallel design with support for different test types.
    
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
        Type of test to use: "normal approximation", "fishers exact", or "likelihood ratio".
        Default is "normal approximation".
    correction : bool, optional
        Whether to apply continuity correction, by default False
        
    Returns
    -------
    dict
        Dictionary containing sample sizes and parameters
    """
    # Validate inputs
    if not 0 <= p1 <= 1 or not 0 <= p2 <= 1:
        raise ValueError("Proportions must be between 0 and 1")
    
    if allocation_ratio <= 0:
        raise ValueError("Allocation ratio must be positive")
    
    # Normalize test_type to lowercase for case-insensitive comparison
    test_type = test_type.lower()
    
    # Get critical values
    z_alpha = stats.norm.ppf(1 - alpha/2)
    z_beta = stats.norm.ppf(power)
    
    # Calculate average (pooled) proportion for some formulas
    p_avg = (p1 + allocation_ratio * p2) / (1 + allocation_ratio)
    
    # Sample size calculation based on test type
    if test_type == "fishers exact" or test_type == "fishers_exact":
        # For Fisher's Exact test, we need a more conservative estimate
        # The inflation factor depends on the expected sample size
        
        # Base calculation using normal approximation
        numerator = (z_alpha + z_beta)**2 * (p1 * (1 - p1) + p2 * (1 - p2) / allocation_ratio)
        denominator = (p1 - p2)**2
        n1_base = numerator / denominator
        
        # Estimate total sample size for adjustment
        n_total_est = n1_base * (1 + allocation_ratio)
        
        # For Fisher's exact test with small to moderate samples, use the normal approximation
        # as the starting point. The exact test will be used at analysis time.
        # No arbitrary adjustment factors - use the mathematically derived result
        n1 = math.ceil(n1_base)
        
    elif test_type == "likelihood_ratio":
        # For likelihood ratio test, use a slightly different formula
        # This is an approximation as exact formulas for likelihood ratio tests are complex
        p_diff = abs(p1 - p2)
        pooled_var = p_avg * (1 - p_avg) * (1 + 1/allocation_ratio)
        
        n1 = math.ceil((z_alpha + z_beta)**2 * pooled_var / (p_diff**2))
        
    else:  # Default to normal approximation
        # Calculate n1 using the normal approximation formula
        numerator = (z_alpha + z_beta)**2 * (p1 * (1 - p1) + p2 * (1 - p2) / allocation_ratio)
        denominator = (p1 - p2)**2
        n1 = numerator / denominator
        
        # Apply continuity correction if requested
        if correction:
            # Standard continuity correction adds 0.5 to each cell in the 2x2 table
            # This modifies the effect size calculation, following established methods
            # Reference: Fleiss, J.L. et al. (2003). Statistical Methods for Rates and Proportions
            corrected_p1 = (p1 * n1 + 0.5) / (n1 + 1)
            corrected_p2 = (p2 * n1 + 0.5) / (n1 + 1)  # Approximate with same n
            effect_diff = abs(corrected_p1 - corrected_p2)
            # Recalculate with corrected effect size - no arbitrary multipliers
            numerator = (z_alpha + z_beta)**2 * (corrected_p1 * (1 - corrected_p1) + corrected_p2 * (1 - corrected_p2) / allocation_ratio)
            denominator = effect_diff**2
            n1 = numerator / denominator
            
        n1 = math.ceil(n1)
    
    # Calculate n2 based on allocation ratio
    n2 = math.ceil(n1 * allocation_ratio)
    
    # Return results
    return {
        "sample_size_1": n1,
        "sample_size_2": n2,
        "total_sample_size": n1 + n2,
        "p1": p1,
        "p2": p2,
        "power": power,
        "alpha": alpha,
        "allocation_ratio": allocation_ratio,
        "test_type": test_type,
        "correction": correction,
        "method": "analytical"
    }

def sample_size_binary_non_inferiority(p1, non_inferiority_margin, power=0.8, alpha=0.05, 
                                     allocation_ratio=1.0, assumed_difference=0.0, direction="lower",
                                     continuity_correction=False):
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
    continuity_correction : bool, optional
        Whether to apply continuity correction, by default False
    
    Returns
    -------
    dict
        Dictionary containing the required sample sizes
    """
    # Validate inputs
    if not 0 <= p1 <= 1:
        raise ValueError("Proportion must be between 0 and 1")
    
    if non_inferiority_margin <= 0:
        raise ValueError("Non-inferiority margin must be positive")
    
    if allocation_ratio <= 0:
        raise ValueError("Allocation ratio must be positive")
    
    # Calculate the expected p2 based on assumed_difference
    p2 = p1 + assumed_difference
    
    # Ensure p2 is valid
    if not 0 <= p2 <= 1:
        raise ValueError("The resulting p2 based on assumed difference is not between 0 and 1")
    
    # Get critical values (one-sided alpha for non-inferiority)
    z_alpha = stats.norm.ppf(1 - alpha)
    z_beta = stats.norm.ppf(power)
    
    # Use simpler pooled variance approach for better accuracy with literature
    # This matches the standard formulation in Chow & Liu and other references
    if direction == "lower":
        # For lower non-inferiority, we're testing p2 >= p1 - margin
        effect = non_inferiority_margin  # Detectable difference under null
    else:  # upper
        # For upper non-inferiority, we're testing p2 <= p1 + margin  
        effect = non_inferiority_margin  # Detectable difference under null
    
    # Pooled variance approach (standard in most references)
    # Assumes equal variance for both groups under null hypothesis
    var_pooled = p1 * (1 - p1) + p2 * (1 - p2) / allocation_ratio
    
    # Calculate n1 using the standard formula
    numerator = (z_alpha + z_beta)**2 * var_pooled
    denominator = effect**2
    
    # Handle case where denominator is zero
    if denominator == 0:
        raise ValueError("Cannot calculate sample size when effect size is zero")
    
    n1_basic = numerator / denominator
    
    # Apply continuity correction if requested
    if continuity_correction:
        # Add continuity correction term
        correction_term = (z_alpha + z_beta) / (2 * effect)
        n1_corrected = (numerator + correction_term) / denominator
        n1 = math.ceil(n1_corrected)
    else:
        n1 = math.ceil(n1_basic)
    
    n2 = math.ceil(n1 * allocation_ratio)
    
    # Return results
    return {
        "sample_size_1": n1,
        "sample_size_2": n2,
        "total_sample_size": n1 + n2,
        "p1": p1,
        "p2": p2,
        "non_inferiority_margin": non_inferiority_margin,
        "assumed_difference": assumed_difference,
        "direction": direction,
        "power": power,
        "alpha": alpha,
        "allocation_ratio": allocation_ratio,
        "continuity_correction": continuity_correction,
        "method": "analytical"
    }

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
    # Validate inputs
    if not 0 <= p1 <= 1:
        raise ValueError("Proportion must be between 0 and 1")
    
    if non_inferiority_margin <= 0:
        raise ValueError("Non-inferiority margin must be positive")
    
    # Calculate p2 from assumed difference
    p2 = p1 + assumed_difference
    
    # Ensure p2 is valid
    if not 0 <= p2 <= 1:
        raise ValueError("The resulting p2 based on assumed difference is not between 0 and 1")
    
    # Calculate critical value (one-sided alpha for non-inferiority)
    z_alpha = stats.norm.ppf(1 - alpha)
    
    # Calculate test statistics based on direction
    if direction == "lower":
        # For lower non-inferiority, we're testing p2 >= p1 - margin
        # Null: p2 = p1 - margin, Alt: p2 > p1 - margin
        p2_null = p1 - non_inferiority_margin
        se = math.sqrt(p1 * (1 - p1) / n1 + p2_null * (1 - p2_null) / n2)
        z_crit = -z_alpha  # Left tail critical value
        
        # Expected z-statistic under the alternative
        se_alt = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
        z_exp = (p2 - p2_null) / se_alt
        
        # Calculate power
        power = stats.norm.cdf(z_exp - z_crit)
        
    else:  # upper
        # For upper non-inferiority, we're testing p2 <= p1 + margin
        # Null: p2 = p1 + margin, Alt: p2 < p1 + margin
        p2_null = p1 + non_inferiority_margin
        se = math.sqrt(p1 * (1 - p1) / n1 + p2_null * (1 - p2_null) / n2)
        z_crit = z_alpha  # Right tail critical value
        
        # Expected z-statistic under the alternative
        se_alt = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
        z_exp = (p2 - p2_null) / se_alt
        
        # Calculate power
        power = stats.norm.cdf(z_crit - z_exp)
    
    # Return results
    return {
        "power": power,
        "p1": p1,
        "p2": p2,
        "n1": n1,
        "n2": n2,
        "non_inferiority_margin": non_inferiority_margin,
        "assumed_difference": assumed_difference,
        "direction": direction,
        "alpha": alpha,
        "method": "analytical"
    }


def min_detectable_effect_binary(n1, n2, p1, power=0.8, alpha=0.05, precision=0.01, test_type="normal_approximation"):
    """
    Calculate minimum detectable effect for binary outcomes using analytical approach.
    
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
    alpha : float, optional
        Significance level, by default 0.05
    precision : float, optional
        Desired precision for the effect size, by default 0.01
    test_type : str, optional
        Type of statistical test to use, by default "normal_approximation"
    
    Returns
    -------
    dict
        Dictionary containing the minimum detectable difference in proportions
    """
    # Validate inputs
    if n1 <= 0 or n2 <= 0:
        raise ValueError("Sample sizes must be positive integers")
    
    if p1 < 0 or p1 > 1:
        raise ValueError("Proportion must be between 0 and 1")
    
    if power <= 0 or power >= 1:
        raise ValueError("Power must be between 0 and 1")
    
    if alpha <= 0 or alpha >= 1:
        raise ValueError("Alpha must be between 0 and 1")
    
    # Calculate critical values
    z_alpha = stats.norm.ppf(1 - alpha/2)  # Two-sided
    z_beta = stats.norm.ppf(power)
    
    # Determine direction based on p1
    # If p1 is small, look for an increase (positive effect)
    # If p1 is large, look for a decrease (negative effect)
    if p1 <= 0.5:
        direction = 1  # Positive direction (searching for an increase)
    else:
        direction = -1  # Negative direction (searching for a decrease)
    
    # Binary search to find the minimum detectable effect
    # Initialize search range
    if direction == 1:
        low = precision  # Smallest positive difference
        high = 1.0 - p1 - precision  # Largest possible increase
    else:
        low = precision  # Smallest positive difference
        high = p1 - precision  # Largest possible decrease
    
    current_diff = (low + high) / 2
    iterations = 0
    max_iterations = 20  # Prevent infinite loops
    
    while (high - low) > precision and iterations < max_iterations:
        iterations += 1
        
        # Calculate p2 based on current difference
        p2 = p1 + direction * current_diff
        
        # Ensure p2 is within valid range [0,1]
        p2 = max(0, min(1, p2))
        
        # Calculate standard error for the difference in proportions
        pooled_p = ((p1 * n1) + (p2 * n2)) / (n1 + n2)
        se_diff = math.sqrt(pooled_p * (1 - pooled_p) * (1/n1 + 1/n2))
        
        # Calculate effect size
        effect_size = abs(p2 - p1) / se_diff
        
        # Calculate achieved power based on effect size
        # Power = Φ(effect_size - z_alpha) where Φ is the standard normal CDF
        achieved_power = stats.norm.cdf(effect_size - z_alpha)
        
        # Binary search: adjust the effect size based on achieved power
        if achieved_power < power:
            # Effect too small, increase it
            low = current_diff
        else:
            # Effect sufficient, try to decrease it
            high = current_diff
        
        # Set new midpoint for next iteration
        current_diff = (low + high) / 2
    
    # Calculate final p2 and effect measures
    p2 = p1 + direction * current_diff
    p2 = max(0, min(1, p2))  # Ensure p2 is valid
    
    # For relative risk and odds ratio
    relative_risk = p2 / p1 if p1 > 0 else float('inf')
    odds1 = p1 / (1 - p1) if p1 < 1 else float('inf')
    odds2 = p2 / (1 - p2) if p2 < 1 else float('inf')
    odds_ratio = odds2 / odds1 if odds1 > 0 else float('inf')
    
    # Return results
    return {
        "minimum_detectable_p2": p2,
        "minimum_detectable_difference": abs(p2 - p1),
        "direction": "increase" if direction > 0 else "decrease",
        "p1": p1,
        "n1": n1,
        "n2": n2,
        "power": power,
        "alpha": alpha,
        "relative_risk": relative_risk,
        "odds_ratio": odds_ratio,
        "iterations": iterations,
        "method": "analytical"
    }
