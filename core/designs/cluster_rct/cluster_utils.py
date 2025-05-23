"""Utility functions for cluster randomized controlled trials.

This module provides common utility functions used across different types
of cluster randomized controlled trials, including functions for calculating
design effects with various assumptions.
"""

import math
import numpy as np
from scipy import stats


def design_effect_equal(cluster_size, icc):
    """
    Calculate the design effect for equal cluster sizes.
    
    Parameters
    ----------
    cluster_size : int or float
        Average number of individuals per cluster
    icc : float
        Intracluster correlation coefficient
    
    Returns
    -------
    float
        Design effect
    """
    return 1 + (cluster_size - 1) * icc


def design_effect_unequal(cluster_size, icc, cv=None, cluster_sizes=None):
    """
    Calculate the design effect accounting for unequal cluster sizes.
    
    Parameters
    ----------
    cluster_size : int or float
        Average number of individuals per cluster
    icc : float
        Intracluster correlation coefficient
    cv : float, optional
        Coefficient of variation of cluster sizes.
        If None, it will be calculated from cluster_sizes if provided.
    cluster_sizes : list or array, optional
        List of cluster sizes. If provided and cv is None, 
        the cv will be calculated from these values.
    
    Returns
    -------
    float
        Design effect adjusted for unequal cluster sizes
    """
    # Calculate CV if not provided but cluster sizes are
    if cv is None and cluster_sizes is not None:
        cv = np.std(cluster_sizes) / np.mean(cluster_sizes)
    elif cv is None:
        cv = 0  # Default to equal cluster sizes
    
    # Calculate design effect with CV adjustment
    return 1 + ((1 + cv**2) * cluster_size - 1) * icc


def convert_icc_logit_to_linear(icc_logit, p):
    """
    Convert ICC from logit scale to linear scale for binary outcomes.
    
    For binary outcomes, ICCs are sometimes reported on the logit scale.
    This function converts from logit to linear scale.
    
    Parameters
    ----------
    icc_logit : float
        ICC on the logit scale
    p : float
        Overall prevalence/proportion
    
    Returns
    -------
    float
        ICC on the linear scale
    """
    # Variance on linear scale
    var_linear = p * (1 - p)
    
    # Variance on logit scale
    var_logit = math.pi**2 / 3
    
    # Convert ICC using variance relationship
    icc_linear = icc_logit * (var_logit / var_linear)
    
    # Ensure ICC is within valid range [0, 1]
    return min(max(icc_linear, 0), 1)


def convert_icc_linear_to_logit(icc_linear, p):
    """
    Convert ICC from linear scale to logit scale for binary outcomes.
    
    Parameters
    ----------
    icc_linear : float
        ICC on the linear scale
    p : float
        Overall prevalence/proportion
    
    Returns
    -------
    float
        ICC on the logit scale
    """
    # Variance on linear scale
    var_linear = p * (1 - p)
    
    # Variance on logit scale
    var_logit = math.pi**2 / 3
    
    # Convert ICC using variance relationship
    icc_logit = icc_linear * (var_linear / var_logit)
    
    # Ensure ICC is within valid range [0, 1]
    return min(max(icc_logit, 0), 1)


def validate_cluster_parameters(n_clusters, min_recommended=40):
    """
    Validate cluster parameters and provide warnings for small numbers of clusters.
    
    Parameters
    ----------
    n_clusters : int
        Total number of clusters across all arms
    min_recommended : int, optional
        Minimum recommended number of clusters, by default 40
    
    Returns
    -------
    dict
        Dictionary with validation results including warnings if applicable
    """
    warnings = []
    
    if n_clusters < min_recommended:
        warnings.append(f"The total number of clusters ({n_clusters}) is below the "
                       f"recommended minimum ({min_recommended}). Statistical inference "
                       f"may not be reliable. Consider increasing the number of clusters "
                       f"or using methods specifically designed for small numbers of clusters.")
    
    if n_clusters < 30:
        warnings.append("With fewer than 30 total clusters, consider using permutation tests "
                       "or small-sample corrections in the analysis.")
    
    if n_clusters < 20:
        warnings.append("CAUTION: With fewer than 20 total clusters, the risk of Type I error "
                       "inflation is substantial. Results should be interpreted with extreme caution.")
    
    return {
        "valid": n_clusters >= min_recommended,
        "n_clusters": n_clusters,
        "min_recommended": min_recommended,
        "warnings": warnings
    }


def convert_effect_measures(p1, measure_type, measure_value):
    """
    Convert between different effect size measures for binary outcomes.
    
    Parameters
    ----------
    p1 : float
        Proportion in control group
    measure_type : str
        Type of effect measure: 'risk_difference', 'risk_ratio', or 'odds_ratio'
    measure_value : float
        Value of the effect measure
    
    Returns
    -------
    dict
        Dictionary with all effect measures
    """
    if measure_type == 'risk_difference':
        p2 = p1 + measure_value
    elif measure_type == 'risk_ratio':
        p2 = p1 * measure_value
    elif measure_type == 'odds_ratio':
        odds1 = p1 / (1 - p1)
        odds2 = odds1 * measure_value
        p2 = odds2 / (1 + odds2)
    else:
        raise ValueError("measure_type must be one of: 'risk_difference', 'risk_ratio', 'odds_ratio'")
    
    # Ensure p2 is within valid range [0, 1]
    p2 = min(max(p2, 0), 1)
    
    # Calculate all effect measures
    risk_difference = p2 - p1
    risk_ratio = p2 / p1 if p1 > 0 else float('inf')
    odds_ratio = (p2 / (1 - p2)) / (p1 / (1 - p1)) if 0 < p1 < 1 and 0 < p2 < 1 else float('inf')
    
    return {
        'p1': p1,
        'p2': p2,
        'risk_difference': risk_difference,
        'risk_ratio': risk_ratio,
        'odds_ratio': odds_ratio
    }
