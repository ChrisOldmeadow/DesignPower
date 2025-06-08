"""
Permutation tests for cluster randomized controlled trials.

This module provides exact permutation inference methods specifically designed
for cluster RCTs with small numbers of clusters (5-15 per arm) where 
distributional assumptions may be questionable.

Features:
- Exact permutation tests for cluster-level data
- Multiple test statistics (mean difference, t-statistic, rank-based)
- Handles both continuous and binary outcomes
- Confidence intervals via permutation
- Robust to any distribution assumptions

References:
- Young, A. (2019). Channeling Fisher: Randomization tests and the statistical insignificance of seemingly significant experimental results
- Leyrat, C. et al. (2018). Cluster randomized trials with a small number of clusters: a review of methods
- Murray, D.M. (1998). Design and analysis of group-randomized trials
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Any, List, Tuple, Optional, Union
import warnings
from itertools import combinations
import math


def cluster_permutation_test(
    data: Union[pd.DataFrame, Dict[str, Any]], 
    cluster_col: str = 'cluster',
    treatment_col: str = 'treatment', 
    outcome_col: str = 'outcome',
    test_statistic: str = 'mean_difference',
    n_permutations: int = 10000,
    confidence_level: float = 0.95,
    alternative: str = 'two-sided',
    return_distribution: bool = False,
    random_seed: Optional[int] = None
) -> Dict[str, Any]:
    """
    Perform exact permutation test for cluster randomized trial.
    
    This function implements cluster randomization inference (CRI) following
    Young (2019) methodology, providing exact p-values without distributional
    assumptions.
    
    Parameters
    ----------
    data : pd.DataFrame or dict
        Either a DataFrame with individual-level data or a dict with cluster-level summaries.
        If DataFrame: columns should include cluster_col, treatment_col, outcome_col
        If dict: should contain 'control_clusters' and 'treatment_clusters' with outcome values
    cluster_col : str, optional
        Name of cluster identifier column, by default 'cluster'
    treatment_col : str, optional
        Name of treatment assignment column (0=control, 1=treatment), by default 'treatment'
    outcome_col : str, optional
        Name of outcome variable column, by default 'outcome'
    test_statistic : str, optional
        Test statistic to use: 'mean_difference', 't_statistic', 'rank_sum', by default 'mean_difference'
    n_permutations : int, optional
        Number of permutations (use 'exact' for all possible), by default 10000
    confidence_level : float, optional
        Confidence level for permutation CI, by default 0.95
    alternative : str, optional
        Alternative hypothesis: 'two-sided', 'greater', 'less', by default 'two-sided'
    return_distribution : bool, optional
        Whether to return full permutation distribution, by default False
    random_seed : int, optional
        Random seed for reproducibility, by default None
        
    Returns
    -------
    dict
        Dictionary containing:
        - p_value: Exact permutation p-value
        - observed_statistic: Observed test statistic
        - confidence_interval: Permutation-based CI for effect
        - n_permutations_used: Actual number of permutations performed
        - method: Description of test performed
        - distribution: Permutation distribution (if return_distribution=True)
        
    Examples
    --------
    >>> # Cluster-level data
    >>> cluster_data = {
    ...     'control_clusters': [0.45, 0.52, 0.41, 0.38, 0.49],
    ...     'treatment_clusters': [0.58, 0.61, 0.55, 0.63, 0.59]
    ... }
    >>> result = cluster_permutation_test(cluster_data)
    >>> print(f"P-value: {result['p_value']:.4f}")
    
    >>> # Individual-level DataFrame
    >>> df = pd.DataFrame({
    ...     'cluster': [1,1,1,2,2,2,3,3,3,4,4,4],
    ...     'treatment': [0,0,0,0,0,0,1,1,1,1,1,1], 
    ...     'outcome': [0,1,0,1,1,0,1,1,1,0,1,1]
    ... })
    >>> result = cluster_permutation_test(df)
    """
    
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Parse input data format
    if isinstance(data, dict):
        control_outcomes, treatment_outcomes = _parse_dict_data(data)
    elif isinstance(data, pd.DataFrame):
        control_outcomes, treatment_outcomes = _parse_dataframe_data(
            data, cluster_col, treatment_col, outcome_col
        )
    else:
        raise ValueError("Data must be pandas DataFrame or dictionary")
    
    # Validate cluster sizes
    n_control = len(control_outcomes)
    n_treatment = len(treatment_outcomes)
    total_clusters = n_control + n_treatment
    
    if total_clusters < 6:
        warnings.warn(
            f"Very few clusters (n={total_clusters}). Permutation test may have limited power.",
            UserWarning
        )
    
    # Combine all cluster outcomes
    all_outcomes = np.concatenate([control_outcomes, treatment_outcomes])
    
    # Calculate observed test statistic
    observed_stat = _calculate_test_statistic(
        control_outcomes, treatment_outcomes, test_statistic
    )
    
    # Determine if we can do exact permutation
    max_permutations = math.comb(total_clusters, n_control)
    
    if n_permutations == 'exact' or n_permutations >= max_permutations:
        # Exact permutation test
        permutation_stats = _exact_permutation_distribution(
            all_outcomes, n_control, test_statistic
        )
        n_perms_used = max_permutations
        method = f"Exact permutation test ({max_permutations} permutations)"
    else:
        # Monte Carlo permutation test
        permutation_stats = _monte_carlo_permutation_distribution(
            all_outcomes, n_control, n_permutations, test_statistic
        )
        n_perms_used = n_permutations
        method = f"Monte Carlo permutation test ({n_permutations} permutations)"
    
    # Calculate p-value
    p_value = _calculate_permutation_pvalue(
        observed_stat, permutation_stats, alternative
    )
    
    # Calculate permutation-based confidence interval
    confidence_interval = _permutation_confidence_interval(
        all_outcomes, n_control, confidence_level, test_statistic
    )
    
    # Prepare results
    results = {
        'p_value': p_value,
        'observed_statistic': observed_stat,
        'confidence_interval': confidence_interval,
        'n_permutations_used': n_perms_used,
        'method': method,
        'test_statistic': test_statistic,
        'alternative': alternative,
        'n_control_clusters': n_control,
        'n_treatment_clusters': n_treatment,
        'total_clusters': total_clusters
    }
    
    if return_distribution:
        results['distribution'] = permutation_stats
    
    # Add interpretation
    results['interpretation'] = _interpret_permutation_result(results)
    
    return results


def cluster_permutation_power(
    effect_size: float,
    control_mean: float,
    control_sd: float,
    n_control: int,
    n_treatment: int,
    cluster_size: int,
    test_statistic: str = 'mean_difference',
    n_simulations: int = 1000,
    n_permutations: int = 1000,
    alpha: float = 0.05,
    random_seed: Optional[int] = None
) -> Dict[str, Any]:
    """
    Estimate power of cluster permutation test via simulation.
    
    Parameters
    ----------
    effect_size : float
        True effect size (difference in cluster means)
    control_mean : float
        Mean outcome in control clusters
    control_sd : float
        Standard deviation of cluster means
    n_control : int
        Number of control clusters
    n_treatment : int
        Number of treatment clusters  
    cluster_size : int
        Number of individuals per cluster
    test_statistic : str, optional
        Test statistic to use, by default 'mean_difference'
    n_simulations : int, optional
        Number of simulation runs, by default 1000
    n_permutations : int, optional
        Number of permutations per test, by default 1000
    alpha : float, optional
        Significance level, by default 0.05
    random_seed : int, optional
        Random seed for reproducibility, by default None
        
    Returns
    -------
    dict
        Dictionary containing power estimate and simulation details
    """
    
    if random_seed is not None:
        np.random.seed(random_seed)
    
    treatment_mean = control_mean + effect_size
    significant_tests = 0
    
    p_values = []
    
    for sim in range(n_simulations):
        # Generate cluster-level outcomes
        control_outcomes = np.random.normal(control_mean, control_sd, n_control)
        treatment_outcomes = np.random.normal(treatment_mean, control_sd, n_treatment)
        
        # Create data dictionary
        data = {
            'control_clusters': control_outcomes,
            'treatment_clusters': treatment_outcomes
        }
        
        # Perform permutation test
        result = cluster_permutation_test(
            data=data,
            test_statistic=test_statistic,
            n_permutations=n_permutations,
            random_seed=None  # Let each test be random
        )
        
        p_values.append(result['p_value'])
        
        if result['p_value'] <= alpha:
            significant_tests += 1
    
    power = significant_tests / n_simulations
    
    return {
        'power': power,
        'effect_size': effect_size,
        'n_control_clusters': n_control,
        'n_treatment_clusters': n_treatment,
        'total_clusters': n_control + n_treatment,
        'cluster_size': cluster_size,
        'n_simulations': n_simulations,
        'n_permutations': n_permutations,
        'alpha': alpha,
        'test_statistic': test_statistic,
        'mean_p_value': np.mean(p_values),
        'p_value_distribution': p_values if n_simulations <= 100 else None
    }


def _parse_dict_data(data: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """Parse dictionary format cluster data."""
    try:
        control_outcomes = np.array(data['control_clusters'])
        treatment_outcomes = np.array(data['treatment_clusters'])
    except KeyError as e:
        raise ValueError(f"Dictionary must contain 'control_clusters' and 'treatment_clusters' keys. Missing: {e}")
    
    return control_outcomes, treatment_outcomes


def _parse_dataframe_data(
    df: pd.DataFrame, 
    cluster_col: str, 
    treatment_col: str, 
    outcome_col: str
) -> Tuple[np.ndarray, np.ndarray]:
    """Parse DataFrame format and compute cluster-level summaries."""
    
    # Validate columns exist
    required_cols = [cluster_col, treatment_col, outcome_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"DataFrame missing required columns: {missing_cols}")
    
    # Calculate cluster-level means
    cluster_summaries = df.groupby([cluster_col, treatment_col])[outcome_col].mean().reset_index()
    
    # Separate by treatment group
    control_data = cluster_summaries[cluster_summaries[treatment_col] == 0]
    treatment_data = cluster_summaries[cluster_summaries[treatment_col] == 1]
    
    if len(control_data) == 0 or len(treatment_data) == 0:
        raise ValueError("Must have clusters in both treatment groups")
    
    control_outcomes = control_data[outcome_col].values
    treatment_outcomes = treatment_data[outcome_col].values
    
    return control_outcomes, treatment_outcomes


def _calculate_test_statistic(
    control_outcomes: np.ndarray,
    treatment_outcomes: np.ndarray, 
    test_statistic: str
) -> float:
    """Calculate the specified test statistic."""
    
    if test_statistic == 'mean_difference':
        return np.mean(treatment_outcomes) - np.mean(control_outcomes)
    
    elif test_statistic == 't_statistic':
        # Unpaired t-statistic
        n1, n2 = len(control_outcomes), len(treatment_outcomes)
        mean1, mean2 = np.mean(control_outcomes), np.mean(treatment_outcomes)
        var1, var2 = np.var(control_outcomes, ddof=1), np.var(treatment_outcomes, ddof=1)
        
        pooled_se = np.sqrt(var1/n1 + var2/n2)
        if pooled_se == 0:
            return 0
        return (mean2 - mean1) / pooled_se
    
    elif test_statistic == 'rank_sum':
        # Mann-Whitney U statistic
        from scipy.stats import mannwhitneyu
        try:
            statistic, _ = mannwhitneyu(treatment_outcomes, control_outcomes, alternative='two-sided')
            return statistic
        except ValueError:
            # Handle ties or identical distributions
            return len(control_outcomes) * len(treatment_outcomes) / 2
    
    else:
        raise ValueError(f"Unknown test statistic: {test_statistic}")


def _exact_permutation_distribution(
    all_outcomes: np.ndarray,
    n_control: int,
    test_statistic: str
) -> np.ndarray:
    """Generate exact permutation distribution."""
    
    total_clusters = len(all_outcomes)
    permutation_stats = []
    
    # Generate all possible combinations
    for control_indices in combinations(range(total_clusters), n_control):
        control_mask = np.zeros(total_clusters, dtype=bool)
        control_mask[list(control_indices)] = True
        
        perm_control = all_outcomes[control_mask]
        perm_treatment = all_outcomes[~control_mask]
        
        stat = _calculate_test_statistic(perm_control, perm_treatment, test_statistic)
        permutation_stats.append(stat)
    
    return np.array(permutation_stats)


def _monte_carlo_permutation_distribution(
    all_outcomes: np.ndarray,
    n_control: int,
    n_permutations: int,
    test_statistic: str
) -> np.ndarray:
    """Generate Monte Carlo permutation distribution."""
    
    total_clusters = len(all_outcomes)
    permutation_stats = []
    
    for _ in range(n_permutations):
        # Random permutation
        perm_indices = np.random.permutation(total_clusters)
        perm_control = all_outcomes[perm_indices[:n_control]]
        perm_treatment = all_outcomes[perm_indices[n_control:]]
        
        stat = _calculate_test_statistic(perm_control, perm_treatment, test_statistic)
        permutation_stats.append(stat)
    
    return np.array(permutation_stats)


def _calculate_permutation_pvalue(
    observed_stat: float,
    permutation_stats: np.ndarray,
    alternative: str
) -> float:
    """Calculate permutation p-value."""
    
    n_perms = len(permutation_stats)
    
    if alternative == 'two-sided':
        # Two-sided: count permutations as or more extreme than observed
        more_extreme = np.sum(np.abs(permutation_stats) >= np.abs(observed_stat))
        p_value = more_extreme / n_perms
        
    elif alternative == 'greater':
        # One-sided: treatment > control
        more_extreme = np.sum(permutation_stats >= observed_stat)
        p_value = more_extreme / n_perms
        
    elif alternative == 'less':
        # One-sided: treatment < control
        more_extreme = np.sum(permutation_stats <= observed_stat)
        p_value = more_extreme / n_perms
        
    else:
        raise ValueError(f"Unknown alternative: {alternative}")
    
    return p_value


def _permutation_confidence_interval(
    all_outcomes: np.ndarray,
    n_control: int,
    confidence_level: float,
    test_statistic: str,
    max_permutations: int = 5000
) -> Tuple[float, float]:
    """Calculate permutation-based confidence interval for effect size."""
    
    # For computational efficiency, limit permutations for CI
    total_clusters = len(all_outcomes)
    max_exact = math.comb(total_clusters, n_control)
    
    if max_exact <= max_permutations:
        # Use exact permutation
        permutation_stats = _exact_permutation_distribution(all_outcomes, n_control, test_statistic)
    else:
        # Use Monte Carlo
        permutation_stats = _monte_carlo_permutation_distribution(
            all_outcomes, n_control, max_permutations, test_statistic
        )
    
    # Calculate percentiles for CI
    alpha = 1 - confidence_level
    lower_percentile = (alpha/2) * 100
    upper_percentile = (1 - alpha/2) * 100
    
    ci_lower = np.percentile(permutation_stats, lower_percentile)
    ci_upper = np.percentile(permutation_stats, upper_percentile)
    
    return (ci_lower, ci_upper)


def _interpret_permutation_result(result: Dict[str, Any]) -> str:
    """Generate interpretation text for permutation test result."""
    
    p_val = result['p_value']
    n_clusters = result['total_clusters']
    effect = result['observed_statistic']
    ci = result['confidence_interval']
    
    # Significance interpretation
    if p_val <= 0.001:
        sig_text = "highly significant (p ≤ 0.001)"
    elif p_val <= 0.01:
        sig_text = "significant (p ≤ 0.01)"
    elif p_val <= 0.05:
        sig_text = "significant (p ≤ 0.05)"
    elif p_val <= 0.10:
        sig_text = "marginally significant (p ≤ 0.10)"
    else:
        sig_text = "not significant (p > 0.10)"
    
    # Effect direction
    if effect > 0:
        direction = "favors treatment"
    elif effect < 0:
        direction = "favors control"
    else:
        direction = "shows no difference"
    
    # Sample size assessment
    if n_clusters < 10:
        size_note = "Very small cluster trial - exact permutation test provides valid inference."
    elif n_clusters < 20:
        size_note = "Small cluster trial - permutation test robust to distributional assumptions."
    else:
        size_note = "Permutation test provides distribution-free inference."
    
    interpretation = (
        f"Permutation test result: {sig_text} (p = {p_val:.4f}). "
        f"Observed effect {direction} (effect = {effect:.3f}, "
        f"95% CI: [{ci[0]:.3f}, {ci[1]:.3f}]). "
        f"{size_note}"
    )
    
    return interpretation


# Convenience functions for specific outcome types

def cluster_permutation_test_binary(
    control_clusters: List[float],
    treatment_clusters: List[float],
    **kwargs
) -> Dict[str, Any]:
    """
    Convenience function for binary outcome cluster permutation test.
    
    Parameters
    ----------
    control_clusters : list
        List of success rates for control clusters
    treatment_clusters : list  
        List of success rates for treatment clusters
    **kwargs
        Additional arguments passed to cluster_permutation_test
        
    Returns
    -------
    dict
        Permutation test results
    """
    data = {
        'control_clusters': control_clusters,
        'treatment_clusters': treatment_clusters
    }
    
    return cluster_permutation_test(data, **kwargs)


def cluster_permutation_test_continuous(
    control_clusters: List[float],
    treatment_clusters: List[float],
    **kwargs
) -> Dict[str, Any]:
    """
    Convenience function for continuous outcome cluster permutation test.
    
    Parameters
    ----------
    control_clusters : list
        List of mean outcomes for control clusters
    treatment_clusters : list
        List of mean outcomes for treatment clusters  
    **kwargs
        Additional arguments passed to cluster_permutation_test
        
    Returns
    -------
    dict
        Permutation test results
    """
    data = {
        'control_clusters': control_clusters,
        'treatment_clusters': treatment_clusters
    }
    
    return cluster_permutation_test(data, **kwargs)