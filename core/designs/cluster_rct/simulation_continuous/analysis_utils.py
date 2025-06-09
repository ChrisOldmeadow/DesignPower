"""
Analysis utility functions for cluster randomized controlled trials with continuous outcomes.

This module provides helper functions for statistical analysis and data processing.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


def _ols_cluster_test(df):
    """Return t-value and p-value for treatment using OLS with cluster robust SE."""
    try:
        ols = sm.OLS(df["y"], sm.add_constant(df[["treatment"]])).fit(
            cov_type="cluster", cov_kwds={"groups": df["cluster"]}
        )
        tval = ols.tvalues["treatment"]
        pval = 2 * stats.t.sf(abs(tval), df=max(1, df["cluster"].nunique() - 2))
        return tval, pval, {"converged": True, "model": "ols_cluster"}
    except Exception as e:
        return np.nan, 1.0, {"converged": False, "model": "ols_cluster", "error": str(e)}


def _analyze_continuous_trial(df, icc=0.05, analysis_model="permutation", return_details=True):
    """
    Analyze a continuous outcome cluster RCT trial.
    
    This function provides a simplified interface to the core analysis logic,
    allowing for easy integration with permutation tests and other analysis methods.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns 'y' (outcome), 'treatment' (0/1), 'cluster' (cluster ID)
    icc : float, optional
        Intracluster correlation coefficient (for non-permutation methods), by default 0.05
    analysis_model : str, optional
        Analysis method ("permutation", "ttest", "mixedlm", "gee"), by default "permutation"
    return_details : bool, optional
        Whether to return detailed analysis information, by default True
        
    Returns
    -------
    tuple
        (t_stat, p_value, details) if return_details=True, else (t_stat, p_value)
        
    Notes
    -----
    This function extracts the core analysis logic from simulate_continuous_trial
    for use in permutation test integration and unit testing.
    """
    # Ensure proper data types
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Data must be a pandas DataFrame")
    
    required_cols = ['y', 'treatment', 'cluster']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Convert cluster to categorical if not already
    if not pd.api.types.is_categorical_dtype(df['cluster']):
        df = df.copy()
        df['cluster'] = df['cluster'].astype('category')
    
    # Route to the appropriate analysis method using the existing logic
    if analysis_model == "permutation":
        # Exact permutation test - recommended for very small clusters (5-10 per arm)
        from .permutation_tests import cluster_permutation_test
        
        # Aggregate to cluster means
        cluster_means = df.groupby(['cluster', 'treatment'], observed=True)['y'].mean().reset_index()
        control_means = cluster_means[cluster_means['treatment'] == 0]['y'].values
        interv_means = cluster_means[cluster_means['treatment'] == 1]['y'].values
        
        # Prepare data for permutation test
        perm_data = {
            'control_clusters': control_means,
            'treatment_clusters': interv_means
        }
        
        # Determine number of permutations based on cluster size
        total_clusters = len(control_means) + len(interv_means)
        if total_clusters <= 12:
            n_perms = 'exact'  # Use exact permutation for very small trials
        elif total_clusters <= 20:
            n_perms = 10000    # High precision for small trials
        else:
            n_perms = 5000     # Standard precision for larger trials
        
        perm_result = cluster_permutation_test(
            data=perm_data,
            test_statistic='t_statistic',  # Use t-statistic for compatibility
            n_permutations=n_perms,
            alternative='two-sided'
        )
        
        t_stat = perm_result['observed_statistic']
        p_value = perm_result['p_value']
        
        if return_details:
            return t_stat, p_value, {
                "converged": True, 
                "model": "permutation",
                "permutation_method": perm_result['method'],
                "n_permutations": perm_result['n_permutations_used'],
                "confidence_interval": perm_result['confidence_interval']
            }
        return t_stat, p_value
        
    elif analysis_model == "ttest":
        # Simple cluster-level t-test
        cluster_means = df.groupby(['cluster', 'treatment'], observed=True)['y'].mean().reset_index()
        control_means = cluster_means[cluster_means['treatment'] == 0]['y'].values
        interv_means = cluster_means[cluster_means['treatment'] == 1]['y'].values
        
        t_stat, p_value = stats.ttest_ind(control_means, interv_means, equal_var=True)
        
        if return_details:
            return t_stat, p_value, {"converged": True, "model": "ttest"}
        return t_stat, p_value
        
    else:
        # For other methods, use the OLS cluster test as a fallback
        # This avoids circular imports while providing basic functionality
        return _ols_cluster_test(df)