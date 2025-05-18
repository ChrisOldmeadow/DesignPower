"""
Utility functions for sample size calculation and power analysis.

This module provides helper functions and shared utilities
for use across the sample size calculation project.
"""
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


def generate_code_snippet(method_name, parameters):
    """
    Generate a reproducible code snippet for a given method and parameters.
    
    Parameters
    ----------
    method_name : str
        Name of the method used for calculation
    parameters : dict
        Dictionary of parameters used in the calculation
    
    Returns
    -------
    str
        Python code snippet that reproduces the calculation
    """
    if method_name == "sample_size_difference_in_means":
        params = parameters.copy()
        snippet = f"from core.power import sample_size_difference_in_means\n\n"
        snippet += f"result = sample_size_difference_in_means(\n"
        snippet += f"    delta={params.get('delta')},\n"
        snippet += f"    std_dev={params.get('std_dev')},\n"
        snippet += f"    power={params.get('power')},\n"
        snippet += f"    alpha={params.get('alpha')},\n"
        snippet += f"    allocation_ratio={params.get('allocation_ratio')}\n"
        snippet += f")\n\n"
        snippet += f"print(f\"Required sample sizes: {result['n1']} and {result['n2']} (total: {result['total_n']})\")"
        
    elif method_name == "power_binary_cluster_rct":
        params = parameters.copy()
        snippet = f"from core.power import power_binary_cluster_rct\n\n"
        snippet += f"result = power_binary_cluster_rct(\n"
        snippet += f"    n_clusters={params.get('n_clusters')},\n"
        snippet += f"    cluster_size={params.get('cluster_size')},\n"
        snippet += f"    icc={params.get('icc')},\n"
        snippet += f"    p1={params.get('p1')},\n"
        snippet += f"    p2={params.get('p2')},\n"
        snippet += f"    alpha={params.get('alpha')}\n"
        snippet += f")\n\n"
        snippet += f"print(f\"Statistical power: {result['power']:.4f}\")"
        
    elif method_name == "sample_size_binary_cluster_rct":
        params = parameters.copy()
        snippet = f"from core.power import sample_size_binary_cluster_rct\n\n"
        snippet += f"result = sample_size_binary_cluster_rct(\n"
        snippet += f"    p1={params.get('p1')},\n"
        snippet += f"    p2={params.get('p2')},\n"
        snippet += f"    icc={params.get('icc')},\n"
        snippet += f"    cluster_size={params.get('cluster_size')},\n"
        snippet += f"    power={params.get('power')},\n"
        snippet += f"    alpha={params.get('alpha')}\n"
        snippet += f")\n\n"
        snippet += f"print(f\"Required clusters per arm: {result['n_clusters_per_arm']} (total: {result['total_clusters']})\")"

    elif method_name == "simulate_stepped_wedge":
        params = parameters.copy()
        snippet = f"from core.simulation import simulate_stepped_wedge\n\n"
        snippet += f"result = simulate_stepped_wedge(\n"
        snippet += f"    clusters={params.get('clusters')},\n"
        snippet += f"    steps={params.get('steps')},\n"
        snippet += f"    individuals_per_cluster={params.get('individuals_per_cluster')},\n"
        snippet += f"    icc={params.get('icc')},\n"
        snippet += f"    treatment_effect={params.get('treatment_effect')},\n"
        snippet += f"    std_dev={params.get('std_dev')},\n"
        snippet += f"    nsim={params.get('nsim', 1000)},\n"
        snippet += f"    alpha={params.get('alpha')}\n"
        snippet += f")\n\n"
        snippet += f"print(f\"Estimated power: {result['power']:.4f}\")"
        
    elif method_name == "julia_stepped_wedge":
        params = parameters.copy()
        snippet = f"from julia import Main\n"
        snippet += f"from julia.api import Julia\n\n"
        snippet += f"# Initialize Julia\n"
        snippet += f"jl = Julia(compiled_modules=False)\n"
        snippet += f"Main.include(\"julia_backend/stepped_wedge.jl\")\n\n"
        snippet += f"result = Main.simulate_stepped_wedge(\n"
        snippet += f"    {params.get('clusters')},  # clusters\n"
        snippet += f"    {params.get('steps')},  # steps\n"
        snippet += f"    {params.get('individuals_per_cluster')},  # individuals per cluster\n"
        snippet += f"    {params.get('icc')},  # ICC\n"
        snippet += f"    {params.get('treatment_effect')},  # treatment effect\n"
        snippet += f"    {params.get('std_dev')},  # standard deviation\n"
        snippet += f"    {params.get('nsim', 1000)},  # number of simulations\n"
        snippet += f"    {params.get('alpha')}  # alpha\n"
        snippet += f")\n\n"
        snippet += f"print(f\"Estimated power: {result['power']:.4f}\")"
        
    else:
        snippet = "# Unknown method"
    
    return snippet


def generate_plain_language_summary(method_name, result):
    """
    Generate a plain language summary of the calculation results.
    
    Parameters
    ----------
    method_name : str
        Name of the method used for calculation
    result : dict
        Dictionary containing the calculation results
    
    Returns
    -------
    str
        Plain language summary of the results
    """
    if method_name == "sample_size_difference_in_means":
        delta = result['parameters']['delta']
        std_dev = result['parameters']['std_dev']
        power = result['parameters']['power']
        alpha = result['parameters']['alpha']
        n1 = result['n1']
        n2 = result['n2']
        total = result['total_n']
        
        summary = f"To detect a difference in means of {delta} (standardized effect size: {delta/std_dev:.2f}) "
        summary += f"with {power*100}% power at a {alpha*100}% significance level, "
        summary += f"you need {n1} participants in the first group and {n2} in the second group "
        summary += f"(total: {total} participants)."
        
    elif method_name == "power_binary_cluster_rct":
        n_clusters = result['parameters']['n_clusters']
        cluster_size = result['parameters']['cluster_size']
        icc = result['parameters']['icc']
        p1 = result['parameters']['p1']
        p2 = result['parameters']['p2']
        power = result['power']
        total_n = result['parameters']['total_n']
        
        summary = f"With {n_clusters} clusters per arm and {cluster_size} individuals per cluster "
        summary += f"(total: {total_n} individuals), you have {power*100:.1f}% power to detect "
        summary += f"a difference in proportions from {p1*100:.1f}% to {p2*100:.1f}% "
        summary += f"with an intracluster correlation coefficient (ICC) of {icc}."
        
    elif method_name == "sample_size_binary_cluster_rct":
        p1 = result['parameters']['p1']
        p2 = result['parameters']['p2']
        icc = result['parameters']['icc']
        cluster_size = result['parameters']['cluster_size']
        power = result['parameters']['power']
        n_clusters = result['n_clusters_per_arm']
        total_clusters = result['total_clusters']
        total_n = result['total_n']
        
        summary = f"To detect a change in proportion from {p1*100:.1f}% to {p2*100:.1f}% "
        summary += f"with {power*100}% power, given an ICC of {icc} and {cluster_size} individuals per cluster, "
        summary += f"you need {n_clusters} clusters per arm (total: {total_clusters} clusters, {total_n} individuals)."
        
    elif method_name == "simulate_stepped_wedge":
        clusters = result['parameters']['clusters']
        steps = result['parameters']['steps']
        indiv_per_cluster = result['parameters']['individuals_per_cluster']
        icc = result['parameters']['icc']
        effect = result['parameters']['treatment_effect']
        std_dev = result['parameters']['std_dev']
        power = result['power']
        total_n = result['parameters']['total_n']
        
        summary = f"Based on {result['nsim']} simulations of a stepped wedge design with "
        summary += f"{clusters} clusters, {steps} steps, and {indiv_per_cluster} individuals per cluster per step "
        summary += f"(total: {total_n} observations), you have approximately {power*100:.1f}% power to detect "
        summary += f"a treatment effect of {effect} (standardized effect: {effect/std_dev:.2f}) "
        summary += f"with an ICC of {icc}."
        
    else:
        summary = "Result summary not available for this calculation method."
    
    return summary


def create_power_curve(method, param_range, param_name, fixed_params):
    """
    Create data for a power curve by varying one parameter.
    
    Parameters
    ----------
    method : callable
        Method to call for each parameter value
    param_range : list or array
        Range of values for the parameter to vary
    param_name : str
        Name of the parameter being varied
    fixed_params : dict
        Dictionary of fixed parameters
    
    Returns
    -------
    tuple
        (x_values, y_values) for plotting
    """
    powers = []
    
    for val in param_range:
        # Update parameters with current value
        params = fixed_params.copy()
        params[param_name] = val
        
        # Calculate power for current parameters
        result = method(**params)
        
        # Extract power from result
        if 'power' in result:
            powers.append(result['power'])
        else:
            powers.append(None)
    
    return param_range, powers
