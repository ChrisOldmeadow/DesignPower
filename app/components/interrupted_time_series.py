"""
Component module for Interrupted Time Series designs.

This module provides UI rendering functions and calculation functions for
Interrupted Time Series designs with continuous and binary outcomes.
"""
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import graphviz

# Import design-specific modules
from core.designs.interrupted_time_series.analytical import (
    power_continuous, sample_size_continuous, power_binary, sample_size_binary
)
from core.designs.interrupted_time_series.simulation import (
    simulate_continuous, simulate_binary
)


def render_interrupted_time_series_continuous(calc_type, hypothesis_type):
    """
    Render the UI for Interrupted Time Series design with continuous outcome.
    
    Args:
        calc_type: String indicating calculation type (Sample Size, Power, or Minimum Detectable Effect)
        hypothesis_type: String indicating hypothesis type (Superiority or Non-Inferiority)
        
    Returns:
        dict: Parameters collected from the UI
    """
    params = {}
    
    # Display header with calculation type
    st.write(f"### Interrupted Time Series with Continuous Outcome ({calc_type})")
    
    # Method selection
    with st.container():
        st.subheader("Analysis Method")
        
        params["method"] = st.radio("Method", 
                                  options=["Analytical", "Simulation"],
                                  help="Analytical uses closed-form formulas; Simulation uses Monte Carlo methods",
                                  key="its_method_continuous")

    # Basic parameters UI
    with st.container():
        st.subheader("Study Design Parameters")
        
        # Parameter inputs
        col1, col2 = st.columns(2)
        
        with col1:
            if calc_type == "Sample Size":
                params["ratio"] = st.slider("Post/Pre Time Points Ratio", 
                                          min_value=0.5, max_value=3.0, value=1.0, step=0.1,
                                          help="Ratio of post-intervention to pre-intervention time points",
                                          key="its_ratio_continuous")
            else:
                params["n_pre"] = st.number_input("Pre-intervention Time Points", 
                                                value=12, step=1, min_value=3, max_value=100,
                                                help="Number of observations before intervention",
                                                key="its_n_pre_continuous")
                
                params["n_post"] = st.number_input("Post-intervention Time Points", 
                                                 value=12, step=1, min_value=3, max_value=100,
                                                 help="Number of observations after intervention",
                                                 key="its_n_post_continuous")
            
            params["mean_change"] = st.number_input("Expected Change in Level", 
                                                  value=0.5, step=0.1,
                                                  help="Expected difference in means between pre and post periods",
                                                  key="its_mean_change_continuous")
            
        with col2:
            params["std_dev"] = st.number_input("Standard Deviation", 
                                              value=2.0, step=0.1, min_value=0.1,
                                              help="Standard deviation of the outcome",
                                              key="its_sd_continuous")
            
            params["autocorr"] = st.slider("Autocorrelation", 
                                         min_value=0.0, max_value=0.9, value=0.2, step=0.05,
                                         help="Temporal correlation between consecutive observations",
                                         key="its_autocorr_continuous")

    # Statistical parameters
    with st.container():
        st.subheader("Statistical Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            params["alpha"] = st.slider("Significance Level (α)", 
                                      min_value=0.01, max_value=0.20, value=0.05, step=0.01,
                                      key="its_alpha_continuous")
        
        with col2:
            if calc_type == "Sample Size":
                params["power"] = st.slider("Power (1-β)", 
                                          min_value=0.70, max_value=0.99, value=0.80, step=0.01,
                                          key="its_power_continuous")
            elif params["method"] == "Simulation":
                params["nsim"] = st.selectbox("Number of Simulations", 
                                            options=[100, 500, 1000, 2000, 5000], 
                                            index=2,
                                            help="More simulations = more accurate results but slower computation",
                                            key="its_nsim_continuous")

    # Display effect size information
    if calc_type != "Sample Size":
        with st.expander("Study Design Summary", expanded=False):
            total_n = params["n_pre"] + params["n_post"]
            effective_n_factor = (1 - params["autocorr"]) / (1 + params["autocorr"]) if params["autocorr"] > 0 else 1.0
            effective_n = total_n * effective_n_factor
            
            st.write(f"**Total Time Points:** {total_n}")
            st.write(f"**Effective Sample Size:** {effective_n:.1f}")
            st.write(f"**Autocorrelation Adjustment Factor:** {effective_n_factor:.3f}")
            st.write(f"**Standardized Effect Size:** {abs(params['mean_change']) / params['std_dev']:.3f}")

    return params


def render_interrupted_time_series_binary(calc_type, hypothesis_type):
    """
    Render the UI for Interrupted Time Series design with binary outcome.
    
    Args:
        calc_type: String indicating calculation type (Sample Size, Power, or Minimum Detectable Effect)
        hypothesis_type: String indicating hypothesis type (Superiority or Non-Inferiority)
        
    Returns:
        dict: Parameters collected from the UI
    """
    params = {}
    
    # Display header with calculation type
    st.write(f"### Interrupted Time Series with Binary Outcome ({calc_type})")
    
    # Method selection
    with st.container():
        st.subheader("Analysis Method")
        
        params["method"] = st.radio("Method", 
                                  options=["Analytical", "Simulation"],
                                  help="Analytical uses closed-form formulas; Simulation uses Monte Carlo methods",
                                  key="its_method_binary")

    # Basic parameters UI
    with st.container():
        st.subheader("Study Design Parameters")
        
        # Parameter inputs
        col1, col2 = st.columns(2)
        
        with col1:
            if calc_type == "Sample Size":
                params["ratio"] = st.slider("Post/Pre Observations Ratio", 
                                          min_value=0.5, max_value=3.0, value=1.0, step=0.1,
                                          help="Ratio of post-intervention to pre-intervention observations",
                                          key="its_ratio_binary")
            else:
                params["n_pre"] = st.number_input("Pre-intervention Observations", 
                                                value=100, step=10, min_value=20, max_value=5000,
                                                help="Number of observations before intervention",
                                                key="its_n_pre_binary")
                
                params["n_post"] = st.number_input("Post-intervention Observations", 
                                                 value=100, step=10, min_value=20, max_value=5000,
                                                 help="Number of observations after intervention",
                                                 key="its_n_post_binary")
            
            params["p_pre"] = st.slider("Pre-intervention Proportion", 
                                      min_value=0.01, max_value=0.99, value=0.30, step=0.01,
                                      help="Expected proportion of events before intervention",
                                      key="its_p_pre")
            
        with col2:
            params["p_post"] = st.slider("Post-intervention Proportion", 
                                       min_value=0.01, max_value=0.99, value=0.45, step=0.01,
                                       help="Expected proportion of events after intervention",
                                       key="its_p_post")
            
            params["autocorr"] = st.slider("Autocorrelation", 
                                         min_value=0.0, max_value=0.9, value=0.2, step=0.05,
                                         help="Temporal correlation between consecutive observations",
                                         key="its_autocorr_binary")

    # Statistical parameters
    with st.container():
        st.subheader("Statistical Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            params["alpha"] = st.slider("Significance Level (α)", 
                                      min_value=0.01, max_value=0.20, value=0.05, step=0.01,
                                      key="its_alpha_binary")
        
        with col2:
            if calc_type == "Sample Size":
                params["power"] = st.slider("Power (1-β)", 
                                          min_value=0.70, max_value=0.99, value=0.80, step=0.01,
                                          key="its_power_binary")
            elif params["method"] == "Simulation":
                params["nsim"] = st.selectbox("Number of Simulations", 
                                            options=[100, 500, 1000, 2000, 5000], 
                                            index=2,
                                            help="More simulations = more accurate results but slower computation",
                                            key="its_nsim_binary")

    # Display effect size information
    with st.expander("Effect Size Information", expanded=False):
        risk_diff = params["p_post"] - params["p_pre"]
        if params["p_pre"] > 0:
            risk_ratio = params["p_post"] / params["p_pre"]
            odds_ratio = (params["p_post"] / (1 - params["p_post"])) / (params["p_pre"] / (1 - params["p_pre"]))
        else:
            risk_ratio = float('inf')
            odds_ratio = float('inf')
        
        st.write(f"**Risk Difference:** {risk_diff:.3f}")
        if risk_ratio != float('inf'):
            st.write(f"**Risk Ratio:** {risk_ratio:.3f}")
            st.write(f"**Odds Ratio:** {odds_ratio:.3f}")

    # Display study design summary
    if calc_type != "Sample Size":
        with st.expander("Study Design Summary", expanded=False):
            total_n = params["n_pre"] + params["n_post"]
            effective_n_factor = (1 - params["autocorr"]) / (1 + params["autocorr"]) if params["autocorr"] > 0 else 1.0
            effective_n = total_n * effective_n_factor
            
            st.write(f"**Total Observations:** {total_n:,}")
            st.write(f"**Effective Sample Size:** {effective_n:.0f}")
            st.write(f"**Autocorrelation Adjustment Factor:** {effective_n_factor:.3f}")

    return params


def calculate_interrupted_time_series_continuous(params):
    """
    Calculate power or sample size for interrupted time series design with continuous outcome.
    
    Args:
        params: Dictionary of parameters from the UI
        
    Returns:
        dict: Results from the calculation
    """
    try:
        method = params["method"].lower()
        calc_type = params.get("calculation_type", "Power")
        
        if method == "analytical":
            if calc_type == "Sample Size":
                # Sample size calculation
                result = sample_size_continuous(
                    mean_change=float(params["mean_change"]),
                    std_dev=float(params["std_dev"]),
                    power=float(params["power"]),
                    alpha=float(params["alpha"]),
                    autocorr=float(params["autocorr"]),
                    ratio=float(params["ratio"])
                )
            else:
                # Power calculation
                result = power_continuous(
                    n_pre=int(params["n_pre"]),
                    n_post=int(params["n_post"]),
                    mean_change=float(params["mean_change"]),
                    std_dev=float(params["std_dev"]),
                    alpha=float(params["alpha"]),
                    autocorr=float(params["autocorr"])
                )
        else:
            # Simulation method
            if calc_type == "Sample Size":
                st.warning("Sample size calculation via simulation not yet implemented for ITS. Using analytical method.")
                result = sample_size_continuous(
                    mean_change=float(params["mean_change"]),
                    std_dev=float(params["std_dev"]),
                    power=float(params["power"]),
                    alpha=float(params["alpha"]),
                    autocorr=float(params["autocorr"]),
                    ratio=float(params["ratio"])
                )
            else:
                # For now, use analytical method as simulation isn't fully implemented
                st.info("Using analytical method for ITS continuous outcomes.")
                result = power_continuous(
                    n_pre=int(params["n_pre"]),
                    n_post=int(params["n_post"]),
                    mean_change=float(params["mean_change"]),
                    std_dev=float(params["std_dev"]),
                    alpha=float(params["alpha"]),
                    autocorr=float(params["autocorr"])
                )
        
        # Add method information for display
        result["method"] = params["method"]
        result["design_type"] = "Interrupted Time Series"
        result["outcome_type"] = "Continuous"
        
        return result
        
    except Exception as e:
        st.error(f"Error in calculation: {str(e)}")
        return None


def calculate_interrupted_time_series_binary(params):
    """
    Calculate power or sample size for interrupted time series design with binary outcome.
    
    Args:
        params: Dictionary of parameters from the UI
        
    Returns:
        dict: Results from the calculation
    """
    try:
        method = params["method"].lower()
        calc_type = params.get("calculation_type", "Power")
        
        if method == "analytical":
            if calc_type == "Sample Size":
                # Sample size calculation
                result = sample_size_binary(
                    p_pre=float(params["p_pre"]),
                    p_post=float(params["p_post"]),
                    power=float(params["power"]),
                    alpha=float(params["alpha"]),
                    autocorr=float(params["autocorr"]),
                    ratio=float(params["ratio"])
                )
            else:
                # Power calculation
                result = power_binary(
                    n_pre=int(params["n_pre"]),
                    n_post=int(params["n_post"]),
                    p_pre=float(params["p_pre"]),
                    p_post=float(params["p_post"]),
                    alpha=float(params["alpha"]),
                    autocorr=float(params["autocorr"])
                )
        else:
            # Simulation method
            if calc_type == "Sample Size":
                st.warning("Sample size calculation via simulation not yet implemented for ITS. Using analytical method.")
                result = sample_size_binary(
                    p_pre=float(params["p_pre"]),
                    p_post=float(params["p_post"]),
                    power=float(params["power"]),
                    alpha=float(params["alpha"]),
                    autocorr=float(params["autocorr"]),
                    ratio=float(params["ratio"])
                )
            else:
                # For now, use analytical method as simulation isn't fully implemented
                st.info("Using analytical method for ITS binary outcomes.")
                result = power_binary(
                    n_pre=int(params["n_pre"]),
                    n_post=int(params["n_post"]),
                    p_pre=float(params["p_pre"]),
                    p_post=float(params["p_post"]),
                    alpha=float(params["alpha"]),
                    autocorr=float(params["autocorr"])
                )
        
        # Add method information for display
        result["method"] = params["method"]
        result["design_type"] = "Interrupted Time Series"
        result["outcome_type"] = "Binary"
        
        return result
        
    except Exception as e:
        st.error(f"Error in calculation: {str(e)}")
        return None


def generate_cli_code_interrupted_time_series_continuous(params):
    """
    Generate CLI code for interrupted time series continuous outcome calculation.
    
    Args:
        params: Dictionary of parameters from the UI
        
    Returns:
        str: CLI command string
    """
    calc_type = params.get("calculation_type", "Power")
    
    if calc_type == "Sample Size":
        mean_change = params["mean_change"]
        std_dev = params["std_dev"]
        power = params["power"]
        alpha = params["alpha"]
        autocorr = params["autocorr"]
        ratio = params["ratio"]
        
        cli_code = f"""# Interrupted Time Series - Continuous Outcome (Sample Size)
from core.designs.interrupted_time_series.analytical import sample_size_continuous

result = sample_size_continuous(
    mean_change={mean_change},
    std_dev={std_dev},
    power={power},
    alpha={alpha},
    autocorr={autocorr},
    ratio={ratio}
)

print(f"Required pre-intervention time points: {{result['n_pre']}}")
print(f"Required post-intervention time points: {{result['n_post']}}")
print(f"Total time points: {{result['total_n']}}")"""
    else:
        n_pre = params["n_pre"]
        n_post = params["n_post"]
        mean_change = params["mean_change"]
        std_dev = params["std_dev"]
        alpha = params["alpha"]
        autocorr = params["autocorr"]
        
        cli_code = f"""# Interrupted Time Series - Continuous Outcome (Power)
from core.designs.interrupted_time_series.analytical import power_continuous

result = power_continuous(
    n_pre={n_pre},
    n_post={n_post},
    mean_change={mean_change},
    std_dev={std_dev},
    alpha={alpha},
    autocorr={autocorr}
)

print(f"Power: {{result['power']:.3f}}")"""
    
    return cli_code


def generate_cli_code_interrupted_time_series_binary(params):
    """
    Generate CLI code for interrupted time series binary outcome calculation.
    
    Args:
        params: Dictionary of parameters from the UI
        
    Returns:
        str: CLI command string
    """
    calc_type = params.get("calculation_type", "Power")
    
    if calc_type == "Sample Size":
        p_pre = params["p_pre"]
        p_post = params["p_post"]
        power = params["power"]
        alpha = params["alpha"]
        autocorr = params["autocorr"]
        ratio = params["ratio"]
        
        cli_code = f"""# Interrupted Time Series - Binary Outcome (Sample Size)
from core.designs.interrupted_time_series.analytical import sample_size_binary

result = sample_size_binary(
    p_pre={p_pre},
    p_post={p_post},
    power={power},
    alpha={alpha},
    autocorr={autocorr},
    ratio={ratio}
)

print(f"Required pre-intervention observations: {{result['n_pre']}}")
print(f"Required post-intervention observations: {{result['n_post']}}")
print(f"Total observations: {{result['total_n']}}")"""
    else:
        n_pre = params["n_pre"]
        n_post = params["n_post"]
        p_pre = params["p_pre"]
        p_post = params["p_post"]
        alpha = params["alpha"]
        autocorr = params["autocorr"]
        
        cli_code = f"""# Interrupted Time Series - Binary Outcome (Power)
from core.designs.interrupted_time_series.analytical import power_binary

result = power_binary(
    n_pre={n_pre},
    n_post={n_post},
    p_pre={p_pre},
    p_post={p_post},
    alpha={alpha},
    autocorr={autocorr}
)

print(f"Power: {{result['power']:.3f}}")"""
    
    return cli_code