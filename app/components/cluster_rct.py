"""Component module for Cluster RCT designs.

This module provides UI rendering functions and calculation functions for
Cluster Randomized Controlled Trial designs with continuous and binary outcomes.
"""
import streamlit as st
import numpy as np
import pandas as pd
import math

# Import specific analytical and simulation modules
from core.designs.cluster_rct import analytical_continuous
from core.designs.cluster_rct import simulation_continuous
from core.designs.cluster_rct import analytical_binary
from core.designs.cluster_rct import simulation_binary

# Shared functions
def render_binary_advanced_options():
    """
    Render advanced options for binary outcome designs in cluster RCTs.
    
    Returns:
        dict: Advanced parameters
    """
    advanced_params = {}
    
    # Method selection (analytical vs simulation)
    advanced_params["method"] = st.radio(
        "Calculation Method",
        ["Analytical", "Simulation"],
        index=0,
        key="cluster_binary_method_radio",
        horizontal=True
    )
    
    # Convert to lowercase for function calls
    advanced_params["method"] = advanced_params["method"].lower()
    
    # Simulation-specific options
    if advanced_params["method"] == "simulation":
        st.markdown("#### Simulation Parameters")
        col1, col2 = st.columns(2)
        
        with col1:
            advanced_params["nsim"] = st.number_input(
                "Number of Simulations", 
                value=1000, 
                min_value=100, 
                max_value=10000,
                step=100,
                key="cluster_binary_nsim"
            )
        
        with col2:
            advanced_params["seed"] = st.number_input(
                "Random Seed", 
                value=42, 
                min_value=1, 
                max_value=99999,
                key="cluster_binary_seed"
            )
    
    return advanced_params


def render_continuous_advanced_options():
    """
    Render advanced options for continuous outcome designs in cluster RCTs.
    
    Returns:
        dict: Advanced parameters
    """
    advanced_params = {}
    
    # Method selection (analytical vs simulation)
    advanced_params["method"] = st.radio(
        "Calculation Method",
        ["Analytical", "Simulation"],
        index=0,
        key="cluster_continuous_method_radio",
        horizontal=True
    )
    
    # Convert to lowercase for function calls
    advanced_params["method"] = advanced_params["method"].lower()
    
    # Simulation-specific options
    if advanced_params["method"] == "simulation":
        st.markdown("#### Simulation Parameters")
        col1, col2 = st.columns(2)
        
        with col1:
            advanced_params["nsim"] = st.number_input(
                "Number of Simulations", 
                value=1000, 
                min_value=100, 
                max_value=10000,
                step=100,
                key="cluster_continuous_nsim"
            )
        
        with col2:
            advanced_params["seed"] = st.number_input(
                "Random Seed", 
                value=42, 
                min_value=1, 
                max_value=99999,
                key="cluster_continuous_seed"
            )
    
    return advanced_params


def render_cluster_continuous(calc_type, hypothesis_type):
    """
    Render the UI for Cluster RCT with continuous outcome.
    
    Args:
        calc_type: String indicating calculation type (Sample Size, Power, or Minimum Detectable Effect)
        hypothesis_type: String indicating hypothesis type (Superiority or Non-Inferiority)
    
    Returns:
        dict: Parameters collected from the UI
    """
    params = {}
    params["calc_type"] = calc_type
    params["hypothesis_type"] = hypothesis_type
    params["outcome_type"] = "continuous"
    
    st.markdown("### Study Parameters")
    
    # For continuous outcomes in cluster RCTs
    col1, col2 = st.columns(2)
    
    with col1:
        # Cluster parameters
        st.markdown("#### Cluster Parameters")
        params["cluster_size"] = st.number_input(
            "Average Cluster Size", 
            value=20, 
            min_value=2, 
            help="Average number of individuals per cluster"
        )
        
        params["icc"] = st.number_input(
            "Intracluster Correlation Coefficient (ICC)", 
            value=0.05, 
            min_value=0.0, 
            max_value=1.0, 
            format="%f", 
            help="Correlation between individuals within the same cluster"
        )
        
        if calc_type == "Sample Size":
            params["power"] = st.slider(
                "Power (1-β)", 
                min_value=0.7, 
                max_value=0.99, 
                value=0.8, 
                step=0.01, 
                format="%0.2f"
            )
        elif calc_type == "Power":
            params["n_clusters"] = st.number_input(
                "Number of Clusters per Arm", 
                min_value=2, 
                value=15,
                help="Number of clusters in each treatment arm"
            )
        elif calc_type == "Minimum Detectable Effect":
            params["n_clusters"] = st.number_input(
                "Number of Clusters per Arm", 
                min_value=2, 
                value=15,
                help="Number of clusters in each treatment arm"
            )
            params["power"] = st.slider(
                "Power (1-β)", 
                min_value=0.7, 
                max_value=0.99, 
                value=0.8, 
                step=0.01, 
                format="%0.2f"
            )
    
    with col2:
        # Outcome parameters
        st.markdown("#### Outcome Parameters")
        if hypothesis_type == "Superiority":
            if calc_type != "Minimum Detectable Effect":
                params["mean1"] = st.number_input(
                    "Mean Outcome in Control Group", 
                    value=0.0, 
                    format="%f"
                )
                params["mean2"] = st.number_input(
                    "Mean Outcome in Intervention Group", 
                    value=0.5, 
                    format="%f"
                )
            else:  # Minimum Detectable Effect
                params["mean1"] = st.number_input(
                    "Mean Outcome in Control Group", 
                    value=0.0, 
                    format="%f"
                )
                # mean2 will be calculated
            
            params["std_dev"] = st.number_input(
                "Standard Deviation", 
                value=1.0, 
                min_value=0.01, 
                format="%f"
            )
        
        # Significance level
        params["alpha"] = st.select_slider(
            "Significance Level (α)", 
            options=[0.001, 0.01, 0.05, 0.1], 
            value=0.05, 
            format="%0.3f"
        )
    
    # Advanced options in an expander
    with st.expander("Advanced Options"):
        advanced_params = render_continuous_advanced_options()
        params.update(advanced_params)
    
    return params


def render_cluster_binary(calc_type, hypothesis_type):
    """
    Render the UI for Cluster RCT with binary outcome.
    
    Args:
        calc_type: String indicating calculation type (Sample Size, Power, or Minimum Detectable Effect)
        hypothesis_type: String indicating hypothesis type (Superiority or Non-Inferiority)
        
    Returns:
        dict: Parameters collected from the UI
    """
    params = {}
    params["calc_type"] = calc_type
    params["hypothesis_type"] = hypothesis_type
    params["outcome_type"] = "binary"
    
    st.markdown("### Study Parameters")
    
    # For binary outcomes in cluster RCTs
    col1, col2 = st.columns(2)
    
    with col1:
        # Cluster parameters
        st.markdown("#### Cluster Parameters")
        params["cluster_size"] = st.number_input(
            "Average Cluster Size", 
            value=20, 
            min_value=2, 
            help="Average number of individuals per cluster"
        )
        
        params["icc"] = st.number_input(
            "Intracluster Correlation Coefficient (ICC)", 
            value=0.05, 
            min_value=0.0, 
            max_value=1.0, 
            format="%f", 
            help="Correlation between individuals within the same cluster"
        )
        
        if calc_type == "Sample Size":
            params["power"] = st.slider(
                "Power (1-β)", 
                min_value=0.7, 
                max_value=0.99, 
                value=0.8, 
                step=0.01, 
                format="%0.2f"
            )
        elif calc_type == "Power":
            params["n_clusters"] = st.number_input(
                "Number of Clusters per Arm", 
                min_value=2, 
                value=15,
                help="Number of clusters in each treatment arm"
            )
        elif calc_type == "Minimum Detectable Effect":
            params["n_clusters"] = st.number_input(
                "Number of Clusters per Arm", 
                min_value=2, 
                value=15,
                help="Number of clusters in each treatment arm"
            )
            params["power"] = st.slider(
                "Power (1-β)", 
                min_value=0.7, 
                max_value=0.99, 
                value=0.8, 
                step=0.01, 
                format="%0.2f"
            )
    
    with col2:
        # Outcome parameters
        st.markdown("#### Outcome Parameters")
        if hypothesis_type == "Superiority":
            if calc_type != "Minimum Detectable Effect":
                params["p1"] = st.slider(
                    "Proportion in Control Group", 
                    min_value=0.01, 
                    max_value=0.99, 
                    value=0.3, 
                    step=0.01, 
                    format="%0.2f"
                )
                params["p2"] = st.slider(
                    "Proportion in Intervention Group", 
                    min_value=0.01, 
                    max_value=0.99, 
                    value=0.5, 
                    step=0.01, 
                    format="%0.2f"
                )
            else:  # Minimum Detectable Effect
                params["p1"] = st.slider(
                    "Proportion in Control Group", 
                    min_value=0.01, 
                    max_value=0.99, 
                    value=0.3, 
                    step=0.01, 
                    format="%0.2f"
                )
                # p2 will be calculated
        
        # Significance level
        params["alpha"] = st.select_slider(
            "Significance Level (α)", 
            options=[0.001, 0.01, 0.05, 0.1], 
            value=0.05, 
            format="%0.3f"
        )
    
    # Advanced options in an expander
    with st.expander("Advanced Options"):
        advanced_params = render_binary_advanced_options()
        params.update(advanced_params)
    
    return params


def calculate_cluster_continuous(params):
    """
    Calculate results for Cluster RCT with continuous outcome.
    
    Args:
        params: Dictionary of parameters
        
    Returns:
        dict: Results of calculations
    """
    # Extract parameters
    calc_type = params.get("calc_type")
    method = params.get("method", "analytical")
    
    try:
        # Check for required parameters based on calculation type
        if calc_type == "Sample Size":
            required_params = ["mean1", "mean2", "std_dev", "icc", "cluster_size", "power", "alpha"]
        elif calc_type == "Power":
            required_params = ["n_clusters", "cluster_size", "icc", "mean1", "mean2", "std_dev", "alpha"]
        elif calc_type == "Minimum Detectable Effect":
            required_params = ["n_clusters", "cluster_size", "icc", "std_dev", "power", "alpha"]
        
        # Validate required parameters
        for param in required_params:
            if params.get(param) is None:
                return {"error": f"Missing required parameter: {param}"}
        
        # Call appropriate function based on calculation type and method
        if calc_type == "Sample Size":
            if method == "analytical":
                results = analytical_continuous.sample_size_continuous(
                    mean1=params["mean1"],
                    mean2=params["mean2"],
                    std_dev=params["std_dev"],
                    icc=params["icc"],
                    cluster_size=params["cluster_size"],
                    power=params["power"],
                    alpha=params["alpha"]
                )
            else:  # simulation
                results = simulation_continuous.sample_size_continuous_sim(
                    mean1=params["mean1"],
                    mean2=params["mean2"],
                    std_dev=params["std_dev"],
                    icc=params["icc"],
                    cluster_size=params["cluster_size"],
                    power=params["power"],
                    alpha=params["alpha"],
                    nsim=params.get("nsim", 1000),
                    seed=params.get("seed", 42)
                )
        
        elif calc_type == "Power":
            if method == "analytical":
                results = analytical_continuous.power_continuous(
                    n_clusters=params["n_clusters"],
                    cluster_size=params["cluster_size"],
                    icc=params["icc"],
                    mean1=params["mean1"],
                    mean2=params["mean2"],
                    std_dev=params["std_dev"],
                    alpha=params["alpha"]
                )
            else:  # simulation
                results = simulation_continuous.power_continuous_sim(
                    n_clusters=params["n_clusters"],
                    cluster_size=params["cluster_size"],
                    icc=params["icc"],
                    mean1=params["mean1"],
                    mean2=params["mean2"],
                    std_dev=params["std_dev"],
                    alpha=params["alpha"],
                    nsim=params.get("nsim", 1000),
                    seed=params.get("seed", 42)
                )
        
        elif calc_type == "Minimum Detectable Effect":
            if method == "analytical":
                results = analytical_continuous.min_detectable_effect_continuous(
                    n_clusters=params["n_clusters"],
                    cluster_size=params["cluster_size"],
                    icc=params["icc"],
                    std_dev=params["std_dev"],
                    power=params["power"],
                    alpha=params["alpha"]
                )
            else:  # simulation
                results = simulation_continuous.min_detectable_effect_continuous_sim(
                    n_clusters=params["n_clusters"],
                    cluster_size=params["cluster_size"],
                    icc=params["icc"],
                    std_dev=params["std_dev"],
                    power=params["power"],
                    alpha=params["alpha"],
                    nsim=params.get("nsim", 1000),
                    seed=params.get("seed", 42)
                )
        
        # Add calculation method and design method to results
        results["design_method"] = "Cluster RCT"
        
        return results
    
    except Exception as e:
        return {"error": f"Error in calculation: {str(e)}"}


def calculate_cluster_binary(params):
    """
    Calculate results for Cluster RCT with binary outcome.
    
    Args:
        params: Dictionary of parameters
        
    Returns:
        dict: Results of calculations
    """
    # Extract parameters
    calc_type = params.get("calc_type")
    method = params.get("method", "analytical")
    
    try:
        # Check for required parameters based on calculation type
        if calc_type == "Sample Size":
            required_params = ["p1", "p2", "icc", "cluster_size", "power", "alpha"]
        elif calc_type == "Power":
            required_params = ["n_clusters", "cluster_size", "icc", "p1", "p2", "alpha"]
        elif calc_type == "Minimum Detectable Effect":
            required_params = ["n_clusters", "cluster_size", "icc", "p1", "power", "alpha"]
        
        # Validate required parameters
        for param in required_params:
            if params.get(param) is None:
                return {"error": f"Missing required parameter: {param}"}
        
        # Call appropriate function based on calculation type and method
        if calc_type == "Sample Size":
            if method == "analytical":
                results = analytical_binary.sample_size_binary(
                    p1=params["p1"],
                    p2=params["p2"],
                    icc=params["icc"],
                    cluster_size=params["cluster_size"],
                    power=params["power"],
                    alpha=params["alpha"]
                )
            else:  # simulation
                results = simulation_binary.sample_size_binary_sim(
                    p1=params["p1"],
                    p2=params["p2"],
                    icc=params["icc"],
                    cluster_size=params["cluster_size"],
                    power=params["power"],
                    alpha=params["alpha"],
                    nsim=params.get("nsim", 1000),
                    seed=params.get("seed", 42)
                )
        
        elif calc_type == "Power":
            if method == "analytical":
                results = analytical_binary.power_binary(
                    n_clusters=params["n_clusters"],
                    cluster_size=params["cluster_size"],
                    icc=params["icc"],
                    p1=params["p1"],
                    p2=params["p2"],
                    alpha=params["alpha"]
                )
            else:  # simulation
                results = simulation_binary.power_binary_sim(
                    n_clusters=params["n_clusters"],
                    cluster_size=params["cluster_size"],
                    icc=params["icc"],
                    p1=params["p1"],
                    p2=params["p2"],
                    alpha=params["alpha"],
                    nsim=params.get("nsim", 1000),
                    seed=params.get("seed", 42)
                )
        
        elif calc_type == "Minimum Detectable Effect":
            if method == "analytical":
                results = analytical_binary.min_detectable_effect_binary(
                    n_clusters=params["n_clusters"],
                    cluster_size=params["cluster_size"],
                    icc=params["icc"],
                    p1=params["p1"],
                    power=params["power"],
                    alpha=params["alpha"]
                )
            else:  # simulation
                results = simulation_binary.min_detectable_effect_binary_sim(
                    n_clusters=params["n_clusters"],
                    cluster_size=params["cluster_size"],
                    icc=params["icc"],
                    p1=params["p1"],
                    power=params["power"],
                    alpha=params["alpha"],
                    nsim=params.get("nsim", 1000),
                    seed=params.get("seed", 42)
                )
        
        # Add calculation method and design method to results
        results["design_method"] = "Cluster RCT"
        
        return results
    
    except Exception as e:
        return {"error": f"Error in calculation: {str(e)}"}