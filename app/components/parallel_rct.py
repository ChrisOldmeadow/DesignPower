"""
Parallel RCT components for the DesignPower application.

This module contains UI components for rendering Parallel Randomized
Controlled Trial design options and handling related calculations.
"""
import streamlit as st
import numpy as np
import math

import os
import sys

# Add the project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Now import from core
from core.designs.parallel import analytical
from core.designs.parallel import binary_simulation, continuous_simulation
from core.designs.parallel.binary_tests import perform_binary_test
from core.utils.formatting import format_number_with_precision

# Import shared component modules
from .advanced_options import (
    render_continuous_advanced_options, 
    render_binary_advanced_options,
    render_survival_advanced_options
)

def render_parallel_continuous(sidebar=True):
    """
    Render the UI for Parallel RCT with continuous outcome.
    
    Args:
        sidebar: Boolean indicating if parameters should be in sidebar
    
    Returns:
        dict: Parameters collected from the UI
    """
    params = {}
    
    # Basic parameters UI
    with st.container():
        st.subheader("Basic Parameters")
        
        # Parameter inputs
        col1, col2 = st.columns(2)
        
        with col1:
            params["mean1"] = st.number_input("Mean (Group 1)", value=0.0, step=0.1, key="mean1_input")
            params["mean2"] = st.number_input("Mean (Group 2)", value=0.5, step=0.1, key="mean2_input")
            params["std_dev"] = st.number_input("Standard Deviation", value=1.0, step=0.1, min_value=0.1, key="sd_input")
            
        with col2:
            params["alpha"] = st.slider("Significance Level (α)", 
                                     min_value=0.01, max_value=0.20, value=0.05, step=0.01,
                                     key="alpha_slider")
            params["power"] = st.slider("Power (1-β)", 
                                     min_value=0.70, max_value=0.99, value=0.80, step=0.01,
                                     key="power_slider")
            params["allocation_ratio"] = st.slider("Allocation Ratio (n2/n1)", 
                                               min_value=0.1, max_value=10.0, value=1.0, step=0.1,
                                               key="allocation_slider")
        
        # Non-inferiority option
        non_inferiority = st.checkbox("Non-inferiority design", value=False, key="non_inferiority_checkbox")
        params["non_inferiority"] = non_inferiority
        
        # Show non-inferiority margin if selected
        if non_inferiority:
            col1, col2 = st.columns(2)
            
            with col1:
                params["margin"] = st.number_input("Non-inferiority Margin", value=0.2, step=0.1, key="margin_input")
            
            with col2:
                params["direction"] = st.selectbox(
                    "Direction",
                    ["lower", "upper"],
                    index=0,
                    key="direction_select"
                )
                
                if params["direction"] == "lower":
                    st.info("Lower means smaller values are better (e.g., pain scores)")
                else:
                    st.info("Upper means larger values are better (e.g., quality of life)")
    
    # Advanced parameters UI
    with st.expander("Advanced Parameters", expanded=False):
        # Get advanced options using our shared component
        advanced_options = render_continuous_advanced_options()
        params.update(advanced_options)
        
        # Simulation-specific options
        if sidebar:
            # This would go in the sidebar - skipping for component extraction
            pass
    
    return params


def render_parallel_binary(sidebar=True):
    """
    Render the UI for Parallel RCT with binary outcome.
    
    Args:
        sidebar: Boolean indicating if parameters should be in sidebar
    
    Returns:
        dict: Parameters collected from the UI
    """
    params = {}
    
    # Basic parameters UI
    with st.container():
        st.subheader("Basic Parameters")
        
        # Parameter inputs
        col1, col2 = st.columns(2)
        
        with col1:
            params["p1"] = st.slider("Proportion (Group 1)", 
                                  min_value=0.01, max_value=0.99, value=0.50, step=0.01,
                                  key="p1_slider")
            params["p2"] = st.slider("Proportion (Group 2)", 
                                  min_value=0.01, max_value=0.99, value=0.60, step=0.01,
                                  key="p2_slider")
            
        with col2:
            params["alpha"] = st.slider("Significance Level (α)", 
                                     min_value=0.01, max_value=0.20, value=0.05, step=0.01,
                                     key="alpha_slider")
            params["power"] = st.slider("Power (1-β)", 
                                     min_value=0.70, max_value=0.99, value=0.80, step=0.01,
                                     key="power_slider")
            params["allocation_ratio"] = st.slider("Allocation Ratio (n2/n1)", 
                                               min_value=0.1, max_value=10.0, value=1.0, step=0.1,
                                               key="allocation_slider")
        
        # Non-inferiority option
        non_inferiority = st.checkbox("Non-inferiority design", value=False, key="non_inferiority_checkbox")
        params["non_inferiority"] = non_inferiority
        
        # Show non-inferiority margin if selected
        if non_inferiority:
            col1, col2 = st.columns(2)
            
            with col1:
                params["margin"] = st.number_input("Non-inferiority Margin", value=0.1, step=0.01, key="margin_input")
                
            with col2:
                params["direction"] = st.selectbox(
                    "Direction",
                    ["lower", "upper"],
                    index=0,
                    key="direction_select"
                )
                
                if params["direction"] == "lower":
                    st.info("Lower means Group 1 can be worse by at most the margin (e.g., treatment not worse than control by more than margin)")
                else:
                    st.info("Upper means Group 1 must be no more than the margin below Group 2 (e.g., experimental treatment can only be slightly less effective)")
    
    # Advanced parameters UI
    with st.expander("Advanced Parameters", expanded=False):
        # Get advanced options using our shared component
        advanced_options = render_binary_advanced_options()
        params.update(advanced_options)
        
        # Simulation-specific options
        if sidebar:
            # This would go in the sidebar - skipping for component extraction
            pass
    
    return params


def render_parallel_survival(sidebar=True):
    """
    Render the UI for Parallel RCT with survival outcome.
    
    Args:
        sidebar: Boolean indicating if parameters should be in sidebar
    
    Returns:
        dict: Parameters collected from the UI
    """
    params = {}
    
    # Basic parameters UI
    with st.container():
        st.subheader("Basic Parameters")
        
        # Parameter inputs
        col1, col2 = st.columns(2)
        
        with col1:
            params["hazard_ratio"] = st.number_input("Hazard Ratio (HR)", 
                                                  value=0.70, step=0.05, min_value=0.05, max_value=10.0,
                                                  key="hr_input")
            params["median_survival1"] = st.number_input("Median Survival Time (Group 1)", 
                                                      value=12.0, step=1.0, min_value=0.1,
                                                      key="median1_input")
            
        with col2:
            params["alpha"] = st.slider("Significance Level (α)", 
                                     min_value=0.01, max_value=0.20, value=0.05, step=0.01,
                                     key="alpha_slider")
            params["power"] = st.slider("Power (1-β)", 
                                     min_value=0.70, max_value=0.99, value=0.80, step=0.01,
                                     key="power_slider")
            params["allocation_ratio"] = st.slider("Allocation Ratio (n2/n1)", 
                                                min_value=0.1, max_value=10.0, value=1.0, step=0.1,
                                                key="allocation_slider")
        
        # For survival, also add study duration and recruitment period
        st.markdown("---")
        st.write("Study Timeline:")
        
        col1, col2 = st.columns(2)
        with col1:
            params["enrollment_period"] = st.number_input("Recruitment/Enrollment Period", 
                                                       value=12.0, step=1.0, min_value=0.1,
                                                       key="enrollment_input")
        
        with col2:
            params["follow_up_period"] = st.number_input("Follow-up Period", 
                                                      value=12.0, step=1.0, min_value=0.1,
                                                      key="follow_up_input")
    
    # Advanced parameters UI
    with st.expander("Advanced Parameters", expanded=False):
        # Get advanced options using our shared component
        advanced_options = render_survival_advanced_options()
        params.update(advanced_options)
        
        # Additional survival-specific options could be added here
        col1, col2 = st.columns(2)
        
        with col1:
            params["dropout_rate"] = st.slider("Dropout Rate", 
                                            min_value=0.0, max_value=0.50, value=0.10, step=0.05,
                                            key="dropout_slider")
        
        with col2:
            params["recruitment_pattern"] = st.selectbox(
                "Recruitment Pattern",
                ["Uniform", "Increasing", "Decreasing"],
                index=0,
                key="recruitment_select"
            )
    
    return params


def calculate_parallel_continuous(params, method="analytical"):
    """
    Calculate results for Parallel RCT with continuous outcome.
    
    Args:
        params: Dictionary of parameters
        method: String indicating calculation method (analytical or simulation)
        
    Returns:
        dict: Results of calculations
    """
    # Extract parameters
    mean1 = params["mean1"]
    mean2 = params["mean2"]
    std_dev = params["std_dev"]
    alpha = params["alpha"]
    power = params["power"]
    allocation_ratio = params["allocation_ratio"]
    
    # Handle advanced options
    unequal_var = params.get("unequal_var", False)
    repeated_measures = params.get("repeated_measures", False)
    
    # Handle non-inferiority
    non_inferiority = params.get("non_inferiority", False)
    margin = params.get("margin", 0.0)
    lower_is_better = params.get("lower_is_better", True)
    
    # Additional parameters for repeated measures
    correlation = params.get("correlation", 0.5)
    analysis_method = params.get("analysis_method", "ANCOVA")
    
    # Second standard deviation for unequal variance
    std_dev2 = params.get("std_dev2", std_dev)
    
    # Prepare result dictionary
    result = {}
    
    # Calculate sample size
    if method == "analytical":
        if non_inferiority:
            # Non-inferiority test
            sample_size = analytical.sample_size_continuous_ni(
                mean1=mean1,
                std_dev=std_dev,
                margin=margin,
                power=power,
                alpha=alpha,
                allocation_ratio=allocation_ratio,
                lower_is_better=lower_is_better
            )
        else:
            # Standard superiority test
            sample_size = analytical.sample_size_continuous(
                mean1=mean1, 
                mean2=mean2, 
                std_dev=std_dev, 
                std_dev2=std_dev2 if unequal_var else std_dev,
                power=power,
                alpha=alpha,
                allocation_ratio=allocation_ratio,
                unequal_var=unequal_var,
                repeated_measures=repeated_measures,
                correlation=correlation if repeated_measures else None,
                analysis_method=analysis_method if repeated_measures else None
            )
            
        result.update(sample_size)
        result["method"] = "Analytical"
        
    else:  # simulation
        # Get simulation parameters from params or defaults
        nsim = params.get("nsim", 1000)
        min_n = params.get("min_n", 10)
        max_n = params.get("max_n", 1000)
        step = params.get("step", 10)
        
        # Call the simulation function
        sim_result = continuous_simulation.sample_size_continuous_sim(
            mean1=mean1,
            mean2=mean2,
            std_dev=std_dev,
            std_dev2=std_dev2 if unequal_var else std_dev,
            power=power,
            alpha=alpha,
            allocation_ratio=allocation_ratio,
            nsim=nsim,
            min_n=min_n,
            max_n=max_n,
            step=step,
            unequal_var=unequal_var
        )
        
        result.update(sim_result)
        result["method"] = "Simulation"
    
    return result


def calculate_parallel_binary(params, method="analytical"):
    """
    Calculate results for Parallel RCT with binary outcome.
    
    Args:
        params: Dictionary of parameters
        method: String indicating calculation method (analytical or simulation)
        
    Returns:
        dict: Results of calculations
    """
    # Extract parameters
    p1 = params["p1"]
    p2 = params["p2"]
    alpha = params["alpha"]
    power = params["power"]
    allocation_ratio = params["allocation_ratio"]
    test_type = params.get("test_type", "Normal Approximation")
    
    # Handle non-inferiority
    non_inferiority = params.get("non_inferiority", False)
    margin = params.get("margin", 0.0)
    lower_is_better = params.get("lower_is_better", True)
    
    # Prepare result dictionary
    result = {}
    
    if method == "analytical":
        # Call appropriate analytical function based on hypothesis type
        if non_inferiority:
            sample_size = analytical.sample_size_binary_ni(
                p1=p1,
                margin=margin,
                power=power,
                alpha=alpha,
                allocation_ratio=allocation_ratio,
                lower_is_better=lower_is_better,
                test_type=test_type
            )
        else:
            sample_size = analytical.sample_size_binary(
                p1=p1, 
                p2=p2, 
                power=power,
                alpha=alpha,
                allocation_ratio=allocation_ratio,
                test_type=test_type
            )
            
        result.update(sample_size)
        result["method"] = "Analytical"
        result["test_type"] = test_type
        
    else:  # simulation
        # Get simulation parameters
        nsim = params.get("nsim", 1000)
        min_n = params.get("min_n", 10)
        max_n = params.get("max_n", 1000)
        step = params.get("step", 10)
        
        # Call the simulation function
        sim_result = binary_simulation.sample_size_binary_sim(
            p1=p1,
            p2=p2,
            power=power,
            alpha=alpha,
            allocation_ratio=allocation_ratio,
            nsim=nsim,
            min_n=min_n,
            max_n=max_n,
            step=step,
            test_type=test_type
        )
        
        result.update(sim_result)
        result["method"] = "Simulation"
        result["test_type"] = test_type
        
    return result


def calculate_parallel_survival(params, method="analytical"):
    """
    Calculate results for Parallel RCT with survival outcome.
    
    Args:
        params: Dictionary of parameters
        method: String indicating calculation method (analytical or simulation)
        
    Returns:
        dict: Results of calculations
    """
    # Extract parameters
    hazard_ratio = params["hazard_ratio"]
    median_survival1 = params["median_survival1"]
    alpha = params["alpha"]
    power = params["power"]
    allocation_ratio = params["allocation_ratio"]
    enrollment_period = params["enrollment_period"]
    follow_up_period = params["follow_up_period"]
    dropout_rate = params.get("dropout_rate", 0.1)
    
    # This is a placeholder - actual implementation would call the appropriate functions
    # from the survival analysis modules
    
    # Prepare result dictionary
    result = {
        "n1": 100,  # Placeholder
        "n2": 100,  # Placeholder
        "total_n": 200,  # Placeholder
        "events": 150,  # Placeholder
    }
    
    return result
