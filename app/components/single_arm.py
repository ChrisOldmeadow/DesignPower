"""
Single Arm Trial components for the DesignPower application.

This module contains UI components for rendering Single Arm Trial design options
and handling related calculations for continuous, binary, and survival outcomes.
"""
import streamlit as st
import numpy as np
import math
from scipy import stats

import os
import sys

# Add the project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Now import from core
from core.stats import power_calculations
from core.designs.single_arm import analytical as single_arm_analytical

# Import shared component modules
from .advanced_options import (
    render_continuous_advanced_options, 
    render_binary_advanced_options,
    render_survival_advanced_options
)

def render_single_arm_continuous(sidebar=True):
    """
    Render the UI for Single Arm Trial with continuous outcome.
    
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
            params["mean_null"] = st.number_input("Null Hypothesis Mean", value=0.0, step=0.1, key="mean_null_input")
            params["mean_alt"] = st.number_input("Alternative Hypothesis Mean", value=0.5, step=0.1, key="mean_alt_input")
            params["std_dev"] = st.number_input("Standard Deviation", value=1.0, step=0.1, min_value=0.1, key="sd_input")
            
        with col2:
            params["alpha"] = st.slider("Significance Level (α)", 
                                      min_value=0.01, max_value=0.20, value=0.05, step=0.01,
                                      key="alpha_slider")
            params["power"] = st.slider("Power (1-β)", 
                                      min_value=0.70, max_value=0.99, value=0.80, step=0.01,
                                      key="power_slider")
        
        # Historical data option
        has_historical = st.checkbox("Use historical control data", value=False, key="historical_checkbox")
        params["has_historical"] = has_historical
        
        # Show historical data parameters if selected
        if has_historical:
            st.markdown("---")
            st.write("Historical Control Data:")
            
            col1, col2 = st.columns(2)
            
            with col1:
                params["historical_n"] = st.number_input("Historical Sample Size", 
                                                      value=50, step=1, min_value=1,
                                                      key="historical_n_input")
                
            with col2:
                params["discount_factor"] = st.slider("Historical Data Weight", 
                                                   min_value=0.0, max_value=1.0, value=0.5, step=0.05,
                                                   key="discount_slider")
                st.info("A weight of 1.0 gives full weight to historical data, while 0.0 ignores it.")
    
    # Advanced parameters UI
    with st.expander("Advanced Parameters", expanded=False):
        # For continuous outcomes, we might use a subset of the continuous advanced options
        advanced_options = render_continuous_advanced_options()
        params.update(advanced_options)
    
    return params


def render_single_arm_binary(sidebar=True):
    """
    Render the UI for Single Arm Trial with binary outcome.
    
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
            params["p_null"] = st.slider("Null Hypothesis Proportion", 
                                       min_value=0.01, max_value=0.99, value=0.20, step=0.01,
                                       key="p_null_slider")
            params["p_alt"] = st.slider("Alternative Hypothesis Proportion", 
                                      min_value=0.01, max_value=0.99, value=0.40, step=0.01,
                                      key="p_alt_slider")
            
        with col2:
            params["alpha"] = st.slider("Significance Level (α)", 
                                      min_value=0.01, max_value=0.20, value=0.05, step=0.01,
                                      key="alpha_slider")
            params["power"] = st.slider("Power (1-β)", 
                                      min_value=0.70, max_value=0.99, value=0.80, step=0.01,
                                      key="power_slider")
        
        # Historical data option
        has_historical = st.checkbox("Use historical control data", value=False, key="historical_checkbox")
        params["has_historical"] = has_historical
        
        # Show historical data parameters if selected
        if has_historical:
            st.markdown("---")
            st.write("Historical Control Data:")
            
            col1, col2 = st.columns(2)
            
            with col1:
                params["historical_n"] = st.number_input("Historical Sample Size", 
                                                      value=50, step=1, min_value=1,
                                                      key="historical_n_input")
                params["historical_events"] = st.number_input("Historical Events", 
                                                          value=10, step=1, min_value=0,
                                                          key="historical_events_input")
                
            with col2:
                params["discount_factor"] = st.slider("Historical Data Weight", 
                                                   min_value=0.0, max_value=1.0, value=0.5, step=0.05,
                                                   key="discount_slider")
                st.info("A weight of 1.0 gives full weight to historical data, while 0.0 ignores it.")
    
    # Advanced parameters UI
    with st.expander("Advanced Parameters", expanded=False):
        # For binary outcomes, we can use the existing binary test options
        advanced_options = render_binary_advanced_options()
        params.update(advanced_options)
    
    return params


def render_single_arm_survival(sidebar=True):
    """
    Render the UI for Single Arm Trial with survival outcome.
    
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
            params["median_null"] = st.number_input("Null Median Survival", 
                                                 value=6.0, step=0.5, min_value=0.1,
                                                 key="median_null_input")
            params["median_alt"] = st.number_input("Alternative Median Survival", 
                                                value=9.0, step=0.5, min_value=0.1,
                                                key="median_alt_input")
            
        with col2:
            params["alpha"] = st.slider("Significance Level (α)", 
                                      min_value=0.01, max_value=0.20, value=0.05, step=0.01,
                                      key="alpha_slider")
            params["power"] = st.slider("Power (1-β)", 
                                      min_value=0.70, max_value=0.99, value=0.80, step=0.01,
                                      key="power_slider")
        
        # Study timeline
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
        
        # Historical data option
        has_historical = st.checkbox("Use historical control data", value=False, key="historical_checkbox")
        params["has_historical"] = has_historical
        
        # Show historical data parameters if selected
        if has_historical:
            st.markdown("---")
            st.write("Historical Control Data:")
            
            col1, col2 = st.columns(2)
            
            with col1:
                params["historical_n"] = st.number_input("Historical Sample Size", 
                                                      value=50, step=1, min_value=1,
                                                      key="historical_n_input")
                
            with col2:
                params["discount_factor"] = st.slider("Historical Data Weight", 
                                                   min_value=0.0, max_value=1.0, value=0.5, step=0.05,
                                                   key="discount_slider")
    
    # Advanced parameters UI
    with st.expander("Advanced Parameters", expanded=False):
        # Advanced survival options
        advanced_options = render_survival_advanced_options()
        params.update(advanced_options)
        
        # Additional survival-specific options
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


def calculate_single_arm_continuous(params, method="analytical"):
    """
    Calculate sample size for Single Arm Trial with continuous outcome.
    
    Args:
        params: Dictionary of parameters
        method: String indicating calculation method (analytical or simulation)
        
    Returns:
        dict: Results of calculations
    """
    # Extract parameters
    mean_null = params["mean_null"]
    mean_alt = params["mean_alt"]
    std_dev = params["std_dev"]
    alpha = params["alpha"]
    power = params["power"]
    has_historical = params.get("has_historical", False)
    
    # Calculate effect size for reporting
    effect_size = abs(mean_alt - mean_null) / std_dev
    
    if method == "analytical":
        # Use our power_calculations module for one-sample continuous tests
        n = power_calculations.one_sample_t_test_sample_size(
            mean_null=mean_null,
            mean_alt=mean_alt,
            std_dev=std_dev,
            alpha=alpha,
            power=power
        )
        
        # If we have historical data, adjust the sample size
        if has_historical:
            historical_n = params["historical_n"]
            discount_factor = params["discount_factor"]
            
            # Adjust sample size based on historical data weight
            # Historical data effectively increases the "information" we have
            n_adjusted = n / (1 + discount_factor * historical_n / n)
            n = math.ceil(n_adjusted)
    else:
        # Simulation method would be implemented here
        # For now, just use the analytical result
        n = power_calculations.one_sample_t_test_sample_size(
            mean_null=mean_null,
            mean_alt=mean_alt,
            std_dev=std_dev,
            alpha=alpha,
            power=power
        )
    
    result = {
        "n": n,
        "effect_size": effect_size,
        "method": method.capitalize()
    }
    
    return result


def calculate_single_arm_binary(params, method="analytical"):
    """
    Calculate sample size for Single Arm Trial with binary outcome.
    
    Args:
        params: Dictionary of parameters
        method: String indicating calculation method (analytical or simulation)
        
    Returns:
        dict: Results of calculations
    """
    # Extract parameters
    p_null = params["p_null"]
    p_alt = params["p_alt"]
    alpha = params["alpha"]
    power = params["power"]
    test_type = params.get("test_type", "Normal Approximation")
    has_historical = params.get("has_historical", False)
    
    # Calculate effect size for reporting
    effect_size = abs(p_alt - p_null)
    
    if method == "analytical":
        # Use our power_calculations module for one-sample proportion tests
        n = power_calculations.one_sample_proportion_test_sample_size(
            p_null=p_null,
            p_alt=p_alt,
            alpha=alpha,
            power=power,
            test_type=test_type
        )
        
        # If we have historical data, adjust the sample size
        if has_historical:
            historical_n = params["historical_n"]
            discount_factor = params["discount_factor"]
            historical_events = params.get("historical_events", int(historical_n * p_null))
            
            # Adjust sample size based on historical data weight
            n_adjusted = n / (1 + discount_factor * historical_n / n)
            n = math.ceil(n_adjusted)
    else:
        # Simulation method would be implemented here
        # For now, just use the analytical result
        n = power_calculations.one_sample_proportion_test_sample_size(
            p_null=p_null,
            p_alt=p_alt,
            alpha=alpha,
            power=power,
            test_type=test_type
        )
    
    result = {
        "n": n,
        "effect_size": effect_size,
        "test_type": test_type,
        "method": method.capitalize()
    }
    
    return result


def calculate_single_arm_survival(params, method="analytical"):
    """
    Calculate sample size for Single Arm Trial with survival outcome.
    
    Args:
        params: Dictionary of parameters
        method: String indicating calculation method (analytical or simulation)
        
    Returns:
        dict: Results of calculations
    """
    # Extract parameters
    median_null = params["median_null"]
    median_alt = params["median_alt"]
    alpha = params["alpha"]
    power = params["power"]
    enrollment_period = params["enrollment_period"]
    follow_up_period = params["follow_up_period"]
    dropout_rate = params.get("dropout_rate", 0.1)
    has_historical = params.get("has_historical", False)
    
    # Convert medians to hazard rates
    hazard_null = math.log(2) / median_null
    hazard_alt = math.log(2) / median_alt
    
    # Calculate hazard ratio for reporting
    hazard_ratio = hazard_alt / hazard_null
    
    if method == "analytical":
        # Use power_calculations for one-sample survival analysis
        result = power_calculations.one_sample_survival_test_sample_size(
            median_null=median_null,
            median_alt=median_alt,
            enrollment_period=enrollment_period,
            follow_up_period=follow_up_period,
            alpha=alpha,
            power=power,
            dropout_rate=dropout_rate
        )
        
        # Extract values from result
        n = result["n"]
        events = result["events"]
        
        # If we have historical data, adjust the sample size
        if has_historical:
            historical_n = params["historical_n"]
            discount_factor = params["discount_factor"]
            
            # Adjust sample size based on historical data weight
            n_adjusted = n / (1 + discount_factor * historical_n / n)
            n = math.ceil(n_adjusted)
            # Events would also need to be adjusted proportionally
            events = math.ceil(events * (n_adjusted / n))
    else:
        # Simulation method would be implemented here
        # For now, just use the analytical result
        result = power_calculations.one_sample_survival_test_sample_size(
            median_null=median_null,
            median_alt=median_alt,
            enrollment_period=enrollment_period,
            follow_up_period=follow_up_period,
            alpha=alpha,
            power=power,
            dropout_rate=dropout_rate
        )
        n = result["n"]
        events = result["events"]
    
    final_result = {
        "n": n,
        "events": events,
        "hazard_ratio": hazard_ratio,
        "method": method.capitalize()
    }
    
    return final_result
