"""
Component module for Single Arm Trial designs.

This module provides UI rendering functions and calculation functions for
Single Arm Trial designs with continuous, binary, and survival outcomes.
"""
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import design-specific modules
from core.designs.single_arm.continuous import one_sample_t_test_sample_size
from core.designs.single_arm.continuous import one_sample_t_test_power
from core.designs.single_arm.binary import one_sample_proportion_test_sample_size
from core.designs.single_arm.binary import one_sample_proportion_test_power
from core.designs.single_arm.survival import one_sample_survival_test_sample_size
from core.designs.single_arm.survival import one_sample_survival_test_power

def render_single_arm_continuous(calc_type, hypothesis_type):
    """
    Render the UI for Single Arm Trial with continuous outcome.
    
    Args:
        calc_type: String indicating calculation type (Sample Size, Power, or Minimum Detectable Effect)
        hypothesis_type: String indicating hypothesis type (Superiority or Non-Inferiority)
        
    Returns:
        dict: Parameters collected from the UI
    """
    params = {}
    
    # Display header with calculation type
    st.write(f"### Single Arm Trial with Continuous Outcome ({calc_type})")
    
    # Basic parameters UI
    with st.container():
        st.subheader("Basic Parameters")
        
        # Parameter inputs
        col1, col2 = st.columns(2)
        
        with col1:
            params["mean"] = st.number_input("Sample Mean", value=0.5, step=0.1, key="mean_input_single")
            params["std_dev"] = st.number_input("Standard Deviation", value=1.0, step=0.1, min_value=0.1, key="sd_input_single")
            params["null_mean"] = st.number_input("Null Hypothesis Mean", value=0.0, step=0.1, key="null_mean_input")
            
        with col2:
            params["alpha"] = st.slider("Significance Level (α)", 
                                     min_value=0.01, max_value=0.20, value=0.05, step=0.01,
                                     key="alpha_slider_single")
                                     
            # Only show power slider when not calculating power
            if calc_type != "Power":
                params["power"] = st.slider("Power (1-β)", 
                                         min_value=0.70, max_value=0.99, value=0.80, step=0.01,
                                         key="power_slider_single")
            else:
                # When calculating power, use a default value (not shown in UI)
                params["power"] = None
                # When calculating power, we need sample size
                params["n"] = st.number_input("Sample Size", value=20, step=1, min_value=5, key="n_input_single")
        
        # Historical data option
        has_historical = st.checkbox("Use historical control data", value=False, key="historical_checkbox")
        params["has_historical"] = has_historical
        
        # Show historical data parameters if selected
        if has_historical:
            st.markdown("---")
            st.write("Historical Control Data:")
            
            col1, col2 = st.columns(2)
            
            with col1:
                params["historical_mean"] = st.number_input("Historical Mean", value=0.0, step=0.1, key="historical_mean_input")
                params["historical_std_dev"] = st.number_input("Historical Standard Deviation", 
                                                            value=1.0, step=0.1, min_value=0.1,
                                                            key="historical_sd_input")
                
            with col2:
                params["historical_n"] = st.number_input("Historical Sample Size", 
                                                      value=20, step=1, min_value=1,
                                                      key="historical_n_input")
                params["correlation"] = st.slider("Prior-Data Correlation", 
                                               min_value=0.0, max_value=1.0, value=0.5, step=0.01,
                                               key="correlation_slider_historical")
    
    # Advanced parameters UI
    with st.expander("Advanced Parameters", expanded=False):
        # Test type
        params["test"] = st.selectbox(
            "Test Type",
            ["One-sample t-test", "Wilcoxon signed-rank test"],
            index=0,
            key="test_select_single"
        )
        
        # Alternative hypothesis
        params["alternative"] = st.selectbox(
            "Alternative Hypothesis",
            ["two-sided", "greater", "less"],
            index=0,
            key="alternative_select_single"
        )
    
    return params


def render_single_arm_binary(calc_type, hypothesis_type):
    """
    Render the UI for Single Arm Trial with binary outcome.
    
    Args:
        calc_type: String indicating calculation type (Sample Size, Power, or Minimum Detectable Effect)
        hypothesis_type: String indicating hypothesis type (Superiority or Non-Inferiority)
        
    Returns:
        dict: Parameters collected from the UI
    """
    params = {}
    
    # Display header with calculation type
    st.write(f"### Single Arm Trial with Binary Outcome ({calc_type})")
    
    # Basic parameters UI
    with st.container():
        st.subheader("Basic Parameters")
        
        # Parameter inputs
        col1, col2 = st.columns(2)
        
        with col1:
            params["p"] = st.slider("Sample Proportion", 
                                 min_value=0.01, max_value=0.99, value=0.3, step=0.01,
                                 key="p_slider_single")
            params["p0"] = st.slider("Null Hypothesis Proportion", 
                                  min_value=0.01, max_value=0.99, value=0.5, step=0.01,
                                  key="p0_slider_single")
            
        with col2:
            params["alpha"] = st.slider("Significance Level (α)", 
                                     min_value=0.01, max_value=0.20, value=0.05, step=0.01,
                                     key="alpha_slider_single_binary")
                                     
            # Only show power slider when not calculating power
            if calc_type != "Power":
                params["power"] = st.slider("Power (1-β)", 
                                         min_value=0.70, max_value=0.99, value=0.80, step=0.01,
                                         key="power_slider_single_binary")
            else:
                # When calculating power, use a default value (not shown in UI)
                params["power"] = None
                # When calculating power, we need sample size
                params["n"] = st.number_input("Sample Size", value=30, step=1, min_value=5, key="n_input_single_binary")
    
    # Advanced parameters UI
    with st.expander("Advanced Parameters", expanded=False):
        # Correction methods
        params["correction"] = st.selectbox(
            "Correction Method",
            ["None", "Continuity", "Exact"],
            index=0,
            key="correction_select_single"
        )
        
        # Alternative hypothesis
        params["alternative"] = st.selectbox(
            "Alternative Hypothesis",
            ["two-sided", "greater", "less"],
            index=0,
            key="alternative_select_single_binary"
        )
    
    return params


def render_single_arm_survival(calc_type, hypothesis_type):
    """
    Render the UI for Single Arm Trial with survival outcome.
    
    Args:
        calc_type: String indicating calculation type (Sample Size, Power, or Minimum Detectable Effect)
        hypothesis_type: String indicating hypothesis type (Superiority or Non-Inferiority)
        
    Returns:
        dict: Parameters collected from the UI
    """
    params = {}
    
    # Display header with calculation type
    st.write(f"### Single Arm Trial with Survival Outcome ({calc_type})")
    
    # Basic parameters UI
    with st.container():
        st.subheader("Basic Parameters")
        
        # Parameter inputs
        col1, col2 = st.columns(2)
        
        with col1:
            params["median_survival"] = st.number_input("Median Survival (months)", 
                                                  value=12.0, step=1.0, min_value=0.1,
                                                  key="median_survival_input_single")
            params["null_median_survival"] = st.number_input("Null Hypothesis Median Survival (months)", 
                                                       value=6.0, step=1.0, min_value=0.1,
                                                       key="null_median_input_single")
        
        with col2:
            params["alpha"] = st.slider("Significance Level (α)", 
                                     min_value=0.01, max_value=0.20, value=0.05, step=0.01,
                                     key="alpha_slider_single_survival")
                                     
            # Only show power slider when not calculating power
            if calc_type != "Power":
                params["power"] = st.slider("Power (1-β)", 
                                         min_value=0.70, max_value=0.99, value=0.80, step=0.01,
                                         key="power_slider_single_survival")
            else:
                # When calculating power, use a default value (not shown in UI)
                params["power"] = None
                # When calculating power, we need sample size
                params["n"] = st.number_input("Sample Size", value=30, step=1, min_value=5, key="n_input_single_survival")
    
    # Advanced parameters UI
    with st.expander("Advanced Parameters", expanded=False):
        # Study parameters
        col1, col2 = st.columns(2)
        
        with col1:
            params["accrual_time"] = st.number_input("Accrual Time (months)", 
                                                 value=12.0, step=1.0, min_value=0.1,
                                                 key="accrual_input_single")
            
        with col2:
            params["follow_up_time"] = st.number_input("Follow-up Time (months)", 
                                                    value=24.0, step=1.0, min_value=0.1,
                                                    key="followup_input_single")
            
        # Dropout parameter
        params["dropout_rate"] = st.slider("Dropout Rate", 
                                        min_value=0.0, max_value=0.5, value=0.1, step=0.01,
                                        key="dropout_slider_single")
        
        # Test type
        params["test"] = st.selectbox(
            "Test Type",
            ["Log-rank", "Cox model"],
            index=0,
            key="test_select_single_survival"
        )
    
    return params


def calculate_single_arm_continuous(params):
    """
    Calculate results for Single Arm Trial with continuous outcome.
    
    Args:
        params: Dictionary of parameters
        
    Returns:
        dict: Results of calculations
    """
    # Extract parameters with defaults
    mean = params.get("mean", 0.5)
    std_dev = params.get("std_dev", 1.0)
    null_mean = params.get("null_mean", 0.0)
    alpha = params.get("alpha", 0.05)
    power = params.get("power", 0.8)
    
    # Historical data parameters
    has_historical = params.get("has_historical", False)
    historical_mean = params.get("historical_mean", 0.0)
    historical_std_dev = params.get("historical_std_dev", 1.0)
    historical_n = params.get("historical_n", 20)
    correlation = params.get("correlation", 0.5)
    
    # Advanced parameters
    test = params.get("test", "One-sample t-test")
    alternative = params.get("alternative", "two-sided")
    
    # Prepare result dictionary
    result = {}
    
    # Calculate based on calculation_type
    calculation_type = params.get("calculation_type", "Sample Size")
    
    if calculation_type == "Sample Size":
        # Calculate sample size
        if has_historical:
            # Use historical control data
            # Simplified approach - adjust for prior information
            adjusted_std_dev = std_dev * (1 - correlation**2)**0.5
            sample_size = one_sample_t_test_sample_size(
                mean=mean,
                null_mean=null_mean,
                std_dev=adjusted_std_dev,
                power=power,
                alpha=alpha,
                alternative=alternative
            )
        else:
            # Standard single arm design
            sample_size = one_sample_t_test_sample_size(
                mean=mean,
                null_mean=null_mean,
                std_dev=std_dev,
                power=power,
                alpha=alpha,
                alternative=alternative
            )
            
        # Format results
        result["n"] = round(sample_size["n"])
        result["effect_size"] = round((mean - null_mean) / std_dev, 3)
        
        return result
        
    elif calculation_type == "Power":
        # Get sample size
        n = params.get("n", 20)
        
        # Calculate power
        if has_historical:
            # Use historical control data
            # Simplified approach - adjust for prior information
            adjusted_std_dev = std_dev * (1 - correlation**2)**0.5
            power_result = one_sample_t_test_power(
                n=n,
                mean=mean,
                null_mean=null_mean,
                std_dev=adjusted_std_dev,
                alpha=alpha,
                alternative=alternative
            )
        else:
            # Standard single arm design
            power_result = one_sample_t_test_power(
                n=n,
                mean=mean,
                null_mean=null_mean,
                std_dev=std_dev,
                alpha=alpha,
                alternative=alternative
            )
        
        # Format results
        result["power"] = round(power_result["power"], 3)
        result["n"] = n
        result["effect_size"] = round((mean - null_mean) / std_dev, 3)
        
        return result
        
    elif calculation_type == "Minimum Detectable Effect":
        # Get sample size
        n = params.get("n", 20)
        
        # Calculate MDE
        # This is the minimum difference detectable with the given sample size and power
        if has_historical:
            # Use historical control data
            # Simplified approach - adjust for prior information
            adjusted_std_dev = std_dev * (1 - correlation**2)**0.5
            t_crit = stats.t.ppf(1 - alpha / (2 if alternative == "two-sided" else 1), n - 1)
            t_pow = stats.t.ppf(power, n - 1)
            mde = (t_crit + t_pow) * adjusted_std_dev / (n**0.5)
        else:
            # Standard single arm design
            t_crit = stats.t.ppf(1 - alpha / (2 if alternative == "two-sided" else 1), n - 1)
            t_pow = stats.t.ppf(power, n - 1)
            mde = (t_crit + t_pow) * std_dev / (n**0.5)
        
        # Format results
        result["mde"] = round(mde, 3)
        result["n"] = n
        
        return result
    
    return result


def calculate_single_arm_binary(params):
    """
    Calculate results for Single Arm Trial with binary outcome.
    
    Args:
        params: Dictionary of parameters
        
    Returns:
        dict: Results of calculations
    """
    # Extract parameters with default values
    p = params.get("p", 0.3)
    p0 = params.get("p0", 0.5)
    alpha = params.get("alpha", 0.05)
    power = params.get("power", 0.8)
    
    # Advanced parameters
    correction = params.get("correction", "None")
    alternative = params.get("alternative", "two-sided")
    
    # Prepare result dictionary
    result = {}
    
    # Calculate based on calculation_type
    calculation_type = params.get("calculation_type", "Sample Size")
    
    if calculation_type == "Sample Size":
        # Calculate sample size
        sample_size = one_sample_proportion_test_sample_size(
            p=p,
            p0=p0,
            power=power,
            alpha=alpha,
            alternative=alternative,
            correction=correction != "None"
        )
            
        # Format results
        result["n"] = round(sample_size["n"])
        result["absolute_risk_difference"] = round(p - p0, 3)
        result["relative_risk"] = round(p / p0, 3) if p0 > 0 else "Infinity"
        
        return result
        
    elif calculation_type == "Power":
        # Get sample size
        n = params.get("n", 30)
        
        # Calculate power
        power_result = one_sample_proportion_test_power(
            n=n,
            p=p,
            p0=p0,
            alpha=alpha,
            alternative=alternative,
            correction=correction != "None"
        )
        
        # Format results
        result["power"] = round(power_result["power"], 3)
        result["n"] = n
        result["absolute_risk_difference"] = round(p - p0, 3)
        result["relative_risk"] = round(p / p0, 3) if p0 > 0 else "Infinity"
        
        return result
        
    elif calculation_type == "Minimum Detectable Effect":
        # Get sample size
        n = params.get("n", 30)
        
        # Calculate MDE
        # This is the minimum detectable proportion with the given null proportion,
        # sample size, and power
        
        # Approximate calculation based on normal approximation
        z_alpha = stats.norm.ppf(1 - alpha / (2 if alternative == "two-sided" else 1))
        z_beta = stats.norm.ppf(power)
        
        # Calculate MDE (minimum detectable proportion)
        if alternative == "two-sided" or alternative == "greater":
            # For p > p0
            p_mde = p0 + (z_alpha + z_beta) * (p0 * (1 - p0) / n)**0.5
            mde = p_mde - p0
        else:
            # For p < p0
            p_mde = p0 - (z_alpha + z_beta) * (p0 * (1 - p0) / n)**0.5
            mde = p0 - p_mde
        
        # Format results
        result["mde"] = round(mde, 3)
        result["p_mde"] = round(p_mde, 3)
        result["n"] = n
        
        return result
    
    return result


def calculate_single_arm_survival(params):
    """
    Calculate results for Single Arm Trial with survival outcome.
    
    Args:
        params: Dictionary of parameters
        
    Returns:
        dict: Results of calculations
    """
    # Extract parameters with default values
    median_survival = params.get("median_survival", 12.0)
    null_median_survival = params.get("null_median_survival", 6.0)
    alpha = params.get("alpha", 0.05)
    power = params.get("power", 0.8)
    
    # Study parameters
    accrual_time = params.get("accrual_time", 12.0)
    follow_up_time = params.get("follow_up_time", 24.0)
    dropout_rate = params.get("dropout_rate", 0.1)
    
    # Advanced parameters
    test = params.get("test", "Log-rank")
    
    # Convert median survival to lambda (rate parameter)
    lambda1 = math.log(2) / median_survival
    lambda0 = math.log(2) / null_median_survival
    
    # Prepare result dictionary
    result = {}
    
    # Calculate based on calculation_type
    calculation_type = params.get("calculation_type", "Sample Size")
    
    if calculation_type == "Sample Size":
        # Calculate sample size
        sample_size = one_sample_survival_test_sample_size(
            lambda0=lambda0,
            lambda1=lambda1,
            accrual_time=accrual_time,
            follow_up_time=follow_up_time,
            dropout_rate=dropout_rate,
            power=power,
            alpha=alpha
        )
            
        # Format results
        result["n"] = round(sample_size["n"])
        result["events"] = round(sample_size["events"])
        result["hazard_ratio"] = round(lambda1 / lambda0, 3)
        result["median_survival"] = median_survival
        result["null_median_survival"] = null_median_survival
        
        return result
        
    elif calculation_type == "Power":
        # Get sample size
        n = params.get("n", 30)
        
        # Calculate power
        power_result = one_sample_survival_test_power(
            n=n,
            lambda0=lambda0,
            lambda1=lambda1,
            accrual_time=accrual_time,
            follow_up_time=follow_up_time,
            dropout_rate=dropout_rate,
            alpha=alpha
        )
        
        # Format results
        result["power"] = round(power_result["power"], 3)
        result["n"] = n
        result["events"] = round(power_result["events"])
        result["hazard_ratio"] = round(lambda1 / lambda0, 3)
        result["median_survival"] = median_survival
        result["null_median_survival"] = null_median_survival
        
        return result
        
    elif calculation_type == "Minimum Detectable Effect":
        # Get sample size
        n = params.get("n", 30)
        
        # Calculate MDE (minimum detectable hazard ratio)
        # Approximate calculation based on log-rank test
        # This is the hazard ratio detectable with the given sample size and power
        
        # Calculate expected number of events (simplified)
        study_duration = accrual_time + follow_up_time
        expected_events = n * (1 - math.exp(-lambda0 * study_duration)) * (1 - dropout_rate)
        
        # Calculate MDE (minimum detectable hazard ratio)
        z_alpha = stats.norm.ppf(1 - alpha / 2)  # Two-sided by default for survival
        z_beta = stats.norm.ppf(power)
        
        # Calculate HR assuming proportional hazards
        hr_mde = math.exp(2 * (z_alpha + z_beta) / math.sqrt(expected_events))
        
        # Convert HR to median survival
        median_survival_mde = null_median_survival / hr_mde
        
        # Format results
        result["mde"] = round(hr_mde, 3)
        result["median_survival_mde"] = round(median_survival_mde, 2)
        result["n"] = n
        result["expected_events"] = round(expected_events)
        
        return result
    
    return result
