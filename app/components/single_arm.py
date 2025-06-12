"""
Component module for Single Arm Trial designs.

This module provides UI rendering functions and calculation functions for
Single Arm Trial designs with continuous, binary, and survival outcomes.
"""
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import graphviz

# Import design-specific modules
from core.designs.single_arm.continuous import one_sample_t_test_sample_size
from core.designs.single_arm.continuous import one_sample_t_test_power
from core.designs.single_arm.binary import one_sample_proportion_test_sample_size
from core.designs.single_arm.binary import one_sample_proportion_test_power
from core.designs.single_arm.binary import ahern_sample_size
from core.designs.single_arm.binary import ahern_power
from core.designs.single_arm.binary import simons_two_stage_design
from core.designs.single_arm.binary import simons_power
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
        # Design method
        params["design_method"] = st.selectbox(
            "Design Method",
            ["Standard", "A'Hern", "Simon's Two-Stage"],
            index=0,
            key="design_method_select_single_binary",
            help="'Standard' uses normal approximation. 'A\'Hern' uses exact binomial probabilities for single-stage trials. 'Simon\'s Two-Stage' allows early stopping for futility, both recommended for phase II trials."
        )
        
        # Handle different design methods
        if params["design_method"] == "Standard":
            # Correction methods for standard design
            params["correction"] = st.selectbox(
                "Correction Method",
                ["None", "Continuity", "Exact"],
                index=0,
                key="correction_select_single"
            )
            
            # Alternative hypothesis for standard design
            params["alternative"] = st.selectbox(
                "Alternative Hypothesis",
                ["two-sided", "greater", "less"],
                index=0,
                key="alternative_select_single_binary"
            )
        
        elif params["design_method"] == "A'Hern":
            # Adjust the UI for A'Hern design
            st.info("⚠️ A'Hern's design requires that p > p0. Please ensure your sample proportion (p) "
                   "is greater than your null hypothesis proportion (p0).")
            
            # A'Hern always uses one-sided "greater" alternative hypothesis
            params["alternative"] = "greater"
            params["correction"] = "Exact"
            
            # Make sure p > p0 for A'Hern design
            if params["p"] <= params["p0"]:
                # Swap the values if they're in the wrong order
                temp_p = params["p"]
                params["p"] = max(min(params["p0"] + 0.2, 0.95), params["p0"] + 0.05)  # p0 + (0.05 to 0.2)
                params["p0"] = min(temp_p, params["p0"])
                
                st.warning(f"Values adjusted to satisfy A'Hern design requirements (p > p0). "
                          f"New values: p = {params['p']:.2f}, p0 = {params['p0']:.2f}")
            
            # For A'Hern, display the information about acceptance/rejection boundaries
            st.info("A'Hern's design will calculate the minimum number of responses (r) needed to "
                   "reject the null hypothesis that the response rate is less than or equal to p0.")
        
        else:  # Simon's Two-Stage design
            # Create a container with a highlighted background for the Simon's design section
            simons_container = st.container()
            with simons_container:
                # Add a more visually distinct header
                st.markdown("#### 📊 Simon's Two-Stage Design Settings")
                
                # Parameter validation notice with improved formatting
                st.warning("⚠️ Simon's Two-Stage design requires that p > p0. Please ensure your sample proportion (p) "
                         "is greater than your null hypothesis proportion (p0).")
                
                # Simon's design always uses one-sided "greater" alternative hypothesis
                params["alternative"] = "greater"
                params["correction"] = "Exact"
                
                # Make sure p > p0 for Simon's design
                if params["p"] <= params["p0"]:
                    # Swap the values if they're in the wrong order
                    temp_p = params["p"]
                    params["p"] = max(min(params["p0"] + 0.2, 0.95), params["p0"] + 0.05)  # p0 + (0.05 to 0.2)
                    params["p0"] = min(temp_p, params["p0"])
                    
                    st.error(f"Values automatically adjusted to satisfy Simon's design requirements (p > p0). "
                              f"New values: p = {params['p']:.2f}, p0 = {params['p0']:.2f}")
                
                # Create a two-column layout for parameters
                col1, col2 = st.columns(2)
                
                # Design type selection with enhanced UI (optimal vs. minimax)
                with col1:
                    params["simon_design_type"] = st.radio(
                        "Design Type",
                        ["Optimal", "Minimax"],
                        index=0,
                        key="simon_design_type_select",
                        help="'Optimal' minimizes the expected sample size under H0. 'Minimax' minimizes the maximum sample size."
                    )
                
                # Add a visual diagram to help understand the two-stage design using graphviz
                with col2:
                    st.markdown("##### Design Visualization")
                    
                    # Create a graphviz object for the flowchart
                    simon_graph = graphviz.Digraph()
                    simon_graph.attr('node', shape='box', style='filled', color='lightblue', fontname='Arial', 
                                   fontsize='11', margin='0.2,0.1')
                    simon_graph.attr('edge', fontname='Arial', fontsize='10')
                    
                    # Define the nodes - simplified to focus on key decision points
                    simon_graph.node('stage1', 'Stage 1:\nEnroll n₁ patients')
                    simon_graph.node('decision1', 'Responses > r₁?', shape='diamond', color='lightgreen')
                    simon_graph.node('stop', 'Stop trial\nfor futility', color='#ffcccc')
                    simon_graph.node('stage2', 'Stage 2:\nEnroll n-n₁\nmore patients')
                    simon_graph.node('decision2', 'Total responses > r?', shape='diamond', color='lightgreen')
                    simon_graph.node('ineffective', 'Treatment ineffective\n(Accept H₀)', color='#ffcccc')
                    simon_graph.node('effective', 'Treatment effective\n(Reject H₀)', color='#ccffcc')
                    
                    # Add edges to connect the nodes in the simplified flowchart
                    simon_graph.edge('stage1', 'decision1')
                    simon_graph.edge('decision1', 'stop', label='NO')
                    simon_graph.edge('decision1', 'stage2', label='YES')
                    simon_graph.edge('stage2', 'decision2')
                    simon_graph.edge('decision2', 'ineffective', label='NO')
                    simon_graph.edge('decision2', 'effective', label='YES')
                    
                    # Display the graphviz chart in Streamlit
                    st.graphviz_chart(simon_graph)
                
                # Show explanation of Simon's two-stage design with enhanced formatting
                st.info("""
                **How Simon's Two-Stage Design Works:**
                
                1. **First Stage**: Enroll n₁ patients
                   - If responses ≤ r₁: Stop for futility
                   - If responses > r₁: Continue to stage 2
                
                2. **Second Stage**: Enroll additional patients to reach total n
                   - Final decision: If total responses > r, reject H₀ (treatment works)
                
                This design reduces expected sample size by allowing early stopping when treatment is ineffective.
                """)
    
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
                mean_alt=mean,
                mean_null=null_mean,
                std_dev=adjusted_std_dev,
                alpha=alpha,
                sides=2 if alternative == "two-sided" else 1
            )
        else:
            # Standard single arm design
            power_result = one_sample_t_test_power(
                n=n,
                mean_alt=mean,
                mean_null=null_mean,
                std_dev=std_dev,
                alpha=alpha,
                sides=2 if alternative == "two-sided" else 1
            )
        
        # Format results
        result["power"] = round(power_result, 3)
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
    p = params.get("p", 0.3)  # Sample proportion (p1 in A'Hern's notation)
    p0 = params.get("p0", 0.5)  # Null hypothesis proportion
    alpha = params.get("alpha", 0.05)
    power = params.get("power", 0.8)
    beta = 1 - power  # Convert power to beta for A'Hern's method
    
    # Advanced parameters
    correction = params.get("correction", "None")
    alternative = params.get("alternative", "two-sided")
    design_method = params.get("design_method", "Standard")
    
    # Prepare result dictionary
    result = {}
    
    # Calculate based on calculation_type
    calculation_type = params.get("calculation_type", "Sample Size")
    
    if calculation_type == "Sample Size":
        # Choose the appropriate calculation method based on design_method
        if design_method == "A'Hern":
            try:
                # Use A'Hern's method for phase II trials
                # Note: A'Hern uses p1 as the desirable response rate and p0 as the unacceptable rate
                # Ensure p0 < p for proper calculation
                if p <= p0:
                    result["error"] = "For A'Hern design, p must be greater than p0"
                    result["n"] = None
                    return result
                    
                ahern_result = ahern_sample_size(
                    p0=p0,
                    p1=p,
                    alpha=alpha,
                    beta=beta
                )
                
                # Format results
                result["n"] = ahern_result["n"]
                result["r"] = ahern_result["r"]
                result["actual_alpha"] = round(ahern_result["actual_alpha"], 4)
                result["actual_power"] = round(ahern_result["actual_power"], 4)
                result["absolute_risk_difference"] = round(p - p0, 3)
                result["relative_risk"] = round(p / p0, 3) if p0 > 0 else "Infinity"
                result["design_method"] = "A'Hern"
                
            except Exception as e:
                result["error"] = str(e)
                result["n"] = None
        
        elif design_method == "Simon's Two-Stage":
            try:
                # Use Simon's two-stage design for phase II trials
                # Ensure p0 < p for proper calculation
                if p <= p0:
                    result["error"] = "For Simon's two-stage design, p must be greater than p0"
                    result["n"] = None
                    return result
                
                # Get design type (optimal or minimax)
                simon_design_type = params.get("simon_design_type", "Optimal").lower()
                
                # Calculate Simon's two-stage design
                simons_result = simons_two_stage_design(
                    p0=p0,
                    p1=p,
                    alpha=alpha,
                    beta=beta,
                    design_type=simon_design_type
                )
                
                # Format results
                result["n1"] = simons_result["n1"]  # First stage sample size
                result["r1"] = simons_result["r1"]  # First stage rejection threshold
                result["n"] = simons_result["n"]   # Total sample size
                result["r"] = simons_result["r"]   # Final rejection threshold
                result["EN0"] = round(simons_result["EN0"], 2)  # Expected sample size under H0
                result["PET0"] = round(simons_result["PET0"], 4)  # Probability of early termination under H0
                result["actual_alpha"] = round(simons_result["actual_alpha"], 4)
                result["actual_power"] = round(simons_result["actual_power"], 4)
                result["absolute_risk_difference"] = round(p - p0, 3)
                result["relative_risk"] = round(p / p0, 3) if p0 > 0 else "Infinity"
                result["design_method"] = "Simon's Two-Stage"
                result["design_type"] = simon_design_type.capitalize()
                
            except Exception as e:
                result["error"] = str(e)
                result["n"] = None
                
        else:  # Standard method
            try:
                # Use standard normal approximation method
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
                result["design_method"] = "Standard"
                
            except Exception as e:
                result["error"] = str(e)
                result["n"] = None
        
        return result
        
    elif calculation_type == "Power":
        # Extract sample size from parameters
        n = params.get("n", 30)
        
        # Choose the appropriate power calculation method based on design_method
        if design_method == "A'Hern":
            try:
                # For A'Hern design, we need the rejection threshold r
                # If not provided, we can estimate it using the sample size function
                r = params.get("r", None)
                if r is None:
                    # Calculate r for the given parameters
                    if p <= p0:
                        result["error"] = "For A'Hern design, p must be greater than p0"
                        result["power"] = None
                        return result
                        
                    # Get r from ahern_sample_size function with the user-provided n
                    temp_result = ahern_sample_size(p0=p0, p1=p, alpha=alpha, beta=beta, fixed_n=n)
                    r = temp_result["r"]
                
                # Calculate power for A'Hern design with the given parameters
                power_result = ahern_power(n=n, r=r, p0=p0, p1=p)
                
                # Format results
                result["power"] = round(power_result["power"], 3)
                result["actual_alpha"] = round(power_result["actual_alpha"], 4)
                result["r"] = r
                result["absolute_risk_difference"] = round(p - p0, 3)
                result["relative_risk"] = round(p / p0, 3) if p0 > 0 else "Infinity"
                result["design_method"] = "A'Hern"
                
            except Exception as e:
                result["error"] = str(e)
                result["power"] = None
        
        elif design_method == "Simon's Two-Stage":
            try:
                # For Simon's two-stage design, we need several parameters
                n1 = params.get("n1", None)  # First stage sample size
                r1 = params.get("r1", None)  # First stage rejection threshold
                r = params.get("r", None)    # Final rejection threshold
                
                if n1 is None or r1 is None or r is None:
                    # If parameters aren't provided, we can't calculate power
                    result["error"] = "For Simon's design power calculation, please provide n1, r1, and r"
                    result["power"] = None
                    return result
                
                # Ensure p > p0
                if p <= p0:
                    result["error"] = "For Simon's two-stage design, p must be greater than p0"
                    result["power"] = None
                    return result
                    
                # Calculate power for Simon's two-stage design
                power = simons_power(n1=n1, r1=r1, n=n, r=r, p=p)
                
                # Calculate type I error (alpha) by calculating power at p0
                alpha_actual = simons_power(n1=n1, r1=r1, n=n, r=r, p=p0)
                
                # Calculate expected sample size under H0
                # Probability of early termination under H0
                from scipy.stats import binom
                PET0 = binom.cdf(r1, n1, p0)
                EN0 = n1 + (n - n1) * (1 - PET0)
                
                # Format results
                result["power"] = round(power, 3)
                result["actual_alpha"] = round(alpha_actual, 4)
                result["n1"] = n1
                result["r1"] = r1
                result["n"] = n
                result["r"] = r
                result["EN0"] = round(EN0, 2)  # Expected sample size under H0
                result["PET0"] = round(PET0, 4)  # Probability of early termination under H0
                result["absolute_risk_difference"] = round(p - p0, 3)
                result["relative_risk"] = round(p / p0, 3) if p0 > 0 else "Infinity"
                result["design_method"] = "Simon's Two-Stage"
                
            except Exception as e:
                result["error"] = str(e)
                result["power"] = None
                
        else:  # Standard method
            try:
                # Calculate power using standard method
                power_result = one_sample_proportion_test_power(
                    n=n,
                    p0=p0,
                    p1=p,
                    alpha=alpha,
                    alternative=alternative,
                    correction=correction != "None"
                )
                
                # Format results
                result["power"] = round(power_result["power"], 3)
                result["absolute_risk_difference"] = round(p - p0, 3)
                result["relative_risk"] = round(p / p0, 3) if p0 > 0 else "Infinity"
                result["design_method"] = "Standard"
                
            except Exception as e:
                result["error"] = str(e)
                result["power"] = None
        
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


def generate_cli_code_single_arm_continuous(params):
    """
    Generate reproducible CLI code for single-arm continuous outcomes.
    """
    calculation_type = params.get('calculation_type', 'Sample Size')
    mean = params.get('mean', 0.0)
    std_dev = params.get('std_dev', 1.0)
    null_mean = params.get('null_mean', 0.0)
    alpha = params.get('alpha', 0.05)
    power = params.get('power', 0.8)
    n = params.get('n')
    
    script = f'''#!/usr/bin/env python3
"""
Single-Arm Trial Analysis - Continuous Outcome ({calculation_type})
Generated by DesignPower CLI

This script performs {calculation_type.lower()} calculation for a single-arm trial
with continuous outcomes using one-sample t-test.
"""

from core.designs.single_arm.continuous import (
    one_sample_t_test_sample_size,
    one_sample_t_test_power
)

def main():
    """Main analysis function."""
    
    # Study parameters
    mean = {mean}  # Sample mean
    null_mean = {null_mean}  # Null hypothesis mean
    std_dev = {std_dev}  # Standard deviation
    alpha = {alpha}  # Significance level
    power = {power}  # Desired power
'''
    
    if calculation_type == 'Sample Size':
        script += f'''    
    # Calculate sample size
    result = one_sample_t_test_sample_size(
        mean=mean,
        null_mean=null_mean,
        std_dev=std_dev,
        alpha=alpha,
        power=power
    )
    
    print("=== Single-Arm Trial Sample Size Calculation ===")
    print(f"Sample mean: {{mean}}")
    print(f"Null hypothesis mean: {{null_mean}}")
    print(f"Standard deviation: {{std_dev}}")
    print(f"Effect size (Cohen's d): {{(mean - null_mean) / std_dev:.3f}}")
    print(f"Significance level (α): {{alpha}}")
    print(f"Power (1-β): {{power}}")
    print(f"\\nRequired sample size: {{result['n']}}")
'''
    
    elif calculation_type == 'Power':
        script += f'''
    n = {n}  # Sample size
    
    # Calculate power
    result = one_sample_t_test_power(
        n=n,
        mean=mean,
        null_mean=null_mean,
        std_dev=std_dev,
        alpha=alpha
    )
    
    print("=== Single-Arm Trial Power Calculation ===")
    print(f"Sample size: {{n}}")
    print(f"Sample mean: {{mean}}")
    print(f"Null hypothesis mean: {{null_mean}}")
    print(f"Standard deviation: {{std_dev}}")
    print(f"Effect size (Cohen's d): {{(mean - null_mean) / std_dev:.3f}}")
    print(f"Significance level (α): {{alpha}}")
    print(f"\\nCalculated power: {{result['power']:.3f}}")
'''
    
    script += '''

if __name__ == "__main__":
    main()
'''
    
    return script


def generate_cli_code_single_arm_binary(params):
    """
    Generate reproducible CLI code for single-arm binary outcomes.
    """
    calculation_type = params.get('calculation_type', 'Sample Size')
    p = params.get('p', 0.3)
    p0 = params.get('p0', 0.1)
    alpha = params.get('alpha', 0.05)
    power = params.get('power', 0.8)
    design_method = params.get('design_method', 'standard')
    simon_type = params.get('simon_design_type', 'optimal')
    n = params.get('n')
    n1 = params.get('n1')
    r1 = params.get('r1')
    r = params.get('r')
    
    script = f'''#!/usr/bin/env python3
"""
Single-Arm Trial Analysis - Binary Outcome ({calculation_type})
Generated by DesignPower CLI

This script performs {calculation_type.lower()} calculation for a single-arm trial
with binary outcomes using {design_method} method.
"""

from core.designs.single_arm.binary import (
    one_sample_proportion_test_sample_size,
    one_sample_proportion_test_power,
    ahern_sample_size,
    ahern_power,
    simons_two_stage_design,
    simons_power
)

def main():
    """Main analysis function."""
    
    # Study parameters
    p = {p}  # Expected proportion
    p0 = {p0}  # Null hypothesis proportion
    alpha = {alpha}  # Significance level
    power = {power}  # Desired power
    design_method = "{design_method}"  # Design method
'''

    if design_method.lower() == 'ahern':
        script += f'''
    
    # A'Hern's Exact Single-Stage Design
    if design_method.lower() == "ahern":
        beta = 1 - power
        
'''
        if calculation_type == 'Sample Size':
            script += '''        # Calculate sample size using A'Hern method
        result = ahern_sample_size(
            p0=p0,
            p1=p,
            alpha=alpha,
            beta=beta
        )
        
        print("=== A'Hern Single-Stage Design Sample Size ===")
        print(f"Expected proportion (p1): {p}")
        print(f"Null hypothesis proportion (p0): {p0}")
        print(f"Absolute risk difference: {p - p0:.3f}")
        print(f"Relative risk: {p / p0:.3f}")
        print(f"Significance level (α): {alpha}")
        print(f"Power (1-β): {power}")
        print(f"\\nRequired sample size (n): {result['n']}")
        print(f"Rejection threshold (r): {result['r']}")
        print(f"Actual α: {result['actual_alpha']:.4f}")
        print(f"Actual power: {result['actual_power']:.4f}")
        print(f"\\nDecision rule: Reject H0 if ≥ {result['r']} responses observed in {result['n']} patients")
'''
        elif calculation_type == 'Power':
            script += f'''
        n = {n}  # Sample size
        r = {r}  # Rejection threshold
        
        # Calculate power using A'Hern method
        result = ahern_power(
            n=n,
            r=r,
            p0=p0,
            p1=p
        )
        
        print("=== A'Hern Single-Stage Design Power Calculation ===")
        print(f"Sample size (n): {{n}}")
        print(f"Rejection threshold (r): {{r}}")
        print(f"Expected proportion (p1): {{p}}")
        print(f"Null hypothesis proportion (p0): {{p0}}")
        print(f"\\nCalculated power: {{result['power']:.3f}}")
        print(f"Actual α: {{result['actual_alpha']:.4f}}")
'''

    elif design_method.lower() == 'simons':
        script += f'''
    
    # Simon's Two-Stage Design
    simon_type = "{simon_type}"  # optimal or minimax
    
'''
        if calculation_type == 'Sample Size':
            script += '''        beta = 1 - power
        
        # Calculate sample size using Simon's two-stage design
        result = simons_two_stage_design(
            p0=p0,
            p1=p,
            alpha=alpha,
            beta=beta,
            design_type=simon_type.lower()
        )
        
        print("=== Simon's Two-Stage Design Sample Size ===")
        print(f"Design type: {simon_type.capitalize()}")
        print(f"Expected proportion (p1): {p}")
        print(f"Null hypothesis proportion (p0): {p0}")
        print(f"Absolute risk difference: {p - p0:.3f}")
        print(f"Relative risk: {p / p0:.3f}")
        print(f"Significance level (α): {alpha}")
        print(f"Power (1-β): {power}")
        print(f"\\n--- Stage 1 ---")
        print(f"Sample size (n1): {result['n1']}")
        print(f"Rejection threshold (r1): {result['r1']}")
        print(f"\\n--- Final Analysis ---")
        print(f"Total sample size (n): {result['n']}")
        print(f"Final rejection threshold (r): {result['r']}")
        print(f"\\n--- Operating Characteristics ---")
        print(f"Expected sample size under H0: {result['EN0']:.1f}")
        print(f"Probability of early termination under H0: {result['PET0']:.3f}")
        print(f"Actual α: {result['actual_alpha']:.4f}")
        print(f"Actual power: {result['actual_power']:.4f}")
        print(f"\\nDecision rules:")
        print(f"  Stage 1: Stop for futility if ≤ {result['r1']} responses in {result['n1']} patients")
        print(f"  Final: Reject H0 if > {result['r']} total responses in {result['n']} patients")
'''
        elif calculation_type == 'Power':
            script += f'''
        n1 = {n1}  # Stage 1 sample size
        r1 = {r1}  # Stage 1 rejection threshold
        n = {n}  # Total sample size
        r = {r}  # Final rejection threshold
        
        # Calculate power using Simon's two-stage design
        power_val = simons_power(n1=n1, r1=r1, n=n, r=r, p=p)
        
        print("=== Simon's Two-Stage Design Power Calculation ===")
        print(f"Stage 1 sample size (n1): {{n1}}")
        print(f"Stage 1 threshold (r1): {{r1}}")
        print(f"Total sample size (n): {{n}}")
        print(f"Final threshold (r): {{r}}")
        print(f"Expected proportion (p1): {{p}}")
        print(f"Null hypothesis proportion (p0): {{p0}}")
        print(f"\\nCalculated power: {{power_val:.3f}}")
'''

    else:  # Standard
        script += '''
    
    # Standard normal approximation method
'''
        if calculation_type == 'Sample Size':
            script += '''    # Calculate sample size using standard method
    sample_size = one_sample_proportion_test_sample_size(
        p0=p0,
        p1=p,
        alpha=alpha,
        power=power
    )
    
    print("=== Standard Single-Arm Design Sample Size ===")
    print(f"Expected proportion (p1): {p}")
    print(f"Null hypothesis proportion (p0): {p0}")
    print(f"Absolute risk difference: {p - p0:.3f}")
    print(f"Relative risk: {p / p0:.3f}")
    print(f"Significance level (α): {alpha}")
    print(f"Power (1-β): {power}")
    print(f"\\nRequired sample size: {sample_size}")
'''
        elif calculation_type == 'Power':
            script += f'''
    n = {n}  # Sample size
    
    # Calculate power using standard method
    power_val = one_sample_proportion_test_power(
        n=n,
        p0=p0,
        p1=p,
        alpha=alpha
    )
    
    print("=== Standard Single-Arm Design Power Calculation ===")
    print(f"Sample size: {{n}}")
    print(f"Expected proportion (p1): {{p}}")
    print(f"Null hypothesis proportion (p0): {{p0}}")
    print(f"Significance level (α): {{alpha}}")
    print(f"\\nCalculated power: {{power_val:.3f}}")
'''
    
    script += '''

if __name__ == "__main__":
    main()
'''
    
    return script
def generate_cli_code_single_arm_survival(params):
    """
    Generate reproducible CLI code for single-arm survival outcomes.
    """
    calculation_type = params.get('calculation_type', 'Sample Size')
    median_null = params.get('median_null', 6.0)
    median_alt = params.get('median_alt', 12.0)
    enrollment_period = params.get('enrollment_period', 12.0)
    follow_up_period = params.get('follow_up_period', 12.0)
    dropout_rate = params.get('dropout_rate', 0.1)
    alpha = params.get('alpha', 0.05)
    power = params.get('power', 0.8)
    sides = params.get('sides', 2)
    n = params.get('n')
    
    script = f'''#!/usr/bin/env python3
"""
Single-Arm Survival Analysis ({calculation_type})
Generated by DesignPower CLI

This script performs {calculation_type.lower()} calculation for a single-arm survival study
using log-rank test methodology.
"""

from core.designs.single_arm.survival import (
    one_sample_survival_test_sample_size,
    one_sample_survival_test_power
)
import math

def main():
    """Main analysis function."""
    
    # Study parameters
    median_null = {median_null}  # Null hypothesis median survival (months)
    median_alt = {median_alt}  # Alternative hypothesis median survival (months)
    enrollment_period = {enrollment_period}  # Enrollment period (months)
    follow_up_period = {follow_up_period}  # Follow-up period (months)
    dropout_rate = {dropout_rate}  # Expected dropout rate
    alpha = {alpha}  # Significance level
    power = {power}  # Desired power
    sides = {sides}  # One-sided (1) or two-sided (2) test
'''
    
    if calculation_type == 'Sample Size':
        script += '''    
    # Calculate sample size
    result = one_sample_survival_test_sample_size(
        median_null=median_null,
        median_alt=median_alt,
        enrollment_period=enrollment_period,
        follow_up_period=follow_up_period,
        dropout_rate=dropout_rate,
        alpha=alpha,
        power=power,
        sides=sides
    )
    
    print("=== Single-Arm Survival Study Sample Size ===")
    print(f"Null hypothesis median survival: {median_null} months")
    print(f"Alternative hypothesis median survival: {median_alt} months")
    print(f"Hazard ratio: {result['hazard_ratio']:.3f}")
    print(f"Enrollment period: {enrollment_period} months")
    print(f"Follow-up period: {follow_up_period} months")
    print(f"Expected dropout rate: {dropout_rate*100:.1f}%")
    print(f"Significance level (α): {alpha}")
    print(f"Power (1-β): {power}")
    print(f"Test sides: {sides}")
    print(f"\\nRequired sample size: {result['sample_size']}")
    print(f"Expected number of events: {result['events']}")
'''
    
    elif calculation_type == 'Power':
        script += f'''
    n = {n}  # Sample size
    
    # Calculate power
    result = one_sample_survival_test_power(
        n=n,
        median_null=median_null,
        median_alt=median_alt,
        enrollment_period=enrollment_period,
        follow_up_period=follow_up_period,
        dropout_rate=dropout_rate,
        alpha=alpha,
        sides=sides
    )
    
    print("=== Single-Arm Survival Study Power Calculation ===")
    print(f"Sample size: {{n}}")
    print(f"Null hypothesis median survival: {{median_null}} months")
    print(f"Alternative hypothesis median survival: {{median_alt}} months")
    print(f"Hazard ratio: {{result['hazard_ratio']:.3f}}")
    print(f"Enrollment period: {{enrollment_period}} months")
    print(f"Follow-up period: {{follow_up_period}} months")
    print(f"Expected dropout rate: {{dropout_rate*100:.1f}}%")
    print(f"Significance level (α): {{alpha}}")
    print(f"Test sides: {{sides}}")
    print(f"\\nCalculated power: {{result['power']:.3f}}")
    print(f"Expected number of events: {{result['expected_events']:.0f}}")
'''
    
    script += '''

if __name__ == "__main__":
    main()
'''
    
    return script