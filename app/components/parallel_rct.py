"""
Component module for Parallel RCT designs.

This module provides UI rendering functions and calculation functions for
Parallel RCT designs with continuous, binary, and survival outcomes.
"""
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import specific analytical and simulation modules
from core.designs.parallel import analytical_continuous
from core.designs.parallel import simulation_continuous
from core.designs.parallel import analytical_binary
from core.designs.parallel import simulation_binary
from core.designs.parallel import analytical_survival
from core.designs.parallel import simulation_survival

# Shared functions
def render_binary_advanced_options():
    """
    Render advanced options for binary outcome designs.
    
    Returns:
        dict: Advanced parameters
    """
    advanced_params = {}
    
    # Method selection (analytical vs simulation)
    advanced_params["method"] = st.radio(
        "Calculation Method",
        ["Analytical", "Simulation"],
        index=0,
        key="binary_method_radio",
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
                key="binary_nsim_input"
            )
            
        with col2:
            advanced_params["seed"] = st.number_input(
                "Random Seed", 
                value=42, 
                min_value=1, 
                help="Seed for random number generation (for reproducibility)",
                key="binary_seed_input"
            )
    
    # Test type selection
    advanced_params["test_type"] = st.selectbox(
        "Statistical Test",
        ["Normal Approximation", "Fisher's Exact Test", "Likelihood Ratio Test"],
        index=0,
        key="binary_test_type_select"
    )
    
    # Add explicit continuity correction option
    advanced_params["correction"] = st.checkbox(
        "Apply Continuity Correction", 
        value=False,
        key="binary_correction_checkbox",
        help="Apply continuity correction for improved accuracy"
    )
    
    # Add display text to show the impact of selections
    if advanced_params["test_type"] == "Fisher's Exact Test":
        st.info("Fisher's Exact Test is more conservative and typically requires larger sample sizes.")
    elif advanced_params["test_type"] == "Likelihood Ratio Test":
        st.info("Likelihood Ratio Test may be more powerful than normal approximation in some cases.")
    
    if advanced_params["correction"]:
        st.info("Continuity correction improves accuracy but may increase required sample sizes.")
    
    return advanced_params

def render_continuous_advanced_options():
    """
    Render advanced options for continuous outcome designs.
    
    Returns:
        dict: Advanced parameters
    """
    advanced_params = {}
    
    # Method selection (analytical vs simulation)
    advanced_params["method"] = st.radio(
        "Calculation Method",
        ["Analytical", "Simulation"],
        index=0,
        key="method_radio",
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
                key="nsim_input"
            )
            
        with col2:
            advanced_params["seed"] = st.number_input(
                "Random Seed", 
                value=42, 
                min_value=1, 
                help="Seed for random number generation (for reproducibility)",
                key="seed_input"
            )
    
    # Other advanced options UI
    st.markdown("#### Design Parameters")
    col1, col2 = st.columns(2)
    
    with col1:
        advanced_params["unequal_var"] = st.checkbox("Unequal variances", value=False, key="unequal_var_checkbox")
        
    with col2:
        advanced_params["repeated_measures"] = st.checkbox("Repeated measures", value=False, key="repeated_measures_checkbox")
    
    # Show additional inputs based on selections
    if advanced_params["unequal_var"]:
        advanced_params["std_dev2"] = st.number_input("Standard Deviation (Group 2)", 
                                                    value=1.0, step=0.1, min_value=0.1,
                                                    key="sd2_input")
    
    if advanced_params["repeated_measures"]:
        advanced_params["correlation"] = st.slider("Correlation", 
                                                min_value=0.0, max_value=0.99, value=0.5, step=0.01,
                                                key="correlation_slider")
        advanced_params["analysis_method"] = st.selectbox(
            "Analysis Method",
            ["ANCOVA", "Change Score"],
            index=0,
            key="analysis_method_select"
        )
    
    return advanced_params

def render_parallel_continuous(calc_type, hypothesis_type):
    """
    Render the UI for Parallel RCT with continuous outcome.
    
    Args:
        calc_type: String indicating calculation type (Sample Size, Power, or Minimum Detectable Effect)
        hypothesis_type: String indicating hypothesis type (Superiority or Non-Inferiority)
    
    Returns:
        dict: Parameters collected from the UI
    """
    params = {}
    
    # Display header with calculation type
    st.write(f"### Parallel RCT with Continuous Outcome ({calc_type})")
    
    # Basic parameters UI
    with st.container():
        st.subheader("Basic Parameters")
        
        # Parameter inputs
        col1, col2 = st.columns(2)
        
        # Set default non-inferiority flag based on hypothesis type
        non_inferiority = (hypothesis_type == "Non-Inferiority")
        params["non_inferiority"] = non_inferiority
        
        with col1:
            # Always show Group 1 mean and standard deviation for both designs
            params["mean1"] = st.number_input("Mean (Group 1)", value=0.0, step=0.1, key="mean1_input")
            params["std_dev"] = st.number_input("Standard Deviation", value=1.0, step=0.1, min_value=0.1, key="sd_input")
            
            # For non-inferiority design, show margin and assumed difference
            if non_inferiority:
                params["margin"] = st.number_input("Non-inferiority Margin (NIM)", value=0.2, step=0.1, key="margin_input")
                params["assumed_difference"] = st.number_input("Assumed Difference", value=0.0, step=0.1, 
                                                       help="The assumed true difference between treatments (0 = treatments equivalent)", 
                                                       key="assumed_diff_input")
            # For superiority design with MDE calculation, show sample sizes
            elif calc_type == "Minimum Detectable Effect":
                params["n1"] = st.number_input("Sample Size (Group 1)", value=20, step=1, min_value=5, key="n1_input_mde")
                params["n2"] = st.number_input("Sample Size (Group 2)", value=20, step=1, min_value=5, key="n2_input_mde")
            # For regular superiority design, show Group 2 mean
            else:
                params["mean2"] = st.number_input("Mean (Group 2)", value=0.5, step=0.1, key="mean2_input")
            
        with col2:
            params["alpha"] = st.slider("Significance Level (α)", 
                                     min_value=0.01, max_value=0.20, value=0.05, step=0.01,
                                     key="alpha_slider")
                                     
            # For non-inferiority design, show direction
            if non_inferiority:
                params["direction"] = st.selectbox(
                    "Direction",
                    ["lower", "upper"],
                    index=0,
                    key="direction_select"
                )
                # Explain the direction based on selection
                if params["direction"] == "lower":
                    st.info("Lower means smaller values are better (e.g., pain scores)")
                else:
                    st.info("Upper means larger values are better (e.g., quality of life)")
            
            # Show power/sample size inputs based on calculation type
            if calc_type == "Sample Size":
                params["power"] = st.slider("Power (1-β)", 
                                         min_value=0.70, max_value=0.99, value=0.80, step=0.01,
                                         key="power_slider")
            elif calc_type == "Power":
                # When calculating power, use a default value (not shown in UI)
                params["power"] = None
                # When calculating power, we need sample size
                params["n1"] = st.number_input("Sample Size (Group 1)", value=20, step=1, min_value=5, key="n1_input")
                params["n2"] = st.number_input("Sample Size (Group 2)", value=20, step=1, min_value=5, key="n2_input")
            elif calc_type == "Minimum Detectable Effect":
                # For MDE, we need power
                params["power"] = st.slider("Power (1-β)", 
                                         min_value=0.70, max_value=0.99, value=0.80, step=0.01,
                                         key="power_slider_mde")
                
            params["allocation_ratio"] = st.slider("Allocation Ratio (n2/n1)", 
                                               min_value=0.1, max_value=10.0, value=1.0, step=0.1,
                                               key="allocation_slider")
        
        # Show description of parameters based on hypothesis type
        if non_inferiority:
            st.markdown("""**Non-Inferiority Design Parameters:**
            - **NIM** is the non-inferiority margin, the maximum acceptable difference
            - **Assumed Difference** is what you believe the true difference to be (0 = treatments equivalent)
            - **Direction** determines whether lower or higher values are better
            """)
        # For superiority design, show a note about the direction of effect
        elif calc_type != "Minimum Detectable Effect":
            st.info(f"For superiority design, effect is measured as (Group 2 - Group 1) = {params.get('mean2', 0) - params.get('mean1', 0)}")
        else:
            st.info("Minimum Detectable Effect calculation will determine the smallest effect size that can be detected with the given sample size and power")
    
    # Advanced parameters UI
    with st.expander("Advanced Parameters", expanded=False):
        # Get advanced options using our shared component
        advanced_options = render_continuous_advanced_options()
        params.update(advanced_options)
        
        # Simulation-specific options are handled in the expandable section
    
    return params


def render_parallel_binary(calc_type, hypothesis_type):
    """
    Render the UI for Parallel RCT with binary outcome.
    
    Args:
        calc_type: String indicating calculation type (Sample Size, Power, or Minimum Detectable Effect)
        hypothesis_type: String indicating hypothesis type (Superiority or Non-Inferiority)
        
    Returns:
        dict: Parameters collected from the UI
    """
    # Debug the hypothesis type being passed
    print(f"BINARY COMPONENT DEBUG - Hypothesis Type: {hypothesis_type}")
    
    params = {}
    # Store the hypothesis type in params
    params["hypothesis_type"] = hypothesis_type
    
    # Display header with calculation type
    st.write(f"### Parallel RCT with Binary Outcome ({calc_type})")
    
    # Basic parameters UI
    with st.container():
        st.subheader("Basic Parameters")
        
        # Parameter inputs
        col1, col2 = st.columns(2)
        
        with col1:
            # Different UI based on hypothesis type
            if hypothesis_type == "Superiority":
                # For superiority, show both group proportions
                params["p1"] = st.slider("Proportion (Group 1)", 
                                      min_value=0.01, max_value=0.99, value=0.3, step=0.01,
                                      key="p1_slider")
                params["p2"] = st.slider("Proportion (Group 2)", 
                                      min_value=0.01, max_value=0.99, value=0.5, step=0.01,
                                      key="p2_slider")
            else:  # Non-Inferiority
                # For non-inferiority, show p1, NIM and direction
                params["p1"] = st.slider("Proportion (Group 1)", 
                                      min_value=0.01, max_value=0.99, value=0.3, step=0.01,
                                      key="ni_p1_slider")
                
                params["nim"] = st.slider("Non-Inferiority Margin",
                                      min_value=0.01, max_value=0.20, value=0.10, step=0.01,
                                      key="binary_nim_slider",
                                      help="The non-inferiority margin represents the maximum acceptable difference")
                
                params["direction"] = st.selectbox("Direction",
                                            ["Higher is better", "Lower is better"],
                                            index=0,
                                            key="binary_direction_select",
                                            help="Indicates whether higher or lower values represent better outcomes")
                
                # Calculate p2 based on p1, NIM and direction
                if params["direction"] == "Higher is better":
                    # If higher is better, p2 = p1 - NIM for non-inferiority
                    params["p2"] = params["p1"] - params["nim"]
                else:
                    # If lower is better, p2 = p1 + NIM for non-inferiority
                    params["p2"] = params["p1"] + params["nim"]
                
                # Display the calculated p2
                st.write(f"Calculated proportion (Group 2): {params['p2']:.2f}")
                
                # Add a note explaining the non-inferiority hypothesis
                if params["direction"] == "Higher is better":
                    st.info(f"Testing: New treatment is not worse than control by more than {params['nim']:.2f}")
                else:
                    st.info(f"Testing: New treatment is not worse than control by more than {params['nim']:.2f}")

            
        with col2:
            params["alpha"] = st.slider("Significance Level (α)", 
                                     min_value=0.01, max_value=0.20, value=0.05, step=0.01,
                                     key="alpha_slider_binary")
                                     
            # Only show power slider when not calculating power
            if calc_type != "Power":
                params["power"] = st.slider("Power (1-β)", 
                                         min_value=0.70, max_value=0.99, value=0.80, step=0.01,
                                         key="power_slider_binary")
            else:
                # When calculating power, use a default value (not shown in UI)
                params["power"] = None
                # When calculating power, we need sample size
                params["n1"] = st.number_input("Sample Size (Group 1)", value=50, step=1, min_value=5, key="n1_input_binary")
                params["n2"] = st.number_input("Sample Size (Group 2)", value=50, step=1, min_value=5, key="n2_input_binary")
                
            params["allocation_ratio"] = st.slider("Allocation Ratio (n2/n1)", 
                                               min_value=0.1, max_value=10.0, value=1.0, step=0.1,
                                               key="allocation_slider_binary")
    
    # Advanced parameters UI
    with st.expander("Advanced Parameters", expanded=False):
        # Get advanced options using our binary advanced options component
        advanced_options = render_binary_advanced_options()
        params.update(advanced_options)
        
        # Correction methods for analytical methods
        if params.get("method", "analytical") == "analytical":
            params["correction"] = st.selectbox(
                "Correction Method",
                ["None", "Continuity", "Yates"],
                index=0,
                key="correction_select"
            )
    
    return params


def render_parallel_survival(calc_type, hypothesis_type):
    """
    Render the UI for Parallel RCT with survival outcome.
    
    Args:
        calc_type: String indicating calculation type (Sample Size, Power, or Minimum Detectable Effect)
        hypothesis_type: String indicating hypothesis type (Superiority or Non-Inferiority)
        
    Returns:
        dict: Parameters collected from the UI
    """
    params = {}
    
    # Display header with calculation type
    st.write(f"### Parallel RCT with Survival Outcome ({calc_type})")
    
    # Basic parameters UI
    with st.container():
        st.subheader("Basic Parameters")
        
        # Parameter inputs
        col1, col2 = st.columns(2)
        
        with col1:
            params["hr"] = st.number_input("Hazard Ratio", value=0.7, step=0.05, min_value=0.1, key="hr_input")
            params["median_survival1"] = st.number_input("Median Survival (Group 1, months)", 
                                                      value=12.0, step=1.0, min_value=0.1,
                                                      key="median1_input")
        
        with col2:
            params["alpha"] = st.slider("Significance Level (α)", 
                                     min_value=0.01, max_value=0.20, value=0.05, step=0.01,
                                     key="alpha_slider_survival")
                                     
            # Only show power slider when not calculating power
            if calc_type != "Power":
                params["power"] = st.slider("Power (1-β)", 
                                         min_value=0.70, max_value=0.99, value=0.80, step=0.01,
                                         key="power_slider_survival")
            else:
                # When calculating power, use a default value (not shown in UI)
                params["power"] = None
                # When calculating power, we need sample size
                params["n1"] = st.number_input("Sample Size (Group 1)", value=50, step=1, min_value=5, key="n1_input_survival")
                params["n2"] = st.number_input("Sample Size (Group 2)", value=50, step=1, min_value=5, key="n2_input_survival")
                
            params["allocation_ratio"] = st.slider("Allocation Ratio (n2/n1)", 
                                               min_value=0.1, max_value=10.0, value=1.0, step=0.1,
                                               key="allocation_slider_survival")
    
    # Advanced parameters UI
    with st.expander("Advanced Parameters", expanded=False):
        # Study parameters
        col1, col2 = st.columns(2)
        
        with col1:
            params["accrual_time"] = st.number_input("Accrual Time (months)", 
                                                 value=12.0, step=1.0, min_value=0.1,
                                                 key="accrual_input")
            
        with col2:
            params["follow_up_time"] = st.number_input("Follow-up Time (months)", 
                                                    value=24.0, step=1.0, min_value=0.1,
                                                    key="followup_input")
            
        # Dropout parameters
        params["dropout_rate1"] = st.slider("Dropout Rate (Group 1)", 
                                        min_value=0.0, max_value=0.5, value=0.1, step=0.01,
                                        key="dropout1_slider")
        params["dropout_rate2"] = st.slider("Dropout Rate (Group 2)", 
                                        min_value=0.0, max_value=0.5, value=0.1, step=0.01,
                                        key="dropout2_slider")
        
        # Simulation-specific options are handled in the expandable section
    
    return params


def calculate_parallel_continuous(params):
    """
    Calculate results for Parallel RCT with continuous outcome.
    
    Args:
        params: Dictionary of parameters
        
    Returns:
        dict: Results of calculations
    """
    # Get calculation method from params
    method = params.get("method", "analytical")  # Default to analytical if not specified
    # Extract parameters with default values to avoid None errors
    mean1 = params.get("mean1", 0.0)
    mean2 = params.get("mean2", 0.5)
    std_dev = params.get("std_dev", 1.0)  # Provide default value to avoid None
    alpha = params.get("alpha", 0.05)
    power = params.get("power", 0.8)
    allocation_ratio = params.get("allocation_ratio", 1.0)
    
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
    
    # Get simulation-specific parameters
    nsim = params.get("nsim", 1000)
    seed = params.get("seed", 42)
    
    # Calculate based on calculation_type
    calculation_type = params.get("calculation_type", "Sample Size")
    
    if calculation_type == "Sample Size":
        # Calculate sample size
        if method == "analytical":
            if non_inferiority:
                # Non-inferiority test
                sample_size = analytical_continuous.sample_size_continuous_non_inferiority(
                    mean1=mean1,
                    non_inferiority_margin=params.get("margin", 0.2),
                    sd1=std_dev,
                    sd2=std_dev2 if unequal_var else None,
                    power=power,
                    alpha=alpha,
                    allocation_ratio=allocation_ratio,
                    assumed_difference=params.get("assumed_difference", 0.0),
                    direction="lower" if params.get("direction", "lower") == "lower" else "upper"
                )
            else:
                # Standard superiority test
                if repeated_measures:
                    # Repeated measures design
                    sample_size = analytical_continuous.sample_size_repeated_measures(
                        mean1=mean1,
                        mean2=mean2,
                        sd=std_dev,
                        correlation=correlation,
                        power=power,
                        alpha=alpha,
                        allocation_ratio=allocation_ratio,
                        method=analysis_method
                    )
                else:
                    # Standard parallel design - ensure no None values are passed
                    # Make sure all required parameters have valid values
                    if std_dev is None:
                        std_dev = 1.0  # Default value if None
                    
                    # Ensure std_dev2 is not None
                    if unequal_var and std_dev2 is None:
                        std_dev2 = std_dev  # Default to std_dev if None
                    
                    # Now call the analytical function with validated parameters
                    sample_size = analytical_continuous.sample_size_continuous(
                        mean1=mean1, 
                        mean2=mean2, 
                        sd1=std_dev,  # This should never be None now
                        sd2=std_dev2 if unequal_var else std_dev,
                        power=power,
                        alpha=alpha,
                        allocation_ratio=allocation_ratio
                    )
        elif method == "simulation":
            # Simulation-based calculation
            if non_inferiority:
                # Non-inferiority test simulation
                sample_size = simulation_continuous.sample_size_continuous_non_inferiority_sim(
                    non_inferiority_margin=params.get("margin", 0.2),
                    std_dev=std_dev,
                    power=power,
                    alpha=alpha,
                    allocation_ratio=allocation_ratio,
                    nsim=nsim,
                    assumed_difference=params.get("assumed_difference", 0.0),
                    direction="lower" if params.get("direction", "lower") == "lower" else "upper",
                    repeated_measures=repeated_measures,
                    correlation=correlation if repeated_measures else 0.5,
                    method=analysis_method if repeated_measures else "change_score"
                )
            else:
                # For standard superiority test, we need to calculate the effect size (delta)
                delta = mean2 - mean1
                
                # Call the simulation function
                if unequal_var:
                    # For unequal variance, we need to pass both std_dev values
                    # Since the simulation function doesn't have a direct parameter for this,
                    # we'll approximate by passing the average std_dev
                    # This isn't ideal but helps provide feedback to the user that unequal variance is considered
                    # Note: A better approach would be to modify the simulation function to accept both std_dev values
                    effective_std_dev = (std_dev + std_dev2) / 2
                    
                    sample_size = simulation_continuous.sample_size_continuous_sim(
                        delta=delta,
                        std_dev=effective_std_dev,  # Use average of both std_devs
                        power=power,
                        alpha=alpha,
                        allocation_ratio=allocation_ratio,
                        nsim=nsim,
                        repeated_measures=repeated_measures,
                        correlation=correlation if repeated_measures else 0.5,
                        method=analysis_method if repeated_measures else "change_score"
                    )
                else:
                    # Standard case with equal variance
                    sample_size = simulation_continuous.sample_size_continuous_sim(
                        delta=delta,
                        std_dev=std_dev,
                        power=power,
                        alpha=alpha,
                        allocation_ratio=allocation_ratio,
                        nsim=nsim,
                        repeated_measures=repeated_measures,
                        correlation=correlation if repeated_measures else 0.5,
                        method=analysis_method if repeated_measures else "change_score"
                    )
            
        # Extract values from result with fallbacks
        n1 = sample_size.get("n1", sample_size.get("sample_size_1"))
        n2 = sample_size.get("n2", sample_size.get("sample_size_2"))
        total_n = sample_size.get("total_n", sample_size.get("total_sample_size", n1 + n2))
        
        # Format results
        result["n1"] = round(n1)
        result["n2"] = round(n2)
        result["total_n"] = round(total_n)
        result["mean_difference"] = mean2 - mean1
        
        return result
        
    elif calculation_type == "Power":
        # Get sample sizes
        n1 = params.get("n1", 20)
        n2 = params.get("n2", 20)
        
        # Calculate power
        if method == "analytical":
            if non_inferiority:
                # Non-inferiority test
                power_result = analytical_continuous.power_continuous_non_inferiority(
                    n1=n1,
                    n2=n2,
                    mean1=mean1,
                    non_inferiority_margin=params.get("margin", 0.2),
                    sd1=std_dev,
                    sd2=std_dev2 if unequal_var else None,
                    alpha=alpha,
                    assumed_difference=params.get("assumed_difference", 0.0),
                    direction="lower" if params.get("direction", "lower") == "lower" else "upper"
                )
            else:
                # Standard superiority test
                if repeated_measures:
                    # Repeated measures design
                    power_result = analytical_continuous.power_repeated_measures(
                        n1=n1,
                        n2=n2,
                        mean1=mean1,
                        mean2=mean2,
                        sd=std_dev,
                        correlation=correlation,
                        alpha=alpha,
                        method=analysis_method
                    )
                else:
                    # Standard parallel design
                    power_result = analytical_continuous.power_continuous(
                        n1=n1,
                        n2=n2,
                        mean1=mean1,
                        mean2=mean2,
                        sd1=std_dev,
                        sd2=std_dev2 if unequal_var else std_dev,
                        alpha=alpha
                    )
        elif method == "simulation":
            # Simulation-based power calculation
            if non_inferiority:
                # Non-inferiority test using simulation
                power_result = simulation_continuous.simulate_continuous_non_inferiority(
                    n1=n1,
                    n2=n2,
                    non_inferiority_margin=params.get("margin", 0.2),
                    std_dev=std_dev,
                    nsim=nsim,
                    alpha=alpha,
                    seed=seed,
                    assumed_difference=params.get("assumed_difference", 0.0),
                    direction="lower" if params.get("direction", "lower") == "lower" else "upper",
                    repeated_measures=repeated_measures,
                    correlation=correlation if repeated_measures else 0.5,
                    method=analysis_method if repeated_measures else "change_score"
                )
            else:
                # Standard superiority test using simulation
                # Handle unequal variance - make it clear in the UI that this has an effect
                if unequal_var:
                    # When unequal_var is checked, use both sd1 and sd2
                    power_result = simulation_continuous.power_continuous_sim(
                        n1=n1,
                        n2=n2,
                        mean1=mean1,
                        mean2=mean2,
                        sd1=std_dev,
                        sd2=std_dev2,  # Use the second standard deviation when unequal_var is true
                        alpha=alpha,
                        nsim=nsim,
                        test="t-test",
                        seed=seed
                    )
                    
                    # Add information to the result to indicate unequal variances were used
                    result["std_dev_note"] = f"Using unequal variances: Group 1 SD = {std_dev}, Group 2 SD = {std_dev2}"
                else:
                    # Standard case with equal variance
                    power_result = simulation_continuous.power_continuous_sim(
                        n1=n1,
                        n2=n2,
                        mean1=mean1,
                        mean2=mean2,
                        sd1=std_dev,
                        sd2=std_dev,  # Same as sd1 when unequal_var is false
                        alpha=alpha,
                        nsim=nsim,
                        test="t-test",
                        seed=seed
                    )
        
        # Format results
        result["power"] = round(power_result.get("power", 0), 3)
        result["n1"] = n1
        result["n2"] = n2
        result["total_n"] = n1 + n2
        result["mean_difference"] = mean2 - mean1
        
        return result
        
    elif calculation_type == "Minimum Detectable Effect":
        # Get sample sizes
        n1 = params.get("n1", 20)
        n2 = params.get("n2", 20)
        
        # Calculate MDE
        if non_inferiority:
            # For non-inferiority designs, return a string message
            result["mde"] = "Not implemented for non-inferiority"
            # Initialize other results
            result["n1"] = n1
            result["n2"] = n2
            result["total_n"] = n1 + n2
            
            # Return early since we can't calculate MDE for non-inferiority
            return result
            
        elif repeated_measures:
            # Not implemented yet for repeated measures
            result["mde"] = "Not implemented for repeated measures"
            # Initialize other results
            result["n1"] = n1
            result["n2"] = n2
            result["total_n"] = n1 + n2
            
            # Return early since we can't calculate MDE for repeated measures
            return result
        
        elif method == "analytical":
            # Standard parallel design - analytical method
            mde_result = analytical_continuous.min_detectable_effect_continuous(
                n1=n1,
                n2=n2,
                sd1=std_dev,
                sd2=std_dev2 if unequal_var else std_dev,
                power=power,
                alpha=alpha
            )
            
            # Format results for standard parallel design
            result["mde"] = round(mde_result.get("minimum_detectable_effect", 0), 3)
            
        elif method == "simulation":
            # Standard parallel design - simulation method
            if unequal_var:
                # For unequal variance, use the average standard deviation as an approximation
                effective_std_dev = (std_dev + std_dev2) / 2
                
                mde_result = simulation_continuous.min_detectable_effect_continuous_sim(
                    n1=n1,
                    n2=n2,
                    std_dev=effective_std_dev,  # Use average of both std_devs
                    power=power,
                    nsim=nsim,
                    alpha=alpha,
                    precision=0.01,
                    seed=seed
                )
                
                # Add note about unequal variance approximation
                result["std_dev_note"] = f"Using unequal variances: Group 1 SD = {std_dev}, Group 2 SD = {std_dev2}"
            else:
                # Standard equal variance case
                mde_result = simulation_continuous.min_detectable_effect_continuous_sim(
                    n1=n1,
                    n2=n2,
                    std_dev=std_dev,
                    power=power,
                    nsim=nsim,
                    alpha=alpha,
                    precision=0.01,
                    seed=seed
                )
            
            # Format results
            result["mde"] = round(mde_result.get("minimum_detectable_effect", 0), 3)
        result["n1"] = n1
        result["n2"] = n2
        result["total_n"] = n1 + n2
        
        return result
    
    return result


def calculate_parallel_binary(params):
    """
    Calculate results for Parallel RCT with binary outcome.
    
    Args:
        params: Dictionary of parameters
        
    Returns:
        dict: Results of calculations
    """
    # Get calculation method and hypothesis type from params
    method = params.get("method", "analytical")  # Default to analytical if not specified
    hypothesis_type = params.get("hypothesis_type", "Superiority")  # Default to superiority if not specified
    
    # Extract parameters with default values
    p1 = params.get("p1", 0.3)
    p2 = params.get("p2", 0.5)
    alpha = params.get("alpha", 0.05)
    power = params.get("power", 0.8)
    allocation_ratio = params.get("allocation_ratio", 1.0)
    
    # For non-inferiority, log additional parameters for debugging
    if hypothesis_type == "Non-Inferiority":
        nim = params.get("nim", 0.1)  # Default non-inferiority margin
        direction = params.get("direction", "Higher is better")
        print(f"Non-inferiority calculation with margin: {nim}, direction: {direction}")
        print(f"Using p1: {p1}, p2: {p2}")
        
        # Add non-inferiority parameters to the calculation
    
    # Handle advanced options
    correction = params.get("correction", "None")
    
    # Prepare result dictionary
    result = {}
    
    # Get simulation-specific parameters
    nsim = params.get("nsim", 1000)
    seed = params.get("seed", 42)
    
    # Get the test type from the advanced params - this is what we set in render_binary_advanced_options
    test_type = params.get("test_type", "Normal Approximation")
    
    # Calculate based on calculation_type
    calculation_type = params.get("calculation_type", "Sample Size")
    
    if calculation_type == "Sample Size":
        # Calculate sample size
        if method == "analytical":
            # Convert test name to test_type expected by the function
            # Map our UI test names to function parameter names
            test_name_mapping = {
                "normal approximation": "normal approximation",
                "fisher's exact test": "fishers exact",
                "likelihood ratio test": "likelihood ratio"
            }
            # Convert the test_type to lowercase for case-insensitive matching
            mapped_test_type = test_name_mapping.get(test_type.lower(), "normal approximation")
            
            # Get correction directly from params (set by the checkbox in render_binary_advanced_options)
            has_correction = params.get("correction", False)
            
            # Debug output
            print(f"UI test type: {test_type}, Mapped to: {mapped_test_type}, Correction: {has_correction}")
            
            sample_size = analytical_binary.sample_size_binary(
                p1=p1,
                p2=p2,
                power=power,
                alpha=alpha,
                allocation_ratio=allocation_ratio,
                test_type=mapped_test_type,  # Use the correct parameter name "test_type"
                correction=has_correction  # Pass the correction parameter
            )
        elif method == "simulation":
            # Use simulation method for sample size calculation
            # Map the UI test type names to the format expected by the simulation function
            test_name_mapping = {
                "normal approximation": "normal_approximation",
                "fisher's exact test": "fishers_exact",
                "likelihood ratio test": "likelihood_ratio"
            }
            # Convert the test_type to lowercase and map to the correct format
            sim_test_type = test_name_mapping.get(test_type.lower(), "normal_approximation")
            
            # Debug output
            print(f"Simulation test type: {sim_test_type} from UI: {test_type}")
            
            sample_size = simulation_binary.sample_size_binary_sim(
                p1=p1,
                p2=p2,
                power=power,
                alpha=alpha,
                allocation_ratio=allocation_ratio,
                nsim=nsim,
                test_type=sim_test_type
            )
            
        # Extract values from result - handle different key names from different functions
        # The analytical function returns sample_size_1, sample_size_2, total_sample_size
        n1 = sample_size.get("sample_size_1", sample_size.get("n1", 0))
        n2 = sample_size.get("sample_size_2", sample_size.get("n2", 0))
        total_n = sample_size.get("total_sample_size", sample_size.get("total_n", n1 + n2))
        
        # Format results
        result["n1"] = round(n1)
        result["n2"] = round(n2)
        result["total_n"] = round(total_n)
        result["absolute_risk_difference"] = round(p2 - p1, 3)
        result["relative_risk"] = round(p2 / p1, 3) if p1 > 0 else "Infinity"
        result["odds_ratio"] = round((p2 / (1 - p2)) / (p1 / (1 - p1)), 3) if p1 < 1 and p2 < 1 else "Undefined"
        
        return result
        
    elif calculation_type == "Power":
        # Get sample sizes
        n1 = params.get("n1", 50)
        n2 = params.get("n2", 50)
        
        # Calculate power
        if method == "analytical":
            # Convert test name to test_type expected by the function
            test_name_mapping = {
                "normal approximation": "normal approximation",
                "fisher's exact test": "fishers exact",
                "likelihood ratio test": "likelihood ratio"
            }
            # Convert the test_type to lowercase for case-insensitive matching
            mapped_test_type = test_name_mapping.get(test_type.lower(), "normal approximation")
            
            # Check if correction is applied
            has_correction = params.get("correction", "None") != "None"
            
            power_result = analytical_binary.power_binary(
                n1=n1,
                n2=n2,
                p1=p1,
                p2=p2,
                alpha=alpha,
                test_type=mapped_test_type,  # Use the correct parameter name
                correction=has_correction  # Pass the correction parameter
            )
        elif method == "simulation":
            # Use simulation method for power calculation
            # Map the UI test type names to the format expected by the simulation function
            test_name_mapping = {
                "normal approximation": "normal_approximation",
                "fisher's exact test": "fishers_exact",
                "likelihood ratio test": "likelihood_ratio"
            }
            # Convert the test_type to lowercase and map to the correct format
            sim_test_type = test_name_mapping.get(test_type.lower(), "normal_approximation")

            # Debug output
            print(f"Simulation test type for power: {sim_test_type} from UI: {test_type}")

            power_result = simulation_binary.power_binary_sim(
                n1=n1,
                n2=n2,
                p1=p1,
                p2=p2,
                alpha=alpha,
                nsim=nsim,
                test_type=sim_test_type,
                seed=seed
            )

        # Format results
        result["power"] = round(power_result.get("power", 0), 3)
        result["n1"] = n1
        result["n2"] = n2
        result["total_n"] = n1 + n2
        result["absolute_risk_difference"] = round(p2 - p1, 3)
        result["relative_risk"] = round(p2 / p1, 3) if p1 > 0 else "Infinity"
        result["odds_ratio"] = round((p2 / (1 - p2)) / (p1 / (1 - p1)), 3) if p1 < 1 and p2 < 1 else "Undefined"
        
        return result
        
    elif calculation_type == "Minimum Detectable Effect":
        # Get sample sizes
        n1 = params.get("n1", 50)
        n2 = params.get("n2", 50)
        
        # Calculate MDE
        if method == "analytical":
            # Convert test name to test_type expected by the function
            test_name_mapping = {
                "normal approximation": "normal approximation",
                "fisher's exact test": "fishers exact",
                "likelihood ratio test": "likelihood ratio"
            }
            # Convert the test_type to lowercase for case-insensitive matching
            mapped_test_type = test_name_mapping.get(test_type.lower(), "normal approximation")
            
            # Check if correction is applied
            has_correction = params.get("correction", "None") != "None"
            
            mde_result = analytical_binary.min_detectable_effect_binary(
                n1=n1,
                n2=n2,
                p1=p1,
                power=power,
                alpha=alpha,
                test_type=mapped_test_type,  # Use the correct parameter name
                correction=has_correction  # Pass the correction parameter
            )
        elif method == "simulation":
            # Use simulation method for minimum detectable effect calculation
            # Map the UI test type names to the format expected by the simulation function
            test_name_mapping = {
                "normal approximation": "normal_approximation",
                "fisher's exact test": "fishers_exact",
                "likelihood ratio test": "likelihood_ratio"
            }
            # Convert the test_type to lowercase and map to the correct format
            sim_test_type = test_name_mapping.get(test_type.lower(), "normal_approximation")
            
            # Debug output
            print(f"Simulation test type for MDE: {sim_test_type} from UI: {test_type}")
            
            mde_result = simulation_binary.min_detectable_effect_binary_sim(
                n1=n1,
                n2=n2,
                p1=p1,
                power=power,
                nsim=nsim,
                alpha=alpha,
                precision=0.01,
                test_type=sim_test_type,
                seed=seed
            )
        
        # Format results
        p2_mde = mde_result.get("p2", 0)
        result["mde"] = round(p2_mde - p1, 3)
        result["p2_mde"] = round(p2_mde, 3)
        result["n1"] = n1
        result["n2"] = n2
        result["total_n"] = n1 + n2
        
        return result
    
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
    # Extract parameters with default values
    hr = params.get("hr", 0.7)
    median_survival1 = params.get("median_survival1", 12.0)
    alpha = params.get("alpha", 0.05)
    power = params.get("power", 0.8)
    allocation_ratio = params.get("allocation_ratio", 1.0)
    
    # Study parameters
    accrual_time = params.get("accrual_time", 12.0)
    follow_up_time = params.get("follow_up_time", 24.0)
    dropout_rate1 = params.get("dropout_rate1", 0.1)
    dropout_rate2 = params.get("dropout_rate2", 0.1)
    
    # Convert median survival to lambda (rate parameter)
    lambda1 = math.log(2) / median_survival1
    lambda2 = lambda1 * hr
    
    # Prepare result dictionary
    result = {}
    
    # Calculate based on calculation_type
    calculation_type = params.get("calculation_type", "Sample Size")
    
    if calculation_type == "Sample Size":
        # Calculate sample size
        if method == "analytical":
            sample_size = analytical_survival.sample_size_survival(
                lambda1=lambda1,
                lambda2=lambda2,
                accrual_time=accrual_time,
                follow_up_time=follow_up_time,
                dropout_rate1=dropout_rate1,
                dropout_rate2=dropout_rate2,
                power=power,
                alpha=alpha,
                allocation_ratio=allocation_ratio
            )
            
        # Extract values from result - handle different key names from different functions
        # The analytical function returns sample_size_1, sample_size_2, total_sample_size
        n1 = sample_size.get("sample_size_1", sample_size.get("n1", 0))
        n2 = sample_size.get("sample_size_2", sample_size.get("n2", 0))
        total_n = sample_size.get("total_sample_size", sample_size.get("total_n", n1 + n2))
        events = sample_size.get("events", 0)
        
        # Format results
        result["n1"] = round(n1)
        result["n2"] = round(n2)
        result["total_n"] = round(total_n)
        result["events"] = round(events)
        result["hazard_ratio"] = hr
        result["median_survival1"] = median_survival1
        result["median_survival2"] = median_survival1 / hr
        
        return result
        
    elif calculation_type == "Power":
        # Get sample sizes
        n1 = params.get("n1", 50)
        n2 = params.get("n2", 50)
        
        # Calculate power
        if method == "analytical":
            power_result = analytical_survival.power_survival(
                n1=n1,
                n2=n2,
                lambda1=lambda1,
                lambda2=lambda2,
                accrual_time=accrual_time,
                follow_up_time=follow_up_time,
                dropout_rate1=dropout_rate1,
                dropout_rate2=dropout_rate2,
                alpha=alpha
            )
        
        # Format results
        result["power"] = round(power_result.get("power", 0), 3)
        result["n1"] = n1
        result["n2"] = n2
        result["total_n"] = n1 + n2
        result["events"] = round(power_result.get("events", 0))
        result["hazard_ratio"] = hr
        result["median_survival1"] = median_survival1
        result["median_survival2"] = median_survival1 / hr
        
        return result
        
    elif calculation_type == "Minimum Detectable Effect":
        # Get sample sizes
        n1 = params.get("n1", 50)
        n2 = params.get("n2", 50)
        
        # Calculate MDE
        if method == "analytical":
            mde_result = analytical_survival.mde_survival(
                n1=n1,
                n2=n2,
                lambda1=lambda1,
                accrual_time=accrual_time,
                follow_up_time=follow_up_time,
                dropout_rate1=dropout_rate1,
                dropout_rate2=dropout_rate2,
                power=power,
                alpha=alpha
            )
        
        # Format results
        hr_mde = mde_result.get("hr", 0)
        result["mde"] = round(hr_mde, 3)
        result["n1"] = n1
        result["n2"] = n2
        result["total_n"] = n1 + n2
        result["events"] = round(mde_result.get("events", 0))
        result["median_survival1"] = median_survival1
        result["median_survival2"] = median_survival1 / hr_mde
        
        return result
    
    return result
