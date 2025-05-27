"""
Component module for Parallel RCT designs.

This module provides UI rendering functions and calculation functions for
Parallel RCT designs with continuous, binary, and survival outcomes.
"""
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

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

def render_continuous_advanced_options(calc_type):
    """
    Render advanced options for continuous outcome designs.
    
    Args:
        calc_type: String indicating calculation type (Sample Size, Power, or Minimum Detectable Effect)
        
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
        advanced_params["use_simulation"] = True
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
        
        # Add min_n, max_n, step_n for sample size simulation
        if calc_type == "Sample Size":
            st.markdown("##### Sample Size Simulation Parameters")
            col_sim_ss1, col_sim_ss2, col_sim_ss3 = st.columns(3)
            with col_sim_ss1:
                advanced_params["min_n"] = st.number_input(
                    "Min N per Group", value=10, min_value=2, step=1, key="continuous_min_n_sim"
                )
            with col_sim_ss2:
                advanced_params["max_n"] = st.number_input(
                    "Max N per Group", value=500, min_value=10, step=10, key="continuous_max_n_sim"
                )
            with col_sim_ss3:
                advanced_params["step_n"] = st.number_input(
                    "Step for N", value=10, min_value=1, step=1, key="continuous_step_n_sim"
                )
            
    else:
        advanced_params["use_simulation"] = False
            
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
        
        with col1:
            # Always show Group 1 mean and standard deviation for both designs
            params["mean1"] = st.number_input("Mean (Group 1)", value=0.0, step=0.1, key="mean1_input")
            params["std_dev"] = st.number_input("Standard Deviation", value=1.0, step=0.1, min_value=0.1, key="sd_input")
            
            # For non-inferiority design, show margin and assumed difference
            if hypothesis_type == "Non-Inferiority":
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
            if hypothesis_type == "Non-Inferiority":
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
        if hypothesis_type == "Non-Inferiority":
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
        advanced_options = render_continuous_advanced_options(calc_type)
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
    st.write(f"### Parallel RCT with Survival Outcome ({calc_type} - {hypothesis_type})")
    
    # Basic parameters UI
    with st.container():
        st.subheader("Basic Parameters")
        
        # Parameter inputs
        col1, col2 = st.columns(2)
        
        with col1:
            if hypothesis_type == "Superiority":
                params["hr"] = st.number_input("Hazard Ratio (Treatment vs Control)", value=0.7, step=0.05, min_value=0.1, key="hr_input_survival_sup")
                params["median_survival1"] = st.number_input("Median Survival (Control Group, months)", 
                                                          value=12.0, step=1.0, min_value=0.1,
                                                          key="median1_input_survival_sup")
            elif hypothesis_type == "Non-Inferiority":
                params["median_survival1"] = st.number_input("Median Survival (Control Group, months)", 
                                                          value=12.0, step=1.0, min_value=0.1,
                                                          key="median1_input_survival_ni")
                params["non_inferiority_margin_hr"] = st.number_input(
                    "Non-Inferiority Margin (Hazard Ratio)", 
                    value=1.3, step=0.05, min_value=1.01, 
                    help="Upper acceptable limit for HR (Treatment/Control). Must be > 1.",
                    key="nim_hr_input_survival_ni"
                )
                params["assumed_true_hr"] = st.number_input(
                    "Assumed True Hazard Ratio (Treatment/Control)", 
                    value=1.0, step=0.05, min_value=0.1, 
                    help="Expected HR under the alternative hypothesis (e.g., 1.0 for true equivalence).",
                    key="assumed_hr_input_survival_ni"
                )
                if calc_type == "Minimum Detectable Effect":
                    st.warning("Minimum Detectable Effect is typically not the primary calculation for non-inferiority designs with a pre-specified margin.")

        with col2:
            params["alpha"] = st.slider("Significance Level (α)", 
                                     min_value=0.01, max_value=0.20, value=0.05, step=0.01,
                                     key="alpha_slider_survival")
                                     
            if calc_type != "Power":
                params["power"] = st.slider("Power (1-β)", 
                                         min_value=0.70, max_value=0.99, value=0.80, step=0.01,
                                         key="power_slider_survival")
            else:
                params["power"] = None # Default for power calculation, not shown
                params["n1"] = st.number_input("Sample Size (Group 1)", value=50, step=1, min_value=5, key="n1_input_survival_power")
                params["n2"] = st.number_input("Sample Size (Group 2)", value=50, step=1, min_value=5, key="n2_input_survival_power")
                
            # Allocation ratio is always relevant
            if calc_type == "Sample Size":
                 params["allocation_ratio"] = st.slider("Allocation Ratio (n2/n1)", 
                                                   min_value=0.1, max_value=10.0, value=1.0, step=0.1,
                                                   key="allocation_slider_survival_ss")
            elif 'n1' in params and 'n2' in params and params['n1'] > 0: # For Power and MDE if n1, n2 are provided
                params["allocation_ratio"] = params['n2'] / params['n1']
                st.write(f"Allocation Ratio (n2/n1): {params['allocation_ratio']:.2f}") 
            else: # Default if not sample size and n1/n2 not set (e.g. MDE without n1/n2 yet)
                params["allocation_ratio"] = 1.0 # Default, might be overridden if n1/n2 are set for MDE

    # Advanced parameters UI
    with st.expander("Advanced Parameters", expanded=False):
        col_adv1, col_adv2 = st.columns(2)
        with col_adv1:
            params["accrual_time"] = st.number_input("Enrollment Period (months)", 
                                                 value=12.0, step=1.0, min_value=0.1,
                                                 key="accrual_input_survival")
            if hypothesis_type == "Superiority": # Only show sides for superiority
                sides_options = ["Two-sided", "One-sided"]
                selected_sides = st.radio(
                    "Test Type", 
                    options=sides_options, 
                    index=0, 
                    key="sides_radio_survival_sup",
                    horizontal=True
                )
                params["sides"] = 1 if selected_sides == "One-sided" else 2
            # For Non-Inferiority, sides is implicitly 1, handled in calculation function
            
        with col_adv2:
            params["follow_up_time"] = st.number_input("Follow-up Period (months)", 
                                                    value=24.0, step=1.0, min_value=0.1,
                                                    key="followup_input_survival")
            
        params["dropout_rate"] = st.slider("Overall Dropout Rate", 
                                        min_value=0.0, max_value=0.5, value=0.1, step=0.01,
                                        key="dropout_slider_survival")

        method_options = ["Analytical", "Simulation"]
        selected_method = st.selectbox(
            "Calculation Method",
            options=method_options,
            index=method_options.index(params.get("method", "Analytical")), 
            key="method_survival_selectbox"
        )
        params["method"] = selected_method

        if params["method"] == "Simulation":
            col_sim_basic1, col_sim_basic2 = st.columns(2)
            with col_sim_basic1:
                params["nsim"] = st.number_input("Number of Simulations", value=1000, min_value=100, step=100, key="nsim_survival_input")
            with col_sim_basic2:
                params["seed"] = st.number_input(
                    "Random Seed", 
                    value=42, 
                    min_value=1, 
                    help="Seed for random number generation (for reproducibility)",
                    key="seed_survival_input"
                )

            if calc_type == "Sample Size":
                st.write("Simulation Parameters for Sample Size Optimization:")
                sim_col1, sim_col2, sim_col3 = st.columns(3)
                with sim_col1:
                    params["min_n_sim"] = st.number_input("Min N (per group)", value=10, min_value=5, key="min_n_sim_survival")
                with sim_col2:
                    params["max_n_sim"] = st.number_input("Max N (per group)", value=500, min_value=10, key="max_n_sim_survival")
                with sim_col3:
                    params["step_n_sim"] = st.number_input("Step N", value=5, min_value=1, key="step_n_sim_survival")
            if calc_type == "Minimum Detectable Effect" and hypothesis_type == "Non-Inferiority":
                 # MDE for NI simulation might need specific parameters if we decide to support it meaningfully
                 pass # Placeholder for now
        
    params["calculation_type"] = calc_type
    params["hypothesis_type"] = hypothesis_type

    # Default n1, n2 for MDE if not provided in power section
    if calc_type == "Minimum Detectable Effect":
        if "n1" not in params:
            params["n1"] = st.number_input("Sample Size (Group 1)", value=50, step=1, min_value=5, key="n1_input_survival_mde")
        if "n2" not in params:
            params["n2"] = st.number_input("Sample Size (Group 2)", value=50, step=1, min_value=5, key="n2_input_survival_mde")
        if params['n1'] > 0 : # Recalculate allocation ratio if n1, n2 are set for MDE
            params["allocation_ratio"] = params['n2'] / params['n1']
            # st.write(f"Allocation Ratio (n2/n1): {params['allocation_ratio']:.2f}") # Display if needed

    return params


def calculate_parallel_continuous(params):
    """Calculate results for Parallel RCT with continuous outcomes."""
    calc_type = params.get("calculation_type", "Sample Size")
    hypothesis_type = params.get("hypothesis_type", "Superiority")
    use_simulation = params.get("use_simulation", False)
    results_dict = None 

    # Extract common simulation parameters if simulation is used
    nsim = params.get("nsim", 1000) if use_simulation else None
    seed = params.get("seed", 42) if use_simulation else None 
    ui_std_dev = params.get("std_dev", 1.0) # Primary SD from UI
    ui_std_dev2 = params.get("std_dev2", ui_std_dev) # Secondary SD from UI, defaults to primary if not set

    # Extract parameters for sample size simulation if applicable
    min_n_sim = params.get("min_n", 10) if use_simulation and calc_type == "Sample Size" else None
    max_n_sim = params.get("max_n", 1000) if use_simulation and calc_type == "Sample Size" else None
    step_sim = params.get("step_n", 10) if use_simulation and calc_type == "Sample Size" else None 

    if use_simulation:
        if calc_type == "Sample Size":
            if hypothesis_type == "Superiority" or hypothesis_type == "Equivalence":
                delta = abs(params.get("mean1", 0) - params.get("mean2", 1))
                # sample_size_continuous_sim expects std_dev
                results_dict = simulation_continuous.sample_size_continuous_sim(
                    delta=delta,
                    std_dev=ui_std_dev,
                    power=params.get("power", 0.8),
                    alpha=params.get("alpha", 0.05),
                    allocation_ratio=params.get("allocation_ratio", 1.0),
                    nsim=nsim,
                    min_n=min_n_sim,
                    max_n=max_n_sim,
                    step=step_sim,
                    repeated_measures=params.get("repeated_measures", False),
                    correlation=params.get("correlation", 0.5),
                    method=params.get("repeated_measures_method", "change_score"),
                    seed=seed
                )
            elif hypothesis_type == "Non-Inferiority":
                # sample_size_continuous_non_inferiority_sim expects std_dev
                results_dict = simulation_continuous.sample_size_continuous_non_inferiority_sim(
                    non_inferiority_margin=params.get("non_inferiority_margin", 0.1),
                    std_dev=ui_std_dev,
                    power=params.get("power", 0.8),
                    alpha=params.get("alpha", 0.05),
                    allocation_ratio=params.get("allocation_ratio", 1.0),
                    nsim=nsim,
                    min_n=min_n_sim,
                    max_n=max_n_sim,
                    step=step_sim,
                    assumed_difference=params.get("assumed_difference", 0.0),
                    direction=params.get("non_inferiority_direction", "lower"),
                    repeated_measures=params.get("repeated_measures", False),
                    correlation=params.get("correlation", 0.5),
                    method=params.get("repeated_measures_method", "change_score"),
                    seed=seed
                )
        elif calc_type == "Power":
            if hypothesis_type == "Superiority" or hypothesis_type == "Equivalence":
                # power_continuous_sim expects sd1, sd2
                results_dict = simulation_continuous.power_continuous_sim(
                    n1=params.get("n1", 100),
                    n2=params.get("n2", 100),
                    mean1=params.get("mean1", 0),
                    mean2=params.get("mean2", 1),
                    sd1=ui_std_dev,
                    sd2=ui_std_dev2, 
                    alpha=params.get("alpha", 0.05),
                    nsim=nsim,
                    test=params.get("test_type_continuous", "t-test"),
                    seed=seed
                )
            elif hypothesis_type == "Non-Inferiority":
                # simulate_continuous_non_inferiority (for power) expects std_dev
                results_dict = simulation_continuous.simulate_continuous_non_inferiority(
                    n1=params.get("n1", 100),
                    n2=params.get("n2", 100),
                    non_inferiority_margin=params.get("non_inferiority_margin", 0.1),
                    std_dev=ui_std_dev,
                    nsim=nsim,
                    alpha=params.get("alpha", 0.05),
                    seed=seed,
                    assumed_difference=params.get("assumed_difference", 0.0),
                    direction=params.get("non_inferiority_direction", "lower"),
                    repeated_measures=params.get("repeated_measures", False),
                    correlation=params.get("correlation", 0.5),
                    method=params.get("repeated_measures_method", "change_score")
                )
        elif calc_type == "Minimum Detectable Effect":
            if hypothesis_type == "Superiority" or hypothesis_type == "Equivalence":
                # min_detectable_effect_continuous_sim expects std_dev
                results_dict = simulation_continuous.min_detectable_effect_continuous_sim(
                    n1=params.get("n1", 100),
                    n2=params.get("n2", 100),
                    std_dev=ui_std_dev,
                    power=params.get("power", 0.8),
                    alpha=params.get("alpha", 0.05),
                    nsim=nsim,
                    precision=params.get("mde_precision", 0.01),
                    repeated_measures=params.get("repeated_measures", False),
                    correlation=params.get("correlation", 0.5),
                    method=params.get("repeated_measures_method", "change_score"),
                    seed=seed
                )
            elif hypothesis_type == "Non-Inferiority":
                # Assuming min_detectable_effect_non_inferiority_sim also expects std_dev
                # If this function doesn't exist or has different params, it will need adjustment
                if hasattr(simulation_continuous, 'min_detectable_effect_non_inferiority_sim'):
                    results_dict = simulation_continuous.min_detectable_effect_non_inferiority_sim(
                        n1=params.get("n1", 100),
                        n2=params.get("n2", 100),
                        std_dev=ui_std_dev,
                        power=params.get("power", 0.8),
                        alpha=params.get("alpha", 0.05),
                        nsim=nsim,
                        non_inferiority_margin=params.get("non_inferiority_margin", 0.1),
                        assumed_difference=params.get("assumed_difference", 0.0),
                        direction=params.get("non_inferiority_direction", "lower"),
                        repeated_measures=params.get("repeated_measures", False),
                        correlation=params.get("correlation", 0.5),
                        method=params.get("repeated_measures_method", "change_score"),
                        seed=seed
                    )
                else:
                    # Placeholder if the function is missing, to avoid crashing
                    results_dict = {"error": "min_detectable_effect_non_inferiority_sim not found"}
                    st.error("MDE Non-Inferiority simulation function not available.")

    else:  # Analytical calculations
        # Extract common parameters with default values to avoid None errors
        mean1 = params.get("mean1", 0.0)
        mean2 = params.get("mean2", 0.5)
        # Analytical functions expect sd1, sd2
        # sd1 comes from ui_std_dev, sd2 from ui_std_dev2 (which defaults to ui_std_dev if unequal_var is false)
        sd1_val = ui_std_dev 
        sd2_val = ui_std_dev2
        alpha = params.get("alpha", 0.05)
        power = params.get("power", 0.8)
        allocation_ratio = params.get("allocation_ratio", 1.0)
        
        unequal_var = params.get("unequal_var", False)
        repeated_measures = params.get("repeated_measures", False)
        correlation = params.get("correlation", 0.5)
        analysis_method = params.get("repeated_measures_method", "ANCOVA") 

        if calc_type == "Sample Size":
            if hypothesis_type == "Superiority" or hypothesis_type == "Equivalence":
                if repeated_measures:
                    results_dict = analytical_continuous.sample_size_repeated_measures(
                        mean1=mean1, mean2=mean2, sd1=sd1_val, correlation=correlation,
                        power=power, alpha=alpha, allocation_ratio=allocation_ratio, method=analysis_method
                    )
                else:
                    results_dict = analytical_continuous.sample_size_continuous(
                        mean1=mean1, mean2=mean2, sd1=sd1_val, sd2=sd2_val,
                        power=power, alpha=alpha, allocation_ratio=allocation_ratio
                    )
            elif hypothesis_type == "Non-Inferiority":
                results_dict = analytical_continuous.sample_size_continuous_non_inferiority(
                    mean1=mean1, 
                    non_inferiority_margin=params.get("non_inferiority_margin", 0.1),
                    sd1=sd1_val, 
                    sd2=sd2_val, 
                    power=power, 
                    alpha=alpha, 
                    allocation_ratio=allocation_ratio,
                    assumed_difference=params.get("assumed_difference", 0.0),
                    direction=params.get("non_inferiority_direction", "lower")
                )
        elif calc_type == "Power":
            n1 = params.get("n1", 100)
            n2 = params.get("n2", 100)
            if hypothesis_type == "Superiority" or hypothesis_type == "Equivalence":
                if repeated_measures:
                    results_dict = analytical_continuous.power_repeated_measures(
                        n1=n1, n2=n2, mean1=mean1, mean2=mean2, sd=sd1_val, 
                        correlation=correlation, alpha=alpha, method=analysis_method
                    )
                else:
                    results_dict = analytical_continuous.power_continuous(
                        n1=n1, n2=n2, mean1=mean1, mean2=mean2, sd1=sd1_val, sd2=sd2_val, alpha=alpha
                    )
            elif hypothesis_type == "Non-Inferiority":
                results_dict = analytical_continuous.power_continuous_non_inferiority(
                    n1=n1, n2=n2, mean1=mean1, 
                    non_inferiority_margin=params.get("non_inferiority_margin", 0.1),
                    sd1=sd1_val, sd2=sd2_val, alpha=alpha,
                    assumed_difference=params.get("assumed_difference", 0.0),
                    direction=params.get("non_inferiority_direction", "lower")
                )
        elif calc_type == "Minimum Detectable Effect":
            n1 = params.get("n1", 100)
            n2 = params.get("n2", 100)
            if hypothesis_type == "Superiority" or hypothesis_type == "Equivalence":
                if repeated_measures:
                    # MDE for repeated measures might need specific handling or may not be directly available
                    # For now, let's assume it falls back to standard if not specifically implemented
                    # Or, we can state it's not supported for analytical repeated measures MDE yet.
                    # This part needs to align with what analytical_continuous.min_detectable_effect_repeated_measures expects or if it exists
                    # For now, using standard MDE as a placeholder if repeated_measures MDE isn't distinct
                     results_dict = analytical_continuous.min_detectable_effect_continuous(
                        n1=n1, n2=n2, sd1=sd1_val, sd2=sd2_val, power=power, alpha=alpha
                    ) 
                else:
                    results_dict = analytical_continuous.min_detectable_effect_continuous(
                        n1=n1, n2=n2, sd1=sd1_val, sd2=sd2_val, power=power, alpha=alpha
                    )
            elif hypothesis_type == "Non-Inferiority":
                # MDE for non-inferiority is typically not calculated in the same way.
                # Usually, the margin is fixed. We can return a message or the margin itself.
                results_dict = {"mde_non_inferiority_info": "MDE is not applicable for non-inferiority in the same sense; the margin is key."}

    # Process and return results
    final_results = {}
    if results_dict:
        if calc_type == "Sample Size":
            final_results["n1"] = round(results_dict.get("n1", results_dict.get("sample_size_1", 0)))
            final_results["n2"] = round(results_dict.get("n2", results_dict.get("sample_size_2", 0)))
            final_results["total_n"] = final_results["n1"] + final_results["n2"]
            if not use_simulation: 
                 final_results["mean_difference"] = params.get("mean2", 1) - params.get("mean1", 0)
            elif "delta" in results_dict: 
                 final_results["mean_difference"] = results_dict.get("delta")

        elif calc_type == "Power":
            final_results["power"] = round(results_dict.get("power", results_dict.get("empirical_power", 0)), 3)
            final_results["n1"] = params.get("n1", 0)
            final_results["n2"] = params.get("n2", 0)
            final_results["total_n"] = final_results["n1"] + final_results["n2"]
            if not use_simulation:
                 final_results["mean_difference"] = params.get("mean2", 1) - params.get("mean1", 0)
            elif "mean1" in results_dict and "mean2" in results_dict: 
                 final_results["mean_difference"] = results_dict.get("mean2") - results_dict.get("mean1")
            if use_simulation and "std_dev_note" in results_dict:
                final_results["std_dev_note"] = results_dict["std_dev_note"]

        elif calc_type == "Minimum Detectable Effect":
            if "mde_non_inferiority_info" in results_dict:
                final_results["mde_info"] = results_dict["mde_non_inferiority_info"]
            else:
                final_results["mde"] = round(results_dict.get("mde", results_dict.get("minimum_detectable_effect", 0)), 3)
            final_results["n1"] = params.get("n1", 0)
            final_results["n2"] = params.get("n2", 0)
            final_results["total_n"] = final_results["n1"] + final_results["n2"]
            if use_simulation and "std_dev_note" in results_dict:
                final_results["std_dev_note"] = results_dict["std_dev_note"]
        
        # Add simulation specific info if used
        if use_simulation:
            final_results["simulations"] = nsim
            final_results["seed"] = seed
            if "mean_p_value" in results_dict: 
                final_results["mean_p_value"] = round(results_dict["mean_p_value"], 4)

    return final_results


def calculate_parallel_binary(params):
    """Calculate results for Parallel RCT with binary outcomes."""
    # Get calculation method and hypothesis type from params
    method = params.get("method", "analytical")  
    hypothesis_type = params.get("hypothesis_type", "Superiority")  

    # Extract parameters with default values
    p1 = params.get("p1", 0.3)
    p2 = params.get("p2", 0.5)
    alpha = params.get("alpha", 0.05)
    power = params.get("power", 0.8)
    allocation_ratio = params.get("allocation_ratio", 1.0)
    
    # For non-inferiority, log additional parameters for debugging
    if hypothesis_type == "Non-Inferiority":
        nim = params.get("nim", 0.1)  
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
            test_name_mapping = {
                "normal approximation": "normal approximation",
                "fisher's exact test": "fishers exact",
                "likelihood ratio test": "likelihood ratio"
            }
            # Convert the test_type to lowercase for case-insensitive matching
            mapped_test_type = test_name_mapping.get(test_type.lower(), "normal approximation")
            
            # Get correction directly from params (set by the checkbox in render_binary_advanced_options)
            has_correction = params.get("correction", "None") != "None"
            
            # Debug output
            print(f"UI test type: {test_type}, Mapped to: {mapped_test_type}, Correction: {has_correction}")
            
            sample_size = analytical_binary.sample_size_binary(
                p1=p1,
                p2=p2,
                power=power,
                alpha=alpha,
                allocation_ratio=allocation_ratio,
                test_type=mapped_test_type,  
                correction=has_correction  
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
                test_type=mapped_test_type,  
                correction=has_correction  
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
                test_type=mapped_test_type,  
                correction=has_correction  
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


def calculate_parallel_survival(params):
    """Calculate results for Parallel RCT with survival outcome."""
    # Extract parameters with default values
    calculation_type = params.get("calculation_type", "Sample Size")
    hypothesis_type = params.get("hypothesis_type", "Superiority")
    method = params.get("method", "Analytical")

    alpha = params.get("alpha", 0.05)
    power = params.get("power", 0.8)
    allocation_ratio = params.get("allocation_ratio", 1.0)
    
    # Study parameters
    accrual_time = params.get("accrual_time", 12.0)
    follow_up_time = params.get("follow_up_time", 24.0)
    dropout_rate = params.get("dropout_rate", 0.1) # Single dropout rate from UI

    # Simulation specific parameters
    nsim = params.get("nsim", 1000)
    min_n_sim = params.get("min_n_sim", 10)
    max_n_sim = params.get("max_n_sim", 500)
    step_n_sim = params.get("step_n_sim", 5)
    seed = params.get("seed", 42) # Added seed for simulation

    # Initialize effective parameters for calculation
    median_survival1 = params.get("median_survival1", 12.0)
    hr_for_calc = None
    median2_for_calc = None
    sides = params.get("sides", 2) # Default to 2 for superiority unless specified
    
    # Specific parameters based on hypothesis type
    non_inferiority_margin_hr_param = None
    assumed_true_hr_param = None

    if hypothesis_type == "Non-Inferiority":
        sides = 1 # Non-inferiority is one-sided by definition
        non_inferiority_margin_hr_param = params.get("non_inferiority_margin_hr", 1.3)
        assumed_true_hr_param = params.get("assumed_true_hr", 1.0)
        
        # For display and simulation, power is based on the assumed true HR
        hr_for_calc = assumed_true_hr_param 
        median2_for_calc = median_survival1 / hr_for_calc if hr_for_calc is not None and hr_for_calc > 0 else float('inf')

    elif hypothesis_type == "Superiority":
        hr_for_calc = params.get("hr", 0.7)
        median2_for_calc = median_survival1 / hr_for_calc if hr_for_calc is not None and hr_for_calc > 0 else float('inf')
        # `sides` will be taken from params.get("sides", 2) as set above
    else:
        return {"error": f"Unknown hypothesis type: {hypothesis_type}"}

    # Prepare result dictionary, including input params for display
    result = {
        "calculation_type_param": calculation_type,
        "hypothesis_type_param": hypothesis_type,
        "method_param": method,
        "alpha_param": alpha,
        "power_param": power if calculation_type != "Power" else None,
        "median_survival1_param": median_survival1,
        "accrual_time_param": accrual_time,
        "follow_up_time_param": follow_up_time,
        "dropout_rate_param": dropout_rate,
        "allocation_ratio_param": allocation_ratio,
        "sides_param": sides
    }
    if hypothesis_type == "Non-Inferiority":
        result["non_inferiority_margin_hr_param"] = non_inferiority_margin_hr_param
        result["assumed_true_hr_param"] = assumed_true_hr_param
        # For NI, calculations use assumed_true_hr to derive median2 for simulation/analytical calls if needed
        # The NI margin itself is passed directly to NI-specific functions.
        result["hr_for_display"] = assumed_true_hr_param # Display the assumed true HR
        result["median_survival2_derived"] = median2_for_calc
    else: # Superiority
        result["hr_param"] = hr_for_calc # hr is the primary effect size parameter for superiority
        result["hr_for_display"] = hr_for_calc # Display the specified HR for superiority
        result["median_survival2_derived"] = median2_for_calc

    if calculation_type == "Sample Size":
        if method == "Analytical":
            if hypothesis_type == "Non-Inferiority":
                sample_size_result = analytical_survival.sample_size_survival_non_inferiority(
                    median1=median_survival1,
                    non_inferiority_margin=non_inferiority_margin_hr_param,
                    enrollment_period=accrual_time,
                    follow_up_period=follow_up_time,
                    dropout_rate=dropout_rate,
                    power=power,
                    alpha=alpha, # NI function handles one-sided alpha internally
                    allocation_ratio=allocation_ratio,
                    assumed_hazard_ratio=assumed_true_hr_param
                )
            else: # Superiority
                sample_size_result = analytical_survival.sample_size_survival(
                    median1=median_survival1,
                    median2=median2_for_calc,
                    enrollment_period=accrual_time,
                    follow_up_period=follow_up_time,
                    dropout_rate=dropout_rate,
                    power=power,
                    alpha=alpha, 
                    allocation_ratio=allocation_ratio,
                    sides=sides 
                )
        elif method == "Simulation":
            if hypothesis_type == "Non-Inferiority":                   
                sample_size_result = simulation_survival.sample_size_survival_non_inferiority_sim(
                    median1=median_survival1,
                    non_inferiority_margin=non_inferiority_margin_hr_param,
                    assumed_hazard_ratio=assumed_true_hr_param, # median2 is derived from this internally in sim_ni
                    enrollment_period=accrual_time,
                    follow_up_period=follow_up_time,
                    dropout_rate=dropout_rate,
                    power=power,
                    alpha=alpha, # NI sim function handles one-sided alpha
                    allocation_ratio=allocation_ratio,
                    nsim=nsim,
                    min_n=min_n_sim,
                    max_n=max_n_sim,
                    step=step_n_sim, # Parameter name is 'step' in sim_ni function
                    seed=seed
                )
            else: # Superiority
                sample_size_result = simulation_survival.sample_size_survival_sim(
                    median1=median_survival1,
                    median2=median2_for_calc, 
                    enrollment_period=accrual_time,
                    follow_up_period=follow_up_time,
                    dropout_rate=dropout_rate, 
                    power=power,
                    alpha=alpha,
                    allocation_ratio=allocation_ratio,
                    nsim=nsim,
                    sides=sides, 
                    seed=seed
                )
        else:
            return {"error": f"Unsupported method: {method} for Sample Size calculation."}
            
        result["n1"] = round(sample_size_result.get("n1", sample_size_result.get("sample_size_1", 0)))
        result["n2"] = round(sample_size_result.get("n2", sample_size_result.get("sample_size_2", 0)))
        result["total_n"] = result["n1"] + result["n2"]
        result["events"] = round(sample_size_result.get("events", sample_size_result.get("total_events", 0)))
        if 'power_curve_data' in sample_size_result:
            result['power_curve_data'] = sample_size_result['power_curve_data']
        
    elif calculation_type == "Power":
        n1 = params.get("n1", 50)
        n2 = params.get("n2", 50)
        result["n1_param"] = n1 
        result["n2_param"] = n2
        if method == "Analytical":
            if hypothesis_type == "Non-Inferiority":
                power_result_dict = analytical_survival.power_survival_non_inferiority(
                    n1=n1, n2=n2,
                    median1=median_survival1,
                    non_inferiority_margin=non_inferiority_margin_hr_param,
                    enrollment_period=accrual_time,
                    follow_up_period=follow_up_time,
                    dropout_rate=dropout_rate,
                    alpha=alpha, # NI function handles one-sided alpha internally
                    assumed_hazard_ratio=assumed_true_hr_param
                )
            else: # Superiority
                power_result_dict = analytical_survival.power_survival(
                    n1=n1, n2=n2,
                    median1=median_survival1,
                    median2=median2_for_calc,
                    enrollment_period=accrual_time,
                    follow_up_period=follow_up_time,
                    dropout_rate=dropout_rate,
                    alpha=alpha,
                    sides=sides
                )
        elif method == "Simulation":
            if hypothesis_type == "Non-Inferiority":
                power_result_dict = simulation_survival.power_survival_non_inferiority_sim(
                    n1=n1, n2=n2,
                    median1=median_survival1,
                    non_inferiority_margin=non_inferiority_margin_hr_param,
                    assumed_hazard_ratio=assumed_true_hr_param, # median2 derived internally
                    enrollment_period=accrual_time,
                    follow_up_period=follow_up_time,
                    dropout_rate=dropout_rate,
                    alpha=alpha, # NI sim function handles one-sided alpha
                    nsim=nsim,
                    seed=seed
                )
            else: # Superiority
                power_result_dict = simulation_survival.power_survival_sim(
                    n1=n1, n2=n2,
                    median1=median_survival1,
                    median2=median2_for_calc,
                    enrollment_period=accrual_time,
                    follow_up_period=follow_up_time,
                    dropout_rate=dropout_rate, 
                    alpha=alpha,
                    nsim=nsim,
                    sides=sides, 
                    seed=seed
                )
        else:
            return {"error": f"Unsupported method: {method} for Power calculation."}
        
        result["power"] = round(power_result_dict.get("power", 0), 3)
        result["events"] = round(power_result_dict.get("events", power_result_dict.get("total_events", 0)))
        if 'survival_curves' in power_result_dict:
            result['survival_curves'] = power_result_dict['survival_curves']

    elif calculation_type == "Minimum Detectable Effect":
        if hypothesis_type == "Non-Inferiority":
            result["mde_not_applicable"] = True 
            result["message"] = "Minimum Detectable Effect for Non-Inferiority is typically defined as the largest margin (NIM) for which non-inferiority can be claimed, or the true effect (HR) detectable against a fixed NIM. The calculation below provides the detectable HR for a superiority hypothesis. A dedicated NI MDE simulation is not yet implemented."
        
        n1 = params.get("n1", 50)
        n2 = params.get("n2", 50)
        result["n1_param"] = n1
        result["n2_param"] = n2
        
        if method == "Analytical":
            mde_result_dict = analytical_survival.min_detectable_effect_survival(
                n1=n1, n2=n2,
                median1=median_survival1,
                enrollment_period=accrual_time,
                follow_up_period=follow_up_time,
                dropout_rate=dropout_rate,
                power=power,
                alpha=alpha,
                sides=sides 
            )
        elif method == "Simulation":
            # For NI MDE, we'd ideally call a specific NI MDE simulation if it existed.
            # Since it doesn't, we fall back to superiority MDE simulation.
            if hypothesis_type == "Non-Inferiority":
                # Placeholder: A true NI MDE simulation would iterate on assumed_true_hr or NIM.
                # For now, we'll just use the superiority MDE sim and the message above explains this.
                pass # No specific NI MDE sim function to call yet.

            mde_result_dict = simulation_survival.min_detectable_effect_survival_sim(
                n1=n1, n2=n2,
                median1=median_survival1,
                enrollment_period=accrual_time,
                follow_up_period=follow_up_time,
                dropout_rate=dropout_rate,
                power=power,
                alpha=alpha,
                nsim=nsim,
                sides=sides, 
                seed=seed
            )
        else:
            return {"error": f"Unsupported method: {method} for MDE calculation."}

        detectable_hr = mde_result_dict.get("hr", None)
        result["mde"] = round(detectable_hr, 3) if detectable_hr is not None else None
        result["median_survival2_mde"] = round(median_survival1 / detectable_hr, 1) if detectable_hr and detectable_hr > 0 else None
        result["events"] = round(mde_result_dict.get("events", mde_result_dict.get("total_events", 0)))
        if 'power_vs_hr_data' in mde_result_dict:
             result['power_vs_hr_data'] = mde_result_dict['power_vs_hr_data']

    result['nsim'] = nsim if method == "Simulation" else None

    return result

def display_survival_results(result, calculation_type, hypothesis_type, use_simulation):
    """Display formatted results for survival outcome calculations."""
    st.subheader("Results")

    if not result:
        st.error("No results to display.")
        return

    # Display based on calculation type
    if calculation_type == "Sample Size":
        st.write(f"**Required Sample Size (Total):** {result.get('total_n', 'N/A')}")
        st.write(f"- Group 1: {result.get('n1', 'N/A')}")
        st.write(f"- Group 2: {result.get('n2', 'N/A')}")
        st.write(f"**Expected Number of Events:** {result.get('events', 'N/A')}")
        df_data = {
            "Parameter": ["Hazard Ratio", "Median Survival (Group 1)", "Median Survival (Group 2)", "Power", "Alpha"],
            "Value": [
                result.get('hr_for_display', 'N/A'), 
                result.get('median_survival1_param', 'N/A'), 
                result.get('median_survival2_derived', 'N/A'),
                result.get('power_param', 'N/A'), # Power used as input
                result.get('alpha_param', 'N/A')  # Alpha used as input
            ]
        }

    elif calculation_type == "Power":
        power_val = result.get('power', 'N/A')
        if isinstance(power_val, (int, float)):
            st.write(f"**Calculated Power:** {power_val:.3f}")
        else:
            st.write(f"**Calculated Power:** {power_val}")
        st.write(f"**Expected Number of Events:** {result.get('events', 'N/A')}")
        df_data = {
            "Parameter": ["Hazard Ratio", "Median Survival (Group 1)", "Median Survival (Group 2)", 
                          "Sample Size (Group 1)", "Sample Size (Group 2)", "Total Sample Size", "Alpha"],
            "Value": [
                result.get('hr_for_display', 'N/A'), 
                result.get('median_survival1_param', 'N/A'), 
                result.get('median_survival2_derived', 'N/A'),
                result.get('n1_param', 'N/A'),
                result.get('n2_param', 'N/A'),
                result.get('total_n', 'N/A'),
                result.get('alpha_param', 'N/A') # Alpha used as input
            ]
        }

    elif calculation_type == "Minimum Detectable Effect":
        mde_val = result.get('mde', 'N/A')
        if isinstance(mde_val, (int, float)):
            st.write(f"**Minimum Detectable Hazard Ratio:** {mde_val:.3f}")
        else:
            st.write(f"**Minimum Detectable Hazard Ratio:** {mde_val}")
        st.write(f"**Expected Number of Events:** {result.get('events', 'N/A')}")
        df_data = {
            "Parameter": ["Median Survival (Group 1)", "Sample Size (Group 1)", 
                          "Sample Size (Group 2)", "Total Sample Size", "Power", "Alpha"],
            "Value": [
                result.get('median_survival1_param', 'N/A'),
                result.get('n1_param', 'N/A'),
                result.get('n2_param', 'N/A'),
                result.get('total_n', 'N/A'),
                result.get('power_param', 'N/A'), # Power used as input
                result.get('alpha_param', 'N/A')  # Alpha used as input
            ]
        }
    else:
        st.error("Invalid calculation type for display.")
        return

    # Display parameters in a table
    df = pd.DataFrame(df_data)
    st.table(df.set_index('Parameter'))

    if use_simulation:
        st.info("Results obtained using simulation.")

    # Display non-inferiority specific information
    if hypothesis_type == "Non-Inferiority":
        st.markdown("#### Non-Inferiority Interpretation")
        if calculation_type == "Sample Size":
            st.write(f"Sample size calculated to test if the upper bound of the confidence interval for the hazard ratio is below the margin of {result.get('non_inferiority_margin_hr_param', 'N/A')}, assuming a true hazard ratio of {result.get('assumed_true_hr_param', 'N/A')}.")
        elif calculation_type == "Power":
            st.write(f"Power to detect non-inferiority, defined as the upper bound of the confidence interval for the hazard ratio being below the margin of {result.get('non_inferiority_margin_hr_param', 'N/A')}, given an assumed true hazard ratio of {result.get('assumed_true_hr_param', 'N/A')}.")
        # MDE is typically not the primary focus for NI, but if calculated as HR, context is similar.


def create_survival_visualization(result, calculation_type, hypothesis_type):
    """Create visualization for survival outcome results."""
    st.subheader("Visualizations")

    if not result:
        st.warning("No data available for visualization.")
        return

    fig, ax = plt.subplots()

    if calculation_type == "Sample Size" and 'power_curve_data' in result and result['power_curve_data']:
        df_power_curve = pd.DataFrame(result['power_curve_data'])
        if not df_power_curve.empty:
            ax.plot(df_power_curve['total_n'], df_power_curve['power'], marker='o')
            ax.set_xlabel("Total Sample Size")
            ax.set_ylabel("Power")
            ax.set_title("Power vs. Sample Size")
            ax.grid(True)
            st.pyplot(fig)

            if 'events' in df_power_curve.columns:
                fig_events, ax_events = plt.subplots()
                ax_events.plot(df_power_curve['total_n'], df_power_curve['events'], marker='o', color='green')
                ax_events.set_xlabel("Total Sample Size")
                ax_events.set_ylabel("Number of Events")
                ax_events.set_title("Events vs. Sample Size")
                ax_events.grid(True)
                st.pyplot(fig_events)
        else:
            st.info("No power curve data available for visualization.")

    elif calculation_type == "Power" and 'survival_curves' in result and result['survival_curves']:
        df_curves = pd.DataFrame(result['survival_curves'])
        if not df_curves.empty:
            ax.plot(df_curves['time'], df_curves['survival_group1'], label='Group 1')
            ax.plot(df_curves['time'], df_curves['survival_group2'], label='Group 2')
            ax.set_xlabel("Time")
            ax.set_ylabel("Survival Probability")
            ax.set_title("Survival Curves")
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)
        else:
            st.info("No survival curve data available for visualization.")
            
    elif calculation_type == "Minimum Detectable Effect" and 'power_vs_hr_data' in result and result['power_vs_hr_data']:
        df_mde_curve = pd.DataFrame(result['power_vs_hr_data'])
        if not df_mde_curve.empty:
            ax.plot(df_mde_curve['hr'], df_mde_curve['power'], marker='o')
            ax.set_xlabel("Hazard Ratio")
            ax.set_ylabel("Power")
            ax.set_title("Power vs. Hazard Ratio")
            ax.grid(True)
            st.pyplot(fig)
        else:
            st.info("No MDE curve data available for visualization.")
    else:
        st.info("Visualization for this combination of calculation type and results is not currently available.")
