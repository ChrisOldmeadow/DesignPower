"""Continuous outcome UI components for Parallel RCT designs.

This module provides UI rendering functions for continuous outcomes
in Parallel RCT designs.
"""
import streamlit as st


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
                params["non_inferiority_margin"] = st.number_input("Non-inferiority Margin (NIM)", value=0.2, step=0.1, key="margin_input")
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
                params["non_inferiority_direction"] = st.selectbox(
                    "Direction",
                    ["lower", "upper"],
                    index=0,
                    key="direction_select"
                )
                # Explain the direction based on selection
                if params["non_inferiority_direction"] == "lower":
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