"""Binary outcome UI components for Parallel RCT designs.

This module provides UI rendering functions for binary outcomes
in Parallel RCT designs.
"""
import streamlit as st


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