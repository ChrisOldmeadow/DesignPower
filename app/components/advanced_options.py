"""
Advanced options components for the DesignPower application.

This module contains reusable components for rendering advanced options
UI elements across different design types.
"""
import streamlit as st

def render_continuous_advanced_options():
    """
    Render advanced options for continuous outcomes.
    
    Returns:
        dict: Dictionary containing the advanced options settings
    """
    # For continuous outcomes, show variance and repeated measures options
    st.write("Advanced Analysis Options:")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        unequal_var = st.checkbox("Unequal Variances", value=False, key="unequal_var_checkbox")
        
    with col2:
        repeated_measures = st.checkbox("Repeated Measures (Baseline + Follow-up)", value=False, key="repeated_measures_checkbox")
    
    options = {
        "unequal_var": unequal_var,
        "repeated_measures": repeated_measures,
        "std_dev": None,
        "std_dev2": None,
        "correlation": None,
        "analysis_method": None
    }
    
    # Show a horizontal separator if options are selected
    if unequal_var or repeated_measures:
        st.markdown("---")
    
    # Unequal variances option
    if unequal_var and not repeated_measures:
        st.write("Specify different standard deviations for each group:")
        std_dev = st.number_input("Standard Deviation (Group 1)", value=1.0, step=0.1, min_value=0.1, key="sd_group1_unequal")
        std_dev2 = st.number_input("Standard Deviation (Group 2)", value=1.2, step=0.1, min_value=0.1, key="sd_group2")
        st.info("Using Welch's t-test approximation for unequal variances.")
        
        options["std_dev"] = std_dev
        options["std_dev2"] = std_dev2
    
    # Repeated measures option
    elif repeated_measures:
        st.write("Repeated Measures Analysis Settings:")
        correlation = st.slider("Correlation between Baseline and Follow-up", 
                             min_value=0.0, max_value=0.95, value=0.5, step=0.05,
                             key="correlation_slider")
        
        analysis_method = st.radio(
            "Analysis Method",
            ["Change Score", "ANCOVA"],
            horizontal=True,
            key="analysis_method_radio"
        )
        
        options["correlation"] = correlation
        options["analysis_method"] = analysis_method
        
        # Show explanation based on selected method
        if analysis_method == "Change Score":
            st.info("Change score analysis compares the differences (follow-up minus baseline) between groups.")
            if correlation > 0.5:
                st.warning("Note: When correlation > 0.5, ANCOVA is typically more efficient than change score analysis.")
        else:  # ANCOVA
            st.info("ANCOVA adjusts follow-up scores for baseline differences, typically more efficient when correlation > 0.5.")
    
    # Standard option - only display message if no additional parameters are selected
    if not unequal_var and not repeated_measures:
        st.info("Standard analysis assumes equal variances across groups.")
        
    return options

def render_binary_advanced_options():
    """
    Render advanced options for binary outcomes.
    
    Returns:
        dict: Dictionary containing the advanced options settings
    """
    # Binary outcomes don't use unequal variances or repeated measures
    options = {
        "unequal_var": False,
        "repeated_measures": False,
        "test_type": None
    }
    
    # Binary outcome specific options - immediately show test type options
    test_type = st.radio(
        "Statistical Test",
        ["Normal Approximation", "Likelihood Ratio Test", "Exact Test"],
        index=0,  # Default to Normal Approximation
        key="binary_test_type",
        horizontal=True
    )
    
    options["test_type"] = test_type
    
    # Explain the selected test type
    if test_type == "Normal Approximation":
        st.info("Normal approximation uses the z-test to compare proportions. Fast and reliable for moderate to large sample sizes.")
    elif test_type == "Likelihood Ratio Test":
        st.info("Likelihood Ratio Test often has better statistical properties than the Normal approximation, especially with smaller sample sizes.")
    else:  # Exact Test
        st.info("Fisher's Exact Test provides the most accurate results for small sample sizes, but is computationally intensive for large samples.")
        
    return options

def render_survival_advanced_options():
    """
    Render advanced options for survival outcomes.
    
    Returns:
        dict: Dictionary containing the advanced options settings
    """
    # Basic options for survival analysis
    options = {
        "unequal_var": False,
        "repeated_measures": False,
        "method": "analytical",  # Default to analytical method
        "nsim": 1000,  # Default number of simulations
        "min_n": 10,  # Default minimum sample size to try
        "max_n": 1000,  # Default maximum sample size to try
        "step": 10  # Default step size for incrementing sample size
    }
    
    # Add method selection (Analytical vs Simulation)
    method_col1, method_col2 = st.columns([1, 2])
    with method_col1:
        method = st.radio(
            "Calculation Method",
            ["Analytical", "Simulation"],
            index=0,  # Default to Analytical
            key="survival_calc_method",
            horizontal=True
        )
        options["method"] = method.lower()
    
    # Show simulation options only if simulation method is selected
    if method == "Simulation":
        with method_col2:
            st.info("Simulation provides more realistic estimates but takes longer to compute.")
        
        st.markdown("---")
        st.write("Simulation Settings:")
        
        sim_col1, sim_col2 = st.columns(2)
        
        with sim_col1:
            nsim = st.number_input(
                "Number of Simulations", 
                min_value=100, 
                max_value=10000, 
                value=1000, 
                step=100,
                key="survival_nsim"
            )
            options["nsim"] = nsim
        
        with sim_col2:
            st.write("Sample Size Search Range:")
            min_n = st.number_input(
                "Minimum Sample Size", 
                min_value=2, 
                max_value=100, 
                value=10, 
                step=2,
                key="survival_min_n"
            )
            max_n = st.number_input(
                "Maximum Sample Size", 
                min_value=100, 
                max_value=10000, 
                value=1000, 
                step=50,
                key="survival_max_n"
            )
            step = st.number_input(
                "Step Size", 
                min_value=1, 
                max_value=50, 
                value=10, 
                step=1,
                key="survival_step"
            )
            
            options["min_n"] = min_n
            options["max_n"] = max_n
            options["step"] = step
    else:
        with method_col2:
            st.info("Analytical calculations are fast but make assumptions about the data distribution.")
    
    return options
