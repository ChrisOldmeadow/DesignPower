"""
Streamlit frontend for sample size calculator.

This module provides a web interface for sample size calculation,
power analysis, and simulation-based estimation.
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import requests
import os
import sys
from scipy import stats

# Add parent directory to path to import core modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.compatibility import (
    sample_size_difference_in_means,
    power_difference_in_means,
    power_binary_cluster_rct,
    sample_size_binary_cluster_rct,
    min_detectable_effect_binary_cluster_rct,
    simulate_parallel_rct,
    simulate_cluster_rct,
    simulate_stepped_wedge,
    simulate_binary_cluster_rct,
    simulate_min_detectable_effect,
    simulate_sample_size
)

# Import from new structure (needed for new advanced options)
from core.designs.parallel.analytical import (
    sample_size_repeated_measures,
    power_repeated_measures,
    min_detectable_effect_repeated_measures
)

# Import visualization utilities
from core.utils.visualization import power_curve, sample_size_curve

# Define utility functions that were previously in core.utils
def create_power_curve(x_values, y_values, x_label, y_label, title, target_line=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x_values, y_values, 'b-', linewidth=2)
    
    if target_line is not None:
        ax.axhline(y=target_line, color='r', linestyle='--', alpha=0.7)
    
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    return fig, ax

def generate_code_snippet(function_name, params):
    """Generate a Python code snippet for the calculation."""
    param_str = ', '.join([f'{k}={v}' for k, v in params.items()])
    return f"from core.compatibility import {function_name}\n\nresult = {function_name}({param_str})\nprint(result)"

def generate_plain_language_summary(result, design_type, calculation_type):
    """Generate a plain language summary of the results."""
    # Start with the basic summary
    if calculation_type == "Sample Size":
        if "total_n" in result:
            basic_summary = f"The study requires a total of {result['total_n']} participants to detect the specified effect size with {result['parameters'].get('power', 0.8)*100:.0f}% power at a significance level of {result['parameters'].get('alpha', 0.05)}."
        elif "n_clusters" in result:
            total_n = result['n_clusters'] * 2 * result['parameters'].get('cluster_size', 0)
            basic_summary = f"The study requires {result['n_clusters']} clusters per arm ({result['n_clusters']*2} total clusters) with {result['parameters'].get('cluster_size', 0)} individuals per cluster, for a total of {total_n} participants."
    elif calculation_type == "Power":
        basic_summary = f"With the specified sample size, the study has {result['power']*100:.1f}% power to detect the specified effect size at a significance level of {result['parameters'].get('alpha', 0.05)}."
    elif calculation_type == "Minimum Detectable Effect":
        if "delta" in result:
            basic_summary = f"With the specified sample size, the smallest effect that can be detected with {result['parameters'].get('power', 0.8)*100:.0f}% power is {result['delta']:.3f}."
        elif "p2" in result:
            basic_summary = f"With the specified sample size and control proportion of {result['parameters'].get('p1', 0)}, the smallest detectable proportion in the intervention group is {result['p2']:.3f}."
    else:
        return "Results summary not available for this calculation type."
    
    # Add method information and references
    method_info = get_method_information(result, design_type, calculation_type)
    
    return f"{basic_summary}\n\n{method_info}"    


def get_method_information(result, design_type, calculation_type):
    """Get information about the method used and relevant references."""
    # Check for repeated measures
    repeated_measures = False
    analysis_method = None
    if "parameters" in result and "method" in result["parameters"]:
        repeated_measures = True
        analysis_method = result["parameters"]["method"]
    
    # Check for unequal variances
    unequal_var = False
    if "parameters" in result and "std_dev2" in result["parameters"] and result["parameters"]["std_dev2"] is not None:
        unequal_var = True
    
    # Check if simulation was used
    simulation = "nsim" in result["parameters"] if "parameters" in result else False
    
    # Build method description and references based on design type
    if "Parallel RCT" in design_type and "Continuous Outcome" in design_type:
        if "nsim" in result["parameters"] or simulation_used:
            if calculation_type == "Sample Size":
                achieved_power = result["parameters"].get("achieved_power", 0.8)
                return (f"**Method:** Simulation-based sample size determination. Achieved power: {achieved_power:.3f}.\n\n"
                       f"**Reference:** Burton A, Altman DG, Royston P, Holder RL. The design of simulation studies in medical statistics. *Statistics in Medicine*. 2006;25(24):4279-4292.")
            elif calculation_type == "Minimum Detectable Effect (MDE)":
                return ("**Method:** Simulation-based optimization approach for minimum detectable effect.\n\n"
                       "**Reference:** Morris TP, White IR, Crowther MJ. Using simulation studies to evaluate statistical methods. *Statistics in Medicine*. 2019;38(11):2074-2102.")
            else:
                return "**Method:** Simulation-based estimation with Monte Carlo methods."
        else:
            if repeated_measures:
                method_type = "change score analysis" if analysis_method == "change_score" else "ANCOVA"
                return (f"**Method:** Analytical formula for repeated measures design using {method_type}.\n\n"
                        f"**Reference:** Frison L, Pocock SJ. Repeated measures in clinical trials: analysis using mean summary statistics and its implications for design. *Statistics in Medicine*. 1992;11(13):1685-1704.")
            elif unequal_var:
                return ("**Method:** Analytical formula for two-sample t-test with unequal variances (Welch's t-test).\n\n"
                        "**Reference:** Welch BL. The generalization of 'Student's' problem when several different population variances are involved. *Biometrika*. 1947;34(1-2):28-35.")
            else:
                return ("**Method:** Analytical formula for two-sample t-test with equal variances.\n\n"
                        "**Reference:** Cohen J. Statistical Power Analysis for the Behavioral Sciences. 2nd ed. Lawrence Erlbaum Associates; 1988.")
    
    elif "Cluster RCT" in design_type and "Continuous Outcome" in design_type:
        if simulation:
            return "**Method:** Simulation-based estimation accounting for clustering effects."
        else:
            return ("**Method:** Analytical formula for cluster randomized trials.\n\n"
                    "**Reference:** Donner A, Klar N. Design and Analysis of Cluster Randomization Trials in Health Research. Arnold; 2000.")
    
    elif "Cluster RCT" in design_type and "Binary Outcome" in design_type:
        if simulation:
            return "**Method:** Simulation-based estimation for binary outcomes in cluster randomized trials."
        else:
            return ("**Method:** Analytical formula for binary outcomes in cluster randomized trials.\n\n"
                    "**Reference:** Hayes RJ, Moulton LH. Cluster Randomised Trials. 2nd ed. Chapman & Hall/CRC; 2017.")
    
    elif "Stepped Wedge" in design_type:
        return ("**Method:** Simulation-based estimation for stepped wedge cluster randomized trial.\n\n"
                "**Reference:** Hussey MA, Hughes JP. Design and analysis of stepped wedge cluster randomized trials. *Contemporary Clinical Trials*. 2007;28(2):182-191.")
    
    elif "Parallel RCT" in design_type and "Binary Outcome" in design_type:
        if simulation:
            return "**Method:** Simulation-based estimation for binary outcomes."
        else:
            return ("**Method:** Analytical formula for comparing proportions.\n\n"
                    "**Reference:** Fleiss JL, Levin B, Paik MC. Statistical Methods for Rates and Proportions. 3rd ed. Wiley; 2003.")
    
    # Default case
    return "**Method:** Standard analytical formula."

# Set page config
st.set_page_config(
    page_title="Sample Size Calculator",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# App title
st.title("Study Design Power Calculator")
st.markdown("A tool for sample size calculation, power analysis, and simulation-based estimation.")

# Create tabs for different functionalities
tab1, tab2, tab3 = st.tabs(["Calculator", "About", "Documentation"])

with tab1:
    # Sidebar for input parameters
    with st.sidebar:
        st.header("Study Design Parameters")
        
        # Select study design and outcome type separately
        study_design = st.selectbox(
            "Study Design",
            [
                "Parallel RCT",
                "Cluster RCT",
                "Stepped Wedge Trial",
                "Interrupted Time Series"
            ]
        )
        
        # Select outcome type (disabled for Stepped Wedge for now since it only supports continuous)
        outcome_type = "Continuous" if study_design == "Stepped Wedge Trial" else st.selectbox(
            "Outcome Type",
            [
                "Continuous",
                "Binary",
                "Count",
                "Survival"
            ],
            disabled=study_design == "Stepped Wedge Trial"
        )
        
        # Combine for backward compatibility
        if study_design == "Parallel RCT":
            if outcome_type == "Continuous":
                design_type = "Parallel RCT (Continuous Outcome)"
            elif outcome_type == "Binary":
                design_type = "Parallel RCT (Binary Outcome)"
            else:
                design_type = f"Parallel RCT ({outcome_type} Outcome)"
        elif study_design == "Cluster RCT":
            if outcome_type == "Continuous":
                design_type = "Cluster RCT (Continuous Outcome)"
            elif outcome_type == "Binary":
                design_type = "Cluster RCT (Binary Outcome)"
            else:
                design_type = f"Cluster RCT ({outcome_type} Outcome)"
        else:
            design_type = study_design
            
        # Inform user if combination is not yet implemented
        if (outcome_type in ["Count", "Survival"]) or (study_design == "Interrupted Time Series"):
            st.sidebar.warning(f"{study_design} with {outcome_type} outcome is not yet fully implemented in the UI. Some calculations may not work.")
        
        # Select calculation type
        calculation_type = st.selectbox(
            "Calculation Type",
            [
                "Sample Size",
                "Power",
                "Minimum Detectable Effect (MDE)"
            ]
        )
        
        # Show parameters based on selected design
        with st.expander("Basic Parameters", expanded=True):
            if "Continuous Outcome" in design_type:
                if calculation_type == "Sample Size":
                    # Option to input effect size directly or as means (for all continuous outcome designs)
                    effect_size_input_method = st.radio(
                        "Effect Size Input Method",
                        ["Direct Effect Size", "Group Means"],
                        horizontal=True
                    )
                    
                    if effect_size_input_method == "Direct Effect Size":
                        delta = st.number_input("Effect Size (Difference in Means)", value=0.5, step=0.1)
                    else:  # Group Means option
                        if "Stepped Wedge" in design_type:
                            mean_control = st.number_input("Mean Control Phase", value=0.0, step=0.1)
                            mean_intervention = st.number_input("Mean Intervention Phase", value=0.5, step=0.1)
                        else:  # Parallel or Cluster
                            mean_control = st.number_input("Mean Control Group", value=0.0, step=0.1)
                            mean_intervention = st.number_input("Mean Intervention Group", value=0.5, step=0.1)
                        delta = abs(mean_intervention - mean_control)  # Calculate effect size automatically
                        st.info(f"Calculated effect size (absolute difference): {delta}")
                        
                    std_dev = st.number_input("Standard Deviation", value=1.0, step=0.1, min_value=0.1)
                    power = st.slider("Power", min_value=0.5, max_value=0.99, value=0.8, step=0.01)
                    alpha = st.slider("Significance Level (α)", min_value=0.01, max_value=0.1, value=0.05, step=0.01)
                    
                elif calculation_type == "Power":
                    # Set up sample size inputs based on design
                    if "Parallel" in design_type:
                        n1 = st.number_input("Sample Size (Group 1)", value=50, step=1, min_value=2)
                        n2 = st.number_input("Sample Size (Group 2)", value=50, step=1, min_value=2)
                    elif "Cluster" in design_type:
                        n_clusters = st.number_input("Number of Clusters per Arm", value=10, step=1, min_value=2)
                        cluster_size = st.number_input("Cluster Size", value=20, step=1, min_value=2)
                    elif "Stepped Wedge" in design_type:
                        clusters = st.number_input("Number of Clusters", value=12, step=1, min_value=2)
                        steps = st.number_input("Number of Time Steps", value=4, step=1, min_value=2)
                        individuals_per_cluster = st.number_input("Individuals per Cluster per Step", value=10, step=1, min_value=1)
                    
                    # Option to input effect size directly or as means for all continuous designs
                    effect_size_input_method = st.radio(
                        "Effect Size Input Method",
                        ["Direct Effect Size", "Group Means"],
                        horizontal=True
                    )
                    
                    if effect_size_input_method == "Direct Effect Size":
                        delta = st.number_input("Effect Size (Difference in Means)", value=0.5, step=0.1)
                    else:  # Group Means option
                        if "Stepped Wedge" in design_type:
                            mean_control = st.number_input("Mean Control Phase", value=0.0, step=0.1)
                            mean_intervention = st.number_input("Mean Intervention Phase", value=0.5, step=0.1)
                        else:  # Parallel or Cluster
                            mean_control = st.number_input("Mean Control Group", value=0.0, step=0.1)
                            mean_intervention = st.number_input("Mean Intervention Group", value=0.5, step=0.1)
                        delta = abs(mean_intervention - mean_control)  # Calculate effect size automatically
                        st.info(f"Calculated effect size (absolute difference): {delta}")
                    
                    std_dev = st.number_input("Standard Deviation", value=1.0, step=0.1, min_value=0.1)
                    alpha = st.slider("Significance Level (α)", min_value=0.01, max_value=0.1, value=0.05, step=0.01)
                    
                elif calculation_type == "Minimum Detectable Effect (MDE)":
                    if "Parallel" in design_type:
                        n1 = st.number_input("Sample Size (Group 1)", value=50, step=1, min_value=2)
                        n2 = st.number_input("Sample Size (Group 2)", value=50, step=1, min_value=2)
                    else:  # Cluster
                        n_clusters = st.number_input("Number of Clusters per Arm", value=10, step=1, min_value=2)
                        cluster_size = st.number_input("Cluster Size", value=20, step=1, min_value=2)
                    
                    std_dev = st.number_input("Standard Deviation", value=1.0, step=0.1, min_value=0.1)
                    power = st.slider("Power", min_value=0.5, max_value=0.99, value=0.8, step=0.01)
                    alpha = st.slider("Significance Level (α)", min_value=0.01, max_value=0.1, value=0.05, step=0.01)
                
            elif "Binary Outcome" in design_type:
                if calculation_type == "Sample Size":
                    p1 = st.slider("Control Group Proportion", min_value=0.01, max_value=0.99, value=0.5, step=0.01)
                    p2 = st.slider("Intervention Group Proportion", min_value=0.01, max_value=0.99, value=0.6, step=0.01)
                    power = st.slider("Power", min_value=0.5, max_value=0.99, value=0.8, step=0.01)
                    alpha = st.slider("Significance Level (α)", min_value=0.01, max_value=0.1, value=0.05, step=0.01)
                    
                elif calculation_type == "Power":
                    if "Parallel" in design_type:
                        n1 = st.number_input("Sample Size (Group 1)", value=100, step=1, min_value=2)
                        n2 = st.number_input("Sample Size (Group 2)", value=100, step=1, min_value=2)
                    else:  # Cluster
                        n_clusters = st.number_input("Number of Clusters per Arm", value=10, step=1, min_value=2)
                        cluster_size = st.number_input("Cluster Size", value=20, step=1, min_value=2)
                    
                    p1 = st.slider("Control Group Proportion", min_value=0.01, max_value=0.99, value=0.5, step=0.01)
                    p2 = st.slider("Intervention Group Proportion", min_value=0.01, max_value=0.99, value=0.6, step=0.01)
                    alpha = st.slider("Significance Level (α)", min_value=0.01, max_value=0.1, value=0.05, step=0.01)
                    
                elif calculation_type == "Minimum Detectable Effect (MDE)":
                    if "Parallel" in design_type:
                        n1 = st.number_input("Sample Size (Group 1)", value=100, step=1, min_value=2)
                        n2 = st.number_input("Sample Size (Group 2)", value=100, step=1, min_value=2)
                    else:  # Cluster
                        n_clusters = st.number_input("Number of Clusters per Arm", value=10, step=1, min_value=2)
                        cluster_size = st.number_input("Cluster Size", value=20, step=1, min_value=2)
                    
                    p1 = st.slider("Control Group Proportion", min_value=0.01, max_value=0.99, value=0.5, step=0.01)
                    power = st.slider("Power", min_value=0.5, max_value=0.99, value=0.8, step=0.01)
                    alpha = st.slider("Significance Level (α)", min_value=0.01, max_value=0.1, value=0.05, step=0.01)
            
            elif design_type == "Stepped Wedge Trial":
                if calculation_type != "Power":  # We already handle Power case in the combined section above
                    clusters = st.number_input("Number of Clusters", value=12, step=1, min_value=2)
                    steps = st.number_input("Number of Time Steps", value=4, step=1, min_value=2)
                    individuals_per_cluster = st.number_input("Individuals per Cluster per Step", value=10, step=1, min_value=1)
                icc = st.slider("Intracluster Correlation Coefficient (ICC)", min_value=0.0, max_value=0.5, value=0.05, step=0.01)
                
                # For Stepped Wedge, add effect size input method if not already handled
                if calculation_type != "Power":
                    effect_size_input_method = st.radio(
                        "Effect Size Input Method",
                        ["Direct Effect Size", "Group Means"],
                        horizontal=True
                    )
                    
                    if effect_size_input_method == "Direct Effect Size":
                        delta = st.number_input("Effect Size (Difference in Means)", value=0.5, step=0.1)
                    else:  # Group Means option
                        mean_control = st.number_input("Mean Control Phase", value=0.0, step=0.1)
                        mean_intervention = st.number_input("Mean Intervention Phase", value=0.5, step=0.1)
                        delta = abs(mean_intervention - mean_control)  # Calculate effect size automatically
                        st.info(f"Calculated effect size (absolute difference): {delta}")
                
                if calculation_type == "Power":
                    treatment_effect = st.number_input("Treatment Effect", value=0.5, step=0.1)
                    std_dev = st.number_input("Standard Deviation", value=1.0, step=0.1, min_value=0.1)
                    alpha = st.slider("Significance Level (α)", min_value=0.01, max_value=0.1, value=0.05, step=0.01)
                
                elif calculation_type == "Sample Size":
                    treatment_effect = st.number_input("Treatment Effect", value=0.5, step=0.1)
                    std_dev = st.number_input("Standard Deviation", value=1.0, step=0.1, min_value=0.1)
                    power = st.slider("Power", min_value=0.5, max_value=0.99, value=0.8, step=0.01)
                    alpha = st.slider("Significance Level (α)", min_value=0.01, max_value=0.1, value=0.05, step=0.01)
                
                elif calculation_type == "Minimum Detectable Effect (MDE)":
                    std_dev = st.number_input("Standard Deviation", value=1.0, step=0.1, min_value=0.1)
                    power = st.slider("Power", min_value=0.5, max_value=0.99, value=0.8, step=0.01)
                    alpha = st.slider("Significance Level (α)", min_value=0.01, max_value=0.1, value=0.05, step=0.01)
                
        # Advanced parameters for specific designs
        with st.expander("Advanced Parameters", expanded=False):
            # Advanced options for Parallel RCT with Continuous Outcome
            if "Parallel RCT (Continuous Outcome)" == design_type:
                st.write("Advanced Analysis Options:")
                
                # Use checkboxes for the main options
                col1, col2 = st.columns(2)
                
                with col1:
                    unequal_var = st.checkbox("Unequal Variances", value=False, key="unequal_var_checkbox")
                    
                with col2:
                    repeated_measures = st.checkbox("Repeated Measures (Baseline + Follow-up)", value=False, key="repeated_measures_checkbox")
                
                # Show a horizontal separator
                if unequal_var or repeated_measures:
                    st.markdown("---")
                
                # Unequal variances option
                if unequal_var and not repeated_measures:  # Only show if unequal selected and repeated not selected
                    st.write("Specify different standard deviations for each group:")
                    std_dev = st.number_input("Standard Deviation (Group 1)", value=1.0, step=0.1, min_value=0.1, key="sd_group1_unequal")
                    std_dev2 = st.number_input("Standard Deviation (Group 2)", value=1.2, step=0.1, min_value=0.1, key="sd_group2")
                    st.info("Using Welch's t-test approximation for unequal variances.")
                
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
                    
                    # Show explanation based on selected method
                    if analysis_method == "Change Score":
                        st.info("Change score analysis compares the differences (follow-up minus baseline) between groups.")
                        if correlation > 0.5:
                            st.warning("Note: When correlation > 0.5, ANCOVA is typically more efficient than change score analysis.")
                    else:  # ANCOVA
                        st.info("ANCOVA adjusts follow-up scores for baseline differences, typically more efficient when correlation > 0.5.")
                
                # Standard option - no additional parameters needed
                elif not unequal_var and not repeated_measures:
                    st.info("Standard analysis assumes equal variances across groups.")
            
            # Add simulation-specific options
            if design_type == "Stepped Wedge Trial" or "Cluster" in design_type:
                if calculation_type == "Power":
                    nsim = st.slider("Number of Simulations", min_value=100, max_value=10000, value=1000, step=100, key="nsim_cluster_slider")
            
            # Original advanced parameters for other designs
            if "Cluster" in design_type or design_type == "Stepped Wedge Trial":
                st.markdown("---")
                if "Cluster" in design_type and calculation_type in ["Power", "Sample Size"]:
                    icc = st.slider("Intracluster Correlation Coefficient (ICC)", min_value=0.0, max_value=0.5, value=0.05, step=0.01)
                
                # Always show simulation options for cluster or stepped wedge designs
                if design_type == "Stepped Wedge Trial" or ("Cluster" in design_type and calculation_type == "Power"):
                    nsim = st.slider("Number of Simulations", min_value=100, max_value=10000, value=1000, step=100, key="nsim_cluster_slider")
            
            if calculation_type == "Sample Size" and "Parallel" in design_type:
                st.markdown("---")
                allocation_ratio = st.slider("Allocation Ratio (n2/n1)", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
        
        # Simulation toggle for designs that support both analytical and simulation methods
        if design_type != "Stepped Wedge Trial" and "Cluster" not in design_type and calculation_type in ["Sample Size", "Power", "Minimum Detectable Effect (MDE)"]:
            use_simulation = st.checkbox("Use Simulation-based Estimation", value=False, key="use_simulation_checkbox")
            if use_simulation:
                nsim = st.slider("Number of Simulations", min_value=100, max_value=10000, value=1000, step=100, key="nsim_slider")
                if calculation_type == "Sample Size":
                    st.info("Simulation iteratively finds the minimum sample size that achieves the target power.")
                    max_n = st.number_input("Maximum Sample Size to Try", value=1000, step=50, min_value=100)
                    step_size = st.number_input("Sample Size Step", value=10, step=5, min_value=1)
                elif calculation_type == "Power":
                    st.info("Simulation provides empirical power estimates based on Monte Carlo methods.")
                else:  # MDE
                    st.info("Simulation uses an optimization approach to find the minimum detectable effect that achieves the desired power.")
        else:
            # Set simulation to always true for stepped wedge and cluster designs
            use_simulation = True
        
        # Option to generate power curve
        generate_curve = st.checkbox("Generate Power Curve", value=False)
        
        if generate_curve:
            curve_parameter = st.selectbox(
                "Parameter to Vary",
                ["Sample Size", "Effect Size", "ICC"] if "Cluster" in design_type else ["Sample Size", "Effect Size"]
            )
            
            if curve_parameter == "Sample Size":
                if "Parallel" in design_type:
                    min_n = st.number_input("Minimum Sample Size (per group)", value=10, step=10, min_value=2)
                    max_n = st.number_input("Maximum Sample Size (per group)", value=200, step=10, min_value=10)
                    step_n = st.number_input("Step Size", value=10, step=5, min_value=1)
                else:  # Cluster or Stepped Wedge
                    min_n = st.number_input("Minimum Number of Clusters", value=4, step=2, min_value=2)
                    max_n = st.number_input("Maximum Number of Clusters", value=30, step=2, min_value=4)
                    step_n = st.number_input("Step Size", value=2, step=1, min_value=1)
            
            elif curve_parameter == "Effect Size":
                min_effect = st.number_input("Minimum Effect Size", value=0.1, step=0.1, min_value=0.01)
                max_effect = st.number_input("Maximum Effect Size", value=1.0, step=0.1, min_value=0.1)
                step_effect = st.number_input("Step Size", value=0.1, step=0.05, min_value=0.01)
            
            elif curve_parameter == "ICC":
                min_icc = st.number_input("Minimum ICC", value=0.01, step=0.01, min_value=0.0)
                max_icc = st.number_input("Maximum ICC", value=0.3, step=0.01, min_value=0.01)
                step_icc = st.number_input("Step Size", value=0.02, step=0.01, min_value=0.01)

    # Main area for results
    st.header("Results")
    
    # Run calculation button
    run_calculation = st.button("Run Calculation")
    
    if run_calculation:
        with st.spinner("Calculating..."):
            # Logic for different designs and calculation types
            try:
                result = None
                
                # Parallel RCT with Continuous Outcome
                if design_type == "Parallel RCT (Continuous Outcome)":
                    # Extract variables from advanced options
                    std_dev2 = None
                    correlation = None
                    analysis_method = None
                    
                    # Check if advanced options were used
                    try:
                        unequal_var = st.session_state.get("unequal_var_checkbox", False)
                        repeated_measures = st.session_state.get("repeated_measures_checkbox", False)
                        
                        if unequal_var and not repeated_measures:
                            std_dev2 = st.session_state.get("sd_group2", 1.2)
                        elif repeated_measures:
                            correlation = st.session_state.get("correlation_slider", 0.5)
                            analysis_method = "change_score" if st.session_state.get("analysis_method_radio", "Change Score") == "Change Score" else "ancova"
                    except:
                        pass  # Fallback to standard analysis if advanced options not found
                    
                    if calculation_type == "Sample Size":
                        # Get simulation checkbox state from session state
                        use_simulation = st.session_state.get("use_simulation_checkbox", False)
                        
                        if use_simulation:
                            # Use simulation-based approach for sample size calculation
                            nsim_value = st.session_state.get("nsim_slider", 1000)
                            max_n_value = st.session_state.get("Maximum Sample Size to Try", 1000)
                            step_size_value = st.session_state.get("Sample Size Step", 10)
                            
                            # Handle repeated measures parameters if enabled
                            if repeated_measures:
                                result = simulate_sample_size(
                                    delta=delta,
                                    std_dev=std_dev,
                                    power=power,
                                    alpha=alpha,
                                    allocation_ratio=allocation_ratio,
                                    nsim=nsim_value,
                                    max_n=max_n_value,
                                    step=step_size_value,
                                    repeated_measures=True,
                                    correlation=correlation,
                                    method=analysis_method
                                )
                            else:
                                result = simulate_sample_size(
                                    delta=delta,
                                    std_dev=std_dev,
                                    power=power,
                                    alpha=alpha,
                                    allocation_ratio=allocation_ratio,
                                    nsim=nsim_value,
                                    max_n=max_n_value,
                                    step=step_size_value
                                )
                            method_name = "simulate_sample_size"
                        else:
                            # Decide which analytical calculation function to use
                            if repeated_measures:
                                result = sample_size_repeated_measures(
                                    delta=delta,
                                    std_dev=std_dev,
                                    correlation=correlation,
                                    power=power,
                                    alpha=alpha,
                                    allocation_ratio=allocation_ratio,
                                    method=analysis_method
                                )
                            else:  # Standard or Unequal Variances
                                result = sample_size_difference_in_means(
                                    delta=delta,
                                    std_dev=std_dev,
                                    power=power,
                                    alpha=alpha,
                                    allocation_ratio=allocation_ratio,
                                    std_dev2=std_dev2
                                )
                            method_name = "sample_size_difference_in_means"
                    
                    elif calculation_type == "Power":
                        # Get simulation checkbox state (should already be defined from UI)
                        if use_simulation:
                            # Get number of simulations from session state
                            nsim_value = st.session_state.get("nsim_slider", 1000)
                            # Handle repeated measures parameters if enabled
                            if repeated_measures:
                                result = simulate_parallel_rct(
                                    n1=n1,
                                    n2=n2,
                                    mean1=0,
                                    mean2=delta,
                                    std_dev=std_dev,
                                    nsim=nsim_value,
                                    alpha=alpha,
                                    repeated_measures=True,
                                    correlation=correlation,
                                    method=analysis_method
                                )
                            else:
                                result = simulate_parallel_rct(
                                    n1=n1,
                                    n2=n2,
                                    mean1=0,
                                    mean2=delta,
                                    std_dev=std_dev,
                                    nsim=nsim_value,
                                    alpha=alpha
                                )
                            method_name = "simulate_parallel_rct"
                        else:
                            # Decide which calculation function to use
                            if repeated_measures:
                                result = power_repeated_measures(
                                    n1=n1,
                                    n2=n2,
                                    delta=delta,
                                    std_dev=std_dev,
                                    correlation=correlation,
                                    alpha=alpha,
                                    method=analysis_method
                                )
                            else:  # Standard or Unequal Variances
                                result = power_difference_in_means(
                                    n1=n1,
                                    n2=n2,
                                    delta=delta,
                                    std_dev=std_dev,
                                    alpha=alpha,
                                    std_dev2=std_dev2
                                )
                            method_name = "power_difference_in_means"
                    
                    elif calculation_type == "Minimum Detectable Effect (MDE)":
                        # Get simulation checkbox state from session state
                        use_simulation = st.session_state.get("use_simulation_checkbox", False)
                        
                        if use_simulation:
                            # Use simulation-based approach for MDE calculation
                            nsim_value = st.session_state.get("nsim_slider", 1000)
                            # Handle repeated measures parameters if enabled
                            if repeated_measures:
                                result = simulate_min_detectable_effect(
                                    n1=n1,
                                    n2=n2,
                                    std_dev=std_dev,
                                    power=power,
                                    nsim=nsim_value,
                                    alpha=alpha,
                                    repeated_measures=True,
                                    correlation=correlation,
                                    method=analysis_method
                                )
                            else:
                                result = simulate_min_detectable_effect(
                                    n1=n1,
                                    n2=n2,
                                    std_dev=std_dev,
                                    power=power,
                                    nsim=nsim_value,
                                    alpha=alpha
                                )
                            method_name = "simulate_min_detectable_effect"
                        else:
                            # Decide which analytical calculation function to use
                            if repeated_measures:
                                result = min_detectable_effect_repeated_measures(
                                    n1=n1,
                                    n2=n2,
                                    std_dev=std_dev,
                                    correlation=correlation,
                                    power=power,
                                    alpha=alpha,
                                    method=analysis_method
                                )
                            else:  # Standard or Unequal Variances
                                # For unequal variances, calculate MDE
                                z_alpha = stats.norm.ppf(1 - alpha/2)
                                z_beta = stats.norm.ppf(power)
                                
                                if unequal_var and std_dev2 is not None:
                                    # Welch-Satterthwaite approximation for unequal variances
                                    delta = (z_alpha + z_beta) * math.sqrt((std_dev**2 / n1) + (std_dev2**2 / n2))
                                else:
                                    # Standard calculation for equal variances
                                    delta = (z_alpha + z_beta) * std_dev * math.sqrt(1/n1 + 1/n2)
                                
                                result = {
                                    "delta": delta,
                                    "parameters": {
                                        "n1": n1,
                                        "n2": n2,
                                        "std_dev": std_dev,
                                        "std_dev2": std_dev2,
                                        "power": power,
                                        "alpha": alpha
                                    }
                                }
                            method_name = "min_detectable_effect"
                
                # Cluster RCT with Continuous Outcome
                elif design_type == "Cluster RCT (Continuous Outcome)":
                    if calculation_type == "Power":
                        result = simulate_cluster_rct(
                            n_clusters=n_clusters,
                            cluster_size=cluster_size,
                            icc=icc,
                            mean1=0,
                            mean2=delta,
                            std_dev=std_dev,
                            nsim=st.session_state.get("nsim_cluster_slider", 1000),
                            alpha=alpha
                        )
                        method_name = "simulate_cluster_rct"
                
                # Cluster RCT with Binary Outcome
                elif design_type == "Cluster RCT (Binary Outcome)":
                    if calculation_type == "Sample Size":
                        result = sample_size_binary_cluster_rct(
                            p1=p1,
                            p2=p2,
                            icc=icc,
                            cluster_size=cluster_size,
                            power=power,
                            alpha=alpha
                        )
                        method_name = "sample_size_binary_cluster_rct"
                    
                    elif calculation_type == "Power":
                        if use_simulation:
                            result = simulate_binary_cluster_rct(
                                n_clusters=n_clusters,
                                cluster_size=cluster_size,
                                icc=icc,
                                p1=p1,
                                p2=p2,
                                nsim=st.session_state.get("nsim_cluster_slider", 1000),
                                alpha=alpha
                            )
                            method_name = "simulate_binary_cluster_rct"
                        else:
                            result = power_binary_cluster_rct(
                                n_clusters=n_clusters,
                                cluster_size=cluster_size,
                                icc=icc,
                                p1=p1,
                                p2=p2,
                                alpha=alpha
                            )
                            method_name = "power_binary_cluster_rct"
                    
                    elif calculation_type == "Minimum Detectable Effect (MDE)":
                        result = min_detectable_effect_binary_cluster_rct(
                            n_clusters=n_clusters,
                            cluster_size=cluster_size,
                            icc=icc,
                            p1=p1,
                            power=power,
                            alpha=alpha
                        )
                        method_name = "min_detectable_effect_binary_cluster_rct"
                
                # Stepped Wedge Trial
                elif design_type == "Stepped Wedge Trial":
                    if calculation_type == "Power":
                        result = simulate_stepped_wedge(
                            clusters=clusters,
                            steps=steps,
                            individuals_per_cluster=individuals_per_cluster,
                            icc=icc,
                            treatment_effect=treatment_effect,
                            std_dev=std_dev,
                            nsim=st.session_state.get("nsim_cluster_slider", 1000),
                            alpha=alpha
                        )
                        method_name = "simulate_stepped_wedge"
                
                # Display results
                if result:
                    # Create three columns for results display
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Calculation Results")
                        for key, value in result.items():
                            if key != "parameters" and not isinstance(value, dict):
                                st.metric(label=key.replace("_", " ").title(), value=f"{value:.4f}" if isinstance(value, float) else value)
                    
                    with col2:
                        st.subheader("Input Parameters")
                        for key, value in result.get("parameters", {}).items():
                            if key != "nsim":
                                st.metric(label=key.replace("_", " ").title(), value=f"{value:.4f}" if isinstance(value, float) else value)
                    
                    # Summary and code snippet
                    st.subheader("Plain Language Summary")
                    summary = generate_plain_language_summary(result, design_type, calculation_type)
                    st.markdown(summary)
                    
                    st.subheader("Reproducible Code Snippet")
                    code_snippet = generate_code_snippet(method_name, result.get("parameters", {}))
                    st.code(code_snippet, language="python")
                    
                    # Optional power curve
                    if generate_curve:
                        st.subheader("Power Curve")
                        
                        fig, ax = plt.subplots(figsize=(10, 6))
                        
                        if curve_parameter == "Sample Size":
                            if "Parallel" in design_type:
                                # Sample size range for parallel design
                                n_range = np.arange(min_n, max_n + 1, step_n)
                                powers = []
                                
                                for n in n_range:
                                    if "Continuous Outcome" in design_type:
                                        power_val = power_difference_in_means(
                                            n1=n, 
                                            n2=n, 
                                            delta=delta, 
                                            std_dev=std_dev, 
                                            alpha=alpha
                                        )["power"]
                                    else:  # Binary outcome
                                        # Implementation for binary outcome would go here
                                        power_val = 0.5  # Placeholder
                                    
                                    powers.append(power_val)
                                
                                ax.plot(n_range, powers, 'o-', linewidth=2)
                                ax.set_xlabel("Sample Size per Group")
                                ax.set_ylabel("Statistical Power")
                                ax.set_title(f"Power Curve for Varying Sample Size")
                                ax.grid(True, linestyle='--', alpha=0.7)
                                ax.axhline(y=0.8, color='r', linestyle='--', alpha=0.5, label="Power = 0.8")
                                ax.legend()
                                
                            else:  # Cluster design
                                # Number of clusters range
                                n_range = np.arange(min_n, max_n + 1, step_n)
                                powers = []
                                
                                for n in n_range:
                                    if design_type == "Cluster RCT (Binary Outcome)":
                                        power_val = power_binary_cluster_rct(
                                            n_clusters=n,
                                            cluster_size=cluster_size,
                                            icc=icc,
                                            p1=p1,
                                            p2=p2,
                                            alpha=alpha
                                        )["power"]
                                    else:  # Stepped wedge or continuous outcome
                                        # Run a few simulations for the curve
                                        power_val = 0.5  # Placeholder
                                    
                                    powers.append(power_val)
                                
                                ax.plot(n_range, powers, 'o-', linewidth=2)
                                ax.set_xlabel("Number of Clusters per Arm")
                                ax.set_ylabel("Statistical Power")
                                ax.set_title(f"Power Curve for Varying Number of Clusters")
                                ax.grid(True, linestyle='--', alpha=0.7)
                                ax.axhline(y=0.8, color='r', linestyle='--', alpha=0.5, label="Power = 0.8")
                                ax.legend()
                        
                        elif curve_parameter == "Effect Size":
                            # Effect size range
                            effect_range = np.arange(min_effect, max_effect + 0.01, step_effect)
                            powers = []
                            
                            for effect in effect_range:
                                if "Continuous Outcome" in design_type:
                                    if "Parallel" in design_type:
                                        power_val = power_difference_in_means(
                                            n1=n1, 
                                            n2=n2, 
                                            delta=effect, 
                                            std_dev=std_dev, 
                                            alpha=alpha
                                        )["power"]
                                    else:  # Cluster
                                        # Would need simulation here
                                        power_val = 0.5  # Placeholder
                                else:  # Binary outcome
                                    if "Parallel" in design_type:
                                        # Implementation for binary outcome would go here
                                        power_val = 0.5  # Placeholder
                                    else:  # Cluster
                                        # Calculate p2 based on p1 and effect
                                        p2_effect = min(p1 + effect, 0.99)
                                        power_val = power_binary_cluster_rct(
                                            n_clusters=n_clusters,
                                            cluster_size=cluster_size,
                                            icc=icc,
                                            p1=p1,
                                            p2=p2_effect,
                                            alpha=alpha
                                        )["power"]
                                
                                powers.append(power_val)
                            
                            ax.plot(effect_range, powers, 'o-', linewidth=2)
                            ax.set_xlabel("Effect Size")
                            ax.set_ylabel("Statistical Power")
                            ax.set_title(f"Power Curve for Varying Effect Size")
                            ax.grid(True, linestyle='--', alpha=0.7)
                            ax.axhline(y=0.8, color='r', linestyle='--', alpha=0.5, label="Power = 0.8")
                            ax.legend()
                        
                        elif curve_parameter == "ICC":
                            # ICC range
                            icc_range = np.arange(min_icc, max_icc + 0.01, step_icc)
                            powers = []
                            
                            for icc_val in icc_range:
                                if design_type == "Cluster RCT (Binary Outcome)":
                                    power_val = power_binary_cluster_rct(
                                        n_clusters=n_clusters,
                                        cluster_size=cluster_size,
                                        icc=icc_val,
                                        p1=p1,
                                        p2=p2,
                                        alpha=alpha
                                    )["power"]
                                else:  # Other designs would need simulation
                                    power_val = 0.5  # Placeholder
                                
                                powers.append(power_val)
                            
                            ax.plot(icc_range, powers, 'o-', linewidth=2)
                            ax.set_xlabel("Intracluster Correlation Coefficient (ICC)")
                            ax.set_ylabel("Statistical Power")
                            ax.set_title(f"Power Curve for Varying ICC")
                            ax.grid(True, linestyle='--', alpha=0.7)
                            ax.axhline(y=0.8, color='r', linestyle='--', alpha=0.5, label="Power = 0.8")
                            ax.legend()
                        
                        st.pyplot(fig)
                
                else:
                    st.error(f"Calculation not implemented for {design_type} with {calculation_type}")
            
            except Exception as e:
                st.error(f"Error in calculation: {str(e)}")

with tab2:
    st.header("About Sample Size Calculator")
    st.markdown("""
    This tool helps researchers and statisticians plan studies by calculating:
    
    - **Sample size requirements**: How many participants are needed to detect an effect of interest
    - **Statistical power**: The probability of detecting an effect if it exists
    - **Minimum detectable effect**: The smallest effect size that can be detected with a given sample size
    
    The calculator supports various study designs, including:
    - Parallel randomized controlled trials (RCTs)
    - Cluster randomized controlled trials
    - Stepped wedge designs
    
    It handles both continuous and binary outcomes, and provides simulation-based estimates for complex designs.
    """)
    
    st.subheader("How to Use")
    st.markdown("""
    1. Select your study design and calculation type in the sidebar
    2. Enter the required parameters
    3. Click "Run Calculation"
    4. View results, including a plain-language summary and reproducible code
    5. Optionally, generate a power curve to explore how changing parameters affects power
    """)
    
    st.subheader("Methods")
    st.markdown("""
    This tool implements both analytical and simulation-based methods:
    
    - **Analytical methods**: Closed-form equations for simple designs
    - **Simulation-based methods**: Monte Carlo simulations for complex designs
    
    For stepped wedge designs and other complex scenarios, simulation is the default approach.
    """)

with tab3:
    st.header("Technical Documentation")
    st.markdown("""
    ### API Documentation
    
    The calculator is built on a FastAPI backend with endpoints for all supported calculations:
    
    - `/calculate/sample-size/*`: Endpoints for sample size calculation
    - `/calculate/power/*`: Endpoints for power calculation
    - `/calculate/simulation/*`: Endpoints for simulation-based estimation
    
    Each endpoint accepts JSON input and returns calculation results along with a reproducible code snippet.
    
    ### Using the Python Package
    
    The core functionality is available as importable Python modules:
    
    ```python
    from core.power import sample_size_difference_in_means
    
    result = sample_size_difference_in_means(
        delta=0.5,
        std_dev=1.0,
        power=0.8,
        alpha=0.05
    )
    ```
    
    ### Command Line Interface
    
    The package includes a CLI tool for running calculations from the terminal:
    
    ```bash
    python cli.py sample-size --delta 0.5 --power 0.8 --alpha 0.05
    ```
    
    ### High-Performance Computing with Julia
    
    For computationally intensive simulations, the package provides Julia implementations that can be called from Python:
    
    ```python
    from julia import Main
    from julia.api import Julia
    
    jl = Julia(compiled_modules=False)
    Main.include("julia_backend/stepped_wedge.jl")
    
    result = Main.simulate_stepped_wedge(
        12,  # clusters
        4,   # steps
        10,  # individuals per cluster
        0.05,  # ICC
        0.5,   # treatment effect
        1.0,   # standard deviation
        1000,  # number of simulations
        0.05   # alpha
    )
    ```
    """)

if __name__ == "__main__":
    # Run the Streamlit app
    pass
