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
from scipy import optimize

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
    simulate_sample_size,
    # Continuous non-inferiority functions
    sample_size_continuous_non_inferiority,
    power_continuous_non_inferiority,
    min_detectable_non_inferiority_margin,
    simulate_continuous_non_inferiority,
    # Binary analytical functions
    sample_size_binary_non_inferiority,
    power_binary_non_inferiority,
    min_detectable_binary_non_inferiority_margin,
    # Binary simulation functions
    sample_size_binary_sim,
    min_detectable_effect_binary_sim,
    simulate_binary,
    simulate_binary_non_inferiority,
    sample_size_binary_non_inferiority_sim,
    min_detectable_binary_non_inferiority_margin_sim
)

# Import binary outcome functions from analytical module
from core.designs.parallel.analytical import (
    sample_size_binary,
    power_binary
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
    # Check if this is a non-inferiority test based on parameters
    is_non_inferiority = result['parameters'].get('hypothesis_type') == 'non-inferiority' if 'parameters' in result and 'hypothesis_type' in result['parameters'] else False
    
    if calculation_type == "Sample Size":
        if "Continuous Outcome" in design_type:
            n1 = result["n1"]
            n2 = result["n2"]
            total = result["total_n"]
            
            if is_non_inferiority:
                nim = result['parameters'].get('non_inferiority_margin', 0)
                direction = result['parameters'].get('direction', 'lower')
                assumed_diff = result['parameters'].get('assumed_difference', 0)
                direction_text = "not worse than" if direction == "lower" else "not better than"
                
                return f"For a non-inferiority test with margin of {nim:.2f} units (testing that the new treatment is {direction_text} standard by more than this margin), assuming a true difference of {assumed_diff:.2f}, with {result['parameters'].get('power', 0)*100:.0f}% power and a {result['parameters'].get('alpha', 0)*100:.0f}% significance level, you need **{n1}** participants in group 1 and **{n2}** participants in group 2, for a total of **{total}** participants."
            else:  # Superiority
                return f"For detecting an effect size of {result['parameters'].get('delta', 0):.2f} units with {result['parameters'].get('power', 0)*100:.0f}% power and a {result['parameters'].get('alpha', 0)*100:.0f}% significance level, you need **{n1}** participants in group 1 and **{n2}** participants in group 2, for a total of **{total}** participants."
        
        elif "Binary Outcome" in design_type:
            n1 = result["n1"]
            n2 = result["n2"]
            total = result["total_n"]
            p1 = result['parameters'].get('p1', 0)
            
            if is_non_inferiority:
                nim = result['parameters'].get('non_inferiority_margin', 0)
                direction = result['parameters'].get('direction', 'lower')
                assumed_diff = result['parameters'].get('assumed_difference', 0)
                direction_text = "not worse than" if direction == "lower" else "not better than"
                
                return f"For a non-inferiority test with margin of {nim:.2f} (testing that the new treatment proportion is {direction_text} standard by more than this margin), assuming a true difference of {assumed_diff:.2f}, with {result['parameters'].get('power', 0)*100:.0f}% power and a {result['parameters'].get('alpha', 0)*100:.0f}% significance level, you need **{n1}** participants in group 1 and **{n2}** participants in group 2, for a total of **{total}** participants."
            else:  # Superiority
                p2 = result['parameters'].get('p2', 0)
                return f"For detecting a difference between proportions of {p1:.2f} and {p2:.2f} with {result['parameters'].get('power', 0)*100:.0f}% power and a {result['parameters'].get('alpha', 0)*100:.0f}% significance level, you need **{n1}** participants in group 1 and **{n2}** participants in group 2, for a total of **{total}** participants."
    
    elif calculation_type == "Power":
        if "Continuous Outcome" in design_type:
            power = result["power"]
            n1 = result['parameters'].get('n1', 0)
            n2 = result['parameters'].get('n2', 0)
            
            if is_non_inferiority:
                nim = result['parameters'].get('non_inferiority_margin', 0)
                direction = result['parameters'].get('direction', 'lower')
                assumed_diff = result['parameters'].get('assumed_difference', 0)
                direction_text = "not worse than" if direction == "lower" else "not better than"
                
                return f"With {n1} participants in group 1 and {n2} participants in group 2, you have **{power*100:.1f}%** power for a non-inferiority test with margin of {nim:.2f} units (testing that the new treatment is {direction_text} standard by more than this margin), assuming a true difference of {assumed_diff:.2f}, at a {result['parameters'].get('alpha', 0)*100:.0f}% significance level."
            else:  # Superiority
                delta = result['parameters'].get('delta', 0)
                return f"With {n1} participants in group 1 and {n2} participants in group 2, you have **{power*100:.1f}%** power to detect an effect size of {delta:.2f} units at a {result['parameters'].get('alpha', 0)*100:.0f}% significance level."
        
        elif "Binary Outcome" in design_type:
            power = result["power"]
            n1 = result['parameters'].get('n1', 0)
            n2 = result['parameters'].get('n2', 0)
            p1 = result['parameters'].get('p1', 0)
            
            if is_non_inferiority:
                nim = result['parameters'].get('non_inferiority_margin', 0)
                direction = result['parameters'].get('direction', 'lower')
                assumed_diff = result['parameters'].get('assumed_difference', 0)
                direction_text = "not worse than" if direction == "lower" else "not better than"
                
                return f"With {n1} participants in group 1 and {n2} participants in group 2, you have **{power*100:.1f}%** power for a non-inferiority test with margin of {nim:.2f} (testing that the new treatment proportion is {direction_text} standard by more than this margin), assuming a true difference of {assumed_diff:.2f}, at a {result['parameters'].get('alpha', 0)*100:.0f}% significance level."
            else:  # Superiority
                p2 = result['parameters'].get('p2', 0)
                return f"With {n1} participants in group 1 and {n2} participants in group 2, you have **{power*100:.1f}%** power to detect a difference between proportions of {p1:.2f} and {p2:.2f} at a {result['parameters'].get('alpha', 0)*100:.0f}% significance level."
    
    elif calculation_type == "Minimum Detectable Effect (MDE)":
        if "Continuous Outcome" in design_type:
            n1 = result['parameters'].get('n1', 0)
            n2 = result['parameters'].get('n2', 0)
            
            if is_non_inferiority:
                margin = result["margin"]
                direction = result['parameters'].get('direction', 'lower')
                assumed_diff = result['parameters'].get('assumed_difference', 0)
                direction_text = "not worse than" if direction == "lower" else "not better than"
                
                if "Continuous Outcome" in design_type:
                    return f"With {n1} participants in group 1 and {n2} participants in group 2, the minimum non-inferiority margin you can detect with {result['parameters'].get('power', 0)*100:.0f}% power at a {result['parameters'].get('alpha', 0)*100:.0f}% significance level is **{margin:.2f}** units (testing that the new treatment is {direction_text} standard by more than this margin), assuming a true difference of {assumed_diff:.2f}."
                else:  # Binary Outcome
                    p1 = result['parameters'].get('p1', 0)
                    return f"With {n1} participants in group 1 and {n2} participants in group 2, the minimum non-inferiority margin you can detect with {result['parameters'].get('power', 0)*100:.0f}% power at a {result['parameters'].get('alpha', 0)*100:.0f}% significance level is **{margin:.2f}** (testing that the new treatment proportion is {direction_text} standard proportion of {p1:.2f} by more than this margin), assuming a true difference of {assumed_diff:.2f}."
            else:  # Superiority
                delta = result["delta"]
                return f"With {n1} participants in group 1 and {n2} participants in group 2, the smallest effect size you can detect with {result['parameters'].get('power', 0)*100:.0f}% power at a {result['parameters'].get('alpha', 0)*100:.0f}% significance level is **{delta:.2f}** units."
    
    # Default case
    return "Sample size calculation complete. See the parameters and results above."

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
    
    # Continuous outcome methods
    if "Continuous Outcome" in design_type:
        # Handle non-inferiority tests
        if is_non_inferiority:
            method_text = "Non-inferiority test sample size calculation for comparison of means."
            ref = "Blackwelder WC. 'Proving the null hypothesis' in clinical trials. Controlled Clinical Trials. 1982;3(4):345-353."
            additional_ref = "Julious SA. Sample sizes for clinical trials with normal data. Statistics in Medicine. 2004;23(12):1921-1986."
            
            # Add direction-specific information
            if "parameters" in result and "direction" in result["parameters"]:
                direction = result["parameters"]["direction"]
                if direction == "lower":
                    method_text += " Testing that new treatment is not worse than standard by more than the margin."
                else:  # upper
                    method_text += " Testing that new treatment is not better than standard by more than the margin (non-superiority)."
            
            # Add information about one-sided test
            method_text += " Uses one-sided alpha level as is standard for non-inferiority tests."
        
        # Parallel RCT methods for superiority
        elif "Parallel" in design_type:
            # Check for repeated measures
            repeated_measures = False
            if "parameters" in result and "correlation" in result["parameters"]:
                repeated_measures = True
            
            if repeated_measures:
                if "method" in result["parameters"] and result["parameters"]["method"] == "ancova":
                    method_text = "ANCOVA analysis for repeated measures design."
                    ref = "Frison L, Pocock SJ. Repeated measures in clinical trials: analysis using mean summary statistics and its implications for design. Statistics in Medicine. 1992;11(13):1685-1704."
                else:  # Default to change score
                    method_text = "Change score analysis for repeated measures design."
                    ref = "Vickers AJ, Altman DG. Statistics Notes: Analysing controlled trials with baseline and follow up measurements. BMJ. 2001;323(7321):1123-1124."
            else:
                # Standard parallel design
                if "std_dev2" in result["parameters"] and result["parameters"]["std_dev2"] is not None:
                    method_text = "Sample size calculation for unequal variances (Welch's t-test)."
                    ref = "Chow S, Shao J, Wang H. Sample Size Calculations in Clinical Research. CRC Press; 2008."
                else:
                    method_text = "Standard sample size calculation for comparison of means in parallel groups."
                    ref = "Lachin JM. Introduction to sample size determination and power analysis for clinical trials. Controlled Clinical Trials. 1981;2(2):93-113."
        
        # Cluster RCT methods
        elif "Cluster" in design_type:
            method_text = "Cluster randomized trial design accounting for intraclass correlation."
            ref = "Donner A, Klar N. Design and Analysis of Cluster Randomization Trials in Health Research. Arnold; 2000."
    
    # Binary outcome methods
    elif "Binary Outcome" in design_type:
        if "Cluster" in design_type:
            method_text = "Sample size calculation for binary outcomes in cluster randomized trials."
            ref = "Hayes RJ, Moulton LH. Cluster Randomised Trials. CRC Press; 2009."
        else:
            method_text = "Sample size calculation for comparison of proportions in parallel groups."
            ref = "Fleiss JL, Levin B, Paik MC. Statistical Methods for Rates and Proportions. Wiley; 2003."
    
    # Stepped wedge design
    elif "Stepped Wedge" in design_type:
        method_text = "Simulation-based power calculation for stepped wedge cluster randomized design."
        ref = "Hussey MA, Hughes JP. Design and analysis of stepped wedge cluster randomized trials. Contemporary Clinical Trials. 2007;28(2):182-191."
    
    # Simulation-specific information
    if simulation_used:
        simulation_info = f"Results estimated using Monte Carlo simulation with {result['parameters']['nsim']} iterations."
    else:
        simulation_info = "Results calculated using analytical formulas."
    
    # For non-inferiority, add the additional reference
    if is_non_inferiority:
        return f"**Method:** {method_text}\n\n{simulation_info}\n\n**References:**\n1. {ref}\n2. {additional_ref}"
    else:
        return f"**Method:** {method_text}\n\n{simulation_info}\n\n**Reference:** {ref}"

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
            # Hypothesis type selection for Parallel RCT designs
            if "Parallel RCT" in design_type and ("Continuous Outcome" in design_type or "Binary Outcome" in design_type):
                hypothesis_type = st.radio(
                    "Hypothesis Type",
                    ["Superiority", "Non-inferiority"], 
                    key="hypothesis_type_radio",
                    horizontal=True,
                    help="Superiority tests if one treatment is better than another. Non-inferiority tests if a new treatment is not worse than the standard by more than an acceptable margin."
                )
                
            if "Continuous Outcome" in design_type:
                if calculation_type == "Sample Size":
                    # Show different parameters based on hypothesis type
                    if "hypothesis_type_radio" in st.session_state and st.session_state["hypothesis_type_radio"] == "Non-inferiority":
                        # Non-inferiority parameters for continuous outcomes
                        std_dev = st.number_input("Standard Deviation", value=1.0, step=0.1, min_value=0.1)
                        
                        # Non-inferiority specific parameters
                        default_margin = 0.5
                        step_size = 0.1
                        
                        nim = st.number_input(
                            "Non-inferiority Margin", 
                            value=default_margin, 
                            step=step_size, 
                            min_value=0.01, 
                            key="nim_input",
                            help="The maximum acceptable difference for still considering the new treatment non-inferior to the standard."
                        )
                        
                        direction = st.radio(
                            "Non-inferiority Direction",
                            ["Lower", "Upper"], 
                            index=0, 
                            key="direction_radio",
                            help="Lower: Test if new treatment is not worse than standard by more than margin. Upper: Test if new treatment is not better than standard by more than margin (rare)."
                        )
                        
                        assumed_difference = st.number_input(
                            "Assumed True Difference", 
                            value=0.0, 
                            step=0.1, 
                            key="assumed_diff_input",
                            help="The true difference you expect between means. Typically 0 for non-inferiority (treatments are actually equivalent)."
                        )
                        
                        st.info("Non-inferiority tests use a one-sided alpha level (significance level).")
                    else:
                        # Superiority parameters
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
                    alpha = st.slider("Significance Level (\u03b1)", min_value=0.01, max_value=0.1, value=0.05, step=0.01)
                    
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
                    
                    # Show different parameters based on hypothesis type
                    if "hypothesis_type_radio" in st.session_state and st.session_state["hypothesis_type_radio"] == "Non-inferiority":
                        # Non-inferiority parameters for continuous outcomes
                        std_dev = st.number_input("Standard Deviation", value=1.0, step=0.1, min_value=0.1)
                        
                        # Non-inferiority specific parameters
                        default_margin = 0.5
                        step_size = 0.1
                        
                        nim = st.number_input(
                            "Non-inferiority Margin", 
                            value=default_margin, 
                            step=step_size, 
                            min_value=0.01, 
                            key="nim_input",
                            help="The maximum acceptable difference for still considering the new treatment non-inferior to the standard."
                        )
                        
                        direction = st.radio(
                            "Non-inferiority Direction",
                            ["Lower", "Upper"], 
                            index=0, 
                            key="direction_radio",
                            help="Lower: Test if new treatment is not worse than standard by more than margin. Upper: Test if new treatment is not better than standard by more than margin (rare)."
                        )
                        
                        assumed_difference = st.number_input(
                            "Assumed True Difference", 
                            value=0.0, 
                            step=0.1, 
                            key="assumed_diff_input",
                            help="The true difference you expect between means. Typically 0 for non-inferiority (treatments are actually equivalent)."
                        )
                        
                        st.info("Non-inferiority tests use a one-sided alpha level (significance level).")
                    else:
                        # Superiority parameters - Option to input effect size directly or as means
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
                    
                    alpha = st.slider("Significance Level (\u03b1)", min_value=0.01, max_value=0.1, value=0.05, step=0.01)
                
                elif calculation_type == "Minimum Detectable Effect (MDE)":
                    if "Parallel" in design_type:
                        n1 = st.number_input("Sample Size (Group 1)", value=50, step=1, min_value=2)
                        n2 = st.number_input("Sample Size (Group 2)", value=50, step=1, min_value=2)
                    else:  # Cluster
                        n_clusters = st.number_input("Number of Clusters per Arm", value=10, step=1, min_value=2)
                        cluster_size = st.number_input("Cluster Size", value=20, step=1, min_value=2)
                    
                    # Show different parameters based on hypothesis type
                    if "hypothesis_type_radio" in st.session_state and st.session_state["hypothesis_type_radio"] == "Non-inferiority":
                        # Non-inferiority parameters for MDE (which is non-inferiority margin)
                        std_dev = st.number_input("Standard Deviation", value=1.0, step=0.1, min_value=0.1)
                        
                        direction = st.radio(
                            "Non-inferiority Direction",
                            ["Lower", "Upper"], 
                            index=0, 
                            key="direction_radio",
                            help="Lower: Test if new treatment is not worse than standard by more than margin. Upper: Test if new treatment is not better than standard by more than margin (rare)."
                        )
                        
                        assumed_difference = st.number_input(
                            "Assumed True Difference", 
                            value=0.0, 
                            step=0.1, 
                            key="assumed_diff_input",
                            help="The true difference you expect between means. Typically 0 for non-inferiority (treatments are actually equivalent)."
                        )
                        
                        st.info("Non-inferiority tests use a one-sided alpha level (significance level).")
                    else:
                        # For superiority MDE
                        std_dev = st.number_input("Standard Deviation", value=1.0, step=0.1, min_value=0.1)
                    
                    power = st.slider("Power", min_value=0.5, max_value=0.99, value=0.8, step=0.01)
                    alpha = st.slider("Significance Level (\u03b1)", min_value=0.01, max_value=0.1, value=0.05, step=0.01)
            
            elif "Binary Outcome" in design_type:
                if calculation_type == "Sample Size":
                    # Always show control group proportion
                    p1 = st.slider("Control Group Proportion", min_value=0.01, max_value=0.99, value=0.5, step=0.01)
                    
                    # Show different parameters based on hypothesis type
                    if "hypothesis_type_radio" in st.session_state and st.session_state["hypothesis_type_radio"] == "Non-inferiority":
                        # Non-inferiority parameters
                        default_margin = 0.1
                        step_size = 0.01
                        
                        nim = st.number_input(
                            "Non-inferiority Margin", 
                            value=default_margin, 
                            step=step_size, 
                            min_value=0.01, 
                            key="nim_input",
                            help="The maximum acceptable difference for still considering the new treatment non-inferior to the standard."
                        )
                        
                        direction = st.radio(
                            "Non-inferiority Direction",
                            ["Lower", "Upper"], 
                            index=0, 
                            key="direction_radio",
                            help="Lower: Test if new treatment is not worse than standard by more than margin. Upper: Test if new treatment is not better than standard by more than margin (rare)."
                        )
                        
                        assumed_difference = st.number_input(
                            "Assumed True Difference", 
                            value=0.0, 
                            step=0.01, 
                            key="assumed_diff_input",
                            help="The true difference you expect between proportions. Typically 0 for non-inferiority (treatments are actually equivalent)."
                        )
                        
                        st.info("Non-inferiority tests use a one-sided alpha level (significance level).")
                    else:
                        # Superiority parameters
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
                    
                    # Always show control group proportion
                    p1 = st.slider("Control Group Proportion", min_value=0.01, max_value=0.99, value=0.5, step=0.01)
                    
                    # Show different parameters based on hypothesis type
                    if "hypothesis_type_radio" in st.session_state and st.session_state["hypothesis_type_radio"] == "Non-inferiority":
                        # Non-inferiority parameters
                        default_margin = 0.1
                        step_size = 0.01
                        
                        nim = st.number_input(
                            "Non-inferiority Margin", 
                            value=default_margin, 
                            step=step_size, 
                            min_value=0.01, 
                            key="nim_input",
                            help="The maximum acceptable difference for still considering the new treatment non-inferior to the standard."
                        )
                        
                        direction = st.radio(
                            "Non-inferiority Direction",
                            ["Lower", "Upper"], 
                            index=0, 
                            key="direction_radio",
                            help="Lower: Test if new treatment is not worse than standard by more than margin. Upper: Test if new treatment is not better than standard by more than margin (rare)."
                        )
                        
                        assumed_difference = st.number_input(
                            "Assumed True Difference", 
                            value=0.0, 
                            step=0.01, 
                            key="assumed_diff_input",
                            help="The true difference you expect between proportions. Typically 0 for non-inferiority (treatments are actually equivalent)."
                        )
                        
                        st.info("Non-inferiority tests use a one-sided alpha level (significance level).")
                    else:
                        # Superiority parameters
                        p2 = st.slider("Intervention Group Proportion", min_value=0.01, max_value=0.99, value=0.6, step=0.01)
                    
                    alpha = st.slider("Significance Level (α)", min_value=0.01, max_value=0.1, value=0.05, step=0.01)
                    
                elif calculation_type == "Minimum Detectable Effect (MDE)":
                    if "Parallel" in design_type:
                        n1 = st.number_input("Sample Size (Group 1)", value=100, step=1, min_value=2)
                        n2 = st.number_input("Sample Size (Group 2)", value=100, step=1, min_value=2)
                    else:  # Cluster
                        n_clusters = st.number_input("Number of Clusters per Arm", value=10, step=1, min_value=2)
                        cluster_size = st.number_input("Cluster Size", value=20, step=1, min_value=2)
                    
                    # Always show control group proportion
                    p1 = st.slider("Control Group Proportion", min_value=0.01, max_value=0.99, value=0.5, step=0.01)
                    
                    # Show different parameters based on hypothesis type
                    if "hypothesis_type_radio" in st.session_state and st.session_state["hypothesis_type_radio"] == "Non-inferiority":
                        # Non-inferiority parameters
                        direction = st.radio(
                            "Non-inferiority Direction",
                            ["Lower", "Upper"], 
                            index=0, 
                            key="direction_radio",
                            help="Lower: Test if new treatment is not worse than standard by more than margin. Upper: Test if new treatment is not better than standard by more than margin (rare)."
                        )
                        
                        assumed_difference = st.number_input(
                            "Assumed True Difference", 
                            value=0.0, 
                            step=0.01, 
                            key="assumed_diff_input",
                            help="The true difference you expect between proportions. Typically 0 for non-inferiority (treatments are actually equivalent)."
                        )
                        
                        st.info("Non-inferiority tests use a one-sided alpha level (significance level).")
                    
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
            # Advanced options for Parallel RCT with either Continuous or Binary Outcome
            # Advanced options section - different handling for continuous vs binary outcomes
            if "Parallel RCT (Continuous Outcome)" == design_type:
                # Show title and separator for continuous outcomes
                st.write("Advanced Analysis Options:")
                st.markdown("---")
                
                # For continuous outcomes, show variance and repeated measures options
                col1, col2 = st.columns(2)
                
                with col1:
                    unequal_var = st.checkbox("Unequal Variances", value=False, key="unequal_var_checkbox")
                    
                with col2:
                    repeated_measures = st.checkbox("Repeated Measures (Baseline + Follow-up)", value=False, key="repeated_measures_checkbox")
                
                # Show a horizontal separator if options are selected
                if unequal_var or repeated_measures:
                    st.markdown("---")
                
                # Unequal variances option
                if unequal_var and not repeated_measures:
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
                
                # Standard option - only display message if no additional parameters are selected
                if not unequal_var and not repeated_measures:
                    st.info("Standard analysis assumes equal variances across groups.")
                
            elif "Parallel RCT (Binary Outcome)" == design_type:
                # For binary outcomes, skip the header and separator
                # Set default values that will be used in calculations
                unequal_var = False
                repeated_measures = False
                
                # Binary outcome specific options - immediately show test type options
                test_type = st.radio(
                    "Statistical Test",
                    ["Normal Approximation", "Likelihood Ratio Test", "Exact Test"],
                    index=0,  # Default to Normal Approximation
                    key="binary_test_type",
                    horizontal=True
                )
                
                # Explain the selected test type
                if test_type == "Normal Approximation":
                    st.info("Normal approximation uses the z-test to compare proportions. Fast and reliable for moderate to large sample sizes.")
                elif test_type == "Likelihood Ratio Test":
                    st.info("Likelihood Ratio Test often has better statistical properties than the Normal approximation, especially with smaller sample sizes.")
                else:  # Exact Test
                    st.info("Fisher's Exact Test provides the most accurate results for small sample sizes, but is computationally intensive for large samples.")
                
                # This section has been moved up to the continuous outcome section
            
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
                    
                    # Parameters for sample size simulations
                    if "Binary Outcome" in design_type:
                        min_n = st.number_input("Minimum Sample Size to Try", value=10, min_value=2, step=1, key="min_n_slider")
                        max_n = st.number_input("Maximum Sample Size to Try", value=1000, min_value=10, step=10, key="max_n_slider")
                        step = st.number_input("Sample Size Step", value=10, min_value=1, step=1, key="step_slider")
                    else:  # Continuous Outcome
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
                        
                        # Get hypothesis type from session state
                        hypothesis_type = st.session_state.get("hypothesis_type_radio", "Superiority")
                        
                        if hypothesis_type == "Non-inferiority":
                            # Get non-inferiority parameters from session state
                            nim = st.session_state.get("nim_input", 0.5)
                            direction = st.session_state.get("direction_radio", "Lower").lower()
                            assumed_difference = st.session_state.get("assumed_diff_input", 0.0)
                            
                            if use_simulation:
                                # Get simulation parameters from session state
                                nsim_value = st.session_state.get("nsim_slider", 1000)
                                max_n_value = st.session_state.get("max_n_slider", 200)
                                step_size_value = st.session_state.get("step_size_slider", 10)
                                
                                # Call the appropriate non-inferiority simulation function
                                result = simulate_sample_size_non_inferiority(
                                    non_inferiority_margin=nim,
                                    std_dev=std_dev,
                                    power=power,
                                    alpha=alpha,
                                    allocation_ratio=allocation_ratio,
                                    nsim=nsim_value,
                                    min_n=10,
                                    max_n=max_n_value,
                                    step=step_size_value,
                                    assumed_difference=assumed_difference,
                                    direction=direction,
                                    repeated_measures=repeated_measures,
                                    correlation=correlation if repeated_measures else 0.5,
                                    method=analysis_method if repeated_measures else "change_score"
                                )
                                method_name = "simulate_sample_size_non_inferiority"
                            else:
                                # Use analytical non-inferiority function
                                result = sample_size_continuous_non_inferiority(
                                    non_inferiority_margin=nim,
                                    std_dev=std_dev,
                                    power=power,
                                    alpha=alpha,
                                    allocation_ratio=allocation_ratio,
                                    assumed_difference=assumed_difference,
                                    direction=direction
                                )
                                method_name = "sample_size_continuous_non_inferiority"
                        else:  # Superiority hypothesis
                            if use_simulation:
                                # Get number of simulations from session state
                                nsim_value = st.session_state.get("nsim_slider", 1000)
                                max_n_value = st.session_state.get("max_n_slider", 200)
                                step_size_value = st.session_state.get("step_size_slider", 10)
                                
                                # Call the appropriate simulation function
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
                        # Get hypothesis type from session state
                        hypothesis_type = st.session_state.get("hypothesis_type_radio", "Superiority")
                        
                        if hypothesis_type == "Non-inferiority":
                            # Get non-inferiority parameters from session state
                            nim = st.session_state.get("nim_input", 0.5)
                            direction = st.session_state.get("direction_radio", "Lower").lower()
                            assumed_difference = st.session_state.get("assumed_diff_input", 0.0)
                            
                            if use_simulation:
                                # Get simulation parameters
                                nsim_value = st.session_state.get("nsim_slider", 1000)
                                
                                # Call non-inferiority simulation for power
                                result = simulate_continuous_non_inferiority(
                                    n1=n1,
                                    n2=n2,
                                    non_inferiority_margin=nim,
                                    std_dev=std_dev,
                                    nsim=nsim_value,
                                    alpha=alpha,
                                    assumed_difference=assumed_difference,
                                    direction=direction,
                                    repeated_measures=repeated_measures,
                                    correlation=correlation if repeated_measures else 0.5,
                                    method=analysis_method if repeated_measures else "change_score"
                                )
                                method_name = "simulate_continuous_non_inferiority"
                            else:
                                # Use analytical non-inferiority function for power
                                result = power_continuous_non_inferiority(
                                    n1=n1,
                                    n2=n2,
                                    non_inferiority_margin=nim,
                                    std_dev=std_dev,
                                    alpha=alpha,
                                    assumed_difference=assumed_difference,
                                    direction=direction
                                )
                                method_name = "power_continuous_non_inferiority"
                        else:  # Superiority hypothesis
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
                        # Get hypothesis type from session state
                        hypothesis_type = st.session_state.get("hypothesis_type_radio", "Superiority")
                        # Get simulation checkbox state from session state
                        use_simulation = st.session_state.get("use_simulation_checkbox", False)
                        
                        if hypothesis_type == "Non-inferiority":
                            # In non-inferiority context, MDE is the minimum non-inferiority margin
                            # Get non-inferiority direction from session state
                            direction = st.session_state.get("direction_radio", "Lower").lower()
                            assumed_difference = st.session_state.get("assumed_diff_input", 0.0)
                            
                            # For non-inferiority, we use analytical calculation as simulation is complex
                            result = min_detectable_non_inferiority_margin(
                                n1=n1,
                                n2=n2,
                                std_dev=std_dev,
                                power=power,
                                alpha=alpha,
                                assumed_difference=assumed_difference,
                                direction=direction
                            )
                            method_name = "min_detectable_non_inferiority_margin"
                        else:  # Superiority hypothesis
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
                                else:  # Standard parallel design
                                    # For MDE, we use the power function but solve for delta
                                    # We know given n1, n2, power, and alpha, what delta can we detect?
                                    # Calculate the critical values for hypothesis test
                                    z_alpha = stats.norm.ppf(1 - alpha/2)
                                    z_beta = stats.norm.ppf(power)
                                    
                                    # Calculate MDE
                                    if std_dev2 is not None:  # Unequal variances
                                        delta = (z_alpha + z_beta) * np.sqrt((std_dev**2 / n1) + (std_dev2**2 / n2))
                                    else:  # Equal variances
                                        delta = (z_alpha + z_beta) * std_dev * np.sqrt((1/n1) + (1/n2))
                                    
                                    result = {
                                        'delta': delta,
                                        'parameters': {
                                            'n1': n1,
                                            'n2': n2,
                                            'std_dev': std_dev,
                                            'std_dev2': std_dev2,
                                            'power': power,
                                            'alpha': alpha
                                        }
                                    }
                                method_name = "min_detectable_effect_analytical"
                
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
                
                # Parallel RCT with Binary Outcome
                elif design_type == "Parallel RCT (Binary Outcome)":
                    # Get hypothesis type from session state
                    hypothesis_type = st.session_state.get("hypothesis_type_radio", "Superiority")
                    
                    # Get simulation checkbox state from session state
                    use_simulation = st.session_state.get("use_simulation_checkbox", False)
                    
                    if hypothesis_type == "Non-inferiority":
                        # Get non-inferiority parameters from session state
                        nim = st.session_state.get("nim_input", 0.1)
                        direction = st.session_state.get("direction_radio", "Lower").lower()
                        assumed_difference = st.session_state.get("assumed_diff_input", 0.0)
                        
                        if calculation_type == "Sample Size":
                            if use_simulation:
                                # Get simulation parameters from session state
                                nsim_value = st.session_state.get("nsim_slider", 1000)
                                min_n_value = st.session_state.get("min_n_slider", 10)
                                max_n_value = st.session_state.get("max_n_slider", 1000)
                                step_value = st.session_state.get("step_slider", 10)
                                
                                result = sample_size_binary_non_inferiority_sim(
                                    p1=p1,
                                    non_inferiority_margin=nim,
                                    power=power,
                                    alpha=alpha,
                                    allocation_ratio=allocation_ratio,
                                    assumed_difference=assumed_difference,
                                    direction=direction,
                                    nsim=nsim_value,
                                    min_n=min_n_value,
                                    max_n=max_n_value,
                                    step=step_value
                                )
                                method_name = "sample_size_binary_non_inferiority_sim"
                            else:
                                result = sample_size_binary_non_inferiority(
                                    p1=p1,
                                    non_inferiority_margin=nim,
                                    power=power,
                                    alpha=alpha,
                                    allocation_ratio=allocation_ratio,
                                    assumed_difference=assumed_difference,
                                    direction=direction
                                )
                                method_name = "sample_size_binary_non_inferiority"
                        
                        elif calculation_type == "Power":
                            if use_simulation:
                                # Get number of simulations from session state
                                nsim_value = st.session_state.get("nsim_slider", 1000)
                                
                                result = simulate_binary_non_inferiority(
                                    n1=n1,
                                    n2=n2,
                                    p1=p1,
                                    non_inferiority_margin=nim,
                                    alpha=alpha,
                                    assumed_difference=assumed_difference,
                                    direction=direction,
                                    nsim=nsim_value
                                )
                                method_name = "simulate_binary_non_inferiority"
                            else:
                                result = power_binary_non_inferiority(
                                    n1=n1,
                                    n2=n2,
                                    p1=p1,
                                    non_inferiority_margin=nim,
                                    alpha=alpha,
                                    assumed_difference=assumed_difference,
                                    direction=direction
                                )
                                method_name = "power_binary_non_inferiority"
                        
                        elif calculation_type == "Minimum Detectable Effect (MDE)":
                            if use_simulation:
                                # Get number of simulations from session state
                                nsim_value = st.session_state.get("nsim_slider", 1000)
                                
                                result = min_detectable_binary_non_inferiority_margin_sim(
                                    n1=n1,
                                    n2=n2,
                                    p1=p1,
                                    power=power,
                                    alpha=alpha,
                                    assumed_difference=assumed_difference,
                                    direction=direction,
                                    nsim=nsim_value,
                                    precision=0.01
                                )
                                method_name = "min_detectable_binary_non_inferiority_margin_sim"
                            else:
                                result = min_detectable_binary_non_inferiority_margin(
                                    n1=n1,
                                    n2=n2,
                                    p1=p1,
                                    power=power,
                                    alpha=alpha,
                                    assumed_difference=assumed_difference,
                                    direction=direction
                                )
                                method_name = "min_detectable_binary_non_inferiority_margin"
                    
                    else:  # Superiority hypothesis
                        if calculation_type == "Sample Size":
                            if use_simulation:
                                # Get simulation parameters from session state
                                nsim_value = st.session_state.get("nsim_slider", 1000)
                                min_n_value = st.session_state.get("min_n_slider", 10)
                                max_n_value = st.session_state.get("max_n_slider", 1000)
                                step_value = st.session_state.get("step_slider", 10)
                                
                                # Get the selected test type from session state
                                test_type = st.session_state.get("binary_test_type", "Normal Approximation")
                                
                                result = sample_size_binary_sim(
                                    p1=p1,
                                    p2=p2,
                                    power=power,
                                    alpha=alpha,
                                    allocation_ratio=allocation_ratio,
                                    nsim=nsim_value,
                                    min_n=min_n_value,
                                    max_n=max_n_value,
                                    step=step_value,
                                    test_type=test_type
                                )
                                method_name = "sample_size_binary_sim"
                            else:
                                # Get the selected test type from session state
                                test_type = st.session_state.get("binary_test_type", "Normal Approximation")
                                
                                result = sample_size_binary(
                                    p1=p1,
                                    p2=p2,
                                    power=power,
                                    alpha=alpha,
                                    allocation_ratio=allocation_ratio,
                                    test_type=test_type
                                )
                                method_name = "sample_size_binary"
                        
                        elif calculation_type == "Power":
                            if use_simulation:
                                # Get number of simulations from session state
                                nsim_value = st.session_state.get("nsim_slider", 1000)
                                
                                # Get the selected test type from session state
                                test_type = st.session_state.get("binary_test_type", "Normal Approximation")
                                
                                result = simulate_binary(
                                    n1=n1,
                                    n2=n2,
                                    p1=p1,
                                    p2=p2,
                                    nsim=nsim_value,
                                    alpha=alpha,
                                    test_type=test_type
                                )
                                method_name = "simulate_binary"
                            else:
                                # Get the selected test type from session state
                                test_type = st.session_state.get("binary_test_type", "Normal Approximation")
                                
                                result = power_binary(
                                    n1=n1,
                                    n2=n2,
                                    p1=p1,
                                    p2=p2,
                                    alpha=alpha,
                                    test_type=test_type
                                )
                                method_name = "power_binary"
                        
                        # For MDE, we get the minimum detectable p2 for a given p1, sample size, and power
                        elif calculation_type == "Minimum Detectable Effect (MDE)":
                            if use_simulation:
                                # Get number of simulations from session state
                                nsim_value = st.session_state.get("nsim_slider", 1000)
                                
                                # Get the selected test type from session state
                                test_type = st.session_state.get("binary_test_type", "Normal Approximation")
                                
                                result = min_detectable_effect_binary_sim(
                                    n1=n1,
                                    n2=n2,
                                    p1=p1,
                                    power=power,
                                    nsim=nsim_value,
                                    alpha=alpha,
                                    precision=0.01,
                                    test_type=test_type
                                )
                                method_name = "min_detectable_effect_binary_sim"
                            else:
                                if p1 < 0.5:
                                    # If p1 is low, look for increase in p2
                                    min_p2 = p1
                                    max_p2 = 0.99
                                else:
                                    # If p1 is high, look for decrease in p2
                                    min_p2 = 0.01
                                    max_p2 = p1
                                
                                def objective(p2):
                                    # Using the power function to find p2 that gives us target power
                                    return power_binary(n1, n2, p1, p2, alpha)["power"] - power
                                
                                result_p2 = optimize.brentq(objective, min_p2, max_p2)
                                
                                result = {
                                    "p2": result_p2,
                                    "parameters": {
                                        "n1": n1,
                                        "n2": n2,
                                        "p1": p1,
                                        "power": power,
                                        "alpha": alpha
                                    }
                                }
                                method_name = "min_detectable_p2"
                
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
