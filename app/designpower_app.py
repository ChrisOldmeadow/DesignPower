"""
Simplified component-based demo for DesignPower architecture.

This is a minimal working example showing how components enable a more
maintainable and extensible application structure.
"""
import os
import sys
import streamlit as st
import math
import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt

# Make sure the project root is in the path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
# Import core functions
from core.designs.parallel import analytical
from core.designs.parallel import simulation
from core.designs.parallel import binary_simulation
from core.designs.parallel.binary_tests import power_binary_with_test
# Import local power_calculations module for single-arm calculations
from app.power_calculations import one_sample_t_test_sample_size, one_sample_proportion_test_sample_size

# Dictionary of available designs and their parameters
DESIGN_CONFIGS = {
    "parallel_rct": {
        "name": "Parallel RCT",
        "outcomes": ["Continuous Outcome", "Binary Outcome", "Survival Outcome"]
    },
    "single_arm": {
        "name": "Single Arm Trial",
        "outcomes": ["Continuous Outcome", "Binary Outcome", "Survival Outcome"] 
    }
}

def render_parallel_continuous(calc_type, hypothesis_type):
    """Simple component for parallel continuous design"""
    st.write(f"### Parallel RCT with Continuous Outcome ({calc_type})")
    
    # Basic parameters
    col1, col2 = st.columns(2)
    with col1:
        # Different inputs based on hypothesis type and calculation type
        if hypothesis_type == "Superiority":
            # Superiority hypothesis - use traditional difference in means
            if calc_type == "Sample Size":
                mean1 = st.number_input("Mean (Group 1)", value=0.0, step=0.1)
                mean2 = st.number_input("Mean (Group 2)", value=0.5, step=0.1)
                std = st.number_input("Standard Deviation", value=1.0, min_value=0.1, step=0.1)
                power = st.slider("Power", min_value=0.7, max_value=0.99, value=0.8, step=0.05)
            elif calc_type == "Power":
                mean1 = st.number_input("Mean (Group 1)", value=0.0, step=0.1)
                mean2 = st.number_input("Mean (Group 2)", value=0.5, step=0.1)
                std = st.number_input("Standard Deviation", value=1.0, min_value=0.1, step=0.1)
                n1 = st.number_input("Sample Size (Group 1)", value=100, min_value=2, step=1)
                n2 = st.number_input("Sample Size (Group 2)", value=100, min_value=2, step=1)
            elif calc_type == "Minimum Detectable Effect":
                mean1 = st.number_input("Mean (Group 1)", value=0.0, step=0.1)
                std = st.number_input("Standard Deviation", value=1.0, min_value=0.1, step=0.1)
                n1 = st.number_input("Sample Size (Group 1)", value=100, min_value=2, step=1)
                n2 = st.number_input("Sample Size (Group 2)", value=100, min_value=2, step=1)
                power = st.slider("Power", min_value=0.7, max_value=0.99, value=0.8, step=0.05)
        else:
            # Non-inferiority hypothesis - need NIM and assumed difference
            if calc_type == "Sample Size":
                mean1 = st.number_input("Mean (Control Group)", value=0.0, step=0.1)
                std = st.number_input("Standard Deviation", value=1.0, min_value=0.1, step=0.1)
                non_inferiority_margin = st.number_input("Non-Inferiority Margin", value=0.5, min_value=0.01, step=0.1,
                                                      help="The maximum acceptable difference between treatments (must be positive)")
                assumed_difference = st.number_input("Assumed True Difference", value=0.0, step=0.1,
                                                  help="Assumed true difference between treatments (0 = treatments are equivalent)")
                power = st.slider("Power", min_value=0.7, max_value=0.99, value=0.8, step=0.05)
            elif calc_type == "Power":
                mean1 = st.number_input("Mean (Control Group)", value=0.0, step=0.1)
                std = st.number_input("Standard Deviation", value=1.0, min_value=0.1, step=0.1)
                non_inferiority_margin = st.number_input("Non-Inferiority Margin", value=0.5, min_value=0.01, step=0.1)
                assumed_difference = st.number_input("Assumed True Difference", value=0.0, step=0.1)
                n1 = st.number_input("Sample Size (Group 1)", value=100, min_value=2, step=1)
                n2 = st.number_input("Sample Size (Group 2)", value=100, min_value=2, step=1)
            elif calc_type == "Minimum Detectable Effect":
                mean1 = st.number_input("Mean (Control Group)", value=0.0, step=0.1)
                std = st.number_input("Standard Deviation", value=1.0, min_value=0.1, step=0.1)
                assumed_difference = st.number_input("Assumed True Difference", value=0.0, step=0.1)
                n1 = st.number_input("Sample Size (Group 1)", value=100, min_value=2, step=1)
                n2 = st.number_input("Sample Size (Group 2)", value=100, min_value=2, step=1)
                power = st.slider("Power", min_value=0.7, max_value=0.99, value=0.8, step=0.05)
    
    with col2:
        alpha = st.slider("Alpha", min_value=0.01, max_value=0.1, value=0.05, step=0.01)
        
        # Additional inputs based on calculation type
        if calc_type == "Power" or calc_type == "Minimum Detectable Effect":
            st.write("")
            allocation_ratio_calculated = n2 / n1 if n1 > 0 else 1.0
            st.write(f"Allocation Ratio: {allocation_ratio_calculated:.2f}")
    
    # Advanced options
    with st.expander("Advanced Options"):
        # Method selection: Analytical vs Simulation
        method_type = st.radio("Calculation Method", ["Analytical", "Simulation"], horizontal=True)
        use_simulation = method_type == "Simulation"
        
        # Simulation parameters if simulation is selected
        if use_simulation:
            st.write("Simulation Parameters:")
            nsim = st.slider("Number of Simulations", min_value=100, max_value=10000, value=1000, step=100)
            
            if calc_type == "Sample Size":
                min_n = st.number_input("Minimum Sample Size to Try", value=10, min_value=5, step=5)
                max_n = st.number_input("Maximum Sample Size to Try", value=500, min_value=50, step=50)
                step_n = st.number_input("Sample Size Step", value=10, min_value=1, step=1)
            elif calc_type == "Minimum Detectable Effect":
                precision = st.slider("MDE Precision", min_value=0.001, max_value=0.1, value=0.01, step=0.001, format="%.3f")
        else:
            # Default values when not using simulation
            nsim = 1000
            min_n = 10
            max_n = 500
            step_n = 10
            precision = 0.01
            
        # Allocation ratio (for sample size calculation)
        if calc_type == "Sample Size":
            allocation_ratio = st.slider("Allocation Ratio (n2/n1)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
        else:
            allocation_ratio = n2 / n1 if n1 > 0 else 1.0
        
        # Repeated Measures Option
        repeated_measures = st.checkbox("Repeated Measures Design", value=False)
        
        if repeated_measures:
            correlation = st.slider("Correlation between measures", min_value=0.0, max_value=0.99, value=0.5, step=0.05)
            analysis_method = st.radio("Analysis Method", ["Change Score", "ANCOVA"], horizontal=True)
        else:
            correlation = 0
            analysis_method = "Change Score"
        
        # Unequal Variances Option (only for analytical method - simulation always uses equal variance)
        unequal_var = st.checkbox("Unequal Variances", value=False, disabled=use_simulation,
                              help="Unequal variances option is only available with analytical method")
        
        if unequal_var and not use_simulation:
            std2 = st.number_input("Standard Deviation (Group 2)", value=1.2, min_value=0.1, step=0.1)
        else:
            std2 = std
    
    # Convert analysis method to the format expected by the analytical function
    if analysis_method == "Change Score":
        analysis_method_param = "change_score"
    else:  # ANCOVA
        analysis_method_param = "ancova"
        
    # Build the parameter dictionary based on calculation type
    params = {
        "std": std, 
        "alpha": alpha,
        "allocation_ratio": allocation_ratio,
        "unequal_var": unequal_var,
        "std2": std2,
        "repeated_measures": repeated_measures,
        "correlation": correlation,
        "analysis_method": analysis_method_param,
        "use_simulation": use_simulation,
        "nsim": nsim,
        "hypothesis_type": hypothesis_type
    }
    
    # Add non-inferiority specific parameters if applicable
    if hypothesis_type == "Non-Inferiority":
        if calc_type in ["Sample Size", "Power"]:
            params.update({
                "non_inferiority_margin": non_inferiority_margin,
                "assumed_difference": assumed_difference
            })
    
    # Add simulation-specific parameters
    if use_simulation:
        if calc_type == "Sample Size":
            params.update({
                "min_n": min_n,
                "max_n": max_n,
                "step_n": step_n
            })
        elif calc_type == "Minimum Detectable Effect":
            params.update({
                "precision": precision
            })
    
    # Add calculation-specific parameters
    if calc_type == "Sample Size":
        if hypothesis_type == "Superiority":
            params.update({
                "mean1": mean1,
                "mean2": mean2,
                "power": power
            })
        else:  # Non-inferiority
            params.update({
                "mean1": mean1,
                "power": power,
                "non_inferiority_margin": non_inferiority_margin,
                "assumed_difference": assumed_difference
            })
    elif calc_type == "Power":
        if hypothesis_type == "Superiority":
            params.update({
                "mean1": mean1,
                "mean2": mean2,
                "n1": n1,
                "n2": n2
            })
        else:  # Non-inferiority
            params.update({
                "mean1": mean1,
                "n1": n1,
                "n2": n2,
                "non_inferiority_margin": non_inferiority_margin,
                "assumed_difference": assumed_difference
            })
    elif calc_type == "Minimum Detectable Effect":
        if hypothesis_type == "Superiority":
            params.update({
                "mean1": mean1,
                "n1": n1,
                "n2": n2,
                "power": power
            })
        else:  # Non-inferiority
            params.update({
                "mean1": mean1,
                "n1": n1,
                "n2": n2,
                "power": power,
                "assumed_difference": assumed_difference
            })
    
    return params

def render_single_arm_continuous(calc_type, hypothesis_type):
    """Simple component for single arm continuous design"""
    st.write(f"### Single Arm Trial with Continuous Outcome ({calc_type})")
    
    # Basic parameters
    col1, col2 = st.columns(2)
    with col1:
        # Different inputs based on calculation type
        if calc_type == "Sample Size":
            baseline = st.number_input("Baseline/Null Mean", value=0.0, step=0.1)
            target = st.number_input("Target/Alternative Mean", value=0.5, step=0.1)
            std = st.number_input("Standard Deviation", value=1.0, min_value=0.1, step=0.1)
            power = st.slider("Power", min_value=0.7, max_value=0.99, value=0.8, step=0.05)
        elif calc_type == "Power":
            baseline = st.number_input("Baseline/Null Mean", value=0.0, step=0.1)
            target = st.number_input("Target/Alternative Mean", value=0.5, step=0.1)
            std = st.number_input("Standard Deviation", value=1.0, min_value=0.1, step=0.1)
            n = st.number_input("Sample Size", value=50, min_value=2, step=1)
        elif calc_type == "Minimum Detectable Effect":
            baseline = st.number_input("Baseline/Null Mean", value=0.0, step=0.1)
            std = st.number_input("Standard Deviation", value=1.0, min_value=0.1, step=0.1)
            n = st.number_input("Sample Size", value=50, min_value=2, step=1)
            power = st.slider("Power", min_value=0.7, max_value=0.99, value=0.8, step=0.05)
    
    with col2:
        alpha = st.slider("Alpha", min_value=0.01, max_value=0.1, value=0.05, step=0.01)
    
    # Advanced options
    with st.expander("Advanced Options"):
        historical = st.checkbox("Use Historical Control Data", value=False)
        
        if historical:
            col3, col4 = st.columns(2)
            with col3:
                hist_n = st.number_input("Historical Sample Size", value=50, min_value=1, step=1)
            with col4:
                discount = st.slider("Historical Data Weight", min_value=0.0, max_value=1.0, value=0.5, step=0.1)
                st.info("Weight 1.0 = full weight, 0.0 = ignore historical data")
        else:
            hist_n = 0
            discount = 0
    
    # Build the parameter dictionary based on calculation type
    params = {
        "baseline": baseline,
        "std": std, 
        "alpha": alpha,
        "has_historical": historical,
        "historical_n": hist_n,
        "discount_factor": discount
    }
    
    # Add calculation-specific parameters
    if calc_type == "Sample Size":
        params.update({
            "target": target,
            "power": power
        })
    elif calc_type == "Power":
        params.update({
            "target": target,
            "n": n
        })
    elif calc_type == "Minimum Detectable Effect":
        params.update({
            "n": n,
            "power": power
        })
    
    return params

def render_parallel_binary(calc_type, hypothesis_type):
    """Simple component for parallel binary design"""
    st.write(f"### Parallel RCT with Binary Outcome ({calc_type})")
    
    # Basic parameters
    col1, col2 = st.columns(2)
    with col1:
        if hypothesis_type == "Superiority":
            # Superiority - standard comparison of two proportions
            if calc_type == "Sample Size":
                p1 = st.slider("Proportion (Group 1)", min_value=0.0, max_value=1.0, value=0.3, step=0.05)
                p2 = st.slider("Proportion (Group 2)", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
                power = st.slider("Power", min_value=0.7, max_value=0.99, value=0.8, step=0.05)
            elif calc_type == "Power":
                p1 = st.slider("Proportion (Group 1)", min_value=0.0, max_value=1.0, value=0.3, step=0.05)
                p2 = st.slider("Proportion (Group 2)", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
                n1 = st.number_input("Sample Size (Group 1)", value=100, min_value=2, step=1)
                n2 = st.number_input("Sample Size (Group 2)", value=100, min_value=2, step=1)
            elif calc_type == "Minimum Detectable Effect":
                p1 = st.slider("Proportion (Group 1)", min_value=0.0, max_value=1.0, value=0.3, step=0.05)
                n1 = st.number_input("Sample Size (Group 1)", value=100, min_value=2, step=1)
                n2 = st.number_input("Sample Size (Group 2)", value=100, min_value=2, step=1)
                power = st.slider("Power", min_value=0.7, max_value=0.99, value=0.8, step=0.05)
        else:
            # Non-inferiority - need control proportion, NIM, and assumed difference
            if calc_type == "Sample Size":
                p1 = st.slider("Control Group Proportion", min_value=0.0, max_value=1.0, value=0.3, step=0.05,
                               help="The proportion in the control/standard treatment group")
                
                # For Non-Inferiority, we need a margin and direction
                st.markdown("### Non-Inferiority Parameters")
                col1, col2 = st.columns(2)
                with col1:
                    non_inferiority_margin = st.number_input(
                        "Non-Inferiority Margin",
                        min_value=0.01, max_value=0.5, value=0.1, step=0.01,
                        help="The non-inferiority margin (positive value)"
                    )
                with col2:
                    assumed_difference = st.number_input(
                        "Assumed Difference (p2-p1)",
                        min_value=-0.5, max_value=0.5, value=0.0, step=0.01,
                        help="The assumed true difference between proportions"
                    )
                    
                # Direction selection for non-inferiority test
                direction = st.selectbox(
                    "Direction",
                    options=["lower", "upper"],
                    index=0,  # Default to lower
                    help="'lower': Test that new treatment is not worse than standard by more than margin. 'upper': Test that new treatment is not better than standard by more than margin."
                )
                power = st.slider("Power", min_value=0.7, max_value=0.99, value=0.8, step=0.05)
                
                # Calculate the implied treatment proportion for reference
                p2 = p1 + assumed_difference
                st.write(f"Implied treatment proportion: {p2:.2f}")
                
            elif calc_type == "Power":
                p1 = st.slider("Control Group Proportion", min_value=0.0, max_value=1.0, value=0.3, step=0.05,
                               help="The proportion in the control/standard treatment group")
                
                # For Non-Inferiority, we need a margin and direction
                st.markdown("### Non-Inferiority Parameters")
                col1, col2 = st.columns(2)
                with col1:
                    non_inferiority_margin = st.number_input(
                        "Non-Inferiority Margin",
                        min_value=0.01, max_value=0.5, value=0.1, step=0.01,
                        help="The non-inferiority margin (positive value)"
                    )
                with col2:
                    assumed_difference = st.number_input(
                        "Assumed Difference (p2-p1)",
                        min_value=-0.5, max_value=0.5, value=0.0, step=0.01,
                        help="The assumed true difference between proportions"
                    )
                    
                # Direction selection for non-inferiority test
                direction = st.selectbox(
                    "Direction",
                    options=["lower", "upper"],
                    index=0,  # Default to lower
                    help="'lower': Test that new treatment is not worse than standard by more than margin. 'upper': Test that new treatment is not better than standard by more than margin."
                )
                
                # Calculate the implied treatment proportion for reference
                p2 = p1 + assumed_difference
                st.write(f"Implied treatment proportion: {p2:.2f}")
                
                n1 = st.number_input("Sample Size (Group 1)", value=100, min_value=2, step=1)
                n2 = st.number_input("Sample Size (Group 2)", value=100, min_value=2, step=1)
                
            elif calc_type == "Minimum Detectable Effect":
                p1 = st.slider("Control Group Proportion", min_value=0.0, max_value=1.0, value=0.3, step=0.05,
                               help="The proportion in the control/standard treatment group")
                
                # For Non-Inferiority, we need a margin and direction
                st.markdown("### Non-Inferiority Parameters")
                col1, col2 = st.columns(2)
                with col1:
                    non_inferiority_margin = st.number_input(
                        "Non-Inferiority Margin",
                        min_value=0.01, max_value=0.5, value=0.1, step=0.01,
                        help="The non-inferiority margin (positive value)"
                    )
                with col2:
                    assumed_difference = st.number_input(
                        "Assumed Difference (p2-p1)",
                        min_value=-0.5, max_value=0.5, value=0.0, step=0.01,
                        help="The assumed true difference between proportions"
                    )
                
                # Direction selection for non-inferiority test
                direction = st.selectbox(
                    "Direction",
                    options=["lower", "upper"],
                    index=0,  # Default to lower
                    help="'lower': Test that new treatment is not worse than standard by more than margin. 'upper': Test that new treatment is not better than standard by more than margin."
                )
                
                n1 = st.number_input("Sample Size (Group 1)", value=100, min_value=2, step=1)
                n2 = st.number_input("Sample Size (Group 2)", value=100, min_value=2, step=1)
                power = st.slider("Power", min_value=0.7, max_value=0.99, value=0.8, step=0.05)
    
    with col2:
        alpha = st.slider("Alpha", min_value=0.01, max_value=0.1, value=0.05, step=0.01)
        
        # Additional inputs based on calculation type
        if calc_type == "Power" or calc_type == "Minimum Detectable Effect":
            allocation_ratio_calculated = n2 / n1 if n1 > 0 else 1.0
            st.write(f"Allocation Ratio: {allocation_ratio_calculated:.2f}")
    
    # Advanced options
    with st.expander("Advanced Options"):
        # Method selection: Analytical vs Simulation
        method_type = st.radio("Calculation Method", ["Analytical", "Simulation"], horizontal=True)
        use_simulation = method_type == "Simulation"
        
        # Test type selection - available for both analytical and simulation methods
        test_type = st.radio("Statistical Test", 
                          ["Normal Approximation", "Exact Test", "Likelihood Ratio Test"],
                          horizontal=True)
        
        # Simulation parameters if simulation is selected
        if use_simulation:
            st.write("Simulation Parameters:")
            nsim = st.slider("Number of Simulations", min_value=100, max_value=10000, value=1000, step=100)
            
            if calc_type == "Sample Size":
                min_n = st.number_input("Minimum Sample Size to Try", value=10, min_value=5, step=5)
                max_n = st.number_input("Maximum Sample Size to Try", value=500, min_value=50, step=50)
                step_n = st.number_input("Sample Size Step", value=10, min_value=1, step=1)
            elif calc_type == "Minimum Detectable Effect":
                precision = st.slider("MDE Precision", min_value=0.001, max_value=0.1, value=0.01, step=0.001, format="%.3f")
        else:
            # Default values when not using simulation
            nsim = 1000
            min_n = 10
            max_n = 500
            step_n = 10
            precision = 0.01
        
        # Allocation ratio (for sample size calculation)
        if calc_type == "Sample Size":
            allocation_ratio = st.slider("Allocation Ratio (n2/n1)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
        else:
            allocation_ratio = n2 / n1 if n1 > 0 else 1.0
    
    # Build the parameter dictionary based on calculation type
    params = {
        "calculation_type": calc_type,
        "hypothesis_type": hypothesis_type,
        "alpha": alpha,
        "test_type": test_type,
        "use_simulation": use_simulation,
        "nsim": nsim
    }
    
    # Add non-inferiority specific parameters if applicable
    if hypothesis_type == "Non-Inferiority":
        if calc_type in ["Sample Size", "Power"]:
            params.update({
                "non_inferiority_margin": non_inferiority_margin,
                "assumed_difference": assumed_difference
            })
            # For non-inferiority, the p2 value is calculated from p1 + assumed_difference
            p2 = p1 + assumed_difference
    
    # Add simulation-specific parameters
    if use_simulation:
        if calc_type == "Sample Size":
            params.update({
                "min_n": min_n,
                "max_n": max_n,
                "step_n": step_n
            })
        elif calc_type == "Minimum Detectable Effect":
            params.update({
                "precision": precision
            })
    
    # Add calculation-specific parameters
    if calc_type == "Sample Size":
        if hypothesis_type == "Superiority":
            params.update({
                "p1": p1,
                "p2": p2,
                "power": power,
                "allocation_ratio": allocation_ratio
            })
        else:  # Non-inferiority
            params.update({
                "p1": p1,
                "power": power,
                "allocation_ratio": allocation_ratio,
                "non_inferiority_margin": non_inferiority_margin,
                "assumed_difference": assumed_difference,
                "direction": direction
            })
    elif calc_type == "Power":
        if hypothesis_type == "Superiority":
            params.update({
                "p1": p1,
                "p2": p2,
                "n1": n1,
                "n2": n2
            })
        else:  # Non-inferiority
            params.update({
                "p1": p1,
                "n1": n1,
                "n2": n2,
                "non_inferiority_margin": non_inferiority_margin,
                "assumed_difference": assumed_difference,
                "direction": direction
            })
    elif calc_type == "Minimum Detectable Effect":
        if hypothesis_type == "Superiority":
            params.update({
                "p1": p1,
                "n1": n1,
                "n2": n2,
                "power": power
            })
        else:  # Non-inferiority
            params.update({
                "p1": p1,
                "n1": n1,
                "n2": n2,
                "power": power,
                "assumed_difference": assumed_difference,
                "direction": direction
            })
    
    return params

def render_single_arm_binary(calc_type, hypothesis_type):
    """Simple component for single arm binary design"""
    st.write(f"### Single Arm Trial with Binary Outcome ({calc_type})")
    
    # Basic parameters
    col1, col2 = st.columns(2)
    with col1:
        # Different inputs based on calculation type
        if calc_type == "Sample Size":
            p0 = st.slider("Null Hypothesis Proportion", min_value=0.0, max_value=1.0, value=0.3, step=0.05)
            p1 = st.slider("Alternative Hypothesis Proportion", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
            power = st.slider("Power", min_value=0.7, max_value=0.99, value=0.8, step=0.05)
        elif calc_type == "Power":
            p0 = st.slider("Null Hypothesis Proportion", min_value=0.0, max_value=1.0, value=0.3, step=0.05)
            p1 = st.slider("Alternative Hypothesis Proportion", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
            n = st.number_input("Sample Size", value=100, min_value=2, step=1)
        elif calc_type == "Minimum Detectable Effect":
            p0 = st.slider("Null Hypothesis Proportion", min_value=0.0, max_value=1.0, value=0.3, step=0.05)
            n = st.number_input("Sample Size", value=100, min_value=2, step=1)
            power = st.slider("Power", min_value=0.7, max_value=0.99, value=0.8, step=0.05)
    
    with col2:
        alpha = st.slider("Alpha", min_value=0.01, max_value=0.1, value=0.05, step=0.01)
    
    # Advanced options
    with st.expander("Advanced Options"):
        test_type = st.radio("Statistical Test", 
                         ["Normal Approximation", "Exact Test", "Likelihood Ratio Test"],
                         horizontal=True)
        
        historical = st.checkbox("Use Historical Control Data", value=False)
        
        if historical:
            col3, col4 = st.columns(2)
            with col3:
                hist_n = st.number_input("Historical Sample Size", value=50, min_value=1, step=1)
                if calc_type != "Minimum Detectable Effect":
                    hist_events = st.number_input("Historical Events", value=int(50*p0), min_value=0, step=1)
                else:
                    hist_events = st.number_input("Historical Events", value=int(50*p0), min_value=0, step=1)
            with col4:
                discount = st.slider("Historical Data Weight", min_value=0.0, max_value=1.0, value=0.5, step=0.1)
        else:
            hist_n = 0
            hist_events = 0
            discount = 0
    
    # Build the parameter dictionary based on calculation type
    params = {
        "p0": p0,
        "alpha": alpha,
        "test_type": test_type,
        "has_historical": historical,
        "historical_n": hist_n,
        "historical_events": hist_events,
        "discount_factor": discount
    }
    
    # Add calculation-specific parameters
    if calc_type == "Sample Size":
        params.update({
            "p1": p1,
            "power": power
        })
    elif calc_type == "Power":
        params.update({
            "p1": p1,
            "n": n
        })
    elif calc_type == "Minimum Detectable Effect":
        params.update({
            "n": n,
            "power": power
        })
    
    return params

def calc_parallel_continuous(params):
    """Real calculation for parallel continuous design"""
    calculation_type = params["calculation_type"]
    hypothesis_type = params["hypothesis_type"]
    std = params["std"]
    alpha = params["alpha"]
    repeated_measures = params.get("repeated_measures", False)
    
    # Handle different calculation types
    if calculation_type == "Sample Size":
        mean1 = params["mean1"]
        power = params.get("power", 0.8)
        allocation_ratio = params.get("allocation_ratio", 1.0)
        use_simulation = params.get("use_simulation", False)
        
        # Handle different hypothesis types
        if hypothesis_type == "Superiority":
            # For superiority, we need both means and calculate delta
            mean2 = params["mean2"]
            delta = abs(mean2 - mean1)
        else:
            # For non-inferiority, we use the NIM and assumed difference
            non_inferiority_margin = params.get("non_inferiority_margin", 0.5)
            assumed_difference = params.get("assumed_difference", 0.0)
            
            # The "delta" is actually the difference from the margin
            # Usually this is (assumed_difference - (-NIM)) for a lower margin
            delta = abs(assumed_difference - (-non_inferiority_margin))
        
        if use_simulation:
            # Get simulation parameters
            nsim = params.get("nsim", 1000)
            min_n = params.get("min_n", 10)
            max_n = params.get("max_n", 500)
            step_n = params.get("step_n", 10)
            
            # Use simulation-based sample size calculation
            if repeated_measures:
                correlation = params.get("correlation", 0.5)
                analysis_method = params.get("analysis_method", "change_score")
                
                # Use simulation function with repeated measures
                result = simulation.sample_size_continuous(
                    delta=delta,
                    std_dev=std,
                    power=power,
                    alpha=alpha,
                    allocation_ratio=allocation_ratio,
                    nsim=nsim,
                    min_n=min_n,
                    max_n=max_n,
                    step=step_n,
                    repeated_measures=True,
                    correlation=correlation,
                    method=analysis_method
                )
            else:
                # Use standard simulation function
                result = simulation.sample_size_continuous(
                    delta=delta,
                    std_dev=std,
                    power=power,
                    alpha=alpha,
                    allocation_ratio=allocation_ratio,
                    nsim=nsim,
                    min_n=min_n,
                    max_n=max_n,
                    step=step_n
                )
        else:
            # Use analytical methods
            if repeated_measures:
                correlation = params.get("correlation", 0.5)
                analysis_method = params.get("analysis_method", "change_score")
                
                # Use repeated measures function
                result = analytical.sample_size_repeated_measures(
                    delta=delta,
                    std_dev=std,
                    correlation=correlation,
                    power=power,
                    alpha=alpha,
                    allocation_ratio=allocation_ratio,
                    method=analysis_method
                )
            else:
                # Check if we're using unequal variances
                unequal_var = params.get("unequal_var", False)
                std2 = params.get("std2", std)
                
                # Use standard parallel function
                if unequal_var:
                    result = analytical.sample_size_continuous(
                        delta=delta,
                        std_dev=std,
                        std_dev2=std2,  # Pass the second standard deviation for unequal variances
                        power=power,
                        alpha=alpha,
                        allocation_ratio=allocation_ratio
                    )
                else:
                    result = analytical.sample_size_continuous(
                        delta=delta,
                        std_dev=std,
                        power=power,
                        alpha=alpha,
                        allocation_ratio=allocation_ratio
                    )
        
        # Calculate effect size for reporting
        if hypothesis_type == "Superiority":
            effect_size = abs(mean2 - mean1) / std
        else:
            # For non-inferiority, effect size is based on the margin
            effect_size = abs(non_inferiority_margin) / std
        
        # Create result dictionary
        return {
            "n_per_group": result["n1"],  # This assumes equal sample sizes
            "n1": result["n1"],
            "n2": result["n2"],
            "total_n": result["total_n"],
            "effect_size": effect_size
        }
        
    elif calculation_type == "Power":
        mean1 = params["mean1"]
        n1 = params["n1"]
        n2 = params["n2"]
        use_simulation = params.get("use_simulation", False)
        
        # Handle different hypothesis types
        if hypothesis_type == "Superiority":
            # For superiority, we need both means
            mean2 = params["mean2"]
            delta = abs(mean2 - mean1)
            is_non_inferiority = False
        else:
            # For non-inferiority, we use the NIM and assumed difference
            non_inferiority_margin = params.get("non_inferiority_margin", 0.5)
            assumed_difference = params.get("assumed_difference", 0.0)
            
            # For power calculation, we need both the margin and the true (assumed) difference
            delta = abs(assumed_difference - (-non_inferiority_margin))
            is_non_inferiority = True
            
            # For simulation functions, we need to calculate an equivalent mean2
            mean2 = mean1 + assumed_difference
        
        if use_simulation:
            # Get simulation parameters
            nsim = params.get("nsim", 1000)
            
            # Use simulation for power calculation
            if repeated_measures:
                correlation = params.get("correlation", 0.5)
                analysis_method = params.get("analysis_method", "change_score")
                
                # Simulate repeated measures design
                result = simulation.simulate_continuous(
                    n1=n1,
                    n2=n2,
                    mean1=mean1,
                    mean2=mean2,
                    std_dev=std,
                    nsim=nsim,
                    alpha=alpha,
                    repeated_measures=True,
                    correlation=correlation,
                    method=analysis_method
                )
            else:
                # Simulate standard parallel design
                result = simulation.simulate_continuous(
                    n1=n1,
                    n2=n2,
                    mean1=mean1,
                    mean2=mean2,
                    std_dev=std,
                    nsim=nsim,
                    alpha=alpha
                )
                
            # Extract simulated power
            power_value = result["power"]
            result_dict = {
                "power": power_value,
                "n1": n1,
                "n2": n2,
                "total_n": n1 + n2,
                "effect_size": delta / std,
                "simulations": nsim,
                "significant_results": int(power_value * nsim)
            }
        else:
            # Use analytical methods for power calculation
            if repeated_measures:
                correlation = params.get("correlation", 0.5)
                analysis_method = params.get("analysis_method", "change_score")
                
                # Use repeated measures function for power
                result = analytical.power_repeated_measures(
                    n1=n1,
                    n2=n2,
                    delta=delta,
                    std_dev=std,
                    correlation=correlation,
                    alpha=alpha,
                    method=analysis_method
                )
            else:
                # Use standard parallel function for power
                result = analytical.power_continuous(
                    n1=n1,
                    n2=n2,
                    delta=delta,
                    std_dev=std,
                    alpha=alpha
                )
            
            # Create result dictionary
            result_dict = {
                "power": result["power"],
                "n1": n1,
                "n2": n2,
                "total_n": n1 + n2,
                "effect_size": delta / std
            }
            
        return result_dict
        
    elif calculation_type == "Minimum Detectable Effect":
        mean1 = params["mean1"]
        n1 = params["n1"]
        n2 = params["n2"]
        power = params.get("power", 0.8)
        use_simulation = params.get("use_simulation", False)
        
        # Handle different hypothesis types
        if hypothesis_type == "Superiority":
            is_non_inferiority = False
        else:
            # For non-inferiority MDE calculation, we're typically finding the minimum margin
            # that can be detected with the given sample size and power
            assumed_difference = params.get("assumed_difference", 0.0)
            is_non_inferiority = True
        
        if use_simulation:
            # Get simulation parameters
            nsim = params.get("nsim", 1000)
            precision = params.get("precision", 0.01)
            
            # Use simulation for MDE calculation
            if repeated_measures:
                correlation = params.get("correlation", 0.5)
                analysis_method = params.get("analysis_method", "change_score")
                
                # Simulate repeated measures design for MDE
                result = simulation.min_detectable_effect_continuous(
                    n1=n1,
                    n2=n2,
                    std_dev=std,
                    power=power,
                    nsim=nsim,
                    alpha=alpha,
                    precision=precision,
                    repeated_measures=True,
                    correlation=correlation,
                    method=analysis_method
                )
            else:
                # Simulate standard parallel design for MDE
                result = simulation.min_detectable_effect_continuous(
                    n1=n1,
                    n2=n2,
                    std_dev=std,
                    power=power,
                    nsim=nsim,
                    alpha=alpha,
                    precision=precision
                )
                
            # Extract simulated MDE
            if "mde" in result:
                mde_value = result["mde"]
            else:
                mde_value = result["delta"]
                
            # Calculate mean2 based on mean1 and MDE
            mean2 = mean1 + mde_value
            
            # Create result dictionary with simulation details
            result_dict = {
                "mean1": mean1,
                "mean2": mean2,
                "minimum_detectable_effect": mde_value,
                "n1": n1,
                "n2": n2,
                "total_n": n1 + n2,
                "effect_size": mde_value / std,
                "simulations": nsim,
                "precision": precision
            }
        else:
            # Use the appropriate analytical function based on design
            if repeated_measures:
                correlation = params.get("correlation", 0.5)
                analysis_method = params.get("analysis_method", "change_score")
                
                # Use repeated measures function for MDE
                result = analytical.min_detectable_effect_repeated_measures(
                    n1=n1,
                    n2=n2,
                    std_dev=std,
                    correlation=correlation,
                    power=power,
                    alpha=alpha,
                    method=analysis_method
                )
            else:
                # Calculate z-scores for given alpha and power
                z_alpha = stats.norm.ppf(1 - alpha/2)
                z_beta = stats.norm.ppf(power)
                
                # Calculate minimum detectable effect
                variance_factor = (1/n1 + 1/n2)
                mde = (z_alpha + z_beta) * std * math.sqrt(variance_factor)
                
                result = {
                    "mde": mde,
                    "parameters": {
                        "n1": n1,
                        "n2": n2,
                        "std_dev": std,
                        "power": power,
                        "alpha": alpha
                    }
                }
            
            # Calculate mean2 based on mean1 and MDE
            if "mde" in result:
                mde_value = result["mde"]
            else:
                mde_value = result["delta"]
                
            mean2 = mean1 + mde_value
            
            # Create result dictionary
            result_dict = {
                "mean1": mean1,
                "mean2": mean2,
                "minimum_detectable_effect": mde_value,
                "n1": n1,
                "n2": n2,
                "total_n": n1 + n2,
                "effect_size": mde_value / std
            }
            
        return result_dict

def calc_single_arm_continuous(params):
    """Real calculation for single arm continuous design"""
    baseline = params["baseline"]
    target = params["target"]
    std = params["std"]
    alpha = params["alpha"]
    power = params.get("power", 0.8)
    
    # Use the real power calculation function
    n = one_sample_t_test_sample_size(
        mean_null=baseline,
        mean_alt=target,
        std_dev=std,
        alpha=alpha,
        power=power
    )
    
    # Calculate effect size for reporting
    effect_size = abs(target - baseline) / std
    
    return {"n": n, "effect_size": effect_size}

def calc_parallel_binary(params):
    """Real calculation for parallel binary design"""
    calculation_type = params["calculation_type"]
    hypothesis_type = params["hypothesis_type"]
    p1 = params["p1"]
    alpha = params["alpha"]
    test_type = params["test_type"]
    use_simulation = params.get("use_simulation", False)
    
    # Handle different calculation types
    if calculation_type == "Sample Size":
        power = params.get("power", 0.8)
        allocation_ratio = params.get("allocation_ratio", 1.0)
        
        # Handle different hypothesis types
        if hypothesis_type == "Superiority":
            # Standard superiority test
            p2 = params["p2"]
            is_non_inferiority = False
        else:
            # Non-inferiority test
            non_inferiority_margin = params.get("non_inferiority_margin", 0.1)
            assumed_difference = params.get("assumed_difference", 0.0)
            # For binary outcomes, p2 is calculated as p1 + assumed_difference
            p2 = p1 + assumed_difference
            is_non_inferiority = True
        
        if use_simulation:
            # Get simulation parameters
            nsim = params.get("nsim", 1000)
            min_n = params.get("min_n", 10)
            max_n = params.get("max_n", 500)
            step_n = params.get("step_n", 10)
            
            # Use simulation-based sample size calculation
            if is_non_inferiority:
                # Get the direction parameter or default to lower
                direction = params.get("direction", "lower")
                
                # Use non-inferiority simulation
                result = binary_simulation.sample_size_binary_non_inferiority_sim(
                    p1=p1,
                    non_inferiority_margin=non_inferiority_margin,
                    power=power,
                    alpha=alpha,
                    allocation_ratio=allocation_ratio,
                    nsim=nsim,
                    min_n=min_n,
                    max_n=max_n,
                    step=step_n,
                    assumed_difference=assumed_difference,
                    direction=direction  # Use the specified direction
                )
            else:
                # Use standard superiority simulation
                result = binary_simulation.sample_size_binary_sim(
                    p1=p1,
                    p2=p2,
                    power=power,
                    alpha=alpha,
                    allocation_ratio=allocation_ratio,
                    nsim=nsim,
                    min_n=min_n,
                    max_n=max_n,
                    step=step_n,
                    test_type=test_type
                )
            
            # Create result dictionary with simulation details
            return {
                "n_per_group": result["n1"],  # This assumes equal sample sizes
                "n1": result["n1"],
                "n2": result["n2"],
                "total_n": result["total_n"],
                "effect_size": abs(p2 - p1),
                "test_type": test_type,
                "simulation": True,
                "nsim": nsim
            }
        else:
            # Use the appropriate analytical function based on hypothesis type
            if is_non_inferiority:
                # Get the direction parameter or default to lower
                direction = params.get("direction", "lower")
                
                # Use dedicated non-inferiority function, without test_type
                # This is a temporary workaround since the function definition doesn't match
                result = analytical.sample_size_binary_non_inferiority(
                    p1=p1,
                    non_inferiority_margin=non_inferiority_margin,
                    power=power,
                    alpha=alpha,
                    allocation_ratio=allocation_ratio,
                    assumed_difference=assumed_difference,
                    direction=direction  # Use the specified direction
                    # Removed test_type parameter due to function signature mismatch
                )
            else:
                # Use standard superiority function
                result = analytical.sample_size_binary(
                    p1=p1,
                    p2=p2,
                    power=power,
                    alpha=alpha,
                    allocation_ratio=allocation_ratio,
                    test_type=test_type
                )
            
            return {
                "n_per_group": result["n1"],  # This assumes equal sample sizes
                "n1": result["n1"],
                "n2": result["n2"],
                "total_n": result["total_n"],
                "effect_size": abs(p2 - p1),
                "test_type": test_type,
                "simulation": False
            }
    
    elif calculation_type == "Power":
        n1 = params["n1"]
        n2 = params["n2"]
        
        # Handle different hypothesis types
        if hypothesis_type == "Superiority":
            # Standard superiority test
            p2 = params["p2"]
            is_non_inferiority = False
        else:
            # Non-inferiority test
            non_inferiority_margin = params.get("non_inferiority_margin", 0.1)
            assumed_difference = params.get("assumed_difference", 0.0)
            # For binary outcomes, p2 is calculated as p1 + assumed_difference
            p2 = p1 + assumed_difference
            is_non_inferiority = True
        
        if use_simulation:
            # Get simulation parameters
            nsim = params.get("nsim", 1000)
            
            # Use simulation for power calculation
            if is_non_inferiority:
                # Use non-inferiority simulation for power
                result = binary_simulation.simulate_binary_non_inferiority(
                    n1=n1,
                    n2=n2,
                    p1=p1,
                    non_inferiority_margin=non_inferiority_margin,
                    nsim=nsim,
                    alpha=alpha,
                    assumed_difference=assumed_difference,
                    direction="lower"  # Typically non-inferiority is a lower bound
                )
            else:
                # Use standard superiority test
                result = power_binary_with_test(
                    n1=n1,
                    n2=n2,
                    p1=p1,
                    p2=p2,
                    alpha=alpha,
                    test=test_type,
                    nsim=nsim
                )
            
            return {
                "power": result["power"],
                "n1": n1,
                "n2": n2,
                "total_n": n1 + n2,
                "effect_size": abs(p2 - p1),
                "test_type": test_type,
                "simulation": True,
                "nsim": nsim,
                "significant_results": int(result["power"] * nsim)
            }
        else:
            # Use analytical method for power calculation
            result = analytical.power_binary(
                n1=n1,
                n2=n2,
                p1=p1,
                p2=p2,
                alpha=alpha,
                test_type=test_type
            )
            
            return {
                "power": result["power"],
                "n1": n1,
                "n2": n2,
                "total_n": n1 + n2,
                "effect_size": abs(p2 - p1),
                "test_type": test_type,
                "simulation": False
            }
    
    elif calculation_type == "Minimum Detectable Effect":
        n1 = params["n1"]
        n2 = params["n2"]
        power = params.get("power", 0.8)
        
        # Handle different hypothesis types
        if hypothesis_type == "Superiority":
            is_non_inferiority = False
        else:
            # For non-inferiority MDE, we're finding the minimum margin
            assumed_difference = params.get("assumed_difference", 0.0)
            is_non_inferiority = True
        
        if use_simulation:
            # Get simulation parameters
            nsim = params.get("nsim", 1000)
            precision = params.get("precision", 0.01)
            
            # Use simulation for MDE calculation
            if is_non_inferiority:
                # Use non-inferiority margin simulation
                result = binary_simulation.min_detectable_binary_non_inferiority_margin_sim(
                    n1=n1,
                    n2=n2,
                    p1=p1,
                    power=power,
                    alpha=alpha,
                    nsim=nsim,
                    precision=precision,
                    assumed_difference=assumed_difference,
                    direction="lower"  # Typically non-inferiority is a lower bound
                )
                
                # The result key will be different for non-inferiority margin
                mde_value = result.get("margin", 0.0)
            else:
                # Use standard superiority simulation for MDE
                result = binary_simulation.min_detectable_effect_binary_sim(
                    n1=n1,
                    n2=n2,
                    p1=p1,
                    power=power,
                    nsim=nsim,
                    alpha=alpha,
                    precision=precision,
                    test_type=test_type
                )
                
                # Get the MDE value
                mde_value = result.get("mde", 0.0)
            
            p2 = p1 + mde_value
            
            return {
                "p1": p1,
                "p2": p2,
                "minimum_detectable_effect": mde_value,
                "n1": n1,
                "n2": n2,
                "total_n": n1 + n2,
                "effect_size": mde_value,
                "test_type": test_type,
                "simulation": True,
                "nsim": nsim,
                "precision": precision
            }
        else:
            # Calculate minimum detectable effect using analytical methods
            result = analytical.min_detectable_effect_binary(
                n1=n1,
                n2=n2,
                p1=p1,
                power=power,
                alpha=alpha,
                test_type=test_type
            )
            
            p2 = p1 + mde_value
            
            return {
                "p1": p1,
                "p2": p2,
                "minimum_detectable_effect": mde_value,
                "n1": n1,
                "n2": n2,
                "total_n": n1 + n2,
                "effect_size": mde_value,
                "test_type": test_type,
                "simulation": False
            }
    
    # Default case (should never get here)
    return {"error": "Invalid calculation type"}

def calc_single_arm_binary(params):
    """Real calculation for single arm binary design"""
    p0 = params["p0"]
    p1 = params["p1"]
    alpha = params["alpha"]
    power = params.get("power", 0.8)
    test_type = params.get("test_type", "Normal Approximation")
    
    # Use the real power calculation function
    n = one_sample_proportion_test_sample_size(
        p_null=p0,
        p_alt=p1,
        alpha=alpha,
        power=power,
        test_type=test_type
    )
    
    return {
        "n": n,
        "effect_size": abs(p1 - p0),
        "test_type": test_type
    }

# Dictionary mapping components to their render and calculation functions
COMPONENTS = {
    ("Parallel RCT", "Continuous Outcome"): {
        "render": render_parallel_continuous,
        "calculate": calc_parallel_continuous
    },
    ("Parallel RCT", "Binary Outcome"): {
        "render": render_parallel_binary,
        "calculate": calc_parallel_binary
    },
    ("Single Arm Trial", "Continuous Outcome"): {
        "render": render_single_arm_continuous,
        "calculate": calc_single_arm_continuous
    },
    ("Single Arm Trial", "Binary Outcome"): {
        "render": render_single_arm_binary,
        "calculate": calc_single_arm_binary
    }
}

# Add app title
st.title("DesignPower: Component Architecture Demo")
st.write("This demonstrates how a component-based architecture makes adding new designs simpler.")

# Initialize session state
if "design_type" not in st.session_state:
    st.session_state.design_type = "Parallel RCT"
    st.session_state.outcome_type = "Continuous Outcome"
    st.session_state.calculation_type = "Sample Size"
    st.session_state.results = None

# Sidebar for design selection
st.sidebar.header("Study Design")

# Design type selection
design_keys = list(DESIGN_CONFIGS.keys())
selected_design_key = st.sidebar.radio("Design Type", design_keys, 
                                   format_func=lambda x: DESIGN_CONFIGS[x]["name"])

design_name = DESIGN_CONFIGS[selected_design_key]["name"]
st.session_state.design_type = design_name

# Outcome type selection
outcomes = DESIGN_CONFIGS[selected_design_key]["outcomes"]
selected_outcome = st.sidebar.radio("Outcome Type", outcomes)
st.session_state.outcome_type = selected_outcome

# Hypothesis type selection
st.sidebar.header("Hypothesis")
hypothesis_types = ["Superiority", "Non-Inferiority"]
if "hypothesis_type" not in st.session_state:
    st.session_state.hypothesis_type = "Superiority"
selected_hypothesis = st.sidebar.radio("Hypothesis Type", hypothesis_types)
st.session_state.hypothesis_type = selected_hypothesis

# Calculation type selection
st.sidebar.header("Calculation Type")
calculation_types = ["Sample Size", "Power", "Minimum Detectable Effect"]
selected_calculation = st.sidebar.radio("Calculate", calculation_types)
st.session_state.calculation_type = selected_calculation

# Render the appropriate component based on selection
component_key = (st.session_state.design_type, st.session_state.outcome_type)
if component_key in COMPONENTS:
    # Add calculation type and hypothesis type to params for the render function
    calc_type = st.session_state.calculation_type
    hypothesis_type = st.session_state.hypothesis_type
    params = COMPONENTS[component_key]["render"](calc_type, hypothesis_type)
    
    # Add calculation type and hypothesis type to the params
    params["calculation_type"] = calc_type
    params["hypothesis_type"] = hypothesis_type
    
    # Calculate button with dynamic text based on calculation type
    button_text = f"Calculate {calc_type}"
    if st.button(button_text):
        st.session_state.results = COMPONENTS[component_key]["calculate"](params)
        
    # Display results if available
    if st.session_state.results:
        st.markdown("### Results")
        for k, v in st.session_state.results.items():
            st.write(f"**{k}:** {v}")
        
        # Simple visualization of results based on calculation type
        calc_type = st.session_state.calculation_type
        fig, ax = plt.subplots(figsize=(8, 4))
        
        # Check if it's a two-arm study by looking for n1 and n2 keys
        is_two_arm = "n1" in st.session_state.results and "n2" in st.session_state.results
        
        if is_two_arm:
            # Two-arm study
            n1 = st.session_state.results["n1"]
            n2 = st.session_state.results["n2"]
            
            if calc_type == "Sample Size":
                ax.bar(["Group 1", "Group 2"], [n1, n2])
                ax.set_ylabel("Sample Size")
                ax.text(0, n1 + 2, str(n1), ha='center', fontweight='bold')
                ax.text(1, n2 + 2, str(n2), ha='center', fontweight='bold')
                ax.text(0.5, max(n1, n2) + 8, f"Total: {n1 + n2}", ha='center', fontweight='bold')
            elif calc_type == "Power":
                power = st.session_state.results.get("power", 0.0)
                ax.bar(["Power"], [power], color='green')
                ax.set_ylim(0, 1.1)
                ax.set_ylabel("Statistical Power")
                ax.text(0, power + 0.05, f"{power:.2f}", ha='center', fontweight='bold')
            elif calc_type == "Minimum Detectable Effect":
                mde = st.session_state.results.get("minimum_detectable_effect", 0)
                ax.bar(["MDE"], [mde], color='orange')
                ax.set_ylabel("Minimum Detectable Effect")
                ax.text(0, mde + (mde*0.1), f"{mde:.2f}", ha='center', fontweight='bold')
        else:
            # Single arm study
            if calc_type == "Sample Size" and "n" in st.session_state.results:
                n = st.session_state.results["n"]
                ax.bar(["Sample Size"], [n])
                ax.set_ylabel("Sample Size")
                ax.text(0, n + 2, str(n), ha='center', fontweight='bold')
            elif calc_type == "Power" and "power" in st.session_state.results:
                power = st.session_state.results.get("power", 0.0)
                ax.bar(["Power"], [power], color='green')
                ax.set_ylim(0, 1.1)
                ax.set_ylabel("Statistical Power")
                ax.text(0, power + 0.05, f"{power:.2f}", ha='center', fontweight='bold')
            elif calc_type == "Minimum Detectable Effect" and "minimum_detectable_effect" in st.session_state.results:
                mde = st.session_state.results["minimum_detectable_effect"]
                ax.bar(["MDE"], [mde], color='orange')
                ax.set_ylabel("Minimum Detectable Effect")
                ax.text(0, mde + (mde*0.1), f"{mde:.2f}", ha='center', fontweight='bold')
            else:
                # Fallback - just show the important numeric results
                st.info("Results visualization not available for this configuration")
                fig.clf()
                plt.close(fig)
                fig = None
        
        if fig is not None:
            st.pyplot(fig)
else:
    st.warning(f"Component for {component_key} is not implemented yet.")

st.markdown("---")
st.info("""
### Component-Based Architecture Benefits
- Each design type is a separate component
- Adding new designs doesn't require changing existing code
- UI and calculations are encapsulated
- Configuration-driven design makes maintenance easier
""")
