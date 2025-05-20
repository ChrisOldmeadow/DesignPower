"""
Survival outcome components for DesignPower application.

This module contains UI components and calculation functions for survival outcomes,
following the same pattern as binary and continuous outcomes in the main app.
"""
import math
import streamlit as st
import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt

# Import survival outcome modules
from core.designs.parallel import analytical_survival
from core.designs.parallel import simulation_survival

def render_parallel_survival(calc_type, hypothesis_type):
    """Render the UI component for parallel RCT with survival outcome."""
    st.write(f"### Parallel RCT with Survival Outcome ({calc_type})")
    
    # Basic parameters
    col1, col2 = st.columns(2)
    with col1:
        if hypothesis_type == "Superiority":
            # Superiority - standard comparison of two median survival times
            if calc_type == "Sample Size":
                median1 = st.number_input("Median Survival Time (Group 1)", 
                                        min_value=1.0, max_value=100.0, value=10.0, step=1.0)
                median2 = st.number_input("Median Survival Time (Group 2)", 
                                        min_value=1.0, max_value=100.0, value=15.0, step=1.0)
                power = st.slider("Power", min_value=0.7, max_value=0.99, value=0.8, step=0.05)
            elif calc_type == "Power":
                median1 = st.number_input("Median Survival Time (Group 1)", 
                                        min_value=1.0, max_value=100.0, value=10.0, step=1.0)
                median2 = st.number_input("Median Survival Time (Group 2)", 
                                        min_value=1.0, max_value=100.0, value=15.0, step=1.0)
                n1 = st.number_input("Sample Size (Group 1)", value=100, min_value=2, step=1)
                n2 = st.number_input("Sample Size (Group 2)", value=100, min_value=2, step=1)
            elif calc_type == "Minimum Detectable Effect":
                median1 = st.number_input("Median Survival Time (Group 1)", 
                                        min_value=1.0, max_value=100.0, value=10.0, step=1.0)
                n1 = st.number_input("Sample Size (Group 1)", value=100, min_value=2, step=1)
                n2 = st.number_input("Sample Size (Group 2)", value=100, min_value=2, step=1)
                power = st.slider("Power", min_value=0.7, max_value=0.99, value=0.8, step=0.05)
        else:
            # Non-inferiority - need control median, margin, and assumed ratio
            if calc_type == "Sample Size":
                median1 = st.number_input("Control Group Median Survival Time", 
                                        min_value=1.0, max_value=100.0, value=10.0, step=1.0,
                                        help="The median survival time in the control/standard treatment group")
                
                # For Non-Inferiority, we need a margin for the hazard ratio
                st.markdown("### Non-Inferiority Parameters")
                non_inferiority_margin = st.number_input(
                    "Non-Inferiority Margin (Hazard Ratio)",
                    min_value=1.01, max_value=3.0, value=1.3, step=0.05,
                    help="The non-inferiority margin as a hazard ratio (must be > 1)"
                )
                    
                assumed_hazard_ratio = st.number_input(
                    "Assumed Hazard Ratio",
                    min_value=0.1, max_value=2.0, value=1.0, step=0.1,
                    help="The assumed true hazard ratio (1.0 = treatments truly equivalent)"
                )
                
                power = st.slider("Power", min_value=0.7, max_value=0.99, value=0.8, step=0.05)
            elif calc_type == "Power":
                median1 = st.number_input("Control Group Median Survival Time", 
                                        min_value=1.0, max_value=100.0, value=10.0, step=1.0)
                n1 = st.number_input("Sample Size (Group 1)", value=250, min_value=2, step=1)
                n2 = st.number_input("Sample Size (Group 2)", value=250, min_value=2, step=1)
                
                # Non-inferiority margin as hazard ratio
                non_inferiority_margin = st.number_input(
                    "Non-Inferiority Margin (Hazard Ratio)",
                    min_value=1.01, max_value=3.0, value=1.5, step=0.05
                )
                
                assumed_hazard_ratio = st.number_input(
                    "Assumed Hazard Ratio",
                    min_value=0.1, max_value=2.0, value=0.9, step=0.1
                )
            elif calc_type == "Minimum Detectable Effect":
                st.warning("Minimum Detectable Effect is not applicable for non-inferiority tests")
    
    with col2:
        # Common parameters regardless of calculation type
        alpha = st.slider("Significance Level", min_value=0.01, max_value=0.1, value=0.05, step=0.01)
        enrollment_period = st.number_input("Enrollment Period (time units)", 
                                        min_value=1.0, max_value=100.0, value=12.0, step=1.0)
        follow_up_period = st.number_input("Follow-up Period (time units)", 
                                        min_value=1.0, max_value=100.0, value=24.0, step=1.0)
        dropout_rate = st.slider("Dropout Rate", min_value=0.0, max_value=0.5, value=0.1, step=0.05)
        
        # Display calculated allocation ratio for Power and MDE calculations
        if calc_type in ["Power", "Minimum Detectable Effect"]:
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
                min_n = st.number_input("Minimum Sample Size to Try", value=50, min_value=10, step=10)
                max_n = st.number_input("Maximum Sample Size to Try", value=500, min_value=100, step=50)
                step_n = st.number_input("Sample Size Step", value=25, min_value=5, step=5)
            elif calc_type == "Minimum Detectable Effect":
                precision = st.slider("MDE Precision", min_value=0.01, max_value=0.5, value=0.1, step=0.01, format="%.2f")
        else:
            # Default values when not using simulation
            nsim = 1000
            min_n = 50
            max_n = 500
            step_n = 25
            precision = 0.1
        
        # Allocation ratio (for sample size calculation)
        if calc_type == "Sample Size":
            allocation_ratio = st.slider("Allocation Ratio (n2/n1)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
        else:
            allocation_ratio = n2 / n1 if n1 > 0 else 1.0
        
        # Additional options specific to survival analysis
        sides = st.radio("Test Type", ["Two-sided", "One-sided"], horizontal=True, index=0)
        sides_value = 1 if sides == "One-sided" else 2
    
    # Build the parameter dictionary based on calculation type
    params = {
        "calculation_type": calc_type,
        "hypothesis_type": hypothesis_type,
        "alpha": alpha,
        "use_simulation": use_simulation,
        "nsim": nsim,
        "enrollment_period": enrollment_period,
        "follow_up_period": follow_up_period,
        "dropout_rate": dropout_rate,
        "sides": sides_value
    }
    
    # Add non-inferiority specific parameters if applicable
    if hypothesis_type == "Non-Inferiority":
        if calc_type in ["Sample Size", "Power"]:
            params.update({
                "non_inferiority_margin": non_inferiority_margin,
                "assumed_hazard_ratio": assumed_hazard_ratio
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
                "median1": median1,
                "median2": median2,
                "power": power,
                "allocation_ratio": allocation_ratio
            })
        else:  # Non-inferiority
            params.update({
                "median1": median1,
                "power": power,
                "allocation_ratio": allocation_ratio
            })
    elif calc_type == "Power":
        if hypothesis_type == "Superiority":
            params.update({
                "median1": median1,
                "median2": median2,
                "n1": n1,
                "n2": n2
            })
        else:  # Non-inferiority
            params.update({
                "median1": median1,
                "n1": n1,
                "n2": n2
            })
    elif calc_type == "Minimum Detectable Effect":
        if hypothesis_type == "Superiority":
            params.update({
                "median1": median1,
                "n1": n1,
                "n2": n2,
                "power": power
            })
    
    return params

def calc_parallel_survival(params):
    """Calculation function for parallel survival designs."""
    calculation_type = params["calculation_type"]
    hypothesis_type = params["hypothesis_type"]
    median1 = params["median1"]
    alpha = params["alpha"]
    use_simulation = params.get("use_simulation", False)
    enrollment_period = params.get("enrollment_period", 12.0)
    follow_up_period = params.get("follow_up_period", 24.0)
    dropout_rate = params.get("dropout_rate", 0.1)
    sides = params.get("sides", 2)
    
    # Handle different calculation types
    if calculation_type == "Sample Size":
        power = params.get("power", 0.8)
        allocation_ratio = params.get("allocation_ratio", 1.0)
        
        # Handle different hypothesis types
        if hypothesis_type == "Superiority":
            # Standard superiority test
            median2 = params["median2"]
            is_non_inferiority = False
        else:
            # Non-inferiority test
            non_inferiority_margin = params.get("non_inferiority_margin", 1.3)
            assumed_hazard_ratio = params.get("assumed_hazard_ratio", 1.0)
            is_non_inferiority = True
        
        if use_simulation:
            # Get simulation parameters
            nsim = params.get("nsim", 1000)
            min_n = params.get("min_n", 50)
            max_n = params.get("max_n", 500)
            step_n = params.get("step_n", 25)
            
            # Use simulation-based sample size calculation
            if is_non_inferiority:
                # Use non-inferiority simulation
                result = simulation_survival.sample_size_survival_non_inferiority_sim(
                    median1=median1,
                    non_inferiority_margin=non_inferiority_margin,
                    power=power,
                    alpha=alpha,
                    allocation_ratio=allocation_ratio,
                    nsim=nsim,
                    min_n=min_n,
                    max_n=max_n,
                    step=step_n,
                    enrollment_period=enrollment_period,
                    follow_up_period=follow_up_period,
                    dropout_rate=dropout_rate,
                    assumed_hazard_ratio=assumed_hazard_ratio
                )
            else:
                # Use standard superiority simulation
                result = simulation_survival.sample_size_survival_sim(
                    median1=median1,
                    median2=median2,
                    power=power,
                    alpha=alpha,
                    allocation_ratio=allocation_ratio,
                    nsim=nsim,
                    min_n=min_n,
                    max_n=max_n,
                    step=step_n,
                    enrollment_period=enrollment_period,
                    follow_up_period=follow_up_period,
                    dropout_rate=dropout_rate
                )
        else:
            # Use analytical sample size calculation
            if is_non_inferiority:
                # Use non-inferiority analytical calculation
                result = analytical_survival.sample_size_survival_non_inferiority(
                    median1=median1,
                    non_inferiority_margin=non_inferiority_margin,
                    power=power,
                    alpha=alpha,
                    allocation_ratio=allocation_ratio,
                    enrollment_period=enrollment_period,
                    follow_up_period=follow_up_period,
                    dropout_rate=dropout_rate,
                    assumed_hazard_ratio=assumed_hazard_ratio
                )
            else:
                # Use standard superiority analytical calculation
                result = analytical_survival.sample_size_survival(
                    median1=median1,
                    median2=median2,
                    power=power,
                    alpha=alpha,
                    allocation_ratio=allocation_ratio,
                    enrollment_period=enrollment_period,
                    follow_up_period=follow_up_period,
                    dropout_rate=dropout_rate,
                    sides=sides
                )
                
    elif calculation_type == "Power":
        n1 = params["n1"]
        n2 = params["n2"]
        
        if hypothesis_type == "Superiority":
            median2 = params["median2"]
            is_non_inferiority = False
        else:
            non_inferiority_margin = params.get("non_inferiority_margin", 1.3)
            assumed_hazard_ratio = params.get("assumed_hazard_ratio", 1.0)
            is_non_inferiority = True
        
        if use_simulation:
            nsim = params.get("nsim", 1000)
            
            if is_non_inferiority:
                result = simulation_survival.power_survival_non_inferiority_sim(
                    n1=n1,
                    n2=n2,
                    median1=median1,
                    non_inferiority_margin=non_inferiority_margin,
                    enrollment_period=enrollment_period,
                    follow_up_period=follow_up_period,
                    dropout_rate=dropout_rate,
                    alpha=alpha,
                    nsim=nsim,
                    assumed_hazard_ratio=assumed_hazard_ratio
                )
            else:
                result = simulation_survival.power_survival_sim(
                    n1=n1,
                    n2=n2,
                    median1=median1,
                    median2=median2,
                    enrollment_period=enrollment_period,
                    follow_up_period=follow_up_period,
                    dropout_rate=dropout_rate,
                    alpha=alpha,
                    nsim=nsim,
                    seed=None
                )
        else:
            if is_non_inferiority:
                result = analytical_survival.power_survival_non_inferiority(
                    n1=n1,
                    n2=n2,
                    median1=median1,
                    non_inferiority_margin=non_inferiority_margin,
                    enrollment_period=enrollment_period,
                    follow_up_period=follow_up_period,
                    dropout_rate=dropout_rate,
                    alpha=alpha,
                    assumed_hazard_ratio=assumed_hazard_ratio
                )
            else:
                result = analytical_survival.power_survival(
                    n1=n1,
                    n2=n2,
                    median1=median1,
                    median2=median2,
                    enrollment_period=enrollment_period,
                    follow_up_period=follow_up_period,
                    dropout_rate=dropout_rate,
                    alpha=alpha,
                    sides=sides
                )
                
    elif calculation_type == "Minimum Detectable Effect":
        if hypothesis_type == "Non-Inferiority":
            st.error("Minimum Detectable Effect calculation is not applicable for non-inferiority tests")
            return None
        
        n1 = params["n1"]
        n2 = params["n2"]
        power = params["power"]
        
        if use_simulation:
            nsim = params.get("nsim", 1000)
            precision = params.get("precision", 0.1)
            
            result = simulation_survival.min_detectable_effect_survival_sim(
                n1=n1,
                n2=n2,
                median1=median1,
                power=power,
                enrollment_period=enrollment_period,
                follow_up_period=follow_up_period,
                dropout_rate=dropout_rate,
                alpha=alpha,
                nsim=nsim,
                precision=precision,
                sides=sides
            )
        else:
            result = analytical_survival.min_detectable_effect_survival(
                n1=n1,
                n2=n2,
                median1=median1,
                power=power,
                enrollment_period=enrollment_period,
                follow_up_period=follow_up_period,
                dropout_rate=dropout_rate,
                alpha=alpha,
                sides=sides
            )
    
    # Format and display results
    if result:
        display_survival_results(result, calculation_type, hypothesis_type, use_simulation)
    
    return result

def display_survival_results(result, calculation_type, hypothesis_type, use_simulation):
    """Display formatted results for survival outcome calculations."""
    st.write("## Results")
    
    # Create two columns for result display
    col1, col2 = st.columns(2)
    
    # Format results based on calculation type
    with col1:
        if calculation_type == "Sample Size":
            st.write(f"**Sample Size (Group 1):** {result['sample_size_1']:.0f}")
            st.write(f"**Sample Size (Group 2):** {result['sample_size_2']:.0f}")
            st.write(f"**Total Sample Size:** {result['sample_size_1'] + result['sample_size_2']:.0f}")
            
            # Show expected events
            if "expected_events_1" in result and "expected_events_2" in result:
                st.write(f"**Expected Events (Group 1):** {result['expected_events_1']:.1f}")
                st.write(f"**Expected Events (Group 2):** {result['expected_events_2']:.1f}")
                st.write(f"**Total Expected Events:** {result['expected_events_1'] + result['expected_events_2']:.1f}")
        
        elif calculation_type == "Power":
            st.write(f"**Power:** {result['power']:.4f}")
            
            # Show expected events
            if "expected_events_1" in result and "expected_events_2" in result:
                st.write(f"**Expected Events (Group 1):** {result['expected_events_1']:.1f}")
                st.write(f"**Expected Events (Group 2):** {result['expected_events_2']:.1f}")
                st.write(f"**Total Expected Events:** {result['expected_events_1'] + result['expected_events_2']:.1f}")
        
        elif calculation_type == "Minimum Detectable Effect":
            # For survival outcomes, display both the hazard ratio and the median difference
            if "minimum_detectable_hazard_ratio" in result:
                st.write(f"**Minimum Detectable Hazard Ratio:** {result['minimum_detectable_hazard_ratio']:.4f}")
            
            if "minimum_detectable_median" in result:
                st.write(f"**Minimum Detectable Median (Group 2):** {result['minimum_detectable_median']:.2f}")
                median_diff = result['minimum_detectable_median'] - result['median1']
                st.write(f"**Difference in Medians:** {median_diff:.2f}")
    
    with col2:
        # Display input parameters and additional information
        st.write("**Input Parameters:**")
        st.write(f"Median Survival (Group 1): {result.get('median1', 'N/A')}")
        
        if calculation_type in ["Sample Size", "Power"] and hypothesis_type == "Superiority":
            st.write(f"Median Survival (Group 2): {result.get('median2', 'N/A')}")
        
        if hypothesis_type == "Non-Inferiority":
            st.write(f"Non-Inferiority Margin (HR): {result.get('non_inferiority_margin', 'N/A')}")
            st.write(f"Assumed Hazard Ratio: {result.get('assumed_hazard_ratio', 'N/A')}")
        
        st.write(f"Alpha: {result.get('alpha', 'N/A')}")
        st.write(f"Enrollment Period: {result.get('enrollment_period', 'N/A')}")
        st.write(f"Follow-up Period: {result.get('follow_up_period', 'N/A')}")
        st.write(f"Dropout Rate: {result.get('dropout_rate', 'N/A')}")
        
        # Show method used
        method = "Simulation" if use_simulation else "Analytical"
        st.write(f"**Method:** {method}")
        
        # Show simulation-specific results if applicable
        if use_simulation and "simulations" in result:
            st.write(f"**Number of Simulations:** {result['simulations']}")
            if "valid_simulations" in result:
                st.write(f"**Valid Simulations:** {result['valid_simulations']}")
    
    # Add visualization if available
    if use_simulation and calculation_type in ["Power", "Sample Size"]:
        create_survival_visualization(result, calculation_type, hypothesis_type)

def create_survival_visualization(result, calculation_type, hypothesis_type):
    """Create visualization for survival outcome results."""
    st.write("## Visualizations")
    
    # Create a simple visualization based on calculation type
    fig, ax = plt.subplots(figsize=(8, 5))
    
    if calculation_type == "Sample Size":
        # For sample size calculation, show expected survival curves
        times = np.linspace(0, result.get('follow_up_period', 24) * 1.5, 100)
        
        # Calculate survival curves based on exponential model
        ln2 = math.log(2)
        hazard1 = ln2 / result.get('median1', 10)
        
        if hypothesis_type == "Superiority":
            hazard2 = ln2 / result.get('median2', 15)
        else:  # Non-inferiority
            hazard2 = hazard1 * result.get('assumed_hazard_ratio', 1.0)
        
        surv1 = np.exp(-hazard1 * times)
        surv2 = np.exp(-hazard2 * times)
        
        # Plot survival curves
        ax.plot(times, surv1, 'b-', label=f"Group 1 (Median = {result.get('median1', 10):.1f})")
        ax.plot(times, surv2, 'r-', label=f"Group 2 (Median = {result.get('median2', 15):.1f})")
        
        # Add horizontal line at 0.5 for median reference
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        
        # Add vertical lines at median survival times
        ax.axvline(x=result.get('median1', 10), color='blue', linestyle=':', alpha=0.5)
        
        if hypothesis_type == "Superiority":
            ax.axvline(x=result.get('median2', 15), color='red', linestyle=':', alpha=0.5)
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Survival Probability')
        ax.set_title('Expected Survival Curves')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    elif calculation_type == "Power":
        # For power calculation, show the relationship between power and effect size
        
        # Check if we have the necessary information to create a meaningful visualization
        if hypothesis_type == "Superiority" and "median1" in result and "median2" in result:
            # Create a range of median2 values around the actual value
            median1 = result.get('median1', 10)
            median2 = result.get('median2', 15)
            
            # Range of median2 values to plot
            median_range = np.linspace(median1 * 0.8, median1 * 1.5, 20)
            powers = []
            
            # Calculate power for each median2 value using the same parameters
            for m2 in median_range:
                # Use simple approximation for visualization purposes
                hr = math.log(2) / m2 / (math.log(2) / median1)
                effect_size = math.log(hr) / math.sqrt(4 / result.get('n1', 100))
                power = 1 - stats.norm.cdf(1.96 - effect_size)
                powers.append(power)
            
            # Plot power curve
            ax.plot(median_range, powers, 'b-')
            
            # Mark the actual median2 and power
            ax.plot(median2, result.get('power', 0.8), 'ro', markersize=8)
            
            # Add horizontal line at the target power level
            ax.axhline(y=0.8, color='gray', linestyle='--', alpha=0.5)
            
            ax.set_xlabel('Median Survival Time (Group 2)')
            ax.set_ylabel('Power')
            ax.set_title('Power vs. Effect Size')
            ax.grid(True, alpha=0.3)
    
    st.pyplot(fig)
