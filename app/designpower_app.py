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
    
# Import component modules
from core.utils.report_generator import generate_report
from app.components.parallel_rct import (
    render_parallel_continuous, calculate_parallel_continuous,
    render_parallel_binary, calculate_parallel_binary,
    render_parallel_survival, calculate_parallel_survival
)
from app.components.single_arm import (
    render_single_arm_continuous, calculate_single_arm_continuous,
    render_single_arm_binary, calculate_single_arm_binary,
    render_single_arm_survival, calculate_single_arm_survival
)

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

# Dictionary mapping components to their render and calculation functions
COMPONENTS = {
    ("Parallel RCT", "Continuous Outcome"): {
        "render": render_parallel_continuous,
        "calculate": calculate_parallel_continuous
    },
    ("Parallel RCT", "Binary Outcome"): {
        "render": render_parallel_binary,
        "calculate": calculate_parallel_binary
    },
    ("Parallel RCT", "Survival Outcome"): {
        "render": render_parallel_survival,
        "calculate": calculate_parallel_survival
    },
    ("Single Arm Trial", "Continuous Outcome"): {
        "render": render_single_arm_continuous,
        "calculate": calculate_single_arm_continuous
    },
    ("Single Arm Trial", "Binary Outcome"): {
        "render": render_single_arm_binary,
        "calculate": calculate_single_arm_binary
    },
    ("Single Arm Trial", "Survival Outcome"): {
        "render": render_single_arm_survival,
        "calculate": calculate_single_arm_survival
    }
}

# Add app title
st.title("DesignPower: Power and Sample Size Calculator")
st.write("""
    This app calculates power and sample size for various study designs.
    Select a design type and outcome type from the sidebar.
""")

# Initialize session state if needed
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
        
        # Check if there's an error in the results
        if "error" in st.session_state.results:
            st.error(f"Error: {st.session_state.results['error']}")
        else:
            # Format results in a more organized way
            results = st.session_state.results
            
            # Special handling for A'Hern design and Simon's two-stage design
            design_method = results.get("design_method")
            
            if design_method == "A'Hern":
                # Create a two-column layout for A'Hern results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Sample Size Calculation")
                    st.write(f"**Required Sample Size (n):** {results.get('n')}")
                    st.write(f"**Rejection Threshold (r):** {results.get('r')}")
                    st.write(f"**Interpretation:** Reject H₀ if {results.get('r')} or more responses are observed")
                
                with col2:
                    st.subheader("Error Rates")
                    st.write(f"**Target Type I Error (α):** {params.get('alpha')}")
                    st.write(f"**Actual Type I Error:** {results.get('actual_alpha')}")
                    st.write(f"**Target Power:** {params.get('power', 1-params.get('beta', 0.2))}")
                    st.write(f"**Actual Power:** {results.get('actual_power')}")
                
                # Display effect size information
                st.subheader("Effect Size")
                st.write(f"**Null Response Rate (p₀):** {params.get('p0')}")
                st.write(f"**Alternative Response Rate (p₁):** {params.get('p')}")
                st.write(f"**Absolute Risk Difference:** {results.get('absolute_risk_difference')}")
                st.write(f"**Relative Risk:** {results.get('relative_risk')}")
            
            elif design_method == "Simon's Two-Stage":
                # Create a specific layout for Simon's two-stage design results
                st.subheader("Simon's Two-Stage Design Results")
                st.write(f"**Design Type:** {results.get('design_type', 'Optimal')}")
                
                # Create a three-column layout for Simon's results
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.subheader("Stage 1")
                    st.write(f"**First Stage Sample Size (n₁):** {results.get('n1')}")
                    st.write(f"**First Stage Threshold (r₁):** {results.get('r1')}")
                    st.write(f"**Probability of Early Termination (PET₀):** {results.get('PET0')}")
                
                with col2:
                    st.subheader("Overall Design")
                    st.write(f"**Total Sample Size (n):** {results.get('n')}")
                    st.write(f"**Final Threshold (r):** {results.get('r')}")
                    st.write(f"**Expected Sample Size (EN₀):** {results.get('EN0')}")
                
                with col3:
                    st.subheader("Error Rates")
                    st.write(f"**Target Type I Error (α):** {params.get('alpha')}")
                    st.write(f"**Actual Type I Error:** {results.get('actual_alpha')}")
                    st.write(f"**Target Power:** {params.get('power', 1-params.get('beta', 0.2))}")
                    st.write(f"**Actual Power:** {results.get('actual_power')}")
                
                # Display decision rules and effect size information
                st.subheader("Decision Rules")
                st.markdown(f"**Stage 1:** If responses ≤ {results.get('r1')}, stop the trial for futility.")
                st.markdown(f"**Stage 2:** If total responses > {results.get('r')}, reject H₀.")
                
                st.subheader("Effect Size")
                st.write(f"**Null Response Rate (p₀):** {params.get('p0')}")
                st.write(f"**Alternative Response Rate (p₁):** {params.get('p')}")
                st.write(f"**Absolute Risk Difference:** {results.get('absolute_risk_difference')}")
                st.write(f"**Relative Risk:** {results.get('relative_risk')}")
            
            else:
                # Standard display for other results
                for k, v in results.items():
                    # Skip design_method from display
                    if k != "design_method":
                        st.write(f"**{k}:** {v}")
            
        # Generate and display report text
        st.markdown("### Report Text")
        report_text = generate_report(
            st.session_state.results, 
            params, 
            st.session_state.design_type, 
            st.session_state.outcome_type
        )
        
        # Display the report in an expandable section
        with st.expander("Copyable Report for Publication", expanded=True):
            st.markdown(report_text)
            
            # Add a copy button
            if st.button("Copy to Clipboard"):
                try:
                    st.write("Report copied to clipboard!")
                    # Use JavaScript to copy text to clipboard
                    st.markdown(
                        f"""
                        <script>
                            navigator.clipboard.writeText(`{report_text}`);
                        </script>
                        """,
                        unsafe_allow_html=True
                    )
                except Exception as e:
                    st.error(f"Could not copy to clipboard: {e}")
        
        # Simple visualization of results based on calculation type
        calc_type = st.session_state.calculation_type
        fig, ax = plt.subplots(figsize=(8, 4))
        
        # Check if it's a two-arm study by looking for n1 and n2 keys
        is_two_arm = "n1" in st.session_state.results and "n2" in st.session_state.results
        
        if calc_type == "Sample Size":
            # Display a bar chart of sample sizes
            if is_two_arm:
                sizes = [st.session_state.results["n1"], st.session_state.results["n2"]]
                groups = ["Group 1", "Group 2"]
            else:
                sizes = [st.session_state.results.get("n", 0)]
                groups = ["Sample Size"]
                
            ax.bar(groups, sizes, color='skyblue')
            ax.set_ylabel('Sample Size')
            ax.set_title('Required Sample Size by Group')
            
        elif calc_type == "Power":
            # Display the power as a horizontal line
            power = st.session_state.results.get("power", 0)
            ax.axhline(y=power, color='r', linestyle='-')
            ax.set_ylim(0, 1)
            ax.set_ylabel('Power')
            ax.set_title(f'Achieved Power: {power:.2f}')
            ax.set_xticks([])
            
        elif calc_type == "Minimum Detectable Effect":
            # Display the MDE
            mde = st.session_state.results.get("mde", 0)
            ax.bar(["MDE"], [mde], color='salmon')
            ax.set_ylabel('Effect Size')
            ax.set_title(f'Minimum Detectable Effect: {mde:.3f}')
        
        st.pyplot(fig)
