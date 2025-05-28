"""
Simplified component-based demo for DesignPower architecture.

This is a minimal working example showing how components enable a more
maintainable and extensible application structure.
"""
import os
import sys

# Make sure the project root is in the path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import base64
import streamlit as st
import math
import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import graphviz

# Import component modules
from core.utils.report_generator import generate_report
from app.components.parallel_rct import (
    render_parallel_continuous, calculate_parallel_continuous,
    render_parallel_binary, calculate_parallel_binary,
    render_parallel_survival, calculate_parallel_survival,
    display_survival_results, create_survival_visualization
)
from app.components.single_arm import (
    render_single_arm_continuous, calculate_single_arm_continuous,
    render_single_arm_binary, calculate_single_arm_binary,
    render_single_arm_survival, calculate_single_arm_survival
)
from app.components.cluster_rct import (
    render_cluster_continuous,
    render_cluster_binary,
    calculate_cluster_continuous,
    calculate_cluster_binary
)
from app.components.cluster_display_utils import (
    display_sensitivity_analysis,
    display_cluster_variation_info,
    display_icc_conversion_info
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
    },
    "cluster_rct": {
        "name": "Cluster RCT",
        "outcomes": ["Continuous Outcome", "Binary Outcome"]
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
    },
    ("Cluster RCT", "Continuous Outcome"): {
        "render": render_cluster_continuous,
        "calculate": calculate_cluster_continuous
    },
    ("Cluster RCT", "Binary Outcome"): {
        "render": render_cluster_binary,
        "calculate": calculate_cluster_binary
    }
}

# Basic app setup - must be first Streamlit command
st.set_page_config(page_title="DesignPower", page_icon=":chart_with_upwards_trend:")

# Add app title
st.title("DesignPower: Power and Sample Size Calculator")
st.write("""
    This app calculates power and sample size for various study designs.
    Select a design type and outcome type from the sidebar.
""")

# Initialize session state variables if not already set
if "design_type" not in st.session_state:
    st.session_state.design_type = "Parallel RCT"
if "outcome_type" not in st.session_state:
    st.session_state.outcome_type = "Continuous Outcome"
if "results" not in st.session_state:
    st.session_state.results = None
if "calculation_type" not in st.session_state:
    st.session_state.calculation_type = "Sample Size"
if "hypothesis_type" not in st.session_state:
    st.session_state.hypothesis_type = "Superiority"
if "method" not in st.session_state:
    st.session_state.method = "Analytical"

# Add expandable About section at the top of the sidebar
with st.sidebar.expander("ℹ️ About DesignPower", expanded=False):
    st.write("DesignPower is an open-source tool for statistical design of clinical studies, providing accurate sample size and power calculations for various trial designs.")
    
    st.write("**Key Features:**")
    features = [
        "Publication-Ready Reports",
        "Analytical & Simulation Methods",
        "Reproducible & Validated Results"
    ]
    
    # Display each feature on its own line with a bullet
    for feature in features:
        st.write(f"• {feature}")

# Sidebar for design selection
st.sidebar.header("Study Design")

# Design type selection
design_keys = list(DESIGN_CONFIGS.keys())

# Track previous design selection to detect changes
if 'previous_design' not in st.session_state:
    st.session_state.previous_design = None

selected_design_key = st.sidebar.radio("Design Type", design_keys, 
                                   format_func=lambda x: DESIGN_CONFIGS[x]["name"])

design_name = DESIGN_CONFIGS[selected_design_key]["name"]

# Check if design type changed and reset results if it did
if st.session_state.previous_design != design_name:
    if 'results' in st.session_state:
        del st.session_state.results
    st.session_state.previous_design = design_name

st.session_state.design_type = design_name

# Outcome type selection
outcomes = DESIGN_CONFIGS[selected_design_key]["outcomes"]

# Track previous outcome selection to detect changes
if 'previous_outcome' not in st.session_state:
    st.session_state.previous_outcome = None

selected_outcome = st.sidebar.radio("Outcome Type", outcomes)

# Check if outcome type changed and reset results if it did
if st.session_state.previous_outcome != selected_outcome:
    if 'results' in st.session_state:
        del st.session_state.results
    st.session_state.previous_outcome = selected_outcome

st.session_state.outcome_type = selected_outcome

# Hypothesis type selection
st.sidebar.header("Hypothesis")
hypothesis_types = ["Superiority", "Non-Inferiority"]
selected_hypothesis = st.sidebar.radio("Hypothesis Type", hypothesis_types)
st.session_state.hypothesis_type = selected_hypothesis

# Calculation type selection
st.sidebar.header("Calculation Type")
calculation_types = ["Sample Size", "Power", "Minimum Detectable Effect"]
selected_calculation = st.sidebar.radio("Calculate", calculation_types)
st.session_state.calculation_type = selected_calculation

# No redundant key features section - removed

# Documentation and examples
st.sidebar.markdown("---")
st.sidebar.markdown("### Documentation & Examples")
st.sidebar.markdown(
    """<div style='font-size: 0.9em;'>
    <a href='https://github.com/ChrisOldmeadow/DesignPower/blob/main/README.md' target='_blank'>📚 User Guide</a><br>
    <a href='https://github.com/ChrisOldmeadow/DesignPower/blob/main/docs/EXAMPLES.md' target='_blank'>📝 Example Calculations</a><br>
    <a href='https://github.com/ChrisOldmeadow/DesignPower/tree/main/docs/methods' target='_blank'>📊 Statistical Methods</a>
    </div>""", 
    unsafe_allow_html=True
)

# GitHub repository information
st.sidebar.markdown("---")
st.sidebar.markdown(
    "<div style='text-align: center;'>"
    "<a href='https://github.com/ChrisOldmeadow/DesignPower' target='_blank'>"
    "<img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' width='25'/> GitHub Repository</a>"
    "</div>", 
    unsafe_allow_html=True
)

# Render the appropriate component based on selection
component_key = (st.session_state.design_type, st.session_state.outcome_type)
if component_key in COMPONENTS:
    # Add calculation type and hypothesis type to params for the render function
    calc_type = st.session_state.calculation_type
    hypothesis_type = st.session_state.hypothesis_type
    params = COMPONENTS[component_key]["render"](calc_type, hypothesis_type)
    
    # Store the method selected in the component's render function
    # This is important for display_survival_results
    if params and "method" in params:
        st.session_state.method = params["method"]

    # Add calculation type and hypothesis type to the params
    params["calculation_type"] = calc_type
    params["hypothesis_type"] = hypothesis_type
    
    # Calculate button with dynamic text based on calculation type
    button_text = f"Calculate {calc_type}"
    if st.button(button_text):
        st.session_state.results = COMPONENTS[component_key]["calculate"](params)
        
    # Display results if available
    if "results" in st.session_state and st.session_state.results is not None:
        # Check if there's an error in the results
        if isinstance(st.session_state.results, dict) and "error" in st.session_state.results:
            st.error(st.session_state.results["error"])
        else:
            results = st.session_state.results
            design_name = st.session_state.design_type
            outcome_name = st.session_state.outcome_type
            calc_type = st.session_state.calculation_type # Already available
            hypothesis_type = st.session_state.hypothesis_type # Already available
            method_used = st.session_state.method # Get stored method

            # Special handling for Parallel RCT Survival Outcome
            if design_name == "Parallel RCT" and outcome_name == "Survival Outcome":
                display_survival_results(
                    result=results,
                    calculation_type=calc_type,
                    hypothesis_type=hypothesis_type,
                    use_simulation=(method_used.lower() == "simulation")
                )
                create_survival_visualization(
                    result=results,
                    calculation_type=calc_type,
                    hypothesis_type=hypothesis_type
                )
            else:
                # Existing generic results display logic (Corrected Indentation)
                st.markdown("### Results Summary")
                st.markdown("---")
                
                design_method = results.get("design_method")
                
                if design_method == "A'Hern":
                    st.markdown("### A'Hern Design Results")
                    st.markdown("---")
                    tab1, tab2 = st.tabs(["📊 Key Parameters", "📏 Effect Size"])
                    with tab1:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("##### Sample Size Calculation")
                            st.markdown(f"""<div style='background-color:#e6f3ff;padding:10px;border-radius:5px;margin-bottom:5px;'>
                                          <b>Required Sample Size (n):</b> {results.get('n')}
                                          </div>""", unsafe_allow_html=True)
                            st.markdown(f"""<div style='background-color:#e6f3ff;padding:10px;border-radius:5px;margin-bottom:5px;'>
                                          <b>Rejection Threshold (r):</b> {results.get('r')}
                                          </div>""", unsafe_allow_html=True)
                            st.markdown(f"""<div style='background-color:#e6fff0;padding:10px;border-radius:5px;margin-bottom:5px;'>
                                          <b>Interpretation:</b> Reject H₀ if {results.get('r')} or more responses are observed
                                          </div>""", unsafe_allow_html=True)
                        with col2:
                            st.markdown("##### Error Rates")
                            # Ensure 'params' is available here; it should be from the render call scope
                            st.markdown(f"""<div style='background-color:#fff0e6;padding:10px;border-radius:5px;margin-bottom:5px;'>
                                          <b>Target Type I Error (α):</b> {params.get('alpha')}
                                          </div>""", unsafe_allow_html=True)
                            st.markdown(f"""<div style='background-color:#fff0e6;padding:10px;border-radius:5px;margin-bottom:5px;'>
                                          <b>Actual Type I Error:</b> {results.get('actual_alpha')}
                                          </div>""", unsafe_allow_html=True)
                            st.markdown(f"""<div style='background-color:#e6e6ff;padding:10px;border-radius:5px;margin-bottom:5px;'>
                                          <b>Target Power:</b> {params.get('power', 1-params.get('beta', 0.2))}
                                          </div>""", unsafe_allow_html=True)
                            st.markdown(f"""<div style='background-color:#e6e6ff;padding:10px;border-radius:5px;'>
                                          <b>Actual Power:</b> {results.get('actual_power')}
                                          </div>""", unsafe_allow_html=True)
                    with tab2:
                        st.markdown("##### Effect Size Parameters")
                        st.markdown(f"""<div style='background-color:#f0f0f0;padding:10px;border-radius:5px;margin-bottom:5px;'>
                                      <b>Unacceptable Response Rate (p0):</b> {params.get('p0')}
                                      </div>""", unsafe_allow_html=True)
                        st.markdown(f"""<div style='background-color:#f0f0f0;padding:10px;border-radius:5px;'>
                                      <b>Desirable Response Rate (p1):</b> {params.get('p1')}
                                      </div>""", unsafe_allow_html=True)

                elif design_method == "Simon's Two-Stage":
                    st.markdown("### Simon's Two-Stage Design Results")
                    st.markdown("---")
                    st.markdown("#### Stage 1")
                    col1_s1, col2_s1 = st.columns(2)
                    with col1_s1:
                        st.metric(label="Sample Size (n1)", value=results.get("n1"))
                    with col2_s1:
                        st.metric(label="Rejection Threshold (r1)", value=results.get("r1"))
                    st.markdown(f"**Interpretation:** If ≤ {results.get('r1')} responses in {results.get('n1')} patients, stop the trial (futility).")

                    st.markdown("#### Stage 2")
                    col1_s2, col2_s2, col3_s2 = st.columns(3)
                    with col1_s2:
                        st.metric(label="Total Sample Size (N)", value=results.get("N"))
                    with col2_s2:
                        st.metric(label="Overall Rejection Threshold (r)", value=results.get("r"))
                    with col3_s2:
                        st.metric(label="Probability of Early Termination (PET)", value=f"{results.get('PET', 0.0):.3f}")
                    st.markdown(f"**Interpretation:** If > {results.get('r1')} responses in Stage 1, proceed to Stage 2. "
                                f"Overall, if ≤ {results.get('r')} responses in {results.get('N')} patients, reject H₁ (treatment ineffective).")
                    
                    st.markdown("#### Expected Sample Size")
                    st.metric(label="Expected Sample Size (EN)", value=f"{results.get('EN', 0.0):.2f}")

                else: 
                    # Fallback for other generic results
                    filtered_results = {
                        k: v for k, v in results.items() 
                        if k not in [
                            "design_method", "error", "power_curve_data", 
                            "survival_curves", "power_vs_hr_data", "plot_data",
                            "alpha_param", "power_param", "non_inferiority_margin", "assumed_hazard_ratio"
                        ]
                    }
                    if not filtered_results:
                        st.info("No specific tabular results to display for this configuration. Check visualizations if applicable.")
                    
                    for key, value in filtered_results.items():
                        disp_col1, disp_col2 = st.columns([1, 2])
                        with disp_col1:
                            st.markdown(f"**{key.replace('_', ' ').title()}:**")
                        with disp_col2:
                            if isinstance(value, float):
                                st.markdown(f"{value:.3f}")
                            elif isinstance(value, (list, tuple)) and len(value) == 2 and all(isinstance(i, (int, float)) for i in value):
                                 st.markdown(f"({value[0]:.3f}, {value[1]:.3f})") # e.g. confidence interval
                            else:
                                st.markdown(str(value))
                
                # Generate Report Button (Common to all non-survival results displayed in this 'else' block)
                if st.button("Generate Report"):
                    report_text = generate_report(
                        results=results,
                        params=params, # params should contain calc_type, hypothesis_type, method etc.
                        design_type=st.session_state.design_type,
                        outcome_type=st.session_state.outcome_type
                    )
                    report_html_content = f"<pre>{report_text}</pre>"
                    report_name = f"{st.session_state.design_type.replace(' ', '_')}_{st.session_state.outcome_type.replace(' ', '_')}_{st.session_state.calculation_type.replace(' ', '_')}_report"
                    if report_html_content:
                        st.markdown("---<br>Generated Report:", unsafe_allow_html=True) # Add a separator and title
                        st.markdown(report_html_content, unsafe_allow_html=True)
                        b64 = base64.b64encode(report_html_content.encode()).decode()
                        href = f'<a href="data:text/html;base64,{b64}" download="{report_name}.html">Download Report</a>'
                        st.markdown(href, unsafe_allow_html=True)
                    else:
                        st.error("Could not generate report.")
