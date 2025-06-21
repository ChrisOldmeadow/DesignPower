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
import json
import hashlib

# Import component modules
from core.utils.report_generator import generate_report

# Lazy import for survival converter to avoid plotly dependency issues
def _get_survival_converter():
    from app.components.survival_converter import survival_converter_page
    return survival_converter_page
from app.components.parallel_rct import (
    render_parallel_continuous, render_parallel_binary, render_parallel_survival,
    calculate_parallel_continuous, calculate_parallel_binary, calculate_parallel_survival,
    generate_cli_code_parallel_continuous, generate_cli_code_parallel_binary, generate_cli_code_parallel_survival,
    display_survival_results,
    create_survival_visualization
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
    calculate_cluster_binary,
    generate_cli_code_cluster_continuous,
    generate_cli_code_cluster_binary
)
from app.components.cluster_display_utils import (
    display_sensitivity_analysis,
    display_cluster_variation_info,
    display_icc_conversion_info,
    display_cluster_continuous_results
)
from app.components.stepped_wedge import (
    render_stepped_wedge_continuous,
    render_stepped_wedge_binary,
    calculate_stepped_wedge_continuous,
    calculate_stepped_wedge_binary,
    generate_cli_code_stepped_wedge_continuous,
    generate_cli_code_stepped_wedge_binary
)
from app.components.interrupted_time_series import (
    render_interrupted_time_series_continuous,
    render_interrupted_time_series_binary,
    calculate_interrupted_time_series_continuous,
    calculate_interrupted_time_series_binary,
    generate_cli_code_interrupted_time_series_continuous,
    generate_cli_code_interrupted_time_series_binary
)
# Import unified results display system
from app.components.unified_results_display import unified_display
from app.components.display_configs import register_all_configs

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
    },
    "stepped_wedge": {
        "name": "Stepped Wedge",
        "outcomes": ["Continuous Outcome", "Binary Outcome"]
    },
    "interrupted_time_series": {
        "name": "Interrupted Time Series",
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
    },
    ("Stepped Wedge", "Continuous Outcome"): {
        "render": render_stepped_wedge_continuous,
        "calculate": calculate_stepped_wedge_continuous
    },
    ("Stepped Wedge", "Binary Outcome"): {
        "render": render_stepped_wedge_binary,
        "calculate": calculate_stepped_wedge_binary
    },
    ("Interrupted Time Series", "Continuous Outcome"): {
        "render": render_interrupted_time_series_continuous,
        "calculate": calculate_interrupted_time_series_continuous
    },
    ("Interrupted Time Series", "Binary Outcome"): {
        "render": render_interrupted_time_series_binary,
        "calculate": calculate_interrupted_time_series_binary
    }
}

# Basic app setup - must be first Streamlit command
st.set_page_config(page_title="DesignPower", page_icon=":chart_with_upwards_trend:")

# Initialize unified display system
register_all_configs(unified_display)

# Add app title
st.title("DesignPower: Power and Sample Size Calculator")

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

# Sidebar navigation
st.sidebar.header("Navigation")

# Add page selection
page = st.sidebar.radio(
    "Select Page",
    ["📊 Study Design Calculator", "🔄 Survival Parameter Converter"],
    index=0
)

# Handle page routing
if page == "🔄 Survival Parameter Converter":
    try:
        survival_converter_page = _get_survival_converter()
        survival_converter_page()
    except ImportError as e:
        st.error(f"Survival Parameter Converter requires additional dependencies. Error: {e}")
        st.info("To use the survival parameter converter, please install plotly: `pip install plotly`")
    st.stop()  # Stop execution here for survival converter page

# Continue with study design if not survival converter
if page == "📊 Study Design Calculator":
    st.write("""
        This app calculates power and sample size for various study designs.
        Select a design type and outcome type from the sidebar.
    """)
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
        # Use the unified results display system
        unified_display.display_results(
            results=st.session_state.results,
            params=params,
            design_type=st.session_state.design_type,
            outcome_type=st.session_state.outcome_type,
            calculation_type=st.session_state.calculation_type,
            hypothesis_type=st.session_state.hypothesis_type,
            method_used=st.session_state.method
        )
