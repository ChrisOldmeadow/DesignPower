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
import graphviz

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
                # Create an enhanced layout for A'Hern results with consistent styling
                st.markdown("### A'Hern Design Results")
                st.markdown("---")
                
                # Create tabs for different aspects of the results
                tab1, tab2 = st.tabs(["📊 Key Parameters", "📏 Effect Size"])
                
                with tab1:
                    # Create a more visually appealing layout for key parameters
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("##### Sample Size Calculation")
                        st.markdown(f"<div style='background-color:#e6f3ff;padding:10px;border-radius:5px;margin-bottom:5px;'>"
                                  f"<b>Required Sample Size (n):</b> {results.get('n')}"
                                  f"</div>", unsafe_allow_html=True)
                        
                        st.markdown(f"<div style='background-color:#e6f3ff;padding:10px;border-radius:5px;margin-bottom:5px;'>"
                                  f"<b>Rejection Threshold (r):</b> {results.get('r')}"
                                  f"</div>", unsafe_allow_html=True)
                        
                        st.markdown(f"<div style='background-color:#e6fff0;padding:10px;border-radius:5px;margin-bottom:5px;'>"
                                  f"<b>Interpretation:</b> Reject H₀ if {results.get('r')} or more responses are observed"
                                  f"</div>", unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown("##### Error Rates")
                        st.markdown(f"<div style='background-color:#fff0e6;padding:10px;border-radius:5px;margin-bottom:5px;'>"
                                  f"<b>Target Type I Error (α):</b> {params.get('alpha')}"
                                  f"</div>", unsafe_allow_html=True)
                        
                        st.markdown(f"<div style='background-color:#fff0e6;padding:10px;border-radius:5px;margin-bottom:5px;'>"
                                  f"<b>Actual Type I Error:</b> {results.get('actual_alpha')}"
                                  f"</div>", unsafe_allow_html=True)
                        
                        st.markdown(f"<div style='background-color:#e6e6ff;padding:10px;border-radius:5px;margin-bottom:5px;'>"
                                  f"<b>Target Power:</b> {params.get('power', 1-params.get('beta', 0.2))}"
                                  f"</div>", unsafe_allow_html=True)
                        
                        st.markdown(f"<div style='background-color:#e6e6ff;padding:10px;border-radius:5px;'>"
                                  f"<b>Actual Power:</b> {results.get('actual_power')}"
                                  f"</div>", unsafe_allow_html=True)
                
                with tab2:
                    # Create a more visually appealing layout for effect size information
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("##### Response Rates")
                        st.markdown(f"<div style='background-color:#f0e6ff;padding:10px;border-radius:5px;margin-bottom:5px;'>"
                                  f"<b>Null Response Rate (p₀):</b> {params.get('p0')}"
                                  f"</div>", unsafe_allow_html=True)
                        
                        st.markdown(f"<div style='background-color:#f0e6ff;padding:10px;border-radius:5px;'>"
                                  f"<b>Alternative Response Rate (p₁):</b> {params.get('p')}"
                                  f"</div>", unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown("##### Effect Measures")
                        st.markdown(f"<div style='background-color:#ffe6e6;padding:10px;border-radius:5px;margin-bottom:5px;'>"
                                  f"<b>Absolute Risk Difference:</b> {results.get('absolute_risk_difference')}"
                                  f"</div>", unsafe_allow_html=True)
                        
                        st.markdown(f"<div style='background-color:#ffe6e6;padding:10px;border-radius:5px;'>"
                                  f"<b>Relative Risk:</b> {results.get('relative_risk')}"
                                  f"</div>", unsafe_allow_html=True)
                
                # Add an information card explaining A'Hern design
                with st.expander("💡 About A'Hern Design", expanded=False):
                    st.markdown("""
                    **A'Hern Design** is a single-stage design for phase II trials using exact binomial probabilities instead of normal approximations.
                    This approach is more appropriate for smaller sample sizes and provides more precise results.
                    
                    The design calculates the minimum number of responses (r) needed to reject the null hypothesis that the 
                    response rate is less than or equal to p₀, with a specific significance level (α) and desired power.
                    """)
            
            elif design_method == "Simon's Two-Stage":
                # Create a specific layout for Simon's two-stage design results with enhanced styling
                st.markdown("### 📊 Simon's Two-Stage Design Results")
                
                # Create a horizontal line for better separation
                st.markdown("---")
                
                # Show design type with better highlighting
                design_type = results.get('design_type', 'Optimal')
                st.markdown(f"<div style='background-color:#f0f2f6;padding:10px;border-radius:5px;margin-bottom:10px;'><h4>Design Type: {design_type}</h4></div>", unsafe_allow_html=True)
                
                # Display flowchart of the design for visual understanding
                with st.expander("📋 View Design Flowchart", expanded=True):
                    n1 = results.get('n1')
                    r1 = results.get('r1')
                    n = results.get('n')
                    r = results.get('r')
                    
                    # Create a customized flowchart based on actual parameters using graphviz
                    st.markdown("##### Design Flowchart with Calculated Parameters")
                    
                    # Create a graphviz object for the flowchart
                    results_graph = graphviz.Digraph()
                    results_graph.attr('node', shape='box', style='filled', color='lightblue', fontname='Arial', 
                                    fontsize='12', margin='0.2,0.1')
                    results_graph.attr('edge', fontname='Arial', fontsize='11')
                    
                    # Define the nodes with actual parameter values - simplified to focus on key decision points
                    results_graph.node('stage1', f'Stage 1:\nEnroll {n1} patients')
                    results_graph.node('decision1', f'Responses > {r1}?', shape='diamond', color='lightgreen')
                    results_graph.node('stop', 'Stop trial\nfor futility', color='#ffcccc')
                    results_graph.node('stage2', f'Stage 2:\nEnroll {n-n1}\nmore patients')
                    results_graph.node('decision2', f'Total responses > {r}?', shape='diamond', color='lightgreen')
                    results_graph.node('ineffective', 'Treatment ineffective\n(Accept H₀)', color='#ffcccc')
                    results_graph.node('effective', 'Treatment effective\n(Reject H₀)', color='#ccffcc')
                    
                    # Add edges to connect the nodes in the simplified flowchart
                    results_graph.edge('stage1', 'decision1')
                    results_graph.edge('decision1', 'stop', label='NO')
                    results_graph.edge('decision1', 'stage2', label='YES')
                    results_graph.edge('stage2', 'decision2')
                    results_graph.edge('decision2', 'ineffective', label='NO')
                    results_graph.edge('decision2', 'effective', label='YES')
                    
                    # Display the graphviz chart in Streamlit
                    st.graphviz_chart(results_graph)
                
                # Create tabs for different aspects of the results
                tab1, tab2, tab3, tab4 = st.tabs(["📈 Key Parameters", "⚠️ Error Rates", "📏 Effect Size", "📋 Decision Rules"])
                
                with tab1:
                    # Create a more visually appealing layout for key parameters
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("##### Stage 1 Parameters")
                        st.markdown(f"<div style='background-color:#e6f3ff;padding:10px;border-radius:5px;margin-bottom:5px;'>"
                                  f"<b>First Stage Sample Size (n₁):</b> {results.get('n1')}"
                                  f"</div>", unsafe_allow_html=True)
                        
                        st.markdown(f"<div style='background-color:#e6f3ff;padding:10px;border-radius:5px;margin-bottom:5px;'>"
                                  f"<b>First Stage Threshold (r₁):</b> {results.get('r1')}"
                                  f"</div>", unsafe_allow_html=True)
                        
                        st.markdown(f"<div style='background-color:#e6f3ff;padding:10px;border-radius:5px;'>"
                                  f"<b>Probability of Early Termination (PET₀):</b> {results.get('PET0')}"
                                  f"</div>", unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown("##### Overall Design Parameters")
                        st.markdown(f"<div style='background-color:#eff8e6;padding:10px;border-radius:5px;margin-bottom:5px;'>"
                                  f"<b>Total Sample Size (n):</b> {results.get('n')}"
                                  f"</div>", unsafe_allow_html=True)
                        
                        st.markdown(f"<div style='background-color:#eff8e6;padding:10px;border-radius:5px;margin-bottom:5px;'>"
                                  f"<b>Final Threshold (r):</b> {results.get('r')}"
                                  f"</div>", unsafe_allow_html=True)
                        
                        st.markdown(f"<div style='background-color:#eff8e6;padding:10px;border-radius:5px;'>"
                                  f"<b>Expected Sample Size (EN₀):</b> {results.get('EN0')}"
                                  f"</div>", unsafe_allow_html=True)
                
                with tab2:
                    # Create a more visually appealing layout for error rates
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("##### Type I Error (False Positive)")
                        target_alpha = params.get('alpha')
                        actual_alpha = results.get('actual_alpha')
                        
                        st.markdown(f"<div style='background-color:#fff0e6;padding:10px;border-radius:5px;margin-bottom:5px;'>"
                                  f"<b>Target Type I Error (α):</b> {target_alpha}"
                                  f"</div>", unsafe_allow_html=True)
                        
                        st.markdown(f"<div style='background-color:#fff0e6;padding:10px;border-radius:5px;'>"
                                  f"<b>Actual Type I Error:</b> {actual_alpha}"
                                  f"</div>", unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown("##### Power (1 - Type II Error)")
                        target_power = params.get('power', 1-params.get('beta', 0.2))
                        actual_power = results.get('actual_power')
                        
                        st.markdown(f"<div style='background-color:#e6e6ff;padding:10px;border-radius:5px;margin-bottom:5px;'>"
                                  f"<b>Target Power:</b> {target_power}"
                                  f"</div>", unsafe_allow_html=True)
                        
                        st.markdown(f"<div style='background-color:#e6e6ff;padding:10px;border-radius:5px;'>"
                                  f"<b>Actual Power:</b> {actual_power}"
                                  f"</div>", unsafe_allow_html=True)
                
                with tab3:
                    # Create a more visually appealing layout for effect size information
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("##### Response Rates")
                        p0 = params.get('p0')
                        p1 = params.get('p')
                        
                        st.markdown(f"<div style='background-color:#f0e6ff;padding:10px;border-radius:5px;margin-bottom:5px;'>"
                                  f"<b>Null Response Rate (p₀):</b> {p0}"
                                  f"</div>", unsafe_allow_html=True)
                        
                        st.markdown(f"<div style='background-color:#f0e6ff;padding:10px;border-radius:5px;'>"
                                  f"<b>Alternative Response Rate (p₁):</b> {p1}"
                                  f"</div>", unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown("##### Effect Measures")
                        risk_diff = results.get('absolute_risk_difference')
                        rel_risk = results.get('relative_risk')
                        
                        st.markdown(f"<div style='background-color:#ffe6e6;padding:10px;border-radius:5px;margin-bottom:5px;'>"
                                  f"<b>Absolute Risk Difference:</b> {risk_diff}"
                                  f"</div>", unsafe_allow_html=True)
                        
                        st.markdown(f"<div style='background-color:#ffe6e6;padding:10px;border-radius:5px;'>"
                                  f"<b>Relative Risk:</b> {rel_risk}"
                                  f"</div>", unsafe_allow_html=True)
                
                with tab4:
                    # Create a more visually appealing layout for decision rules
                    st.markdown("##### Trial Conduct Decision Rules")
                    
                    # Stage 1 decision rule
                    st.markdown(f"<div style='background-color:#e6fff0;padding:15px;border-radius:5px;margin-bottom:10px;'>"
                              f"<h5>Stage 1 Decision Rule:</h5>"
                              f"<ul>"
                              f"<li>Enroll {results.get('n1')} patients in the first stage</li>"
                              f"<li>Count the number of responses (r)</li>"
                              f"<li>If r ≤ {results.get('r1')}, <b>stop the trial</b> for futility</li>"
                              f"<li>If r > {results.get('r1')}, <b>continue</b> to the second stage</li>"
                              f"</ul>"
                              f"</div>", unsafe_allow_html=True)
                    
                    # Stage 2 decision rule
                    st.markdown(f"<div style='background-color:#e6fff0;padding:15px;border-radius:5px;'>"
                              f"<h5>Stage 2 Decision Rule:</h5>"
                              f"<ul>"
                              f"<li>Enroll additional {results.get('n') - results.get('n1')} patients</li>"
                              f"<li>Count the total number of responses (r) across both stages</li>"
                              f"<li>If total r > {results.get('r')}, <b>reject H₀</b> (treatment is effective)</li>"
                              f"<li>If total r ≤ {results.get('r')}, <b>accept H₀</b> (treatment is not effective)</li>"
                              f"</ul>"
                              f"</div>", unsafe_allow_html=True)
            
            else:
                # Enhanced display for standard results with consistent styling
                st.markdown("### Results Summary")
                st.markdown("---")
                
                # Group results into logical categories
                sample_size_keys = ["n", "n1", "n2", "total_n"]
                power_keys = ["power", "actual_power"]
                effect_size_keys = ["mde", "delta", "difference", "risk_difference", "hazard_ratio", "hr"]
                
                # Use an expander for detailed results
                with st.expander("📊 Detailed Results", expanded=True):
                    # Create a multi-column layout
                    if calc_type == "Sample Size":
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("##### Sample Size")
                            for k in sample_size_keys:
                                if k in results:
                                    # Format sample size with thousands separator
                                    value = results[k]
                                    if isinstance(value, (int, float)):
                                        formatted_value = f"{value:,}" if isinstance(value, int) else f"{value:.2f}"
                                    else:
                                        formatted_value = value
                                    
                                    st.markdown(f"<div style='background-color:#e6f3ff;padding:10px;border-radius:5px;margin-bottom:5px;'>"
                                             f"<b>{k.replace('_', ' ').title()}:</b> {formatted_value}"
                                             f"</div>", unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown("##### Error Rates & Parameters")
                            for k, v in results.items():
                                if k not in sample_size_keys and k not in ["design_method"]:
                                    formatted_value = v
                                    if isinstance(v, float):
                                        formatted_value = f"{v:.4f}"
                                    
                                    st.markdown(f"<div style='background-color:#eff8e6;padding:10px;border-radius:5px;margin-bottom:5px;'>"
                                             f"<b>{k.replace('_', ' ').title()}:</b> {formatted_value}"
                                             f"</div>", unsafe_allow_html=True)
                    
                    elif calc_type == "Power":
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("##### Power Analysis")
                            for k in power_keys:
                                if k in results:
                                    value = results[k]
                                    formatted_value = f"{value:.4f}" if isinstance(value, float) else value
                                    
                                    st.markdown(f"<div style='background-color:#e6e6ff;padding:10px;border-radius:5px;margin-bottom:5px;'>"
                                             f"<b>{k.replace('_', ' ').title()}:</b> {formatted_value}"
                                             f"</div>", unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown("##### Study Parameters")
                            for k, v in results.items():
                                if k not in power_keys and k not in ["design_method"]:
                                    formatted_value = v
                                    if isinstance(v, float):
                                        formatted_value = f"{v:.4f}"
                                    
                                    st.markdown(f"<div style='background-color:#fff0e6;padding:10px;border-radius:5px;margin-bottom:5px;'>"
                                             f"<b>{k.replace('_', ' ').title()}:</b> {formatted_value}"
                                             f"</div>", unsafe_allow_html=True)
                    
                    elif calc_type == "Minimum Detectable Effect":
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("##### Effect Size")
                            for k in effect_size_keys:
                                if k in results:
                                    value = results[k]
                                    formatted_value = f"{value:.4f}" if isinstance(value, float) else value
                                    
                                    st.markdown(f"<div style='background-color:#f0e6ff;padding:10px;border-radius:5px;margin-bottom:5px;'>"
                                             f"<b>{k.replace('_', ' ').title()}:</b> {formatted_value}"
                                             f"</div>", unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown("##### Study Parameters")
                            for k, v in results.items():
                                if k not in effect_size_keys and k not in ["design_method"]:
                                    formatted_value = v
                                    if isinstance(v, float):
                                        formatted_value = f"{v:.4f}"
                                    
                                    st.markdown(f"<div style='background-color:#ffe6e6;padding:10px;border-radius:5px;margin-bottom:5px;'>"
                                             f"<b>{k.replace('_', ' ').title()}:</b> {formatted_value}"
                                             f"</div>", unsafe_allow_html=True)
                    
                    else:
                        # Fallback for any other calculation type
                        for k, v in results.items():
                            if k != "design_method":
                                formatted_value = v
                                if isinstance(v, float):
                                    formatted_value = f"{v:.4f}"
                                
                                st.markdown(f"<div style='background-color:#f0f2f6;padding:10px;border-radius:5px;margin-bottom:5px;'>"
                                         f"<b>{k.replace('_', ' ').title()}:</b> {formatted_value}"
                                         f"</div>", unsafe_allow_html=True)
            
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
        
        # No visualization plots needed for Simon's two-stage design results
