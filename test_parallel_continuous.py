#!/usr/bin/env python3
"""Test the parallel continuous report generation"""
import streamlit as st
from app.components.parallel_rct.calculations import calculate_parallel_continuous
from core.utils.report_generator import generate_report

st.set_page_config(page_title="Test Parallel Continuous", layout="wide")
st.title("Test Parallel RCT Continuous Report")

# Test parameters - NO repeated measures
params = {
    "calculation_type": "Sample Size",
    "hypothesis_type": "Superiority",
    "mean1": 0.0,
    "mean2": 0.5,
    "std_dev": 1.0,
    "power": 0.8,
    "alpha": 0.05,
    "allocation_ratio": 1.0,
    "method": "analytical",
    "repeated_measures": False,  # No repeated measures
    "unequal_var": False
}

# Calculate results
results = calculate_parallel_continuous(params)

st.subheader("Calculation Results")
st.json(results)

# Generate report
report = generate_report(
    design_type="Parallel RCT",
    outcome_type="Continuous Outcome",
    params=params,
    results=results
)

st.subheader("HTML Report Display")
st.markdown(report, unsafe_allow_html=True)

# Show raw HTML for debugging
with st.expander("View Raw HTML"):
    st.code(report, language='html')
    
# Check for problematic patterns
if "' + f'" in report or "f'{" in report:
    st.error("Found problematic f-string pattern in report!")
if "&lt;" in report or "&gt;" in report or "&quot;" in report:
    st.error("Found escaped HTML entities in report!")