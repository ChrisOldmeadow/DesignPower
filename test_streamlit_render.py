#!/usr/bin/env python3
import streamlit as st
from core.utils.report_generator import generate_sample_size_report

st.title("Test Report Rendering")

# Test parameters
results = {
    'n1': 50,
    'n2': 50,
    'effect_size': 0.5
}

params = {
    'mean1': 0,
    'mean2': 0.5,
    'std_dev': 1.0,
    'power': 0.8,
    'alpha': 0.05,
    'hypothesis_type': 'Superiority',
    'method': 'analytical',
    'repeated_measures': True,
    'correlation': 0.7
}

# Generate report
report = generate_sample_size_report(results, params, 'Parallel RCT', 'Continuous Outcome')

# Display using markdown
st.markdown(report, unsafe_allow_html=True)

# Also show raw HTML for debugging
with st.expander("View Raw HTML"):
    st.code(report, language='html')