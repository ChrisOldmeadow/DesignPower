#!/usr/bin/env python3
import streamlit as st

st.title("Test References HTML Issue")

# Simulate the reference dictionary
reference = {
    'citation': 'Cohen J. (1988). Statistical Power Analysis for the Behavioral Sciences. New York, NY: Routledge Academic',
    'doi': 'https://doi.org/10.4324/9780203771587'
}

# Test 1: Current approach (might have issues)
html1 = f"""
<h3>Test 1: Current Approach</h3>
<div style="background-color: #f8f9fa; padding: 15px;">
    <p><strong>Primary Reference:</strong></p>
    <p style="font-style: italic; margin: 10px 0;">{reference['citation']}</p>
    <p><strong>DOI:</strong> <a href="{reference['doi']}" target="_blank" style="color: #2E86AB;">{reference['doi']}</a></p>
    
    <p style="margin-top: 15px;"><strong>Additional Key References:</strong></p>
    <ul style="margin-top: 10px;">
    <li>Chow SC, Shao J, Wang H, Lokhnygina Y. (2017). Sample Size Calculations in Clinical Research. 3rd Edition. CRC Press.</li>
    <li>Julious SA. (2010). Sample Sizes for Clinical Trials. CRC Press.</li>
    </ul>
</div>
"""

st.markdown(html1, unsafe_allow_html=True)

# Test 2: Without blank line
html2 = f"""
<h3>Test 2: Without Blank Line</h3>
<div style="background-color: #f8f9fa; padding: 15px;">
    <p><strong>Primary Reference:</strong></p>
    <p style="font-style: italic; margin: 10px 0;">{reference['citation']}</p>
    <p><strong>DOI:</strong> <a href="{reference['doi']}" target="_blank" style="color: #2E86AB;">{reference['doi']}</a></p>
    <p style="margin-top: 15px;"><strong>Additional Key References:</strong></p>
    <ul style="margin-top: 10px;">
    <li>Chow SC, Shao J, Wang H, Lokhnygina Y. (2017). Sample Size Calculations in Clinical Research. 3rd Edition. CRC Press.</li>
    <li>Julious SA. (2010). Sample Sizes for Clinical Trials. CRC Press.</li>
    </ul>
</div>
"""

st.markdown(html2, unsafe_allow_html=True)

# Test 3: Pre-extracted values
ref_citation = reference['citation']
ref_doi = reference['doi']

html3 = f"""
<h3>Test 3: Pre-extracted Values</h3>
<div style="background-color: #f8f9fa; padding: 15px;">
    <p><strong>Primary Reference:</strong></p>
    <p style="font-style: italic; margin: 10px 0;">{ref_citation}</p>
    <p><strong>DOI:</strong> <a href="{ref_doi}" target="_blank" style="color: #2E86AB;">{ref_doi}</a></p>
    <p style="margin-top: 15px;"><strong>Additional Key References:</strong></p>
    <ul style="margin-top: 10px;">
    <li>Chow SC, Shao J, Wang H, Lokhnygina Y. (2017). Sample Size Calculations in Clinical Research. 3rd Edition. CRC Press.</li>
    <li>Julious SA. (2010). Sample Sizes for Clinical Trials. CRC Press.</li>
    </ul>
</div>
"""

st.markdown(html3, unsafe_allow_html=True)