#!/usr/bin/env python3
import streamlit as st

# Test different ways of constructing HTML
st.title("HTML Quote Test")

# Method 1: Current approach with nested quotes
html1 = f"""
<div style="background-color: #e6f3ff; padding: 20px;">
    <p style="font-size: 0.9em; color: #666;">
    <strong>Tip:</strong> Copy the text above directly into your grant application or study protocol.
    </p>
</div>
"""

st.subheader("Method 1: Current approach")
st.markdown(html1, unsafe_allow_html=True)

# Method 2: Using single quotes for attributes
html2 = f"""
<div style='background-color: #e6f3ff; padding: 20px;'>
    <p style='font-size: 0.9em; color: #666;'>
    <strong>Tip:</strong> Copy the text above directly into your grant application or study protocol.
    </p>
</div>
"""

st.subheader("Method 2: Single quotes")
st.markdown(html2, unsafe_allow_html=True)

# Method 3: Escaped quotes
html3 = """
<div style="background-color: #e6f3ff; padding: 20px;">
    <p style="font-size: 0.9em; color: #666;">
    <strong>Tip:</strong> Copy the text above directly into your grant application or study protocol.
    </p>
</div>
"""

st.subheader("Method 3: Raw string")
st.markdown(html3, unsafe_allow_html=True)

# Show what we're actually generating
st.subheader("Debug: Raw HTML being generated")
st.code(html1)