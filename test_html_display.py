#!/usr/bin/env python3
import streamlit as st

# Test HTML display
html_content = """
<div style="font-family: Arial, sans-serif; line-height: 1.6; max-width: 800px;">
<h2 style="color: #2E86AB; border-bottom: 2px solid #2E86AB; padding-bottom: 10px;">
Test HTML Display
</h2>
<div style="background-color: #e6f3ff; padding: 20px; border-radius: 8px; border-left: 4px solid #0066cc; margin: 20px 0;">
    <h4 style="color: #0052a3; margin-top: 0; margin-bottom: 15px;">📝 Methodological Description</h4>
    <div style="background-color: white; padding: 15px; border-radius: 6px; border: 1px solid #cce7ff;">
        <p style="font-style: italic; line-height: 1.8; margin: 0; color: #333;">
        This is a test of the HTML display functionality. If you see this text properly formatted, 
        then the HTML is rendering correctly.
        </p>
    </div>
    <p style="font-size: 0.9em; color: #666; margin-top: 10px; margin-bottom: 0;">
    <strong>Tip:</strong> Copy the text above directly into your grant application or study protocol.
    </p>
</div>
</div>
"""

st.title("HTML Display Test")
st.markdown(html_content, unsafe_allow_html=True)

# Also test with raw display
st.subheader("Raw HTML (for comparison)")
st.code(html_content)