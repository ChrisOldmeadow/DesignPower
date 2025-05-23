"""Display utilities for cluster RCT results.

This module provides helper functions to display the results of
cluster randomized controlled trial calculations, including
sensitivity analysis visualizations.
"""
import streamlit as st
import pandas as pd


def display_sensitivity_analysis(results, calc_type):
    """
    Display sensitivity analysis results with a chart.
    
    Args:
        results: Dictionary containing sensitivity analysis results
        calc_type: String indicating calculation type (Sample Size, Power, or MDE)
    """
    if "sensitivity_analysis" not in results:
        return
    
    sens_data = results["sensitivity_analysis"]
    
    st.markdown("### ICC Sensitivity Analysis")
    st.write("This analysis shows how results vary across different ICC values.")
    
    # Create a dataframe for plotting
    icc_values = sens_data["icc_range"]
    sens_results = sens_data["results"]
    
    # Extract the relevant metric based on calculation type
    if calc_type == "Sample Size":
        metric = "n_clusters"
        y_label = "Number of Clusters per Arm"
        chart_title = "Effect of ICC on Required Number of Clusters"
    elif calc_type == "Power":
        metric = "power"
        y_label = "Power"
        chart_title = "Effect of ICC on Power"
    else:  # Minimum Detectable Effect
        metric = "mde"
        y_label = "Minimum Detectable Effect"
        chart_title = "Effect of ICC on Minimum Detectable Effect"
    
    # Create two dataframes: one for the primary metric and one for design effect
    metric_data = pd.DataFrame({
        "ICC": icc_values,
        y_label: [result[metric] for result in sens_results]
    })
    
    deff_data = pd.DataFrame({
        "ICC": icc_values,
        "Design Effect": [result["design_effect"] for result in sens_results]
    })
    
    # Display two charts side by side
    col1, col2 = st.columns(2)
    
    with col1:
        # Primary metric chart
        st.markdown(f"**{chart_title}**")
        chart = {
            "mark": "line",
            "encoding": {
                "x": {"field": "ICC", "type": "quantitative", "title": "ICC"},
                "y": {"field": y_label, "type": "quantitative", "title": y_label},
                "tooltip": [{"field": "ICC", "type": "quantitative", "format": ".3f"},
                           {"field": y_label, "type": "quantitative", "format": ".3f"}]
            },
            "data": metric_data,
            "width": 300,
            "height": 250
        }
        st.vega_lite_chart(metric_data, chart)
    
    with col2:
        # Design effect chart
        st.markdown("**Effect of ICC on Design Effect**")
        chart = {
            "mark": "line",
            "encoding": {
                "x": {"field": "ICC", "type": "quantitative", "title": "ICC"},
                "y": {"field": "Design Effect", "type": "quantitative", "title": "Design Effect"},
                "tooltip": [{"field": "ICC", "type": "quantitative", "format": ".3f"},
                           {"field": "Design Effect", "type": "quantitative", "format": ".3f"}]
            },
            "data": deff_data,
            "width": 300,
            "height": 250
        }
        st.vega_lite_chart(deff_data, chart)
    
    # Display the data as a table
    st.markdown("**Sensitivity Analysis Results**")
    
    # Create a table with all relevant metrics
    table_data = {"ICC": icc_values, "Design Effect": [result["design_effect"] for result in sens_results]}
    
    if calc_type == "Sample Size":
        table_data["Number of Clusters"] = [result["n_clusters"] for result in sens_results]
        table_data["Total Sample Size"] = [result["total_n"] for result in sens_results]
    elif calc_type == "Power":
        table_data["Power"] = [result["power"] for result in sens_results]
    else:  # Minimum Detectable Effect
        table_data["MDE"] = [result["mde"] for result in sens_results]
    
    # Convert to dataframe and display
    table_df = pd.DataFrame(table_data)
    st.dataframe(table_df)


def display_cluster_variation_info(results):
    """
    Display information about cluster size variation.
    
    Args:
        results: Dictionary containing cluster size variation information
    """
    if "cv_cluster_size" not in results or results["cv_cluster_size"] == 0:
        return
    
    st.markdown("### Cluster Size Variation")
    
    cv = results["cv_cluster_size"]
    deff = results["design_effect"]
    
    st.info(f"""
    **Coefficient of Variation (CV) = {cv:.2f}**
    
    This calculation accounts for variation in cluster sizes (CV = {cv:.2f}). 
    With unequal cluster sizes, the design effect is {deff:.2f}, which is larger 
    than it would be with equal-sized clusters. This inflation is incorporated into 
    all calculations.
    """)


def display_icc_conversion_info(results):
    """
    Display information about ICC scale conversion.
    
    Args:
        results: Dictionary containing ICC conversion information
    """
    if "icc_scale_original" not in results or results["icc_scale_original"] != "Logit":
        return
    
    st.markdown("### ICC Scale Conversion")
    
    icc_original = results["icc_original"]
    icc_converted = results["icc_converted"]
    
    st.info(f"""
    **ICC Conversion Applied**
    
    The ICC value was converted from logit scale ({icc_original:.4f}) to linear scale ({icc_converted:.4f}) 
    for calculations. This conversion depends on the control group proportion.
    """)
