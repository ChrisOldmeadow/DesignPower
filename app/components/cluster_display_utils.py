"""Display utilities for cluster RCT results.

This module provides helper functions to display the results of
cluster randomized controlled trial calculations, including
sensitivity analysis visualizations.
"""
import streamlit as st
import pandas as pd
import base64
from core.utils.report_generator import generate_report


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


def display_cluster_continuous_results(results, params, calc_type, hypothesis_type):
    """
    Display results for Cluster RCT with continuous outcome.
    
    Args:
        results: Dictionary containing calculation results
        params: Dictionary containing input parameters
        calc_type: String indicating calculation type (Sample Size, Power, or Minimum Detectable Effect)
        hypothesis_type: String indicating hypothesis type (Superiority or Non-Inferiority)
    """
    if "error" in results:
        st.error(f"Error in calculation: {results['error']}")
        return
    
    # Display main results based on calculation type
    if calc_type == "Sample Size":
        st.markdown("### Sample Size Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            n_clusters = results.get("n_clusters", "N/A")
            st.metric("Clusters per Arm", n_clusters)
        
        with col2:
            cluster_size = results.get("cluster_size", params.get("cluster_size", "N/A"))
            st.metric("Cluster Size", cluster_size)
        
        with col3:
            total_n = results.get("total_n", "N/A")
            if total_n == "N/A" and n_clusters != "N/A" and cluster_size != "N/A":
                total_n = n_clusters * cluster_size * 2  # 2 arms
            st.metric("Total Sample Size", total_n)
            
    elif calc_type == "Power":
        st.markdown("### Power Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            power = results.get("power", "N/A")
            if isinstance(power, float):
                st.metric("Power", f"{power:.3f}")
            else:
                st.metric("Power", power)
        
        with col2:
            alpha = results.get("alpha", params.get("alpha", "N/A"))
            st.metric("Significance Level (α)", alpha)
            
    elif calc_type == "Minimum Detectable Effect":
        st.markdown("### Minimum Detectable Effect Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            mde = results.get("mde", results.get("min_detectable_effect", "N/A"))
            if isinstance(mde, float):
                st.metric("Minimum Detectable Effect", f"{mde:.3f}")
            else:
                st.metric("Minimum Detectable Effect", mde)
        
        with col2:
            effect_size = results.get("effect_size", "N/A")
            if isinstance(effect_size, float):
                st.metric("Effect Size", f"{effect_size:.3f}")
            else:
                st.metric("Effect Size", effect_size)
    
    # Display study parameters
    st.markdown("### Study Parameters")
    
    param_cols = st.columns(4)
    
    with param_cols[0]:
        mean1 = results.get("mean1", params.get("mean1", "N/A"))
        st.metric("Mean Group 1", mean1)
    
    with param_cols[1]:
        mean2 = results.get("mean2", params.get("mean2", "N/A"))
        st.metric("Mean Group 2", mean2)
    
    with param_cols[2]:
        std_dev = results.get("std_dev", params.get("std_dev", "N/A"))
        st.metric("Standard Deviation", std_dev)
    
    with param_cols[3]:
        icc = results.get("icc", params.get("icc", "N/A"))
        if isinstance(icc, float):
            st.metric("ICC", f"{icc:.3f}")
        else:
            st.metric("ICC", icc)
    
    # Display design effect if available
    if "design_effect" in results:
        st.markdown("### Design Effect")
        design_effect = results["design_effect"]
        st.info(f"""
        **Design Effect: {design_effect:.2f}**
        
        The design effect accounts for clustering in the data. A design effect of {design_effect:.2f} 
        means that {design_effect:.0f}x more participants are needed compared to an individually 
        randomized trial to achieve the same statistical power.
        """)
    
    # Display additional cluster-specific information
    display_cluster_variation_info(results)
    display_icc_conversion_info(results)
    
    # Display sensitivity analysis if available
    display_sensitivity_analysis(results, calc_type)


