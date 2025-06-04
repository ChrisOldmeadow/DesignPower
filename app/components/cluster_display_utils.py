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


def display_cluster_continuous_results(results, params, calc_type):
    """
    Display formatted results for Cluster RCT with continuous outcome.
    
    Args:
        results: Dictionary containing calculation results.
        params: Dictionary of input parameters.
        calc_type: String indicating calculation type (Sample Size, Power, or MDE).
    """
    st.markdown("## Cluster RCT: Continuous Outcome Results")
    st.markdown("---")

    # --- Key Metrics ---
    st.markdown("### Key Metrics")
    
    # Adjust number of columns based on calc_type for better layout
    if calc_type == "Sample Size":
        cols = st.columns(3)
        if params.get("determine_ss_param") == "Number of Clusters (k)":
            cols[0].metric("Required Clusters per Arm (k)", results.get("n_clusters", "N/A"))
            cols[1].metric("Input Average Cluster Size (m)", params.get("cluster_size_input_for_k_calc", "N/A"))
        elif params.get("determine_ss_param") == "Average Cluster Size (m)":
            cols[0].metric("Input Clusters per Arm (k)", params.get("n_clusters_input_for_m_calc", "N/A"))
            cols[1].metric("Required Average Cluster Size (m)", results.get("cluster_size", "N/A"))
        else: # Should not happen with current UI, but good to have a fallback
             cols[0].metric("Required Clusters per Arm (k)", results.get("n_clusters", "N/A"))
             cols[1].metric("Required Average Cluster Size (m)", results.get("cluster_size", "N/A"))

        cols[2].metric("Achieved Power", f"{results.get('achieved_power', 0.0):.3f}" if results.get('achieved_power') is not None else "N/A")
        st.metric("Total Sample Size (N)", results.get("total_n", "N/A"))

    elif calc_type == "Power":
        cols = st.columns(3)
        cols[0].metric("Achieved Power", f"{results.get('power', 0.0):.3f}" if results.get('power') is not None else "N/A")
        cols[1].metric("Input Clusters per Arm (k)", params.get("n_clusters", "N/A"))
        cols[2].metric("Input Average Cluster Size (m)", params.get("cluster_size", "N/A"))
        st.metric("Total Sample Size (N)", results.get("total_n", "N/A"))

    elif calc_type == "Minimum Detectable Effect":
        cols = st.columns(3)
        cols[0].metric("Minimum Detectable Effect (MDE)", f"{results.get('mde', 0.0):.3f}" if results.get('mde') is not None else "N/A")
        cols[1].metric("Input Clusters per Arm (k)", params.get("n_clusters", "N/A"))
        cols[2].metric("Input Average Cluster Size (m)", params.get("cluster_size", "N/A"))
        st.metric("Total Sample Size (N)", results.get("total_n", "N/A"))

    st.markdown("---")

    # --- LMM Fit Statistics (if applicable) ---
    if params.get("analysis_model") == "mixedlm" and "lmm_fit_stats" in results:
        st.markdown("### LMM Fit Statistics")
        stats = results["lmm_fit_stats"]
        
        total_sims_from_stats = sum(stats.values())
        # Prefer nsim from main results if available, fallback to sum from stats dict
        total_sims_run = results.get("nsim", total_sims_from_stats if isinstance(total_sims_from_stats, (int, float)) and total_sims_from_stats > 0 else params.get("nsim", 0) )

        if total_sims_run > 0:
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Total Simulations Run:** {total_sims_run}")
                st.write(f"- Successful Fits: {stats.get('success', 0)} ({stats.get('success', 0)/total_sims_run:.1%})")
                st.write(f"- Success (Convergence Warning): {stats.get('success_convergence_warnings', 0)} ({stats.get('success_convergence_warnings', 0)/total_sims_run:.1%})")
            with col2:
                st.write(f"- OLS Fallback (Boundary): {stats.get('success_boundary_ols_fallbacks', 0)} ({stats.get('success_boundary_ols_fallbacks', 0)/total_sims_run:.1%})")
                st.write(f"- OLS Fallback (Fit Error): {stats.get('ols_fallbacks_errors', 0)} ({stats.get('ols_fallbacks_errors', 0)/total_sims_run:.1%})")
                st.write(f"- Outer T-test Fallback (Error): {stats.get('ttest_fallbacks_outer_errors', 0)} ({stats.get('ttest_fallbacks_outer_errors', 0)/total_sims_run:.1%})")
            
            if "lmm_total_considered_for_power" in stats:
                st.caption(f"Empirical power for LMM based on {stats.get('lmm_total_considered_for_power', 0)} simulations "
                           f"(excludes outer t-test fallbacks and unknown statuses).")
        else:
            st.write("LMM fit statistics reported, but total simulations count is zero or unavailable.")
        st.markdown("---")

    # --- Other Information (Sensitivity, CV, ICC Conversion) ---
    # These functions will only display if relevant data is in results
    display_sensitivity_analysis(results, calc_type)
    display_cluster_variation_info(results)
    display_icc_conversion_info(results)

    # --- Expander for Full Details ---
    with st.expander("View Full Simulation Details & Input Parameters"):
        st.markdown("#### Input Parameters")
        # Filter params to avoid showing redundant/internal keys if any
        params_to_display = {k: v for k, v in params.items() if k not in ['calc_type', 'hypothesis_type', 'outcome_type', 'calculation_type']}
        st.json(params_to_display)
        
        st.markdown("#### Full Results Dictionary")
        # Filter out potentially large data for cleaner display in expander
        filtered_results_for_expander = {
            k: v for k, v in results.items() 
            if k not in ["sensitivity_analysis", "power_curve_data", "p_values_list", "lmm_fit_stats"] # lmm_fit_stats already shown
        }
        st.json(filtered_results_for_expander)

    # --- Generate Report Button ---
    st.markdown("---")
    if st.button("Generate Report", key="cluster_cont_report_button"):
        report_text = generate_report(
            results=results,
            params=params, # params should contain calc_type, hypothesis_type, method etc.
            design_type="Cluster RCT", # Hardcoded as this function is specific
            outcome_type="Continuous Outcome" # Hardcoded as this function is specific
        )
        # The report_text from generate_report is already pre-formatted text
        report_html_content = f"<pre>{report_text}</pre>"
        report_name = f"Cluster_RCT_Continuous_{calc_type.replace(' ', '_')}_report"
        
        if report_html_content:
            st.markdown("#### Generated Report:", unsafe_allow_html=True)
            st.markdown(report_html_content, unsafe_allow_html=True)
            b64 = base64.b64encode(report_html_content.encode()).decode()
            href = f'<a href="data:text/html;base64,{b64}" download="{report_name}.html">Download Report as HTML</a>'
            st.markdown(href, unsafe_allow_html=True)
        else:
            st.error("Could not generate report.")

