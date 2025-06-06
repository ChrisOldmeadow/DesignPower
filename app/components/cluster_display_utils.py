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


import textwrap # For dedenting the script template

# Helper function to generate CLI code for Cluster RCT Continuous Outcome
def generate_cli_code_cluster_continuous(params):
    calc_type = params.get("calc_type")
    method = params.get("method", "analytical")
    determine_ss_param = params.get("determine_ss_param") # Specific to Sample Size

    # Prepare parameters for the script, using UI values as defaults
    script_params = {
        "calc_type": calc_type,
        "method": method,
        "mean1": params.get("mean1"),
        "mean2": params.get("mean2"),
        "std_dev": params.get("std_dev"),
        "icc": params.get("icc"),
        "alpha": params.get("alpha"),
    }

    if calc_type == "Sample Size":
        script_params["power"] = params.get("power")
        script_params["determine_ss_param"] = determine_ss_param
        if determine_ss_param == "Number of Clusters (k)":
            script_params["cluster_size_input"] = params.get("cluster_size_input_for_k_calc")
            script_params["n_clusters_input"] = None
        elif determine_ss_param == "Average Cluster Size (m)":
            script_params["cluster_size_input"] = None
            script_params["n_clusters_input"] = params.get("n_clusters_input_for_m_calc")
    elif calc_type == "Power":
        script_params["n_clusters_input"] = params.get("n_clusters")
        script_params["cluster_size_input"] = params.get("cluster_size")
        # mean2 is direct input for power calc
    elif calc_type == "Minimum Detectable Effect":
        script_params["n_clusters_input"] = params.get("n_clusters")
        script_params["cluster_size_input"] = params.get("cluster_size")
        script_params["power"] = params.get("power")
        # mean2 is calculated for MDE, mean1 is input

    if method == "simulation":
        script_params.update({
            "nsim": params.get("nsim", 1000),
            "seed": params.get("seed"), # Can be None
            "analysis_model": params.get("analysis_model", "ttest"),
            "use_satterthwaite": params.get("use_satterthwaite", False),
            "use_bias_correction": params.get("use_bias_correction", False),
            "bayes_draws": params.get("bayes_draws", 500),
            "bayes_warmup": params.get("bayes_warmup", 500),
            "lmm_method": params.get("lmm_method", "auto"),
            "lmm_reml": params.get("lmm_reml", True),
            "lmm_cov_penalty_weight": params.get("lmm_cov_penalty_weight", 0.0),
        })

    # Use a dictionary for cli_defaults to handle None values correctly in f-string
    cli_defaults = script_params.copy()

    script_template = f"""
import argparse
import json
import sys

# Attempt to import DesignPower modules
try:
    from core.designs.cluster_rct import analytical_continuous, simulation_continuous
except ImportError:
    print("Error: Could not import DesignPower modules. "
          "Ensure DesignPower is installed and accessible, or run from project root.", file=sys.stderr)
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Reproducible CLI for Cluster RCT Continuous Outcome Calculations")

    parser.add_argument("--calc_type", type=str, required=True, choices=["Sample Size", "Power", "Minimum Detectable Effect"], default='{cli_defaults.get('calc_type')}', help="Type of calculation")
    parser.add_argument("--method", type=str, required=True, choices=["analytical", "simulation"], default='{cli_defaults.get('method')}', help="Calculation method")
    
    parser.add_argument("--mean1", type=float, required=True, default={cli_defaults.get('mean1')}, help="Mean in group 1")
    parser.add_argument("--mean2", type=float, default={cli_defaults.get('mean2')}, help="Mean in group 2 (required for Sample Size/Power, not for MDE)")
    parser.add_argument("--std_dev", type=float, required=True, default={cli_defaults.get('std_dev')}, help="Standard deviation")
    parser.add_argument("--icc", type=float, required=True, default={cli_defaults.get('icc')}, help="Intraclass Correlation Coefficient")
    parser.add_argument("--alpha", type=float, default={cli_defaults.get('alpha', 0.05)}, help="Significance level (alpha)")

    # Parameters for Sample Size
    parser.add_argument("--power", type=float, default={cli_defaults.get('power', 0.8)}, help="Desired power (for Sample Size or MDE)")
    parser.add_argument("--determine_ss_param", type=str, choices=["Number of Clusters (k)", "Average Cluster Size (m)"], default='{cli_defaults.get('determine_ss_param')}', help="Parameter to determine for Sample Size")
    
    # Parameters for Power/MDE, or fixed inputs for Sample Size
    parser.add_argument("--n_clusters_input", type=int, default={cli_defaults.get('n_clusters_input')}, help="Number of clusters per arm (input for Power/MDE, or fixed for SS if solving for m)")
    parser.add_argument("--cluster_size_input", type=int, default={cli_defaults.get('cluster_size_input')}, help="Average cluster size (input for Power/MDE, or fixed for SS if solving for k)")

    # Simulation specific parameters
    parser.add_argument("--nsim", type=int, default={cli_defaults.get('nsim', 1000)}, help="Number of simulations")
    parser.add_argument("--seed", type=int, default={cli_defaults.get('seed')}, help="Random seed for simulations (optional)")
    parser.add_argument("--analysis_model", type=str, default='{cli_defaults.get('analysis_model', 'ttest')}', choices=["ttest", "mixedlm", "bayes"], help="Analysis model for simulation")
    parser.add_argument("--use_satterthwaite", type=lambda x: (str(x).lower() == 'true'), default={cli_defaults.get('use_satterthwaite', False)}, help="Use Satterthwaite approximation (simulation ttest)")
    parser.add_argument("--use_bias_correction", type=lambda x: (str(x).lower() == 'true'), default={cli_defaults.get('use_bias_correction', False)}, help="Use bias correction (simulation ttest)")
    parser.add_argument("--bayes_draws", type=int, default={cli_defaults.get('bayes_draws', 500)}, help="Bayesian draws")
    parser.add_argument("--bayes_warmup", type=int, default={cli_defaults.get('bayes_warmup', 500)}, help="Bayesian warmup draws")
    parser.add_argument("--lmm_method", type=str, default='{cli_defaults.get('lmm_method', 'auto')}', help="LMM fitting method")
    parser.add_argument("--lmm_reml", type=lambda x: (str(x).lower() == 'true'), default={cli_defaults.get('lmm_reml', True)}, help="Use REML for LMM")
    parser.add_argument("--lmm_cov_penalty_weight", type=float, default={cli_defaults.get('lmm_cov_penalty_weight', 0.0)}, help="LMM covariance penalty weight")

    args = parser.parse_args()
    results = {{}}

    try:
        if args.calc_type == "Sample Size":
            if args.determine_ss_param == "Number of Clusters (k)" and args.cluster_size_input is None:
                parser.error("--cluster_size_input is required when --determine_ss_param is 'Number of Clusters (k)'")
            if args.determine_ss_param == "Average Cluster Size (m)" and args.n_clusters_input is None:
                parser.error("--n_clusters_input is required when --determine_ss_param is 'Average Cluster Size (m)'")
            if args.mean2 is None:
                 parser.error("--mean2 is required for Sample Size calculation.")

            cs_arg = args.cluster_size_input if args.determine_ss_param == "Number of Clusters (k)" else None
            ncf_arg = args.n_clusters_input if args.determine_ss_param == "Average Cluster Size (m)" else None
            
            if args.method == "analytical":
                results = analytical_continuous.sample_size_continuous(
                    mean1=args.mean1, mean2=args.mean2, std_dev=args.std_dev, icc=args.icc,
                    cluster_size=cs_arg, n_clusters_fixed=ncf_arg,
                    power=args.power, alpha=args.alpha
                )
            else: # simulation
                results = simulation_continuous.sample_size_continuous_sim(
                    mean1=args.mean1, mean2=args.mean2, std_dev=args.std_dev, icc=args.icc,
                    cluster_size=cs_arg, n_clusters_fixed=ncf_arg,
                    power=args.power, alpha=args.alpha, nsim=args.nsim, seed=args.seed,
                    analysis_model=args.analysis_model, use_satterthwaite=args.use_satterthwaite,
                    use_bias_correction=args.use_bias_correction, bayes_draws=args.bayes_draws,
                    bayes_warmup=args.bayes_warmup, lmm_method=args.lmm_method, lmm_reml=args.lmm_reml,
                    lmm_cov_penalty_weight=args.lmm_cov_penalty_weight
                )
        elif args.calc_type == "Power":
            if args.n_clusters_input is None or args.cluster_size_input is None or args.mean2 is None:
                parser.error("--n_clusters_input, --cluster_size_input and --mean2 are required for Power calculation.")
            if args.method == "analytical":
                results = analytical_continuous.power_continuous(
                    n_clusters=args.n_clusters_input, cluster_size=args.cluster_size_input, icc=args.icc,
                    mean1=args.mean1, mean2=args.mean2, std_dev=args.std_dev, alpha=args.alpha
                )
            else: # simulation
                results = simulation_continuous.power_continuous_sim(
                    n_clusters=args.n_clusters_input, cluster_size=args.cluster_size_input, icc=args.icc,
                    mean1=args.mean1, mean2=args.mean2, std_dev=args.std_dev, alpha=args.alpha,
                    nsim=args.nsim, seed=args.seed, analysis_model=args.analysis_model,
                    use_satterthwaite=args.use_satterthwaite, use_bias_correction=args.use_bias_correction,
                    bayes_draws=args.bayes_draws, bayes_warmup=args.bayes_warmup,
                    lmm_method=args.lmm_method, lmm_reml=args.lmm_reml,
                    lmm_cov_penalty_weight=args.lmm_cov_penalty_weight
                )
        elif args.calc_type == "Minimum Detectable Effect":
            if args.n_clusters_input is None or args.cluster_size_input is None:
                parser.error("--n_clusters_input and --cluster_size_input are required for MDE calculation.")
            if args.method == "analytical":
                results = analytical_continuous.mde_continuous(
                    n_clusters=args.n_clusters_input, cluster_size=args.cluster_size_input, icc=args.icc,
                    mean1=args.mean1, std_dev=args.std_dev, power=args.power, alpha=args.alpha
                )
            else: # simulation
                results = simulation_continuous.mde_continuous_sim(
                    n_clusters=args.n_clusters_input, cluster_size=args.cluster_size_input, icc=args.icc,
                    mean1=args.mean1, std_dev=args.std_dev, power=args.power, alpha=args.alpha,
                    nsim=args.nsim, seed=args.seed, analysis_model=args.analysis_model,
                    use_satterthwaite=args.use_satterthwaite, use_bias_correction=args.use_bias_correction,
                    bayes_draws=args.bayes_draws, bayes_warmup=args.bayes_warmup,
                    lmm_method=args.lmm_method, lmm_reml=args.lmm_reml,
                    lmm_cov_penalty_weight=args.lmm_cov_penalty_weight
                )
        
        print(json.dumps(results, indent=4))

    except Exception as e:
        print(f"Error during calculation: {{e}}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
"""
    return textwrap.dedent(script_template)


def display_cluster_continuous_results(results, params, calc_type, hypothesis_type="Superiority"):
    """
    Display formatted results for Cluster RCT with continuous outcome.
    
    Args:
        results: Dictionary containing calculation results.
        params: Dictionary of input parameters.
        calc_type: String indicating calculation type (Sample Size, Power, or MDE).
        hypothesis_type: String indicating hypothesis type (Superiority, Non-Inferiority).
    """
    # --- Key Metrics ---
    
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

    # Show non-inferiority margin as an additional metric if applicable
    if hypothesis_type == "Non-Inferiority":
        nim_value = params.get('non_inferiority_margin')
        if nim_value is not None:
            st.metric("Non-Inferiority Margin", f"{nim_value:.3f}")
    
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
                st.write(f"- Successful Fits: {stats.get('successful_fits', 0)} ({stats.get('successful_fits', 0)/total_sims_run:.1%})")
                st.write(f"- Success (Convergence Warning): {stats.get('convergence_warnings', 0)} ({stats.get('convergence_warnings', 0)/total_sims_run:.1%})")
            with col2:
                st.write(f"- OLS Fallback (Boundary): {stats.get('success_boundary_ols_fallbacks', 0)} ({stats.get('success_boundary_ols_fallbacks', 0)/total_sims_run:.1%})")
                st.write(f"- OLS Fallback (Fit Error): {stats.get('ols_fallbacks_errors', 0)} ({stats.get('ols_fallbacks_errors', 0)/total_sims_run:.1%})")
                st.write(f"- Outer T-test Fallback (Error): {stats.get('ttest_fallbacks_outer_errors', 0)} ({stats.get('ttest_fallbacks_outer_errors', 0)/total_sims_run:.1%})")
            
            if "lmm_total_considered_for_power" in stats:
                st.caption(f"Empirical power for LMM based on {stats.get('lmm_total_considered_for_power', 0)} simulations "
                           f"(excludes outer t-test fallbacks and unknown statuses).")
                
                # Warning for high fallback rates
                successful_lmm = stats.get('successful_fits', 0) + stats.get('convergence_warnings', 0)
                ols_fallbacks = stats.get('success_boundary_ols_fallbacks', 0)
                total_valid = stats.get('lmm_total_considered_for_power', 1)
                
                lmm_success_rate = successful_lmm / total_valid
                ols_fallback_rate = ols_fallbacks / total_valid
                
                if lmm_success_rate < 0.2 and ols_fallback_rate > 0.7:
                    st.warning(
                        f"⚠️ **Low Mixed Model Success Rate ({lmm_success_rate:.1%})**\n\n"
                        f"Most simulations ({ols_fallback_rate:.1%}) fell back to cluster-robust OLS due to boundary conditions "
                        f"(very small cluster variance estimates). This suggests:\n\n"
                        f"• **Clustering effects are minimal** in your scenario\n"
                        f"• **Results are valid** but based on cluster-level analysis rather than true mixed models\n"
                        f"• **Consider**: \n"
                        f"  - Increasing cluster size or number of clusters\n"
                        f"  - Using 'T-test on Aggregate Data' method (designed for few clusters)\n"
                        f"  - Using 'Bayesian (Stan)' method (handles small cluster counts better than LMM)\n\n"
                        f"The power calculation is still accurate but reflects cluster-level analysis."
                    )
        else:
            st.write("LMM fit statistics reported, but total simulations count is zero or unavailable.")
        st.markdown("---")

    # --- Other Information (Sensitivity, CV, ICC Conversion) ---
    # These functions will only display if relevant data is in results
    display_sensitivity_analysis(results, calc_type)
    display_cluster_variation_info(results)
    display_icc_conversion_info(results)

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

            # Add button for reproducible CLI code
            st.markdown("---") # Separator
            st.markdown("#### Reproducible CLI Code")
            
            # Generate a unique key suffix based on dynamic parts of params to avoid conflicts
            button_key_suffix_list = []
            if params.get('calc_type'): button_key_suffix_list.append(params['calc_type'].replace(' ', '_').lower())
            if params.get('method'): button_key_suffix_list.append(params['method'].lower())
            # Add more params if needed to ensure uniqueness, e.g., a timestamp or random part if function called many times
            button_key_suffix = "_" + "_".join(filter(None, button_key_suffix_list)) if button_key_suffix_list else "_default"

            if st.button("Generate Reproducible CLI Code", key=f"generate_cli_code_cluster_continuous{button_key_suffix}"):
                try:
                    cli_script_string = generate_cli_code_cluster_continuous(params) # Call the new function
                    st.code(cli_script_string, language='python')
                    
                    script_name = f"run_cluster_rct_continuous_{params.get('calc_type', 'calc').replace(' ', '_')}.py"
                    st.download_button(
                        label="Download CLI Script",
                        data=cli_script_string,
                        file_name=script_name,
                        mime="text/x-python",
                        key=f"download_cli_cluster_continuous{button_key_suffix}"
                    )
                except Exception as e:
                    st.error(f"Error generating CLI code: {e}")

