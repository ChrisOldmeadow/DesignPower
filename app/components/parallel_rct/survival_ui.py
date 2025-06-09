"""Survival outcome UI components for Parallel RCT designs.

This module provides UI rendering functions for survival outcomes
in Parallel RCT designs.
"""
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


def render_parallel_survival(calc_type, hypothesis_type):
    """
    Render the UI for Parallel RCT with survival outcome.
    
    Args:
        calc_type: String indicating calculation type (Sample Size, Power, or Minimum Detectable Effect)
        hypothesis_type: String indicating hypothesis type (Superiority or Non-Inferiority)
        
    Returns:
        dict: Parameters collected from the UI
    """
    params = {}
    
    # Display header with calculation type
    st.write(f"### Parallel RCT with Survival Outcome ({calc_type} - {hypothesis_type})")
    
    # Basic parameters UI
    with st.container():
        st.subheader("Basic Parameters")
        
        # Parameter inputs
        col1, col2 = st.columns(2)
        
        with col1:
            if hypothesis_type == "Superiority":
                params["hr"] = st.number_input("Hazard Ratio (Treatment vs Control)", value=0.7, step=0.05, min_value=0.1, key="hr_input_survival_sup")
                params["median_survival1"] = st.number_input("Median Survival (Control Group, months)", 
                                                          value=12.0, step=1.0, min_value=0.1,
                                                          key="median1_input_survival_sup")
            elif hypothesis_type == "Non-Inferiority":
                params["median_survival1"] = st.number_input("Median Survival (Control Group, months)", 
                                                          value=12.0, step=1.0, min_value=0.1,
                                                          key="median1_input_survival_ni")
                params["non_inferiority_margin_hr"] = st.number_input(
                    "Non-Inferiority Margin (Hazard Ratio)", 
                    value=1.3, step=0.05, min_value=1.01, 
                    help="Upper acceptable limit for HR (Treatment/Control). Must be > 1.",
                    key="nim_hr_input_survival_ni"
                )
                params["assumed_true_hr"] = st.number_input(
                    "Assumed True Hazard Ratio (Treatment/Control)", 
                    value=1.0, step=0.05, min_value=0.1, 
                    help="Expected HR under the alternative hypothesis (e.g., 1.0 for true equivalence).",
                    key="assumed_hr_input_survival_ni"
                )
                if calc_type == "Minimum Detectable Effect":
                    st.warning("Minimum Detectable Effect is typically not the primary calculation for non-inferiority designs with a pre-specified margin.")

        with col2:
            params["alpha"] = st.slider("Significance Level (α)", 
                                     min_value=0.01, max_value=0.20, value=0.05, step=0.01,
                                     key="alpha_slider_survival")
                                     
            if calc_type != "Power":
                params["power"] = st.slider("Power (1-β)", 
                                         min_value=0.70, max_value=0.99, value=0.80, step=0.01,
                                         key="power_slider_survival")
            else:
                params["power"] = None # Default for power calculation, not shown
                params["n1"] = st.number_input("Sample Size (Group 1)", value=50, step=1, min_value=5, key="n1_input_survival_power")
                params["n2"] = st.number_input("Sample Size (Group 2)", value=50, step=1, min_value=5, key="n2_input_survival_power")
                
            # Allocation ratio is always relevant
            if calc_type == "Sample Size":
                 params["allocation_ratio"] = st.slider("Allocation Ratio (n2/n1)", 
                                                   min_value=0.1, max_value=10.0, value=1.0, step=0.1,
                                                   key="allocation_slider_survival_ss")
            elif 'n1' in params and 'n2' in params and params['n1'] > 0: # For Power and MDE if n1, n2 are provided
                params["allocation_ratio"] = params['n2'] / params['n1']
                st.write(f"Allocation Ratio (n2/n1): {params['allocation_ratio']:.2f}") 
            else: # Default if not sample size and n1/n2 not set (e.g. MDE without n1/n2 yet)
                params["allocation_ratio"] = 1.0 # Default, might be overridden if n1/n2 are set for MDE

    # Advanced parameters UI
    with st.expander("Advanced Parameters", expanded=False):
        col_adv1, col_adv2 = st.columns(2)
        with col_adv1:
            params["accrual_time"] = st.number_input("Enrollment Period (months)", 
                                                 value=12.0, step=1.0, min_value=0.1,
                                                 key="accrual_input_survival")
            if hypothesis_type == "Superiority": # Only show sides for superiority
                sides_options = ["Two-sided", "One-sided"]
                selected_sides = st.radio(
                    "Test Type", 
                    options=sides_options, 
                    index=0, 
                    key="sides_radio_survival_sup",
                    horizontal=True
                )
                params["sides"] = 1 if selected_sides == "One-sided" else 2
            # For Non-Inferiority, sides is implicitly 1, handled in calculation function
            
        with col_adv2:
            params["follow_up_time"] = st.number_input("Follow-up Period (months)", 
                                                    value=24.0, step=1.0, min_value=0.1,
                                                    key="followup_input_survival")
            
        params["dropout_rate"] = st.slider("Overall Dropout Rate", 
                                        min_value=0.0, max_value=0.5, value=0.1, step=0.01,
                                        key="dropout_slider_survival")

        method_options = ["Analytical", "Simulation"]
        selected_method = st.selectbox(
            "Calculation Method",
            options=method_options,
            index=method_options.index(params.get("method", "Analytical")), 
            key="method_survival_selectbox"
        )
        params["method"] = selected_method

        if params["method"] == "Simulation":
            col_sim_basic1, col_sim_basic2 = st.columns(2)
            with col_sim_basic1:
                params["nsim"] = st.number_input("Number of Simulations", value=1000, min_value=100, step=100, key="nsim_survival_input")
            with col_sim_basic2:
                params["seed"] = st.number_input(
                    "Random Seed", 
                    value=42, 
                    min_value=1, 
                    help="Seed for random number generation (for reproducibility)",
                    key="seed_survival_input"
                )

            if calc_type == "Sample Size":
                st.write("Simulation Parameters for Sample Size Optimization:")
                sim_col1, sim_col2, sim_col3 = st.columns(3)
                with sim_col1:
                    params["min_n_sim"] = st.number_input("Min N (per group)", value=10, min_value=5, key="min_n_sim_survival")
                with sim_col2:
                    params["max_n_sim"] = st.number_input("Max N (per group)", value=500, min_value=10, key="max_n_sim_survival")
                with sim_col3:
                    params["step_n_sim"] = st.number_input("Step N", value=5, min_value=1, key="step_n_sim_survival")
            if calc_type == "Minimum Detectable Effect" and hypothesis_type == "Non-Inferiority":
                 # MDE for NI simulation might need specific parameters if we decide to support it meaningfully
                 pass # Placeholder for now
        
    params["calculation_type"] = calc_type
    params["hypothesis_type"] = hypothesis_type

    # Default n1, n2 for MDE if not provided in power section
    if calc_type == "Minimum Detectable Effect":
        if "n1" not in params:
            params["n1"] = st.number_input("Sample Size (Group 1)", value=50, step=1, min_value=5, key="n1_input_survival_mde")
        if "n2" not in params:
            params["n2"] = st.number_input("Sample Size (Group 2)", value=50, step=1, min_value=5, key="n2_input_survival_mde")
        if params['n1'] > 0 : # Recalculate allocation ratio if n1, n2 are set for MDE
            params["allocation_ratio"] = params['n2'] / params['n1']
            # st.write(f"Allocation Ratio (n2/n1): {params['allocation_ratio']:.2f}") # Display if needed

    return params


def display_survival_results(result, calculation_type, hypothesis_type, use_simulation):
    """Display formatted results for survival outcome calculations."""
    st.subheader("Results")

    if not result:
        st.error("No results to display.")
        return

    # Display based on calculation type
    if calculation_type == "Sample Size":
        st.write(f"**Required Sample Size (Total):** {result.get('total_n', 'N/A')}")
        st.write(f"- Group 1: {result.get('n1', 'N/A')}")
        st.write(f"- Group 2: {result.get('n2', 'N/A')}")
        st.write(f"**Expected Number of Events:** {result.get('events', 'N/A')}")
        df_data = {
            "Parameter": ["Hazard Ratio", "Median Survival (Group 1)", "Median Survival (Group 2)", "Power", "Alpha"],
            "Value": [
                result.get('hr_for_display', 'N/A'), 
                result.get('median_survival1_param', 'N/A'), 
                result.get('median_survival2_derived', 'N/A'),
                result.get('power_param', 'N/A'), # Power used as input
                result.get('alpha_param', 'N/A')  # Alpha used as input
            ]
        }

    elif calculation_type == "Power":
        power_val = result.get('power', 'N/A')
        if isinstance(power_val, (int, float)):
            st.write(f"**Calculated Power:** {power_val:.3f}")
        else:
            st.write(f"**Calculated Power:** {power_val}")
        st.write(f"**Expected Number of Events:** {result.get('events', 'N/A')}")
        df_data = {
            "Parameter": ["Hazard Ratio", "Median Survival (Group 1)", "Median Survival (Group 2)", 
                          "Sample Size (Group 1)", "Sample Size (Group 2)", "Total Sample Size", "Alpha"],
            "Value": [
                result.get('hr_for_display', 'N/A'), 
                result.get('median_survival1_param', 'N/A'), 
                result.get('median_survival2_derived', 'N/A'),
                result.get('n1_param', 'N/A'),
                result.get('n2_param', 'N/A'),
                result.get('total_n', 'N/A'),
                result.get('alpha_param', 'N/A') # Alpha used as input
            ]
        }

    elif calculation_type == "Minimum Detectable Effect":
        mde_val = result.get('mde', 'N/A')
        if isinstance(mde_val, (int, float)):
            st.write(f"**Minimum Detectable Hazard Ratio:** {mde_val:.3f}")
        else:
            st.write(f"**Minimum Detectable Hazard Ratio:** {mde_val}")
        st.write(f"**Expected Number of Events:** {result.get('events', 'N/A')}")
        df_data = {
            "Parameter": ["Median Survival (Group 1)", "Sample Size (Group 1)", 
                          "Sample Size (Group 2)", "Total Sample Size", "Power", "Alpha"],
            "Value": [
                result.get('median_survival1_param', 'N/A'),
                result.get('n1_param', 'N/A'),
                result.get('n2_param', 'N/A'),
                result.get('total_n', 'N/A'),
                result.get('power_param', 'N/A'), # Power used as input
                result.get('alpha_param', 'N/A')  # Alpha used as input
            ]
        }
    else:
        st.error("Invalid calculation type for display.")
        return

    # Display parameters in a table
    df = pd.DataFrame(df_data)
    st.table(df.set_index('Parameter'))

    if use_simulation:
        st.info("Results obtained using simulation.")

    # Display non-inferiority specific information
    if hypothesis_type == "Non-Inferiority":
        st.markdown("#### Non-Inferiority Interpretation")
        if calculation_type == "Sample Size":
            st.write(f"Sample size calculated to test if the upper bound of the confidence interval for the hazard ratio is below the margin of {result.get('non_inferiority_margin_hr_param', 'N/A')}, assuming a true hazard ratio of {result.get('assumed_true_hr_param', 'N/A')}.")
        elif calculation_type == "Power":
            st.write(f"Power to detect non-inferiority, defined as the upper bound of the confidence interval for the hazard ratio being below the margin of {result.get('non_inferiority_margin_hr_param', 'N/A')}, given an assumed true hazard ratio of {result.get('assumed_true_hr_param', 'N/A')}.")
        # MDE is typically not the primary focus for NI, but if calculated as HR, context is similar.


def create_survival_visualization(result, calculation_type, hypothesis_type):
    """Create visualization for survival outcome results."""
    st.subheader("Visualizations")

    if not result:
        st.warning("No data available for visualization.")
        return

    fig, ax = plt.subplots()

    if calculation_type == "Sample Size" and 'power_curve_data' in result and result['power_curve_data']:
        df_power_curve = pd.DataFrame(result['power_curve_data'])
        if not df_power_curve.empty:
            ax.plot(df_power_curve['total_n'], df_power_curve['power'], marker='o')
            ax.set_xlabel("Total Sample Size")
            ax.set_ylabel("Power")
            ax.set_title("Power vs. Sample Size")
            ax.grid(True)
            st.pyplot(fig)

            if 'events' in df_power_curve.columns:
                fig_events, ax_events = plt.subplots()
                ax_events.plot(df_power_curve['total_n'], df_power_curve['events'], marker='o', color='green')
                ax_events.set_xlabel("Total Sample Size")
                ax_events.set_ylabel("Number of Events")
                ax_events.set_title("Events vs. Sample Size")
                ax_events.grid(True)
                st.pyplot(fig_events)
        else:
            st.info("No power curve data available for visualization.")

    elif calculation_type == "Power" and 'survival_curves' in result and result['survival_curves']:
        df_curves = pd.DataFrame(result['survival_curves'])
        if not df_curves.empty:
            ax.plot(df_curves['time'], df_curves['survival_group1'], label='Group 1')
            ax.plot(df_curves['time'], df_curves['survival_group2'], label='Group 2')
            ax.set_xlabel("Time")
            ax.set_ylabel("Survival Probability")
            ax.set_title("Survival Curves")
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)
        else:
            st.info("No survival curve data available for visualization.")
            
    elif calculation_type == "Minimum Detectable Effect" and 'power_vs_hr_data' in result and result['power_vs_hr_data']:
        df_mde_curve = pd.DataFrame(result['power_vs_hr_data'])
        if not df_mde_curve.empty:
            ax.plot(df_mde_curve['hr'], df_mde_curve['power'], marker='o')
            ax.set_xlabel("Hazard Ratio")
            ax.set_ylabel("Power")
            ax.set_title("Power vs. Hazard Ratio")
            ax.grid(True)
            st.pyplot(fig)
        else:
            st.info("No MDE curve data available for visualization.")
    else:
        st.info("Visualization for this combination of calculation type and results is not currently available.")