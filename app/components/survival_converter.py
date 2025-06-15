"""
Survival Parameter Converter Component for Streamlit Dashboard.

This component provides a user-friendly interface for converting between different
survival analysis parameters including median survival, hazard rates, survival
fractions, and event rates.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from core.utils.survival_converters import (
    convert_survival_parameters,
    convert_hazard_ratio_scenario,
    convert_time_units
)


def survival_converter_page():
    """Main survival parameter converter page."""
    
    st.title("🔄 Survival Parameter Converter")
    st.markdown("""
    Convert between different survival analysis parameters for clinical trial design.
    Provide any parameter to get all equivalent values.
    """)
    
    # Create tabs for different conversion types
    tab1, tab2, tab3, tab4 = st.tabs([
        "Single Parameter", "Hazard Ratio Scenario", "Unit Conversion", "Examples"
    ])
    
    with tab1:
        _single_parameter_converter()
    
    with tab2:
        _hazard_ratio_converter()
    
    with tab3:
        _unit_converter()
    
    with tab4:
        _examples_and_help()


def _single_parameter_converter():
    """Single parameter conversion interface."""
    
    st.header("Convert Single Parameter")
    st.markdown("Provide any one survival parameter to get all equivalent parameters.")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Input Parameters")
        
        # Parameter selection
        param_type = st.selectbox(
            "Parameter Type",
            ["Median Survival", "Hazard Rate", "Survival Fraction", "Event Rate"],
            help="Choose which parameter you know"
        )
        
        # Time settings
        time_unit = st.selectbox("Time Unit", ["months", "years", "weeks", "days"])
        
        if param_type in ["Survival Fraction", "Event Rate"]:
            time_point = st.number_input(
                f"Time Point ({time_unit})",
                min_value=0.1,
                value=12.0 if time_unit == "months" else 1.0,
                step=0.1,
                help="Time at which survival fraction/event rate is measured"
            )
        else:
            time_point = 12.0 if time_unit == "months" else 1.0
        
        # Parameter value input
        if param_type == "Median Survival":
            value = st.number_input(
                f"Median Survival ({time_unit})",
                min_value=0.1,
                value=12.0 if time_unit == "months" else 1.0,
                step=0.1
            )
            kwargs = {"median_survival": value}
            
        elif param_type == "Hazard Rate":
            value = st.number_input(
                f"Hazard Rate (per {time_unit})",
                min_value=0.001,
                value=0.058 if time_unit == "months" else 0.693,
                step=0.001,
                format="%.4f"
            )
            kwargs = {"hazard_rate": value}
            
        elif param_type == "Survival Fraction":
            value = st.slider(
                "Survival Fraction",
                min_value=0.01,
                max_value=0.99,
                value=0.70,
                step=0.01,
                format="%.2f"
            )
            kwargs = {"survival_fraction": value}
            
        else:  # Event Rate
            value = st.slider(
                "Event Rate",
                min_value=0.01,
                max_value=0.99,
                value=0.30,
                step=0.01,
                format="%.2f"
            )
            kwargs = {"event_rate": value}
        
        kwargs["time_point"] = time_point
    
    with col2:
        st.subheader("Conversion Results")
        
        try:
            result = convert_survival_parameters(**kwargs)
            
            # Display results in a nice format
            st.markdown("**All Equivalent Parameters:**")
            
            # Create results dataframe
            results_data = [
                ["Median Survival", f"{result['median_survival']:.3f} {time_unit}"],
                ["Hazard Rate", f"{result['hazard_rate']:.4f} per {time_unit}"],
                ["Survival Fraction", f"{result['survival_fraction']:.1%} at {result['time_point']:.1f} {time_unit}"],
                ["Event Rate", f"{result['event_rate']:.1%} by {result['time_point']:.1f} {time_unit}"]
            ]
            
            df = pd.DataFrame(results_data, columns=["Parameter", "Value"])
            st.dataframe(df, hide_index=True)
            
            # Visual representation
            _plot_survival_curve(result, time_unit)
            
        except Exception as e:
            st.error(f"Error: {e}")


def _hazard_ratio_converter():
    """Hazard ratio scenario conversion interface."""
    
    st.header("Hazard Ratio Scenario")
    st.markdown("Convert hazard ratio scenario to complete parameter sets for both groups.")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Input Parameters")
        
        hazard_ratio = st.number_input(
            "Hazard Ratio (Treatment vs Control)",
            min_value=0.1,
            max_value=5.0,
            value=0.67,
            step=0.01,
            help="HR < 1 favors treatment, HR > 1 favors control"
        )
        
        time_unit = st.selectbox("Time Unit", ["months", "years", "weeks", "days"], key="hr_unit")
        
        time_point = st.number_input(
            f"Time Point for Fractions ({time_unit})",
            min_value=0.1,
            value=12.0 if time_unit == "months" else 1.0,
            step=0.1
        )
        
        # Group parameter input
        input_group = st.selectbox("Provide parameter for:", ["Control Group", "Treatment Group"])
        param_type = st.selectbox("Parameter Type", ["Median Survival", "Hazard Rate"])
        
        if param_type == "Median Survival":
            value = st.number_input(
                f"{input_group} Median Survival ({time_unit})",
                min_value=0.1,
                value=12.0 if time_unit == "months" else 1.0,
                step=0.1
            )
            if input_group == "Control Group":
                kwargs = {"control_median": value}
            else:
                kwargs = {"treatment_median": value}
        else:  # Hazard Rate
            value = st.number_input(
                f"{input_group} Hazard Rate (per {time_unit})",
                min_value=0.001,
                value=0.058 if time_unit == "months" else 0.693,
                step=0.001,
                format="%.4f"
            )
            if input_group == "Control Group":
                kwargs = {"control_hazard": value}
            else:
                kwargs = {"treatment_hazard": value}
    
    with col2:
        st.subheader("Scenario Results")
        
        try:
            result = convert_hazard_ratio_scenario(
                hazard_ratio=hazard_ratio,
                time_point=time_point,
                **kwargs
            )
            
            control = result['control']
            treatment = result['treatment']
            
            # Summary metrics
            st.markdown("**Hazard Ratio Summary:**")
            st.metric("Hazard Ratio", f"{result['hazard_ratio']:.3f}")
            
            col_c, col_t = st.columns(2)
            
            with col_c:
                st.markdown("**Control Group**")
                st.metric("Median Survival", f"{control['median_survival']:.2f} {time_unit}")
                st.metric("Hazard Rate", f"{control['hazard_rate']:.4f}")
                st.metric("Survival Fraction", f"{control['survival_fraction']:.1%}")
                st.metric("Event Rate", f"{control['event_rate']:.1%}")
            
            with col_t:
                st.markdown("**Treatment Group**")
                st.metric("Median Survival", f"{treatment['median_survival']:.2f} {time_unit}")
                st.metric("Hazard Rate", f"{treatment['hazard_rate']:.4f}")
                st.metric("Survival Fraction", f"{treatment['survival_fraction']:.1%}")
                st.metric("Event Rate", f"{treatment['event_rate']:.1%}")
            
            # Comparison plot
            _plot_survival_comparison(result, time_unit)
            
        except Exception as e:
            st.error(f"Error: {e}")


def _unit_converter():
    """Unit conversion interface."""
    
    st.header("Time Unit Conversion")
    st.markdown("Convert survival parameters between different time units.")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Input")
        
        value = st.number_input("Value", value=1.0, step=0.1)
        param_type = st.selectbox(
            "Parameter Type",
            ["median", "hazard"],
            format_func=lambda x: "Median Survival" if x == "median" else "Hazard Rate"
        )
        from_unit = st.selectbox("From Unit", ["days", "weeks", "months", "years"])
        to_unit = st.selectbox("To Unit", ["days", "weeks", "months", "years"])
    
    with col2:
        st.subheader("Conversion Result")
        
        if from_unit != to_unit:
            try:
                if param_type == "median":
                    converted = convert_time_units(value, from_unit, to_unit)
                else:  # hazard
                    # Hazard rates are per time unit, so conversion is inverse
                    converted = convert_time_units(value, to_unit, from_unit)
                
                st.success(f"{value} {param_type} in {from_unit} = {converted:.4f} in {to_unit}")
                
                # Show complete parameter conversion
                if param_type == "median":
                    params = convert_survival_parameters(median_survival=converted)
                else:
                    params = convert_survival_parameters(hazard_rate=converted)
                
                st.markdown("**Complete Parameter Set:**")
                st.write(f"- Median: {params['median_survival']:.3f} {to_unit}")
                st.write(f"- Hazard: {params['hazard_rate']:.4f} per {to_unit}")
                st.write(f"- Survival: {params['survival_fraction']:.1%} at {params['time_point']:.1f} {to_unit}")
                
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.info("Select different units for conversion")


def _examples_and_help():
    """Examples and help interface."""
    
    st.header("Examples & Use Cases")
    
    examples = [
        {
            "title": "🏥 Cancer Trial Planning",
            "scenario": "You have control median = 12 months, target HR = 0.7",
            "need": "Treatment median, event rates, hazard rates",
            "example": {
                "hazard_ratio": 0.7,
                "control_median": 12,
                "time_point": 24
            }
        },
        {
            "title": "📚 Literature Meta-Analysis", 
            "scenario": "Study reports '60% 5-year survival'",
            "need": "Median survival, hazard rate for power calculation",
            "example": {
                "survival_fraction": 0.6,
                "time_point": 60
            }
        },
        {
            "title": "📋 Protocol Development",
            "scenario": "Historical event rate of 40% at 2 years",
            "need": "Median survival for sample size calculation",
            "example": {
                "event_rate": 0.4,
                "time_point": 24
            }
        }
    ]
    
    for i, ex in enumerate(examples):
        with st.expander(ex["title"]):
            st.markdown(f"**Scenario:** {ex['scenario']}")
            st.markdown(f"**Need:** {ex['need']}")
            
            if st.button(f"Run Example {i+1}", key=f"example_{i}"):
                try:
                    if "hazard_ratio" in ex["example"]:
                        result = convert_hazard_ratio_scenario(**ex["example"])
                        st.success("Hazard Ratio Scenario Results:")
                        st.json(result)
                    else:
                        result = convert_survival_parameters(**ex["example"])
                        st.success("Parameter Conversion Results:")
                        st.json(result)
                except Exception as e:
                    st.error(f"Error: {e}")
    
    # Help section
    st.header("📖 Parameter Definitions")
    
    definitions = {
        "Median Survival": "Time at which 50% of subjects have experienced the event",
        "Hazard Rate": "Instantaneous risk of event occurrence (λ in exponential distribution)",
        "Survival Fraction": "Proportion surviving to a specific time point",
        "Event Rate": "Proportion experiencing event by a specific time point",
        "Hazard Ratio": "Ratio of hazard rates between treatment and control groups"
    }
    
    for term, definition in definitions.items():
        st.markdown(f"**{term}:** {definition}")


def _plot_survival_curve(params: dict, time_unit: str):
    """Plot survival curve from parameters."""
    
    # Generate time points
    max_time = params['median_survival'] * 3
    time_points = np.linspace(0, max_time, 100)
    
    # Calculate survival probabilities
    hazard = params['hazard_rate']
    survival_probs = np.exp(-hazard * time_points)
    
    # Create plot
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=time_points,
        y=survival_probs,
        mode='lines',
        name='Survival Curve',
        line=dict(color='blue', width=2)
    ))
    
    # Add median survival line
    fig.add_vline(
        x=params['median_survival'],
        line_dash="dash",
        line_color="red",
        annotation_text=f"Median: {params['median_survival']:.1f} {time_unit}"
    )
    
    # Add time point marker if relevant
    if params['time_point'] <= max_time:
        fig.add_trace(go.Scatter(
            x=[params['time_point']],
            y=[params['survival_fraction']],
            mode='markers',
            name=f'{params["survival_fraction"]:.1%} survival',
            marker=dict(color='red', size=8)
        ))
    
    fig.update_layout(
        title="Survival Curve",
        xaxis_title=f"Time ({time_unit})",
        yaxis_title="Survival Probability",
        yaxis=dict(range=[0, 1]),
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)


def _plot_survival_comparison(result: dict, time_unit: str):
    """Plot survival curves for both groups in hazard ratio scenario."""
    
    control = result['control']
    treatment = result['treatment']
    
    # Generate time points
    max_time = max(control['median_survival'], treatment['median_survival']) * 2.5
    time_points = np.linspace(0, max_time, 100)
    
    # Calculate survival probabilities
    control_survival = np.exp(-control['hazard_rate'] * time_points)
    treatment_survival = np.exp(-treatment['hazard_rate'] * time_points)
    
    # Create plot
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=time_points,
        y=control_survival,
        mode='lines',
        name='Control',
        line=dict(color='red', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=time_points,
        y=treatment_survival,
        mode='lines',
        name='Treatment',
        line=dict(color='blue', width=2)
    ))
    
    # Add median survival lines
    fig.add_vline(
        x=control['median_survival'],
        line_dash="dash",
        line_color="red",
        annotation_text=f"Control Median: {control['median_survival']:.1f}"
    )
    
    fig.add_vline(
        x=treatment['median_survival'],
        line_dash="dash",
        line_color="blue",
        annotation_text=f"Treatment Median: {treatment['median_survival']:.1f}"
    )
    
    fig.update_layout(
        title=f"Survival Comparison (HR = {result['hazard_ratio']:.3f})",
        xaxis_title=f"Time ({time_unit})",
        yaxis_title="Survival Probability",
        yaxis=dict(range=[0, 1]),
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    survival_converter_page()