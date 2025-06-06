"""
Display configurations for different design types in the DesignPower application.

This module contains the specific display configurations for each combination
of design type and outcome type, defining how results should be presented.
"""
from typing import Dict, Any
import streamlit as st
from .unified_results_display import (
    DisplayConfig, SectionConfig, MetricConfig,
    format_sample_size, format_power, format_effect_size, format_percentage
)


def create_parallel_rct_continuous_config() -> DisplayConfig:
    """Create display configuration for Parallel RCT with continuous outcome."""
    
    def get_sample_size_metrics(calc_type: str) -> list:
        """Get metrics for sample size calculation."""
        if calc_type == "Sample Size":
            return [
                MetricConfig("total_n", "Total Sample Size (N)", format_sample_size, style="primary"),
                MetricConfig("n1", "Sample Size per Group (n1)", format_sample_size, style="primary"),
                MetricConfig("n2", "Sample Size per Group (n2)", format_sample_size, style="primary")
            ]
        elif calc_type == "Power":
            return [
                MetricConfig("power", "Power", format_power, style="primary")
            ]
        else:  # Minimum Detectable Effect
            return [
                MetricConfig("mde", "Minimum Detectable Effect (MDE)", format_effect_size, style="primary")
            ]
    
    def display_main_results(results: Dict[str, Any], params: Dict[str, Any], calc_type: str, hypothesis_type: str):
        """Custom display function for main results."""
        metrics = get_sample_size_metrics(calc_type)
        
        # Add non-inferiority margin if applicable
        if hypothesis_type == "Non-Inferiority":
            metrics.append(
                MetricConfig("params.non_inferiority_margin", "Non-Inferiority Margin", format_effect_size, style="secondary")
            )
        
        # Display metrics in columns
        if metrics:
            cols = st.columns(min(len(metrics), 3))
            for i, metric in enumerate(metrics):
                with cols[i % 3]:
                    if metric.key.startswith('params.'):
                        value = params.get(metric.key.split('.', 1)[1])
                    else:
                        value = results.get(metric.key, 'N/A')
                    formatted_value = metric.format_value(value)
                    st.metric(label=metric.label, value=formatted_value, help=metric.help_text)
    
    main_section = SectionConfig(
        title="",  # No title, this is the main display
        metrics=[],
        custom_display_func=display_main_results
    )
    
    return DisplayConfig(
        design_name="Parallel RCT",
        outcome_name="Continuous Outcome",
        sections=[main_section],
        cli_code_generator=None  # Will be set when imported
    )


def create_parallel_rct_binary_config() -> DisplayConfig:
    """Create display configuration for Parallel RCT with binary outcome."""
    
    def display_main_results(results: Dict[str, Any], params: Dict[str, Any], calc_type: str, hypothesis_type: str):
        """Custom display function for binary results."""
        if calc_type == "Sample Size":
            metrics = [
                MetricConfig("total_n", "Total Sample Size (N)", format_sample_size, style="primary"),
                MetricConfig("n1", "Sample Size per Group (n1)", format_sample_size, style="primary"),
                MetricConfig("n2", "Sample Size per Group (n2)", format_sample_size, style="primary")
            ]
        elif calc_type == "Power":
            metrics = [
                MetricConfig("power", "Power", format_power, style="primary")
            ]
        else:  # Minimum Detectable Effect
            metrics = [
                MetricConfig("mde", "Minimum Detectable Effect", format_effect_size, style="primary")
            ]
        
        # Handle non-inferiority margins for binary outcomes
        if hypothesis_type == "Non-Inferiority":
            nim_value_rd = params.get('non_inferiority_margin_rd')
            nim_value_rr = params.get('non_inferiority_margin_rr')
            
            if nim_value_rd is not None:
                metrics.append(MetricConfig("params.non_inferiority_margin_rd", "Non-Inferiority Margin (Risk Difference)", format_effect_size))
            elif nim_value_rr is not None:
                metrics.append(MetricConfig("params.non_inferiority_margin_rr", "Non-Inferiority Margin (Risk Ratio)", format_effect_size))
            elif params.get('non_inferiority_margin') is not None:
                metrics.append(MetricConfig("params.non_inferiority_margin", "Non-Inferiority Margin", format_effect_size))
        
        # Display metrics
        if metrics:
            cols = st.columns(min(len(metrics), 3))
            for i, metric in enumerate(metrics):
                with cols[i % 3]:
                    if metric.key.startswith('params.'):
                        value = params.get(metric.key.split('.', 1)[1])
                    else:
                        value = results.get(metric.key, 'N/A')
                    formatted_value = metric.format_value(value)
                    st.metric(label=metric.label, value=formatted_value)
    
    main_section = SectionConfig(
        title="",
        metrics=[],
        custom_display_func=display_main_results
    )
    
    return DisplayConfig(
        design_name="Parallel RCT",
        outcome_name="Binary Outcome",
        sections=[main_section],
        cli_code_generator=None
    )


def create_cluster_rct_continuous_config() -> DisplayConfig:
    """Create display configuration for Cluster RCT with continuous outcome."""
    
    def display_main_results(results: Dict[str, Any], params: Dict[str, Any], calc_type: str, hypothesis_type: str):
        """Custom display function for cluster continuous results."""
        # Import here to avoid circular imports
        from .cluster_display_utils import display_cluster_continuous_results
        # Now properly pass the hypothesis_type parameter
        display_cluster_continuous_results(results, params, calc_type, hypothesis_type)
    
    main_section = SectionConfig(
        title="",
        metrics=[],
        custom_display_func=display_main_results
    )
    
    return DisplayConfig(
        design_name="Cluster RCT",
        outcome_name="Continuous Outcome", 
        sections=[main_section],
        cli_code_generator=None
    )


def create_cluster_rct_binary_config() -> DisplayConfig:
    """Create display configuration for Cluster RCT with binary outcome."""
    
    def display_main_results(results: Dict[str, Any], params: Dict[str, Any], calc_type: str, hypothesis_type: str):
        """Custom display function for cluster binary results."""
        if calc_type == "Sample Size":
            metrics = [
                MetricConfig("total_n", "Total Sample Size (N)", format_sample_size, style="primary"),
                MetricConfig("n_clusters", "Number of Clusters (K)", format_sample_size, style="primary"),
                MetricConfig("cluster_size_actual", "Avg. Cluster Size (m)", format_sample_size, style="primary", 
                           help_text="Average cluster size")
            ]
        elif calc_type == "Power":
            metrics = [
                MetricConfig("power", "Power", format_power, style="primary")
            ]
        else:  # Minimum Detectable Effect
            # Handle multiple possible MDE keys for cluster binary
            mde_key = None
            for key in ['mde_p2', 'mde_absolute_risk_reduction', 'mde']:
                if key in results:
                    mde_key = key
                    break
            
            metrics = [
                MetricConfig(mde_key or "mde", "Minimum Detectable Effect", format_effect_size, style="primary")
            ]
        
        # Display metrics
        if metrics:
            cols = st.columns(min(len(metrics), 3))
            for i, metric in enumerate(metrics):
                with cols[i % 3]:
                    value = results.get(metric.key, results.get('cluster_size', 'N/A') if metric.key == 'cluster_size_actual' else 'N/A')
                    if isinstance(value, (int, float)) and metric.format_func:
                        formatted_value = metric.format_func(value)
                    else:
                        formatted_value = metric.format_value(value)
                    st.metric(label=metric.label, value=formatted_value, help=metric.help_text)
    
    main_section = SectionConfig(
        title="",
        metrics=[],
        custom_display_func=display_main_results
    )
    
    return DisplayConfig(
        design_name="Cluster RCT",
        outcome_name="Binary Outcome",
        sections=[main_section],
        cli_code_generator=None
    )


def create_single_arm_binary_config() -> DisplayConfig:
    """Create display configuration for Single Arm with binary outcome."""
    
    def display_main_results(results: Dict[str, Any], params: Dict[str, Any], calc_type: str, hypothesis_type: str):
        """Custom display function for single arm binary results."""
        design_method = results.get("design_method")
        
        if design_method == "A'Hern":
            st.markdown("### A'Hern Design Results")
            st.markdown("---")
            
            tab1, tab2 = st.tabs(["📊 Key Parameters", "📏 Effect Size"])
            
            with tab1:
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("##### Sample Size Calculation")
                    st.markdown(f"""<div style='background-color:#e6f3ff;padding:10px;border-radius:5px;margin-bottom:5px;'>
                              <b>Required Sample Size (n):</b> {results.get('n')}
                              </div>""", unsafe_allow_html=True)
                    st.markdown(f"""<div style='background-color:#e6f3ff;padding:10px;border-radius:5px;margin-bottom:5px;'>
                              <b>Rejection Threshold (r):</b> {results.get('r')}
                              </div>""", unsafe_allow_html=True)
                    st.markdown(f"""<div style='background-color:#e6fff0;padding:10px;border-radius:5px;margin-bottom:5px;'>
                              <b>Interpretation:</b> Reject H₀ if {results.get('r')} or more responses are observed
                              </div>""", unsafe_allow_html=True)
                
                with col2:
                    st.markdown("##### Error Rates")
                    st.markdown(f"""<div style='background-color:#fff0e6;padding:10px;border-radius:5px;margin-bottom:5px;'>
                              <b>Target Type I Error (α):</b> {params.get('alpha')}
                              </div>""", unsafe_allow_html=True)
                    st.markdown(f"""<div style='background-color:#fff0e6;padding:10px;border-radius:5px;margin-bottom:5px;'>
                              <b>Actual Type I Error:</b> {results.get('actual_alpha')}
                              </div>""", unsafe_allow_html=True)
                    st.markdown(f"""<div style='background-color:#e6e6ff;padding:10px;border-radius:5px;margin-bottom:5px;'>
                              <b>Target Power:</b> {params.get('power', 1-params.get('beta', 0.2))}
                              </div>""", unsafe_allow_html=True)
                    st.markdown(f"""<div style='background-color:#e6e6ff;padding:10px;border-radius:5px;'>
                              <b>Actual Power:</b> {results.get('actual_power')}
                              </div>""", unsafe_allow_html=True)
            
            with tab2:
                st.markdown("##### Effect Size Parameters")
                st.markdown(f"""<div style='background-color:#f0f0f0;padding:10px;border-radius:5px;margin-bottom:5px;'>
                          <b>Unacceptable Response Rate (p0):</b> {params.get('p0')}
                          </div>""", unsafe_allow_html=True)
                st.markdown(f"""<div style='background-color:#f0f0f0;padding:10px;border-radius:5px;'>
                          <b>Desirable Response Rate (p1):</b> {params.get('p1')}
                          </div>""", unsafe_allow_html=True)
        
        elif design_method == "Simon's Two-Stage":
            st.markdown("### Simon's Two-Stage Design Results")
            st.markdown("---")
            
            st.markdown("#### Stage 1")
            col1_s1, col2_s1 = st.columns(2)
            with col1_s1:
                st.metric(label="Sample Size (n1)", value=results.get("n1"))
            with col2_s1:
                st.metric(label="Rejection Threshold (r1)", value=results.get("r1"))
            st.markdown(f"**Interpretation:** If ≤ {results.get('r1')} responses in {results.get('n1')} patients, stop the trial (futility).")

            st.markdown("#### Stage 2")
            col1_s2, col2_s2, col3_s2 = st.columns(3)
            with col1_s2:
                st.metric(label="Total Sample Size (N)", value=results.get("N"))
            with col2_s2:
                st.metric(label="Overall Rejection Threshold (r)", value=results.get("r"))
            with col3_s2:
                st.metric(label="Probability of Early Termination (PET)", value=f"{results.get('PET', 0.0):.3f}")
            
            st.markdown(f"**Interpretation:** If > {results.get('r1')} responses in Stage 1, proceed to Stage 2. "
                        f"Overall, if ≤ {results.get('r')} responses in {results.get('N')} patients, reject H₁ (treatment ineffective).")
            
            st.markdown("#### Expected Sample Size")
            st.metric(label="Expected Sample Size (EN)", value=f"{results.get('EN', 0.0):.2f}")
    
    main_section = SectionConfig(
        title="",
        metrics=[],
        custom_display_func=display_main_results
    )
    
    return DisplayConfig(
        design_name="Single Arm Trial",
        outcome_name="Binary Outcome",
        sections=[main_section],
        show_cli_code=False  # Single arm doesn't have CLI code generation yet
    )


def create_parallel_rct_survival_config() -> DisplayConfig:
    """Create display configuration for Parallel RCT with survival outcome."""
    
    def display_main_results(results: Dict[str, Any], params: Dict[str, Any], calc_type: str, hypothesis_type: str):
        """Custom display function for survival results."""
        # Import here to avoid circular imports
        from .parallel_rct import display_survival_results
        display_survival_results(
            result=results,
            calculation_type=calc_type,
            hypothesis_type=hypothesis_type,
            use_simulation=(params.get('method', 'analytical').lower() == "simulation")
        )
    
    main_section = SectionConfig(
        title="",
        metrics=[],
        custom_display_func=display_main_results
    )
    
    return DisplayConfig(
        design_name="Parallel RCT",
        outcome_name="Survival Outcome",
        sections=[main_section],
        cli_code_generator=None
    )


def register_all_configs(unified_display):
    """Register all display configurations with the unified display system."""
    
    # Import CLI code generators
    try:
        from .parallel_rct import generate_cli_code_parallel_continuous, generate_cli_code_parallel_binary, generate_cli_code_parallel_survival
        from .cluster_rct import generate_cli_code_cluster_continuous, generate_cli_code_cluster_binary
    except ImportError:
        # Handle case where CLI generators aren't available
        generate_cli_code_parallel_continuous = None
        generate_cli_code_parallel_binary = None
        generate_cli_code_parallel_survival = None
        generate_cli_code_cluster_continuous = None
        generate_cli_code_cluster_binary = None
    
    # Parallel RCT configurations
    parallel_continuous_config = create_parallel_rct_continuous_config()
    parallel_continuous_config.cli_code_generator = generate_cli_code_parallel_continuous
    unified_display.register_config(("Parallel RCT", "Continuous Outcome"), parallel_continuous_config)
    
    parallel_binary_config = create_parallel_rct_binary_config()
    parallel_binary_config.cli_code_generator = generate_cli_code_parallel_binary
    unified_display.register_config(("Parallel RCT", "Binary Outcome"), parallel_binary_config)
    
    parallel_survival_config = create_parallel_rct_survival_config()
    parallel_survival_config.cli_code_generator = generate_cli_code_parallel_survival
    unified_display.register_config(("Parallel RCT", "Survival Outcome"), parallel_survival_config)
    
    # Cluster RCT configurations
    cluster_continuous_config = create_cluster_rct_continuous_config()
    cluster_continuous_config.cli_code_generator = generate_cli_code_cluster_continuous
    unified_display.register_config(("Cluster RCT", "Continuous Outcome"), cluster_continuous_config)
    
    cluster_binary_config = create_cluster_rct_binary_config()
    cluster_binary_config.cli_code_generator = generate_cli_code_cluster_binary
    unified_display.register_config(("Cluster RCT", "Binary Outcome"), cluster_binary_config)
    
    # Single Arm configurations
    single_arm_binary_config = create_single_arm_binary_config()
    unified_display.register_config(("Single Arm Trial", "Binary Outcome"), single_arm_binary_config)
    
    # Add other single arm configurations as needed
    # unified_display.register_config(("Single Arm Trial", "Continuous Outcome"), create_single_arm_continuous_config())
    # unified_display.register_config(("Single Arm Trial", "Survival Outcome"), create_single_arm_survival_config())