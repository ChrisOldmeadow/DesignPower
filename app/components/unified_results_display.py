"""
Unified Results Display System for DesignPower Application.

This module provides a standardized interface for displaying results across all
design types, replacing the inconsistent approaches previously used.
"""
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import json
import hashlib
import base64
from typing import Dict, Any, List, Optional, Union, Callable
from dataclasses import dataclass
from core.utils.report_generator import generate_report


@dataclass
class MetricConfig:
    """Configuration for a single metric display."""
    key: str
    label: str
    format_func: Optional[Callable] = None
    help_text: Optional[str] = None
    style: Optional[str] = None  # 'primary', 'secondary', 'success', 'warning', 'error'
    
    def format_value(self, value: Any) -> str:
        """Format the value for display."""
        if value is None or value == 'N/A':
            return 'N/A'
        
        if self.format_func:
            return self.format_func(value)
        
        if isinstance(value, float):
            return f"{value:.3f}"
        elif isinstance(value, (list, tuple)) and len(value) == 2:
            return f"({value[0]:.3f}, {value[1]:.3f})"
        else:
            return str(value)


@dataclass
class SectionConfig:
    """Configuration for a results section."""
    title: str
    metrics: List[MetricConfig]
    expanded: bool = True
    show_divider: bool = True
    custom_display_func: Optional[Callable] = None


@dataclass
class DisplayConfig:
    """Complete display configuration for a design type."""
    design_name: str
    outcome_name: str
    sections: List[SectionConfig]
    show_simulation_details: bool = True
    show_parameters_summary: bool = True
    show_html_report: bool = True
    show_cli_code: bool = True
    cli_code_generator: Optional[Callable] = None
    custom_visualization: Optional[Callable] = None


class UnifiedResultsDisplay:
    """Unified results display system for all design types."""
    
    def __init__(self):
        """Initialize the unified results display system."""
        self.display_configs = {}
        self._register_default_configs()
    
    def register_config(self, key: str, config: DisplayConfig):
        """Register a display configuration for a specific design/outcome combination."""
        self.display_configs[key] = config
    
    def display_results(
        self,
        results: Dict[str, Any],
        params: Dict[str, Any],
        design_type: str,
        outcome_type: str,
        calculation_type: str,
        hypothesis_type: str = "Superiority",
        method_used: str = "analytical"
    ):
        """
        Display results using the unified system.
        
        Args:
            results: Dictionary containing calculation results
            params: Dictionary containing input parameters
            design_type: Type of study design
            outcome_type: Type of outcome
            calculation_type: Type of calculation performed
            hypothesis_type: Type of hypothesis being tested
            method_used: Method used for calculation (analytical/simulation)
        """
        # Check for errors first
        if isinstance(results, dict) and "error" in results:
            st.error(results["error"])
            return
        
        # Ensure hypothesis_type and calculation_type are available in params for report generation
        params = params.copy()  # Don't modify the original
        params['hypothesis_type'] = hypothesis_type
        params['calculation_type'] = calculation_type
        
        # Get the appropriate display configuration
        config_key = (design_type, outcome_type)
        config = self.display_configs.get(config_key)
        
        if not config:
            # Fallback to generic display
            self._display_generic_results(results, params, design_type, outcome_type, calculation_type)
            return
        
        # Display main results header
        st.markdown(f"### {config.design_name}: {config.outcome_name} - Results Summary")
        st.markdown("---")
        
        # Display custom visualization if available
        if config.custom_visualization:
            config.custom_visualization(results, params, calculation_type, hypothesis_type, method_used)
        
        # Display configured sections
        for section in config.sections:
            self._display_section(section, results, params, calculation_type, hypothesis_type)
        
        # Display simulation details if applicable
        if config.show_simulation_details and self._is_simulation_method(results, params, method_used):
            self._display_simulation_details(results, params)
        
        st.markdown("---")
        
        # Display standard components
        if config.show_parameters_summary:
            self._display_parameters_summary(params)
        
        if config.show_html_report:
            self._display_html_report(results, params, design_type, outcome_type)
        
        if config.show_cli_code and config.cli_code_generator:
            self._display_cli_code(params, config.cli_code_generator, design_type, outcome_type)
    
    def _display_section(
        self,
        section: SectionConfig,
        results: Dict[str, Any],
        params: Dict[str, Any],
        calculation_type: str,
        hypothesis_type: str
    ):
        """Display a configured results section."""
        if section.custom_display_func:
            # Use custom display function
            section.custom_display_func(results, params, calculation_type, hypothesis_type)
        else:
            # Use standard metric display
            if section.title:
                if section.expanded:
                    st.markdown(f"#### {section.title}")
                else:
                    with st.expander(section.title):
                        self._display_metrics(section.metrics, results, params)
                    return
            
            self._display_metrics(section.metrics, results, params)
        
        if section.show_divider:
            st.markdown("---")
    
    def _display_metrics(self, metrics: List[MetricConfig], results: Dict[str, Any], params: Dict[str, Any]):
        """Display a list of metrics."""
        # Group metrics by style for better layout
        primary_metrics = [m for m in metrics if m.style in [None, 'primary']]
        secondary_metrics = [m for m in metrics if m.style == 'secondary']
        
        # Display primary metrics in columns
        if primary_metrics:
            cols = st.columns(min(len(primary_metrics), 3))
            for i, metric in enumerate(primary_metrics):
                with cols[i % 3]:
                    value = self._get_metric_value(metric.key, results, params)
                    formatted_value = metric.format_value(value)
                    st.metric(label=metric.label, value=formatted_value, help=metric.help_text)
        
        # Display secondary metrics
        if secondary_metrics:
            st.markdown("##### Additional Details")
            for metric in secondary_metrics:
                value = self._get_metric_value(metric.key, results, params)
                formatted_value = metric.format_value(value)
                
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.markdown(f"**{metric.label}:**")
                with col2:
                    if metric.style == 'warning':
                        st.warning(formatted_value)
                    elif metric.style == 'error':
                        st.error(formatted_value)
                    elif metric.style == 'success':
                        st.success(formatted_value)
                    else:
                        st.markdown(formatted_value)
    
    def _get_metric_value(self, key: str, results: Dict[str, Any], params: Dict[str, Any]) -> Any:
        """Get a metric value from results or params, supporting nested keys."""
        # Support nested keys like 'result.power' or 'params.alpha'
        if '.' in key:
            source, nested_key = key.split('.', 1)
            if source == 'results':
                return results.get(nested_key)
            elif source == 'params':
                return params.get(nested_key)
        
        # Try results first, then params
        return results.get(key, params.get(key))
    
    def _is_simulation_method(self, results: Dict[str, Any], params: Dict[str, Any], method_used: str) -> bool:
        """Check if simulation method was used."""
        return (
            method_used.lower() == "simulation" or
            results.get('method') == 'simulation' or
            params.get('method') == 'simulation' or
            params.get('use_simulation', False)
        )
    
    def _display_simulation_details(self, results: Dict[str, Any], params: Dict[str, Any]):
        """Display simulation-specific details."""
        st.markdown("#### Simulation Details")
        with st.expander("View Simulation Specifics"):
            nsim = results.get('nsim_run', results.get('nsim', params.get('nsim', 'N/A')))
            st.write(f"Number of simulations run: {nsim}")
            
            if 'fallback_reason' in results and results['fallback_reason']:
                st.warning(f"Simulation used fallback values: {results['fallback_reason']}")
            
            # Display other simulation-specific metrics
            sim_metrics = ['convergence_rate', 'simulation_time', 'seed']
            for metric in sim_metrics:
                if metric in results:
                    st.write(f"{metric.replace('_', ' ').title()}: {results[metric]}")
    
    def _display_parameters_summary(self, params: Dict[str, Any]):
        """Display input parameters summary with enhanced visual presentation."""
        with st.expander("📋 Input Parameters Summary", expanded=False):
            filtered_params = {
                k: v for k, v in params.items() 
                if not k.startswith('_') and k not in [
                    'results', 'design_type', 'outcome_type', 
                    'previous_design', 'previous_outcome', 'calculation_type_changed',
                    'button_calculate_clicked'
                ]
            }
            
            if not filtered_params:
                st.info("No input parameters to display.")
                return
            
            # Create a visually appealing parameters table
            param_data = []
            for key, value in filtered_params.items():
                # Format parameter names to be more readable
                readable_name = key.replace('_', ' ').title()
                
                # Format values appropriately
                if isinstance(value, bool):
                    formatted_value = "✅ Yes" if value else "❌ No"
                elif isinstance(value, float):
                    if 0 < value < 1:  # Likely a proportion/probability
                        formatted_value = f"{value:.3f}"
                    else:
                        formatted_value = f"{value:.2f}"
                elif isinstance(value, int):
                    formatted_value = f"{value:,}"
                elif isinstance(value, str):
                    formatted_value = value
                elif value is None:
                    formatted_value = "Not specified"
                else:
                    formatted_value = str(value)
                
                param_data.append({
                    "Parameter": readable_name,
                    "Value": formatted_value
                })
            
            # Display as a nicely formatted table
            if param_data:
                df_params = pd.DataFrame(param_data)
                st.dataframe(
                    df_params,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Parameter": st.column_config.TextColumn(
                            "Parameter",
                            width="medium",
                            help="Input parameter name"
                        ),
                        "Value": st.column_config.TextColumn(
                            "Value",
                            width="medium", 
                            help="Parameter value"
                        )
                    }
                )
                
                # Add download options
                st.markdown("---")
                st.markdown("**📥 Download Options:**")
                
                col1, col2, col3 = st.columns(3)
                
                # Generate unique key suffix for download buttons
                params_key_suffix = hashlib.md5(json.dumps(filtered_params, sort_keys=True).encode()).hexdigest()[:8]
                
                with col1:
                    # CSV download
                    csv_data = df_params.to_csv(index=False)
                    st.download_button(
                        label="📊 CSV File",
                        data=csv_data,
                        file_name=f"designpower_parameters_{params_key_suffix}.csv",
                        mime="text/csv",
                        key=f"download_params_csv_{params_key_suffix}",
                        help="Download parameters as CSV file"
                    )
                
                with col2:
                    # JSON download (formatted)
                    json_data = json.dumps(filtered_params, indent=2, sort_keys=True)
                    st.download_button(
                        label="📄 JSON File", 
                        data=json_data,
                        file_name=f"designpower_parameters_{params_key_suffix}.json",
                        mime="application/json",
                        key=f"download_params_json_{params_key_suffix}",
                        help="Download parameters as JSON file"
                    )
                
                with col3:
                    # Text summary download
                    text_summary = self._generate_text_summary(filtered_params)
                    st.download_button(
                        label="📝 Text Summary",
                        data=text_summary,
                        file_name=f"designpower_parameters_{params_key_suffix}.txt",
                        mime="text/plain",
                        key=f"download_params_txt_{params_key_suffix}",
                        help="Download parameters as readable text summary"
                    )
    
    def _generate_text_summary(self, params: Dict[str, Any]) -> str:
        """Generate a human-readable text summary of parameters."""
        from datetime import datetime
        
        summary_lines = [
            "DesignPower - Input Parameters Summary",
            "=" * 40,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "ANALYSIS PARAMETERS:",
            "-" * 20
        ]
        
        # Group parameters by category for better organization
        design_params = []
        statistical_params = []
        simulation_params = []
        other_params = []
        
        for key, value in params.items():
            readable_name = key.replace('_', ' ').title()
            
            # Format value for text display
            if isinstance(value, bool):
                formatted_value = "Yes" if value else "No"
            elif isinstance(value, float):
                if 0 < value < 1:
                    formatted_value = f"{value:.3f}"
                else:
                    formatted_value = f"{value:.2f}"
            elif isinstance(value, int):
                formatted_value = f"{value:,}"
            elif value is None:
                formatted_value = "Not specified"
            else:
                formatted_value = str(value)
            
            param_line = f"{readable_name}: {formatted_value}"
            
            # Categorize parameters
            if any(keyword in key.lower() for keyword in ['design', 'calculation', 'hypothesis', 'method']):
                design_params.append(param_line)
            elif any(keyword in key.lower() for keyword in ['alpha', 'power', 'effect', 'mean', 'proportion', 'p1', 'p2', 'std', 'icc', 'cluster']):
                statistical_params.append(param_line)
            elif any(keyword in key.lower() for keyword in ['nsim', 'seed', 'simulation']):
                simulation_params.append(param_line)
            else:
                other_params.append(param_line)
        
        # Add categorized parameters to summary
        if design_params:
            summary_lines.extend(["", "Study Design:", "-" * 13])
            summary_lines.extend(design_params)
        
        if statistical_params:
            summary_lines.extend(["", "Statistical Parameters:", "-" * 21])
            summary_lines.extend(statistical_params)
        
        if simulation_params:
            summary_lines.extend(["", "Simulation Parameters:", "-" * 21])
            summary_lines.extend(simulation_params)
        
        if other_params:
            summary_lines.extend(["", "Other Parameters:", "-" * 17])
            summary_lines.extend(other_params)
        
        summary_lines.extend([
            "",
            "=" * 40,
            "This summary was generated by DesignPower.",
            "For more information, visit the DesignPower documentation."
        ])
        
        return "\n".join(summary_lines)
    
    def _display_html_report(self, results: Dict[str, Any], params: Dict[str, Any], design_type: str, outcome_type: str):
        """Display HTML report generation."""
        with st.expander("Detailed HTML Report"):
            try:
                report_html = generate_report(
                    design_type=design_type,
                    outcome_type=outcome_type,
                    params=params,
                    results=results
                )
                st.markdown(report_html, unsafe_allow_html=True)
                
                report_key_suffix = hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()[:8]
                st.download_button(
                    label="Download Report (.html)",
                    data=report_html,
                    file_name=f"designpower_report_{report_key_suffix}.html",
                    mime="text/html",
                    key=f"download_report_{design_type.replace(' ', '_')}_{outcome_type.replace(' ', '_')}_{report_key_suffix}"
                )
            except Exception as e:
                st.error(f"Error generating HTML report: {e}")
    
    def _display_cli_code(self, params: Dict[str, Any], cli_generator: Callable, design_type: str, outcome_type: str):
        """Display CLI code generation."""
        with st.expander("Reproducible Python CLI Code"):
            try:
                cli_code = cli_generator(params)
                st.code(cli_code, language="python")
                
                cli_key_suffix = hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()[:8]
                st.download_button(
                    label="Download CLI Script (.py)",
                    data=cli_code,
                    file_name=f"designpower_{design_type.replace(' ', '_')}_{outcome_type.replace(' ', '_')}_cli_{cli_key_suffix}.py",
                    mime="text/x-python",
                    key=f"download_cli_{design_type.replace(' ', '_')}_{outcome_type.replace(' ', '_')}_{cli_key_suffix}"
                )
            except Exception as e:
                st.error(f"Error generating CLI code: {e}")
    
    def _display_generic_results(self, results: Dict[str, Any], params: Dict[str, Any], design_type: str, outcome_type: str, calculation_type: str):
        """Fallback generic results display."""
        st.markdown("### Results Summary")
        st.markdown("---")
        
        # Filter out non-displayable results
        filtered_results = {
            k: v for k, v in results.items() 
            if k not in [
                "design_method", "error", "power_curve_data", 
                "survival_curves", "power_vs_hr_data", "plot_data",
                "alpha_param", "power_param"
            ]
        }
        
        if not filtered_results:
            st.info("No specific tabular results to display for this configuration.")
            return
        
        for key, value in filtered_results.items():
            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown(f"**{key.replace('_', ' ').title()}:**")
            with col2:
                if isinstance(value, float):
                    st.markdown(f"{value:.3f}")
                elif isinstance(value, (list, tuple)) and len(value) == 2:
                    st.markdown(f"({value[0]:.3f}, {value[1]:.3f})")
                else:
                    st.markdown(str(value))
    
    def _register_default_configs(self):
        """Register default display configurations for all design types."""
        # This will be populated with specific configurations for each design type
        pass


# Custom formatting functions for common metrics
def format_sample_size(value: Any) -> str:
    """Format sample size values."""
    if isinstance(value, (int, float)):
        return f"{int(value)}"
    return str(value)

def format_power(value: Any) -> str:
    """Format power values."""
    if isinstance(value, (int, float)):
        return f"{value:.3f}"
    return str(value)

def format_effect_size(value: Any) -> str:
    """Format effect size values."""
    if isinstance(value, (int, float)):
        return f"{value:.4f}"
    return str(value)

def format_percentage(value: Any) -> str:
    """Format percentage values."""
    if isinstance(value, (int, float)):
        return f"{value:.1%}"
    return str(value)


# Global instance
unified_display = UnifiedResultsDisplay()