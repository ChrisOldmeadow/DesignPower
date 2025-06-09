"""Binary outcome UI components for cluster RCT designs.

This module provides UI rendering functions for cluster RCT designs
with binary outcomes, including advanced options and main interface.
"""

import streamlit as st
import numpy as np
import pandas as pd
import math

# Import CLI generation utilities
from .cli_generation import _detect_resource_constraints


def render_binary_advanced_options():
    """
    Render advanced options for binary outcome designs in cluster RCTs.
    
    Returns:
        dict: Advanced parameters
    """
    advanced_params = {}
    
    # Method selection (analytical vs simulation)
    advanced_params["method"] = st.radio(
        "Calculation Method",
        ["Analytical", "Permutation Test", "Simulation"],
        index=0,
        key="cluster_binary_method_radio",
        horizontal=True,
        help="Analytical: Fast closed-form calculations. Permutation Test: Exact inference without distributional assumptions (ideal for small clusters). Simulation: Monte Carlo simulation with various analysis models."
    )
    
    # Convert to method identifier for function calls
    method_map = {
        "analytical": "analytical",
        "permutation test": "permutation",
        "simulation": "simulation"
    }
    advanced_params["method"] = method_map.get(advanced_params["method"].lower(), "analytical")
    
    # Cluster Size Variation tab
    st.markdown("#### Cluster Size Variation")
    advanced_params["cv_cluster_size"] = st.slider(
        "Coefficient of Variation for Cluster Sizes",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.05,
        format="%0.2f",
        help="Coefficient of variation for cluster sizes. 0 = equal cluster sizes, larger values indicate more variation."
    )
    
    # ICC Scale Conversion tab
    st.markdown("#### ICC Scale Conversion")
    icc_scales = ["Linear", "Logit"]
    advanced_params["icc_scale"] = st.radio(
        "ICC Scale",
        icc_scales,
        index=0,
        key="icc_scale_radio",
        horizontal=True,
        help="ICC can be specified on linear or logit scale. ICC values on different scales may not be directly comparable."
    )
    
    # Only show conversion when logit scale is selected
    if advanced_params["icc_scale"] == "Logit":
        st.info("The ICC value will be converted from logit to linear scale for calculations. Conversion depends on the control group proportion.")
    
    # Effect Measure Options
    st.markdown("#### Effect Measure")
    effect_measures = ["Risk Difference", "Risk Ratio", "Odds Ratio"]
    advanced_params["effect_measure"] = st.radio(
        "Effect Measure",
        effect_measures,
        index=0,
        key="effect_measure_radio",
        horizontal=True,
        help="Specify which effect measure to use for the calculation."
    ).lower().replace(" ", "_")
    
    # Simulation-specific options
    if advanced_params["method"] == "simulation":
        st.markdown("#### Simulation Parameters")
        col1, col2 = st.columns(2)
        
        with col1:
            advanced_params["nsim"] = st.number_input(
                "Number of Simulations", 
                value=1000, 
                min_value=100, 
                max_value=10000,
                step=100,
                key="cluster_binary_nsim"
            )
        
        with col2:
            advanced_params["seed"] = st.number_input(
                "Random Seed", 
                value=42, 
                min_value=1, 
                max_value=99999,
                key="cluster_binary_seed"
            )

        st.markdown("#### Analysis Model")
        # Check availability of Bayesian backends
        try:
            import cmdstanpy
            stan_available = True
        except ImportError:
            stan_available = False
            
        try:
            import pymc
            pymc_available = True
        except ImportError:
            pymc_available = False
        
        model_options_binary = [
            "Design Effect Adjusted Z-test",
            "T-test on Aggregate Data",
            "GLMM (Individual-Level)",
            "GEE (Individual-Level)"
        ]
        
        # Add Bayesian options based on availability
        if stan_available:
            model_options_binary.append("Bayesian (Stan)")
        else:
            model_options_binary.append("Bayesian (Stan) - Not Available")
            
        if pymc_available:
            model_options_binary.append("Bayesian (PyMC)")
        else:
            model_options_binary.append("Bayesian (PyMC) - Not Available")
            
        # Approximate Bayesian methods (lightweight, always available with scipy)
        model_options_binary.append("Bayesian (Variational) - Fast")
        model_options_binary.append("Bayesian (ABC) - Lightweight")
        
        # Smart suggestions based on environment
        is_constrained = _detect_resource_constraints()
        if is_constrained:
            st.info(
                "🌐 **Resource-Constrained Environment Detected**\n\n"
                "For optimal performance in this environment, consider:\n"
                "• **Bayesian (ABC) - Lightweight** for basic Bayesian inference\n"
                "• **Bayesian (Variational) - Fast** for faster approximate inference\n"
                "• **Design Effect Adjusted Z-test** for fastest non-Bayesian analysis\n\n"
                "Full MCMC methods (Stan/PyMC) may be slow or unavailable."
            )
        elif not stan_available and not pymc_available:
            st.info(
                "💡 **Bayesian Analysis Available**\n\n"
                "Stan/PyMC not detected, but approximate Bayesian methods are available:\n"
                "• **Bayesian (Variational) - Fast** for quick approximate inference\n"
                "• **Bayesian (ABC) - Lightweight** for simulation-based inference"
            )
        
        selected_model_display_binary = st.selectbox(
            "Statistical Model Used to Analyse Each Simulated Trial",
            options=model_options_binary,
            index=0, # Default to Z-test
            key="cluster_binary_model_select",
            help="Select the statistical analysis model to be used in the simulation. \n- 'Design Effect Adjusted Z-test': Uses a z-test adjusted for clustering. \n- 'T-test on Aggregate Data': Performs a t-test on cluster-level summaries. \n- 'GLMM (Individual-Level)': Uses a Generalized Linear Mixed Model (requires individual data simulation). \n- 'GEE (Individual-Level)': Uses Generalized Estimating Equations (requires individual data simulation). \n- 'Bayesian' options: Hierarchical Bayesian models with multiple backends and inference methods."
        )
        
        # Show installation message if Bayesian is selected but not available
        if "Bayesian (Stan)" in selected_model_display_binary and not stan_available:
            st.error(
                "📦 **Stan backend requires additional installation**\n\n"
                "To use Stan for Bayesian analysis, please install cmdstanpy:\n"
                "```bash\n"
                "pip install cmdstanpy\n"
                "```\n"
                "The calculation will fall back to Design Effect Z-test if you proceed."
            )
        elif "Bayesian (PyMC)" in selected_model_display_binary and not pymc_available:
            st.error(
                "📦 **PyMC backend requires additional installation**\n\n"
                "To use PyMC for Bayesian analysis, please install pymc:\n"
                "```bash\n"
                "pip install pymc\n"
                "```\n"
                "The calculation will fall back to variational approximation if you proceed."
            )
        elif "Variational" in selected_model_display_binary:
            st.info(
                "⚡ **Fast Variational Bayes**\n\n"
                "Uses Laplace approximation for fast approximate Bayesian inference on logit scale. "
                "Results are approximate but much faster than full MCMC. "
                "Good for initial exploration or resource-constrained environments."
            )
        elif "ABC" in selected_model_display_binary:
            st.info(
                "🎯 **Approximate Bayesian Computation**\n\n"
                "Uses simulation-based approximate inference for binary outcomes. Very lightweight and "
                "suitable for low-resource servers. Results are approximate but "
                "provide valid uncertainty quantification."
            )
        
        model_map_binary = {
            "Design Effect Adjusted Z-test": "deff_ztest",
            "T-test on Aggregate Data": "aggregate_ttest",
            "GLMM (Individual-Level)": "glmm",
            "GEE (Individual-Level)": "gee",
            "Bayesian (Stan)": "bayes",
            "Bayesian (Stan) - Not Available": "bayes",  # Will fall back automatically
            "Bayesian (PyMC)": "bayes",
            "Bayesian (PyMC) - Not Available": "bayes",  # Will fall back automatically
            "Bayesian (Variational) - Fast": "bayes",
            "Bayesian (ABC) - Lightweight": "bayes",
        }
        advanced_params["analysis_method"] = model_map_binary[selected_model_display_binary]
        advanced_params["analysis_method_ui"] = selected_model_display_binary  # Keep original UI selection for backend detection
        
        # Set Bayesian backend based on selection
        if "Bayesian (PyMC)" in selected_model_display_binary:
            advanced_params["bayes_backend"] = "pymc"
        elif "Variational" in selected_model_display_binary:
            advanced_params["bayes_backend"] = "variational"
        elif "ABC" in selected_model_display_binary:
            advanced_params["bayes_backend"] = "abc"
        else:
            advanced_params["bayes_backend"] = "stan"  # Default to Stan
        
        # Add Bayesian-specific options
        if advanced_params["analysis_method"] == "bayes":
            colb1, colb2 = st.columns(2)
            with colb1:
                advanced_params["bayes_draws"] = st.number_input(
                    "Posterior draws",
                    min_value=100,
                    max_value=5000,
                    value=500,
                    step=100,
                    key="cluster_binary_bayes_draws",
                )
            with colb2:
                advanced_params["bayes_warmup"] = st.number_input(
                    "Warm-up iterations",
                    min_value=100,
                    max_value=5000,
                    value=500,
                    step=100,
                    key="cluster_binary_bayes_warmup",
                )
            
            # Show backend information and resource implications
            backend = advanced_params["bayes_backend"]
            if backend == "pymc":
                backend_name = "PyMC"
                sampling_type = "MCMC (NUTS)"
                resource_note = "Full MCMC - High accuracy, moderate resource use"
            elif backend == "stan":
                backend_name = "Stan"
                sampling_type = "MCMC (NUTS)"
                resource_note = "Full MCMC - High accuracy, moderate resource use"
            elif backend == "variational":
                backend_name = "Variational Bayes"
                sampling_type = "Laplace Approximation"
                resource_note = "⚡ Fast approximation - Low resource use, good for exploration"
            elif backend == "abc":
                backend_name = "ABC"
                sampling_type = "Simulation-based"
                resource_note = "🌐 Lightweight - Very low resource use, suitable for web deployment"
            else:
                backend_name = backend
                sampling_type = "Unknown"
                resource_note = ""
            
            st.info(f"🔧 **Bayesian Backend**: {backend_name} ({sampling_type})\n\n{resource_note}")
            
            # Show limitations for approximate methods
            if backend in ["variational", "abc"]:
                st.warning(
                    "⚠️ **Approximate Method Limitations**:\n"
                    "• Results are approximate, not exact posterior samples\n"
                    "• May underestimate uncertainty in some cases\n"
                    "• Best used for initial exploration or resource-limited environments\n"
                    "• For final analyses, consider full MCMC when possible"
                )
            
            # Bayesian inference method selection
            st.markdown("**Bayesian Inference Method**")
            inference_options = {
                "Credible Interval": "credible_interval",
                "Posterior Probability": "posterior_probability", 
                "ROPE (Region of Practical Equivalence)": "rope"
            }
            selected_inference = st.selectbox(
                "Method for determining statistical significance",
                options=list(inference_options.keys()),
                index=0,
                key="cluster_binary_bayes_inference",
                help=f"""Choose how to determine statistical significance (using {backend_name} backend):
                • **Credible Interval**: 95% credible interval excludes zero (most standard)
                • **Posterior Probability**: >97.5% probability effect is in favorable direction
                • **ROPE**: <5% probability effect is in Region of Practical Equivalence around zero
                
                **Available Backends**:
                • Stan/PyMC: Full MCMC with NUTS sampler (high accuracy)
                • Variational: Fast Laplace approximation (good for exploration)
                • ABC: Lightweight simulation-based inference (web-friendly)"""
            )
            advanced_params["bayes_inference_method"] = inference_options[selected_inference]

    # ICC Sensitivity Analysis section without using an expander
    st.markdown("#### ICC Sensitivity Analysis")
    st.markdown("Explore how results vary across a range of ICC values")
    advanced_params["run_sensitivity"] = st.checkbox(
        "Run ICC Sensitivity Analysis",
        value=False,
        help="Calculate results across a range of ICC values to see how sensitive the results are to ICC assumptions."
    )
    
    if advanced_params["run_sensitivity"]:
        col1, col2 = st.columns(2)
        with col1:
            advanced_params["icc_min"] = st.number_input(
                "Minimum ICC",
                value=0.01,
                min_value=0.0,
                max_value=0.99,
                format="%0.2f"
            )
        with col2:
            advanced_params["icc_max"] = st.number_input(
                "Maximum ICC",
                value=0.10,
                min_value=0.01,
                max_value=0.99,
                format="%0.2f"
            )
        
        advanced_params["icc_steps"] = st.slider(
            "Number of ICC Values",
            min_value=3,
            max_value=10,
            value=5,
            help="Number of equally spaced ICC values to evaluate between the minimum and maximum."
        )
    
    return advanced_params


def render_cluster_binary(calc_type, hypothesis_type):
    """
    Render the UI for Cluster RCT with binary outcome.
    
    Args:
        calc_type: String indicating calculation type (Sample Size, Power, or Minimum Detectable Effect)
        hypothesis_type: String indicating hypothesis type (Superiority or Non-Inferiority)
        
    Returns:
        dict: Parameters collected from the UI
    """
    params = {}
    params["calc_type"] = calc_type
    params["hypothesis_type"] = hypothesis_type
    params["outcome_type"] = "Binary Outcome"
    
    st.markdown("### Study Parameters")
    
    # For binary outcomes in cluster RCTs
    col1, col2 = st.columns(2)
    
    with col1:
        # Cluster parameters
        st.markdown("#### Cluster Parameters")
        params["cluster_size"] = st.number_input(
            "Average Cluster Size", 
            value=20, 
            min_value=2, 
            help="Average number of individuals per cluster"
        )
        
        params["icc"] = st.number_input(
            "Intracluster Correlation Coefficient (ICC)", 
            value=0.05, 
            min_value=0.0, 
            max_value=1.0, 
            format="%f", 
            help="Correlation between individuals within the same cluster"
        )
        
        if calc_type == "Sample Size":
            params["power"] = st.slider(
                "Power (1-β)", 
                min_value=0.7, 
                max_value=0.99, 
                value=0.8, 
                step=0.01, 
                format="%0.2f"
            )
        elif calc_type == "Power":
            params["n_clusters"] = st.number_input(
                "Number of Clusters per Arm", 
                min_value=2, 
                value=15,
                help="Number of clusters in each treatment arm"
            )
        elif calc_type == "Minimum Detectable Effect":
            params["n_clusters"] = st.number_input(
                "Number of Clusters per Arm", 
                min_value=2, 
                value=15,
                help="Number of clusters in each treatment arm"
            )
            params["power"] = st.slider(
                "Power (1-β)", 
                min_value=0.7, 
                max_value=0.99, 
                value=0.8, 
                step=0.01, 
                format="%0.2f"
            )
    
    with col2:
        # Outcome parameters
        st.markdown("#### Outcome Parameters")
        if hypothesis_type == "Superiority":
            if calc_type != "Minimum Detectable Effect":
                params["p1"] = st.slider(
                    "Proportion in Control Group", 
                    min_value=0.01, 
                    max_value=0.99, 
                    value=0.3, 
                    step=0.01, 
                    format="%0.2f"
                )
                params["p2"] = st.slider(
                    "Proportion in Intervention Group", 
                    min_value=0.01, 
                    max_value=0.99, 
                    value=0.5, 
                    step=0.01, 
                    format="%0.2f"
                )
            else:  # Minimum Detectable Effect
                params["p1"] = st.slider(
                    "Proportion in Control Group", 
                    min_value=0.01, 
                    max_value=0.99, 
                    value=0.3, 
                    step=0.01, 
                    format="%0.2f"
                )
                # p2 will be calculated
        
        # Significance level
        params["alpha"] = st.select_slider(
            "Significance Level (α)", 
            options=[0.001, 0.01, 0.05, 0.1], 
            value=0.05
        )
    
    # Advanced options in an expander
    with st.expander("Advanced Options"):
        advanced_params = render_binary_advanced_options()
        params.update(advanced_params)
    
    return params