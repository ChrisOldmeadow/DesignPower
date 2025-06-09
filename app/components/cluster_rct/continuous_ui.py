"""Continuous outcome UI components for cluster RCT designs.

This module provides UI rendering functions for cluster RCT designs
with continuous outcomes, including advanced options and main interface.
"""

import streamlit as st
import numpy as np
import pandas as pd
import math

# Import CLI generation utilities
from .cli_generation import _detect_resource_constraints


def render_continuous_advanced_options():
    """
    Render advanced options for continuous outcome designs in cluster RCTs.
    
    Returns:
        dict: Advanced parameters
    """
    advanced_params = {}
    
    # Method selection (analytical vs simulation)
    advanced_params["method"] = st.radio(
        "Calculation Method",
        ["Analytical", "Permutation Test", "Simulation"],
        index=0,
        key="cluster_continuous_method_radio",
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
                key="cluster_continuous_nsim",
                help="Total number of Monte-Carlo replicates to run. Larger values give more stable estimates at the cost of speed."
            )
        
        with col2:
            advanced_params["seed"] = st.number_input(
                "Random Seed", 
                value=42, 
                min_value=1, 
                max_value=99999,
                key="cluster_continuous_seed",
                help="Set a seed for reproducibility."
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
        
        model_options = [
            "T-test (cluster-level)",
            "Linear Mixed Model (REML)",
            "GEE (Exchangeable)",
        ]
        
        # Add Bayesian options based on availability
        if stan_available:
            model_options.append("Bayesian (Stan)")
        else:
            model_options.append("Bayesian (Stan) - Not Available")
            
        if pymc_available:
            model_options.append("Bayesian (PyMC)")
        else:
            model_options.append("Bayesian (PyMC) - Not Available")
            
        # Approximate Bayesian methods (lightweight, always available with scipy)
        model_options.append("Bayesian (Variational) - Fast")
        model_options.append("Bayesian (ABC) - Lightweight")
        
        # Smart suggestions based on environment
        is_constrained = _detect_resource_constraints()
        if is_constrained:
            st.info(
                "🌐 **Resource-Constrained Environment Detected**\n\n"
                "For optimal performance in this environment, consider:\n"
                "• **Bayesian (ABC) - Lightweight** for basic Bayesian inference\n"
                "• **Bayesian (Variational) - Fast** for faster approximate inference\n"
                "• **T-test (cluster-level)** for fastest non-Bayesian analysis\n\n"
                "Full MCMC methods (Stan/PyMC) may be slow or unavailable."
            )
        elif not stan_available and not pymc_available:
            st.info(
                "💡 **Bayesian Analysis Available**\n\n"
                "Stan/PyMC not detected, but approximate Bayesian methods are available:\n"
                "• **Bayesian (Variational) - Fast** for quick approximate inference\n"
                "• **Bayesian (ABC) - Lightweight** for simulation-based inference"
            )
        
        model_display = st.selectbox(
            "Statistical Model Used to Analyse Each Simulated Trial",
            model_options,
            index=0,
            key="cluster_continuous_model_select",
            help="Choose the analysis model applied to each simulated dataset. The simple two-sample t-test analyses cluster-level data. Mixed models explicitly model random cluster intercepts and can provide more power when cluster counts are moderate to large. GEE provides marginal (population-averaged) inference and is robust to some model misspecification, but small-sample bias can be an issue."
        )
        
        # Show installation message if Bayesian is selected but not available
        if "Bayesian (Stan)" in model_display and not stan_available:
            st.error(
                "📦 **Stan backend requires additional installation**\n\n"
                "To use Stan for Bayesian analysis, please install cmdstanpy:\n"
                "```bash\n"
                "pip install cmdstanpy\n"
                "```\n"
                "The calculation will fall back to cluster-level t-test if you proceed."
            )
        elif "Bayesian (PyMC)" in model_display and not pymc_available:
            st.error(
                "📦 **PyMC backend requires additional installation**\n\n"
                "To use PyMC for Bayesian analysis, please install pymc:\n"
                "```bash\n"
                "pip install pymc\n"
                "```\n"
                "The calculation will fall back to variational approximation if you proceed."
            )
        elif "Variational" in model_display:
            st.info(
                "⚡ **Fast Variational Bayes**\n\n"
                "Uses Laplace approximation for fast approximate Bayesian inference. "
                "Results are approximate but much faster than full MCMC. "
                "Good for initial exploration or resource-constrained environments."
            )
        elif "ABC" in model_display:
            st.info(
                "🎯 **Approximate Bayesian Computation**\n\n"
                "Uses simulation-based approximate inference. Very lightweight and "
                "suitable for low-resource servers. Results are approximate but "
                "provide valid uncertainty quantification."
            )
        advanced_params["lmm_cov_penalty_weight"] = 0.0 # Default if not LMM
        if "Linear Mixed Model" in model_display:
            advanced_params["lmm_cov_penalty_weight"] = st.number_input(
                "LMM Covariance L2 Penalty Weight",
                min_value=0.0,
                value=0.0,
                step=0.001,
                format="%.4f",
                key="cluster_continuous_lmm_penalty",
                help="L2 penalty weight for LMM random effects covariance structure. Helps stabilize model fitting, especially with few clusters or complex structures. 0.0 means no penalty. Small positive values (e.g., 0.001, 0.01) can sometimes help convergence or prevent singular fits. Use with caution."
            )

        model_map = {
            "T-test (cluster-level)": "ttest",
            "Linear Mixed Model (REML)": "mixedlm",
            "GEE (Exchangeable)": "gee",
            "Bayesian (Stan)": "bayes",
            "Bayesian (Stan) - Not Available": "bayes",  # Will fall back automatically
            "Bayesian (PyMC)": "bayes",
            "Bayesian (PyMC) - Not Available": "bayes",  # Will fall back automatically
            "Bayesian (Variational) - Fast": "bayes",
            "Bayesian (ABC) - Lightweight": "bayes",
        }
        advanced_params["analysis_model"] = model_map[model_display]
        
        # Set Bayesian backend based on selection
        if "Bayesian (PyMC)" in model_display:
            advanced_params["bayes_backend"] = "pymc"
        elif "Variational" in model_display:
            advanced_params["bayes_backend"] = "variational"
        elif "ABC" in model_display:
            advanced_params["bayes_backend"] = "abc"
        else:
            advanced_params["bayes_backend"] = "stan"  # Default to Stan
        
        # Model-specific options
        if advanced_params["analysis_model"] == "mixedlm":
            advanced_params["use_satterthwaite"] = st.checkbox(
                "Use Satterthwaite approximation for degrees of freedom",
                value=False,
                key="cluster_continuous_satt",
                help="Applies Satterthwaite adjustment which can improve type-I error control with a moderate number of clusters (< ~40)."
            )
            # Optimizer selection
            optim = st.selectbox(
                "LMM Optimizer",
                ["auto", "lbfgs", "powell", "cg", "bfgs", "newton", "nm"],
                index=0,
                key="cluster_continuous_lmm_opt",
                help="Choose the optimizer for the mixed-model fit. 'auto' tries several in order until one converges."
            )
            advanced_params["lmm_method"] = optim
            advanced_params["lmm_reml"] = st.checkbox(
                "Use REML (vs ML)",
                value=True,
                key="cluster_continuous_lmm_reml",
                help="Restricted maximum likelihood is typically preferred for variance component estimation."
            )
        elif advanced_params["analysis_model"] == "gee":
            advanced_params["use_bias_correction"] = st.checkbox(
                "Use small-sample bias correction (Mancl & DeRouen)",
                value=False,
                key="cluster_continuous_bias_corr",
                help="Bias-reduced sandwich covariance estimator to mitigate downward bias when the number of clusters is small (< ~50)."
            )
        elif advanced_params["analysis_model"] == "bayes":
            colb1, colb2 = st.columns(2)
            with colb1:
                advanced_params["bayes_draws"] = st.number_input(
                    "Posterior draws",
                    min_value=100,
                    max_value=5000,
                    value=500,
                    step=100,
                    key="cluster_continuous_bayes_draws",
                )
            with colb2:
                advanced_params["bayes_warmup"] = st.number_input(
                    "Warm-up iterations",
                    min_value=100,
                    max_value=5000,
                    value=500,
                    step=100,
                    key="cluster_continuous_bayes_warmup",
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
                key="cluster_continuous_bayes_inference",
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
    
    return advanced_params


def render_cluster_continuous(calc_type, hypothesis_type):
    """
    Render the UI for Cluster RCT with continuous outcome.
    
    Args:
        calc_type: String indicating calculation type (Sample Size, Power, or Minimum Detectable Effect)
        hypothesis_type: String indicating hypothesis type (Superiority or Non-Inferiority)
    
    Returns:
        dict: Parameters collected from the UI
    """
    params = {}
    params["calc_type"] = calc_type
    params["hypothesis_type"] = hypothesis_type
    params["outcome_type"] = "Continuous Outcome"
    
    st.markdown("### Study Parameters")
    
    # For continuous outcomes in cluster RCTs
    col1, col2 = st.columns(2)
    
    with col1:
        # Cluster parameters
        st.markdown("#### Cluster Parameters")
        
        params["icc"] = st.number_input(
            "Intracluster Correlation Coefficient (ICC)", 
            value=0.05, 
            min_value=0.0, 
            max_value=1.0, 
            format="%f", 
            help="Correlation between individuals within the same cluster",
            key="cluster_cont_icc"
        )
        
        if calc_type == "Sample Size":
            params["determine_ss_param"] = st.radio(
                "Determine which sample size parameter:",
                ("Number of Clusters (k)", "Average Cluster Size (m)"),
                key="cluster_cont_determine_ss_radio",
                horizontal=True,
                index=0, # Default to determining Number of Clusters
                help="Select whether to calculate the number of clusters (given average size) or the average cluster size (given number of clusters)."
            )

            if params["determine_ss_param"] == "Number of Clusters (k)":
                params["cluster_size_input_for_k_calc"] = st.number_input(
                    "Average Cluster Size (m)", 
                    value=20, 
                    min_value=2, 
                    key="cluster_cont_m_for_k_calc",
                    help="Assumed average number of individuals per cluster."
                )
                # k will be the output
            elif params["determine_ss_param"] == "Average Cluster Size (m)":
                params["n_clusters_input_for_m_calc"] = st.number_input(
                    "Number of Clusters per Arm (k)", 
                    min_value=2, 
                    value=10, 
                    key="cluster_cont_k_for_m_calc",
                    help="Assumed number of clusters in each treatment arm."
                )
                # m will be the output
            
            params["power"] = st.slider(
                "Power (1-β)", 
                min_value=0.7, 
                max_value=0.99, 
                value=0.8, 
                step=0.01, 
                format="%0.2f",
                key="cluster_cont_power_ss"
            )
        elif calc_type == "Power":
            # Original inputs for Power calculation - k and m are both inputs
            params["cluster_size"] = st.number_input(
                "Average Cluster Size (m)", 
                value=20, 
                min_value=2, 
                key="cluster_cont_m_for_power_calc",
                help="Average number of individuals per cluster"
            )
            params["n_clusters"] = st.number_input(
                "Number of Clusters per Arm (k)", 
                min_value=2, 
                value=15,
                key="cluster_cont_k_for_power_calc",
                help="Number of clusters in each treatment arm"
            )
        elif calc_type == "Minimum Detectable Effect":
            # Original inputs for MDE calculation - k and m are both inputs
            params["cluster_size"] = st.number_input(
                "Average Cluster Size (m)", 
                value=20, 
                min_value=2, 
                key="cluster_cont_m_for_mde_calc",
                help="Average number of individuals per cluster"
            )
            params["n_clusters"] = st.number_input(
                "Number of Clusters per Arm (k)", 
                min_value=2, 
                value=15,
                key="cluster_cont_k_for_mde_calc",
                help="Number of clusters in each treatment arm"
            )
            params["power"] = st.slider(
                "Power (1-β)", 
                min_value=0.7, 
                max_value=0.99, 
                value=0.8, 
                step=0.01, 
                format="%0.2f",
                key="cluster_cont_power_mde"
            )
    
    with col2:
        # Outcome parameters
        st.markdown("#### Outcome Parameters")
        if hypothesis_type == "Superiority":
            if calc_type != "Minimum Detectable Effect":
                params["mean1"] = st.number_input(
                    "Mean Outcome in Control Group", 
                    value=0.0, 
                    format="%f",
                    key="cluster_cont_mean1_sup"
                )
                params["mean2"] = st.number_input(
                    "Mean Outcome in Intervention Group", 
                    value=0.5, 
                    format="%f",
                    key="cluster_cont_mean2_sup"
                )
            else:  # Minimum Detectable Effect
                params["mean1"] = st.number_input(
                    "Mean Outcome in Control Group", 
                    value=0.0, 
                    format="%f",
                    key="cluster_cont_mean1_mde"
                )
                # mean2 will be calculated
            
            params["std_dev"] = st.number_input(
                "Standard Deviation", 
                value=1.0, 
                min_value=0.01, 
                format="%f",
                key="cluster_cont_std_sup"
            )
        elif hypothesis_type == "Non-Inferiority":
            params["mean1"] = st.number_input(
                "Mean Outcome in Control Group", 
                value=0.0, 
                format="%f",
                key="cluster_cont_mean1_ni"
            )
            
            params["non_inferiority_margin"] = st.number_input(
                "Non-Inferiority Margin", 
                value=0.2, 
                step=0.1, 
                help="The maximum acceptable difference showing non-inferiority",
                key="cluster_cont_nim"
            )
            
            params["assumed_difference"] = st.number_input(
                "Assumed True Difference", 
                value=0.0, 
                step=0.1,
                help="The assumed true difference between treatments (0 = treatments equivalent)",
                key="cluster_cont_assumed_diff"
            )
            
            params["non_inferiority_direction"] = st.selectbox(
                "Direction",
                ["lower", "upper"],
                index=0,
                help="Lower: smaller values are better (e.g., pain scores). Upper: larger values are better (e.g., quality of life)",
                key="cluster_cont_direction"
            )
            
            params["std_dev"] = st.number_input(
                "Standard Deviation", 
                value=1.0, 
                min_value=0.01, 
                format="%f",
                key="cluster_cont_std_ni"
            )
        
        # Significance level
        params["alpha"] = st.select_slider(
            "Significance Level (α)", 
            options=[0.001, 0.01, 0.05, 0.1], 
            value=0.05
        )
    
    # Advanced options in an expander
    with st.expander("Advanced Options"):
        advanced_params = render_continuous_advanced_options()
        params.update(advanced_params)
    
    return params