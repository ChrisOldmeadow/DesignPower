"""Component module for Cluster RCT designs.

This module provides UI rendering functions and calculation functions for
Cluster Randomized Controlled Trial designs with continuous and binary outcomes.
"""
import streamlit as st
import numpy as np
import pandas as pd
import math

# Import specific analytical and simulation modules
from core.designs.cluster_rct import analytical_continuous
from core.designs.cluster_rct import simulation_continuous
from core.designs.cluster_rct import analytical_binary
from core.designs.cluster_rct import simulation_binary

# Shared functions
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
        ["Analytical", "Simulation"],
        index=0,
        key="cluster_binary_method_radio",
        horizontal=True
    )
    
    # Convert to lowercase for function calls
    advanced_params["method"] = advanced_params["method"].lower()
    
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
        ["Analytical", "Simulation"],
        index=0,
        key="cluster_continuous_method_radio",
        horizontal=True
    )
    
    # Convert to lowercase for function calls
    advanced_params["method"] = advanced_params["method"].lower()
    
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
        model_display = st.selectbox(
            "Statistical Model Used to Analyse Each Simulated Trial",
            [
                "T-test (cluster-level)",
                "Linear Mixed Model (REML)",
                "GEE (Exchangeable)",
                "Bayesian (Stan)"
            ],
            index=0,
            key="cluster_continuous_model_select",
            help="Choose the analysis model applied to each simulated dataset. The simple two-sample t-test analyses individual-level data ignoring clustering but with design-effect adjustment. Mixed models explicitly model random cluster intercepts and can provide more power when cluster counts are moderate to large. GEE provides marginal (population-averaged) inference and is robust to some model misspecification, but small-sample bias can be an issue."
        )
        model_map = {
            "T-test (cluster-level)": "ttest",
            "Linear Mixed Model (REML)": "mixedlm",
            "GEE (Exchangeable)": "gee",
            "Bayesian (Stan)": "bayes",
        }
        advanced_params["analysis_model"] = model_map[model_display]
        
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
                params["mean1"] = st.number_input(
                    "Mean Outcome in Control Group", 
                    value=0.0, 
                    format="%f"
                )
                params["mean2"] = st.number_input(
                    "Mean Outcome in Intervention Group", 
                    value=0.5, 
                    format="%f"
                )
            else:  # Minimum Detectable Effect
                params["mean1"] = st.number_input(
                    "Mean Outcome in Control Group", 
                    value=0.0, 
                    format="%f"
                )
                # mean2 will be calculated
            
            params["std_dev"] = st.number_input(
                "Standard Deviation", 
                value=1.0, 
                min_value=0.01, 
                format="%f"
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


def calculate_cluster_continuous(params):
    """
    Calculate results for Cluster RCT with continuous outcome.
    
    Args:
        params: Dictionary of parameters
        
    Returns:
        dict: Results of calculations
    """
    # Extract parameters
    calc_type = params.get("calc_type")
    method = params.get("method", "analytical")
    
    try:
        # Check for required parameters based on calculation type
        if calc_type == "Sample Size":
            required_params = ["mean1", "mean2", "std_dev", "icc", "cluster_size", "power", "alpha"]
        elif calc_type == "Power":
            required_params = ["n_clusters", "cluster_size", "icc", "mean1", "mean2", "std_dev", "alpha"]
        elif calc_type == "Minimum Detectable Effect":
            required_params = ["n_clusters", "cluster_size", "icc", "std_dev", "power", "alpha"]
        
        # Validate required parameters
        for param in required_params:
            if params.get(param) is None:
                return {"error": f"Missing required parameter: {param}"}
        
        # Call appropriate function based on calculation type and method
        if calc_type == "Sample Size":
            if method == "analytical":
                results = analytical_continuous.sample_size_continuous(
                    mean1=params["mean1"],
                    mean2=params["mean2"],
                    std_dev=params["std_dev"],
                    icc=params["icc"],
                    cluster_size=params["cluster_size"],
                    power=params["power"],
                    alpha=params["alpha"]
                )
            else:  # simulation
                results = simulation_continuous.sample_size_continuous_sim(
                    mean1=params["mean1"],
                    mean2=params["mean2"],
                    std_dev=params["std_dev"],
                    icc=params["icc"],
                    cluster_size=params["cluster_size"],
                    power=params["power"],
                    alpha=params["alpha"],
                    nsim=params.get("nsim", 1000),
                    seed=params.get("seed", 42),
                    analysis_model=params.get("analysis_model", "ttest"),
                    use_satterthwaite=params.get("use_satterthwaite", False),
                    use_bias_correction=params.get("use_bias_correction", False),
                    bayes_draws=params.get("bayes_draws", 500),
                    bayes_warmup=params.get("bayes_warmup", 500),
                    lmm_method=params.get("lmm_method", "auto"),
                    lmm_reml=params.get("lmm_reml", True),
                )
        
        elif calc_type == "Power":
            if method == "analytical":
                results = analytical_continuous.power_continuous(
                    n_clusters=params["n_clusters"],
                    cluster_size=params["cluster_size"],
                    icc=params["icc"],
                    mean1=params["mean1"],
                    mean2=params["mean2"],
                    std_dev=params["std_dev"],
                    alpha=params["alpha"]
                )
            else:  # simulation
                progress_bar = st.progress(0.0)
                
                def _update_progress(i, total):
                    progress_bar.progress(i / total)
                
                results = simulation_continuous.power_continuous_sim(
                    n_clusters=params["n_clusters"],
                    cluster_size=params["cluster_size"],
                    icc=params["icc"],
                    mean1=params["mean1"],
                    mean2=params["mean2"],
                    std_dev=params["std_dev"],
                    alpha=params["alpha"],
                    nsim=params.get("nsim", 1000),
                    seed=params.get("seed", 42),
                    analysis_model=params.get("analysis_model", "ttest"),
                    use_satterthwaite=params.get("use_satterthwaite", False),
                    use_bias_correction=params.get("use_bias_correction", False),
                    bayes_draws=params.get("bayes_draws", 500),
                    bayes_warmup=params.get("bayes_warmup", 500),
                    lmm_method=params.get("lmm_method", "auto"),
                    lmm_reml=params.get("lmm_reml", True),
                    progress_callback=_update_progress,
                )
                progress_bar.empty()
        
        elif calc_type == "Minimum Detectable Effect":
            if method == "analytical":
                results = analytical_continuous.min_detectable_effect_continuous(
                    n_clusters=params["n_clusters"],
                    cluster_size=params["cluster_size"],
                    icc=params["icc"],
                    std_dev=params["std_dev"],
                    power=params["power"],
                    alpha=params["alpha"]
                )
            else:  # simulation
                results = simulation_continuous.min_detectable_effect_continuous_sim(
                    n_clusters=params["n_clusters"],
                    cluster_size=params["cluster_size"],
                    icc=params["icc"],
                    std_dev=params["std_dev"],
                    power=params["power"],
                    alpha=params["alpha"],
                    nsim=params.get("nsim", 1000),
                    seed=params.get("seed", 42),
                    analysis_model=params.get("analysis_model", "ttest"),
                    use_satterthwaite=params.get("use_satterthwaite", False),
                    use_bias_correction=params.get("use_bias_correction", False),
                    bayes_draws=params.get("bayes_draws", 500),
                    bayes_warmup=params.get("bayes_warmup", 500),
                    lmm_method=params.get("lmm_method", "auto"),
                    lmm_reml=params.get("lmm_reml", True),
                )
        
        # Add calculation method and design method to results
        results["design_method"] = "Cluster RCT"
        
        return results
    
    except Exception as e:
        return {"error": f"Error in calculation: {str(e)}"}


def calculate_cluster_binary(params):
    """
    Calculate results for Cluster RCT with binary outcome.
    
    Args:
        params: Dictionary of parameters
        
    Returns:
        dict: Results of calculations
    """
    # Extract parameters
    calc_type = params.get("calc_type")
    method = params.get("method", "analytical")
    icc_scale = params.get("icc_scale", "Linear")
    cv_cluster_size = params.get("cv_cluster_size", 0.0)
    effect_measure = params.get("effect_measure", "risk_difference")
    
    try:
        # Check for required parameters based on calculation type
        if calc_type == "Sample Size":
            required_params = ["p1", "p2", "icc", "cluster_size", "power", "alpha"]
        elif calc_type == "Power":
            required_params = ["n_clusters", "cluster_size", "icc", "p1", "p2", "alpha"]
        elif calc_type == "Minimum Detectable Effect":
            required_params = ["n_clusters", "cluster_size", "icc", "p1", "power", "alpha"]
        
        # Validate required parameters
        for param in required_params:
            if params.get(param) is None:
                return {"error": f"Missing required parameter: {param}"}
        
        # Process ICC based on scale
        icc = params["icc"]
        if icc_scale == "Logit":
            # Import the conversion function from cluster_utils
            from core.designs.cluster_rct.cluster_utils import convert_icc_logit_to_linear
            icc = convert_icc_logit_to_linear(icc, params["p1"])
            
        # Check if we need to run sensitivity analysis
        run_sensitivity = params.get("run_sensitivity", False)
        sensitivity_results = []
        
        if run_sensitivity:
            icc_min = params.get("icc_min", 0.01)
            icc_max = params.get("icc_max", 0.10)
            icc_steps = params.get("icc_steps", 5)
            
            # Create a range of ICC values
            icc_values = np.linspace(icc_min, icc_max, icc_steps)
            
            # Store the original ICC for the main calculation
            original_icc = icc
        
        # Call appropriate function based on calculation type and method
        if calc_type == "Sample Size":
            if method == "analytical":
                results = analytical_binary.sample_size_binary(
                    p1=params["p1"],
                    p2=params["p2"],
                    icc=icc,
                    cluster_size=params["cluster_size"],
                    power=params["power"],
                    alpha=params["alpha"],
                    cv_cluster_size=cv_cluster_size,
                    effect_measure=None  # Not needed here since p2 is provided
                )
            else:  # simulation
                results = simulation_binary.sample_size_binary_sim(
                    p1=params["p1"],
                    p2=params["p2"],
                    icc=icc,
                    cluster_size=params["cluster_size"],
                    power=params["power"],
                    alpha=params["alpha"],
                    nsim=params.get("nsim", 1000),
                    seed=params.get("seed", 42),
                    cv_cluster_size=cv_cluster_size
                )
            
            # Run sensitivity analysis if requested
            if run_sensitivity:
                for test_icc in icc_values:
                    if method == "analytical":
                        sens_result = analytical_binary.sample_size_binary(
                            p1=params["p1"],
                            p2=params["p2"],
                            icc=test_icc,
                            cluster_size=params["cluster_size"],
                            power=params["power"],
                            alpha=params["alpha"],
                            cv_cluster_size=cv_cluster_size
                        )
                    else:  # simulation
                        sens_result = simulation_binary.sample_size_binary_sim(
                            p1=params["p1"],
                            p2=params["p2"],
                            icc=test_icc,
                            cluster_size=params["cluster_size"],
                            power=params["power"],
                            alpha=params["alpha"],
                            nsim=params.get("nsim", 1000),
                            seed=params.get("seed", 42),
                            cv_cluster_size=cv_cluster_size
                        )
                    sensitivity_results.append({
                        "icc": test_icc,
                        "n_clusters": sens_result["n_clusters"],
                        "total_n": sens_result["total_n"],
                        "design_effect": sens_result["design_effect"]
                    })
        
        elif calc_type == "Power":
            if method == "analytical":
                results = analytical_binary.power_binary(
                    n_clusters=params["n_clusters"],
                    cluster_size=params["cluster_size"],
                    icc=icc,
                    p1=params["p1"],
                    p2=params["p2"],
                    alpha=params["alpha"],
                    cv_cluster_size=cv_cluster_size
                )
            else:  # simulation
                results = simulation_binary.power_binary_sim(
                    n_clusters=params["n_clusters"],
                    cluster_size=params["cluster_size"],
                    icc=icc,
                    p1=params["p1"],
                    p2=params["p2"],
                    alpha=params["alpha"],
                    nsim=params.get("nsim", 1000),
                    seed=params.get("seed", 42),
                    cv_cluster_size=cv_cluster_size
                )
            
            # Run sensitivity analysis if requested
            if run_sensitivity:
                for test_icc in icc_values:
                    if method == "analytical":
                        sens_result = analytical_binary.power_binary(
                            n_clusters=params["n_clusters"],
                            cluster_size=params["cluster_size"],
                            icc=test_icc,
                            p1=params["p1"],
                            p2=params["p2"],
                            alpha=params["alpha"],
                            cv_cluster_size=cv_cluster_size
                        )
                    else:  # simulation
                        sens_result = simulation_binary.power_binary_sim(
                            n_clusters=params["n_clusters"],
                            cluster_size=params["cluster_size"],
                            icc=test_icc,
                            p1=params["p1"],
                            p2=params["p2"],
                            alpha=params["alpha"],
                            nsim=params.get("nsim", 1000),
                            seed=params.get("seed", 42),
                            cv_cluster_size=cv_cluster_size
                        )
                    sensitivity_results.append({
                        "icc": test_icc,
                        "power": sens_result["power"],
                        "design_effect": sens_result["design_effect"]
                    })
        
        elif calc_type == "Minimum Detectable Effect":
            if method == "analytical":
                results = analytical_binary.min_detectable_effect_binary(
                    n_clusters=params["n_clusters"],
                    cluster_size=params["cluster_size"],
                    icc=icc,
                    p1=params["p1"],
                    power=params["power"],
                    alpha=params["alpha"],
                    cv_cluster_size=cv_cluster_size,
                    effect_measure=effect_measure
                )
            else:  # simulation
                results = simulation_binary.min_detectable_effect_binary_sim(
                    n_clusters=params["n_clusters"],
                    cluster_size=params["cluster_size"],
                    icc=icc,
                    p1=params["p1"],
                    power=params["power"],
                    alpha=params["alpha"],
                    nsim=params.get("nsim", 1000),
                    seed=params.get("seed", 42),
                    cv_cluster_size=cv_cluster_size,
                    effect_measure=effect_measure
                )
            
            # Run sensitivity analysis if requested
            if run_sensitivity:
                for test_icc in icc_values:
                    if method == "analytical":
                        sens_result = analytical_binary.min_detectable_effect_binary(
                            n_clusters=params["n_clusters"],
                            cluster_size=params["cluster_size"],
                            icc=test_icc,
                            p1=params["p1"],
                            power=params["power"],
                            alpha=params["alpha"],
                            cv_cluster_size=cv_cluster_size,
                            effect_measure=effect_measure
                        )
                    else:  # simulation
                        sens_result = simulation_binary.min_detectable_effect_binary_sim(
                            n_clusters=params["n_clusters"],
                            cluster_size=params["cluster_size"],
                            icc=test_icc,
                            p1=params["p1"],
                            power=params["power"],
                            alpha=params["alpha"],
                            nsim=params.get("nsim", 1000),
                            seed=params.get("seed", 42),
                            cv_cluster_size=cv_cluster_size,
                            effect_measure=effect_measure
                        )
                    sensitivity_results.append({
                        "icc": test_icc,
                        "mde": sens_result["mde"],
                        "design_effect": sens_result["design_effect"]
                    })
        
        # Add calculation method and design method to results
        results["design_method"] = "Cluster RCT"
        results["calculation_method"] = method
        
        # Add ICC scale information
        results["icc_original"] = params["icc"]
        results["icc_scale_original"] = icc_scale
        if icc_scale == "Logit":
            results["icc_converted"] = icc
            results["icc_conversion_note"] = f"ICC converted from logit scale ({params['icc']}) to linear scale ({icc:.4f})"
            
        # Add sensitivity analysis results if available
        if run_sensitivity:
            results["sensitivity_analysis"] = {
                "icc_range": [float(icc) for icc in icc_values],
                "results": sensitivity_results
            }
            
        return results
    
    except Exception as e:
        return {"error": f"Error in calculation: {str(e)}"}
