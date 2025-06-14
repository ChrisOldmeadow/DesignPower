"""Calculation functions for Cluster RCT designs.

This module contains the core calculation functions for cluster randomized
controlled trials with continuous and binary outcomes, extracted from the
main cluster_rct.py component module.
"""

import streamlit as st
import numpy as np

# Import specific analytical and simulation modules
from core.designs.cluster_rct import analytical_continuous
from core.designs.cluster_rct import simulation_continuous
from core.designs.cluster_rct import analytical_binary
from core.designs.cluster_rct import simulation_binary


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
            if params.get("hypothesis_type") == "Non-Inferiority":
                required_params = ["mean1", "non_inferiority_margin", "assumed_difference", "std_dev", "icc", "power", "alpha"]
            else:
                required_params = ["mean1", "mean2", "std_dev", "icc", "power", "alpha"]
            if params.get("determine_ss_param") == "Number of Clusters (k)":
                required_params.append("cluster_size_input_for_k_calc")
            elif params.get("determine_ss_param") == "Average Cluster Size (m)":
                required_params.append("n_clusters_input_for_m_calc")
        elif calc_type == "Power":
            if params.get("hypothesis_type") == "Non-Inferiority":
                required_params = ["n_clusters", "cluster_size", "icc", "mean1", "non_inferiority_margin", "assumed_difference", "std_dev", "alpha"]
            else:
                required_params = ["n_clusters", "cluster_size", "icc", "mean1", "mean2", "std_dev", "alpha"]
        elif calc_type == "Minimum Detectable Effect":
            if params.get("hypothesis_type") == "Non-Inferiority":
                required_params = ["n_clusters", "cluster_size", "icc", "mean1", "non_inferiority_margin", "assumed_difference", "std_dev", "power", "alpha"]
            else:
                required_params = ["n_clusters", "cluster_size", "icc", "mean1", "std_dev", "power", "alpha"]
        
        # Validate required parameters
        for param in required_params:
            if params.get(param) is None:
                return {"error": f"Missing required parameter: {param}"}
        
        # Calculate mean2 for non-inferiority scenarios
        if params.get("hypothesis_type") == "Non-Inferiority" and calc_type in ["Power", "Sample Size", "Minimum Detectable Effect"]:
            mean1 = params["mean1"]
            non_inferiority_margin = params["non_inferiority_margin"]
            assumed_difference = params["assumed_difference"]
            direction = params.get("non_inferiority_direction", "lower")
            
            # Calculate mean2 based on non-inferiority parameters
            # For proper non-inferiority sample size calculation, we need to use the effective delta
            if calc_type == "Sample Size":
                # For sample size, calculate the difference we need to detect
                if direction == "lower":
                    # Testing that new treatment is not worse than control by more than margin
                    # Effective delta = assumed_difference + non_inferiority_margin
                    effective_difference = assumed_difference + non_inferiority_margin
                else:
                    # Testing that new treatment is not better than control by more than margin
                    # Effective delta = non_inferiority_margin - assumed_difference
                    effective_difference = non_inferiority_margin - assumed_difference
                
                mean2 = mean1 + effective_difference
            else:
                # For power and MDE calculations, use the assumed difference
                mean2 = mean1 + assumed_difference
            
            # Add calculated mean2 to params for use in function calls
            params["mean2"] = mean2
        
        # Call appropriate function based on calculation type and method
        if calc_type == "Sample Size":
            if method == "analytical":
                if params.get("determine_ss_param") == "Number of Clusters (k)":
                    results = analytical_continuous.sample_size_continuous(
                        mean1=params["mean1"],
                        mean2=params["mean2"],
                        std_dev=params["std_dev"],
                        icc=params["icc"],
                        cluster_size=params["cluster_size_input_for_k_calc"],
                        power=params["power"],
                        alpha=params["alpha"]
                    )
                elif params.get("determine_ss_param") == "Average Cluster Size (m)":
                    results = analytical_continuous.sample_size_continuous(
                        mean1=params["mean1"],
                        mean2=params["mean2"],
                        std_dev=params["std_dev"],
                        icc=params["icc"],
                        cluster_size=None,
                        n_clusters_fixed=params["n_clusters_input_for_m_calc"],
                        power=params["power"],
                        alpha=params["alpha"]
                    )
            elif method == "permutation":
                if params.get("determine_ss_param") == "Number of Clusters (k)":
                    results = analytical_continuous.sample_size_continuous_permutation(
                        mean1=params["mean1"],
                        mean2=params["mean2"],
                        std_dev=params["std_dev"],
                        icc=params["icc"],
                        cluster_size=params["cluster_size_input_for_k_calc"],
                        power=params["power"],
                        alpha=params["alpha"]
                    )
                elif params.get("determine_ss_param") == "Average Cluster Size (m)":
                    results = analytical_continuous.sample_size_continuous_permutation(
                        mean1=params["mean1"],
                        mean2=params["mean2"],
                        std_dev=params["std_dev"],
                        icc=params["icc"],
                        cluster_size=None,
                        n_clusters_fixed=params["n_clusters_input_for_m_calc"],
                        power=params["power"],
                        alpha=params["alpha"]
                    )
            else:  # simulation
                progress_bar = st.progress(0.0)
                
                def _update_progress(i, total):
                    progress_bar.progress(i / total)
                
                if params.get("determine_ss_param") == "Number of Clusters (k)":
                    results = simulation_continuous.sample_size_continuous_sim(
                        mean1=params["mean1"],
                        mean2=params["mean2"],
                        std_dev=params["std_dev"],
                        icc=params["icc"],
                        cluster_size=params["cluster_size_input_for_k_calc"],
                        power=params["power"],
                        alpha=params["alpha"],
                        nsim=params.get("nsim", 1000),
                        seed=params.get("seed", 42),
                        analysis_model=params.get("analysis_model", "ttest"),
                        use_satterthwaite=params.get("use_satterthwaite", False),
                        use_bias_correction=params.get("use_bias_correction", False),
                        bayes_draws=params.get("bayes_draws", 500),
                        bayes_warmup=params.get("bayes_warmup", 500),
                        bayes_inference_method=params.get("bayes_inference_method", "credible_interval"),
                        lmm_method=params.get("lmm_method", "auto"),
                        lmm_reml=params.get("lmm_reml", True),
                        lmm_cov_penalty_weight=params.get("lmm_cov_penalty_weight", 0.0),
                        progress_callback=_update_progress
                    )
                elif params.get("determine_ss_param") == "Average Cluster Size (m)":
                    results = simulation_continuous.sample_size_continuous_sim(
                        mean1=params["mean1"],
                        mean2=params["mean2"],
                        std_dev=params["std_dev"],
                        icc=params["icc"],
                        cluster_size=None,
                        n_clusters_fixed=params["n_clusters_input_for_m_calc"],
                        power=params["power"],
                        alpha=params["alpha"],
                        nsim=params.get("nsim", 1000),
                        seed=params.get("seed", 42),
                        analysis_model=params.get("analysis_model", "ttest"),
                        use_satterthwaite=params.get("use_satterthwaite", False),
                        use_bias_correction=params.get("use_bias_correction", False),
                        bayes_draws=params.get("bayes_draws", 500),
                        bayes_warmup=params.get("bayes_warmup", 500),
                        bayes_inference_method=params.get("bayes_inference_method", "credible_interval"),
                        lmm_method=params.get("lmm_method", "auto"),
                        lmm_reml=params.get("lmm_reml", True),
                        lmm_cov_penalty_weight=params.get("lmm_cov_penalty_weight", 0.0),
                        progress_callback=_update_progress
                    )
                progress_bar.empty()
        
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
            elif method == "permutation":
                progress_bar = st.progress(0.0)
                
                def _update_progress(i, total):
                    progress_bar.progress(i / total)
                
                results = analytical_continuous.power_continuous_permutation(
                    n_clusters=params["n_clusters"],
                    cluster_size=params["cluster_size"],
                    icc=params["icc"],
                    mean1=params["mean1"],
                    mean2=params["mean2"],
                    std_dev=params["std_dev"],
                    alpha=params["alpha"],
                    progress_callback=_update_progress
                )
                progress_bar.empty()
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
                    bayes_inference_method=params.get("bayes_inference_method", "credible_interval"),
                    bayes_backend=params.get("bayes_backend", "stan"),
                    lmm_method=params.get("lmm_method", "auto"),
                    lmm_reml=params.get("lmm_reml", True),
                    lmm_cov_penalty_weight=params.get("lmm_cov_penalty_weight", 0.0),
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
            elif method == "permutation":
                results = analytical_continuous.min_detectable_effect_continuous_permutation(
                    n_clusters=params["n_clusters"],
                    cluster_size=params["cluster_size"],
                    icc=params["icc"],
                    std_dev=params["std_dev"],
                    power=params["power"],
                    alpha=params["alpha"]
                )
            else:  # simulation
                progress_bar = st.progress(0.0)
                
                def _update_progress(i, total):
                    progress_bar.progress(i / total)
                
                results = simulation_continuous.min_detectable_effect_continuous_sim(
                    mean1=params["mean1"],
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
                    bayes_inference_method=params.get("bayes_inference_method", "credible_interval"),
                    bayes_backend=params.get("bayes_backend", "stan"),
                    lmm_method=params.get("lmm_method", "auto"),
                    lmm_reml=params.get("lmm_reml", True),
                    progress_callback=_update_progress,
                )
                progress_bar.empty()

                # Check for Bayesian MDE simulation failure and fallback
                if params.get("analysis_model") == "bayes" and \
                   (results.get("mde") is None or results.get("error")):
                    warning_message = (
                        "Bayesian MDE simulation failed to converge or returned an error. "
                        "Falling back to analytical MDE calculation. "
                        "The analytical result may differ and does not account for "
                        "Bayesian posterior uncertainty."
                    )
                    st.warning(warning_message)
                    
                    # Store nsim from the attempted simulation, if available
                    nsim_attempted = results.get("nsim", params.get("nsim", 1000))

                    results = analytical_continuous.min_detectable_effect_continuous(
                        n_clusters=params["n_clusters"],
                        cluster_size=params["cluster_size"],
                        icc=params["icc"],
                        std_dev=params["std_dev"],
                        power=params["power"],
                        alpha=params["alpha"]
                    )
                    results["warning"] = warning_message
                    results["nsim"] = nsim_attempted # Preserve nsim from the sim attempt
        
        # Add calculation method and design method to results
        results["design_method"] = "Cluster RCT"
        
        # Check for infinity results and enhance warnings
        import math
        main_result_keys = ['n_clusters', 'cluster_size', 'power', 'mde']
        has_infinity = any(
            key in results and isinstance(results[key], (int, float)) and math.isinf(results[key])
            for key in main_result_keys
        )
        
        if has_infinity and 'warning' in results:
            # Make the warning more prominent for infinity cases
            results['calculation_status'] = 'warning'
            results['warning_level'] = 'high'
            # Add a user-friendly summary
            if calc_type == "Sample Size" and params.get("determine_ss_param") == "Average Cluster Size (m)":
                results['user_message'] = (
                    f"Cannot determine cluster size with the given constraints. "
                    f"The required cluster size would be infinite. See detailed guidance below."
                )
            elif calc_type == "Sample Size":
                results['user_message'] = (
                    f"Cannot determine number of clusters with the given constraints. "
                    f"The required number of clusters would be infinite. See detailed guidance below."
                )
        
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
            required_params = ["p1", "p2", "icc", "power", "alpha"]
        elif calc_type == "Power":
            required_params = ["n_clusters", "cluster_size", "icc", "p1", "p2", "alpha"]
        elif calc_type == "Minimum Detectable Effect":
            required_params = ["n_clusters", "cluster_size", "icc", "p1", "power", "alpha", "effect_measure"]
        
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
        
        # Map UI analysis_method to backend keyword for simulations
        ui_analysis_method = params.get("analysis_method_ui", params.get("analysis_method", "Design Effect Adjusted Z-test"))
        if ui_analysis_method == "Design Effect Adjusted Z-test":
            backend_analysis_method = "deff_ztest"
        elif ui_analysis_method == "T-test on Aggregate Data":
            backend_analysis_method = "aggregate_ttest"
        elif ui_analysis_method == "GLMM (Individual-Level)":
            backend_analysis_method = "glmm"
        elif ui_analysis_method == "GEE (Individual-Level)":
            backend_analysis_method = "gee"
        elif ui_analysis_method in [
            "Bayesian (Stan)",
            "Bayesian (Stan) - Not Available",
            "Bayesian (PyMC)",
            "Bayesian (PyMC) - Not Available",
            "Bayesian (Variational) - Fast",
            "Bayesian (ABC) - Lightweight"
        ]:
            backend_analysis_method = "bayes"
        else:
            backend_analysis_method = "deff_ztest"  # Default fallback
        
        # Extract Bayesian parameters if using Bayesian analysis
        bayes_backend = "stan"  # Default
        bayes_draws = params.get("bayes_draws", 500)
        bayes_warmup = params.get("bayes_warmup", 500)
        bayes_inference_method = params.get("bayes_inference_method", "credible_interval")
        
        # Map UI Bayesian method to backend
        if "Bayesian (PyMC)" in ui_analysis_method:
            bayes_backend = "pymc"
        elif "Variational" in ui_analysis_method:
            bayes_backend = "variational"
        elif "ABC" in ui_analysis_method:
            bayes_backend = "abc"
        elif "Bayesian (Stan)" in ui_analysis_method:
            bayes_backend = "stan"
        
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
            elif method == "permutation":
                results = analytical_binary.sample_size_binary_permutation(
                    p1=params["p1"],
                    p2=params["p2"],
                    icc=icc,
                    cluster_size=params["cluster_size"],
                    power=params["power"],
                    alpha=params["alpha"],
                    cv_cluster_size=cv_cluster_size
                )
            else:  # simulation
                progress_bar = st.progress(0.0)
                
                def _update_progress(i, total):
                    progress_bar.progress(i / total)
                
                results = simulation_binary.sample_size_binary_sim(
                    p1=params["p1"],
                    p2=params["p2"],
                    icc=icc,
                    cluster_size=params["cluster_size"],
                    power=params["power"],
                    alpha=params["alpha"],
                    nsim=params.get("nsim", 1000),
                    seed=params.get("seed", 42),
                    cv_cluster_size=cv_cluster_size,
                    analysis_method=backend_analysis_method,
                    bayes_backend=bayes_backend,
                    bayes_draws=bayes_draws,
                    bayes_warmup=bayes_warmup,
                    bayes_inference_method=bayes_inference_method,
                    progress_callback=_update_progress
                )
                progress_bar.empty()
            
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
                            cv_cluster_size=cv_cluster_size,
                            analysis_method=backend_analysis_method,
                            bayes_backend=bayes_backend,
                            bayes_draws=bayes_draws,
                            bayes_warmup=bayes_warmup,
                            bayes_inference_method=bayes_inference_method
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
            elif method == "permutation":
                progress_bar = st.progress(0.0)
                
                def _update_progress(i, total):
                    progress_bar.progress(i / total)
                
                results = analytical_binary.power_binary_permutation(
                    n_clusters=params["n_clusters"],
                    cluster_size=params["cluster_size"],
                    icc=icc,
                    p1=params["p1"],
                    p2=params["p2"],
                    alpha=params["alpha"],
                    cv_cluster_size=cv_cluster_size,
                    progress_callback=_update_progress
                )
                progress_bar.empty()
            else:  # simulation
                progress_bar = st.progress(0.0)
                
                def _update_progress(i, total):
                    progress_bar.progress(i / total)
                
                results = simulation_binary.power_binary_sim(
                    n_clusters=params["n_clusters"],
                    cluster_size=params["cluster_size"],
                    icc=icc,
                    p1=params["p1"],
                    p2=params["p2"],
                    alpha=params["alpha"],
                    nsim=params.get("nsim", 1000),
                    seed=params.get("seed", 42),
                    cv_cluster_size=cv_cluster_size,
                    analysis_method=backend_analysis_method,
                    bayes_backend=bayes_backend,
                    bayes_draws=bayes_draws,
                    bayes_warmup=bayes_warmup,
                    bayes_inference_method=bayes_inference_method,
                    progress_callback=_update_progress
                )
                progress_bar.empty()
            
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
                            cv_cluster_size=cv_cluster_size,
                            analysis_method=backend_analysis_method,
                            bayes_backend=bayes_backend,
                            bayes_draws=bayes_draws,
                            bayes_warmup=bayes_warmup,
                            bayes_inference_method=bayes_inference_method
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
            elif method == "permutation":
                results = analytical_binary.min_detectable_effect_binary_permutation(
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
                progress_bar = st.progress(0.0)
                
                def _update_progress(i, total):
                    progress_bar.progress(i / total)
                
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
                    effect_measure=effect_measure,
                    analysis_method=backend_analysis_method,
                    bayes_backend=bayes_backend,
                    bayes_draws=bayes_draws,
                    bayes_warmup=bayes_warmup,
                    bayes_inference_method=bayes_inference_method,
                    progress_callback=_update_progress
                )
                progress_bar.empty()
            
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
                            effect_measure=effect_measure,
                            analysis_method=backend_analysis_method,
                            bayes_backend=bayes_backend,
                            bayes_draws=bayes_draws,
                            bayes_warmup=bayes_warmup,
                            bayes_inference_method=bayes_inference_method
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