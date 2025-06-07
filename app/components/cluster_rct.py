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
import textwrap
import argparse
import sys
import os
import json
try:
    import psutil
    _PSUTIL_AVAILABLE = True
except ImportError:
    _PSUTIL_AVAILABLE = False


def _detect_resource_constraints():
    """Detect if we're in a resource-constrained environment."""
    if not _PSUTIL_AVAILABLE:
        return False  # Can't detect, assume normal resources
    
    try:
        # Check available memory (suggest lightweight if < 2GB)
        memory_gb = psutil.virtual_memory().available / (1024**3)
        
        # Check if this looks like a free hosting environment
        # (very rough heuristics)
        cpu_count = psutil.cpu_count()
        
        # Suggest lightweight methods for very constrained environments
        is_constrained = memory_gb < 2.0 or cpu_count <= 1
        
        return is_constrained
    except:
        return False  # Default to not constrained if detection fails


# For mapping UI analysis method names to backend keywords
analysis_method_map_continuous_sim = {
    "Linear Mixed Model (LMM)": "mixedlm",
    "T-test on Aggregate Data": "aggregate_ttest",
    "GEE (Individual-Level)": "gee" # Placeholder, assuming GEE might be used for continuous
}

analysis_method_map_binary_sim = {
    "Design Effect Adjusted Z-test": "deff_ztest",
    "T-test on Aggregate Data": "aggregate_ttest",
    "GLMM (Individual-Level)": "glmm",
    "GEE (Individual-Level)": "gee"
}


def generate_cli_code_cluster_continuous(params):
    """
    Generate clean, simple reproducible code for cluster RCT continuous outcome calculations.
    
    This matches the style in EXAMPLES.md for consistency and simplicity.
    """
    # Extract key parameters from UI
    calc_type = params.get('calculation_type', 'Power')
    method = params.get('method', 'analytical')
    
    # Core parameters 
    n_clusters = params.get('n_clusters', 10)
    cluster_size = params.get('cluster_size', 20)
    icc = params.get('icc', 0.05)
    mean1 = params.get('mean1', 3.0)
    mean2 = params.get('mean2', 3.5)
    std_dev = params.get('std_dev', 1.2)
    alpha = params.get('alpha', 0.05)
    power = params.get('power', 0.8)
    
    # Simulation-specific parameters
    nsim = params.get('nsim', 1000)
    seed = params.get('seed')
    
    # Analysis method and Bayesian parameters
    analysis_model = params.get('analysis_model', 'ttest')
    
    # Map to backend method
    if analysis_model == 'bayes':
        # Extract Bayesian-specific parameters
        bayes_backend = params.get('bayes_backend', 'stan')
        bayes_draws = params.get('bayes_draws', 500)
        bayes_warmup = params.get('bayes_warmup', 500)
        bayes_inference_method = params.get('bayes_inference_method', 'credible_interval')
        
        backend_method = 'bayes'
        bayes_params = f"""    bayes_backend="{bayes_backend}",
    bayes_draws={bayes_draws},
    bayes_warmup={bayes_warmup},
    bayes_inference_method="{bayes_inference_method}","""
    else:
        backend_method = analysis_model
        bayes_params = ""
    
    # Build import statement
    if method == "analytical":
        import_line = "from core.designs.cluster_rct.analytical_continuous import"
        module_prefix = ""
    else:
        import_line = "from core.designs.cluster_rct.simulation_continuous import"
        module_prefix = ""
    
    # Build function call based on calculation type
    if calc_type == "Power":
        function_name = f"power_continuous{'_sim' if method == 'simulation' else ''}"
        
        # Build parameters for power calculation
        core_params = f"""n_clusters={n_clusters},
    cluster_size={cluster_size},
    icc={icc},
    mean1={mean1},
    mean2={mean2},
    std_dev={std_dev},
    alpha={alpha}"""
        
        if method == "simulation":
            sim_params = f"""nsim={nsim},"""
            if seed is not None:
                sim_params += f"""
    seed={seed},"""
            
            sim_params += f"""
    analysis_model="{backend_method}","""
            
            all_params = core_params + ",\n    " + sim_params.strip()
            if bayes_params:
                all_params += "\n    " + bayes_params.strip()
        else:
            all_params = core_params
            
        result_display = 'result["power"]'
        
    elif calc_type == "Sample Size":
        function_name = f"sample_size_continuous{'_sim' if method == 'simulation' else ''}"
        
        core_params = f"""mean1={mean1},
    mean2={mean2},
    std_dev={std_dev},
    icc={icc},
    cluster_size={cluster_size},
    power={power},
    alpha={alpha}"""
        
        if method == "simulation":
            sim_params = f"""nsim={nsim},"""
            if seed is not None:
                sim_params += f"""
    seed={seed},"""
                
            sim_params += f"""
    analysis_model="{backend_method}","""
            
            all_params = core_params + ",\n    " + sim_params.strip()
            if bayes_params:
                all_params += "\n    " + bayes_params.strip()
        else:
            all_params = core_params
            
        result_display = 'result["n_clusters"]'
        
    elif calc_type == "Minimum Detectable Effect":
        function_name = f"min_detectable_effect_continuous{'_sim' if method == 'simulation' else ''}"
        
        core_params = f"""n_clusters={n_clusters},
    cluster_size={cluster_size},
    icc={icc},
    std_dev={std_dev},
    power={power},
    alpha={alpha}"""
        
        if method == "simulation":
            sim_params = f"""nsim={nsim},"""
            if seed is not None:
                sim_params += f"""
    seed={seed},"""
                
            sim_params += f"""
    analysis_model="{backend_method}","""
            
            all_params = core_params + ",\n    " + sim_params.strip()
            if bayes_params:
                all_params += "\n    " + bayes_params.strip()
        else:
            all_params = core_params
            
        result_display = 'result["mde"]'
    
    # Generate clean, simple code with usage instructions
    code = f"""# Cluster RCT Continuous Outcome - {calc_type} Analysis
# Generated by DesignPower
#
# HOW TO USE THIS SCRIPT:
# 1. Save this code to a file with .py extension (e.g., 'my_analysis.py')
# 2. SETUP REQUIREMENTS:
#    - Install Python (3.8 or later)
#    - Download/clone the DesignPower codebase from GitHub
#    - Install required packages: pip install -r requirements.txt
# 3. RUN THE SCRIPT:
#    - Option A: Run from DesignPower project directory: python my_analysis.py
#    - Option B: Add DesignPower to Python path, then run from anywhere
#    - Option C: Run in Jupyter/IDE with DesignPower project as working directory
#
# The script will print the main result and full details in JSON format

{import_line} {function_name}

# Calculate {calc_type.lower()}
result = {function_name}(
    {all_params}
)

print(f"{calc_type}: {{result['{result_display.split('\"')[1]}']:.3f}}")
print(f"Design effect: {{result['design_effect']:.2f}}")

# Full results
import json
print(json.dumps(result, indent=2))"""

    return code


def generate_cli_code_cluster_binary(params):
    """
    Generate clean, simple reproducible code for cluster RCT binary outcome calculations.
    
    This matches the style in EXAMPLES.md for consistency and simplicity.
    """
    # Extract key parameters from UI
    calc_type = params.get('calc_type', 'Power')
    method = params.get('method', 'analytical')
    
    # Core parameters 
    n_clusters = params.get('n_clusters', 10)
    cluster_size = params.get('cluster_size', 20)
    icc = params.get('icc', 0.05)
    p1 = params.get('p1', 0.3)
    p2 = params.get('p2', 0.5)
    alpha = params.get('alpha', 0.05)
    power = params.get('power', 0.8)
    
    # Simulation-specific parameters
    nsim = params.get('nsim', 1000)
    seed = params.get('seed')
    
    # Analysis method and Bayesian parameters
    analysis_method = params.get('analysis_method', 'deff_ztest')
    analysis_method_ui = params.get('analysis_method_ui', '')
    
    # Map UI analysis method to backend
    if analysis_method == 'bayes':
        # Extract Bayesian-specific parameters
        bayes_backend = params.get('bayes_backend', 'stan')
        bayes_draws = params.get('bayes_draws', 500)
        bayes_warmup = params.get('bayes_warmup', 500)
        bayes_inference_method = params.get('bayes_inference_method', 'credible_interval')
        
        backend_method = 'bayes'
        bayes_params = f"""    bayes_backend="{bayes_backend}",
    bayes_draws={bayes_draws},
    bayes_warmup={bayes_warmup},
    bayes_inference_method="{bayes_inference_method}","""
    else:
        backend_method = analysis_method
        bayes_params = ""
    
    # Build import statement
    if method == "analytical":
        import_line = "from core.designs.cluster_rct.analytical_binary import"
        module_prefix = ""
    else:
        import_line = "from core.designs.cluster_rct.simulation_binary import"
        module_prefix = ""
    
    # Build function call based on calculation type
    if calc_type == "Power":
        function_name = f"power_binary{'_sim' if method == 'simulation' else ''}"
        
        # Build parameters for power calculation
        core_params = f"""n_clusters={n_clusters},
    cluster_size={cluster_size},
    icc={icc},
    p1={p1},
    p2={p2},
    alpha={alpha}"""
        
        if method == "simulation":
            sim_params = f"""nsim={nsim},"""
            if seed is not None:
                sim_params += f"""
    seed={seed},"""
            
            sim_params += f"""
    analysis_method="{backend_method}","""
            
            all_params = core_params + ",\n    " + sim_params.strip()
            if bayes_params:
                all_params += "\n    " + bayes_params.strip()
        else:
            all_params = core_params
            
        result_display = 'result["power"]'
        
    elif calc_type == "Sample Size":
        function_name = f"sample_size_binary{'_sim' if method == 'simulation' else ''}"
        
        core_params = f"""p1={p1},
    p2={p2},
    icc={icc},
    cluster_size={cluster_size},
    power={power},
    alpha={alpha}"""
        
        if method == "simulation":
            sim_params = f"""nsim={nsim},"""
            if seed is not None:
                sim_params += f"""
    seed={seed},"""
                
            sim_params += f"""
    analysis_method="{backend_method}","""
            
            all_params = core_params + ",\n    " + sim_params.strip()
            if bayes_params:
                all_params += "\n    " + bayes_params.strip()
        else:
            all_params = core_params
            
        result_display = 'result["n_clusters"]'
        
    elif calc_type == "Minimum Detectable Effect":
        function_name = f"min_detectable_effect_binary{'_sim' if method == 'simulation' else ''}"
        
        core_params = f"""n_clusters={n_clusters},
    cluster_size={cluster_size},
    icc={icc},
    p1={p1},
    power={power},
    alpha={alpha}"""
        
        if method == "simulation":
            sim_params = f"""nsim={nsim},"""
            if seed is not None:
                sim_params += f"""
    seed={seed},"""
                
            sim_params += f"""
    analysis_method="{backend_method}","""
            
            all_params = core_params + ",\n    " + sim_params.strip()
            if bayes_params:
                all_params += "\n    " + bayes_params.strip()
        else:
            all_params = core_params
            
        result_display = 'result["mde"]'
    
    # Generate clean, simple code with usage instructions
    code = f"""# Cluster RCT Binary Outcome - {calc_type} Analysis
# Generated by DesignPower
#
# HOW TO USE THIS SCRIPT:
# 1. Save this code to a file with .py extension (e.g., 'my_analysis.py')
# 2. SETUP REQUIREMENTS:
#    - Install Python (3.8 or later)
#    - Download/clone the DesignPower codebase from GitHub
#    - Install required packages: pip install -r requirements.txt
# 3. RUN THE SCRIPT:
#    - Option A: Run from DesignPower project directory: python my_analysis.py
#    - Option B: Add DesignPower to Python path, then run from anywhere
#    - Option C: Run in Jupyter/IDE with DesignPower project as working directory
#
# The script will print the main result and full details in JSON format

{import_line} {function_name}

# Calculate {calc_type.lower()}
result = {function_name}(
    {all_params}
)

print(f"{calc_type}: {{result['{result_display.split('"')[1]}']:.3f}}")
print(f"Design effect: {{result['design_effect']:.2f}}")

# Full results
import json
print(json.dumps(result, indent=2))"""

    return code


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
                "🌐 **Resource-Constrained Environment Detected**\\n\\n"
                "For optimal performance in this environment, consider:\\n"
                "• **Bayesian (ABC) - Lightweight** for basic Bayesian inference\\n"
                "• **Bayesian (Variational) - Fast** for faster approximate inference\\n"
                "• **Design Effect Adjusted Z-test** for fastest non-Bayesian analysis\\n\\n"
                "Full MCMC methods (Stan/PyMC) may be slow or unavailable."
            )
        elif not stan_available and not pymc_available:
            st.info(
                "💡 **Bayesian Analysis Available**\\n\\n"
                "Stan/PyMC not detected, but approximate Bayesian methods are available:\\n"
                "• **Bayesian (Variational) - Fast** for quick approximate inference\\n"
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
                "📦 **Stan backend requires additional installation**\\n\\n"
                "To use Stan for Bayesian analysis, please install cmdstanpy:\\n"
                "```bash\\n"
                "pip install cmdstanpy\\n"
                "```\\n"
                "The calculation will fall back to Design Effect Z-test if you proceed."
            )
        elif "Bayesian (PyMC)" in selected_model_display_binary and not pymc_available:
            st.error(
                "📦 **PyMC backend requires additional installation**\\n\\n"
                "To use PyMC for Bayesian analysis, please install pymc:\\n"
                "```bash\\n"
                "pip install pymc\\n"
                "```\\n"
                "The calculation will fall back to variational approximation if you proceed."
            )
        elif "Variational" in selected_model_display_binary:
            st.info(
                "⚡ **Fast Variational Bayes**\\n\\n"
                "Uses Laplace approximation for fast approximate Bayesian inference on logit scale. "
                "Results are approximate but much faster than full MCMC. "
                "Good for initial exploration or resource-constrained environments."
            )
        elif "ABC" in selected_model_display_binary:
            st.info(
                "🎯 **Approximate Bayesian Computation**\\n\\n"
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
            
            st.info(f"🔧 **Bayesian Backend**: {backend_name} ({sampling_type})\\n\\n{resource_note}")
            
            # Show limitations for approximate methods
            if backend in ["variational", "abc"]:
                st.warning(
                    "⚠️ **Approximate Method Limitations**:\\n"
                    "• Results are approximate, not exact posterior samples\\n"
                    "• May underestimate uncertainty in some cases\\n"
                    "• Best used for initial exploration or resource-limited environments\\n"
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
            help="Choose the analysis model applied to each simulated dataset. The simple two-sample t-test analyses individual-level data ignoring clustering but with design-effect adjustment. Mixed models explicitly model random cluster intercepts and can provide more power when cluster counts are moderate to large. GEE provides marginal (population-averaged) inference and is robust to some model misspecification, but small-sample bias can be an issue."
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
            required_params = ["mean1", "mean2", "std_dev", "icc", "power", "alpha"]
            if params.get("determine_ss_param") == "Number of Clusters (k)":
                required_params.append("cluster_size_input_for_k_calc")
            elif params.get("determine_ss_param") == "Average Cluster Size (m)":
                required_params.append("n_clusters_input_for_m_calc")
        elif calc_type == "Power":
            required_params = ["n_clusters", "cluster_size", "icc", "mean1", "mean2", "std_dev", "alpha"]
        elif calc_type == "Minimum Detectable Effect":
            required_params = ["n_clusters", "cluster_size", "icc", "mean1", "std_dev", "power", "alpha"]
        
        # Validate required parameters
        for param in required_params:
            if params.get(param) is None:
                return {"error": f"Missing required parameter: {param}"}
        
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
            else:  # simulation
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
                    cv_cluster_size=cv_cluster_size,
                    analysis_method=backend_analysis_method,
                    bayes_backend=bayes_backend,
                    bayes_draws=bayes_draws,
                    bayes_warmup=bayes_warmup,
                    bayes_inference_method=bayes_inference_method
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
                    cv_cluster_size=cv_cluster_size,
                    analysis_method=backend_analysis_method,
                    bayes_backend=bayes_backend,
                    bayes_draws=bayes_draws,
                    bayes_warmup=bayes_warmup,
                    bayes_inference_method=bayes_inference_method
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
                    effect_measure=effect_measure,
                    analysis_method=backend_analysis_method,
                    bayes_backend=bayes_backend,
                    bayes_draws=bayes_draws,
                    bayes_warmup=bayes_warmup,
                    bayes_inference_method=bayes_inference_method
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
