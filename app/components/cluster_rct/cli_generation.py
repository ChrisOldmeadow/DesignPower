"""CLI code generation utilities for cluster RCT components.

This module provides functions to generate reproducible Python code
for cluster RCT power analysis and sample size calculations.
"""

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
    
    # Cluster size variation
    cv_cluster_size = params.get('cv_cluster_size', 0.0)
    
    # Simulation-specific parameters
    nsim = params.get('nsim', 1000)
    seed = params.get('seed')
    
    # Analysis method and model-specific parameters
    analysis_model = params.get('analysis_model', 'ttest')
    
    # Extract model-specific parameters
    model_params = ""
    
    if analysis_model == 'bayes':
        # Extract Bayesian-specific parameters
        bayes_backend = params.get('bayes_backend', 'stan')
        bayes_draws = params.get('bayes_draws', 500)
        bayes_warmup = params.get('bayes_warmup', 500)
        bayes_inference_method = params.get('bayes_inference_method', 'credible_interval')
        
        backend_method = 'bayes'
        model_params = f"""    bayes_backend="{bayes_backend}",
    bayes_draws={bayes_draws},
    bayes_warmup={bayes_warmup},
    bayes_inference_method="{bayes_inference_method}","""
    elif analysis_model == 'mixedlm':
        # Extract LMM-specific parameters
        use_satterthwaite = params.get('use_satterthwaite', False)
        lmm_method = params.get('lmm_method', 'auto')
        lmm_reml = params.get('lmm_reml', True)
        lmm_cov_penalty_weight = params.get('lmm_cov_penalty_weight', 0.0)
        
        backend_method = 'mixedlm'
        model_params = f"""    use_satterthwaite={use_satterthwaite},
    lmm_method="{lmm_method}",
    lmm_reml={lmm_reml},"""
        if lmm_cov_penalty_weight > 0:
            model_params += f"""
    lmm_cov_penalty_weight={lmm_cov_penalty_weight},"""
    elif analysis_model == 'gee':
        # Extract GEE-specific parameters
        use_bias_correction = params.get('use_bias_correction', False)
        
        backend_method = 'gee'
        model_params = f"""    use_bias_correction={use_bias_correction},"""
    else:
        backend_method = analysis_model
        model_params = ""
    
    # Build import statement
    if method == "analytical":
        import_line = "from core.designs.cluster_rct.analytical_continuous import"
        module_prefix = ""
    elif method == "permutation":
        import_line = "from core.designs.cluster_rct.analytical_continuous import"
        module_prefix = ""
    else:
        import_line = "from core.designs.cluster_rct.simulation_continuous import"
        module_prefix = ""
    
    # Build function call based on calculation type
    if calc_type == "Power":
        if method == "permutation":
            function_name = "power_continuous_permutation"
        else:
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
            if model_params:
                all_params += "\n    " + model_params.strip()
        else:
            all_params = core_params
            
        result_display = 'result["power"]'
        
    elif calc_type == "Sample Size":
        if method == "permutation":
            function_name = "sample_size_continuous_permutation"
        else:
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
            if model_params:
                all_params += "\n    " + model_params.strip()
        else:
            all_params = core_params
            
        result_display = 'result["n_clusters"]'
        
    elif calc_type == "Minimum Detectable Effect":
        if method == "permutation":
            function_name = "min_detectable_effect_continuous_permutation"
        else:
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
            if model_params:
                all_params += "\n    " + model_params.strip()
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
    
    # Cluster size variation
    cv_cluster_size = params.get('cv_cluster_size', 0.0)
    
    # Effect measure and ICC scale (for binary)
    effect_measure = params.get('effect_measure', 'risk_difference')
    icc_scale = params.get('icc_scale', 'Linear')
    
    # Simulation-specific parameters
    nsim = params.get('nsim', 1000)
    seed = params.get('seed')
    
    # Analysis method and model-specific parameters
    analysis_method = params.get('analysis_method', 'deff_ztest')
    analysis_method_ui = params.get('analysis_method_ui', '')
    
    # Extract model-specific parameters
    model_params = ""
    
    # Map UI analysis method to backend
    if analysis_method == 'bayes':
        # Extract Bayesian-specific parameters
        bayes_backend = params.get('bayes_backend', 'stan')
        bayes_draws = params.get('bayes_draws', 500)
        bayes_warmup = params.get('bayes_warmup', 500)
        bayes_inference_method = params.get('bayes_inference_method', 'credible_interval')
        
        backend_method = 'bayes'
        model_params = f"""    bayes_backend="{bayes_backend}",
    bayes_draws={bayes_draws},
    bayes_warmup={bayes_warmup},
    bayes_inference_method="{bayes_inference_method}","""
    elif analysis_method == 'glmm':
        # GLMM doesn't have additional parameters in the current implementation
        backend_method = 'glmm'
        model_params = ""
    elif analysis_method == 'gee':
        # GEE doesn't have bias correction for binary outcomes in current implementation
        backend_method = 'gee'
        model_params = ""
    else:
        backend_method = analysis_method
        model_params = ""
    
    # Build import statement
    if method == "analytical":
        import_line = "from core.designs.cluster_rct.analytical_binary import"
        module_prefix = ""
    elif method == "permutation":
        import_line = "from core.designs.cluster_rct.analytical_binary import"
        module_prefix = ""
    else:
        import_line = "from core.designs.cluster_rct.simulation_binary import"
        module_prefix = ""
    
    # Build function call based on calculation type
    if calc_type == "Power":
        if method == "permutation":
            function_name = "power_binary_permutation"
        else:
            function_name = f"power_binary{'_sim' if method == 'simulation' else ''}"
        
        # Build parameters for power calculation
        core_params = f"""n_clusters={n_clusters},
    cluster_size={cluster_size},
    icc={icc},
    p1={p1},
    p2={p2},
    alpha={alpha}"""
        
        # Add optional parameters if non-default
        if cv_cluster_size > 0:
            core_params += f",\n    cv_cluster_size={cv_cluster_size}"
        
        if method == "simulation":
            sim_params = f"""nsim={nsim},"""
            if seed is not None:
                sim_params += f"""
    seed={seed},"""
            
            sim_params += f"""
    analysis_method="{backend_method}","""
            
            all_params = core_params + ",\n    " + sim_params.strip()
            if model_params:
                all_params += "\n    " + model_params.strip()
        else:
            all_params = core_params
            
        result_display = 'result["power"]'
        
    elif calc_type == "Sample Size":
        if method == "permutation":
            function_name = "sample_size_binary_permutation"
        else:
            function_name = f"sample_size_binary{'_sim' if method == 'simulation' else ''}"
        
        core_params = f"""p1={p1},
    p2={p2},
    icc={icc},
    cluster_size={cluster_size},
    power={power},
    alpha={alpha}"""
        
        # Add optional parameters if non-default
        if cv_cluster_size > 0:
            core_params += f",\n    cv_cluster_size={cv_cluster_size}"
        
        if method == "simulation":
            sim_params = f"""nsim={nsim},"""
            if seed is not None:
                sim_params += f"""
    seed={seed},"""
                
            sim_params += f"""
    analysis_method="{backend_method}","""
            
            all_params = core_params + ",\n    " + sim_params.strip()
            if model_params:
                all_params += "\n    " + model_params.strip()
        else:
            all_params = core_params
            
        result_display = 'result["n_clusters"]'
        
    elif calc_type == "Minimum Detectable Effect":
        if method == "permutation":
            function_name = "min_detectable_effect_binary_permutation"
        else:
            function_name = f"min_detectable_effect_binary{'_sim' if method == 'simulation' else ''}"
        
        core_params = f"""n_clusters={n_clusters},
    cluster_size={cluster_size},
    icc={icc},
    p1={p1},
    power={power},
    alpha={alpha}"""
        
        # Add optional parameters if non-default
        if cv_cluster_size > 0:
            core_params += f",\n    cv_cluster_size={cv_cluster_size}"
        if effect_measure != 'risk_difference':
            core_params += f",\n    effect_measure='{effect_measure}'"
        
        if method == "simulation":
            sim_params = f"""nsim={nsim},"""
            if seed is not None:
                sim_params += f"""
    seed={seed},"""
                
            sim_params += f"""
    analysis_method="{backend_method}","""
            
            all_params = core_params + ",\n    " + sim_params.strip()
            if model_params:
                all_params += "\n    " + model_params.strip()
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