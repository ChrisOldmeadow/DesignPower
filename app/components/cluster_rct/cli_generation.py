"""CLI code generation utilities for cluster RCT components.

This module provides functions to generate reproducible Python code
for cluster RCT power analysis and sample size calculations with
ACTUAL parameter values from the UI.
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


def generate_cli_code_cluster_continuous(params):
    """
    Generate reproducible Python code for cluster RCT continuous outcome calculations.
    
    Uses ACTUAL parameter values from the UI to create a runnable script.
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
    
    # Build import statement and function name
    if method == "analytical":
        import_line = "from core.designs.cluster_rct.analytical_continuous import"
    elif method == "permutation":
        import_line = "from core.designs.cluster_rct.analytical_continuous import"
    else:
        import_line = "from core.designs.cluster_rct.simulation_continuous import"
    
    # Determine function name and parameters based on calculation type
    if calc_type == "Power":
        if method == "permutation":
            function_name = "power_continuous_permutation"
        else:
            function_name = f"power_continuous{'_sim' if method == 'simulation' else ''}"
        
        # Build actual parameter string with real values
        if method == "simulation":
            param_lines = [
                f"    n_clusters={n_clusters},",
                f"    cluster_size={cluster_size},", 
                f"    icc={icc},",
                f"    mean1={mean1},",
                f"    mean2={mean2},",
                f"    std_dev={std_dev},",
                f"    alpha={alpha},",
                f"    nsim={nsim},"
            ]
            if seed is not None:
                param_lines.append(f"    seed={seed},")
            param_lines.append(f"    analysis_model='{analysis_model}'")
            
            # Add model-specific parameters if needed
            if analysis_model == 'bayes':
                bayes_backend = params.get('bayes_backend', 'stan')
                bayes_draws = params.get('bayes_draws', 500)
                bayes_warmup = params.get('bayes_warmup', 500)
                bayes_inference_method = params.get('bayes_inference_method', 'credible_interval')
                param_lines.extend([
                    f"    bayes_backend='{bayes_backend}',",
                    f"    bayes_draws={bayes_draws},",
                    f"    bayes_warmup={bayes_warmup},",
                    f"    bayes_inference_method='{bayes_inference_method}'"
                ])
            elif analysis_model == 'mixedlm':
                use_satterthwaite = params.get('use_satterthwaite', False)
                lmm_method = params.get('lmm_method', 'auto')
                lmm_reml = params.get('lmm_reml', True)
                param_lines.extend([
                    f"    use_satterthwaite={use_satterthwaite},",
                    f"    lmm_method='{lmm_method}',",
                    f"    lmm_reml={lmm_reml}"
                ])
                lmm_cov_penalty_weight = params.get('lmm_cov_penalty_weight', 0.0)
                if lmm_cov_penalty_weight > 0:
                    param_lines.append(f"    lmm_cov_penalty_weight={lmm_cov_penalty_weight}")
            elif analysis_model == 'gee':
                use_bias_correction = params.get('use_bias_correction', False)
                param_lines.append(f"    use_bias_correction={use_bias_correction}")
        else:
            # Analytical method
            param_lines = [
                f"    n_clusters={n_clusters},",
                f"    cluster_size={cluster_size},",
                f"    icc={icc},",
                f"    mean1={mean1},",
                f"    mean2={mean2},",
                f"    std_dev={std_dev},",
                f"    alpha={alpha}"
            ]
        
        result_key = "power"
        
    elif calc_type == "Sample Size":
        if method == "permutation":
            function_name = "sample_size_continuous_permutation"
        else:
            function_name = f"sample_size_continuous{'_sim' if method == 'simulation' else ''}"
        
        if method == "simulation":
            param_lines = [
                f"    mean1={mean1},",
                f"    mean2={mean2},",
                f"    std_dev={std_dev},",
                f"    icc={icc},",
                f"    cluster_size={cluster_size},",
                f"    power={power},",
                f"    alpha={alpha},",
                f"    nsim={nsim},"
            ]
            if seed is not None:
                param_lines.append(f"    seed={seed},")
            param_lines.append(f"    analysis_model='{analysis_model}'")
        else:
            param_lines = [
                f"    mean1={mean1},",
                f"    mean2={mean2},",
                f"    std_dev={std_dev},",
                f"    icc={icc},",
                f"    cluster_size={cluster_size},",
                f"    power={power},",
                f"    alpha={alpha}"
            ]
        
        result_key = "n_clusters"
        
    elif calc_type == "Minimum Detectable Effect":
        if method == "permutation":
            function_name = "min_detectable_effect_continuous_permutation"
        else:
            function_name = f"min_detectable_effect_continuous{'_sim' if method == 'simulation' else ''}"
        
        if method == "simulation":
            param_lines = [
                f"    n_clusters={n_clusters},",
                f"    cluster_size={cluster_size},",
                f"    icc={icc},",
                f"    std_dev={std_dev},",
                f"    power={power},",
                f"    alpha={alpha},",
                f"    nsim={nsim},"
            ]
            if seed is not None:
                param_lines.append(f"    seed={seed},")
            param_lines.append(f"    analysis_model='{analysis_model}'")
        else:
            param_lines = [
                f"    n_clusters={n_clusters},",
                f"    cluster_size={cluster_size},",
                f"    icc={icc},",
                f"    std_dev={std_dev},",
                f"    power={power},",
                f"    alpha={alpha}"
            ]
        
        result_key = "mde"
    
    # Join parameters properly
    all_params = "\n".join(param_lines)
    
    # Generate the actual reproducible code
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

{import_line} {function_name}

# Calculate {calc_type.lower()} with these specific parameters:
result = {function_name}(
{all_params}
)

# Display main result
print(f"{calc_type}: {{result['{result_key}']:.3f}}")
if 'design_effect' in result:
    print(f"Design effect: {{result['design_effect']:.2f}}")

# Display full results
import json
print("\\nFull results:")
print(json.dumps(result, indent=2))"""

    return code


def generate_cli_code_cluster_binary(params):
    """
    Generate reproducible Python code for cluster RCT binary outcome calculations.
    
    Uses ACTUAL parameter values from the UI to create a runnable script.
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
    
    # Build import statement and function name
    if method == "analytical":
        import_line = "from core.designs.cluster_rct.analytical_binary import"
    elif method == "permutation":
        import_line = "from core.designs.cluster_rct.analytical_binary import"
    else:
        import_line = "from core.designs.cluster_rct.simulation_binary import"
    
    # Determine function name and parameters based on calculation type
    if calc_type == "Power":
        if method == "permutation":
            function_name = "power_binary_permutation"
        else:
            function_name = f"power_binary{'_sim' if method == 'simulation' else ''}"
        
        # Build actual parameter string with real values
        param_lines = [
            f"    n_clusters={n_clusters},",
            f"    cluster_size={cluster_size},",
            f"    icc={icc},",
            f"    p1={p1},",
            f"    p2={p2},",
            f"    alpha={alpha}"
        ]
        
        # Add optional parameters if non-default
        if cv_cluster_size > 0:
            param_lines.append(f"    cv_cluster_size={cv_cluster_size}")
        
        if method == "simulation":
            param_lines.extend([
                f"    nsim={nsim},"
            ])
            if seed is not None:
                param_lines.append(f"    seed={seed},")
            param_lines.append(f"    analysis_method='{analysis_method}'")
            
            # Add model-specific parameters if needed
            if analysis_method == 'bayes':
                bayes_backend = params.get('bayes_backend', 'stan')
                bayes_draws = params.get('bayes_draws', 500)
                bayes_warmup = params.get('bayes_warmup', 500)
                bayes_inference_method = params.get('bayes_inference_method', 'credible_interval')
                param_lines.extend([
                    f"    bayes_backend='{bayes_backend}',",
                    f"    bayes_draws={bayes_draws},",
                    f"    bayes_warmup={bayes_warmup},",
                    f"    bayes_inference_method='{bayes_inference_method}'"
                ])
        
        result_key = "power"
        
    elif calc_type == "Sample Size":
        if method == "permutation":
            function_name = "sample_size_binary_permutation"
        else:
            function_name = f"sample_size_binary{'_sim' if method == 'simulation' else ''}"
        
        param_lines = [
            f"    p1={p1},",
            f"    p2={p2},",
            f"    icc={icc},",
            f"    cluster_size={cluster_size},",
            f"    power={power},",
            f"    alpha={alpha}"
        ]
        
        # Add optional parameters if non-default
        if cv_cluster_size > 0:
            param_lines.append(f"    cv_cluster_size={cv_cluster_size}")
        
        if method == "simulation":
            param_lines.extend([
                f"    nsim={nsim},"
            ])
            if seed is not None:
                param_lines.append(f"    seed={seed},")
            param_lines.append(f"    analysis_method='{analysis_method}'")
        
        result_key = "n_clusters"
        
    elif calc_type == "Minimum Detectable Effect":
        if method == "permutation":
            function_name = "min_detectable_effect_binary_permutation"
        else:
            function_name = f"min_detectable_effect_binary{'_sim' if method == 'simulation' else ''}"
        
        param_lines = [
            f"    n_clusters={n_clusters},",
            f"    cluster_size={cluster_size},",
            f"    icc={icc},",
            f"    p1={p1},",
            f"    power={power},",
            f"    alpha={alpha}"
        ]
        
        # Add optional parameters if non-default
        if cv_cluster_size > 0:
            param_lines.append(f"    cv_cluster_size={cv_cluster_size}")
        if effect_measure != 'risk_difference':
            param_lines.append(f"    effect_measure='{effect_measure}'")
        
        if method == "simulation":
            param_lines.extend([
                f"    nsim={nsim},"
            ])
            if seed is not None:
                param_lines.append(f"    seed={seed},")
            param_lines.append(f"    analysis_method='{analysis_method}'")
        
        result_key = "mde"
    
    # Join parameters properly
    all_params = "\n".join(param_lines)
    
    # Generate the actual reproducible code
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

{import_line} {function_name}

# Calculate {calc_type.lower()} with these specific parameters:
result = {function_name}(
{all_params}
)

# Display main result
print(f"{calc_type}: {{result['{result_key}']:.3f}}")
if 'design_effect' in result:
    print(f"Design effect: {{result['design_effect']:.2f}}")

# Display full results
import json
print("\\nFull results:")
print(json.dumps(result, indent=2))"""

    return code