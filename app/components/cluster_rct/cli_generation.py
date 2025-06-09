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
    
    # Calculate effect size and derived values for display
    effect_size = abs(mean2 - mean1)
    standardized_effect = effect_size / std_dev if std_dev > 0 else 0
    design_effect = 1 + (cluster_size - 1) * icc
    
    study_summary = f"""# STUDY DESIGN SUMMARY:
# Design: Cluster Randomized Controlled Trial
# Outcome Type: Continuous (mean difference analysis)
# Statistical Method: {method.title()}
#
# STUDY PARAMETERS:
# - Control cluster mean: {mean1}
# - Treatment cluster mean: {mean2} (effect size: {effect_size})
# - Standard deviation: {std_dev}
# - Standardized effect size (Cohen's d): {standardized_effect:.3f}
# - Average cluster size: {cluster_size} individuals
# - Intracluster correlation (ICC): {icc:.3f}
# - Design effect: {design_effect:.2f}
# - Desired statistical power: {power*100:.0f}%
# - Significance level (alpha): {alpha*100:.0f}%
#
# CLUSTERING IMPACT:
# The ICC of {icc:.3f} means that {icc*100:.1f}% of the total variance is between clusters.
# The design effect of {design_effect:.2f} means we need {design_effect:.1f}x more participants
# than an equivalent parallel RCT due to clustering.
#
# EFFECT SIZE INTERPRETATION:
# Cohen's d of {standardized_effect:.3f} is considered {"small" if standardized_effect < 0.5 else "medium" if standardized_effect < 0.8 else "large"} effect size.
#
# RESEARCH QUESTION:"""
    
    if calc_type == "Sample Size":
        study_summary += f"""
# How many clusters are needed to detect a mean difference of {effect_size}
# with {power*100:.0f}% power at {alpha*100:.0f}% significance level?"""
    elif calc_type == "Power":
        study_summary += f"""
# With {n_clusters} clusters per arm ({cluster_size} individuals each), what is the
# power to detect a mean difference of {effect_size} at {alpha*100:.0f}% significance?"""
    else:  # MDE
        study_summary += f"""
# With {n_clusters} clusters per arm ({cluster_size} individuals each), what is the
# smallest mean difference detectable with {power*100:.0f}% power at {alpha*100:.0f}% significance?"""

    # Generate the actual reproducible code with detailed study context
    code = f"""{study_summary}

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

# Calculate {calc_type.lower()} with the specified study parameters
result = {function_name}(
{all_params}
)

# RESULTS INTERPRETATION:
print("=" * 60)
print(f"CLUSTER RCT CONTINUOUS OUTCOME - {calc_type.upper()} ANALYSIS")
print("=" * 60)

main_result = result['{result_key}']
print(f"\\n{calc_type}: {{main_result:.3f}}")

if "{calc_type}" == "Sample Size":
    clusters_per_arm = main_result
    total_clusters = clusters_per_arm * 2
    total_individuals = total_clusters * {cluster_size}
    print(f"  - Clusters per arm: {{clusters_per_arm}} clusters")
    print(f"  - Total clusters: {{total_clusters}} clusters")
    print(f"  - Total individuals: {{total_individuals}} participants")
    print(f"  - Design effect: {design_effect:.2f}")
elif "{calc_type}" == "Power":
    print(f"  - Statistical power: {{main_result*100:.1f}}%")
    print(f"  - This means {{main_result*100:.1f}}% chance of detecting the specified difference if it truly exists")
    if 'design_effect' in result:
        print(f"  - Design effect: {{result['design_effect']:.2f}}")
elif "{calc_type}" == "Minimum Detectable Effect":
    mde_value = main_result
    standardized_mde = mde_value / {std_dev} if {std_dev} > 0 else 0
    print(f"  - Minimum detectable effect: {{mde_value:.3f}} units")
    print(f"  - Standardized effect size: {{standardized_mde:.3f}} (Cohen's d)")
    print(f"  - This represents a {{"small" if standardized_mde < 0.5 else "medium" if standardized_mde < 0.8 else "large"}} effect size")
    if 'design_effect' in result:
        print(f"  - Design effect: {{result['design_effect']:.2f}}")

print("\\n" + "=" * 60)
print("DETAILED RESULTS (JSON format):")
print("=" * 60)

# Full results in JSON format for further analysis
import json
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
    
    # Calculate effect size and derived values for display
    effect_size = abs(p2 - p1)
    relative_risk = p2 / p1 if p1 > 0 else float('inf')
    odds_ratio = (p2 / (1 - p2)) / (p1 / (1 - p1)) if p1 < 1 and p2 < 1 else float('inf')
    design_effect = 1 + (cluster_size - 1) * icc
    total_individuals = n_clusters * cluster_size * 2 if calc_type == "Sample Size" else n_clusters * cluster_size * 2
    
    study_summary = f"""# STUDY DESIGN SUMMARY:
# Design: Cluster Randomized Controlled Trial
# Outcome Type: Binary (proportion analysis)
# Statistical Method: {method.title()}
#
# STUDY PARAMETERS:
# - Control cluster proportion: {p1} ({p1*100:.1f}%)
# - Treatment cluster proportion: {p2} ({p2*100:.1f}%)
# - Absolute difference: {effect_size:.3f} ({effect_size*100:.1f} percentage points)
# - Relative risk: {relative_risk:.2f}
# - Odds ratio: {odds_ratio:.2f}
# - Average cluster size: {cluster_size} individuals
# - Intracluster correlation (ICC): {icc:.3f}
# - Design effect: {design_effect:.2f}
# - Desired statistical power: {power*100:.0f}%
# - Significance level (alpha): {alpha*100:.0f}%
#
# CLUSTERING IMPACT:
# The ICC of {icc:.3f} means that {icc*100:.1f}% of the total variance is between clusters.
# The design effect of {design_effect:.2f} means we need {design_effect:.1f}x more participants
# than an equivalent parallel RCT due to clustering.
#
# RESEARCH QUESTION:"""
    
    if calc_type == "Sample Size":
        study_summary += f"""
# How many clusters are needed to detect a difference in proportions from
# {p1:.3f} to {p2:.3f} with {power*100:.0f}% power at {alpha*100:.0f}% significance level?"""
    elif calc_type == "Power":
        study_summary += f"""
# With {n_clusters} clusters per arm ({cluster_size} individuals each), what is the
# power to detect a difference from {p1:.3f} to {p2:.3f} at {alpha*100:.0f}% significance?"""
    else:  # MDE
        study_summary += f"""
# With {n_clusters} clusters per arm ({cluster_size} individuals each), what is the
# smallest difference in proportions detectable with {power*100:.0f}% power at {alpha*100:.0f}% significance?"""

    # Generate the actual reproducible code with detailed study context
    code = f"""{study_summary}

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

# Calculate {calc_type.lower()} with the specified study parameters
result = {function_name}(
{all_params}
)

# RESULTS INTERPRETATION:
print("=" * 60)
print(f"CLUSTER RCT BINARY OUTCOME - {calc_type.upper()} ANALYSIS")
print("=" * 60)

main_result = result['{result_key}']
print(f"\\n{calc_type}: {{main_result:.3f}}")

if "{calc_type}" == "Sample Size":
    clusters_per_arm = main_result
    total_clusters = clusters_per_arm * 2
    total_individuals = total_clusters * {cluster_size}
    print(f"  - Clusters per arm: {{clusters_per_arm}} clusters")
    print(f"  - Total clusters: {{total_clusters}} clusters")
    print(f"  - Total individuals: {{total_individuals}} participants")
    print(f"  - Design effect: {design_effect:.2f}")
elif "{calc_type}" == "Power":
    print(f"  - Statistical power: {{main_result*100:.1f}}%")
    print(f"  - This means {{main_result*100:.1f}}% chance of detecting the specified difference if it truly exists")
    if 'design_effect' in result:
        print(f"  - Design effect: {{result['design_effect']:.2f}}")
elif "{calc_type}" == "Minimum Detectable Effect":
    baseline_prop = {p1}
    detectable_prop = main_result
    abs_diff = abs(detectable_prop - baseline_prop)
    print(f"  - Minimum detectable proportion: {{detectable_prop:.3f}} ({{detectable_prop*100:.1f}}%)")
    print(f"  - Absolute difference: {{abs_diff:.3f}} ({{abs_diff*100:.1f}} percentage points)")
    print(f"  - From baseline {{baseline_prop:.3f}} to {{detectable_prop:.3f}}")
    if 'design_effect' in result:
        print(f"  - Design effect: {{result['design_effect']:.2f}}")

print("\\n" + "=" * 60)
print("DETAILED RESULTS (JSON format):")
print("=" * 60)

# Full results in JSON format for further analysis
import json
print(json.dumps(result, indent=2))"""

    return code