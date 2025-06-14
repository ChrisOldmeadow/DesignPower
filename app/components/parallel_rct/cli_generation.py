"""CLI code generation utilities for parallel RCT components.

This module provides functions to generate reproducible Python code
for parallel RCT power analysis and sample size calculations with
ACTUAL parameter values from the UI.
"""

import textwrap
import json


def generate_cli_code_parallel_binary(params):
    """
    Generate enhanced reproducible code for parallel RCT binary outcome calculations with algorithm transparency.
    
    This includes the complete source code of statistical algorithms for full transparency.
    """
    # Import the source extraction utility for showing algorithm details
    from core.utils.source_extraction import get_function_source
    from core.utils.enhanced_script_generation import get_function_from_name
    
    # Extract key parameters from UI
    calc_type = params.get('calculation_type', 'Power')
    hypothesis_type = params.get('hypothesis_type', 'Superiority')
    method = params.get('method', 'analytical')
    
    # Core parameters 
    n1 = params.get('n1', 100)
    n2 = params.get('n2', 100)
    p1 = params.get('p1', 0.3)
    p2 = params.get('p2', 0.5)
    alpha = params.get('alpha', 0.05)
    power = params.get('power', 0.8)  # May be None for power calculations
    allocation_ratio = params.get('allocation_ratio', 1.0)
    
    # Simulation-specific parameters
    nsim = params.get('nsim', 1000)
    seed = params.get('seed')
    test_type = params.get('test_type', 'normal_approximation')
    
    # Non-inferiority specific
    non_inferiority_margin = params.get('non_inferiority_margin', 0.05)
    assumed_difference_ni = params.get('assumed_difference_ni', 0.0)
    non_inferiority_direction = params.get('non_inferiority_direction', 'higher')
    
    # Build import statement
    import_line = "from core.designs.parallel import"
    
    # Build function call based on calculation type and hypothesis
    if calc_type == "Power":
        if hypothesis_type == "Superiority":
            function_name = f"power_binary{'_sim' if method == 'simulation' else ''}"
            core_params = f"""n1={n1},
    n2={n2},
    p1={p1},
    p2={p2},
    alpha={alpha}"""
        else:  # Non-Inferiority
            function_name = f"power_binary_non_inferiority{'_sim' if method == 'simulation' else ''}"
            core_params = f"""n1={n1},
    n2={n2},
    p1={p1},
    non_inferiority_margin={non_inferiority_margin},
    alpha={alpha},
    assumed_difference={assumed_difference_ni},
    direction="{non_inferiority_direction}",
    allocation_ratio={allocation_ratio}"""
        
        if method == "simulation":
            sim_params = f"""nsim={nsim},"""
            if seed is not None:
                sim_params += f"""
    seed={seed},"""
            sim_params += f"""
    test_type="{test_type}","""
            
            all_params = core_params + ",\n    " + sim_params.strip()
        else:
            all_params = core_params
        
        result_key = "power"
        
    elif calc_type == "Sample Size":
        if hypothesis_type == "Superiority":
            function_name = f"sample_size_binary{'_sim' if method == 'simulation' else ''}"
            core_params = f"""p1={p1},
    p2={p2},
    power={power},
    alpha={alpha},
    allocation_ratio={allocation_ratio}"""
        else:  # Non-Inferiority
            function_name = f"sample_size_binary_non_inferiority{'_sim' if method == 'simulation' else ''}"
            core_params = f"""p1={p1},
    non_inferiority_margin={non_inferiority_margin},
    power={power},
    alpha={alpha},
    allocation_ratio={allocation_ratio},
    assumed_difference={assumed_difference_ni},
    direction="{non_inferiority_direction}" """
        
        if method == "simulation":
            sim_params = f"""nsim={nsim},"""
            if seed is not None:
                sim_params += f"""
    seed={seed},"""
            sim_params += f"""
    test_type="{test_type}","""
            
            all_params = core_params + ",\n    " + sim_params.strip()
        else:
            all_params = core_params
            
        result_key = "total_sample_size"
        
    elif calc_type == "Minimum Detectable Effect":
        if hypothesis_type == "Superiority":
            function_name = f"min_detectable_effect_binary{'_sim' if method == 'simulation' else ''}"
            core_params = f"""n1={n1},
    n2={n2},
    p1={p1},
    power={power},
    alpha={alpha}"""
        else:  # Non-Inferiority
            function_name = f"min_detectable_binary_non_inferiority_margin{'_sim' if method == 'simulation' else ''}"
            core_params = f"""n1={n1},
    n2={n2},
    p1={p1},
    power={power},
    alpha={alpha},
    assumed_difference={assumed_difference_ni},
    direction="{non_inferiority_direction}" """
        
        if method == "simulation":
            sim_params = f"""nsim={nsim},"""
            if seed is not None:
                sim_params += f"""
    seed={seed},"""
            sim_params += f"""
    test_type="{test_type}","""
            
            all_params = core_params + ",\n    " + sim_params.strip()
        else:
            all_params = core_params
            
        if hypothesis_type == "Superiority":
            result_key = "minimum_detectable_p2"
        else:
            result_key = "minimum_detectable_margin"
    
    # Get algorithm source code based on the function being used
    module_path_map = {
        "power_binary": "core.designs.parallel.analytical_binary",
        "power_binary_sim": "core.designs.parallel.simulation_binary", 
        "sample_size_binary": "core.designs.parallel.analytical_binary",
        "sample_size_binary_sim": "core.designs.parallel.simulation_binary",
        "power_binary_non_inferiority": "core.designs.parallel.analytical_binary",
        "sample_size_binary_non_inferiority": "core.designs.parallel.analytical_binary",
        "min_detectable_binary_non_inferiority_margin": "core.designs.parallel.analytical_binary"
    }
    
    # Get the actual function and its source code
    module_path = module_path_map.get(function_name, "core.designs.parallel.analytical_binary")
    actual_function = get_function_from_name(module_path, function_name)
    
    # For simulation functions, we need to import additional dependencies
    additional_imports = ""
    if method == "simulation":
        additional_imports = """from core.designs.parallel.simulation_binary import simulate_binary_trial
from core.designs.parallel.analytical_binary import perform_binary_test"""
    
    if actual_function:
        algorithm_source = get_function_source(actual_function)
        algorithm_section = f"""
# ============================================================================
# STATISTICAL ALGORITHM SOURCE CODE
# 
# The following shows the exact implementation of the {function_name} function
# used in this analysis. This provides full transparency of the statistical
# methodology and allows verification of the approach.
# ============================================================================

{algorithm_source}

# ============================================================================
# MAIN ANALYSIS USING THE ALGORITHM SHOWN ABOVE
# ============================================================================
"""
    else:
        algorithm_section = f"""
# ============================================================================
# STATISTICAL ALGORITHM: {function_name}
# Module: {module_path}
# 
# Note: Algorithm source code could not be extracted automatically.
# The function {function_name} from {module_path} is being used.
# ============================================================================
"""
    
    # Calculate effect size and other derived values for display
    if hypothesis_type == "Superiority":
        effect_size = abs(p2 - p1)
        baseline_rate = min(p1, p2)
        treatment_rate = max(p1, p2)
        relative_risk = p2 / p1 if p1 > 0 else float('inf')
        odds_ratio = (p2 / (1 - p2)) / (p1 / (1 - p1)) if p1 < 1 and p2 < 1 else float('inf')
        
        # Safe power formatting
        power_text = f"{power*100:.0f}%" if power is not None else "TBD (being calculated)"
        
        study_summary = f"""# STUDY DESIGN SUMMARY:
# Design: Parallel Group Randomized Controlled Trial
# Outcome Type: Binary (proportion analysis)
# Hypothesis: {hypothesis_type}
# Statistical Method: {method.title()}
#
# STUDY PARAMETERS:
# - Control group proportion: {p1} ({p1*100:.1f}%)
# - Treatment group proportion: {p2} ({p2*100:.1f}%)
# - Absolute difference: {effect_size:.3f} ({effect_size*100:.1f} percentage points)
# - Relative risk: {relative_risk:.2f}
# - Odds ratio: {odds_ratio:.2f}
# - Desired statistical power: {power_text}
# - Significance level (alpha): {alpha*100:.0f}%
# - Allocation ratio (treatment:control): {allocation_ratio}:1
# - Statistical test: {test_type.replace('_', ' ').title()}
#
# RESEARCH QUESTION:"""
        
        if calc_type == "Sample Size":
            study_summary += f"""
# How many participants are needed to detect a difference in proportions from
# {p1:.3f} to {p2:.3f} with {power_text} power at {alpha*100:.0f}% significance level?"""
        elif calc_type == "Power":
            study_summary += f"""
# With {n1} participants in control and {n2} in treatment group, what is the
# power to detect a difference from {p1:.3f} to {p2:.3f} at {alpha*100:.0f}% significance?"""
        else:  # MDE
            study_summary += f"""
# With {n1} participants in control and {n2} in treatment group, what is the
# smallest difference in proportions detectable with {power_text} power at {alpha*100:.0f}% significance?"""
            
    else:  # Non-Inferiority
        # Safe power formatting for non-inferiority
        power_text = f"{power*100:.0f}%" if power is not None else "TBD (being calculated)"
        
        study_summary = f"""# STUDY DESIGN SUMMARY:
# Design: Parallel Group Randomized Controlled Trial
# Outcome Type: Binary (proportion analysis)
# Hypothesis: {hypothesis_type}
# Statistical Method: {method.title()}
#
# STUDY PARAMETERS:
# - Reference treatment proportion: {p1} ({p1*100:.1f}%)
# - Non-inferiority margin: {non_inferiority_margin:.3f} ({non_inferiority_margin*100:.1f} percentage points)
# - Assumed true difference: {assumed_difference_ni:.3f} ({assumed_difference_ni*100:.1f} percentage points)
# - Direction: Treatment should be no more than {non_inferiority_margin*100:.1f}pp {non_inferiority_direction} than reference
# - Desired statistical power: {power_text}
# - Significance level (alpha): {alpha*100:.0f}% (one-sided for non-inferiority)
# - Allocation ratio (test:reference): {allocation_ratio}:1
# - Statistical test: {test_type.replace('_', ' ').title()}
#
# RESEARCH QUESTION:"""
        
        if calc_type == "Sample Size":
            study_summary += f"""
# How many participants are needed to demonstrate non-inferiority with a margin
# of {non_inferiority_margin:.3f} with {power_text}?"""
        elif calc_type == "Power":
            study_summary += f"""
# With {n1} participants in test and {n2} in reference group, what is the
# power to demonstrate non-inferiority with margin {non_inferiority_margin:.3f}?"""
        else:  # MDE
            study_summary += f"""
# With {n1} participants in test and {n2} in reference group, what is the
# smallest non-inferiority margin demonstrable with {power_text}?"""

    # Generate enhanced code with algorithm transparency
    code = f"""{study_summary}

# ====================================================================
# ENHANCED REPRODUCIBLE ANALYSIS SCRIPT WITH ALGORITHM TRANSPARENCY
# Generated by DesignPower - https://github.com/your-repo/DesignPower
# ====================================================================
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
{additional_imports}

{algorithm_section}

# Calculate {calc_type.lower()} using the algorithm shown above
result = {function_name}(
    {all_params}
)

# RESULTS INTERPRETATION:
print("=" * 60)
print(f"PARALLEL RCT BINARY OUTCOME - {calc_type.upper()} ANALYSIS")
print("=" * 60)

main_result = result['{result_key}']
print(f"\\\\n{calc_type}: {{main_result:.3f}}")

if "{calc_type}" == "Sample Size":
    n1_result = result.get('n1', main_result // 2)
    n2_result = result.get('n2', main_result // 2)
    print(f"  - Group 1 (control): {{n1_result}} participants") 
    print(f"  - Group 2 (treatment): {{n2_result}} participants")
    print(f"  - Total sample size: {{main_result}} participants")
elif "{calc_type}" == "Power":
    print(f"  - Statistical power: {{main_result*100:.1f}}%")
    print(f"  - This means {{main_result*100:.1f}}% chance of detecting the specified difference if it truly exists")
elif "{calc_type}" == "Minimum Detectable Effect":
    if "{hypothesis_type}" == "Superiority":
        baseline_prop = {p1}
        detectable_prop = main_result
        abs_diff = abs(detectable_prop - baseline_prop)
        print(f"  - Minimum detectable proportion: {{detectable_prop:.3f}} ({{detectable_prop*100:.1f}}%)")
        print(f"  - Absolute difference: {{abs_diff:.3f}} ({{abs_diff*100:.1f}} percentage points)")
        print(f"  - From baseline {{baseline_prop:.3f}} to {{detectable_prop:.3f}}")
    else:
        print(f"  - Minimum detectable non-inferiority margin: {{main_result:.3f}} ({{main_result*100:.1f}} percentage points)")

print("\\\\n" + "=" * 60)
print("ALGORITHM INFORMATION")
print("=" * 60)
print(f"Statistical function: {function_name}")
print(f"Source module: {module_path}")
print("Complete algorithm source code: See implementation above")

print("\\\\n" + "=" * 60)
print("DETAILED RESULTS (JSON format):")
print("=" * 60)

# Full results in JSON format for further analysis
import json
print(json.dumps(result, indent=2))"""

    return code


def generate_cli_code_parallel_survival(params):
    """
    Generate enhanced reproducible code for parallel RCT survival outcome calculations with algorithm transparency.
    
    This includes the complete source code of statistical algorithms for full transparency.
    """
    # Extract key parameters from UI
    calc_type = params.get('calculation_type', 'Power')
    hypothesis_type = params.get('hypothesis_type', 'Superiority')
    method = params.get('method', 'analytical')

    # Core parameters 
    n1 = params.get('n1', 150)
    n2 = params.get('n2', 150)
    median1 = params.get('median1', 12.0)
    median2 = params.get('median2', 18.0)
    enrollment_period = params.get('enrollment_period', 12.0)
    follow_up_period = params.get('follow_up_period', 12.0)
    dropout_rate = params.get('dropout_rate', 0.1)
    alpha = params.get('alpha', 0.05)
    power = params.get('power', 0.8)
    allocation_ratio = params.get('allocation_ratio', 1.0)
    
    # Simulation-specific parameters
    nsim = params.get('nsim', 500)
    seed = params.get('seed')
    
    # Build import statement
    import_line = "from core.designs.parallel import"
    
    # Build function call based on calculation type
    if calc_type == "Power":
        function_name = f"power_survival{'_sim' if method == 'simulation' else ''}"
        core_params = f"""n1={n1},
    n2={n2},
    median1={median1},
    median2={median2},
    enrollment_period={enrollment_period},
    follow_up_period={follow_up_period},
    dropout_rate={dropout_rate},
    alpha={alpha}"""
        
        if method == "simulation":
            sim_params = f"""nsim={nsim},"""
            if seed is not None:
                sim_params += f"""
    seed={seed},"""
            
            all_params = core_params + ",\n    " + sim_params.strip()
        else:
            all_params = core_params
        
        result_key = "power"
        
    elif calc_type == "Sample Size":
        function_name = f"sample_size_survival{'_sim' if method == 'simulation' else ''}"
        core_params = f"""median1={median1},
    median2={median2},
    enrollment_period={enrollment_period},
    follow_up_period={follow_up_period},
    dropout_rate={dropout_rate},
    power={power},
    alpha={alpha},
    allocation_ratio={allocation_ratio}"""
        
        if method == "simulation":
            sim_params = f"""nsim={nsim},"""
            if seed is not None:
                sim_params += f"""
    seed={seed},"""
            
            all_params = core_params + ",\n    " + sim_params.strip()
        else:
            all_params = core_params
            
        result_key = "total_sample_size"
        
    elif calc_type == "Minimum Detectable Effect":
        function_name = f"min_detectable_effect_survival{'_sim' if method == 'simulation' else ''}"
        core_params = f"""n1={n1},
    n2={n2},
    median1={median1},
    enrollment_period={enrollment_period},
    follow_up_period={follow_up_period},
    dropout_rate={dropout_rate},
    power={power},
    alpha={alpha}"""
        
        if method == "simulation":
            sim_params = f"""nsim={nsim},"""
            if seed is not None:
                sim_params += f"""
    seed={seed},"""
            
            all_params = core_params + ",\n    " + sim_params.strip()
        else:
            all_params = core_params
            
        result_key = "minimum_detectable_median"
    
    # Get algorithm source code for survival functions
    from core.utils.source_extraction import get_function_source
    from core.utils.enhanced_script_generation import get_function_from_name
    
    survival_module_map = {
        "power_survival": "core.designs.parallel.analytical_survival",
        "power_survival_sim": "core.designs.parallel.simulation_survival", 
        "sample_size_survival": "core.designs.parallel.analytical_survival",
        "sample_size_survival_sim": "core.designs.parallel.simulation_survival"
    }
    
    # Get the actual function and its source code
    survival_module_path = survival_module_map.get(function_name, "core.designs.parallel.analytical_survival")
    survival_function = get_function_from_name(survival_module_path, function_name)
    
    if survival_function:
        survival_algorithm_source = get_function_source(survival_function)
        survival_algorithm_section = f"""
# ============================================================================
# STATISTICAL ALGORITHM SOURCE CODE
# 
# The following shows the exact implementation of the {function_name} function
# used in this analysis. This provides full transparency of the statistical
# methodology and allows verification of the approach.
# ============================================================================

{survival_algorithm_source}

# ============================================================================
# MAIN ANALYSIS USING THE ALGORITHM SHOWN ABOVE
# ============================================================================
"""
    else:
        survival_algorithm_section = f"""
# ============================================================================
# STATISTICAL ALGORITHM: {function_name}
# Module: {survival_module_path}
# 
# Note: Algorithm source code could not be extracted automatically.
# The function {function_name} from {survival_module_path} is being used.
# ============================================================================
"""
    
    # Generate enhanced code with algorithm transparency
    code = f"""# ====================================================================
# ENHANCED PARALLEL RCT SURVIVAL OUTCOME - {calc_type} Analysis ({hypothesis_type})
# Generated by DesignPower with Algorithm Transparency
# ====================================================================
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

{survival_algorithm_section}

# Calculate {calc_type.lower()} using the algorithm shown above
result = {function_name}(
    {all_params}
)

print(f"{calc_type}: {{result['{result_key}']:.3f}}")
# Safe formatting for hazard ratio
hazard_ratio = result.get('hazard_ratio')
if hazard_ratio is not None:
    print(f"Hazard ratio: {{hazard_ratio:.3f}}")
else:
    print("Hazard ratio: N/A")

# Safe formatting for expected events  
total_events = result.get('total_events')
if total_events is not None:
    print(f"Expected events: {{total_events:.0f}}")
else:
    print("Expected events: N/A")

# Algorithm information
print("\\\\n" + "=" * 60)
print("ALGORITHM INFORMATION")
print("=" * 60)
print(f"Statistical function: {function_name}")
print(f"Source module: {survival_module_path}")
print("Complete algorithm source code: See implementation above")

# Full results
print("\\\\n" + "=" * 60)
print("DETAILED RESULTS (JSON format):")
print("=" * 60)
import json
print(json.dumps(result, indent=2))"""

    return code


def generate_cli_code_parallel_continuous(params):
    """
    Generate clean, simple reproducible code for parallel RCT continuous outcome calculations.
    
    This matches the style in EXAMPLES.md for consistency and simplicity.
    """
    # Extract key parameters from UI
    calc_type = params.get('calculation_type', 'Power')
    hypothesis_type = params.get('hypothesis_type', 'Superiority')
    method = params.get('method', 'analytical')
    
    # Core parameters 
    n1 = params.get('n1', 100)
    n2 = params.get('n2', 100)
    mean1 = params.get('mean1', 10.0)
    mean2 = params.get('mean2', 12.0)
    std_dev = params.get('std_dev', 5.0)
    alpha = params.get('alpha', 0.05)
    power = params.get('power', 0.8)  # May be None for power calculations
    allocation_ratio = params.get('allocation_ratio', 1.0)
    
    # Simulation-specific parameters
    nsim = params.get('nsim', 1000)
    seed = params.get('seed')
    
    # Non-inferiority specific
    non_inferiority_margin = params.get('non_inferiority_margin', 1.0)
    assumed_difference_ni = params.get('assumed_difference_ni', 0.0)
    non_inferiority_direction = params.get('non_inferiority_direction', 'higher')
    
    # Repeated measures specific
    repeated_measures = params.get('repeated_measures', False)
    correlation = params.get('correlation', 0.5)
    analysis_method = params.get('analysis_method', 'ANCOVA')
    
    # Advanced options
    unequal_var = params.get('unequal_var', False)
    std_dev2 = params.get('std_dev2', std_dev)
    
    # Build import statement
    import_line = "from core.designs.parallel import"
    
    # Import the source extraction utility for showing algorithm details
    from core.utils.source_extraction import get_function_source
    from core.utils.enhanced_script_generation import get_function_from_name
    
    # Build function call based on calculation type and hypothesis
    if calc_type == "Power":
        if hypothesis_type == "Superiority":
            if repeated_measures:
                function_name = f"power_repeated_measures{'_sim' if method == 'simulation' else ''}"
                # Map UI method names to function expected names
                method_mapping = {"ANCOVA": "ancova", "Change Score": "change_score"}
                mapped_method = method_mapping.get(analysis_method, "ancova")
                
                core_params = f"""n1={n1},
    n2={n2},
    mean1={mean1},
    mean2={mean2},
    sd1={std_dev},
    correlation={correlation},
    alpha={alpha},
    method="{mapped_method}\""""
            elif unequal_var:
                function_name = f"power_continuous{'_sim' if method == 'simulation' else ''}"
                core_params = f"""n1={n1},
    n2={n2},
    mean1={mean1},
    mean2={mean2},
    sd1={std_dev},
    sd2={std_dev2},
    alpha={alpha}"""
            else:
                function_name = f"power_continuous{'_sim' if method == 'simulation' else ''}"
                core_params = f"""n1={n1},
    n2={n2},
    mean1={mean1},
    mean2={mean2},
    std_dev={std_dev},
    alpha={alpha}"""
        else:  # Non-Inferiority
            function_name = f"power_continuous_non_inferiority{'_sim' if method == 'simulation' else ''}"
            core_params = f"""n1={n1},
    n2={n2},
    mean1={mean1},
    std_dev={std_dev},
    non_inferiority_margin={non_inferiority_margin},
    alpha={alpha},
    assumed_difference={assumed_difference_ni},
    direction="{non_inferiority_direction}",
    allocation_ratio={allocation_ratio}"""
        
        if method == "simulation":
            sim_params = f"""nsim={nsim},"""
            if seed is not None:
                sim_params += f"""
    seed={seed},"""
            
            all_params = core_params + ",\n    " + sim_params.strip()
        else:
            all_params = core_params
        
        result_key = "power"
        
    elif calc_type == "Sample Size":
        if hypothesis_type == "Superiority":
            if repeated_measures:
                function_name = f"sample_size_repeated_measures{'_sim' if method == 'simulation' else ''}"
                # Map UI method names to function expected names
                method_mapping = {"ANCOVA": "ancova", "Change Score": "change_score"}
                mapped_method = method_mapping.get(analysis_method, "ancova")
                
                core_params = f"""mean1={mean1},
    mean2={mean2},
    sd1={std_dev},
    correlation={correlation},
    power={power},
    alpha={alpha},
    allocation_ratio={allocation_ratio},
    method="{mapped_method}\""""
            elif unequal_var:
                function_name = f"sample_size_continuous{'_sim' if method == 'simulation' else ''}"
                core_params = f"""mean1={mean1},
    mean2={mean2},
    sd1={std_dev},
    sd2={std_dev2},
    power={power},
    alpha={alpha},
    allocation_ratio={allocation_ratio}"""
            else:
                function_name = f"sample_size_continuous{'_sim' if method == 'simulation' else ''}"
                core_params = f"""mean1={mean1},
    mean2={mean2},
    std_dev={std_dev},
    power={power},
    alpha={alpha},
    allocation_ratio={allocation_ratio}"""
        else:  # Non-Inferiority
            function_name = f"sample_size_continuous_non_inferiority{'_sim' if method == 'simulation' else ''}"
            core_params = f"""mean1={mean1},
    std_dev={std_dev},
    non_inferiority_margin={non_inferiority_margin},
    power={power},
    alpha={alpha},
    allocation_ratio={allocation_ratio},
    assumed_difference={assumed_difference_ni},
    direction="{non_inferiority_direction}" """
        
        if method == "simulation":
            sim_params = f"""nsim={nsim},"""
            if seed is not None:
                sim_params += f"""
    seed={seed},"""
            
            all_params = core_params + ",\n    " + sim_params.strip()
        else:
            all_params = core_params
            
        result_key = "total_sample_size"
        
    elif calc_type == "Minimum Detectable Effect":
        if hypothesis_type == "Superiority":
            if repeated_measures:
                function_name = f"min_detectable_effect_repeated_measures{'_sim' if method == 'simulation' else ''}"
                # Map UI method names to function expected names
                method_mapping = {"ANCOVA": "ancova", "Change Score": "change_score"}
                mapped_method = method_mapping.get(analysis_method, "ancova")
                
                core_params = f"""n1={n1},
    n2={n2},
    sd1={std_dev},
    correlation={correlation},
    power={power},
    alpha={alpha},
    method="{mapped_method}\""""
            elif unequal_var:
                function_name = f"min_detectable_effect_continuous{'_sim' if method == 'simulation' else ''}"
                core_params = f"""n1={n1},
    n2={n2},
    sd1={std_dev},
    sd2={std_dev2},
    power={power},
    alpha={alpha}"""
            else:
                function_name = f"min_detectable_effect_continuous{'_sim' if method == 'simulation' else ''}"
                core_params = f"""n1={n1},
    n2={n2},
    std_dev={std_dev},
    power={power},
    alpha={alpha}"""
        else:  # Non-Inferiority
            function_name = f"min_detectable_continuous_non_inferiority_margin{'_sim' if method == 'simulation' else ''}"
            core_params = f"""n1={n1},
    n2={n2},
    mean1={mean1},
    std_dev={std_dev},
    power={power},
    alpha={alpha},
    assumed_difference={assumed_difference_ni},
    direction="{non_inferiority_direction}" """
        
        if method == "simulation":
            sim_params = f"""nsim={nsim},"""
            if seed is not None:
                sim_params += f"""
    seed={seed},"""
            
            all_params = core_params + ",\n    " + sim_params.strip()
        else:
            all_params = core_params
            
        if hypothesis_type == "Superiority":
            result_key = "minimum_detectable_effect"
        else:
            result_key = "minimum_detectable_margin"
    
    # Get algorithm source code based on the function being used
    module_path_map = {
        "power_continuous": "core.designs.parallel.analytical_continuous",
        "power_continuous_sim": "core.designs.parallel.simulation_continuous", 
        "sample_size_continuous": "core.designs.parallel.analytical_continuous",
        "sample_size_continuous_sim": "core.designs.parallel.simulation_continuous",
        "power_repeated_measures": "core.designs.parallel.analytical_continuous",
        "sample_size_repeated_measures": "core.designs.parallel.analytical_continuous",
        "power_continuous_non_inferiority": "core.designs.parallel.analytical_continuous",
        "sample_size_continuous_non_inferiority": "core.designs.parallel.analytical_continuous",
        "min_detectable_continuous_non_inferiority_margin": "core.designs.parallel.analytical_continuous"
    }
    
    # Get the actual function and its source code
    module_path = module_path_map.get(function_name, "core.designs.parallel.analytical_continuous")
    actual_function = get_function_from_name(module_path, function_name)
    
    if actual_function:
        algorithm_source = get_function_source(actual_function)
        algorithm_section = f"""
# ============================================================================
# STATISTICAL ALGORITHM SOURCE CODE
# 
# The following shows the exact implementation of the {function_name} function
# used in this analysis. This provides full transparency of the statistical
# methodology and allows verification of the approach.
# ============================================================================

{algorithm_source}

# ============================================================================
# MAIN ANALYSIS USING THE ALGORITHM SHOWN ABOVE
# ============================================================================
"""
    else:
        algorithm_section = f"""
# ============================================================================
# STATISTICAL ALGORITHM: {function_name}
# Module: {module_path}
# 
# Note: Algorithm source code could not be extracted automatically.
# The function {function_name} from {module_path} is being used.
# ============================================================================
"""
    
    # Generate detailed study summary
    effect_size = abs(mean2 - mean1) if calc_type != "Minimum Detectable Effect" else "TBD"
    
    # Add design details
    design_type = "Repeated Measures " if repeated_measures else ""
    analysis_note = f" ({analysis_method} analysis)" if repeated_measures else ""
    variance_note = " with unequal variances" if unequal_var and not repeated_measures else ""
    
    study_summary = f"""# STUDY DESIGN SUMMARY:
# Design: {design_type}Parallel Group Randomized Controlled Trial
# Outcome Type: Continuous (mean difference analysis{analysis_note})
# Hypothesis: {hypothesis_type}
# Statistical Method: {method.title()}
#
# STUDY PARAMETERS:"""
    
    if calc_type == "Sample Size":
        study_summary += f"""
# - Control group mean: {mean1}
# - Treatment group mean: {mean2} (effect size: {effect_size})
# - Standard deviation: {std_dev}"""
        
        if repeated_measures:
            study_summary += f"""
# - Repeated measures correlation: {correlation}
# - Analysis method: {analysis_method}"""
        elif unequal_var:
            study_summary += f"""
# - Group 2 standard deviation: {std_dev2}"""
            
        # Safe power formatting for sample size calculations
        power_text = f"{power*100:.0f}%" if power is not None else "TBD"
        
        study_summary += f"""
# - Desired statistical power: {power_text}
# - Significance level (alpha): {alpha*100:.0f}%
# - Allocation ratio (treatment:control): {allocation_ratio}:1
#
# RESEARCH QUESTION:
# How many participants are needed to detect a mean difference of {effect_size}
# with {power_text} power at {alpha*100:.0f}% significance level?"""
    
    elif calc_type == "Power":
        study_summary += f"""
# - Sample size group 1: {n1}
# - Sample size group 2: {n2} (total: {n1 + n2})
# - Control group mean: {mean1}
# - Treatment group mean: {mean2} (effect size: {effect_size})
# - Standard deviation: {std_dev}"""
        
        if repeated_measures:
            study_summary += f"""
# - Repeated measures correlation: {correlation}
# - Analysis method: {analysis_method}"""
        elif unequal_var:
            study_summary += f"""
# - Group 2 standard deviation: {std_dev2}"""
            
        study_summary += f"""
# - Significance level (alpha): {alpha*100:.0f}%
# - Allocation ratio (treatment:control): {allocation_ratio}:1
#
# RESEARCH QUESTION:
# What is the statistical power to detect a mean difference of {effect_size}
# with {n1 + n2} total participants at {alpha*100:.0f}% significance level?"""
    
    elif calc_type == "Minimum Detectable Effect":
        study_summary += f"""
# - Sample size group 1: {n1}
# - Sample size group 2: {n2} (total: {n1 + n2})
# - Control group mean: {mean1}
# - Standard deviation: {std_dev}"""
        
        if repeated_measures:
            study_summary += f"""
# - Repeated measures correlation: {correlation}
# - Analysis method: {analysis_method}"""
        elif unequal_var:
            study_summary += f"""
# - Group 2 standard deviation: {std_dev2}"""
            
        # Safe power formatting for MDE calculations  
        power_text = f"{power*100:.0f}%" if power is not None else "TBD"
        
        study_summary += f"""
# - Desired statistical power: {power_text}
# - Significance level (alpha): {alpha*100:.0f}%
# - Allocation ratio (treatment:control): {allocation_ratio}:1
#
# RESEARCH QUESTION:
# What is the minimum detectable effect size (mean difference)
# with {n1 + n2} participants and {power_text} power at {alpha*100:.0f}% significance?"""

    if method == "simulation":
        study_summary += f"""
#
# SIMULATION SETTINGS:
# - Number of simulations: {nsim:,}
# - Random seed: {seed if seed else 'Not set (results will vary)'}"""

    # Generate clean, simple code with usage instructions
    code = f"""{study_summary}
#
# ====================================================================
# ENHANCED REPRODUCIBLE ANALYSIS SCRIPT WITH ALGORITHM TRANSPARENCY
# Generated by DesignPower - https://github.com/your-repo/DesignPower
# ====================================================================
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

{algorithm_section}

# Calculate {calc_type.lower()} using the algorithm shown above
result = {function_name}(
    {all_params}
)

print(f"\\n{'='*60}")
print(f"RESULT: {calc_type.upper()}")
print(f"{'='*60}")
print(f"{calc_type}: {{result['{result_key}']:.3f}}")
# Safe formatting for effect size
effect_size = result.get('effect_size')
if effect_size is not None:
    print(f"Effect size (Cohen's d): {{effect_size:.3f}}")
else:
    print("Effect size (Cohen's d): N/A")
if 'n1' in result and 'n2' in result:
    print(f"Group 1 sample size: {{result['n1']}}")
    print(f"Group 2 sample size: {{result['n2']}}")

print(f"\\n{'='*60}")
print("INTERPRETATION")
print(f"{'='*60}")"""

    # Add interpretation based on calculation type
    if calc_type == "Sample Size":
        # Safe power formatting for interpretation section
        power_text = f"{power*100:.0f}%" if power is not None else "the specified"
        
        interpretation = f"""
print("To detect a mean difference of {effect_size} between groups:")
print("- Control group (expected mean: {mean1})")  
print("- Treatment group (expected mean: {mean2})")
print("- With {power_text} power at {alpha*100:.0f}% significance level")
print("- You need {{result['{result_key}']:.0f}} total participants")
if 'n1' in result and 'n2' in result:
    print("- Group allocation: {{result['n1']:.0f}} control, {{result['n2']:.0f}} treatment")"""
    
    elif calc_type == "Power":
        interpretation = f"""
print("With {n1 + n2} total participants ({n1} control, {n2} treatment):")
print("- To detect a mean difference of {effect_size}")
print("- At {alpha*100:.0f}% significance level") 
print("- Your study has {{result['{result_key}']*100:.1f}}% statistical power")
print("- This means a {{result['{result_key}']*100:.1f}}% chance of detecting the effect if it exists")"""
    
    elif calc_type == "Minimum Detectable Effect":
        # Safe power formatting for MDE interpretation
        power_text = f"{power*100:.0f}%" if power is not None else "the specified"
        
        interpretation = f"""
print("With {n1 + n2} total participants ({n1} control, {n2} treatment):")
print("- At {alpha*100:.0f}% significance level with {power_text} power")
print("- The minimum detectable mean difference is: {{result['{result_key}']:.3f}}")
if result.get('effect_size') is not None:
    print("- Effect size (Cohen's d): {{result['effect_size']:.3f}}")
else:
    print("- Effect size (Cohen's d): N/A")
print("- Smaller differences would be undetectable with this sample size")"""
    else:
        interpretation = ""
    
    # Complete the code string
    code += interpretation + f"""

print(f"\\n{'='*60}")
print("ALGORITHM INFORMATION")
print(f"{'='*60}")
print(f"Statistical function: {function_name}")
print(f"Source module: {module_path}")
print("Complete algorithm source code: See implementation above")

print(f"\\n{'='*60}")
print("FULL STATISTICAL RESULTS")  
print(f"{'='*60}")

# Complete results with all statistical details
import json
print(json.dumps(result, indent=2))"""

    return code