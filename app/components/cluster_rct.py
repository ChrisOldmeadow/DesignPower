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
    """Generates a Python CLI script string for cluster continuous RCTs."""
    calc_type = params.get("calculation_type", "Sample Size")
    method = params.get("method", "analytical") # analytical or simulation

    alpha = params.get("alpha", 0.05)
    power = params.get("power", 0.8)
    
    mean1 = params.get("mean1", 0.0)
    mean2 = params.get("mean2", 1.0)
    std_dev = params.get("std_dev", 1.0)
    icc = params.get("icc", 0.05)

    # Parameters for specific calculation types
    n_clusters_ui = params.get("n_clusters", 10) # Used for Power/MDE and fixed for SS
    cluster_size_ui = params.get("cluster_size", 50) # Used for Power/MDE and fixed for SS
    
    solve_for_ui = params.get("solve_for_continuous", "Number of Clusters (k)")
    solve_for_core = 'n_clusters' if "Number of Clusters" in solve_for_ui else 'cluster_size'

    # Simulation specific
    nsim = params.get("nsim", 1000)
    seed = params.get("seed", 42)
    
    # Simulation analysis method mapping
    analysis_method_ui = params.get("analysis_method", "T-test on Aggregate Data")
    analysis_model_map = {
        "T-test on Aggregate Data": "ttest",
        "Linear Mixed Model (LMM)": "mixedlm",
        "GEE": "gee",
        "Bayesian GLMM": "bayes" # Assuming 'bayes' is the general Bayesian model for continuous
    }
    analysis_model_core = analysis_model_map.get(analysis_method_ui, "ttest")

    # LMM specific simulation parameters
    lmm_use_satterthwaite = params.get("use_satterthwaite", False)
    lmm_optimization_method_ui = params.get("lmm_method", "Auto")
    lmm_method_core = lmm_optimization_method_ui.lower() if lmm_optimization_method_ui != "Auto" else "auto"
    lmm_use_reml = params.get("lmm_reml", True)

    # Sample Size simulation search range
    min_n_clusters_sim = params.get("min_n_clusters", 2)
    max_n_clusters_sim = params.get("max_n_clusters", 100)
    min_cluster_size_sim = params.get("min_cluster_size", 2)
    max_cluster_size_sim = params.get("max_cluster_size", 500)

    # MDE simulation specific
    precision_mde_sim = params.get("precision", 0.01)
    max_iterations_mde_sim = params.get("max_iterations", 100)

    script_template = textwrap.dedent(f"""
    import argparse
    import json
    import sys
    import os
    import math

    # Attempt to import core modules
    try:
        from core.designs.cluster_rct import analytical_continuous
        from core.designs.cluster_rct import simulation_continuous
    except ImportError:
        sys.stderr.write("Error: Could not import DesignPower's core cluster RCT modules.\n")
        sys.stderr.write("Please ensure the script is run from the DesignPower project root directory,\n")
        sys.stderr.write("or that the DesignPower package is installed / 'core' is in PYTHONPATH.\n")
        sys.exit(1)

    def main():
        parser = argparse.ArgumentParser(description="Reproducible CLI for Cluster RCT Continuous Outcome - DesignPower")
        parser.add_argument('--calculation_type', type=str, required=True, choices=['Sample Size', 'Power', 'Minimum Detectable Effect'], default='{calc_type}')
        parser.add_argument('--method', type=str, required=True, choices=['analytical', 'simulation'], default='{method}')
        
        parser.add_argument('--alpha', type=float, default={alpha})
        parser.add_argument('--power', type=float, default={power})
        
        parser.add_argument('--mean1', type=float, default={mean1})
        parser.add_argument('--mean2', type=float, default={mean2})
        parser.add_argument('--std_dev', type=float, default={std_dev})
        parser.add_argument('--icc', type=float, default={icc})

        # For Power/MDE, n_clusters and cluster_size are inputs.
        # For Sample Size, one is an input (fixed), the other is solved for.
        parser.add_argument('--n_clusters', type=int, default={n_clusters_ui if calc_type != 'Sample Size' or solve_for_core == 'cluster_size' else None}, help="Number of clusters per arm. Input for Power/MDE, or fixed value if solving for cluster_size in Sample Size.")
        parser.add_argument('--cluster_size', type=int, default={cluster_size_ui if calc_type != 'Sample Size' or solve_for_core == 'n_clusters' else None}, help="Average cluster size. Input for Power/MDE, or fixed value if solving for n_clusters in Sample Size.")
        parser.add_argument('--solve_for', type=str, choices=['n_clusters', 'cluster_size'], default='{solve_for_core if calc_type == "Sample Size" else None}', help="Specify for Sample Size calculation: solve for 'n_clusters' or 'cluster_size'.")

        # Simulation specific
        parser.add_argument('--nsim', type=int, default={nsim})
        parser.add_argument('--seed', type=int, default={seed})
        parser.add_argument('--analysis_model', type=str, choices=['ttest', 'mixedlm', 'gee', 'bayes'], default='{analysis_model_core}', help="Analysis model for simulation: ttest, mixedlm, gee, bayes.")
        parser.add_argument('--lmm_use_satterthwaite', action='store_true', default={lmm_use_satterthwaite})
        parser.add_argument('--lmm_method', type=str, default='{lmm_method_core}')
        parser.add_argument('--lmm_reml', action='store_true', default={lmm_use_reml})

        # Sample Size simulation search range
        parser.add_argument('--min_n_clusters_sim', type=int, default={min_n_clusters_sim})
        parser.add_argument('--max_n_clusters_sim', type=int, default={max_n_clusters_sim})
        parser.add_argument('--min_cluster_size_sim', type=int, default={min_cluster_size_sim})
        parser.add_argument('--max_cluster_size_sim', type=int, default={max_cluster_size_sim})

        # MDE simulation specific
        parser.add_argument('--precision_mde_sim', type=float, default={precision_mde_sim})
        parser.add_argument('--max_iterations_mde_sim', type=int, default={max_iterations_mde_sim})

        args = parser.parse_args()

        results = None

        # --- Analytical Calculations --- 
        if args.method == 'analytical':
            if args.calculation_type == 'Sample Size':
                if args.solve_for == 'n_clusters':
                    if args.cluster_size is None:
                        sys.stderr.write("Error: --cluster_size must be provided when solving for n_clusters.\n")
                        sys.exit(1)
                    results = analytical_continuous.sample_size_continuous(
                        mean1=args.mean1, mean2=args.mean2, std_dev=args.std_dev, icc=args.icc,
                        power=args.power, alpha=args.alpha, cluster_size=args.cluster_size, n_clusters_fixed=None
                    )
                elif args.solve_for == 'cluster_size':
                    if args.n_clusters is None:
                        sys.stderr.write("Error: --n_clusters must be provided when solving for cluster_size.\n")
                        sys.exit(1)
                    results = analytical_continuous.sample_size_continuous(
                        mean1=args.mean1, mean2=args.mean2, std_dev=args.std_dev, icc=args.icc,
                        power=args.power, alpha=args.alpha, cluster_size=None, n_clusters_fixed=args.n_clusters
                    )
                else:
                    sys.stderr.write("Error: Invalid --solve_for value for Sample Size calculation.\n")
                    sys.exit(1)
            elif args.calculation_type == 'Power':
                if args.n_clusters is None or args.cluster_size is None:
                    sys.stderr.write("Error: --n_clusters and --cluster_size must be provided for Power calculation.\n")
                    sys.exit(1)
                results = analytical_continuous.power_continuous(
                    n_clusters=args.n_clusters, cluster_size=args.cluster_size, icc=args.icc,
                    mean1=args.mean1, mean2=args.mean2, std_dev=args.std_dev, alpha=args.alpha
                )
            elif args.calculation_type == 'Minimum Detectable Effect':
                if args.n_clusters is None or args.cluster_size is None:
                    sys.stderr.write("Error: --n_clusters and --cluster_size must be provided for MDE calculation.\n")
                    sys.exit(1)
                results = analytical_continuous.min_detectable_effect_continuous(
                    n_clusters=args.n_clusters, cluster_size=args.cluster_size, icc=args.icc,
                    std_dev=args.std_dev, power=args.power, alpha=args.alpha
                )
        # --- Simulation Calculations --- 
        elif args.method == 'simulation':
            sim_params = dict(
                mean1=args.mean1, mean2=args.mean2, std_dev=args.std_dev, icc=args.icc,
                power=args.power, alpha=args.alpha, nsim=args.nsim, seed=args.seed,
                analysis_model=args.analysis_model,
                lmm_use_satterthwaite=args.lmm_use_satterthwaite,
                lmm_method=args.lmm_method,
                lmm_reml=args.lmm_reml
            )
            if args.calculation_type == 'Sample Size':
                ss_sim_params = sim_params.copy()
                if args.solve_for == 'n_clusters':
                    if args.cluster_size is None:
                        sys.stderr.write("Error: --cluster_size must be provided when solving for n_clusters (simulation).\n")
                        sys.exit(1)
                    ss_sim_params.update(dict(
                        cluster_size=args.cluster_size, n_clusters_fixed=None,
                        min_n_clusters=args.min_n_clusters_sim, max_n_clusters=args.max_n_clusters_sim,
                        min_cluster_size=args.min_cluster_size_sim, max_cluster_size=args.max_cluster_size_sim # Pass full range
                    ))
                    results = simulation_continuous.sample_size_continuous_sim(**ss_sim_params)
                elif args.solve_for == 'cluster_size':
                    if args.n_clusters is None:
                        sys.stderr.write("Error: --n_clusters must be provided when solving for cluster_size (simulation).\n")
                        sys.exit(1)
                    ss_sim_params.update(dict(
                        cluster_size=None, n_clusters_fixed=args.n_clusters,
                        min_n_clusters=args.min_n_clusters_sim, max_n_clusters=args.max_n_clusters_sim,
                        min_cluster_size=args.min_cluster_size_sim, max_cluster_size=args.max_cluster_size_sim
                    ))
                    results = simulation_continuous.sample_size_continuous_sim(**ss_sim_params)
                else:
                    sys.stderr.write("Error: Invalid --solve_for value for Sample Size simulation.\n")
                    sys.exit(1)
            elif args.calculation_type == 'Power':
                if args.n_clusters is None or args.cluster_size is None:
                    sys.stderr.write("Error: --n_clusters and --cluster_size must be provided for Power simulation.\n")
                    sys.exit(1)
                power_sim_params = sim_params.copy()
                power_sim_params.update(dict(n_clusters=args.n_clusters, cluster_size=args.cluster_size))
                results = simulation_continuous.power_continuous_sim(**power_sim_params)
            elif args.calculation_type == 'Minimum Detectable Effect':
                if args.n_clusters is None or args.cluster_size is None:
                    sys.stderr.write("Error: --n_clusters and --cluster_size must be provided for MDE simulation.\n")
                    sys.exit(1)
                mde_sim_params = sim_params.copy()
                mde_sim_params.update(dict(
                    n_clusters=args.n_clusters, cluster_size=args.cluster_size,
                    precision=args.precision_mde_sim, max_iterations=args.max_iterations_mde_sim
                ))
                results = simulation_continuous.min_detectable_effect_continuous_sim(**mde_sim_params)

        if results:
            # Convert numpy types to native Python types for JSON serialization
            def convert_numpy_types(obj):
                if isinstance(obj, (np.integer, np.int64)):
                    return int(obj)
                elif isinstance(obj, (np.floating, np.float64)):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_numpy_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(i) for i in obj]
                return obj
            results_serializable = convert_numpy_types(results)
            print(json.dumps(results_serializable, indent=4))
        else:
            sys.stderr.write("Error: Could not calculate results.\n")
            sys.exit(1)

    if __name__ == "__main__":
        # Need to import numpy for the convert_numpy_types function if it's used with results containing numpy types
        import numpy as np 
        main()
    """)
    return script_template




def generate_cli_code_cluster_binary(params):
    calc_type = params.get('calc_type_bin', 'Power')
    method = params.get('method_bin', 'analytical') # 'analytical' or 'simulation'

    # Default analysis method for simulation
    default_analysis_method_sim_ui = params.get('analysis_method_bin_sim', "Design Effect Adjusted Z-test")
    # analysis_method_map_binary_sim is defined at module level
    default_analysis_method_sim_backend = analysis_method_map_binary_sim.get(default_analysis_method_sim_ui, "deff_ztest")

    # For MDE, effect_measure_bin determines the output type.
    default_mde_result_type = params.get('effect_measure_bin', 'risk_difference') # For MDE output type
    
    # Determine default effect measure and value for p2 derivation if p2 is not provided
    p2_bin_val = params.get("p2_bin")
    effect_measure_bin_val = params.get('effect_measure_bin')
    effect_value_bin_val = params.get('effect_value_bin')

    default_effect_measure_input_val_str = f"'{effect_measure_bin_val}'" if p2_bin_val is None and effect_measure_bin_val is not None else 'None'
    default_effect_value_input_val_str = str(effect_value_bin_val) if p2_bin_val is None and effect_value_bin_val is not None else 'None'
    
    p1_default_val = params.get('p1_bin', 0.1)
    p2_default_val_str = 'None' if p2_bin_val is None else str(p2_bin_val)
    
    icc_default_val = params.get('icc_bin', 0.01)
    cluster_size_default_val = params.get('cluster_size_bin', 50)
    n_clusters_default_val = params.get('n_clusters_bin', 10) # For Power/MDE
    power_default_val = params.get('power_bin', 0.8) # For SS/MDE
    alpha_default_val = params.get('alpha_bin', 0.05)
    cv_cluster_size_default_val = params.get('cv_cluster_size_bin', 0.0)
    
    # Simulation specific
    nsim_default_val = params.get('nsim_bin', 1000)
    seed_default_val_str = 'None' if params.get('seed_bin') is None else str(params.get('seed_bin'))
    
    # SS Sim specific
    min_n_clusters_ss_default_val = params.get('min_n_clusters_bin_ss', 2)
    max_n_clusters_ss_default_val = params.get('max_n_clusters_bin_ss', 100)
    
    # MDE Sim specific
    min_effect_mde_default_val = params.get('min_effect_bin_mde', 0.01)
    max_effect_mde_default_val = params.get('max_effect_bin_mde', 0.5)
    precision_mde_default_val = params.get('precision_bin_mde', 0.01)
    max_iterations_mde_default_val = params.get('max_iterations_bin_mde', 10)

    cluster_sizes_list_default_val_str = params.get('cluster_sizes_list_bin_str', '')
    if not cluster_sizes_list_default_val_str: # Ensure it's 'None' for argparse if empty
        cluster_sizes_list_default_val_str = 'None'
    else:
        cluster_sizes_list_default_val_str = f"'{cluster_sizes_list_default_val_str}'"

    script_template = textwrap.dedent(f"""
    import argparse
    import json
    import textwrap
    import sys
    import os
    import numpy as np
    import math # Added math for potential use in core functions if not already there

    # --- Path Setup --- Start
    # This setup allows the script to be run from different locations within the project.
    try:
        current_script_path = os.path.abspath(__file__)
        # Assuming script is in 'scripts' or similar, and 'core' is at 'project_root/core'
        # Adjust os.pardir count based on actual script location relative to project root.
        # If script is at project_root/scripts/my_script.py, then project_root is two levels up.
        project_root = os.path.abspath(os.path.join(os.path.dirname(current_script_path), os.pardir, os.pardir))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        
        # Attempt to import after path adjustment
        from core.designs.cluster_rct import analytical_binary as analytical_binary_module
        from core.designs.cluster_rct import simulation_binary as simulation_binary_module
        from core.designs.cluster_rct.cluster_utils import convert_effect_measures
    except ImportError as e_import:
        # Fallback: Try assuming script is directly in project_root (e.g. for testing)
        try:
            project_root_alt = os.path.abspath(os.path.join(os.path.dirname(current_script_path), os.pardir))
            if project_root_alt not in sys.path:
                sys.path.insert(0, project_root_alt)
            from core.designs.cluster_rct import analytical_binary as analytical_binary_module
            from core.designs.cluster_rct import simulation_binary as simulation_binary_module
            from core.designs.cluster_rct.cluster_utils import convert_effect_measures
        except ImportError:
            print(f"ImportError: {{e_import}}\nFailed to import DesignPower core modules. \n"
                  f"Attempted project_root: {{project_root}}\nAttempted alt_project_root: {{project_root_alt}}\n"
                  f"Current sys.path: {{sys.path}}\n"
                  f"Please ensure the script is placed correctly (e.g., in a 'scripts' folder at the project root) \n"
                  f"or that the DesignPower package is installed and accessible in your PYTHONPATH.")
            sys.exit(1)
    # --- Path Setup --- End

    def parse_cluster_sizes(cs_str):
        if cs_str is None or cs_str.lower() == 'none' or cs_str == '':
            return None
        try:
            return [int(s.strip()) for s in cs_str.split(',') if s.strip()]
        except ValueError:
            raise argparse.ArgumentTypeError("Cluster sizes must be a comma-separated list of integers.")

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            return super(NumpyEncoder, self).default(obj)

    def main():
        parser = argparse.ArgumentParser(
            description="Reproducible CLI for Cluster RCT Binary Outcome - DesignPower",
            formatter_class=argparse.RawTextHelpFormatter
        )
        
        parser.add_argument('--calculation_type', type=str, required=True, 
                            choices=['Sample Size', 'Power', 'Minimum Detectable Effect'], 
                            default='{calc_type}', help="Type of calculation.")
        parser.add_argument('--method', type=str, required=True, 
                            choices=['analytical', 'simulation'], 
                            default='{method}', help="Calculation method.")

        parser.add_argument('--p1', type=float, required=True, default={p1_default_val}, help="Proportion in control group.")
        parser.add_argument('--p2', type=float, default={p2_default_val_str},
                            help="Proportion in intervention group. If None, use --effect_measure_input & --effect_value_input.")
        parser.add_argument('--effect_measure_input', type=str, 
                            choices=['risk_difference', 'risk_ratio', 'odds_ratio', 'None'],
                            default={default_effect_measure_input_val_str},
                            help="Effect measure to define p2 if p2 is None (e.g., 'risk_difference'). Use 'None' if p2 is provided.")
        parser.add_argument('--effect_value_input', type=float, default={default_effect_value_input_val_str},
                            help="Value for --effect_measure_input, if p2 is None.")
        
        parser.add_argument('--icc', type=float, required=True, default={icc_default_val}, help="Intracluster Correlation Coefficient (ICC).")
        parser.add_argument('--cluster_size', type=float, default={cluster_size_default_val}, help="Average individuals per cluster (used if --cluster_sizes_list is not provided or for some sim functions).")
        parser.add_argument('--n_clusters', type=int, default={n_clusters_default_val if calc_type != 'Sample Size' else 'None'},
                            help="Number of clusters PER ARM (for Power/MDE). Required if not Sample Size calc.")
        
        parser.add_argument('--power', type=float, default={power_default_val if calc_type != 'Power' else 'None'},
                            help="Desired power (1 - beta) (for Sample Size/MDE). Required if not Power calc.")
        parser.add_argument('--alpha', type=float, default={alpha_default_val}, help="Significance level (alpha).")
        parser.add_argument('--cv_cluster_size', type=float, default={cv_cluster_size_default_val},
                            help="Coefficient of variation of cluster sizes (used if --cluster_sizes_list is not provided).")
        parser.add_argument('--cluster_sizes_list', type=str, default={cluster_sizes_list_default_val_str},
                            help="Comma-separated list of cluster sizes (e.g., '20,25,30'). If provided, this is passed to simulation functions. For analytical functions, it's used to derive mean cluster_size and cv_cluster_size.")

        # Simulation specific arguments
        parser.add_argument('--nsim', type=int, default={nsim_default_val}, help="Number of simulations (for simulation method).")
        parser.add_argument('--seed', type=int, default={seed_default_val_str}, help="Random seed for simulations (for simulation method).")
        parser.add_argument('--analysis_method_sim', type=str, 
                            choices=['deff_ztest', 'aggregate_ttest', 'glmm', 'gee'], 
                            default='{default_analysis_method_sim_backend}',
                            help="Analysis method for simulations (deff_ztest, aggregate_ttest, glmm, gee). Only for simulation method.")
        
        # Sample Size Simulation specific
        parser.add_argument('--min_n_clusters_ss', type=int, default={min_n_clusters_ss_default_val}, help="Min clusters for Sample Size simulation search.")
        parser.add_argument('--max_n_clusters_ss', type=int, default={max_n_clusters_ss_default_val}, help="Max clusters for Sample Size simulation search.")

        # MDE specific (analytical and simulation)
        parser.add_argument('--mde_result_type', type=str, 
                            choices=['risk_difference', 'risk_ratio', 'odds_ratio'], 
                            default='{default_mde_result_type}', help="Output type for Minimum Detectable Effect (e.g. 'risk_difference').")
        # MDE Simulation specific
        parser.add_argument('--min_effect_mde', type=float, default={min_effect_mde_default_val}, help="Min effect for MDE simulation search.")
        parser.add_argument('--max_effect_mde', type=float, default={max_effect_mde_default_val}, help="Max effect for MDE simulation search.")
        parser.add_argument('--precision_mde', type=float, default={precision_mde_default_val}, help="Precision for MDE simulation search.")
        parser.add_argument('--max_iterations_mde', type=int, default={max_iterations_mde_default_val}, help="Max iterations for MDE simulation search.")

        args = parser.parse_args()
        
        # Process p2 based on inputs
        p2_to_use = args.p2
        if args.p2 is None:
            if args.effect_measure_input and args.effect_measure_input != 'None' and args.effect_value_input is not None:
                try:
                    effect_details = convert_effect_measures(p1=args.p1, measure_type=args.effect_measure_input, measure_value=args.effect_value_input)
                    p2_to_use = effect_details.get('p2')
                    if p2_to_use is None:
                         print(f"Error: Could not derive p2 from {{args.effect_measure_input}}='{{args.effect_value_input}}'.")
                         sys.exit(1)
                except Exception as e_convert:
                    print(f"Error deriving p2: {{e_convert}}")
                    sys.exit(1)
            elif args.calculation_type != 'Minimum Detectable Effect': # MDE calculates p2 internally
                print("Error: p2 is required, or --effect_measure_input and --effect_value_input must be provided to derive p2.")
                sys.exit(1)

        # Process cluster_sizes_list for analytical vs simulation
        parsed_cluster_sizes = parse_cluster_sizes(args.cluster_sizes_list)
        
        # For analytical, derive cluster_size and cv_cluster_size from list if provided
        analytical_cluster_size = args.cluster_size
        analytical_cv_cluster_size = args.cv_cluster_size
        if parsed_cluster_sizes and len(parsed_cluster_sizes) > 0:
            analytical_cluster_size = np.mean(parsed_cluster_sizes)
            if np.mean(parsed_cluster_sizes) > 0:
                analytical_cv_cluster_size = np.std(parsed_cluster_sizes) / np.mean(parsed_cluster_sizes)
            else:
                analytical_cv_cluster_size = 0.0

        results = {{}}
        if args.method == 'analytical':
            common_analytical_params = dict(icc=args.icc, alpha=args.alpha, 
                                            cluster_size=analytical_cluster_size, 
                                            cv_cluster_size=analytical_cv_cluster_size,
                                            cluster_sizes=None) # Analytical functions use derived cv, not the list directly

            if args.calculation_type == 'Sample Size':
                if p2_to_use is None or args.power is None: sys.exit("Error: p2 (or its components) and power are required for analytical sample size.")
                results = analytical_binary_module.sample_size_binary(p1=args.p1, p2=p2_to_use, power=args.power, 
                                                                    **common_analytical_params)
            elif args.calculation_type == 'Power':
                if args.n_clusters is None or p2_to_use is None: sys.exit("Error: n_clusters and p2 (or its components) are required for analytical power.")
                results = analytical_binary_module.power_binary(n_clusters=args.n_clusters, p1=args.p1, p2=p2_to_use,
                                                              **common_analytical_params)
            elif args.calculation_type == 'Minimum Detectable Effect':
                if args.n_clusters is None or args.power is None: sys.exit("Error: n_clusters and power are required for analytical MDE.")
                results = analytical_binary_module.min_detectable_effect_binary(n_clusters=args.n_clusters, p1=args.p1, power=args.power, 
                                                                              effect_measure=args.mde_result_type, 
                                                                              **common_analytical_params)
        elif args.method == 'simulation':
            # For simulation, cluster_size is average if list not given, list is passed directly if given
            sim_cluster_size_param = args.cluster_size if parsed_cluster_sizes is None else None 
            sim_cv_param = args.cv_cluster_size if parsed_cluster_sizes is None else 0.0 # CV is derived if list is given

            common_sim_params = dict(icc=args.icc, alpha=args.alpha, nsim=args.nsim, seed=args.seed, 
                                     cluster_size=sim_cluster_size_param, # Average cluster size
                                     cv_cluster_size=sim_cv_param, # CV of cluster sizes
                                     cluster_sizes=parsed_cluster_sizes, # Actual list of sizes
                                     analysis_method=args.analysis_method_sim)

            if args.calculation_type == 'Sample Size':
                if p2_to_use is None or args.power is None: sys.exit("Error: p2 (or its components) and power are required for simulation sample size.")
                results = simulation_binary_module.sample_size_binary_sim(p1=args.p1, p2=p2_to_use, power=args.power, 
                                                                        min_n=args.min_n_clusters_ss, max_n=args.max_n_clusters_ss,
                                                                        **common_sim_params)
            elif args.calculation_type == 'Power':
                if args.n_clusters is None or p2_to_use is None: sys.exit("Error: n_clusters and p2 (or its components) are required for simulation power.")
                results = simulation_binary_module.power_binary_sim(n_clusters=args.n_clusters, p1=args.p1, p2=p2_to_use, 
                                                                  **common_sim_params)
            elif args.calculation_type == 'Minimum Detectable Effect':
                if args.n_clusters is None or args.power is None: sys.exit("Error: n_clusters and power are required for simulation MDE.")
                results = simulation_binary_module.min_detectable_effect_binary_sim(n_clusters=args.n_clusters, p1=args.p1, power=args.power, 
                                                                                  min_effect=args.min_effect_mde, max_effect=args.max_effect_mde,
                                                                                  precision=args.precision_mde, max_iterations=args.max_iterations_mde,
                                                                                  effect_measure=args.mde_result_type, 
                                                                                  **common_sim_params)
        
        print(json.dumps(results, indent=4, cls=NumpyEncoder))

    if __name__ == "__main__":
        main()
    """)
    return script_template


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
        model_options_binary = {
            "Design Effect Adjusted Z-test": "deff_ztest",
            "T-test on Aggregate Data": "aggregate_ttest",
            "GLMM (Individual-Level)": "glmm",
            "GEE (Individual-Level)": "gee"
        }
        selected_model_display_binary = st.selectbox(
            "Statistical Model Used to Analyse Each Simulated Trial",
            options=list(model_options_binary.keys()),
            index=0, # Default to Z-test
            key="cluster_binary_model_select",
            help="Select the statistical analysis model to be used in the simulation. \n- 'Design Effect Adjusted Z-test': Uses a z-test adjusted for clustering. \n- 'T-test on Aggregate Data': Performs a t-test on cluster-level summaries. \n- 'GLMM (Individual-Level)': Uses a Generalized Linear Mixed Model (requires individual data simulation). \n- 'GEE (Individual-Level)': Uses Generalized Estimating Equations (requires individual data simulation)."
        )
        advanced_params["analysis_method"] = model_options_binary[selected_model_display_binary]

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
        # Check if cmdstanpy is available for Bayesian option
        try:
            import cmdstanpy
            bayesian_available = True
        except ImportError:
            bayesian_available = False
        
        model_options = [
            "T-test (cluster-level)",
            "Linear Mixed Model (REML)",
            "GEE (Exchangeable)",
        ]
        
        if bayesian_available:
            model_options.append("Bayesian (Stan)")
        else:
            model_options.append("Bayesian (Stan) - Not Available")
        
        model_display = st.selectbox(
            "Statistical Model Used to Analyse Each Simulated Trial",
            model_options,
            index=0,
            key="cluster_continuous_model_select",
            help="Choose the analysis model applied to each simulated dataset. The simple two-sample t-test analyses individual-level data ignoring clustering but with design-effect adjustment. Mixed models explicitly model random cluster intercepts and can provide more power when cluster counts are moderate to large. GEE provides marginal (population-averaged) inference and is robust to some model misspecification, but small-sample bias can be an issue."
        )
        
        # Show installation message if Bayesian is selected but not available
        if "Bayesian" in model_display and not bayesian_available:
            st.error(
                "📦 **Bayesian analysis requires additional installation**\n\n"
                "To use Bayesian analysis, please install cmdstanpy:\n"
                "```bash\n"
                "pip install cmdstanpy\n"
                "```\n"
                "The calculation will fall back to cluster-level t-test if you proceed."
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
            "Bayesian (Stan) - Not Available": "bayes",  # Will fall back to ttest automatically
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
                help="""Choose how to determine statistical significance:
                • **Credible Interval**: 95% credible interval excludes zero (most standard)
                • **Posterior Probability**: >97.5% probability effect is in favorable direction
                • **ROPE**: <5% probability effect is in Region of Practical Equivalence around zero"""
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
        ui_analysis_method = params.get("analysis_method", "Design Effect Adjusted Z-test")
        if ui_analysis_method == "Design Effect Adjusted Z-test":
            backend_analysis_method = "deff_ztest"
        elif ui_analysis_method == "T-test on Aggregate Data":
            backend_analysis_method = "aggregate_ttest"
        elif ui_analysis_method == "GLMM (Individual-Level)":
            backend_analysis_method = "glmm"
        elif ui_analysis_method == "GEE (Individual-Level)":
            backend_analysis_method = "gee"
        else:
            backend_analysis_method = "deff_ztest"  # Default fallback
        
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
                    analysis_method=backend_analysis_method
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
                            analysis_method=backend_analysis_method
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
                    analysis_method=backend_analysis_method
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
                            analysis_method=backend_analysis_method
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
                    analysis_method=backend_analysis_method
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
                            analysis_method=backend_analysis_method
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
