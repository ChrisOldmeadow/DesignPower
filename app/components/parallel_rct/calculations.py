"""Calculation functions for Parallel RCT designs.

This module provides all the calculation functions for parallel RCT
designs with binary, continuous, and survival outcomes.
"""
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import specific analytical and simulation modules
from core.designs.parallel import analytical_continuous
from core.designs.parallel import simulation_continuous
from core.designs.parallel import analytical_binary
from core.designs.parallel import simulation_binary
from core.designs.parallel import analytical_survival
from core.designs.parallel import simulation_survival


def calculate_parallel_continuous(params):
    """Calculate results for Parallel RCT with continuous outcomes."""
    calc_type = params.get("calculation_type", "Sample Size")
    hypothesis_type = params.get("hypothesis_type", "Superiority")
    use_simulation = params.get("use_simulation", False)
    results_dict = None 

    # Extract common simulation parameters if simulation is used
    nsim = params.get("nsim", 1000) if use_simulation else None
    seed = params.get("seed", 42) if use_simulation else None 
    ui_std_dev = params.get("std_dev", 1.0) # Primary SD from UI
    ui_std_dev2 = params.get("std_dev2", ui_std_dev) # Secondary SD from UI, defaults to primary if not set

    # Extract parameters for sample size simulation if applicable
    min_n_sim = params.get("min_n", 10) if use_simulation and calc_type == "Sample Size" else None
    max_n_sim = params.get("max_n", 1000) if use_simulation and calc_type == "Sample Size" else None
    step_sim = params.get("step_n", 10) if use_simulation and calc_type == "Sample Size" else None
    
    # Map UI method names to function expected names for repeated measures
    ui_method = params.get("analysis_method", "ANCOVA")
    method_mapping = {
        "ANCOVA": "ancova",
        "Change Score": "change_score"
    }
    analysis_method = method_mapping.get(ui_method, "ancova") 

    if use_simulation:
        if calc_type == "Sample Size":
            if hypothesis_type == "Superiority" or hypothesis_type == "Equivalence":
                delta = abs(params.get("mean1", 0) - params.get("mean2", 1))
                # sample_size_continuous_sim expects std_dev
                results_dict = simulation_continuous.sample_size_continuous_sim(
                    delta=delta,
                    std_dev=ui_std_dev,
                    power=params.get("power", 0.8),
                    alpha=params.get("alpha", 0.05),
                    allocation_ratio=params.get("allocation_ratio", 1.0),
                    nsim=nsim,
                    min_n=min_n_sim,
                    max_n=max_n_sim,
                    step=step_sim,
                    repeated_measures=params.get("repeated_measures", False),
                    correlation=params.get("correlation", 0.5),
                    method=analysis_method,
                    seed=seed
                )
            elif hypothesis_type == "Non-Inferiority":
                # sample_size_continuous_non_inferiority_sim expects std_dev
                results_dict = simulation_continuous.sample_size_continuous_non_inferiority_sim(
                    non_inferiority_margin=params.get("non_inferiority_margin", 0.1),
                    std_dev=ui_std_dev,
                    power=params.get("power", 0.8),
                    alpha=params.get("alpha", 0.05),
                    allocation_ratio=params.get("allocation_ratio", 1.0),
                    nsim=nsim,
                    min_n=min_n_sim,
                    max_n=max_n_sim,
                    step=step_sim,
                    assumed_difference=params.get("assumed_difference", 0.0),
                    direction=params.get("non_inferiority_direction", "lower"),
                    repeated_measures=params.get("repeated_measures", False),
                    correlation=params.get("correlation", 0.5),
                    method=params.get("repeated_measures_method", "change_score"),
                    seed=seed
                )
        elif calc_type == "Power":
            if hypothesis_type == "Superiority" or hypothesis_type == "Equivalence":
                # power_continuous_sim expects sd1, sd2
                results_dict = simulation_continuous.power_continuous_sim(
                    n1=params.get("n1", 100),
                    n2=params.get("n2", 100),
                    mean1=params.get("mean1", 0),
                    mean2=params.get("mean2", 1),
                    sd1=ui_std_dev,
                    sd2=ui_std_dev2, 
                    alpha=params.get("alpha", 0.05),
                    nsim=nsim,
                    test=params.get("test_type_continuous", "t-test"),
                    seed=seed
                )
            elif hypothesis_type == "Non-Inferiority":
                # simulate_continuous_non_inferiority (for power) expects std_dev
                results_dict = simulation_continuous.simulate_continuous_non_inferiority(
                    n1=params.get("n1", 100),
                    n2=params.get("n2", 100),
                    non_inferiority_margin=params.get("non_inferiority_margin", 0.1),
                    std_dev=ui_std_dev,
                    nsim=nsim,
                    alpha=params.get("alpha", 0.05),
                    seed=seed,
                    assumed_difference=params.get("assumed_difference", 0.0),
                    direction=params.get("non_inferiority_direction", "lower"),
                    repeated_measures=params.get("repeated_measures", False),
                    correlation=params.get("correlation", 0.5),
                    method=params.get("repeated_measures_method", "change_score")
                )
        elif calc_type == "Minimum Detectable Effect":
            if hypothesis_type == "Superiority" or hypothesis_type == "Equivalence":
                # min_detectable_effect_continuous_sim expects std_dev
                results_dict = simulation_continuous.min_detectable_effect_continuous_sim(
                    n1=params.get("n1", 100),
                    n2=params.get("n2", 100),
                    std_dev=ui_std_dev,
                    power=params.get("power", 0.8),
                    alpha=params.get("alpha", 0.05),
                    nsim=nsim,
                    precision=params.get("mde_precision", 0.01),
                    repeated_measures=params.get("repeated_measures", False),
                    correlation=params.get("correlation", 0.5),
                    method=params.get("repeated_measures_method", "change_score"),
                    seed=seed
                )
            elif hypothesis_type == "Non-Inferiority":
                # Assuming min_detectable_effect_non_inferiority_sim also expects std_dev
                # If this function doesn't exist or has different params, it will need adjustment
                if hasattr(simulation_continuous, 'min_detectable_effect_non_inferiority_sim'):
                    results_dict = simulation_continuous.min_detectable_effect_non_inferiority_sim(
                        n1=params.get("n1", 100),
                        n2=params.get("n2", 100),
                        std_dev=ui_std_dev,
                        power=params.get("power", 0.8),
                        alpha=params.get("alpha", 0.05),
                        nsim=nsim,
                        non_inferiority_margin=params.get("non_inferiority_margin", 0.1),
                        assumed_difference=params.get("assumed_difference", 0.0),
                        direction=params.get("non_inferiority_direction", "lower"),
                        repeated_measures=params.get("repeated_measures", False),
                        correlation=params.get("correlation", 0.5),
                        method=params.get("repeated_measures_method", "change_score"),
                        seed=seed
                    )
                else:
                    # Placeholder if the function is missing, to avoid crashing
                    results_dict = {"error": "min_detectable_effect_non_inferiority_sim not found"}
                    st.error("MDE Non-Inferiority simulation function not available.")

    else:  # Analytical calculations
        # Extract common parameters with default values to avoid None errors
        mean1 = params.get("mean1", 0.0)
        mean2 = params.get("mean2", 0.5)
        # Analytical functions expect sd1, sd2
        # sd1 comes from ui_std_dev, sd2 from ui_std_dev2 (which defaults to ui_std_dev if unequal_var is false)
        sd1_val = ui_std_dev 
        sd2_val = ui_std_dev2
        alpha = params.get("alpha", 0.05)
        power = params.get("power", 0.8)
        allocation_ratio = params.get("allocation_ratio", 1.0)
        
        unequal_var = params.get("unequal_var", False)
        repeated_measures = params.get("repeated_measures", False)
        correlation = params.get("correlation", 0.5) 

        if calc_type == "Sample Size":
            if hypothesis_type == "Superiority" or hypothesis_type == "Equivalence":
                if repeated_measures:
                    results_dict = analytical_continuous.sample_size_repeated_measures(
                        mean1=mean1, mean2=mean2, sd1=sd1_val, correlation=correlation,
                        power=power, alpha=alpha, allocation_ratio=allocation_ratio, method=analysis_method
                    )
                else:
                    results_dict = analytical_continuous.sample_size_continuous(
                        mean1=mean1, mean2=mean2, sd1=sd1_val, sd2=sd2_val,
                        power=power, alpha=alpha, allocation_ratio=allocation_ratio
                    )
            elif hypothesis_type == "Non-Inferiority":
                results_dict = analytical_continuous.sample_size_continuous_non_inferiority(
                    mean1=mean1, 
                    non_inferiority_margin=params.get("non_inferiority_margin", 0.1),
                    sd1=sd1_val, 
                    sd2=sd2_val, 
                    power=power, 
                    alpha=alpha, 
                    allocation_ratio=allocation_ratio,
                    assumed_difference=params.get("assumed_difference", 0.0),
                    direction=params.get("non_inferiority_direction", "lower")
                )
        elif calc_type == "Power":
            n1 = params.get("n1", 100)
            n2 = params.get("n2", 100)
            if hypothesis_type == "Superiority" or hypothesis_type == "Equivalence":
                if repeated_measures:
                    results_dict = analytical_continuous.power_repeated_measures(
                        n1=n1, n2=n2, mean1=mean1, mean2=mean2, sd1=sd1_val, 
                        correlation=correlation, alpha=alpha, method=analysis_method
                    )
                else:
                    results_dict = analytical_continuous.power_continuous(
                        n1=n1, n2=n2, mean1=mean1, mean2=mean2, sd1=sd1_val, sd2=sd2_val, alpha=alpha
                    )
            elif hypothesis_type == "Non-Inferiority":
                results_dict = analytical_continuous.power_continuous_non_inferiority(
                    n1=n1, n2=n2, mean1=mean1, 
                    non_inferiority_margin=params.get("non_inferiority_margin", 0.1),
                    sd1=sd1_val, sd2=sd2_val, alpha=alpha,
                    assumed_difference=params.get("assumed_difference", 0.0),
                    direction=params.get("non_inferiority_direction", "lower")
                )
        elif calc_type == "Minimum Detectable Effect":
            n1 = params.get("n1", 100)
            n2 = params.get("n2", 100)
            if hypothesis_type == "Superiority" or hypothesis_type == "Equivalence":
                if repeated_measures:
                    results_dict = analytical_continuous.min_detectable_effect_repeated_measures(
                        n1=n1, n2=n2, sd1=sd1_val, correlation=correlation, 
                        power=power, alpha=alpha, method=analysis_method
                    ) 
                else:
                    results_dict = analytical_continuous.min_detectable_effect_continuous(
                        n1=n1, n2=n2, sd1=sd1_val, sd2=sd2_val, power=power, alpha=alpha
                    )
            elif hypothesis_type == "Non-Inferiority":
                # MDE for non-inferiority is typically not calculated in the same way.
                # Usually, the margin is fixed. We can return a message or the margin itself.
                results_dict = {"mde_non_inferiority_info": "MDE is not applicable for non-inferiority in the same sense; the margin is key."}

    # Process and return results
    final_results = {}
    if results_dict:
        if calc_type == "Sample Size":
            final_results["n1"] = round(results_dict.get("n1", results_dict.get("sample_size_1", 0)))
            final_results["n2"] = round(results_dict.get("n2", results_dict.get("sample_size_2", 0)))
            final_results["total_n"] = final_results["n1"] + final_results["n2"]
            if not use_simulation: 
                 final_results["mean_difference"] = params.get("mean2", 1) - params.get("mean1", 0)
            elif "delta" in results_dict: 
                 final_results["mean_difference"] = results_dict.get("delta")
            # Add effect_size if available
            if "effect_size" in results_dict:
                final_results["effect_size"] = results_dict["effect_size"]

        elif calc_type == "Power":
            final_results["power"] = round(results_dict.get("power", results_dict.get("empirical_power", 0)), 3)
            final_results["n1"] = params.get("n1", 0)
            final_results["n2"] = params.get("n2", 0)
            final_results["total_n"] = final_results["n1"] + final_results["n2"]
            if not use_simulation:
                 final_results["mean_difference"] = params.get("mean2", 1) - params.get("mean1", 0)
            elif "mean1" in results_dict and "mean2" in results_dict: 
                 final_results["mean_difference"] = results_dict.get("mean2") - results_dict.get("mean1")
            if use_simulation and "std_dev_note" in results_dict:
                final_results["std_dev_note"] = results_dict["std_dev_note"]

        elif calc_type == "Minimum Detectable Effect":
            if "mde_non_inferiority_info" in results_dict:
                final_results["mde_info"] = results_dict["mde_non_inferiority_info"]
            else:
                final_results["mde"] = round(results_dict.get("mde", results_dict.get("minimum_detectable_effect", 0)), 3)
            final_results["n1"] = params.get("n1", 0)
            final_results["n2"] = params.get("n2", 0)
            final_results["total_n"] = final_results["n1"] + final_results["n2"]
            if use_simulation and "std_dev_note" in results_dict:
                final_results["std_dev_note"] = results_dict["std_dev_note"]
        
        # Add simulation specific info if used
        if use_simulation:
            final_results["simulations"] = nsim
            final_results["seed"] = seed
            if "mean_p_value" in results_dict: 
                final_results["mean_p_value"] = round(results_dict["mean_p_value"], 4)

    return final_results


def calculate_parallel_binary(params):
    """Calculate results for Parallel RCT with binary outcomes."""
    # Get calculation method and hypothesis type from params
    method = params.get("method", "analytical").lower()  # Convert to lowercase
    hypothesis_type = params.get("hypothesis_type", "Superiority")  

    # Extract parameters with default values
    p1 = params.get("p1", 0.3)
    p2 = params.get("p2", 0.5)
    alpha = params.get("alpha", 0.05)
    power = params.get("power", 0.8)
    allocation_ratio = params.get("allocation_ratio", 1.0)
    
    # For non-inferiority, log additional parameters for debugging
    if hypothesis_type == "Non-Inferiority":
        nim = params.get("nim", 0.1)  
        direction = params.get("direction", "Higher is better")
        print(f"Non-inferiority calculation with margin: {nim}, direction: {direction}")
        print(f"Using p1: {p1}, p2: {p2}")
        
        # Add non-inferiority parameters to the calculation
    
    # Handle advanced options
    correction = params.get("correction", "None")
    
    # Prepare result dictionary
    result = {}
    
    # Get simulation-specific parameters
    nsim = params.get("nsim", 1000)
    seed = params.get("seed", 42)
    
    # Get the test type from the advanced params - this is what we set in render_binary_advanced_options
    test_type = params.get("test_type", "Normal Approximation")
    
    # Calculate based on calculation_type
    calculation_type = params.get("calculation_type", "Sample Size")
    
    if calculation_type == "Sample Size":
        # Calculate sample size
        if method == "analytical":
            # Convert test name to test_type expected by the function
            test_name_mapping = {
                "normal approximation": "normal approximation",
                "fisher's exact test": "fishers exact",
                "likelihood ratio test": "likelihood ratio"
            }
            # Convert the test_type to lowercase for case-insensitive matching
            mapped_test_type = test_name_mapping.get(test_type.lower(), "normal approximation")
            
            # Get correction directly from params (set by the checkbox in render_binary_advanced_options)
            has_correction = params.get("correction", False)
            
            # Debug output
            print(f"UI test type: {test_type}, Mapped to: {mapped_test_type}, Correction: {has_correction}")
            
            sample_size = analytical_binary.sample_size_binary(
                p1=p1,
                p2=p2,
                power=power,
                alpha=alpha,
                allocation_ratio=allocation_ratio,
                test_type=mapped_test_type,  
                correction=has_correction  
            )
        elif method == "simulation":
            # Use simulation method for sample size calculation
            # Map the UI test type names to the format expected by the simulation function
            test_name_mapping = {
                "normal approximation": "normal_approximation",
                "fisher's exact test": "fishers_exact",
                "likelihood ratio test": "likelihood_ratio"
            }
            # Convert the test_type to lowercase and map to the correct format
            sim_test_type = test_name_mapping.get(test_type.lower(), "normal_approximation")

            # Debug output
            print(f"Simulation test type: {sim_test_type} from UI: {test_type}")

            sample_size = simulation_binary.sample_size_binary_sim(
                p1=p1,
                p2=p2,
                power=power,
                alpha=alpha,
                allocation_ratio=allocation_ratio,
                nsim=nsim,
                test_type=sim_test_type
            )

        # Extract values from result - handle different key names from different functions
        n1 = sample_size.get("sample_size_1", sample_size.get("n1", 0))
        n2 = sample_size.get("sample_size_2", sample_size.get("n2", 0))
        total_n = sample_size.get("total_sample_size", sample_size.get("total_n", n1 + n2))
        
        # Format results
        result["n1"] = round(n1)
        result["n2"] = round(n2)
        result["total_n"] = round(total_n)
        result["absolute_risk_difference"] = round(p2 - p1, 3)
        result["relative_risk"] = round(p2 / p1, 3) if p1 > 0 else "Infinity"
        result["odds_ratio"] = round((p2 / (1 - p2)) / (p1 / (1 - p1)), 3) if p1 < 1 and p2 < 1 else "Undefined"
        
        return result
        
    elif calculation_type == "Power":
        # Get sample sizes
        n1 = params.get("n1", 50)
        n2 = params.get("n2", 50)
        
        # Calculate power
        if method == "analytical":
            # Convert test name to test_type expected by the function
            test_name_mapping = {
                "normal approximation": "normal approximation",
                "fisher's exact test": "fishers exact",
                "likelihood ratio test": "likelihood ratio"
            }
            # Convert the test_type to lowercase for case-insensitive matching
            mapped_test_type = test_name_mapping.get(test_type.lower(), "normal approximation")
            
            # Check if correction is applied
            has_correction = params.get("correction", "None") != "None"
            
            power_result = analytical_binary.power_binary(
                n1=n1,
                n2=n2,
                p1=p1,
                p2=p2,
                alpha=alpha,
                test_type=mapped_test_type,  
                correction=has_correction  
            )
        elif method == "simulation":
            # Use simulation method for power calculation
            # Map the UI test type names to the format expected by the simulation function
            test_name_mapping = {
                "normal approximation": "normal_approximation",
                "fisher's exact test": "fishers_exact",
                "likelihood ratio test": "likelihood_ratio"
            }
            # Convert the test_type to lowercase and map to the correct format
            sim_test_type = test_name_mapping.get(test_type.lower(), "normal_approximation")

            # Debug output
            print(f"Simulation test type for power: {sim_test_type} from UI: {test_type}")

            power_result = simulation_binary.power_binary_sim(
                n1=n1,
                n2=n2,
                p1=p1,
                p2=p2,
                alpha=alpha,
                nsim=nsim,
                test_type=sim_test_type,
                seed=seed
            )

        # Format results
        result["power"] = round(power_result.get("power", 0), 3)
        result["n1"] = n1
        result["n2"] = n2
        result["total_n"] = n1 + n2
        result["absolute_risk_difference"] = round(p2 - p1, 3)
        result["relative_risk"] = round(p2 / p1, 3) if p1 > 0 else "Infinity"
        result["odds_ratio"] = round((p2 / (1 - p2)) / (p1 / (1 - p1)), 3) if p1 < 1 and p2 < 1 else "Undefined"
        
        return result
        
    elif calculation_type == "Minimum Detectable Effect":
        # Get sample sizes
        n1 = params.get("n1", 50)
        n2 = params.get("n2", 50)
        
        # Calculate MDE
        if method == "analytical":
            # Convert test name to test_type expected by the function
            test_name_mapping = {
                "normal approximation": "normal approximation",
                "fisher's exact test": "fishers exact",
                "likelihood ratio test": "likelihood ratio"
            }
            # Convert the test_type to lowercase for case-insensitive matching
            mapped_test_type = test_name_mapping.get(test_type.lower(), "normal approximation")
            
            # Check if correction is applied
            has_correction = params.get("correction", "None") != "None"
            
            mde_result = analytical_binary.min_detectable_effect_binary(
                n1=n1,
                n2=n2,
                p1=p1,
                power=power,
                alpha=alpha,
                test_type=mapped_test_type,  
                correction=has_correction  
            )
        elif method == "simulation":
            # Use simulation method for minimum detectable effect calculation
            # Map the UI test type names to the format expected by the simulation function
            test_name_mapping = {
                "normal approximation": "normal_approximation",
                "fisher's exact test": "fishers_exact",
                "likelihood ratio test": "likelihood_ratio"
            }
            # Convert the test_type to lowercase and map to the correct format
            sim_test_type = test_name_mapping.get(test_type.lower(), "normal_approximation")
            
            # Debug output
            print(f"Simulation test type for MDE: {sim_test_type} from UI: {test_type}")
            
            mde_result = simulation_binary.min_detectable_effect_binary_sim(
                n1=n1,
                n2=n2,
                p1=p1,
                power=power,
                nsim=nsim,
                alpha=alpha,
                precision=0.01,
                test_type=sim_test_type,
                seed=seed
            )
        
        # Format results
        p2_mde = mde_result.get("p2", 0)
        result["mde"] = round(p2_mde - p1, 3)
        result["p2_mde"] = round(p2_mde, 3)
        result["n1"] = n1
        result["n2"] = n2
        result["total_n"] = n1 + n2
        
        return result
    
    return result


def calculate_parallel_survival(params):
    """Calculate results for Parallel RCT with survival outcome."""
    # Extract parameters with default values
    calculation_type = params.get("calculation_type", "Sample Size")
    hypothesis_type = params.get("hypothesis_type", "Superiority")
    method = params.get("method", "Analytical").lower()
    
    # Advanced method parameters
    advanced_method = params.get("advanced_method", "auto")
    accrual_pattern = params.get("accrual_pattern", "uniform")
    
    # Accrual parameters for non-uniform patterns
    accrual_parameters = {}
    if params.get("ramp_factor"):
        accrual_parameters["ramp_factor"] = params.get("ramp_factor")
    if params.get("growth_rate"):
        accrual_parameters["growth_rate"] = params.get("growth_rate")

    alpha = params.get("alpha", 0.05)
    power = params.get("power", 0.8)
    allocation_ratio = params.get("allocation_ratio", 1.0)
    
    # Study parameters
    accrual_time = params.get("accrual_time", 12.0)
    follow_up_time = params.get("follow_up_time", 24.0)
    dropout_rate = params.get("dropout_rate", 0.1) # Single dropout rate from UI

    # Simulation specific parameters
    nsim = params.get("nsim", 1000)
    min_n_sim = params.get("min_n_sim", 10)
    max_n_sim = params.get("max_n_sim", 500)
    step_n_sim = params.get("step_n_sim", 5)
    seed = params.get("seed", 42) # Added seed for simulation

    # Initialize effective parameters for calculation
    median_survival1 = params.get("median_survival1", 12.0)
    hr_for_calc = None
    median2_for_calc = None
    sides = params.get("sides", 2) # Default to 2 for superiority unless specified
    
    # Specific parameters based on hypothesis type
    non_inferiority_margin_hr_param = None
    assumed_true_hr_param = None

    if hypothesis_type == "Non-Inferiority":
        sides = 1 # Non-inferiority is one-sided by definition
        non_inferiority_margin_hr_param = params.get("non_inferiority_margin_hr", 1.3)
        assumed_true_hr_param = params.get("assumed_true_hr", 1.0)
        
        # For display and simulation, power is based on the assumed true HR
        hr_for_calc = assumed_true_hr_param 
        median2_for_calc = median_survival1 / hr_for_calc if hr_for_calc is not None and hr_for_calc > 0 else float('inf')

    elif hypothesis_type == "Superiority":
        hr_for_calc = params.get("hr", 0.7)
        median2_for_calc = median_survival1 / hr_for_calc if hr_for_calc is not None and hr_for_calc > 0 else float('inf')
        # `sides` will be taken from params.get("sides", 2) as set above
    else:
        return {"error": f"Unknown hypothesis type: {hypothesis_type}"}

    # Prepare result dictionary, including input params for display
    result = {
        "calculation_type_param": calculation_type,
        "hypothesis_type_param": hypothesis_type,
        "method_param": method,
        "alpha_param": alpha,
        "power_param": power if calculation_type != "Power" else None,
        "median_survival1_param": median_survival1,
        "accrual_time_param": accrual_time,
        "follow_up_time_param": follow_up_time,
        "dropout_rate_param": dropout_rate,
        "allocation_ratio_param": allocation_ratio,
        "sides_param": sides
    }
    if hypothesis_type == "Non-Inferiority":
        result["non_inferiority_margin_hr_param"] = non_inferiority_margin_hr_param
        result["assumed_true_hr_param"] = assumed_true_hr_param
        # For NI, calculations use assumed_true_hr to derive median2 for simulation/analytical calls if needed
        # The NI margin itself is passed directly to NI-specific functions.
        result["hr_for_display"] = assumed_true_hr_param # Display the assumed true HR
        result["median_survival2_derived"] = median2_for_calc
    else: # Superiority
        result["hr_param"] = hr_for_calc # hr is the primary effect size parameter for superiority
        result["hr_for_display"] = hr_for_calc # Display the specified HR for superiority
        result["median_survival2_derived"] = median2_for_calc

    if calculation_type == "Sample Size":
        if method == "analytical":
            if hypothesis_type == "Non-Inferiority":
                # Use legacy non-inferiority function for now
                sample_size_result = analytical_survival.sample_size_survival_non_inferiority(
                    median1=median_survival1,
                    non_inferiority_margin=non_inferiority_margin_hr_param,
                    enrollment_period=accrual_time,
                    follow_up_period=follow_up_time,
                    dropout_rate=dropout_rate,
                    power=power,
                    alpha=alpha, # NI function handles one-sided alpha internally
                    allocation_ratio=allocation_ratio,
                    assumed_hazard_ratio=assumed_true_hr_param
                )
            else: # Superiority - use advanced methods
                try:
                    if advanced_method == "compare_all":
                        # Use compare_all to get results from all methods
                        sample_size_result = analytical_survival.sample_size_survival_advanced(
                            hazard_ratio=hr_for_calc,
                            power=power,
                            alpha=alpha,
                            allocation_ratio=allocation_ratio,
                            sides=sides,
                            enrollment_period=accrual_time,
                            follow_up_period=follow_up_time,
                            median_control=median_survival1,
                            dropout_rate=dropout_rate,
                            method="auto",  # Let auto-selection choose the primary method
                            accrual_pattern=accrual_pattern,
                            accrual_parameters=accrual_parameters if accrual_parameters else None,
                            compare_all=True  # This returns comparison data
                        )
                        
                        # Extract the comparison data
                        if "comparison" in sample_size_result:
                            result["comparison"] = sample_size_result["comparison"]
                            result["methods_results"] = sample_size_result.get("methods", {})
                            
                            # Use the recommended method's results as primary
                            recommended = sample_size_result["comparison"].get("recommended_method", "schoenfeld")
                            if recommended in sample_size_result.get("methods", {}):
                                sample_size_result = sample_size_result["methods"][recommended]
                            
                    else:
                        # Single method calculation
                        sample_size_result = analytical_survival.sample_size_survival_advanced(
                            hazard_ratio=hr_for_calc,
                            power=power,
                            alpha=alpha,
                            allocation_ratio=allocation_ratio,
                            sides=sides,
                            enrollment_period=accrual_time,
                            follow_up_period=follow_up_time,
                            median_control=median_survival1,
                            dropout_rate=dropout_rate,
                            method=advanced_method,
                            accrual_pattern=accrual_pattern,
                            accrual_parameters=accrual_parameters if accrual_parameters else None
                        )
                    
                    # Add method guidance information to result
                    result["method_used"] = sample_size_result.get("method_used", advanced_method)
                    result["method_guidance"] = sample_size_result.get("method_guidance", {})
                    
                except Exception as e:
                    # Fallback to legacy method if advanced fails
                    st.warning(f"Advanced method failed, falling back to legacy method: {e}")
                    sample_size_result = analytical_survival.sample_size_survival(
                        median1=median_survival1,
                        median2=median2_for_calc,
                        enrollment_period=accrual_time,
                        follow_up_period=follow_up_time,
                        dropout_rate=dropout_rate,
                        power=power,
                        alpha=alpha, 
                        allocation_ratio=allocation_ratio,
                        sides=sides 
                    )
        elif method == "simulation":
            if hypothesis_type == "Non-Inferiority":                   
                sample_size_result = simulation_survival.sample_size_survival_non_inferiority_sim(
                    median1=median_survival1,
                    non_inferiority_margin=non_inferiority_margin_hr_param,
                    assumed_hazard_ratio=assumed_true_hr_param, # median2 is derived from this internally in sim_ni
                    enrollment_period=accrual_time,
                    follow_up_period=follow_up_time,
                    dropout_rate=dropout_rate,
                    power=power,
                    alpha=alpha, # NI sim function handles one-sided alpha
                    allocation_ratio=allocation_ratio,
                    nsim=nsim,
                    min_n=min_n_sim,
                    max_n=max_n_sim,
                    step=step_n_sim, # Parameter name is 'step' in sim_ni function
                    seed=seed
                )
            else: # Superiority
                sample_size_result = simulation_survival.sample_size_survival_sim(
                    median1=median_survival1,
                    median2=median2_for_calc, 
                    enrollment_period=accrual_time,
                    follow_up_period=follow_up_time,
                    dropout_rate=dropout_rate, 
                    power=power,
                    alpha=alpha,
                    allocation_ratio=allocation_ratio,
                    nsim=nsim,
                    sides=sides, 
                    seed=seed
                )
        else:
            return {"error": f"Unsupported method: {method} for Sample Size calculation."}
            
        result["n1"] = round(sample_size_result.get("n_control", sample_size_result.get("n1", sample_size_result.get("sample_size_1", 0))))
        result["n2"] = round(sample_size_result.get("n_treatment", sample_size_result.get("n2", sample_size_result.get("sample_size_2", 0))))
        result["total_n"] = result["n1"] + result["n2"]
        result["events"] = round(sample_size_result.get("events_required", sample_size_result.get("events", sample_size_result.get("total_events", 0))))
        if 'power_curve_data' in sample_size_result:
            result['power_curve_data'] = sample_size_result['power_curve_data']
        
    elif calculation_type == "Power":
        n1 = params.get("n1", 50)
        n2 = params.get("n2", 50)
        result["n1_param"] = n1 
        result["n2_param"] = n2
        if method == "analytical":
            if hypothesis_type == "Non-Inferiority":
                # Use legacy non-inferiority function for now
                power_result_dict = analytical_survival.power_survival_non_inferiority(
                    n1=n1, n2=n2,
                    median1=median_survival1,
                    non_inferiority_margin=non_inferiority_margin_hr_param,
                    enrollment_period=accrual_time,
                    follow_up_period=follow_up_time,
                    dropout_rate=dropout_rate,
                    alpha=alpha, # NI function handles one-sided alpha internally
                    assumed_hazard_ratio=assumed_true_hr_param
                )
            else: # Superiority - use advanced methods
                try:
                    power_result_dict = analytical_survival.power_survival_advanced(
                        n_control=n1,
                        n_treatment=n2,
                        hazard_ratio=hr_for_calc,
                        alpha=alpha,
                        sides=sides,
                        enrollment_period=accrual_time,
                        follow_up_period=follow_up_time,
                        median_control=median_survival1,
                        dropout_rate=dropout_rate,
                        method=advanced_method
                    )
                    
                    # Add method guidance information to result
                    result["method_used"] = power_result_dict.get("method_used", advanced_method)
                    result["method_guidance"] = power_result_dict.get("method_guidance", {})
                    
                except Exception as e:
                    # Fallback to legacy method if advanced fails
                    st.warning(f"Advanced method failed, falling back to legacy method: {e}")
                    power_result_dict = analytical_survival.power_survival(
                        n1=n1, n2=n2,
                        median1=median_survival1,
                        median2=median2_for_calc,
                        enrollment_period=accrual_time,
                        follow_up_period=follow_up_time,
                        dropout_rate=dropout_rate,
                        alpha=alpha,
                        sides=sides
                    )
        elif method == "simulation":
            if hypothesis_type == "Non-Inferiority":
                power_result_dict = simulation_survival.power_survival_non_inferiority_sim(
                    n1=n1, n2=n2,
                    median1=median_survival1,
                    non_inferiority_margin=non_inferiority_margin_hr_param,
                    assumed_hazard_ratio=assumed_true_hr_param, # median2 derived internally
                    enrollment_period=accrual_time,
                    follow_up_period=follow_up_time,
                    dropout_rate=dropout_rate,
                    alpha=alpha, # NI sim function handles one-sided alpha
                    nsim=nsim,
                    seed=seed
                )
            else: # Superiority
                power_result_dict = simulation_survival.power_survival_sim(
                    n1=n1, n2=n2,
                    median1=median_survival1,
                    median2=median2_for_calc,
                    enrollment_period=accrual_time,
                    follow_up_period=follow_up_time,
                    dropout_rate=dropout_rate, 
                    alpha=alpha,
                    nsim=nsim,
                    sides=sides, 
                    seed=seed
                )
        else:
            return {"error": f"Unsupported method: {method} for Power calculation."}
        
        result["power"] = round(power_result_dict.get("power", 0), 3)
        result["events"] = round(power_result_dict.get("events", power_result_dict.get("total_events", power_result_dict.get("expected_events", 0))))
        if 'survival_curves' in power_result_dict:
            result['survival_curves'] = power_result_dict['survival_curves']

    elif calculation_type == "Minimum Detectable Effect":
        if hypothesis_type == "Non-Inferiority":
            result["mde_not_applicable"] = True 
            result["message"] = "Minimum Detectable Effect for Non-Inferiority is typically defined as the largest margin (NIM) for which non-inferiority can be claimed, or the true effect (HR) detectable against a fixed NIM. The calculation below provides the detectable HR for a superiority hypothesis. A dedicated NI MDE simulation is not yet implemented."
        
        n1 = params.get("n1", 50)
        n2 = params.get("n2", 50)
        result["n1_param"] = n1
        result["n2_param"] = n2
        
        if method == "analytical":
            mde_result_dict = analytical_survival.min_detectable_effect_survival(
                n1=n1, n2=n2,
                median1=median_survival1,
                enrollment_period=accrual_time,
                follow_up_period=follow_up_time,
                dropout_rate=dropout_rate,
                power=power,
                alpha=alpha,
                sides=sides 
            )
        elif method == "simulation":
            # For NI MDE, we'd ideally call a specific NI MDE simulation if it existed.
            # Since it doesn't, we fall back to superiority MDE simulation.
            if hypothesis_type == "Non-Inferiority":
                # Placeholder: A true NI MDE simulation would iterate on assumed_true_hr or NIM.
                # For now, we'll just use the superiority MDE sim and the message above explains this.
                pass # No specific NI MDE sim function to call yet.

            mde_result_dict = simulation_survival.min_detectable_effect_survival_sim(
                n1=n1, n2=n2,
                median1=median_survival1,
                enrollment_period=accrual_time,
                follow_up_period=follow_up_time,
                dropout_rate=dropout_rate,
                power=power,
                alpha=alpha,
                nsim=nsim,
                sides=sides, 
                seed=seed
            )
        else:
            return {"error": f"Unsupported method: {method} for MDE calculation."}

        detectable_hr = mde_result_dict.get("hr", None)
        result["mde"] = round(detectable_hr, 3) if detectable_hr is not None else None
        result["median_survival2_mde"] = round(median_survival1 / detectable_hr, 1) if detectable_hr and detectable_hr > 0 else None
        result["events"] = round(mde_result_dict.get("events", mde_result_dict.get("total_events", 0)))
        if 'power_vs_hr_data' in mde_result_dict:
             result['power_vs_hr_data'] = mde_result_dict['power_vs_hr_data']

    result['nsim'] = nsim if method == "Simulation" else None

    return result