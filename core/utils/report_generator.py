"""
Module for generating human-readable report text based on calculation results.

This module provides functions that format power calculation results into clear,
publication-ready text descriptions with appropriate references.
"""
import textwrap
from core.utils.ahern_report import generate_ahern_report
from core.utils.simons_report import generate_simons_report
from core.utils.cluster_reports import generate_cluster_report

# Dictionary of references for different analysis methods
METHOD_REFERENCES = {
    # Continuous outcomes
    "continuous_analytical": {
        "citation": "Cohen J. (1988). Statistical Power Analysis for the Behavioral Sciences. New York, NY: Routledge Academic",
        "doi": "https://doi.org/10.4324/9780203771587"
    },
    "continuous_simulation": {
        "citation": "Morris TP, White IR, Crowther MJ. (2019). Using simulation studies to evaluate statistical methods. Statistics in Medicine, 38(11), 2074-2102",
        "doi": "https://doi.org/10.1002/sim.8086"
    },
    "continuous_repeated_measures": {
        "citation": "Guo Y, Logan HL, Glueck DH, Muller KE. (2013). Selecting a sample size for studies with repeated measures. BMC Medical Research Methodology, 13(1), 100",
        "doi": "https://doi.org/10.1186/1471-2288-13-100"
    },
    # Survival outcomes
    "survival_exponential": {
        "citation": "Schoenfeld DA. (1983). Sample-size formula for the proportional-hazards regression model. Biometrics, 39(2), 499-503",
        "doi": "https://doi.org/10.2307/2531021"
    },
    # Binary outcomes
    "binary_normal_approximation": {
        "citation": "Fleiss JL, Levin B, Paik MC. (2003). Statistical Methods for Rates and Proportions. New York: John Wiley & Sons",
        "doi": "https://doi.org/10.1002/0471445428"
    },
    "binary_exact": {
        "citation": "Clopper CJ, Pearson ES. (1934). The use of confidence or fiducial limits illustrated in the case of the binomial. Biometrika, 26(4), 404-413",
        "doi": "https://doi.org/10.1093/biomet/26.4.404"
    },
    "binary_ahern": {
        "citation": "A'Hern RP. (2001). Sample size tables for exact single-stage phase II designs. Statistics in Medicine, 20(6), 859-866",
        "doi": "https://doi.org/10.1002/sim.721"
    },
    "binary_likelihood_ratio": {
        "citation": "Self SG, Mauritsen RH. (1988). Power/Sample Size Calculations for Generalized Linear Models. Biometrics, 44(1), 79-86",
        "doi": "https://doi.org/10.2307/2531896"
    },
    # Survival outcomes
    "survival_analytical": {
        "citation": "Schoenfeld DA. (1983). Sample-size formula for the proportional-hazards regression model. Biometrics, 39(2), 499-503",
        "doi": "https://doi.org/10.2307/2531021"
    },
    "survival_simulation": {
        "citation": "Hsieh FY, Lavori PW. (2000). Sample-Size Calculations for the Cox Proportional Hazards Regression Model with Nonbinary Covariates. Controlled Clinical Trials, 21(6), 552-560",
        "doi": "https://doi.org/10.1016/S0197-2456(00)00104-5"
    },
    # Cluster RCT designs
    "cluster_continuous_analytical": {
        "citation": "Donner A, Klar N. (2000). Design and Analysis of Cluster Randomization Trials in Health Research. London: Arnold",
        "doi": "https://doi.org/10.1002/sim.836"
    },
    "cluster_continuous_simulation": {
        "citation": "Burton A, Altman DG, Royston P, Holder RL. (2006). The design of simulation studies in medical statistics. Statistics in Medicine, 25(24), 4279-4292",
        "doi": "https://doi.org/10.1002/sim.2673"
    },
    "cluster_binary_analytical": {
        "citation": "Hayes RJ, Moulton LH. (2017). Cluster Randomised Trials. CRC Press",
        "doi": "https://doi.org/10.1201/9781315370286"
    },
    "cluster_binary_simulation": {
        "citation": "Hemming K, Girling AJ, Sitch AJ, Marsh J, Lilford RJ. (2011). Sample size calculations for cluster randomised controlled trials with a fixed number of clusters. BMC Medical Research Methodology, 11(1), 102",
        "doi": "https://doi.org/10.1186/1471-2288-11-102"
    }
}

def get_method_reference(outcome_type, test_type=None, method="analytical", design=None):
    """
    Get the appropriate reference for the specific calculation method.
    
    Parameters
    ----------
    outcome_type : str
        Type of outcome ('continuous', 'binary', or 'survival')
    test_type : str, optional
        Statistical test type (primarily for binary outcomes)
    method : str, optional
        Calculation method ('analytical' or 'simulation')
    design : str, optional
        Specific design type (e.g., 'repeated_measures')
        
    Returns
    -------
    dict
        Dictionary containing citation and DOI
    """
    # Default reference
    default_ref = {
        "citation": "Lachin JM. (1981). Introduction to sample size determination and power analysis for clinical trials. Controlled Clinical Trials, 2(2), 93-113",
        "doi": "https://doi.org/10.1016/0197-2456(81)90001-5"
    }
    
    if outcome_type == "continuous":
        if design == "repeated_measures":
            # For repeated measures designs
            return METHOD_REFERENCES.get("continuous_repeated_measures", default_ref)
        elif design == "cluster":
            # For cluster randomized trials with continuous outcomes
            key = f"cluster_{outcome_type}_{method}"
            return METHOD_REFERENCES.get(key, default_ref)
        else:
            # For standard continuous designs
            key = f"{outcome_type}_{method}"
            return METHOD_REFERENCES.get(key, default_ref)
    
    elif outcome_type == "binary":
        # Handle cluster design for binary outcomes
        if design == "cluster":
            # For cluster randomized trials with binary outcomes
            key = f"cluster_{outcome_type}_{method}"
            return METHOD_REFERENCES.get(key, default_ref)
            
        # Handle A'Hern design specifically
        elif test_type == "A'Hern":
            return METHOD_REFERENCES.get("binary_ahern", default_ref)
        
        # For other binary outcome tests
        elif test_type:
            # Normalize the test type string
            test_type_normalized = test_type.lower().replace(" ", "_")
            
            key = f"{outcome_type}_{test_type_normalized}"
            return METHOD_REFERENCES.get(key, default_ref)
    
    elif outcome_type == "survival":
        # For survival outcomes, default to exponential model
        return METHOD_REFERENCES.get("survival_exponential", default_ref)
    
    # Default to general analytical method
    return default_ref

def generate_sample_size_report(results, params, design_type, outcome_type):
    """
    Generate a human-readable report for sample size calculations.
    
    Parameters
    ----------
    results : dict
        Results from the calculation function
    params : dict
        Input parameters used for the calculation
    design_type : str
        Study design type (e.g., 'Parallel RCT', 'Single Arm Trial')
    outcome_type : str
        Type of outcome ('Continuous Outcome', 'Binary Outcome', 'Survival Outcome')
        
    Returns
    -------
    str
        Formatted text report
    """
    # Initialize report_text with a default value to prevent UnboundLocalError
    report_text = "No specific report template is available for this design and outcome combination."
    
    # Extract key values
    n1 = results.get('n1', 0)
    n2 = results.get('n2', 0)
    power = params.get('power', 0.8)
    alpha = params.get('alpha', 0.05)
    hypothesis_type = params.get('hypothesis_type', 'Superiority')
    method = params.get('method', 'analytical')
    
    # Extract outcome-specific parameters
    if 'Continuous' in outcome_type:
        # For continuous outcomes
        effect_size = results.get('effect_size', 0)
        mean1 = params.get('mean1', 0)
        mean2 = params.get('mean2', 0)
        std_dev = params.get('std_dev', 1)
        unequal_var = params.get('unequal_var', False)
        std_dev2 = params.get('std_dev2', std_dev) if unequal_var else std_dev
        
        # Check if this is a repeated measures design
        repeated_measures = params.get('repeated_measures', False)
        correlation = params.get('correlation', 0) if repeated_measures else 0
        
        # Get appropriate reference based on design
        design_param = "repeated_measures" if repeated_measures else None
        reference = get_method_reference('continuous', method=method, design=design_param)
        
        if design_type == 'Parallel RCT':
            if hypothesis_type == 'Superiority':
                # Create the variance assumption text based on whether unequal variance was used
                if unequal_var:
                    variance_text = f"assuming unequal variances with standard deviations of {std_dev:.2f} in group 1 and {std_dev2:.2f} in group 2"
                else:
                    variance_text = f"assuming equal variances with a standard deviation of {std_dev:.2f}"
                
                # Check if repeated measures design is being used
                repeated_measures = params.get("repeated_measures", False)
                
                # Check if simulation method was used
                if method == "simulation":
                    nsim = params.get("nsim", 1000)
                    seed = params.get("seed", 42)
                    if repeated_measures:
                        method_text = f"using Monte Carlo simulation ({nsim:,} simulations, random seed {seed}) with a paired t-test"
                    else:
                        method_text = f"using Monte Carlo simulation ({nsim:,} simulations, random seed {seed}) with a two-sided two-sample t-test"
                else:
                    if repeated_measures:
                        method_text = "using a paired t-test"
                    else:
                        method_text = "using a two-sided two-sample t-test"
                
                # Create special text for repeated measures if applicable
                if repeated_measures:
                    correlation = params.get("correlation", 0)
                    repeated_text = f"This is a repeated measures design with a correlation of {correlation:.2f} between measurements."
                else:
                    repeated_text = ""
                
                report_text = textwrap.dedent(f"""
                Sample Size Calculation Report:
                
                A sample size of {n1} participants in group 1 and {n2} participants in group 2 
                (total N = {n1 + n2}) will provide {power * 100:.0f}% power to detect a difference 
                in means of {abs(mean2 - mean1):.2f} (effect size d = {effect_size:.2f}) between 
                groups, {variance_text}, {method_text} with a Type I error rate of {alpha * 100:.0f}%.
                {repeated_text if repeated_text else ''}
                
                Reference: {reference['citation']}
                DOI: {reference['doi']}
                """)
            else:  # Non-inferiority
                nim = params.get('nim', 0)
                direction = params.get('direction', 'Higher is better')
                report_text = textwrap.dedent(f"""
                Non-Inferiority Sample Size Calculation Report:
                
                A sample size of {n1} participants in group 1 and {n2} participants in group 2 
                (total N = {n1 + n2}) will provide {power * 100:.0f}% power to establish 
                non-inferiority with a margin of {nim:.2f}, assuming a standard deviation 
                of {std_dev:.2f}{"" if not unequal_var else f" in group 1 and {std_dev2:.2f} in group 2"}, 
                using a one-sided t-test with a Type I error rate of {alpha * 100:.0f}%.
                
                Reference: {reference['citation']}
                DOI: {reference['doi']}
                """)
                
    elif 'Binary' in outcome_type:
        # For binary outcomes
        p1 = params.get('p1', 0)
        p2 = params.get('p2', 0)
        odds_ratio = results.get('odds_ratio', 0)
        test_type = params.get('test_type', 'Normal Approximation')
        correction = params.get('correction', False)
        
        # Get appropriate reference
        reference = get_method_reference('binary', test_type, method)
        
        if design_type == 'Parallel RCT':
            if hypothesis_type == 'Superiority':
                # Check if simulation method was used and create appropriate method text
                if method == "simulation":
                    nsim = params.get("nsim", 1000)
                    seed = params.get("seed", 42)
                    method_text = f"using Monte Carlo simulation ({nsim:,} simulations, random seed {seed}) with a {test_type}{' with continuity correction' if correction else ''}"
                else:
                    method_text = f"using a {test_type}{' with continuity correction' if correction else ''}"
                
                report_text = textwrap.dedent(f"""
                Sample Size Calculation Report:
                
                A sample size of {n1} participants in group 1 and {n2} participants in group 2 
                (total N = {n1 + n2}) will provide {power * 100:.0f}% power to detect a difference 
                in proportions from {p1:.2f} in group 1 to {p2:.2f} in group 2 (odds ratio = {odds_ratio:.2f}), 
                {method_text}, with a Type I error rate of {alpha * 100:.0f}%.
                
                Reference: {reference['citation']}
                DOI: {reference['doi']}
                """)
            else:  # Non-inferiority
                nim = params.get('nim', 0)
                report_text = textwrap.dedent(f"""
                Non-Inferiority Sample Size Calculation Report:
                
                A sample size of {n1} participants in group 1 and {n2} participants in group 2 
                (total N = {n1 + n2}) will provide {power * 100:.0f}% power to establish 
                non-inferiority with a margin of {nim:.2f}, assuming a proportion of {p1:.2f} in 
                the reference group, using a {test_type}{' with continuity correction' if correction else ''}, 
                with a Type I error rate of {alpha * 100:.0f}%.
                
                Reference: {reference['citation']}
                DOI: {reference['doi']}
                """)
                
    elif 'Survival' in outcome_type:
        # For survival outcomes
        hr = params.get('hr', 0)
        median_survival1 = params.get('median_survival1', 0)
        
        # Get appropriate reference
        reference = get_method_reference('survival', method=method)
        
        if design_type == 'Parallel RCT':
            if hypothesis_type == 'Superiority':
                # Extract accrual and follow-up times
                accrual_time = params.get('accrual_time', 1.0)
                follow_up_time = params.get('follow_up_time', 1.0)
                dropout_rate1 = params.get('dropout_rate1', 0.1)
                
                report_text = textwrap.dedent(f"""
                Sample Size Calculation Report:
                
                A sample size of {n1} participants in group 1 and {n2} participants in group 2 
                (total N = {n1 + n2}) will provide {power * 100:.0f}% power to detect a hazard ratio 
                of {hr:.2f}, assuming a median survival time of {median_survival1:.1f} months in the 
                reference group. The calculation assumes exponential survival distributions with an 
                accrual period of {accrual_time:.1f} months, follow-up period of {follow_up_time:.1f} months, 
                and anticipated dropout rate of {dropout_rate1*100:.1f}%. Analysis will use a log-rank test 
                with a Type I error rate of {alpha * 100:.0f}%.
                
                Reference: {reference['citation']}
                DOI: {reference['doi']}
                """)
            else:  # Non-inferiority
                nim = params.get('nim', 0)
                report_text = textwrap.dedent(f"""
                Non-Inferiority Sample Size Calculation Report:
                
                A sample size of {n1} participants in group 1 and {n2} participants in group 2 
                (total N = {n1 + n2}) will provide {power * 100:.0f}% power to establish 
                non-inferiority with a hazard ratio margin of {nim:.2f}, assuming a median 
                survival time of {median_survival1:.1f} months in the reference group, 
                using a one-sided log-rank test with a Type I error rate of {alpha * 100:.0f}%.
                
                Reference: {reference['citation']}
                DOI: {reference['doi']}
                """)
    else:
        # Default report for other types
        report_text = textwrap.dedent(f"""
        Sample Size Calculation Report:
        
        A total sample size of {n1 + n2} participants will provide {power * 100:.0f}% power 
        for the specified design with a Type I error rate of {alpha * 100:.0f}%.
        """)
    
    return report_text.strip()

def generate_power_report(results, params, design_type, outcome_type):
    """
    Generate a human-readable report for power calculations.
    
    Parameters
    ----------
    results : dict
        Results from the calculation function
    params : dict
        Input parameters used for the calculation
    design_type : str
        Study design type (e.g., 'Parallel RCT', 'Single Arm Trial')
    outcome_type : str
        Type of outcome ('Continuous Outcome', 'Binary Outcome', 'Survival Outcome')
        
    Returns
    -------
    str
        Formatted text report
    """
    # Initialize report_text with a default value to prevent UnboundLocalError
    report_text = "No specific report template is available for this design and outcome combination."
    
    # Extract key values
    n1 = params.get('n1', 0)
    n2 = params.get('n2', 0)
    power = results.get('power', 0)
    alpha = params.get('alpha', 0.05)
    hypothesis_type = params.get('hypothesis_type', 'Superiority')
    method = params.get('method', 'analytical')
    
    if design_type == 'Parallel RCT' and 'Continuous' in outcome_type:
        mean1 = params.get('mean1', 0)
        mean2 = params.get('mean2', 0)
        sd1 = params.get('sd1', 1)
        sd2 = params.get('sd2', params.get('sd1', 1)) # Use sd1 if sd2 is not present
        actual_effect_size = results.get('effect_size') # From power_continuous results
        difference = abs(mean1 - mean2)

        # Determine if sd2 was explicitly provided and different from sd1 for reporting
        sd_text = f"{sd1:.2f}" 
        if params.get('sd2') is not None and sd1 != sd2:
            sd_text += f" in group 1 and {sd2:.2f} in group 2"
        else:
            sd_text += " (pooled or per group)"

        # Handle None effect size gracefully
        if actual_effect_size is not None:
            effect_size_text = f"The standardized effect size (Cohen's d) for this scenario is {actual_effect_size:.2f}."
        else:
            # Calculate effect size manually if not provided
            std_dev = params.get('std_dev', sd1)  # Use std_dev if available, fallback to sd1
            if std_dev and std_dev != 0:
                calculated_effect_size = difference / std_dev
                effect_size_text = f"The standardized effect size (Cohen's d) for this scenario is {calculated_effect_size:.2f}."
            else:
                effect_size_text = "Effect size calculation not available."

        report_text = textwrap.dedent(f"""
        Power Calculation Report (Parallel RCT - Continuous Outcome):

        For a study designed to compare two parallel groups with {n1} participants in group 1 and {n2} in group 2 (total {n1+n2}), targeting a difference between means of {difference:.2f} (group 1 mean: {mean1:.2f}, group 2 mean: {mean2:.2f}), and assuming standard deviation(s) of {sd_text}, the estimated statistical power is {power * 100:.1f}%. This calculation uses a Type I error rate (alpha) of {alpha*100:.0f}%. {effect_size_text}
        """)
        # Add reference
        ref_details = get_method_reference('continuous', method=method, design=None) # Standard parallel design
        report_text += f"\n\nMethod Reference: {ref_details['citation']} ({ref_details['doi']})"
        
    elif design_type == 'Parallel RCT' and 'Binary' in outcome_type:
        # Enhanced binary outcome power reporting
        p1 = params.get('p1', 0)
        p2 = params.get('p2', 0)
        test_type = params.get('test_type', 'Normal Approximation')
        correction = params.get('correction', False)
        odds_ratio = results.get('odds_ratio')
        relative_risk = results.get('relative_risk') 
        absolute_difference = abs(p2 - p1)
        
        # Calculate odds ratio and relative risk if not in results
        if odds_ratio is None and p1 > 0 and p2 > 0 and p1 < 1 and p2 < 1:
            odds_ratio = (p2 / (1 - p2)) / (p1 / (1 - p1))
        if relative_risk is None and p1 > 0:
            relative_risk = p2 / p1
        
        # Format odds ratio and relative risk safely
        or_text = f" (odds ratio = {odds_ratio:.2f})" if odds_ratio is not None else ""
        rr_text = f" (relative risk = {relative_risk:.2f})" if relative_risk is not None else ""
        
        # Create method text based on simulation vs analytical
        if method == "simulation":
            nsim = params.get("nsim", 1000)
            seed = params.get("seed")
            seed_text = f", random seed {seed}" if seed is not None else ""
            method_text = f"using Monte Carlo simulation ({nsim:,} simulations{seed_text}) with {test_type}"
        else:
            method_text = f"using {test_type}"
        
        if correction:
            method_text += " with continuity correction"
        
        report_text = textwrap.dedent(f"""
        Power Calculation Report (Parallel RCT - Binary Outcome):

        For a study designed to compare two parallel groups with {n1} participants in group 1 and {n2} in group 2 (total {n1+n2}), targeting a difference in proportions from {p1:.3f} ({p1*100:.1f}%) in group 1 to {p2:.3f} ({p2*100:.1f}%) in group 2 (absolute difference = {absolute_difference:.3f}{or_text}{rr_text}), the estimated statistical power is {power * 100:.1f}%. This calculation uses {method_text} with a Type I error rate (alpha) of {alpha*100:.0f}%.
        """)
        
        # Add reference
        ref_details = get_method_reference('binary', test_type, method)
        report_text += f"\n\nMethod Reference: {ref_details['citation']} ({ref_details['doi']})"
        
    else:
        # Default report for other types
        report_text = textwrap.dedent(f"""
        Power Calculation Report:
        
        With a total sample size of {n1 + n2} participants, the study will have {power * 100:.1f}% power 
        for the specified design with a Type I error rate of {alpha * 100:.0f}%.
        """)
    
    return report_text.strip()

def generate_mde_report(results, params, design_type, outcome_type):
    """
    Generate a human-readable report for minimum detectable effect calculations.
    
    Parameters
    ----------
    results : dict
        Results from the calculation function
    params : dict
        Input parameters used for the calculation
    design_type : str
        Study design type (e.g., 'Parallel RCT', 'Single Arm Trial')
    outcome_type : str
        Type of outcome ('Continuous Outcome', 'Binary Outcome', 'Survival Outcome')
        
    Returns
    -------
    str
        Formatted text report
    """
    # Initialize report_text with a default value to prevent UnboundLocalError
    report_text = "No specific report template is available for this design and outcome combination."
    
    # Extract key values
    n1 = params.get('n1', 0)
    n2 = params.get('n2', 0)
    power = params.get('power', 0.8)
    alpha = params.get('alpha', 0.05)
    hypothesis_type = params.get('hypothesis_type', 'Superiority')
    method = params.get('method', 'analytical')
    
    if design_type == 'Parallel RCT' and 'Continuous' in outcome_type:
        mde_val = results.get('mde')
        cohen_d_val = results.get('cohen_d')
        sd1 = params.get('sd1', 1)
        sd2 = params.get('sd2', params.get('sd1', 1))

        # Determine if sd2 was explicitly provided and different from sd1 for reporting
        sd_text = f"{sd1:.2f}"
        if params.get('sd2') is not None and sd1 != sd2:
            sd_text += f" in group 1 and {sd2:.2f} in group 2"
        else:
            sd_text += " (pooled or per group)"
        
        # Handle None values gracefully
        if mde_val is not None and cohen_d_val is not None:
            effect_text = f"the minimum detectable difference in means is {mde_val:.2f}. This corresponds to a standardized effect size (Cohen's d) of {cohen_d_val:.2f}."
        elif mde_val is not None:
            effect_text = f"the minimum detectable difference in means is {mde_val:.2f}."
        else:
            effect_text = "the minimum detectable effect calculation is not available."
        
        report_text = textwrap.dedent(f"""
        Minimum Detectable Effect Report (Parallel RCT - Continuous Outcome):

        For a study with {n1} participants in group 1 and {n2} in group 2 (total {n1+n2}), aiming for {power * 100:.0f}% statistical power, and assuming standard deviation(s) of {sd_text}, {effect_text} This calculation uses a Type I error rate (alpha) of {alpha*100:.0f}%.
        """)
        # Add reference
        ref_details = get_method_reference('continuous', method=method, design=None) # Standard parallel design
        report_text += f"\n\nMethod Reference: {ref_details['citation']} ({ref_details['doi']})"
    else:
        # Default report for other types
        report_text = textwrap.dedent(f"""
        Minimum Detectable Effect Report:
        
        With a total sample size of {n1 + n2} participants and {power * 100:.0f}% power, 
        the study can detect the specified minimum effect with a Type I error rate of {alpha * 100:.0f}%.
        """)
    
    return report_text.strip()

def generate_report(results, params, design_type, outcome_type):
    """
    Generate appropriate report based on calculation type.
    
    Parameters
    ----------
    results : dict
        Results from the calculation function
    params : dict
        Input parameters used for the calculation
    design_type : str
        Study design type (e.g., 'Parallel RCT', 'Single Arm Trial')
    outcome_type : str
        Type of outcome ('Continuous Outcome', 'Binary Outcome', 'Survival Outcome')
        
    Returns
    -------
    str
        Formatted text report
    """
    # Check if this is a Cluster RCT design
    if design_type == 'Cluster RCT':
        return generate_cluster_report(results, params)
        
    # Check if this is an A'Hern design for Single Arm Trial with Binary Outcome
    if (design_type == 'Single Arm Trial' and 'Binary' in outcome_type and 
            params.get('design_method', 'Standard') == "A'Hern"):
        return generate_ahern_report(results, params)
        
    # Check if this is a Simon's two-stage design for Single Arm Trial with Binary Outcome
    if (design_type == 'Single Arm Trial' and 'Binary' in outcome_type and 
            params.get('design_method', 'Standard') == "Simon's Two-Stage"):
        return generate_simons_report(results, params)
    
    # Determine the calculation type for standard reports
    # Check for both parameter names since some components use calc_type and others use calculation_type
    calculation_type = params.get("calculation_type", params.get("calc_type", "Sample Size"))
    
    # Generate appropriate report based on calculation type
    if calculation_type == "Sample Size":
        return generate_sample_size_report(results, params, design_type, outcome_type)
    elif calculation_type == "Power":
        return generate_power_report(results, params, design_type, outcome_type)
    elif calculation_type == "Minimum Detectable Effect":
        return generate_mde_report(results, params, design_type, outcome_type)
    else:
        return "No report available for this calculation type."
