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
    # Survival outcomes - Advanced Methods
    "survival_schoenfeld": {
        "citation": "Schoenfeld DA. (1983). Sample-size formula for the proportional-hazards regression model. Biometrics, 39(2), 499-503",
        "doi": "https://doi.org/10.2307/2531021"
    },
    "survival_freedman": {
        "citation": "Freedman LS. (1982). Tables of the number of patients required in clinical trials using the logrank test. Statistics in Medicine, 1(2), 121-129",
        "doi": "https://doi.org/10.1002/sim.4780010204"
    },
    "survival_lakatos": {
        "citation": "Lakatos E. (1988). Sample sizes based on the log-rank statistic in complex clinical trials. Biometrics, 44(1), 229-241",
        "doi": "https://doi.org/10.2307/2531910"
    },
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
    },
    # Stepped wedge designs
    "stepped_wedge_continuous": {
        "citation": "Hemming K, Haines TP, Chilton PJ, Girling AJ, Lilford RJ. (2015). The stepped wedge cluster randomised trial: rationale, design and analysis. BMJ, 350, h391",
        "doi": "https://doi.org/10.1136/bmj.h391"
    },
    "stepped_wedge_binary": {
        "citation": "Hemming K, Haines TP, Chilton PJ, Girling AJ, Lilford RJ. (2015). The stepped wedge cluster randomised trial: rationale, design and analysis. BMJ, 350, h391",
        "doi": "https://doi.org/10.1136/bmj.h391"
    },
    "stepped_wedge_simulation": {
        "citation": "Baio G, Copas A, Ambler G, Hargreaves J, Beard E, Omar RZ. (2015). Sample size calculation for a stepped wedge trial. Trials, 16(1), 354",
        "doi": "https://doi.org/10.1186/s13063-015-0840-9"
    },
    # Interrupted time series designs
    "interrupted_time_series_continuous": {
        "citation": "Wagner AK, Soumerai SB, Zhang F, Ross-Degnan D. (2002). Segmented regression analysis of interrupted time series studies in medication use research. Journal of Clinical Pharmacy and Therapeutics, 27(4), 299-309",
        "doi": "https://doi.org/10.1046/j.1365-2710.2002.00430.x"
    },
    "interrupted_time_series_binary": {
        "citation": "Bernal JL, Cummins S, Gasparrini A. (2017). Interrupted time series regression for the evaluation of public health interventions: a tutorial. International Journal of Epidemiology, 46(1), 348-355",
        "doi": "https://doi.org/10.1093/ije/dyw098"
    },
    "interrupted_time_series_simulation": {
        "citation": "Zhang F, Wagner AK, Soumerai SB, Ross-Degnan D. (2009). Methods for estimating confidence intervals in interrupted time series analyses of health interventions. Journal of Clinical Epidemiology, 62(2), 143-148",
        "doi": "https://doi.org/10.1016/j.jclinepi.2008.08.007"
    }
}

def get_method_reference(outcome_type, test_type=None, method="analytical", design=None, advanced_method=None):
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
    advanced_method : str, optional
        Advanced method for survival outcomes ('schoenfeld', 'freedman', 'lakatos')
        
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
        # For survival outcomes, use advanced method if specified
        if advanced_method and advanced_method in ["schoenfeld", "freedman", "lakatos"]:
            key = f"survival_{advanced_method}"
            return METHOD_REFERENCES.get(key, default_ref)
        else:
            # Default to Schoenfeld method
            return METHOD_REFERENCES.get("survival_schoenfeld", default_ref)
    
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
        
        # Get appropriate reference based on advanced method
        advanced_method = params.get('advanced_method', 'schoenfeld')
        reference = get_method_reference('survival', method=method, advanced_method=advanced_method)
        
        if design_type == 'Parallel RCT':
            if hypothesis_type == 'Superiority':
                # Extract accrual and follow-up times
                accrual_time = params.get('accrual_time', 1.0)
                follow_up_time = params.get('follow_up_time', 1.0)
                dropout_rate = params.get('dropout_rate', 0.1)
                
                # Advanced method information
                method_used = results.get('method_used', advanced_method)
                method_guidance = results.get('method_guidance', {})
                
                # Create method description
                if method_used == 'schoenfeld':
                    method_desc = "Schoenfeld (1983) method - the standard approach for log-rank test sample size calculations"
                elif method_used == 'freedman':
                    method_desc = "Freedman (1982) method - alternative approach with different censoring assumptions"
                elif method_used == 'lakatos':
                    method_desc = "Lakatos (1988) method - advanced approach accounting for complex accrual patterns"
                else:
                    method_desc = "standard log-rank test methodology"
                
                # Add accrual pattern information
                current_accrual_pattern = params.get('accrual_pattern', 'uniform')
                if current_accrual_pattern != 'uniform' and method_used == 'lakatos':
                    accrual_desc = f" with {current_accrual_pattern.replace('_', ' ')} accrual pattern"
                else:
                    accrual_desc = " with uniform accrual"
                
                # Method comparison information
                comparison_text = ""
                if results.get('comparison'):
                    comparison = results['comparison']
                    sample_sizes = comparison.get('sample_sizes', {})
                    max_diff = comparison.get('max_percent_difference', 0)
                    if len(sample_sizes) > 1:
                        sizes_text = ", ".join([f"{k.title()}: {v}" for k, v in sample_sizes.items()])
                        comparison_text = f"\n\nMethod Comparison: {sizes_text} (max difference: {max_diff:.1f}%). "
                        if max_diff < 5:
                            comparison_text += "Methods show excellent agreement."
                        elif max_diff < 15:
                            comparison_text += "Methods show moderate agreement."
                        else:
                            comparison_text += "Methods show substantial differences; consider study design complexity."

                report_text = textwrap.dedent(f"""
                Advanced Survival Analysis - Sample Size Calculation Report:
                
                A sample size of {n1} participants in group 1 and {n2} participants in group 2 
                (total N = {n1 + n2}) will provide {power * 100:.0f}% power to detect a hazard ratio 
                of {hr:.2f}, assuming a median survival time of {median_survival1:.1f} months in the 
                reference group. The calculation uses the {method_desc}{accrual_desc}. 
                
                Study Design: Exponential survival distributions with accrual period of {accrual_time:.1f} months, 
                follow-up period of {follow_up_time:.1f} months, and anticipated dropout rate of {dropout_rate*100:.1f}%. 
                Analysis will use a log-rank test with a Type I error rate of {alpha * 100:.0f}%.{comparison_text}
                
                Primary Reference: {reference['citation']}
                DOI: {reference['doi']}
                """)
            else:  # Non-inferiority
                nim = params.get('non_inferiority_margin_hr', params.get('nim', 0))
                assumed_hr = params.get('assumed_true_hr', 1.0)
                accrual_time = params.get('accrual_time', 1.0)
                follow_up_time = params.get('follow_up_time', 1.0)
                dropout_rate = params.get('dropout_rate', 0.1)
                
                report_text = textwrap.dedent(f"""
                Non-Inferiority Survival Analysis - Sample Size Calculation Report:
                
                A sample size of {n1} participants in group 1 and {n2} participants in group 2 
                (total N = {n1 + n2}) will provide {power * 100:.0f}% power to establish 
                non-inferiority with a hazard ratio margin of {nim:.2f}, assuming a median 
                survival time of {median_survival1:.1f} months in the reference group and 
                a true hazard ratio of {assumed_hr:.2f}. 
                
                Study Design: Exponential survival distributions with accrual period of {accrual_time:.1f} months, 
                follow-up period of {follow_up_time:.1f} months, and anticipated dropout rate of {dropout_rate*100:.1f}%. 
                Analysis will use a one-sided log-rank test with a Type I error rate of {alpha * 100:.0f}%.
                
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
        
    elif design_type == 'Parallel RCT' and 'Survival' in outcome_type:
        # Enhanced survival outcome power reporting
        hr = params.get('hr', 0.7)
        median_survival1 = params.get('median_survival1', 12.0)
        advanced_method = params.get('advanced_method', 'schoenfeld')
        accrual_time = params.get('accrual_time', 12.0)
        follow_up_time = params.get('follow_up_time', 24.0)
        dropout_rate = params.get('dropout_rate', 0.1)
        
        # Method information
        method_used = results.get('method_used', advanced_method)
        events = results.get('events', 0)
        
        # Get median survival for treatment group
        median_survival2 = median_survival1 / hr if hr > 0 else float('inf')
        
        report_text = textwrap.dedent(f"""
        Advanced Survival Analysis - Power Calculation Report:

        For a study designed to compare two parallel groups with {n1} participants in group 1 and {n2} in group 2 (total {n1+n2}), targeting a hazard ratio of {hr:.2f} (median survival: control = {median_survival1:.1f} months, treatment = {median_survival2:.1f} months), the estimated statistical power is {power * 100:.1f}%. 
        
        Study Design: The calculation uses the {method_used.title()} method with exponential survival distributions, accrual period of {accrual_time:.1f} months, follow-up period of {follow_up_time:.1f} months, and anticipated dropout rate of {dropout_rate*100:.1f}%. Expected number of events: {events:.0f}. Analysis will use a log-rank test with a Type I error rate of {alpha*100:.0f}%.
        """)
        
        # Add method comparison if available
        if results.get('comparison'):
            comparison = results['comparison']
            max_diff = comparison.get('max_percent_difference', 0)
            if max_diff > 0:
                report_text += f"\n\nMethod Validation: Cross-method comparison shows {max_diff:.1f}% maximum difference, "
                if max_diff < 5:
                    report_text += "indicating excellent methodological agreement."
                else:
                    report_text += "suggesting methodological sensitivity to study design assumptions."
        
        # Add reference
        ref_details = get_method_reference('survival', method=method, advanced_method=advanced_method)
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
    elif design_type == 'Parallel RCT' and 'Survival' in outcome_type:
        # Enhanced survival MDE reporting
        mde_hr = results.get('mde')
        median_survival1 = params.get('median_survival1', 12.0)
        advanced_method = params.get('advanced_method', 'schoenfeld')
        events = results.get('events', 0)
        
        if mde_hr:
            median_survival2_mde = median_survival1 / mde_hr
            
            report_text = textwrap.dedent(f"""
            Survival Analysis - Minimum Detectable Effect Report:

            For a study with {n1} participants in group 1 and {n2} in group 2 (total {n1+n2}), aiming for {power * 100:.0f}% statistical power, the minimum detectable hazard ratio is {mde_hr:.3f}. This corresponds to detecting a difference between median survival times of {median_survival1:.1f} months (control) vs {median_survival2_mde:.1f} months (treatment). Expected number of events: {events:.0f}.
            
            The calculation uses the {advanced_method.title()} method with a Type I error rate of {alpha*100:.0f}%.
            """)
            
            # Add reference
            ref_details = get_method_reference('survival', method=method, advanced_method=advanced_method)
            report_text += f"\n\nMethod Reference: {ref_details['citation']} ({ref_details['doi']})"
        else:
            report_text = "Minimum detectable effect calculation for survival outcomes is not available."
    else:
        # Default report for other types
        report_text = textwrap.dedent(f"""
        Minimum Detectable Effect Report:
        
        With a total sample size of {n1 + n2} participants and {power * 100:.0f}% power, 
        the study can detect the specified minimum effect with a Type I error rate of {alpha * 100:.0f}%.
        """)
    
    return report_text.strip()

def generate_stepped_wedge_report(results, params):
    """
    Generate a comprehensive report for stepped wedge cluster randomized trials.
    
    Parameters
    ----------
    results : dict
        Results from the stepped wedge calculation
    params : dict
        Input parameters used for the calculation
        
    Returns
    -------
    str
        Formatted HTML report with design overview and references
    """
    # Extract parameters
    clusters = params.get('clusters', 0)
    steps = params.get('steps', 0) 
    individuals_per_cluster = params.get('individuals_per_cluster', 0)
    icc = params.get('icc', 0)
    power = results.get('power', 0)
    alpha = params.get('alpha', 0.05)
    method = params.get('method', 'Simulation')
    nsim = params.get('nsim', 1000)
    outcome_type = results.get('outcome_type', 'Unknown')
    
    # Calculate design metrics
    total_n = clusters * steps * individuals_per_cluster
    total_cluster_periods = clusters * steps
    control_periods = clusters  # Only baseline step
    intervention_periods = clusters * (steps - 1)
    design_efficiency = intervention_periods / total_cluster_periods
    
    # Get appropriate reference based on method used
    method_used = params.get('method', 'Simulation')
    cluster_autocorr = params.get('cluster_autocorr', 0)
    
    if method_used == "Hussey & Hughes Analytical":
        # Use Hussey & Hughes as primary reference for analytical method
        reference = {
            "citation": "Hussey MA, Hughes JP. (2007). Design and analysis of stepped wedge cluster randomized trials. Contemporary Clinical Trials, 28(2), 182-191",
            "doi": "https://doi.org/10.1016/S1551-7144(06)00118-8"
        }
    elif 'continuous' in outcome_type.lower():
        reference = METHOD_REFERENCES.get("stepped_wedge_continuous")
    elif 'binary' in outcome_type.lower():
        reference = METHOD_REFERENCES.get("stepped_wedge_binary")
    else:
        reference = METHOD_REFERENCES.get("stepped_wedge_continuous")  # Default
    
    # Outcome-specific parameters
    if 'continuous' in outcome_type.lower():
        treatment_effect = params.get('treatment_effect', 0)
        std_dev = params.get('std_dev', 1)
        effect_text = f"treatment effect of {treatment_effect:.2f} units with standard deviation {std_dev:.2f}"
    else:  # Binary
        p_control = params.get('p_control', 0)
        p_intervention = params.get('p_intervention', 0) 
        risk_diff = p_intervention - p_control if p_intervention and p_control else 0
        effect_text = f"change in proportion from {p_control:.2f} to {p_intervention:.2f} (risk difference = {risk_diff:.3f})"
    
    # Design effect calculation
    design_effect = 1 + (individuals_per_cluster - 1) * icc
    
    report_html = f"""
    <div style="font-family: Arial, sans-serif; line-height: 1.6; max-width: 800px;">
    
    <h2 style="color: #2E86AB; border-bottom: 2px solid #2E86AB; padding-bottom: 10px;">
    📊 Stepped Wedge Cluster Randomized Trial - Power Analysis Report
    </h2>
    
    <h3 style="color: #495057;">⚡ Power Analysis Results</h3>
    <div style="background-color: #e8f4fd; padding: 15px; border-radius: 8px; border-left: 4px solid #2E86AB;">
    <p><strong>Statistical Power:</strong> <span style="font-size: 1.2em; color: #2E86AB; font-weight: bold;">{power:.1%}</span></p>
    <p><strong>Significance Level (α):</strong> {alpha:.3f}</p>
    <p><strong>Expected Effect:</strong> {effect_text}</p>
    <p><strong>Analysis Method:</strong> {method} {'(' + str(nsim) + ' simulations)' if method.lower() == 'simulation' else ''}</p>
    </div>
    
    <div style="background-color: #e6f3ff; padding: 20px; border-radius: 8px; border-left: 4px solid #0066cc; margin: 20px 0;">
    <h4 style="color: #0052a3; margin-top: 0; margin-bottom: 15px;">📝 Methodological Description</h4>
    <div style="background-color: white; padding: 15px; border-radius: 6px; border: 1px solid #cce7ff;">
    <p style="font-style: italic; line-height: 1.8; margin: 0; color: #333;">
    A sample size of {individuals_per_cluster} individuals per cluster per time period, from each of {clusters} clusters, 
    stepping every time period across {steps} time steps (including baseline), will give the study {power:.0%} power to 
    detect a {'treatment effect of ' + str(treatment_effect) + ' units' if 'continuous' in outcome_type.lower() else 'change from ' + str(p_control) + ' to ' + str(p_intervention) + ' in the proportion'}, 
    assuming {'a standard deviation of ' + str(std_dev) if 'continuous' in outcome_type.lower() else 'the stated proportions'}, an intracluster correlation coefficient (ICC) of {icc:.3f}{', and a cluster autocorrelation coefficient (CAC) of ' + str(cluster_autocorr) if cluster_autocorr > 0 else ''}, 
    with a Type I error rate of {alpha*100:.0f}%. Power calculations were performed using {'analytical methods' if method_used == 'Hussey & Hughes Analytical' else 'simulation-based methods'} as 
    described by {reference['citation'].split('.')[0] + ' et al.'}.
    </p>
    </div>
    <p style="font-size: 0.9em; color: #666; margin-top: 10px; margin-bottom: 0;">
    <strong>Tip:</strong> Copy the text above directly into your grant application or study protocol.
    </p>
    </div>
    
    <h3 style="color: #495057;">🔧 Design Efficiency Metrics</h3>
    <table style="width: 100%; border-collapse: collapse; margin: 15px 0;">
    <tr style="background-color: #f8f9fa;">
        <td style="padding: 8px; border: 1px solid #dee2e6; font-weight: bold;">Total Cluster-Periods</td>
        <td style="padding: 8px; border: 1px solid #dee2e6;">{total_cluster_periods}</td>
    </tr>
    <tr>
        <td style="padding: 8px; border: 1px solid #dee2e6; font-weight: bold;">Control Periods</td>
        <td style="padding: 8px; border: 1px solid #dee2e6;">{control_periods} ({control_periods/total_cluster_periods:.1%})</td>
    </tr>
    <tr style="background-color: #f8f9fa;">
        <td style="padding: 8px; border: 1px solid #dee2e6; font-weight: bold;">Intervention Periods</td>
        <td style="padding: 8px; border: 1px solid #dee2e6;">{intervention_periods} ({intervention_periods/total_cluster_periods:.1%})</td>
    </tr>
    <tr>
        <td style="padding: 8px; border: 1px solid #dee2e6; font-weight: bold;">Design Efficiency</td>
        <td style="padding: 8px; border: 1px solid #dee2e6;"><strong>{design_efficiency:.1%}</strong></td>
    </tr>
    </table>
    
    
    <h3 style="color: #495057;">📚 Methodological Reference</h3>
    <div style="background-color: #f8f9fa; padding: 15px; border-radius: 8px;">
    <p><strong>Primary Reference:</strong></p>
    <p style="font-style: italic; margin: 10px 0;">{reference['citation']}</p>
    <p><strong>DOI:</strong> <a href="{reference['doi']}" target="_blank" style="color: #2E86AB;">{reference['doi']}</a></p>
    
    <p style="margin-top: 15px;"><strong>Additional Key References:</strong></p>
    <ul style="margin-top: 10px;">
    <li>Hussey MA, Hughes JP. (2007). Design and analysis of stepped wedge cluster randomized trials. Contemporary Clinical Trials, 28(2), 182-191.</li>
    <li>Mdege ND, Man MS, Taylor CA, Torgerson DJ. (2011). Systematic review of stepped wedge cluster randomized trials shows that design is particularly used to evaluate interventions during routine implementation. Journal of Clinical Epidemiology, 64(9), 936-948.</li>
    <li>Copas AJ, Lewis JJ, Thompson JA, Davey C, Baio G, Hargreaves JR. (2015). Designing a stepped wedge trial: three main designs, carry-over effects and randomisation approaches. Trials, 16(1), 352.</li>
    </ul>
    </div>
    
    <div style="background-color: #d1ecf1; padding: 15px; border-radius: 8px; border-left: 4px solid #17a2b8; margin: 15px 0;">
    <h4 style="color: #0c5460; margin-top: 0;">⚠️ Important Considerations</h4>
    <ul style="color: #0c5460; margin-bottom: 0;">
    <li><strong>Temporal trends:</strong> Ensure appropriate modeling of time trends in analysis</li>
    <li><strong>Carry-over effects:</strong> Consider potential lasting effects of the intervention</li>
    <li><strong>ICC estimation:</strong> Use pilot data or literature to estimate ICC accurately</li>
    <li><strong>Randomization:</strong> Cluster sequence should be randomized to intervention steps</li>
    </ul>
    </div>
    
    <hr style="margin: 20px 0; border: none; border-top: 1px solid #dee2e6;">
    <p style="font-size: 0.9em; color: #6c757d; text-align: center; margin: 10px 0;">
    Report generated by DesignPower • Stepped Wedge Cluster RCT Module
    </p>
    </div>
    """
    
    return report_html

def generate_interrupted_time_series_report(results, params):
    """
    Generate a comprehensive report for interrupted time series designs.
    
    Parameters
    ----------
    results : dict
        Results from the ITS calculation
    params : dict
        Input parameters used for the calculation
        
    Returns
    -------
    str
        Formatted HTML report with design overview and references
    """
    # Extract parameters
    n_pre = params.get('n_pre', results.get('n_pre', 0))
    n_post = params.get('n_post', results.get('n_post', 0))
    total_n = results.get('total_n', n_pre + n_post)
    autocorr = params.get('autocorr', 0)
    power = results.get('power', 0)
    alpha = params.get('alpha', 0.05)
    method = params.get('method', 'Analytical')
    outcome_type = results.get('outcome_type', 'Unknown')
    calculation_type = params.get('calculation_type', 'Power')
    
    # Get appropriate reference
    if 'continuous' in outcome_type.lower():
        reference = METHOD_REFERENCES.get("interrupted_time_series_continuous")
    elif 'binary' in outcome_type.lower():
        reference = METHOD_REFERENCES.get("interrupted_time_series_binary")
    else:
        reference = METHOD_REFERENCES.get("interrupted_time_series_continuous")  # Default
    
    # Outcome-specific parameters
    if 'continuous' in outcome_type.lower():
        mean_change = params.get('mean_change', 0)
        std_dev = params.get('std_dev', 1)
        effect_size = abs(mean_change) / std_dev if std_dev > 0 else 0
        effect_text = f"expected change in level of {mean_change:.2f} units (standardized effect size = {effect_size:.3f})"
    else:  # Binary
        p_pre = params.get('p_pre', 0)
        p_post = params.get('p_post', 0)
        risk_diff = p_post - p_pre if p_post and p_pre else 0
        effect_text = f"change in proportion from {p_pre:.2f} to {p_post:.2f} (risk difference = {risk_diff:.3f})"
    
    # Autocorrelation adjustment
    if autocorr > 0:
        adjustment_factor = (1 - autocorr) / (1 + autocorr)
        effective_n = total_n * adjustment_factor
        autocorr_text = f"""
        <tr style="background-color: #f8f9fa;">
            <td style="padding: 8px; border: 1px solid #dee2e6; font-weight: bold;">Autocorrelation (ρ)</td>
            <td style="padding: 8px; border: 1px solid #dee2e6;">{autocorr:.3f}</td>
        </tr>
        <tr>
            <td style="padding: 8px; border: 1px solid #dee2e6; font-weight: bold;">Effective Sample Size</td>
            <td style="padding: 8px; border: 1px solid #dee2e6;">{effective_n:.1f} (adjustment factor = {adjustment_factor:.3f})</td>
        </tr>
        """
    else:
        autocorr_text = ""
        effective_n = total_n
    
    # Calculation type specific text
    if calculation_type == "Sample Size":
        power_text = params.get('power', 0.8)
        result_text = f"""
        <p><strong>Required Sample Size:</strong></p>
        <p>• Pre-intervention time points: <span style="font-size: 1.1em; color: #2E86AB; font-weight: bold;">{n_pre}</span></p>
        <p>• Post-intervention time points: <span style="font-size: 1.1em; color: #2E86AB; font-weight: bold;">{n_post}</span></p>
        <p>• Total time points: <span style="font-size: 1.2em; color: #2E86AB; font-weight: bold;">{total_n}</span></p>
        <p><strong>Target Power:</strong> {power_text:.1%}</p>
        """
    else:  # Power calculation
        result_text = f"""
        <p><strong>Statistical Power:</strong> <span style="font-size: 1.2em; color: #2E86AB; font-weight: bold;">{power:.1%}</span></p>
        <p><strong>Sample Size:</strong> {total_n} time points ({n_pre} pre + {n_post} post)</p>
        """
    
    report_html = f"""
    <div style="font-family: Arial, sans-serif; line-height: 1.6; max-width: 800px;">
    
    <h2 style="color: #2E86AB; border-bottom: 2px solid #2E86AB; padding-bottom: 10px;">
    📈 Interrupted Time Series - Power Analysis Report
    </h2>
    
    <h3 style="color: #495057;">⚡ Analysis Results</h3>
    <div style="background-color: #e8f4fd; padding: 15px; border-radius: 8px; border-left: 4px solid #2E86AB;">
    {result_text}
    <p><strong>Significance Level (α):</strong> {alpha:.3f}</p>
    <p><strong>Expected Effect:</strong> {effect_text}</p>
    <p><strong>Analysis Method:</strong> {method}</p>
    </div>
    
    <div style="background-color: #e6f3ff; padding: 20px; border-radius: 8px; border-left: 4px solid #0066cc; margin: 20px 0;">
    <h4 style="color: #0052a3; margin-top: 0; margin-bottom: 15px;">📝 Methodological Description</h4>
    <div style="background-color: white; padding: 15px; border-radius: 6px; border: 1px solid #cce7ff;">
    <p style="font-style: italic; line-height: 1.8; margin: 0; color: #333;">
    {('A sample size of ' + str(n_pre) + ' pre-intervention and ' + str(n_post) + ' post-intervention time points (total ' + str(total_n) + ' observations) will provide ' + str(int(power*100)) + '% power to detect a ' + effect_text) if calculation_type != "Sample Size" else ('To achieve ' + str(int(params.get('power', 0.8)*100)) + '% power to detect a ' + effect_text + ', a minimum of ' + str(n_pre) + ' pre-intervention and ' + str(n_post) + ' post-intervention time points (total ' + str(total_n) + ' observations) is required')}, 
    {('assuming an autocorrelation coefficient of ' + str(autocorr) + ', ') if autocorr > 0 else ''}with a Type I error rate of {alpha*100:.0f}%. 
    Power calculations were performed using {'segmented regression analysis' if method.lower() == 'analytical' else 'simulation-based methods'} as described by {reference['citation'].split('.')[0] + ' et al.'}.
    </p>
    </div>
    <p style="font-size: 0.9em; color: #666; margin-top: 10px; margin-bottom: 0;">
    <strong>Tip:</strong> Copy the text above directly into your grant application or study protocol.
    </p>
    </div>
    
    
    <h3 style="color: #495057;">📊 Statistical Considerations</h3>
    <div style="background-color: #f8f9fa; padding: 15px; border-radius: 8px;">
    <p><strong>Segmented Regression Model:</strong></p>
    <p style="font-family: monospace; background-color: #e9ecef; padding: 10px; border-radius: 4px;">
    Y<sub>t</sub> = β<sub>0</sub> + β<sub>1</sub>×time<sub>t</sub> + β<sub>2</sub>×intervention<sub>t</sub> + β<sub>3</sub>×time_after_intervention<sub>t</sub> + ε<sub>t</sub>
    </p>
    <ul>
    <li><strong>β<sub>0</sub>:</strong> Baseline level at time zero</li>
    <li><strong>β<sub>1</sub>:</strong> Pre-intervention trend (slope)</li>
    <li><strong>β<sub>2</sub>:</strong> Level change immediately after intervention</li>
    <li><strong>β<sub>3</sub>:</strong> Change in trend (slope) after intervention</li>
    </ul>
    </div>
    
    <h3 style="color: #495057;">📚 Methodological Reference</h3>
    <div style="background-color: #f8f9fa; padding: 15px; border-radius: 8px;">
    <p><strong>Primary Reference:</strong></p>
    <p style="font-style: italic; margin: 10px 0;">{reference['citation']}</p>
    <p><strong>DOI:</strong> <a href="{reference['doi']}" target="_blank" style="color: #2E86AB;">{reference['doi']}</a></p>
    
    <p style="margin-top: 15px;"><strong>Additional Key References:</strong></p>
    <ul style="margin-top: 10px;">
    <li>Kontopantelis E, Doran T, Springate DA, Buchan I, Reeves D. (2013). Regression based quasi-experimental approach when randomisation is not possible: interrupted time series analysis. BMJ, 346, f2750.</li>
    <li>Penfold RB, Zhang F. (2013). Use of interrupted time series analysis in evaluating health care quality improvements. Academic Pediatrics, 13(6), S38-S44.</li>
    <li>Lopez Bernal J, Cummins S, Gasparrini A. (2018). The use of controls in interrupted time series studies of public health interventions. International Journal of Epidemiology, 47(6), 2082-2093.</li>
    </ul>
    </div>
    
    <div style="background-color: #d1ecf1; padding: 15px; border-radius: 8px; border-left: 4px solid #17a2b8; margin: 15px 0;">
    <h4 style="color: #0c5460; margin-top: 0;">⚠️ Key Assumptions & Limitations</h4>
    <ul style="color: #0c5460; margin-bottom: 0;">
    <li><strong>Minimum data points:</strong> At least 3 points before and after intervention recommended</li>
    <li><strong>Temporal stability:</strong> Underlying data generating process should be stable</li>
    <li><strong>No confounding events:</strong> Other interventions should not occur during study period</li>
    <li><strong>Autocorrelation:</strong> Must account for serial correlation in time series data</li>
    <li><strong>Stationarity:</strong> Time series should be stationary or appropriately modeled</li>
    </ul>
    </div>
    
    <hr style="margin: 20px 0; border: none; border-top: 1px solid #dee2e6;">
    <p style="font-size: 0.9em; color: #6c757d; text-align: center; margin: 10px 0;">
    Report generated by DesignPower • Interrupted Time Series Module
    </p>
    </div>
    """
    
    return report_html

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
    # Check if this is a Stepped Wedge design
    if design_type == 'Stepped Wedge':
        return generate_stepped_wedge_report(results, params)
        
    # Check if this is an Interrupted Time Series design
    if design_type == 'Interrupted Time Series':
        return generate_interrupted_time_series_report(results, params)
        
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
