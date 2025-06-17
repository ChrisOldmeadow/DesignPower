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
        mean1 = params.get('mean1', 0)
        mean2 = params.get('mean2', 0)
        std_dev = params.get('std_dev', 1)
        unequal_var = params.get('unequal_var', False)
        std_dev2 = params.get('std_dev2', std_dev) if unequal_var else std_dev
        
        # Calculate effect size if not provided
        effect_size = results.get('effect_size', 0)
        if effect_size == 0 and std_dev > 0:
            # Calculate Cohen's d
            effect_size = abs(mean2 - mean1) / std_dev
        
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
                    variance_text = f"unequal variances with standard deviations of {std_dev:.2f} in group 1 and {std_dev2:.2f} in group 2"
                    variance_display = f" (unequal variances: SD₁={std_dev:.2f}, SD₂={std_dev2:.2f})"
                else:
                    variance_text = f"equal variances with a standard deviation of {std_dev:.2f}"
                    variance_display = ""
                
                # Check if repeated measures design is being used
                repeated_measures = params.get("repeated_measures", False)
                repeated_measures_text = ""
                repeated_measures_param = ""
                if repeated_measures:
                    repeated_measures_text = f" This is a repeated measures design with a correlation of {correlation:.2f} between measurements."
                    repeated_measures_param = f"<p><strong>Repeated Measures:</strong> Yes (correlation = {correlation:.2f})</p>"
                
                # Check if simulation method was used
                if method == "simulation":
                    nsim = params.get("nsim", 1000)
                    seed = params.get("seed", 42)
                    method_text = f"using Monte Carlo simulation ({nsim:,} simulations, random seed {seed})"
                else:
                    method_text = "using analytical methods"
                
                # Create HTML report similar to cluster RCT style
                report_html = f"""
<div style="font-family: Arial, sans-serif; line-height: 1.6; max-width: 800px;">

<h2 style="color: #2E86AB; border-bottom: 2px solid #2E86AB; padding-bottom: 10px;">
Parallel RCT - Sample Size Calculation Report
</h2>

<h3 style="color: #495057;">Required Sample Size</h3>
<div style="background-color: #e8f4fd; padding: 20px; border-radius: 8px; border-left: 4px solid #2E86AB;">
    <table style="width: 100%; border-collapse: collapse;">
    <tr>
        <td style="padding: 8px;"><strong>Sample Size Group 1:</strong></td>
        <td style="padding: 8px; font-size: 1.2em; color: #2E86AB;"><strong>{n1}</strong></td>
        <td style="padding: 8px;"><strong>Sample Size Group 2:</strong></td>
        <td style="padding: 8px; font-size: 1.2em; color: #2E86AB;"><strong>{n2}</strong></td>
    </tr>
    <tr>
        <td style="padding: 8px;"><strong>Total Sample Size:</strong></td>
        <td style="padding: 8px; font-size: 1.2em; color: #2E86AB;"><strong>{n1 + n2}</strong></td>
        <td style="padding: 8px;"><strong>Allocation Ratio:</strong></td>
        <td style="padding: 8px;">{n1}:{n2}</td>
    </tr>
    <tr>
        <td style="padding: 8px;"><strong>Target Power:</strong></td>
        <td style="padding: 8px;">{power * 100:.0f}%</td>
        <td style="padding: 8px;"><strong>Significance Level (α):</strong></td>
        <td style="padding: 8px;">{alpha:.3f}</td>
    </tr>
    </table>
</div>

<h3 style="color: #495057;">Effect Size & Study Parameters</h3>
<div style="background-color: #f8f9fa; padding: 15px; border-radius: 8px;">
    <p><strong>Target Difference:</strong> {abs(mean2 - mean1):.2f} units (Mean₁ = {mean1:.2f}, Mean₂ = {mean2:.2f})</p>
    <p><strong>Standardized Effect Size:</strong> Cohen's d = {effect_size:.3f}</p>
    <p><strong>Standard Deviation:</strong> {std_dev:.2f}{variance_display}</p>
    <p><strong>Analysis Method:</strong> {method_text}</p>
    {repeated_measures_param}
</div>

<div style="background-color: #e6f3ff; padding: 20px; border-radius: 8px; border-left: 4px solid #0066cc; margin: 20px 0;">
    <h4 style="color: #0052a3; margin-top: 0; margin-bottom: 15px;">Methodological Description</h4>
    <div style="background-color: white; padding: 15px; border-radius: 6px; border: 1px solid #cce7ff;">
        <p style="font-style: italic; line-height: 1.8; margin: 0; color: #333;">
        A sample size of {n1} participants in group 1 and {n2} participants in group 2 
        (total N = {n1 + n2}) will provide {power * 100:.0f}% power to detect a difference 
        in means of {abs(mean2 - mean1):.2f} (effect size d = {effect_size:.2f}) between 
        groups, assuming {variance_text}, {method_text} with a Type I error rate of {alpha * 100:.0f}%.{repeated_measures_text}
        </p>
    </div>
    <p style="font-size: 0.9em; color: #666; margin-top: 10px; margin-bottom: 0;">
    <strong>Tip:</strong> Copy the text above directly into your grant application or study protocol.
    </p>
</div>

<div style="background-color: #fffacd; padding: 15px; border-radius: 8px; border-left: 4px solid #ffd700; margin: 15px 0;">
    <h4 style="color: #856404; margin-top: 0;">Sample Size Interpretation</h4>
    <p style="color: #856404; margin-bottom: 10px;">
    The calculated sample size ensures:
    </p>
    <ul style="color: #856404; margin-bottom: 0;">
    <li>{power * 100:.0f}% probability of detecting the effect if it truly exists</li>
    <li>Type II error rate (β) = {(1-power)*100:.0f}%</li>
    <li>Effect size of {effect_size:.3f} is {'small' if effect_size < 0.5 else 'medium' if effect_size < 0.8 else 'large'} (Cohen's guidelines)</li>
    <li>Consider adding 10-20% for potential dropout/attrition</li>
    </ul>
</div>

<h3 style="color: #495057;">Methodological References</h3>
<div style="background-color: #f8f9fa; padding: 15px; border-radius: 8px;">
    <p><strong>Primary Reference:</strong></p>
    <p style="font-style: italic; margin: 10px 0;">{reference['citation']}</p>
    <p><strong>DOI:</strong> <a href="{reference['doi']}" target="_blank" style="color: #2E86AB;">{reference['doi']}</a></p>
    <p style="margin-top: 15px;"><strong>Additional Key References:</strong></p>
    <ul style="margin-top: 10px;">
    <li>Chow SC, Shao J, Wang H, Lokhnygina Y. (2017). Sample Size Calculations in Clinical Research. 3rd Edition. CRC Press.</li>
    <li>Julious SA. (2010). Sample Sizes for Clinical Trials. CRC Press.</li>
    <li>Machin D, Campbell MJ, Tan SB, Tan SH. (2018). Sample Sizes for Clinical, Laboratory and Epidemiology Studies. 4th Edition. Wiley-Blackwell.</li>
    </ul>
</div>

<div style="background-color: #d1ecf1; padding: 15px; border-radius: 8px; border-left: 4px solid #17a2b8; margin: 15px 0;">
    <h4 style="color: #0c5460; margin-top: 0;">Important Considerations</h4>
    <ul style="color: #0c5460; margin-bottom: 0;">
    <li><strong>Assumptions:</strong> Results assume normal distributions and {variance_text}</li>
    <li><strong>Effect Size:</strong> Ensure {abs(mean2 - mean1):.2f} units is clinically meaningful</li>
    <li><strong>Attrition:</strong> Consider inflating sample size by 10-20% for expected dropout</li>
    <li><strong>Interim Analysis:</strong> If planned, adjust sample size for multiple testing</li>
    </ul>
</div>

<hr style="margin: 20px 0; border: none; border-top: 1px solid #dee2e6;">
<p style="font-size: 0.9em; color: #6c757d; text-align: center; margin: 10px 0;">
Report generated by DesignPower • Parallel RCT Module • Continuous Outcome
</p>
</div>
                """
                return report_html.strip()
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
                    method_text = f"using Monte Carlo simulation ({nsim:,} simulations, random seed {seed}) with {test_type}"
                else:
                    method_text = f"using {test_type}"
                
                if correction:
                    method_text += " with continuity correction"
                
                # Calculate additional metrics
                risk_diff = abs(p2 - p1)
                if p1 > 0 and p1 < 1 and p2 < 1:
                    if p2 > 0:
                        calc_odds_ratio = (p2 / (1 - p2)) / (p1 / (1 - p1))
                    else:
                        calc_odds_ratio = 0
                else:
                    calc_odds_ratio = odds_ratio
                    
                if p1 > 0:
                    relative_risk = p2 / p1
                else:
                    relative_risk = None
                
                # Create HTML report
                report_html = f"""
<div style="font-family: Arial, sans-serif; line-height: 1.6; max-width: 800px;">

<h2 style="color: #2E86AB; border-bottom: 2px solid #2E86AB; padding-bottom: 10px;">
Parallel RCT - Sample Size Calculation Report
</h2>

<h3 style="color: #495057;">Required Sample Size</h3>
<div style="background-color: #e8f4fd; padding: 20px; border-radius: 8px; border-left: 4px solid #2E86AB;">
    <table style="width: 100%; border-collapse: collapse;">
    <tr>
        <td style="padding: 8px;"><strong>Sample Size Group 1:</strong></td>
        <td style="padding: 8px; font-size: 1.2em; color: #2E86AB;"><strong>{n1}</strong></td>
        <td style="padding: 8px;"><strong>Sample Size Group 2:</strong></td>
        <td style="padding: 8px; font-size: 1.2em; color: #2E86AB;"><strong>{n2}</strong></td>
    </tr>
    <tr>
        <td style="padding: 8px;"><strong>Total Sample Size:</strong></td>
        <td style="padding: 8px; font-size: 1.2em; color: #2E86AB;"><strong>{n1 + n2}</strong></td>
        <td style="padding: 8px;"><strong>Allocation Ratio:</strong></td>
        <td style="padding: 8px;">{n1}:{n2}</td>
    </tr>
    <tr>
        <td style="padding: 8px;"><strong>Target Power:</strong></td>
        <td style="padding: 8px;">{power * 100:.0f}%</td>
        <td style="padding: 8px;"><strong>Significance Level (α):</strong></td>
        <td style="padding: 8px;">{alpha:.3f}</td>
    </tr>
    </table>
</div>

<h3 style="color: #495057;">Effect Size & Study Parameters</h3>
<div style="background-color: #f8f9fa; padding: 15px; border-radius: 8px;">
    <p><strong>Control Group Proportion:</strong> {p1:.3f} ({p1*100:.1f}%)</p>
    <p><strong>Treatment Group Proportion:</strong> {p2:.3f} ({p2*100:.1f}%)</p>
    <p><strong>Risk Difference:</strong> {risk_diff:.3f} ({risk_diff*100:.1f} percentage points)</p>
    <p><strong>Risk Ratio:</strong> {(f'{relative_risk:.3f}') if relative_risk is not None else 'N/A'}</p>
    <p><strong>Odds Ratio:</strong> {calc_odds_ratio:.3f}</p>
    <p><strong>Analysis Method:</strong> {method_text}</p>
</div>

<div style="background-color: #e6f3ff; padding: 20px; border-radius: 8px; border-left: 4px solid #0066cc; margin: 20px 0;">
    <h4 style="color: #0052a3; margin-top: 0; margin-bottom: 15px;">Methodological Description</h4>
    <div style="background-color: white; padding: 15px; border-radius: 6px; border: 1px solid #cce7ff;">
        <p style="font-style: italic; line-height: 1.8; margin: 0; color: #333;">
        A sample size of {n1} participants in group 1 and {n2} participants in group 2 
        (total N = {n1 + n2}) will provide {power * 100:.0f}% power to detect a difference 
        in proportions from {p1:.2f} ({p1*100:.1f}%) to {p2:.2f} ({p2*100:.1f}%), 
        corresponding to a risk difference of {risk_diff:.3f} and odds ratio of {calc_odds_ratio:.2f}, 
        {method_text}, with a Type I error rate of {alpha * 100:.0f}%.
        </p>
    </div>
    <p style="font-size: 0.9em; color: #666; margin-top: 10px; margin-bottom: 0;">
    <strong>Tip:</strong> Copy the text above directly into your grant application or study protocol.
    </p>
</div>

<div style="background-color: #fffacd; padding: 15px; border-radius: 8px; border-left: 4px solid #ffd700; margin: 15px 0;">
    <h4 style="color: #856404; margin-top: 0;">Sample Size Interpretation</h4>
    <p style="color: #856404; margin-bottom: 10px;">
    The calculated sample size ensures:
    </p>
    <ul style="color: #856404; margin-bottom: 0;">
    <li>{power * 100:.0f}% probability of detecting the effect if it truly exists</li>
    <li>Type II error rate (β) = {(1-power)*100:.0f}%</li>
    <li>Number Needed to Treat (NNT) = {int(1/risk_diff) if risk_diff > 0 else 'N/A'}</li>
    <li>Consider 10-20% inflation for potential dropout</li>
    </ul>
</div>

<h3 style="color: #495057;">Methodological References</h3>
<div style="background-color: #f8f9fa; padding: 15px; border-radius: 8px;">
    <p><strong>Primary Reference:</strong></p>
    <p style="font-style: italic; margin: 10px 0;">{reference['citation']}</p>
    <p><strong>DOI:</strong> <a href="{reference['doi']}" target="_blank" style="color: #2E86AB;">{reference['doi']}</a></p>
    <p style="margin-top: 15px;"><strong>Additional Key References:</strong></p>
    <ul style="margin-top: 10px;">
    <li>Fleiss JL, Tytun A, Ury HK. (1980). A simple approximation for calculating sample sizes for comparing independent proportions. Biometrics, 36(2), 343-346.</li>
    <li>Casagrande JT, Pike MC, Smith PG. (1978). An improved approximate formula for calculating sample sizes for comparing two binomial distributions. Biometrics, 34(3), 483-486.</li>
    <li>Farrington CP, Manning G. (1990). Test statistics and sample size formulae for comparative binomial trials with null hypothesis of non-zero risk difference or non-unity relative risk. Statistics in Medicine, 9(12), 1447-1454.</li>
    </ul>
</div>

<div style="background-color: #d1ecf1; padding: 15px; border-radius: 8px; border-left: 4px solid #17a2b8; margin: 15px 0;">
    <h4 style="color: #0c5460; margin-top: 0;">Important Considerations</h4>
    <ul style="color: #0c5460; margin-bottom: 0;">
    <li><strong>Test Selection:</strong> {test_type} {'is exact for small samples' if 'exact' in test_type.lower() else 'uses normal approximation'}</li>
    <li><strong>Rare Events:</strong> {'Consider exact methods' if min(p1, p2) < 0.1 else 'Normal approximation appropriate'}</li>
    <li><strong>Sample Size Balance:</strong> {'Groups are balanced' if n1 == n2 else f'Unbalanced design (ratio {n1/n2:.2f}:1)'}</li>
    <li><strong>Clinical Significance:</strong> Ensure {risk_diff*100:.1f} percentage point difference is meaningful</li>
    </ul>
</div>

<hr style="margin: 20px 0; border: none; border-top: 1px solid #dee2e6;">
<p style="font-size: 0.9em; color: #6c757d; text-align: center; margin: 10px 0;">
Report generated by DesignPower • Parallel RCT Module • Binary Outcome
</p>
</div>
                """
                return report_html.strip()
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

                # Get median survival for treatment group
                median_survival2 = median_survival1 / hr if hr > 0 else float('inf')
                
                # Expected events
                events = results.get('events', 0)
                
                # Create HTML report
                report_html = f"""
<div style="font-family: Arial, sans-serif; line-height: 1.6; max-width: 800px;">

<h2 style="color: #2E86AB; border-bottom: 2px solid #2E86AB; padding-bottom: 10px;">
Parallel RCT - Sample Size Calculation Report (Survival Analysis)
</h2>

<h3 style="color: #495057;">Required Sample Size</h3>
<div style="background-color: #e8f4fd; padding: 20px; border-radius: 8px; border-left: 4px solid #2E86AB;">
    <table style="width: 100%; border-collapse: collapse;">
    <tr>
        <td style="padding: 8px;"><strong>Sample Size Group 1:</strong></td>
        <td style="padding: 8px; font-size: 1.2em; color: #2E86AB;"><strong>{n1}</strong></td>
        <td style="padding: 8px;"><strong>Sample Size Group 2:</strong></td>
        <td style="padding: 8px; font-size: 1.2em; color: #2E86AB;"><strong>{n2}</strong></td>
    </tr>
    <tr>
        <td style="padding: 8px;"><strong>Total Sample Size:</strong></td>
        <td style="padding: 8px; font-size: 1.2em; color: #2E86AB;"><strong>{n1 + n2}</strong></td>
        <td style="padding: 8px;"><strong>Expected Events:</strong></td>
        <td style="padding: 8px; font-size: 1.1em;">{events:.0f}</td>
    </tr>
    <tr>
        <td style="padding: 8px;"><strong>Target Power:</strong></td>
        <td style="padding: 8px;">{power * 100:.0f}%</td>
        <td style="padding: 8px;"><strong>Significance Level (α):</strong></td>
        <td style="padding: 8px;">{alpha:.3f}</td>
    </tr>
    </table>
</div>

<h3 style="color: #495057;">Effect Size & Study Parameters</h3>
<div style="background-color: #f8f9fa; padding: 15px; border-radius: 8px;">
    <p><strong>Target Hazard Ratio:</strong> {hr:.3f}</p>
    <p><strong>Median Survival (Control):</strong> {median_survival1:.1f} months</p>
    <p><strong>Median Survival (Treatment):</strong> {(f'{median_survival2:.1f}') if median_survival2 != float('inf') else '∞'} months</p>
    <p><strong>Accrual Period:</strong> {accrual_time:.1f} months</p>
    <p><strong>Follow-up Period:</strong> {follow_up_time:.1f} months</p>
    <p><strong>Dropout Rate:</strong> {dropout_rate*100:.1f}%</p>
    <p><strong>Analysis Method:</strong> {method_desc}</p>
</div>

<div style="background-color: #e6f3ff; padding: 20px; border-radius: 8px; border-left: 4px solid #0066cc; margin: 20px 0;">
    <h4 style="color: #0052a3; margin-top: 0; margin-bottom: 15px;">Methodological Description</h4>
    <div style="background-color: white; padding: 15px; border-radius: 6px; border: 1px solid #cce7ff;">
        <p style="font-style: italic; line-height: 1.8; margin: 0; color: #333;">
        A sample size of {n1} participants in group 1 and {n2} participants in group 2 
        (total N = {n1 + n2}) will provide {power * 100:.0f}% power to detect a hazard ratio 
        of {hr:.2f}, assuming a median survival time of {median_survival1:.1f} months in the 
        reference group. {('This corresponds to a median survival of ' + f'{median_survival2:.1f}' + ' months in the treatment group.') if median_survival2 != float('inf') else ''} 
        The calculation uses the {method_desc}{accrual_desc}. Study assumes exponential survival distributions with 
        accrual period of {accrual_time:.1f} months, follow-up period of {follow_up_time:.1f} months, 
        and anticipated dropout rate of {dropout_rate*100:.1f}%. Expected number of events is {events:.0f}. 
        Analysis will use a log-rank test with a Type I error rate of {alpha * 100:.0f}%.
        </p>
    </div>
    <p style="font-size: 0.9em; color: #666; margin-top: 10px; margin-bottom: 0;">
    <strong>Tip:</strong> Copy the text above directly into your grant application or study protocol.
    </p>
</div>

{f'''<div style="background-color: #e8f4fd; padding: 15px; border-radius: 8px; border-left: 4px solid #2E86AB; margin: 15px 0;">
    <h4 style="color: #0052a3; margin-top: 0;">Method Comparison</h4>
    <p style="color: #0052a3;">{comparison_text.strip()}</p>
</div>''' if comparison_text else ''}

<div style="background-color: #fffacd; padding: 15px; border-radius: 8px; border-left: 4px solid #ffd700; margin: 15px 0;">
    <h4 style="color: #856404; margin-top: 0;">Sample Size Interpretation</h4>
    <p style="color: #856404; margin-bottom: 10px;">
    The calculated sample size ensures:
    </p>
    <ul style="color: #856404; margin-bottom: 0;">
    <li>{power * 100:.0f}% probability of detecting HR = {hr:.3f} if it truly exists</li>
    <li>Type II error rate (β) = {(1-power)*100:.0f}%</li>
    <li>Expected {events:.0f} events provides {'good' if events > 50 else 'marginal' if events > 30 else 'limited'} precision</li>
    <li>Events per arm: ~{events/2:.0f} (assuming balanced design)</li>
    </ul>
</div>

<h3 style="color: #495057;">Methodological References</h3>
<div style="background-color: #f8f9fa; padding: 15px; border-radius: 8px;">
    <p><strong>Primary Reference:</strong></p>
    <p style="font-style: italic; margin: 10px 0;">{reference['citation']}</p>
    <p><strong>DOI:</strong> <a href="{reference['doi']}" target="_blank" style="color: #2E86AB;">{reference['doi']}</a></p>
    <p style="margin-top: 15px;"><strong>Additional Key References:</strong></p>
    <ul style="margin-top: 10px;">
    <li>Collett D. (2015). Modelling Survival Data in Medical Research. 3rd Edition. CRC Press.</li>
    <li>Machin D, Campbell MJ, Tan SB, Tan SH. (2018). Sample Sizes for Clinical, Laboratory and Epidemiology Studies. 4th Edition. Wiley-Blackwell.</li>
    <li>Hsieh FY, Lavori PW. (2000). Sample-size calculations for the Cox proportional hazards regression model with nonbinary covariates. Controlled Clinical Trials, 21(6), 552-560.</li>
    </ul>
</div>

<div style="background-color: #d1ecf1; padding: 15px; border-radius: 8px; border-left: 4px solid #17a2b8; margin: 15px 0;">
    <h4 style="color: #0c5460; margin-top: 0;">Important Considerations</h4>
    <ul style="color: #0c5460; margin-bottom: 0;">
    <li><strong>Proportional Hazards:</strong> Results assume constant hazard ratio over time</li>
    <li><strong>Censoring Pattern:</strong> Calculations assume administrative censoring only</li>
    <li><strong>Accrual Pattern:</strong> {current_accrual_pattern.replace('_', ' ').title()} accrual assumed</li>
    <li><strong>Sample Size Balance:</strong> {'Groups are balanced' if n1 == n2 else f'Unbalanced design (ratio {n1/n2:.2f}:1)'}</li>
    </ul>
</div>

<hr style="margin: 20px 0; border: none; border-top: 1px solid #dee2e6;">
<p style="font-size: 0.9em; color: #6c757d; text-align: center; margin: 10px 0;">
Report generated by DesignPower • Parallel RCT Module • Survival Outcome
</p>
</div>
                """
                return report_html.strip()
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
        sd1 = params.get('sd1', params.get('std_dev', 1))
        sd2 = params.get('sd2', params.get('std_dev2', sd1))
        std_dev = params.get('std_dev', sd1)
        actual_effect_size = results.get('effect_size')
        difference = abs(mean1 - mean2)
        
        # Calculate effect size if not provided
        if actual_effect_size is None and std_dev and std_dev != 0:
            actual_effect_size = difference / std_dev
        
        # Power interpretation
        if power >= 0.9:
            power_adequacy = "excellent power"
        elif power >= 0.8:
            power_adequacy = "adequate power"
        elif power >= 0.7:
            power_adequacy = "marginal power"
        else:
            power_adequacy = "insufficient power"
        
        # Method text
        if method == "simulation":
            nsim = params.get("nsim", 1000)
            seed = params.get("seed", 42)
            method_text = f"using Monte Carlo simulation ({nsim:,} simulations, random seed {seed})"
        else:
            method_text = "using analytical methods"
        
        # Get appropriate reference
        reference = get_method_reference('continuous', method=method, design=None)
        
        report_html = f"""
<div style="font-family: Arial, sans-serif; line-height: 1.6; max-width: 800px;">

<h2 style="color: #2E86AB; border-bottom: 2px solid #2E86AB; padding-bottom: 10px;">
Parallel RCT - Power Calculation Report
</h2>

<h3 style="color: #495057;">Statistical Power Results</h3>
<div style="background-color: #e8f4fd; padding: 20px; border-radius: 8px; border-left: 4px solid #2E86AB;">
    <table style="width: 100%; border-collapse: collapse;">
    <tr>
        <td style="padding: 8px;"><strong>Statistical Power:</strong></td>
        <td style="padding: 8px; font-size: 1.2em; color: #2E86AB;"><strong>{power * 100:.1f}%</strong></td>
        <td style="padding: 8px;"><strong>Power Assessment:</strong></td>
        <td style="padding: 8px;">{power_adequacy.capitalize()}</td>
    </tr>
    <tr>
        <td style="padding: 8px;"><strong>Sample Size Group 1:</strong></td>
        <td style="padding: 8px;">{n1}</td>
        <td style="padding: 8px;"><strong>Sample Size Group 2:</strong></td>
        <td style="padding: 8px;">{n2}</td>
    </tr>
    <tr>
        <td style="padding: 8px;"><strong>Total Sample Size:</strong></td>
        <td style="padding: 8px;">{n1 + n2}</td>
        <td style="padding: 8px;"><strong>Significance Level (α):</strong></td>
        <td style="padding: 8px;">{alpha:.3f}</td>
    </tr>
    </table>
</div>

<h3 style="color: #495057;">Effect Size & Study Parameters</h3>
<div style="background-color: #f8f9fa; padding: 15px; border-radius: 8px;">
    <p><strong>Target Difference:</strong> {difference:.2f} units (Mean₁ = {mean1:.2f}, Mean₂ = {mean2:.2f})</p>
    <p><strong>Standardized Effect Size:</strong> Cohen's d = {(f'{actual_effect_size:.3f}') if actual_effect_size is not None else 'N/A'}</p>
    <p><strong>Standard Deviation:</strong> {std_dev:.2f}{(' (unequal variances: SD₁=' + f'{sd1:.2f}' + ', SD₂=' + f'{sd2:.2f}' + ')') if sd1 != sd2 else ''}</p>
    <p><strong>Analysis Method:</strong> {method_text}</p>
</div>

<div style="background-color: #e6f3ff; padding: 20px; border-radius: 8px; border-left: 4px solid #0066cc; margin: 20px 0;">
    <h4 style="color: #0052a3; margin-top: 0; margin-bottom: 15px;">Methodological Description</h4>
    <div style="background-color: white; padding: 15px; border-radius: 6px; border: 1px solid #cce7ff;">
        <p style="font-style: italic; line-height: 1.8; margin: 0; color: #333;">
        For a study with {n1} participants in group 1 and {n2} participants in group 2 
        (total N = {n1 + n2}), targeting a difference in means of {difference:.2f} units 
        (from {mean1:.2f} to {mean2:.2f}), the estimated statistical power is {power * 100:.1f}%. 
        This represents a {(f'{actual_effect_size:.3f}') if actual_effect_size is not None else 'N/A'} standardized effect size. 
        The calculation assumes {f'equal variances with standard deviation {std_dev:.2f}' if sd1 == sd2 else f'unequal variances (SD₁={sd1:.2f}, SD₂={sd2:.2f})'} 
        and was performed {method_text} with a Type I error rate of {alpha*100:.0f}%.
        </p>
    </div>
    <p style="font-size: 0.9em; color: #666; margin-top: 10px; margin-bottom: 0;">
    <strong>Tip:</strong> Copy the text above directly into your grant application or study protocol.
    </p>
</div>

<div style="background-color: #fffacd; padding: 15px; border-radius: 8px; border-left: 4px solid #ffd700; margin: 15px 0;">
    <h4 style="color: #856404; margin-top: 0;">Power Interpretation</h4>
    <p style="color: #856404; margin-bottom: 10px;">
    With {power * 100:.0f}% power, this study has {power_adequacy}:
    </p>
    <ul style="color: #856404; margin-bottom: 0;">
    <li>There is a {power * 100:.0f}% probability of detecting the effect if it truly exists</li>
    <li>Type II error rate (β) = {(1-power)*100:.0f}% (probability of missing a true effect)</li>
    <li>{'Consider increasing sample size' if power < 0.8 else 'Sample size appears adequate'} for this effect size</li>
    <li>This represents a {('small' if actual_effect_size and actual_effect_size < 0.5 else 'medium' if actual_effect_size and actual_effect_size < 0.8 else 'large' if actual_effect_size else 'N/A') if actual_effect_size is not None else 'N/A'} effect size</li>
    </ul>
</div>

<h3 style="color: #495057;">Methodological References</h3>
<div style="background-color: #f8f9fa; padding: 15px; border-radius: 8px;">
    <p><strong>Primary Reference:</strong></p>
    <p style="font-style: italic; margin: 10px 0;">{reference['citation']}</p>
    <p><strong>DOI:</strong> <a href="{reference['doi']}" target="_blank" style="color: #2E86AB;">{reference['doi']}</a></p>
    <p style="margin-top: 15px;"><strong>Additional Key References:</strong></p>
    <ul style="margin-top: 10px;">
    <li>Chow SC, Shao J, Wang H, Lokhnygina Y. (2017). Sample Size Calculations in Clinical Research. 3rd Edition. CRC Press.</li>
    <li>Julious SA. (2010). Sample Sizes for Clinical Trials. CRC Press.</li>
    <li>Machin D, Campbell MJ, Tan SB, Tan SH. (2018). Sample Sizes for Clinical, Laboratory and Epidemiology Studies. 4th Edition. Wiley-Blackwell.</li>
    </ul>
</div>

<div style="background-color: #d1ecf1; padding: 15px; border-radius: 8px; border-left: 4px solid #17a2b8; margin: 15px 0;">
    <h4 style="color: #0c5460; margin-top: 0;">Important Considerations</h4>
    <ul style="color: #0c5460; margin-bottom: 0;">
    <li><strong>Assumptions:</strong> Results assume normal distributions and {'equal' if sd1 == sd2 else 'unequal'} variances</li>
    <li><strong>Effect Size Context:</strong> Ensure {difference:.2f} units is clinically meaningful</li>
    <li><strong>Sample Size Balance:</strong> {'Groups are balanced' if n1 == n2 else f'Unbalanced design (ratio {n1/n2:.2f}:1)'}</li>
    <li><strong>Dropout Allowance:</strong> Consider inflating sample size for expected attrition</li>
    </ul>
</div>

<hr style="margin: 20px 0; border: none; border-top: 1px solid #dee2e6;">
<p style="font-size: 0.9em; color: #6c757d; text-align: center; margin: 10px 0;">
Report generated by DesignPower • Parallel RCT Module • Continuous Outcome
</p>
</div>
        """
        return report_html.strip()
        
    elif design_type == 'Parallel RCT' and 'Binary' in outcome_type:
        # Enhanced binary outcome power reporting
        p1 = params.get('p1', 0)
        p2 = params.get('p2', 0)
        test_type = params.get('test_type', 'Normal Approximation')
        correction = params.get('correction', False)
        risk_diff = abs(p2 - p1)
        
        # Calculate odds ratio and relative risk
        if p1 > 0 and p1 < 1 and p2 < 1:
            odds_ratio = (p2 / (1 - p2)) / (p1 / (1 - p1)) if p2 > 0 else 0
        else:
            odds_ratio = None
            
        if p1 > 0:
            relative_risk = p2 / p1
        else:
            relative_risk = None
        
        # Power interpretation
        if power >= 0.9:
            power_adequacy = "excellent power"
        elif power >= 0.8:
            power_adequacy = "adequate power"
        elif power >= 0.7:
            power_adequacy = "marginal power"
        else:
            power_adequacy = "insufficient power"
        
        # Method text
        if method == "simulation":
            nsim = params.get("nsim", 1000)
            seed = params.get("seed", 42)
            method_text = f"using Monte Carlo simulation ({nsim:,} simulations, random seed {seed}) with {test_type}"
        else:
            method_text = f"using {test_type}"
        
        if correction:
            method_text += " with continuity correction"
        
        # Get appropriate reference
        reference = get_method_reference('binary', test_type, method)
        
        report_html = f"""
<div style="font-family: Arial, sans-serif; line-height: 1.6; max-width: 800px;">

<h2 style="color: #2E86AB; border-bottom: 2px solid #2E86AB; padding-bottom: 10px;">
Parallel RCT - Power Calculation Report
</h2>

<h3 style="color: #495057;">Statistical Power Results</h3>
<div style="background-color: #e8f4fd; padding: 20px; border-radius: 8px; border-left: 4px solid #2E86AB;">
    <table style="width: 100%; border-collapse: collapse;">
    <tr>
        <td style="padding: 8px;"><strong>Statistical Power:</strong></td>
        <td style="padding: 8px; font-size: 1.2em; color: #2E86AB;"><strong>{power * 100:.1f}%</strong></td>
        <td style="padding: 8px;"><strong>Power Assessment:</strong></td>
        <td style="padding: 8px;">{power_adequacy.capitalize()}</td>
    </tr>
    <tr>
        <td style="padding: 8px;"><strong>Sample Size Group 1:</strong></td>
        <td style="padding: 8px;">{n1}</td>
        <td style="padding: 8px;"><strong>Sample Size Group 2:</strong></td>
        <td style="padding: 8px;">{n2}</td>
    </tr>
    <tr>
        <td style="padding: 8px;"><strong>Total Sample Size:</strong></td>
        <td style="padding: 8px;">{n1 + n2}</td>
        <td style="padding: 8px;"><strong>Significance Level (α):</strong></td>
        <td style="padding: 8px;">{alpha:.3f}</td>
    </tr>
    </table>
</div>

<h3 style="color: #495057;">Effect Size & Study Parameters</h3>
<div style="background-color: #f8f9fa; padding: 15px; border-radius: 8px;">
    <p><strong>Control Group Proportion:</strong> {p1:.3f} ({p1*100:.1f}%)</p>
    <p><strong>Treatment Group Proportion:</strong> {p2:.3f} ({p2*100:.1f}%)</p>
    <p><strong>Risk Difference:</strong> {risk_diff:.3f} ({risk_diff*100:.1f} percentage points)</p>
    <p><strong>Risk Ratio:</strong> {relative_risk:.3f if relative_risk is not None else 'N/A'}</p>
    <p><strong>Odds Ratio:</strong> {odds_ratio:.3f if odds_ratio is not None else 'N/A'}</p>
    <p><strong>Analysis Method:</strong> {method_text}</p>
</div>

<div style="background-color: #e6f3ff; padding: 20px; border-radius: 8px; border-left: 4px solid #0066cc; margin: 20px 0;">
    <h4 style="color: #0052a3; margin-top: 0; margin-bottom: 15px;">Methodological Description</h4>
    <div style="background-color: white; padding: 15px; border-radius: 6px; border: 1px solid #cce7ff;">
        <p style="font-style: italic; line-height: 1.8; margin: 0; color: #333;">
        For a study with {n1} participants in group 1 and {n2} participants in group 2 
        (total N = {n1 + n2}), targeting a change in proportion from {p1:.3f} ({p1*100:.1f}%) 
        to {p2:.3f} ({p2*100:.1f}%), the estimated statistical power is {power * 100:.1f}%. 
        This represents a risk difference of {risk_diff:.3f} ({risk_diff*100:.1f} percentage points)
        {f', risk ratio of {relative_risk:.2f}' if relative_risk is not None else ''}
        {f', and odds ratio of {odds_ratio:.2f}' if odds_ratio is not None else ''}. 
        The calculation was performed {method_text} with a Type I error rate of {alpha*100:.0f}%.
        </p>
    </div>
    <p style="font-size: 0.9em; color: #666; margin-top: 10px; margin-bottom: 0;">
    <strong>Tip:</strong> Copy the text above directly into your grant application or study protocol.
    </p>
</div>

<div style="background-color: #fffacd; padding: 15px; border-radius: 8px; border-left: 4px solid #ffd700; margin: 15px 0;">
    <h4 style="color: #856404; margin-top: 0;">Power Interpretation</h4>
    <p style="color: #856404; margin-bottom: 10px;">
    With {power * 100:.0f}% power, this study has {power_adequacy}:
    </p>
    <ul style="color: #856404; margin-bottom: 0;">
    <li>There is a {power * 100:.0f}% probability of detecting the effect if it truly exists</li>
    <li>Type II error rate (β) = {(1-power)*100:.0f}% (probability of missing a true effect)</li>
    <li>{'Consider increasing sample size' if power < 0.8 else 'Sample size appears adequate'} for this effect size</li>
    <li>Number Needed to Treat (NNT) = {int(1/risk_diff) if risk_diff > 0 else 'N/A'}</li>
    </ul>
</div>

<h3 style="color: #495057;">Methodological References</h3>
<div style="background-color: #f8f9fa; padding: 15px; border-radius: 8px;">
    <p><strong>Primary Reference:</strong></p>
    <p style="font-style: italic; margin: 10px 0;">{reference['citation']}</p>
    <p><strong>DOI:</strong> <a href="{reference['doi']}" target="_blank" style="color: #2E86AB;">{reference['doi']}</a></p>
    <p style="margin-top: 15px;"><strong>Additional Key References:</strong></p>
    <ul style="margin-top: 10px;">
    <li>Fleiss JL, Tytun A, Ury HK. (1980). A simple approximation for calculating sample sizes for comparing independent proportions. Biometrics, 36(2), 343-346.</li>
    <li>Casagrande JT, Pike MC, Smith PG. (1978). An improved approximate formula for calculating sample sizes for comparing two binomial distributions. Biometrics, 34(3), 483-486.</li>
    <li>Farrington CP, Manning G. (1990). Test statistics and sample size formulae for comparative binomial trials with null hypothesis of non-zero risk difference or non-unity relative risk. Statistics in Medicine, 9(12), 1447-1454.</li>
    </ul>
</div>

<div style="background-color: #d1ecf1; padding: 15px; border-radius: 8px; border-left: 4px solid #17a2b8; margin: 15px 0;">
    <h4 style="color: #0c5460; margin-top: 0;">Important Considerations</h4>
    <ul style="color: #0c5460; margin-bottom: 0;">
    <li><strong>Test Selection:</strong> {test_type} {'is exact for small samples' if 'exact' in test_type.lower() else 'uses normal approximation'}</li>
    <li><strong>Rare Events:</strong> {'Consider exact methods' if min(p1, p2) < 0.1 else 'Normal approximation appropriate'}</li>
    <li><strong>Sample Size Balance:</strong> {'Groups are balanced' if n1 == n2 else f'Unbalanced design (ratio {n1/n2:.2f}:1)'}</li>
    <li><strong>Clinical Significance:</strong> Ensure {risk_diff*100:.1f} percentage point difference is meaningful</li>
    </ul>
</div>

<hr style="margin: 20px 0; border: none; border-top: 1px solid #dee2e6;">
<p style="font-size: 0.9em; color: #6c757d; text-align: center; margin: 10px 0;">
Report generated by DesignPower • Parallel RCT Module • Binary Outcome
</p>
</div>
        """
        return report_html.strip()
        
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
        
        # Power interpretation
        if power >= 0.9:
            power_adequacy = "excellent power"
        elif power >= 0.8:
            power_adequacy = "adequate power"
        elif power >= 0.7:
            power_adequacy = "marginal power"
        else:
            power_adequacy = "insufficient power"
        
        # Method description
        if method_used == 'schoenfeld':
            method_desc = "Schoenfeld (1983) - standard log-rank test formula"
        elif method_used == 'freedman':
            method_desc = "Freedman (1982) - alternative censoring assumptions"
        elif method_used == 'lakatos':
            method_desc = "Lakatos (1988) - complex accrual patterns"
        else:
            method_desc = method_used.title()
        
        # Get appropriate reference
        reference = get_method_reference('survival', method=method, advanced_method=method_used)
        
        report_html = f"""
<div style="font-family: Arial, sans-serif; line-height: 1.6; max-width: 800px;">

<h2 style="color: #2E86AB; border-bottom: 2px solid #2E86AB; padding-bottom: 10px;">
Parallel RCT - Power Calculation Report (Survival Analysis)
</h2>

<h3 style="color: #495057;">Statistical Power Results</h3>
<div style="background-color: #e8f4fd; padding: 20px; border-radius: 8px; border-left: 4px solid #2E86AB;">
    <table style="width: 100%; border-collapse: collapse;">
    <tr>
        <td style="padding: 8px;"><strong>Statistical Power:</strong></td>
        <td style="padding: 8px; font-size: 1.2em; color: #2E86AB;"><strong>{power * 100:.1f}%</strong></td>
        <td style="padding: 8px;"><strong>Power Assessment:</strong></td>
        <td style="padding: 8px;">{power_adequacy.capitalize()}</td>
    </tr>
    <tr>
        <td style="padding: 8px;"><strong>Sample Size Group 1:</strong></td>
        <td style="padding: 8px;">{n1}</td>
        <td style="padding: 8px;"><strong>Sample Size Group 2:</strong></td>
        <td style="padding: 8px;">{n2}</td>
    </tr>
    <tr>
        <td style="padding: 8px;"><strong>Total Sample Size:</strong></td>
        <td style="padding: 8px;">{n1 + n2}</td>
        <td style="padding: 8px;"><strong>Expected Events:</strong></td>
        <td style="padding: 8px;">{events:.0f}</td>
    </tr>
    </table>
</div>

<h3 style="color: #495057;">Effect Size & Study Parameters</h3>
<div style="background-color: #f8f9fa; padding: 15px; border-radius: 8px;">
    <p><strong>Target Hazard Ratio:</strong> {hr:.3f}</p>
    <p><strong>Median Survival (Control):</strong> {median_survival1:.1f} months</p>
    <p><strong>Median Survival (Treatment):</strong> {(f'{median_survival2:.1f}') if median_survival2 != float('inf') else '∞'} months</p>
    <p><strong>Accrual Period:</strong> {accrual_time:.1f} months</p>
    <p><strong>Follow-up Period:</strong> {follow_up_time:.1f} months</p>
    <p><strong>Dropout Rate:</strong> {dropout_rate*100:.1f}%</p>
    <p><strong>Analysis Method:</strong> {method_desc}</p>
    <p><strong>Significance Level (α):</strong> {alpha:.3f}</p>
</div>

<div style="background-color: #e6f3ff; padding: 20px; border-radius: 8px; border-left: 4px solid #0066cc; margin: 20px 0;">
    <h4 style="color: #0052a3; margin-top: 0; margin-bottom: 15px;">Methodological Description</h4>
    <div style="background-color: white; padding: 15px; border-radius: 6px; border: 1px solid #cce7ff;">
        <p style="font-style: italic; line-height: 1.8; margin: 0; color: #333;">
        For a survival study with {n1} participants in group 1 and {n2} participants in group 2 
        (total N = {n1 + n2}), targeting a hazard ratio of {hr:.3f}, the estimated statistical 
        power is {power * 100:.1f}%. With median survival of {median_survival1:.1f} months in 
        the control group, this corresponds to detecting a {'reduction' if hr < 1 else 'increase'} 
        in median survival to {f'{median_survival2:.1f}' if median_survival2 != float('inf') else '∞'} months. 
        The calculation uses the {method_desc} method assuming exponential survival distributions, 
        {accrual_time:.1f} months accrual, {follow_up_time:.1f} months follow-up, and {dropout_rate*100:.1f}% 
        dropout rate. Expected number of events is {events:.0f}. Analysis will use a log-rank test 
        with Type I error rate of {alpha*100:.0f}%.
        </p>
    </div>
    <p style="font-size: 0.9em; color: #666; margin-top: 10px; margin-bottom: 0;">
    <strong>Tip:</strong> Copy the text above directly into your grant application or study protocol.
    </p>
</div>

{f'''<div style="background-color: #e8f4fd; padding: 15px; border-radius: 8px; border-left: 4px solid #2E86AB; margin: 15px 0;">
    <h4 style="color: #0052a3; margin-top: 0;">Method Comparison</h4>
    <p style="color: #0052a3;">Cross-method validation shows {results["comparison"]["max_percent_difference"]:.1f}% maximum difference between methods, 
    {"indicating excellent methodological agreement." if results["comparison"]["max_percent_difference"] < 5 else "suggesting sensitivity to study design assumptions."}</p>
</div>''' if results.get('comparison') else ''}

<div style="background-color: #fffacd; padding: 15px; border-radius: 8px; border-left: 4px solid #ffd700; margin: 15px 0;">
    <h4 style="color: #856404; margin-top: 0;">Power Interpretation</h4>
    <p style="color: #856404; margin-bottom: 10px;">
    With {power * 100:.0f}% power, this study has {power_adequacy}:
    </p>
    <ul style="color: #856404; margin-bottom: 0;">
    <li>There is a {power * 100:.0f}% probability of detecting HR = {hr:.3f} if it truly exists</li>
    <li>Type II error rate (β) = {(1-power)*100:.0f}% (probability of missing the effect)</li>
    <li>Expected {events:.0f} events provides {'good' if events > 50 else 'marginal' if events > 30 else 'limited'} precision</li>
    <li>{'Consider increasing sample size or follow-up' if power < 0.8 else 'Sample size appears adequate'}</li>
    </ul>
</div>

<h3 style="color: #495057;">Methodological References</h3>
<div style="background-color: #f8f9fa; padding: 15px; border-radius: 8px;">
    <p><strong>Primary Reference:</strong></p>
    <p style="font-style: italic; margin: 10px 0;">{reference['citation']}</p>
    <p><strong>DOI:</strong> <a href="{reference['doi']}" target="_blank" style="color: #2E86AB;">{reference['doi']}</a></p>
    <p style="margin-top: 15px;"><strong>Additional Key References:</strong></p>
    <ul style="margin-top: 10px;">
    <li>Collett D. (2015). Modelling Survival Data in Medical Research. 3rd Edition. CRC Press.</li>
    <li>Machin D, Campbell MJ, Tan SB, Tan SH. (2018). Sample Sizes for Clinical, Laboratory and Epidemiology Studies. 4th Edition. Wiley-Blackwell.</li>
    <li>Hsieh FY, Lavori PW. (2000). Sample-size calculations for the Cox proportional hazards regression model with nonbinary covariates. Controlled Clinical Trials, 21(6), 552-560.</li>
    </ul>
</div>

<div style="background-color: #d1ecf1; padding: 15px; border-radius: 8px; border-left: 4px solid #17a2b8; margin: 15px 0;">
    <h4 style="color: #0c5460; margin-top: 0;">Important Considerations</h4>
    <ul style="color: #0c5460; margin-bottom: 0;">
    <li><strong>Proportional Hazards:</strong> Results assume constant hazard ratio over time</li>
    <li><strong>Censoring Pattern:</strong> Calculations assume administrative censoring only</li>
    <li><strong>Accrual Pattern:</strong> {'Uniform accrual assumed' if not params.get('accrual_pattern') or params.get('accrual_pattern') == 'uniform' else 'Non-uniform accrual pattern applied'}</li>
    <li><strong>Sample Size Balance:</strong> {'Groups are balanced' if n1 == n2 else f'Unbalanced design (ratio {n1/n2:.2f}:1)'}</li>
    </ul>
</div>

<hr style="margin: 20px 0; border: none; border-top: 1px solid #dee2e6;">
<p style="font-size: 0.9em; color: #6c757d; text-align: center; margin: 10px 0;">
Report generated by DesignPower • Parallel RCT Module • Survival Outcome
</p>
</div>
        """
        return report_html.strip()
        
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
        sd1 = params.get('sd1', params.get('std_dev', 1))
        sd2 = params.get('sd2', params.get('std_dev2', sd1))
        
        # Get reference
        reference = get_method_reference('continuous', method=method, design=None)
        
        # Method text
        if method == "simulation":
            nsim = params.get("nsim", 1000)
            seed = params.get("seed", 42)
            method_text = f"using Monte Carlo simulation ({nsim:,} simulations, random seed {seed})"
        else:
            method_text = "using analytical methods"
        
        # Create HTML report
        report_html = f"""
<div style="font-family: Arial, sans-serif; line-height: 1.6; max-width: 800px;">

<h2 style="color: #2E86AB; border-bottom: 2px solid #2E86AB; padding-bottom: 10px;">
Parallel RCT - Minimum Detectable Effect Report
</h2>

<h3 style="color: #495057;">MDE Results</h3>
<div style="background-color: #e8f4fd; padding: 20px; border-radius: 8px; border-left: 4px solid #2E86AB;">
    <table style="width: 100%; border-collapse: collapse;">
    <tr>
        <td style="padding: 8px;"><strong>Minimum Detectable Difference:</strong></td>
        <td style="padding: 8px; font-size: 1.2em; color: #2E86AB;"><strong>{mde_val:.3f if mde_val is not None else 'N/A'} units</strong></td>
        <td style="padding: 8px;"><strong>Standardized Effect Size:</strong></td>
        <td style="padding: 8px; font-size: 1.2em; color: #2E86AB;"><strong>d = {cohen_d_val:.3f if cohen_d_val is not None else 'N/A'}</strong></td>
    </tr>
    <tr>
        <td style="padding: 8px;"><strong>Effect Size Category:</strong></td>
        <td style="padding: 8px;">{('Small' if cohen_d_val and cohen_d_val < 0.5 else 'Medium' if cohen_d_val and cohen_d_val < 0.8 else 'Large' if cohen_d_val else 'N/A')}</td>
        <td style="padding: 8px;"><strong>Analysis Method:</strong></td>
        <td style="padding: 8px;">{method_text}</td>
    </tr>
    </table>
</div>

<h3 style="color: #495057;">Study Parameters</h3>
<div style="background-color: #f8f9fa; padding: 15px; border-radius: 8px;">
    <p><strong>Sample Size Group 1:</strong> {n1}</p>
    <p><strong>Sample Size Group 2:</strong> {n2}</p>
    <p><strong>Total Sample Size:</strong> {n1 + n2}</p>
    <p><strong>Standard Deviation:</strong> {sd1:.2f}{f' (unequal variances: SD₁={sd1:.2f}, SD₂={sd2:.2f})' if sd1 != sd2 else ''}</p>
    <p><strong>Target Power:</strong> {power * 100:.0f}%</p>
    <p><strong>Significance Level (α):</strong> {alpha:.3f}</p>
</div>

<div style="background-color: #e6f3ff; padding: 20px; border-radius: 8px; border-left: 4px solid #0066cc; margin: 20px 0;">
    <h4 style="color: #0052a3; margin-top: 0; margin-bottom: 15px;">Methodological Description</h4>
    <div style="background-color: white; padding: 15px; border-radius: 6px; border: 1px solid #cce7ff;">
        <p style="font-style: italic; line-height: 1.8; margin: 0; color: #333;">
        For a study with {n1} participants in group 1 and {n2} in group 2 (total {n1+n2}), 
        aiming for {power * 100:.0f}% statistical power, the minimum detectable difference in means is 
        {mde_val:.3f if mde_val is not None else 'N/A'} units. This corresponds to a standardized effect size 
        (Cohen's d) of {cohen_d_val:.3f if cohen_d_val is not None else 'N/A'}. The calculation assumes 
        {f'equal variances with standard deviation {sd1:.2f}' if sd1 == sd2 else f'unequal variances (SD₁={sd1:.2f}, SD₂={sd2:.2f})'} 
        and was performed {method_text} with a Type I error rate of {alpha*100:.0f}%.
        </p>
    </div>
    <p style="font-size: 0.9em; color: #666; margin-top: 10px; margin-bottom: 0;">
    <strong>Tip:</strong> Copy the text above directly into your grant application or study protocol.
    </p>
</div>

<div style="background-color: #fffacd; padding: 15px; border-radius: 8px; border-left: 4px solid #ffd700; margin: 15px 0;">
    <h4 style="color: #856404; margin-top: 0;">MDE Interpretation</h4>
    <p style="color: #856404; margin-bottom: 10px;">
    Understanding your minimum detectable effect:
    </p>
    <ul style="color: #856404; margin-bottom: 0;">
    <li>The study can reliably detect differences of {mde_val:.3f if mde_val is not None else 'N/A'} units or larger</li>
    <li>Smaller differences may exist but would not be detected with {power*100:.0f}% power</li>
    <li>Effect size d = {cohen_d_val:.3f if cohen_d_val is not None else 'N/A'} is {('small (may require larger sample for clinical relevance)' if cohen_d_val and cohen_d_val < 0.5 else 'medium (typically clinically meaningful)' if cohen_d_val and cohen_d_val < 0.8 else 'large (easily detectable)' if cohen_d_val else 'not calculated')}</li>
    <li>Consider if this MDE aligns with clinically important differences</li>
    </ul>
</div>

<h3 style="color: #495057;">Methodological References</h3>
<div style="background-color: #f8f9fa; padding: 15px; border-radius: 8px;">
    <p><strong>Primary Reference:</strong></p>
    <p style="font-style: italic; margin: 10px 0;">{reference['citation']}</p>
    <p><strong>DOI:</strong> <a href="{reference['doi']}" target="_blank" style="color: #2E86AB;">{reference['doi']}</a></p>
    <p style="margin-top: 15px;"><strong>Additional Key References:</strong></p>
    <ul style="margin-top: 10px;">
    <li>Cohen J. (1992). A power primer. Psychological Bulletin, 112(1), 155-159.</li>
    <li>Faul F, Erdfelder E, Lang AG, Buchner A. (2007). G*Power 3: A flexible statistical power analysis program for the social, behavioral, and biomedical sciences. Behavior Research Methods, 39(2), 175-191.</li>
    <li>Lipsey MW, Wilson DB. (1993). The efficacy of psychological, educational, and behavioral treatment: Confirmation from meta-analysis. American Psychologist, 48(12), 1181-1209.</li>
    </ul>
</div>

<div style="background-color: #d1ecf1; padding: 15px; border-radius: 8px; border-left: 4px solid #17a2b8; margin: 15px 0;">
    <h4 style="color: #0c5460; margin-top: 0;">Important Considerations</h4>
    <ul style="color: #0c5460; margin-bottom: 0;">
    <li><strong>Clinical vs Statistical Significance:</strong> Ensure MDE is clinically meaningful</li>
    <li><strong>Sample Size Trade-offs:</strong> Larger samples detect smaller effects</li>
    <li><strong>Publication Bias:</strong> Studies may miss important small effects</li>
    <li><strong>Effect Size Guidelines:</strong> Cohen's d: 0.2=small, 0.5=medium, 0.8=large</li>
    </ul>
</div>

<hr style="margin: 20px 0; border: none; border-top: 1px solid #dee2e6;">
<p style="font-size: 0.9em; color: #6c757d; text-align: center; margin: 10px 0;">
Report generated by DesignPower • Parallel RCT Module • Continuous Outcome
</p>
</div>
        """
        return report_html.strip()
    elif design_type == 'Parallel RCT' and 'Binary' in outcome_type:
        # Binary MDE reporting
        mde_val = results.get('mde')
        p1 = params.get('p1', 0.5)
        test_type = params.get('test_type', 'Normal Approximation')
        
        # Get reference
        reference = get_method_reference('binary', test_type, method)
        
        # Calculate effect measures
        if mde_val is not None:
            p2_mde = p1 + mde_val
            risk_diff = abs(mde_val)
            if p1 > 0:
                relative_risk = p2_mde / p1
            else:
                relative_risk = None
            if p1 > 0 and p1 < 1 and p2_mde < 1 and p2_mde > 0:
                odds_ratio = (p2_mde / (1 - p2_mde)) / (p1 / (1 - p1))
            else:
                odds_ratio = None
        else:
            p2_mde = risk_diff = relative_risk = odds_ratio = None
        
        # Method text
        if method == "simulation":
            nsim = params.get("nsim", 1000)
            seed = params.get("seed", 42)
            method_text = f"using Monte Carlo simulation ({nsim:,} simulations, random seed {seed})"
        else:
            method_text = f"using {test_type}"
        
        # Create HTML report
        report_html = f"""
<div style="font-family: Arial, sans-serif; line-height: 1.6; max-width: 800px;">

<h2 style="color: #2E86AB; border-bottom: 2px solid #2E86AB; padding-bottom: 10px;">
Parallel RCT - Minimum Detectable Effect Report
</h2>

<h3 style="color: #495057;">MDE Results</h3>
<div style="background-color: #e8f4fd; padding: 20px; border-radius: 8px; border-left: 4px solid #2E86AB;">
    <table style="width: 100%; border-collapse: collapse;">
    <tr>
        <td style="padding: 8px;"><strong>Minimum Detectable Risk Difference:</strong></td>
        <td style="padding: 8px; font-size: 1.2em; color: #2E86AB;"><strong>{f'{risk_diff:.3f}' if risk_diff is not None else 'N/A'} ({f'{risk_diff*100:.1f}' if risk_diff is not None else 'N/A'} percentage points)</strong></td>
    </tr>
    <tr>
        <td style="padding: 8px;"><strong>Control Group Proportion:</strong></td>
        <td style="padding: 8px;">{p1:.3f} ({p1*100:.1f}%)</td>
        <td style="padding: 8px;"><strong>Detectable Treatment Proportion:</strong></td>
        <td style="padding: 8px;">{f'{p2_mde:.3f} ({p2_mde*100:.1f}%)' if p2_mde is not None else 'N/A'}</td>
    </tr>
    <tr>
        <td style="padding: 8px;"><strong>Minimum Detectable Odds Ratio:</strong></td>
        <td style="padding: 8px;">{f'{odds_ratio:.3f}' if odds_ratio is not None else 'N/A'}</td>
        <td style="padding: 8px;"><strong>Minimum Detectable Risk Ratio:</strong></td>
        <td style="padding: 8px;">{f'{relative_risk:.3f}' if relative_risk is not None else 'N/A'}</td>
    </tr>
    </table>
</div>

<h3 style="color: #495057;">Study Parameters</h3>
<div style="background-color: #f8f9fa; padding: 15px; border-radius: 8px;">
    <p><strong>Sample Size Group 1:</strong> {n1}</p>
    <p><strong>Sample Size Group 2:</strong> {n2}</p>
    <p><strong>Total Sample Size:</strong> {n1 + n2}</p>
    <p><strong>Target Power:</strong> {power * 100:.0f}%</p>
    <p><strong>Significance Level (α):</strong> {alpha:.3f}</p>
    <p><strong>Analysis Method:</strong> {method_text}</p>
</div>

<div style="background-color: #e6f3ff; padding: 20px; border-radius: 8px; border-left: 4px solid #0066cc; margin: 20px 0;">
    <h4 style="color: #0052a3; margin-top: 0; margin-bottom: 15px;">Methodological Description</h4>
    <div style="background-color: white; padding: 15px; border-radius: 6px; border: 1px solid #cce7ff;">
        <p style="font-style: italic; line-height: 1.8; margin: 0; color: #333;">
        For a study with {n1} participants in group 1 and {n2} in group 2 (total {n1+n2}), 
        aiming for {power * 100:.0f}% statistical power, the minimum detectable risk difference is 
        {f'{risk_diff:.3f}' if risk_diff is not None else 'N/A'} ({f'{risk_diff*100:.1f}' if risk_diff is not None else 'N/A'} percentage points). 
        With a control group proportion of {p1:.3f}, the study can detect changes to 
        {f'{p2_mde:.3f}' if p2_mde is not None else 'N/A'} or beyond. The calculation was performed 
        {method_text} with a Type I error rate of {alpha*100:.0f}%.
        </p>
    </div>
    <p style="font-size: 0.9em; color: #666; margin-top: 10px; margin-bottom: 0;">
    <strong>Tip:</strong> Copy the text above directly into your grant application or study protocol.
    </p>
</div>

<div style="background-color: #fffacd; padding: 15px; border-radius: 8px; border-left: 4px solid #ffd700; margin: 15px 0;">
    <h4 style="color: #856404; margin-top: 0;">MDE Interpretation</h4>
    <p style="color: #856404; margin-bottom: 10px;">
    Understanding your minimum detectable effect:
    </p>
    <ul style="color: #856404; margin-bottom: 0;">
    <li>The study can detect differences of {f'{risk_diff*100:.1f}' if risk_diff is not None else 'N/A'} percentage points or larger</li>
    <li>Number Needed to Treat (NNT) for MDE: {int(1/risk_diff) if risk_diff and risk_diff > 0 else 'N/A'}</li>
    <li>{'This is a clinically meaningful difference' if risk_diff and risk_diff > 0.1 else 'Consider if this difference is clinically meaningful' if risk_diff else ''}</li>
    <li>Smaller differences may exist but would require larger sample sizes</li>
    </ul>
</div>

<h3 style="color: #495057;">Methodological References</h3>
<div style="background-color: #f8f9fa; padding: 15px; border-radius: 8px;">
    <p><strong>Primary Reference:</strong></p>
    <p style="font-style: italic; margin: 10px 0;">{reference['citation']}</p>
    <p><strong>DOI:</strong> <a href="{reference['doi']}" target="_blank" style="color: #2E86AB;">{reference['doi']}</a></p>
    <p style="margin-top: 15px;"><strong>Additional Key References:</strong></p>
    <ul style="margin-top: 10px;">
    <li>Pocock SJ. (1983). Clinical Trials: A Practical Approach. John Wiley & Sons.</li>
    <li>Sealed Envelope Ltd. (2012). Power calculators for binary outcome superiority trial. Available from: https://www.sealedenvelope.com/power/binary-superiority/</li>
    <li>Noordzij M, Tripepi G, Dekker FW, Zoccali C, Tanck MW, Jager KJ. (2010). Sample size calculations: basic principles and common pitfalls. Nephrology Dialysis Transplantation, 25(5), 1388-1393.</li>
    </ul>
</div>

<div style="background-color: #d1ecf1; padding: 15px; border-radius: 8px; border-left: 4px solid #17a2b8; margin: 15px 0;">
    <h4 style="color: #0c5460; margin-top: 0;">Important Considerations</h4>
    <ul style="color: #0c5460; margin-bottom: 0;">
    <li><strong>Base Rate:</strong> MDE depends on control group proportion ({p1*100:.1f}%)</li>
    <li><strong>Clinical Relevance:</strong> Statistical detectability ≠ clinical importance</li>
    <li><strong>Test Assumptions:</strong> {test_type} {'appropriate for these sample sizes' if n1 + n2 > 40 else 'may need exact methods for small samples'}</li>
    <li><strong>Multiple Comparisons:</strong> Adjust α if testing multiple endpoints</li>
    </ul>
</div>

<hr style="margin: 20px 0; border: none; border-top: 1px solid #dee2e6;">
<p style="font-size: 0.9em; color: #6c757d; text-align: center; margin: 10px 0;">
Report generated by DesignPower • Parallel RCT Module • Binary Outcome
</p>
</div>
        """
        return report_html.strip()
        
    elif design_type == 'Parallel RCT' and 'Survival' in outcome_type:
        # Enhanced survival MDE reporting
        mde_hr = results.get('mde')
        median_survival1 = params.get('median_survival1', 12.0)
        advanced_method = params.get('advanced_method', 'schoenfeld')
        events = results.get('events', 0)
        accrual_time = params.get('accrual_time', 12.0)
        follow_up_time = params.get('follow_up_time', 24.0)
        dropout_rate = params.get('dropout_rate', 0.1)
        
        # Get reference
        reference = get_method_reference('survival', method=method, advanced_method=advanced_method)
        
        if mde_hr:
            median_survival2_mde = median_survival1 / mde_hr if mde_hr > 0 else float('inf')
            
            # Create HTML report
            report_html = f"""
<div style="font-family: Arial, sans-serif; line-height: 1.6; max-width: 800px;">

<h2 style="color: #2E86AB; border-bottom: 2px solid #2E86AB; padding-bottom: 10px;">
Parallel RCT - Minimum Detectable Effect Report (Survival Analysis)
</h2>

<h3 style="color: #495057;">MDE Results</h3>
<div style="background-color: #e8f4fd; padding: 20px; border-radius: 8px; border-left: 4px solid #2E86AB;">
    <table style="width: 100%; border-collapse: collapse;">
    <tr>
        <td style="padding: 8px;"><strong>Minimum Detectable Hazard Ratio:</strong></td>
        <td style="padding: 8px; font-size: 1.2em; color: #2E86AB;"><strong>{mde_hr:.3f}</strong></td>
        <td style="padding: 8px;"><strong>Expected Events:</strong></td>
        <td style="padding: 8px; font-size: 1.1em;">{events:.0f}</td>
    </tr>
    <tr>
        <td style="padding: 8px;"><strong>Control Median Survival:</strong></td>
        <td style="padding: 8px;">{median_survival1:.1f} months</td>
        <td style="padding: 8px;"><strong>Detectable Treatment Median:</strong></td>
        <td style="padding: 8px;">{f'{median_survival2_mde:.1f}' if median_survival2_mde != float('inf') else '∞'} months</td>
    </tr>
    <tr>
        <td style="padding: 8px;"><strong>Survival Improvement:</strong></td>
        <td style="padding: 8px;" colspan="3">{f'{(median_survival2_mde/median_survival1 - 1)*100:.1f}%' if median_survival2_mde != float('inf') and median_survival1 > 0 else 'N/A'}</td>
    </tr>
    </table>
</div>

<h3 style="color: #495057;">Study Parameters</h3>
<div style="background-color: #f8f9fa; padding: 15px; border-radius: 8px;">
    <p><strong>Sample Size Group 1:</strong> {n1}</p>
    <p><strong>Sample Size Group 2:</strong> {n2}</p>
    <p><strong>Total Sample Size:</strong> {n1 + n2}</p>
    <p><strong>Target Power:</strong> {power * 100:.0f}%</p>
    <p><strong>Significance Level (α):</strong> {alpha:.3f}</p>
    <p><strong>Accrual Period:</strong> {accrual_time:.1f} months</p>
    <p><strong>Follow-up Period:</strong> {follow_up_time:.1f} months</p>
    <p><strong>Dropout Rate:</strong> {dropout_rate*100:.1f}%</p>
    <p><strong>Analysis Method:</strong> {advanced_method.title()} method</p>
</div>

<div style="background-color: #e6f3ff; padding: 20px; border-radius: 8px; border-left: 4px solid #0066cc; margin: 20px 0;">
    <h4 style="color: #0052a3; margin-top: 0; margin-bottom: 15px;">Methodological Description</h4>
    <div style="background-color: white; padding: 15px; border-radius: 6px; border: 1px solid #cce7ff;">
        <p style="font-style: italic; line-height: 1.8; margin: 0; color: #333;">
        For a study with {n1} participants in group 1 and {n2} in group 2 (total {n1+n2}), 
        aiming for {power * 100:.0f}% statistical power, the minimum detectable hazard ratio is {mde_hr:.3f}. 
        This corresponds to detecting a difference between median survival times of {median_survival1:.1f} months 
        (control) versus {f'{median_survival2_mde:.1f}' if median_survival2_mde != float('inf') else '∞'} months (treatment). 
        Expected number of events is {events:.0f}. The calculation uses the {advanced_method.title()} method 
        with exponential survival assumptions, {accrual_time:.1f} months accrual, {follow_up_time:.1f} months 
        follow-up, {dropout_rate*100:.1f}% dropout rate, and a Type I error rate of {alpha*100:.0f}%.
        </p>
    </div>
    <p style="font-size: 0.9em; color: #666; margin-top: 10px; margin-bottom: 0;">
    <strong>Tip:</strong> Copy the text above directly into your grant application or study protocol.
    </p>
</div>

<div style="background-color: #fffacd; padding: 15px; border-radius: 8px; border-left: 4px solid #ffd700; margin: 15px 0;">
    <h4 style="color: #856404; margin-top: 0;">MDE Interpretation</h4>
    <p style="color: #856404; margin-bottom: 10px;">
    Understanding your minimum detectable effect:
    </p>
    <ul style="color: #856404; margin-bottom: 0;">
    <li>The study can detect HR ≤ {mde_hr:.3f} (or ≥ {1/mde_hr:.3f if mde_hr > 0 else 'N/A'})</li>
    <li>Corresponds to {f'{(1-mde_hr)*100:.1f}%' if mde_hr < 1 else f'{(mde_hr-1)*100:.1f}% increase' if mde_hr > 1 else '0%'} reduction in hazard</li>
    <li>Expected {events:.0f} events provides {'good' if events > 50 else 'marginal' if events > 30 else 'limited'} precision</li>
    <li>{'Consider longer follow-up for more events' if events < 50 else 'Event count appears adequate'}</li>
    </ul>
</div>

<h3 style="color: #495057;">Methodological References</h3>
<div style="background-color: #f8f9fa; padding: 15px; border-radius: 8px;">
    <p><strong>Primary Reference:</strong></p>
    <p style="font-style: italic; margin: 10px 0;">{reference['citation']}</p>
    <p><strong>DOI:</strong> <a href="{reference['doi']}" target="_blank" style="color: #2E86AB;">{reference['doi']}</a></p>
    <p style="margin-top: 15px;"><strong>Additional Key References:</strong></p>
    <ul style="margin-top: 10px;">
    <li>Collett D. (2015). Modelling Survival Data in Medical Research. 3rd Edition. CRC Press.</li>
    <li>Machin D, Campbell MJ, Tan SB, Tan SH. (2018). Sample Sizes for Clinical, Laboratory and Epidemiology Studies. 4th Edition. Wiley-Blackwell.</li>
    <li>Latouche A, Porcher R, Chevret S. (2004). Sample size formula for proportional hazards modelling of competing risks. Statistics in Medicine, 23(21), 3263-3274.</li>
    </ul>
</div>

<div style="background-color: #d1ecf1; padding: 15px; border-radius: 8px; border-left: 4px solid #17a2b8; margin: 15px 0;">
    <h4 style="color: #0c5460; margin-top: 0;">Important Considerations</h4>
    <ul style="color: #0c5460; margin-bottom: 0;">
    <li><strong>Proportional Hazards:</strong> MDE assumes constant HR over time</li>
    <li><strong>Censoring Impact:</strong> Heavy censoring reduces detectable effects</li>
    <li><strong>Event-Driven:</strong> Power depends on events, not just sample size</li>
    <li><strong>Clinical Relevance:</strong> HR of {mde_hr:.3f} {'is a modest effect' if 0.7 < mde_hr < 1.3 else 'is a substantial effect'}</li>
    </ul>
</div>

<hr style="margin: 20px 0; border: none; border-top: 1px solid #dee2e6;">
<p style="font-size: 0.9em; color: #6c757d; text-align: center; margin: 10px 0;">
Report generated by DesignPower • Parallel RCT Module • Survival Outcome
</p>
</div>
            """
            return report_html.strip()
        else:
            return "Minimum detectable effect calculation for survival outcomes is not available."
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
    Stepped Wedge Cluster Randomized Trial - Power Analysis Report
    </h2>
    
    <h3 style="color: #495057;">Power Analysis Results</h3>
    <div style="background-color: #e8f4fd; padding: 15px; border-radius: 8px; border-left: 4px solid #2E86AB;">
    <p><strong>Statistical Power:</strong> <span style="font-size: 1.2em; color: #2E86AB; font-weight: bold;">{power:.1%}</span></p>
    <p><strong>Significance Level (α):</strong> {alpha:.3f}</p>
    <p><strong>Expected Effect:</strong> {effect_text}</p>
    <p><strong>Analysis Method:</strong> {method} {f'({nsim} simulations)' if method.lower() == 'simulation' else ''}</p>
    </div>
    
    <div style="background-color: #e6f3ff; padding: 20px; border-radius: 8px; border-left: 4px solid #0066cc; margin: 20px 0;">
    <h4 style="color: #0052a3; margin-top: 0; margin-bottom: 15px;">Methodological Description</h4>
    <div style="background-color: white; padding: 15px; border-radius: 6px; border: 1px solid #cce7ff;">
    <p style="font-style: italic; line-height: 1.8; margin: 0; color: #333;">
    A sample size of {individuals_per_cluster} individuals per cluster per time period, from each of {clusters} clusters, 
    stepping every time period across {steps} time steps (including baseline), will give the study {power:.0%} power to 
    detect a {f'treatment effect of {treatment_effect:.2f} units' if 'continuous' in outcome_type.lower() else f'change from {p_control:.2f} to {p_intervention:.2f} in the proportion'}, 
    assuming {f'a standard deviation of {std_dev:.2f}' if 'continuous' in outcome_type.lower() else 'the stated proportions'}, an intracluster correlation coefficient (ICC) of {icc:.3f}{f', and a cluster autocorrelation coefficient (CAC) of {cluster_autocorr:.3f}' if cluster_autocorr > 0 else ''}, 
    with a Type I error rate of {alpha*100:.0f}%. Power calculations were performed using {'analytical methods' if method_used == 'Hussey & Hughes Analytical' else 'simulation-based methods'} as 
    described by {reference['citation'].split('.')[0]} et al.
    </p>
    </div>
    <p style="font-size: 0.9em; color: #666; margin-top: 10px; margin-bottom: 0;">
    <strong>Tip:</strong> Copy the text above directly into your grant application or study protocol.
    </p>
    </div>
    
    <h3 style="color: #495057;">Design Efficiency Metrics</h3>
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
    
    
    <h3 style="color: #495057;">Methodological Reference</h3>
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
    <h4 style="color: #0c5460; margin-top: 0;">Important Considerations</h4>
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
    Interrupted Time Series - Power Analysis Report
    </h2>
    
    <h3 style="color: #495057;">Analysis Results</h3>
    <div style="background-color: #e8f4fd; padding: 15px; border-radius: 8px; border-left: 4px solid #2E86AB;">
    {result_text}
    <p><strong>Significance Level (α):</strong> {alpha:.3f}</p>
    <p><strong>Expected Effect:</strong> {effect_text}</p>
    <p><strong>Analysis Method:</strong> {method}</p>
    </div>
    
    <div style="background-color: #e6f3ff; padding: 20px; border-radius: 8px; border-left: 4px solid #0066cc; margin: 20px 0;">
    <h4 style="color: #0052a3; margin-top: 0; margin-bottom: 15px;">Methodological Description</h4>
    <div style="background-color: white; padding: 15px; border-radius: 6px; border: 1px solid #cce7ff;">
    <p style="font-style: italic; line-height: 1.8; margin: 0; color: #333;">
    {f'A sample size of {n_pre} pre-intervention and {n_post} post-intervention time points (total {total_n} observations) will provide {int(power*100)}% power to detect a {effect_text}' if calculation_type != "Sample Size" else f'To achieve {int(params.get("power", 0.8)*100)}% power to detect a {effect_text}, a minimum of {n_pre} pre-intervention and {n_post} post-intervention time points (total {total_n} observations) is required'}, 
    {f'assuming an autocorrelation coefficient of {autocorr:.3f}, ' if autocorr > 0 else ''}with a Type I error rate of {alpha*100:.0f}%. 
    Power calculations were performed using {'segmented regression analysis' if method.lower() == 'analytical' else 'simulation-based methods'} as described by {reference['citation'].split('.')[0]} et al.
    </p>
    </div>
    <p style="font-size: 0.9em; color: #666; margin-top: 10px; margin-bottom: 0;">
    <strong>Tip:</strong> Copy the text above directly into your grant application or study protocol.
    </p>
    </div>
    
    
    <h3 style="color: #495057;">Statistical Considerations</h3>
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
    
    <h3 style="color: #495057;">Methodological Reference</h3>
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
    <h4 style="color: #0c5460; margin-top: 0;">Key Assumptions & Limitations</h4>
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
