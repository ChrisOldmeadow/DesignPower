"""
Module for generating reports specifically for Cluster RCT designs.

Provides enhanced HTML reports with comprehensive methodological descriptions,
design effect explanations, and practical interpretation aids.
"""
import textwrap

def generate_cluster_report(results, params):
    """
    Generate appropriate report for Cluster RCT designs based on calculation type.
    
    Parameters
    ----------
    results : dict
        Results from the calculation function
    params : dict
        Input parameters used for the calculation
        
    Returns
    -------
    str
        Formatted HTML report with enhanced design interpretation
    """
    # Determine the calculation type
    calc_type = params.get("calc_type", params.get("calculation_type", "Sample Size"))
    outcome_type = params.get("outcome_type", "Continuous Outcome")
    
    # Generate appropriate report based on calculation type
    if calc_type == "Sample Size":
        return generate_cluster_sample_size_report(results, params, outcome_type)
    elif calc_type == "Power":
        return generate_cluster_power_report(results, params, outcome_type)
    elif calc_type == "Minimum Detectable Effect":
        return generate_cluster_mde_report(results, params, outcome_type)
    else:
        return "No report available for this calculation type."

def generate_cluster_sample_size_report(results, params, outcome_type):
    """
    Generate an enhanced HTML report for Cluster RCT sample size calculations.
    
    Provides comprehensive methodological context, design effect interpretation,
    and practical guidance for cluster randomized trials.
    """
    # Extract shared parameters
    determine_ss_param = params.get("determine_ss_param")
    icc = params.get('icc', 0)

    if determine_ss_param == "Average Cluster Size (m)":
        # We solved for cluster_size (m), so n_clusters (k) was an input
        # Input k is params['n_clusters_input_for_m_calc'] from UI, or results['n_clusters_fixed'] from calculation
        report_n_clusters = params.get('n_clusters_input_for_m_calc', results.get('n_clusters_fixed', 0))
        report_cluster_size = results.get('cluster_size', 0)
    elif determine_ss_param == "Number of Clusters (k)":
        # We solved for n_clusters (k), so cluster_size (m) was an input
        # Input m is params['cluster_size_input_for_k_calc'] from UI, or results['cluster_size_fixed'] from calculation
        report_n_clusters = results.get('n_clusters', 0)
        report_cluster_size = params.get('cluster_size_input_for_k_calc', results.get('cluster_size_fixed', 0))
    else:
        # Fallback for older calls or if determine_ss_param is not set
        report_n_clusters = results.get('n_clusters', 0)
        # Try to get calculated cluster_size from results first, then input from params
        report_cluster_size = results.get('cluster_size', params.get('cluster_size', 0)) 

    # Ensure values are integers for display and DE calculation if they are numbers
    report_n_clusters = int(report_n_clusters) if isinstance(report_n_clusters, (int, float)) and report_n_clusters is not None else 0
    report_cluster_size = int(report_cluster_size) if isinstance(report_cluster_size, (int, float)) and report_cluster_size is not None else 0

    total_n = report_n_clusters * 2 * report_cluster_size
    # Calculate design_effect using the appropriate cluster_size for the report
    # If report_cluster_size is 0 or 1, DE is 1. Avoid (0-1)*icc or (1-1)*icc.
    if report_cluster_size > 1:
        design_effect = results.get('design_effect', 1 + (report_cluster_size - 1) * icc)
    else:
        design_effect = 1.0
    alpha = params.get('alpha', 0.05)
    power = params.get('power', 0.8)
    method = params.get('method', 'analytical')
    
    # Method text based on analytical or simulation
    if method == "simulation":
        nsim = params.get("nsim", 1000)
        seed = params.get("seed", 42)
        method_text = f"using Monte Carlo simulation ({nsim:,} simulations, random seed {seed})"
    else:
        method_text = "using analytical methods"
    
    if "Continuous" in outcome_type:
        # Extract continuous-specific parameters
        mean1 = params.get('mean1', 0)
        mean2 = params.get('mean2', 0)
        std_dev = params.get('std_dev', 0)
        difference = abs(mean2 - mean1)
        effect_size = difference / std_dev if std_dev > 0 else 0
        
        # Calculate effective sample size
        effective_n = total_n / design_effect if design_effect > 0 else total_n
        
        # ICC interpretation
        if icc < 0.01:
            icc_level = "negligible"
            icc_impact = "minimal impact on required sample size"
        elif icc < 0.05:
            icc_level = "small"
            icc_impact = "moderate increase in required sample size"
        elif icc < 0.1:
            icc_level = "moderate"
            icc_impact = "substantial increase in required sample size"
        else:
            icc_level = "large"
            icc_impact = "major increase in required sample size"
            
        # Method comparison if simulation
        method_comparison = ""
        if method == "simulation" and 'analytical_comparison' in results:
            analytical_n = results['analytical_comparison'].get('n_clusters', report_n_clusters)
            if analytical_n != report_n_clusters:
                diff_pct = abs(analytical_n - report_n_clusters) / analytical_n * 100
                method_comparison = f"""\n\n<div style="background-color: #e8f4fd; padding: 15px; border-radius: 8px; border-left: 4px solid #2E86AB; margin-top: 15px;">
<strong>Method Comparison:</strong> Analytical method yields {analytical_n} clusters per arm. 
Simulation differs by {diff_pct:.1f}%, likely due to the discrete nature of cluster allocation.
</div>"""
        
        report_html = f"""
<div style="font-family: Arial, sans-serif; line-height: 1.6; max-width: 800px;">

<h2 style="color: #2E86AB; border-bottom: 2px solid #2E86AB; padding-bottom: 10px;">
Cluster RCT - Sample Size Calculation Report
</h2>

<h3 style="color: #495057;">Required Sample Size</h3>
<div style="background-color: #e8f4fd; padding: 20px; border-radius: 8px; border-left: 4px solid #2E86AB;">
    <table style="width: 100%; border-collapse: collapse;">
    <tr>
        <td style="padding: 8px;"><strong>Clusters per Arm:</strong></td>
        <td style="padding: 8px; font-size: 1.2em; color: #2E86AB;"><strong>{report_n_clusters}</strong></td>
        <td style="padding: 8px;"><strong>Cluster Size:</strong></td>
        <td style="padding: 8px; font-size: 1.2em; color: #2E86AB;"><strong>{report_cluster_size}</strong></td>
    </tr>
    <tr>
        <td style="padding: 8px;"><strong>Total Clusters:</strong></td>
        <td style="padding: 8px;">{report_n_clusters * 2}</td>
        <td style="padding: 8px;"><strong>Total Sample Size:</strong></td>
        <td style="padding: 8px; font-size: 1.1em;"><strong>{total_n}</strong></td>
    </tr>
    </table>
</div>

<h3 style="color: #495057;">Effect Size & Study Parameters</h3>
<div style="background-color: #f8f9fa; padding: 15px; border-radius: 8px;">
    <p><strong>Target Difference:</strong> {difference:.2f} units (Mean₁ = {mean1:.2f}, Mean₂ = {mean2:.2f})</p>
    <p><strong>Standardized Effect Size:</strong> Cohen's d = {effect_size:.3f}</p>
    <p><strong>Standard Deviation:</strong> {std_dev:.2f}</p>
    <p><strong>Statistical Power:</strong> {power * 100:.0f}%</p>
    <p><strong>Significance Level (α):</strong> {alpha:.3f}</p>
    <p><strong>Analysis Method:</strong> {method_text}</p>
</div>

<h3 style="color: #495057;">Clustering Effects</h3>
<div style="background-color: #e6f3ff; padding: 15px; border-radius: 8px; border-left: 4px solid #0066cc;">
    <p><strong>Intracluster Correlation (ICC):</strong> {icc:.3f} ({icc_level})</p>
    <p><strong>Design Effect (DEFF):</strong> {design_effect:.2f}</p>
    <p><strong>Effective Sample Size:</strong> {effective_n:.0f} individuals</p>
    <p style="font-style: italic; color: #0066cc;">
    The ICC of {icc:.3f} indicates {icc_level} clustering with {icc_impact}. 
    The design effect of {design_effect:.2f} means this cluster design requires {design_effect:.1f}x 
    more participants than an individually randomized trial.
    </p>
</div>

<div style="background-color: #e6f3ff; padding: 20px; border-radius: 8px; border-left: 4px solid #0066cc; margin: 20px 0;">
    <h4 style="color: #0052a3; margin-top: 0; margin-bottom: 15px;">Methodological Description</h4>
    <div style="background-color: white; padding: 15px; border-radius: 6px; border: 1px solid #cce7ff;">
        <p style="font-style: italic; line-height: 1.8; margin: 0; color: #333;">
        A cluster randomized controlled trial with {report_n_clusters} clusters per arm and {report_cluster_size} individuals per cluster 
        (total N = {total_n}) will provide {power * 100:.0f}% power to detect a difference of {difference:.2f} units between 
        treatment arms (standardized effect size d = {effect_size:.3f}), assuming a within-cluster standard deviation of {std_dev:.2f} 
        and an intracluster correlation coefficient of {icc:.3f}. The design effect of {design_effect:.2f} accounts for the 
        loss of statistical efficiency due to clustering. Power calculations were performed {method_text} with a 
        Type I error rate of {alpha*100:.0f}%.
        </p>
    </div>
    <p style="font-size: 0.9em; color: #666; margin-top: 10px; margin-bottom: 0;">
    <strong>Tip:</strong> Copy the text above directly into your grant application or study protocol.
    </p>
</div>
{method_comparison}

<h3 style="color: #495057;">Methodological References</h3>
<div style="background-color: #f8f9fa; padding: 15px; border-radius: 8px;">
    <p><strong>Primary Reference:</strong></p>
    <p style="font-style: italic; margin: 10px 0;">Donner A, Klar N. (2000). Design and Analysis of Cluster Randomization Trials in Health Research. London: Arnold.</p>
    <p><strong>DOI:</strong> <a href="https://doi.org/10.1002/sim.836" target="_blank" style="color: #2E86AB;">https://doi.org/10.1002/sim.836</a></p>
    
    <p style="margin-top: 15px;"><strong>Additional Key References:</strong></p>
    <ul style="margin-top: 10px;">
    <li>Murray DM. (1998). Design and Analysis of Group-Randomized Trials. Oxford University Press.</li>
    <li>Eldridge SM, Kerry S. (2012). A Practical Guide to Cluster Randomised Trials in Health Services Research. Wiley.</li>
    <li>Campbell MK, Piaggio G, Elbourne DR, Altman DG. (2012). Consort 2010 statement: extension to cluster randomised trials. BMJ, 345:e5661.</li>
    </ul>
</div>

<div style="background-color: #d1ecf1; padding: 15px; border-radius: 8px; border-left: 4px solid #17a2b8; margin: 15px 0;">
    <h4 style="color: #0c5460; margin-top: 0;">Important Considerations</h4>
    <ul style="color: #0c5460; margin-bottom: 0;">
    <li><strong>ICC Estimation:</strong> Use pilot data or literature values for similar populations and outcomes</li>
    <li><strong>Cluster Size Variation:</strong> Unequal cluster sizes inflate the design effect further</li>
    <li><strong>Attrition:</strong> Account for both individual and cluster-level dropout</li>
    <li><strong>Analysis Method:</strong> Use appropriate multilevel or GEE models in the final analysis</li>
    </ul>
</div>

<hr style="margin: 20px 0; border: none; border-top: 1px solid #dee2e6;">
<p style="font-size: 0.9em; color: #6c757d; text-align: center; margin: 10px 0;">
Report generated by DesignPower • Cluster RCT Module • Continuous Outcome
</p>
</div>
        """
        return report_html
    elif "Binary" in outcome_type:
        # Extract binary-specific parameters
        p1 = params.get('p1', 0)
        p2 = params.get('p2', 0)
        risk_diff = abs(p2 - p1)
        risk_ratio = p2 / p1 if p1 > 0 else float('inf')
        odds_ratio = (p2/(1-p2)) / (p1/(1-p1)) if p1 > 0 and p1 < 1 and p2 < 1 else float('inf')
        
        # Calculate effective sample size
        effective_n = total_n / design_effect if design_effect > 0 else total_n
        
        # ICC interpretation for binary outcomes
        if icc < 0.01:
            icc_level = "negligible"
            icc_impact = "minimal impact on required sample size"
        elif icc < 0.05:
            icc_level = "small"
            icc_impact = "moderate increase in required sample size"
        elif icc < 0.1:
            icc_level = "moderate"
            icc_impact = "substantial increase in required sample size"
        else:
            icc_level = "large"
            icc_impact = "major increase in required sample size"
            
        # Method comparison if simulation
        method_comparison = ""
        if method == "simulation" and 'analytical_comparison' in results:
            analytical_n = results['analytical_comparison'].get('n_clusters', report_n_clusters)
            if analytical_n != report_n_clusters:
                diff_pct = abs(analytical_n - report_n_clusters) / analytical_n * 100
                method_comparison = f"""\n\n<div style="background-color: #e8f4fd; padding: 15px; border-radius: 8px; border-left: 4px solid #2E86AB; margin-top: 15px;">
<strong>Method Comparison:</strong> Analytical method yields {analytical_n} clusters per arm. 
Simulation differs by {diff_pct:.1f}%, likely due to the discrete nature of cluster allocation and binary outcome variability.
</div>"""
        
        # ICC scale information if applicable
        icc_scale_info = ""
        if 'icc_scale_original' in results and results['icc_scale_original'] == 'Logit':
            icc_scale_info = f"""\n<div style="background-color: #fff3cd; padding: 10px; border-radius: 6px; margin-top: 10px;">
    <strong>ICC Scale Conversion:</strong> ICC was converted from logit scale ({results.get('icc_original', icc):.4f}) 
    to linear scale ({icc:.4f}) for binary outcome calculations.
</div>"""
        
        report_html = f"""
<div style="font-family: Arial, sans-serif; line-height: 1.6; max-width: 800px;">

<h2 style="color: #2E86AB; border-bottom: 2px solid #2E86AB; padding-bottom: 10px;">
Cluster RCT - Sample Size Calculation Report
</h2>

<h3 style="color: #495057;">Required Sample Size</h3>
<div style="background-color: #e8f4fd; padding: 20px; border-radius: 8px; border-left: 4px solid #2E86AB;">
    <table style="width: 100%; border-collapse: collapse;">
    <tr>
        <td style="padding: 8px;"><strong>Clusters per Arm:</strong></td>
        <td style="padding: 8px; font-size: 1.2em; color: #2E86AB;"><strong>{report_n_clusters}</strong></td>
        <td style="padding: 8px;"><strong>Cluster Size:</strong></td>
        <td style="padding: 8px; font-size: 1.2em; color: #2E86AB;"><strong>{report_cluster_size}</strong></td>
    </tr>
    <tr>
        <td style="padding: 8px;"><strong>Total Clusters:</strong></td>
        <td style="padding: 8px;">{report_n_clusters * 2}</td>
        <td style="padding: 8px;"><strong>Total Sample Size:</strong></td>
        <td style="padding: 8px; font-size: 1.1em;"><strong>{total_n}</strong></td>
    </tr>
    </table>
</div>

<h3 style="color: #495057;">Effect Size & Study Parameters</h3>
<div style="background-color: #f8f9fa; padding: 15px; border-radius: 8px;">
    <p><strong>Control Arm Proportion:</strong> {p1:.3f} ({p1*100:.1f}%)</p>
    <p><strong>Intervention Arm Proportion:</strong> {p2:.3f} ({p2*100:.1f}%)</p>
    <p><strong>Risk Difference:</strong> {risk_diff:.3f} ({risk_diff*100:.1f} percentage points)</p>
    <p><strong>Risk Ratio:</strong> {risk_ratio:.3f if risk_ratio != float('inf') else 'N/A'}</p>
    <p><strong>Odds Ratio:</strong> {odds_ratio:.3f if odds_ratio != float('inf') else 'N/A'}</p>
    <p><strong>Statistical Power:</strong> {power * 100:.0f}%</p>
    <p><strong>Significance Level (α):</strong> {alpha:.3f}</p>
    <p><strong>Analysis Method:</strong> {method_text}</p>
</div>

<h3 style="color: #495057;">Clustering Effects</h3>
<div style="background-color: #e6f3ff; padding: 15px; border-radius: 8px; border-left: 4px solid #0066cc;">
    <p><strong>Intracluster Correlation (ICC):</strong> {icc:.3f} ({icc_level})</p>
    <p><strong>Design Effect (DEFF):</strong> {design_effect:.2f}</p>
    <p><strong>Effective Sample Size:</strong> {effective_n:.0f} individuals</p>
    <p style="font-style: italic; color: #0066cc;">
    The ICC of {icc:.3f} indicates {icc_level} clustering with {icc_impact}. 
    The design effect of {design_effect:.2f} means this cluster design requires {design_effect:.1f}x 
    more participants than an individually randomized trial.
    </p>
    {icc_scale_info}
</div>

<div style="background-color: #e6f3ff; padding: 20px; border-radius: 8px; border-left: 4px solid #0066cc; margin: 20px 0;">
    <h4 style="color: #0052a3; margin-top: 0; margin-bottom: 15px;">Methodological Description</h4>
    <div style="background-color: white; padding: 15px; border-radius: 6px; border: 1px solid #cce7ff;">
        <p style="font-style: italic; line-height: 1.8; margin: 0; color: #333;">
        A cluster randomized controlled trial with {report_n_clusters} clusters per arm and {report_cluster_size} individuals per cluster 
        (total N = {total_n}) will provide {power * 100:.0f}% power to detect a change in proportion from {p1:.3f} ({p1*100:.1f}%) 
        in the control arm to {p2:.3f} ({p2*100:.1f}%) in the intervention arm (risk difference = {risk_diff:.3f}, 
        risk ratio = {risk_ratio:.2f if risk_ratio != float('inf') else 'N/A'}), assuming an intracluster correlation coefficient of {icc:.3f}. 
        The design effect of {design_effect:.2f} accounts for the loss of statistical efficiency due to clustering. 
        Power calculations were performed {method_text} with a Type I error rate of {alpha*100:.0f}%.
        </p>
    </div>
    <p style="font-size: 0.9em; color: #666; margin-top: 10px; margin-bottom: 0;">
    <strong>Tip:</strong> Copy the text above directly into your grant application or study protocol.
    </p>
</div>
{method_comparison}

<h3 style="color: #495057;">Methodological References</h3>
<div style="background-color: #f8f9fa; padding: 15px; border-radius: 8px;">
    <p><strong>Primary Reference:</strong></p>
    <p style="font-style: italic; margin: 10px 0;">Hayes RJ, Moulton LH. (2017). Cluster Randomised Trials. 2nd Edition. CRC Press.</p>
    <p><strong>DOI:</strong> <a href="https://doi.org/10.1201/9781315370286" target="_blank" style="color: #2E86AB;">https://doi.org/10.1201/9781315370286</a></p>
    
    <p style="margin-top: 15px;"><strong>Additional Key References:</strong></p>
    <ul style="margin-top: 10px;">
    <li>Murray DM. (1998). Design and Analysis of Group-Randomized Trials. Oxford University Press.</li>
    <li>Eldridge SM, Ukoumunne OC, Carlin JB. (2009). The Intra-Cluster Correlation Coefficient in Cluster Randomized Trials: A Review of Definitions. International Statistical Review, 77(3), 378-394.</li>
    <li>Rutterford C, Copas A, Eldridge S. (2015). Methods for sample size determination in cluster randomized trials. International Journal of Epidemiology, 44(3), 1051-1067.</li>
    </ul>
</div>

<div style="background-color: #d1ecf1; padding: 15px; border-radius: 8px; border-left: 4px solid #17a2b8; margin: 15px 0;">
    <h4 style="color: #0c5460; margin-top: 0;">Important Considerations</h4>
    <ul style="color: #0c5460; margin-bottom: 0;">
    <li><strong>ICC for Binary Outcomes:</strong> ICC interpretation differs from continuous outcomes; consider using logit scale ICC</li>
    <li><strong>Rare Events:</strong> For proportions < 0.1, consider exact methods or alternative designs</li>
    <li><strong>Cluster Size:</strong> Larger clusters provide diminishing returns due to within-cluster correlation</li>
    <li><strong>Analysis Method:</strong> Use appropriate methods (GEE, mixed models) that account for clustering</li>
    </ul>
</div>

<hr style="margin: 20px 0; border: none; border-top: 1px solid #dee2e6;">
<p style="font-size: 0.9em; color: #6c757d; text-align: center; margin: 10px 0;">
Report generated by DesignPower • Cluster RCT Module • Binary Outcome
</p>
</div>
        """
        return report_html
    else:
        return "No specific report template is available for this outcome type."

def generate_cluster_power_report(results, params, outcome_type):
    """
    Generate an enhanced HTML report for Cluster RCT power calculations.
    
    Provides comprehensive methodological context, design effect interpretation,
    and practical guidance for cluster randomized trials.
    """
    # Extract shared parameters
    n_clusters = params.get('n_clusters', 0)
    cluster_size = params.get('cluster_size', 0)
    total_n = n_clusters * 2 * cluster_size
    icc = params.get('icc', 0)
    design_effect = results.get('design_effect', 1 + (cluster_size - 1) * icc)
    alpha = params.get('alpha', 0.05)
    power = results.get('power', 0)
    method = params.get('method', 'analytical')
    
    # Calculate effective sample size
    effective_n = total_n / design_effect if design_effect > 0 else total_n
    
    # Method text based on analytical or simulation
    if method == "simulation":
        nsim = params.get("nsim", 1000)
        seed = params.get("seed", 42)
        method_text = f"using Monte Carlo simulation ({nsim:,} simulations, random seed {seed})"
    else:
        method_text = "using analytical methods"
    
    # ICC interpretation
    if icc < 0.01:
        icc_level = "negligible"
        icc_impact = "minimal impact on statistical power"
    elif icc < 0.05:
        icc_level = "small"
        icc_impact = "moderate reduction in statistical power"
    elif icc < 0.1:
        icc_level = "moderate"
        icc_impact = "substantial reduction in statistical power"
    else:
        icc_level = "large"
        icc_impact = "major reduction in statistical power"
    
    if "Continuous" in outcome_type:
        # Extract continuous-specific parameters
        mean1 = params.get('mean1', 0)
        mean2 = params.get('mean2', 0)
        std_dev = params.get('std_dev', 0)
        difference = abs(mean2 - mean1)
        effect_size = difference / std_dev if std_dev > 0 else 0
        
        report_html = f"""
<div style="font-family: Arial, sans-serif; line-height: 1.6; max-width: 800px;">

<h2 style="color: #2E86AB; border-bottom: 2px solid #2E86AB; padding-bottom: 10px;">
Cluster RCT - Power Calculation Report
</h2>

<h3 style="color: #495057;">Statistical Power Results</h3>
<div style="background-color: #e8f4fd; padding: 20px; border-radius: 8px; border-left: 4px solid #2E86AB;">
    <table style="width: 100%; border-collapse: collapse;">
    <tr>
        <td style="padding: 8px;"><strong>Statistical Power:</strong></td>
        <td style="padding: 8px; font-size: 1.2em; color: #2E86AB;"><strong>{power * 100:.1f}%</strong></td>
        <td style="padding: 8px;"><strong>Significance Level (α):</strong></td>
        <td style="padding: 8px;">{alpha:.3f}</td>
    </tr>
    <tr>
        <td style="padding: 8px;"><strong>Clusters per Arm:</strong></td>
        <td style="padding: 8px;">{n_clusters}</td>
        <td style="padding: 8px;"><strong>Cluster Size:</strong></td>
        <td style="padding: 8px;">{cluster_size}</td>
    </tr>
    <tr>
        <td style="padding: 8px;"><strong>Total Sample Size:</strong></td>
        <td style="padding: 8px;">{total_n}</td>
        <td style="padding: 8px;"><strong>Analysis Method:</strong></td>
        <td style="padding: 8px;">{method_text.replace('using ', '').capitalize()}</td>
    </tr>
    </table>
</div>

<h3 style="color: #495057;">Effect Size & Study Parameters</h3>
<div style="background-color: #f8f9fa; padding: 15px; border-radius: 8px;">
    <p><strong>Target Difference:</strong> {difference:.2f} units (Mean₁ = {mean1:.2f}, Mean₂ = {mean2:.2f})</p>
    <p><strong>Standardized Effect Size:</strong> Cohen's d = {effect_size:.3f}</p>
    <p><strong>Standard Deviation:</strong> {std_dev:.2f}</p>
</div>

<h3 style="color: #495057;">Clustering Effects</h3>
<div style="background-color: #e6f3ff; padding: 15px; border-radius: 8px; border-left: 4px solid #0066cc;">
    <p><strong>Intracluster Correlation (ICC):</strong> {icc:.3f} ({icc_level})</p>
    <p><strong>Design Effect (DEFF):</strong> {design_effect:.2f}</p>
    <p><strong>Effective Sample Size:</strong> {effective_n:.0f} individuals</p>
    <p style="font-style: italic; color: #0066cc;">
    The ICC of {icc:.3f} indicates {icc_level} clustering with {icc_impact}. 
    The design effect of {design_effect:.2f} means the clustering reduces effective sample size 
    by a factor of {design_effect:.1f}.
    </p>
</div>

<div style="background-color: #e6f3ff; padding: 20px; border-radius: 8px; border-left: 4px solid #0066cc; margin: 20px 0;">
    <h4 style="color: #0052a3; margin-top: 0; margin-bottom: 15px;">Methodological Description</h4>
    <div style="background-color: white; padding: 15px; border-radius: 6px; border: 1px solid #cce7ff;">
        <p style="font-style: italic; line-height: 1.8; margin: 0; color: #333;">
        With {n_clusters} clusters per arm and {cluster_size} individuals per cluster 
        (total N = {total_n}), the study will have {power * 100:.1f}% power to detect a difference of {difference:.2f} units between 
        treatment arms (standardized effect size d = {effect_size:.3f}), assuming a within-cluster standard deviation of {std_dev:.2f} 
        and an intracluster correlation coefficient of {icc:.3f}. The design effect of {design_effect:.2f} accounts for the 
        reduced statistical efficiency due to clustering. Power calculations were performed {method_text} with a 
        Type I error rate of {alpha*100:.0f}%.
        </p>
    </div>
    <p style="font-size: 0.9em; color: #666; margin-top: 10px; margin-bottom: 0;">
    <strong>Tip:</strong> Copy the text above directly into your grant application or study protocol.
    </p>
</div>

<h3 style="color: #495057;">Methodological References</h3>
<div style="background-color: #f8f9fa; padding: 15px; border-radius: 8px;">
    <p><strong>Primary Reference:</strong></p>
    <p style="font-style: italic; margin: 10px 0;">Donner A, Klar N. (2000). Design and Analysis of Cluster Randomization Trials in Health Research. London: Arnold.</p>
    <p><strong>DOI:</strong> <a href="https://doi.org/10.1002/sim.836" target="_blank" style="color: #2E86AB;">https://doi.org/10.1002/sim.836</a></p>
    
    <p style="margin-top: 15px;"><strong>Additional Key References:</strong></p>
    <ul style="margin-top: 10px;">
    <li>Murray DM. (1998). Design and Analysis of Group-Randomized Trials. Oxford University Press.</li>
    <li>Eldridge SM, Kerry S. (2012). A Practical Guide to Cluster Randomised Trials in Health Services Research. Wiley.</li>
    <li>Moerbeek M, Teerenstra S. (2016). Power Analysis of Trials with Multilevel Data. CRC Press.</li>
    </ul>
</div>

<div style="background-color: #d1ecf1; padding: 15px; border-radius: 8px; border-left: 4px solid #17a2b8; margin: 15px 0;">
    <h4 style="color: #0c5460; margin-top: 0;">Important Considerations</h4>
    <ul style="color: #0c5460; margin-bottom: 0;">
    <li><strong>Power Interpretation:</strong> {power * 100:.0f}% power means {power * 100:.0f}% probability of detecting the effect if it truly exists</li>
    <li><strong>Sample Size Adequacy:</strong> {'Consider increasing sample size' if power < 0.8 else 'Sample size appears adequate'}</li>
    <li><strong>ICC Uncertainty:</strong> Power is sensitive to ICC; consider sensitivity analysis</li>
    <li><strong>Analysis Alignment:</strong> Ensure final analysis accounts for clustering</li>
    </ul>
</div>

<hr style="margin: 20px 0; border: none; border-top: 1px solid #dee2e6;">
<p style="font-size: 0.9em; color: #6c757d; text-align: center; margin: 10px 0;">
Report generated by DesignPower • Cluster RCT Module • Continuous Outcome
</p>
</div>
        """
        return report_html
    elif "Binary" in outcome_type:
        # Extract binary-specific parameters
        p1 = params.get('p1', 0)
        p2 = params.get('p2', 0)
        risk_diff = abs(p2 - p1)
        risk_ratio = p2 / p1 if p1 > 0 else float('inf')
        odds_ratio = (p2/(1-p2)) / (p1/(1-p1)) if p1 > 0 and p1 < 1 and p2 < 1 else float('inf')
        
        # ICC scale information if applicable
        icc_scale_info = ""
        if 'icc_scale_original' in results and results['icc_scale_original'] == 'Logit':
            icc_scale_info = f"""\n<div style="background-color: #fff3cd; padding: 10px; border-radius: 6px; margin-top: 10px;">
    <strong>ICC Scale Conversion:</strong> ICC was converted from logit scale ({results.get('icc_original', icc):.4f}) 
    to linear scale ({icc:.4f}) for binary outcome calculations.
</div>"""
        
        report_html = f"""
<div style="font-family: Arial, sans-serif; line-height: 1.6; max-width: 800px;">

<h2 style="color: #2E86AB; border-bottom: 2px solid #2E86AB; padding-bottom: 10px;">
Cluster RCT - Power Calculation Report
</h2>

<h3 style="color: #495057;">Statistical Power Results</h3>
<div style="background-color: #e8f4fd; padding: 20px; border-radius: 8px; border-left: 4px solid #2E86AB;">
    <table style="width: 100%; border-collapse: collapse;">
    <tr>
        <td style="padding: 8px;"><strong>Statistical Power:</strong></td>
        <td style="padding: 8px; font-size: 1.2em; color: #2E86AB;"><strong>{power * 100:.1f}%</strong></td>
        <td style="padding: 8px;"><strong>Significance Level (α):</strong></td>
        <td style="padding: 8px;">{alpha:.3f}</td>
    </tr>
    <tr>
        <td style="padding: 8px;"><strong>Clusters per Arm:</strong></td>
        <td style="padding: 8px;">{n_clusters}</td>
        <td style="padding: 8px;"><strong>Cluster Size:</strong></td>
        <td style="padding: 8px;">{cluster_size}</td>
    </tr>
    <tr>
        <td style="padding: 8px;"><strong>Total Sample Size:</strong></td>
        <td style="padding: 8px;">{total_n}</td>
        <td style="padding: 8px;"><strong>Analysis Method:</strong></td>
        <td style="padding: 8px;">{method_text.replace('using ', '').capitalize()}</td>
    </tr>
    </table>
</div>

<h3 style="color: #495057;">Effect Size & Study Parameters</h3>
<div style="background-color: #f8f9fa; padding: 15px; border-radius: 8px;">
    <p><strong>Control Arm Proportion:</strong> {p1:.3f} ({p1*100:.1f}%)</p>
    <p><strong>Intervention Arm Proportion:</strong> {p2:.3f} ({p2*100:.1f}%)</p>
    <p><strong>Risk Difference:</strong> {risk_diff:.3f} ({risk_diff*100:.1f} percentage points)</p>
    <p><strong>Risk Ratio:</strong> {risk_ratio:.3f if risk_ratio != float('inf') else 'N/A'}</p>
    <p><strong>Odds Ratio:</strong> {odds_ratio:.3f if odds_ratio != float('inf') else 'N/A'}</p>
</div>

<h3 style="color: #495057;">Clustering Effects</h3>
<div style="background-color: #e6f3ff; padding: 15px; border-radius: 8px; border-left: 4px solid #0066cc;">
    <p><strong>Intracluster Correlation (ICC):</strong> {icc:.3f} ({icc_level})</p>
    <p><strong>Design Effect (DEFF):</strong> {design_effect:.2f}</p>
    <p><strong>Effective Sample Size:</strong> {effective_n:.0f} individuals</p>
    <p style="font-style: italic; color: #0066cc;">
    The ICC of {icc:.3f} indicates {icc_level} clustering with {icc_impact}. 
    The design effect of {design_effect:.2f} means the clustering reduces effective sample size 
    by a factor of {design_effect:.1f}.
    </p>
    {icc_scale_info}
</div>

<div style="background-color: #e6f3ff; padding: 20px; border-radius: 8px; border-left: 4px solid #0066cc; margin: 20px 0;">
    <h4 style="color: #0052a3; margin-top: 0; margin-bottom: 15px;">Methodological Description</h4>
    <div style="background-color: white; padding: 15px; border-radius: 6px; border: 1px solid #cce7ff;">
        <p style="font-style: italic; line-height: 1.8; margin: 0; color: #333;">
        With {n_clusters} clusters per arm and {cluster_size} individuals per cluster 
        (total N = {total_n}), the study will have {power * 100:.1f}% power to detect a change in proportion from {p1:.3f} ({p1*100:.1f}%) 
        in the control arm to {p2:.3f} ({p2*100:.1f}%) in the intervention arm (risk difference = {risk_diff:.3f}, 
        risk ratio = {risk_ratio:.2f if risk_ratio != float('inf') else 'N/A'}), assuming an intracluster correlation coefficient of {icc:.3f}. 
        The design effect of {design_effect:.2f} accounts for the reduced statistical efficiency due to clustering. 
        Power calculations were performed {method_text} with a Type I error rate of {alpha*100:.0f}%.
        </p>
    </div>
    <p style="font-size: 0.9em; color: #666; margin-top: 10px; margin-bottom: 0;">
    <strong>Tip:</strong> Copy the text above directly into your grant application or study protocol.
    </p>
</div>

<h3 style="color: #495057;">Methodological References</h3>
<div style="background-color: #f8f9fa; padding: 15px; border-radius: 8px;">
    <p><strong>Primary Reference:</strong></p>
    <p style="font-style: italic; margin: 10px 0;">Hayes RJ, Moulton LH. (2017). Cluster Randomised Trials. 2nd Edition. CRC Press.</p>
    <p><strong>DOI:</strong> <a href="https://doi.org/10.1201/9781315370286" target="_blank" style="color: #2E86AB;">https://doi.org/10.1201/9781315370286</a></p>
    
    <p style="margin-top: 15px;"><strong>Additional Key References:</strong></p>
    <ul style="margin-top: 10px;">
    <li>Murray DM. (1998). Design and Analysis of Group-Randomized Trials. Oxford University Press.</li>
    <li>Donner A, Klar N. (2000). Design and Analysis of Cluster Randomization Trials in Health Research. Arnold.</li>
    <li>Rutterford C, Copas A, Eldridge S. (2015). Methods for sample size determination in cluster randomized trials. International Journal of Epidemiology, 44(3), 1051-1067.</li>
    </ul>
</div>

<div style="background-color: #d1ecf1; padding: 15px; border-radius: 8px; border-left: 4px solid #17a2b8; margin: 15px 0;">
    <h4 style="color: #0c5460; margin-top: 0;">Important Considerations</h4>
    <ul style="color: #0c5460; margin-bottom: 0;">
    <li><strong>Power Interpretation:</strong> {power * 100:.0f}% power means {power * 100:.0f}% probability of detecting the effect if it truly exists</li>
    <li><strong>Sample Size Adequacy:</strong> {'Consider increasing sample size' if power < 0.8 else 'Sample size appears adequate'}</li>
    <li><strong>Binary ICC:</strong> ICC for binary outcomes can be interpreted differently than continuous</li>
    <li><strong>Rare Events:</strong> {'Consider exact methods' if min(p1, p2) < 0.1 else 'Normal approximation appropriate'}</li>
    </ul>
</div>

<hr style="margin: 20px 0; border: none; border-top: 1px solid #dee2e6;">
<p style="font-size: 0.9em; color: #6c757d; text-align: center; margin: 10px 0;">
Report generated by DesignPower • Cluster RCT Module • Binary Outcome
</p>
</div>
        """
        return report_html
    else:
        return "No specific report template is available for this outcome type."

def generate_cluster_mde_report(results, params, outcome_type):
    """
    Generate an enhanced HTML report for Cluster RCT minimum detectable effect calculations.
    
    Provides comprehensive methodological context, design effect interpretation,
    and practical guidance for cluster randomized trials.
    """
    # Extract shared parameters
    n_clusters = params.get('n_clusters', 0)
    cluster_size = params.get('cluster_size', 0)
    total_n = n_clusters * 2 * cluster_size
    icc = params.get('icc', 0)
    design_effect = results.get('design_effect', 1 + (cluster_size - 1) * icc)
    alpha = params.get('alpha', 0.05)
    power = params.get('power', 0.8)
    method = params.get('method', 'analytical')
    
    # Calculate effective sample size
    effective_n = total_n / design_effect if design_effect > 0 else total_n
    
    # Method text based on analytical or simulation
    if method == "simulation":
        nsim = params.get("nsim", 1000)
        seed = params.get("seed", 42)
        method_text = f"using Monte Carlo simulation ({nsim:,} simulations, random seed {seed})"
    else:
        method_text = "using analytical methods"
    
    # ICC interpretation
    if icc < 0.01:
        icc_level = "negligible"
        icc_impact = "minimal impact on minimum detectable effect"
    elif icc < 0.05:
        icc_level = "small"
        icc_impact = "moderate increase in minimum detectable effect"
    elif icc < 0.1:
        icc_level = "moderate"
        icc_impact = "substantial increase in minimum detectable effect"
    else:
        icc_level = "large"
        icc_impact = "major increase in minimum detectable effect"
    
    if "Continuous" in outcome_type:
        # Extract continuous-specific parameters
        std_dev = params.get('std_dev', 0)
        mde = results.get('mde', 0)
        effect_size = results.get('effect_size', mde/std_dev if std_dev != 0 else 0)
        
        # Practical interpretation of effect size
        if effect_size < 0.2:
            effect_interpretation = "very small effect"
        elif effect_size < 0.5:
            effect_interpretation = "small effect"
        elif effect_size < 0.8:
            effect_interpretation = "medium effect"
        else:
            effect_interpretation = "large effect"
        
        report_html = f"""
<div style="font-family: Arial, sans-serif; line-height: 1.6; max-width: 800px;">

<h2 style="color: #2E86AB; border-bottom: 2px solid #2E86AB; padding-bottom: 10px;">
Cluster RCT - Minimum Detectable Effect Report
</h2>

<h3 style="color: #495057;">Minimum Detectable Effect</h3>
<div style="background-color: #e8f4fd; padding: 20px; border-radius: 8px; border-left: 4px solid #2E86AB;">
    <table style="width: 100%; border-collapse: collapse;">
    <tr>
        <td style="padding: 8px;"><strong>MDE (Raw Units):</strong></td>
        <td style="padding: 8px; font-size: 1.2em; color: #2E86AB;"><strong>{mde:.3f}</strong></td>
        <td style="padding: 8px;"><strong>Effect Size (Cohen's d):</strong></td>
        <td style="padding: 8px; font-size: 1.2em; color: #2E86AB;"><strong>{effect_size:.3f}</strong></td>
    </tr>
    <tr>
        <td style="padding: 8px;"><strong>Interpretation:</strong></td>
        <td style="padding: 8px;" colspan="3">{effect_interpretation.capitalize()} (d = {effect_size:.3f})</td>
    </tr>
    <tr>
        <td style="padding: 8px;"><strong>Target Power:</strong></td>
        <td style="padding: 8px;">{power * 100:.0f}%</td>
        <td style="padding: 8px;"><strong>Significance Level (α):</strong></td>
        <td style="padding: 8px;">{alpha:.3f}</td>
    </tr>
    </table>
</div>

<h3 style="color: #495057;">Study Design Parameters</h3>
<div style="background-color: #f8f9fa; padding: 15px; border-radius: 8px;">
    <p><strong>Clusters per Arm:</strong> {n_clusters}</p>
    <p><strong>Cluster Size:</strong> {cluster_size} individuals</p>
    <p><strong>Total Sample Size:</strong> {total_n} individuals</p>
    <p><strong>Standard Deviation:</strong> {std_dev:.2f}</p>
    <p><strong>Analysis Method:</strong> {method_text}</p>
</div>

<h3 style="color: #495057;">Clustering Effects</h3>
<div style="background-color: #e6f3ff; padding: 15px; border-radius: 8px; border-left: 4px solid #0066cc;">
    <p><strong>Intracluster Correlation (ICC):</strong> {icc:.3f} ({icc_level})</p>
    <p><strong>Design Effect (DEFF):</strong> {design_effect:.2f}</p>
    <p><strong>Effective Sample Size:</strong> {effective_n:.0f} individuals</p>
    <p style="font-style: italic; color: #0066cc;">
    The ICC of {icc:.3f} indicates {icc_level} clustering with {icc_impact}. 
    The design effect of {design_effect:.2f} inflates the minimum detectable effect 
    by a factor of √{design_effect:.2f} = {design_effect**0.5:.2f}.
    </p>
</div>

<div style="background-color: #e6f3ff; padding: 20px; border-radius: 8px; border-left: 4px solid #0066cc; margin: 20px 0;">
    <h4 style="color: #0052a3; margin-top: 0; margin-bottom: 15px;">Methodological Description</h4>
    <div style="background-color: white; padding: 15px; border-radius: 6px; border: 1px solid #cce7ff;">
        <p style="font-style: italic; line-height: 1.8; margin: 0; color: #333;">
        With {n_clusters} clusters per arm and {cluster_size} individuals per cluster 
        (total N = {total_n}) and {power * 100:.0f}% power, the minimum detectable difference in means 
        is {mde:.3f} units (standardized effect size d = {effect_size:.3f}), assuming a within-cluster 
        standard deviation of {std_dev:.2f} and an intracluster correlation coefficient of {icc:.3f}. 
        The design effect of {design_effect:.2f} accounts for the reduced statistical efficiency due to clustering. 
        Calculations were performed {method_text} with a Type I error rate of {alpha*100:.0f}%.
        </p>
    </div>
    <p style="font-size: 0.9em; color: #666; margin-top: 10px; margin-bottom: 0;">
    <strong>Tip:</strong> Copy the text above directly into your grant application or study protocol.
    </p>
</div>

<div style="background-color: #fffacd; padding: 15px; border-radius: 8px; border-left: 4px solid #ffd700; margin: 15px 0;">
    <h4 style="color: #856404; margin-top: 0;">Practical Interpretation</h4>
    <p style="color: #856404; margin-bottom: 10px;">
    The minimum detectable effect of {mde:.3f} units represents the smallest difference between treatment 
    groups that this study can reliably detect. This corresponds to a {effect_interpretation} (d = {effect_size:.3f}).
    </p>
    <ul style="color: #856404; margin-bottom: 0;">
    <li>Effects smaller than {mde:.3f} units are unlikely to be detected</li>
    <li>The study has {power*100:.0f}% probability of detecting effects ≥ {mde:.3f} units</li>
    <li>Consider if this MDE is clinically meaningful for your research question</li>
    </ul>
</div>

<h3 style="color: #495057;">Methodological References</h3>
<div style="background-color: #f8f9fa; padding: 15px; border-radius: 8px;">
    <p><strong>Primary Reference:</strong></p>
    <p style="font-style: italic; margin: 10px 0;">Donner A, Klar N. (2000). Design and Analysis of Cluster Randomization Trials in Health Research. London: Arnold.</p>
    <p><strong>DOI:</strong> <a href="https://doi.org/10.1002/sim.836" target="_blank" style="color: #2E86AB;">https://doi.org/10.1002/sim.836</a></p>
    
    <p style="margin-top: 15px;"><strong>Additional Key References:</strong></p>
    <ul style="margin-top: 10px;">
    <li>Cohen J. (1988). Statistical Power Analysis for the Behavioral Sciences. 2nd Edition. Erlbaum.</li>
    <li>Moerbeek M, Teerenstra S. (2016). Power Analysis of Trials with Multilevel Data. CRC Press.</li>
    <li>Bloom HS, Richburg-Hayes L, Black AR. (2007). Using covariates to improve precision for studies that randomize schools to evaluate educational interventions. Educational Evaluation and Policy Analysis, 29(1), 30-59.</li>
    </ul>
</div>

<div style="background-color: #d1ecf1; padding: 15px; border-radius: 8px; border-left: 4px solid #17a2b8; margin: 15px 0;">
    <h4 style="color: #0c5460; margin-top: 0;">Important Considerations</h4>
    <ul style="color: #0c5460; margin-bottom: 0;">
    <li><strong>Clinical Significance:</strong> Ensure the MDE of {mde:.3f} is clinically meaningful</li>
    <li><strong>Sample Size Trade-off:</strong> Increasing clusters or cluster size reduces MDE</li>
    <li><strong>ICC Sensitivity:</strong> MDE is highly sensitive to ICC assumptions</li>
    <li><strong>Covariate Adjustment:</strong> Adding covariates in analysis can reduce MDE</li>
    </ul>
</div>

<hr style="margin: 20px 0; border: none; border-top: 1px solid #dee2e6;">
<p style="font-size: 0.9em; color: #6c757d; text-align: center; margin: 10px 0;">
Report generated by DesignPower • Cluster RCT Module • Continuous Outcome
</p>
</div>
        """
        return report_html
    elif "Binary" in outcome_type:
        # Extract binary-specific parameters
        p1 = params.get('p1', 0)
        p2 = results.get('p2', 0)
        risk_diff = abs(p2 - p1)
        risk_ratio = p2 / p1 if p1 > 0 else float('inf')
        odds_ratio = (p2/(1-p2)) / (p1/(1-p1)) if p1 > 0 and p1 < 1 and p2 < 1 else float('inf')
        
        # Practical interpretation
        if risk_diff < 0.05:
            effect_interpretation = "very small effect"
        elif risk_diff < 0.10:
            effect_interpretation = "small effect"
        elif risk_diff < 0.20:
            effect_interpretation = "moderate effect"
        else:
            effect_interpretation = "large effect"
        
        # ICC scale information if applicable
        icc_scale_info = ""
        if 'icc_scale_original' in results and results['icc_scale_original'] == 'Logit':
            icc_scale_info = f"""\n<div style="background-color: #fff3cd; padding: 10px; border-radius: 6px; margin-top: 10px;">
    <strong>ICC Scale Conversion:</strong> ICC was converted from logit scale ({results.get('icc_original', icc):.4f}) 
    to linear scale ({icc:.4f}) for binary outcome calculations.
</div>"""
        
        report_html = f"""
<div style="font-family: Arial, sans-serif; line-height: 1.6; max-width: 800px;">

<h2 style="color: #2E86AB; border-bottom: 2px solid #2E86AB; padding-bottom: 10px;">
Cluster RCT - Minimum Detectable Effect Report
</h2>

<h3 style="color: #495057;">Minimum Detectable Effect</h3>
<div style="background-color: #e8f4fd; padding: 20px; border-radius: 8px; border-left: 4px solid #2E86AB;">
    <table style="width: 100%; border-collapse: collapse;">
    <tr>
        <td style="padding: 8px;"><strong>Detectable Proportion (p₂):</strong></td>
        <td style="padding: 8px; font-size: 1.2em; color: #2E86AB;"><strong>{p2:.3f} ({p2*100:.1f}%)</strong></td>
        <td style="padding: 8px;"><strong>Risk Difference:</strong></td>
        <td style="padding: 8px; font-size: 1.2em; color: #2E86AB;"><strong>{risk_diff:.3f}</strong></td>
    </tr>
    <tr>
        <td style="padding: 8px;"><strong>Risk Ratio:</strong></td>
        <td style="padding: 8px;">{risk_ratio:.3f if risk_ratio != float('inf') else 'N/A'}</td>
        <td style="padding: 8px;"><strong>Odds Ratio:</strong></td>
        <td style="padding: 8px;">{odds_ratio:.3f if odds_ratio != float('inf') else 'N/A'}</td>
    </tr>
    <tr>
        <td style="padding: 8px;"><strong>Interpretation:</strong></td>
        <td style="padding: 8px;" colspan="3">{effect_interpretation.capitalize()} (Δ = {risk_diff:.3f})</td>
    </tr>
    <tr>
        <td style="padding: 8px;"><strong>Target Power:</strong></td>
        <td style="padding: 8px;">{power * 100:.0f}%</td>
        <td style="padding: 8px;"><strong>Significance Level (α):</strong></td>
        <td style="padding: 8px;">{alpha:.3f}</td>
    </tr>
    </table>
</div>

<h3 style="color: #495057;">Study Design Parameters</h3>
<div style="background-color: #f8f9fa; padding: 15px; border-radius: 8px;">
    <p><strong>Control Arm Proportion (p₁):</strong> {p1:.3f} ({p1*100:.1f}%)</p>
    <p><strong>Clusters per Arm:</strong> {n_clusters}</p>
    <p><strong>Cluster Size:</strong> {cluster_size} individuals</p>
    <p><strong>Total Sample Size:</strong> {total_n} individuals</p>
    <p><strong>Analysis Method:</strong> {method_text}</p>
</div>

<h3 style="color: #495057;">Clustering Effects</h3>
<div style="background-color: #e6f3ff; padding: 15px; border-radius: 8px; border-left: 4px solid #0066cc;">
    <p><strong>Intracluster Correlation (ICC):</strong> {icc:.3f} ({icc_level})</p>
    <p><strong>Design Effect (DEFF):</strong> {design_effect:.2f}</p>
    <p><strong>Effective Sample Size:</strong> {effective_n:.0f} individuals</p>
    <p style="font-style: italic; color: #0066cc;">
    The ICC of {icc:.3f} indicates {icc_level} clustering with {icc_impact}. 
    The design effect of {design_effect:.2f} reduces the effective sample size and increases 
    the minimum detectable effect.
    </p>
    {icc_scale_info}
</div>

<div style="background-color: #e6f3ff; padding: 20px; border-radius: 8px; border-left: 4px solid #0066cc; margin: 20px 0;">
    <h4 style="color: #0052a3; margin-top: 0; margin-bottom: 15px;">Methodological Description</h4>
    <div style="background-color: white; padding: 15px; border-radius: 6px; border: 1px solid #cce7ff;">
        <p style="font-style: italic; line-height: 1.8; margin: 0; color: #333;">
        With {n_clusters} clusters per arm and {cluster_size} individuals per cluster 
        (total N = {total_n}) and {power * 100:.0f}% power, the minimum detectable proportion 
        in the intervention arm is {p2:.3f} ({p2*100:.1f}%), given a proportion of {p1:.3f} ({p1*100:.1f}%) 
        in the control arm. This corresponds to a risk difference of {risk_diff:.3f} ({risk_diff*100:.1f} percentage points). 
        The intracluster correlation coefficient of {icc:.3f} yields a design effect of {design_effect:.2f}. 
        Calculations were performed {method_text} with a Type I error rate of {alpha*100:.0f}%.
        </p>
    </div>
    <p style="font-size: 0.9em; color: #666; margin-top: 10px; margin-bottom: 0;">
    <strong>Tip:</strong> Copy the text above directly into your grant application or study protocol.
    </p>
</div>

<div style="background-color: #fffacd; padding: 15px; border-radius: 8px; border-left: 4px solid #ffd700; margin: 15px 0;">
    <h4 style="color: #856404; margin-top: 0;">Practical Interpretation</h4>
    <p style="color: #856404; margin-bottom: 10px;">
    The minimum detectable risk difference of {risk_diff:.3f} ({risk_diff*100:.1f} percentage points) represents 
    the smallest change in proportion that this study can reliably detect. This corresponds to a {effect_interpretation}.
    </p>
    <ul style="color: #856404; margin-bottom: 0;">
    <li>Changes smaller than {risk_diff*100:.1f} percentage points are unlikely to be detected</li>
    <li>The study has {power*100:.0f}% probability of detecting changes ≥ {risk_diff*100:.1f} percentage points</li>
    <li>In absolute terms: detecting a change from {p1*100:.1f}% to at least {p2*100:.1f}%</li>
    </ul>
</div>

<h3 style="color: #495057;">Methodological References</h3>
<div style="background-color: #f8f9fa; padding: 15px; border-radius: 8px;">
    <p><strong>Primary Reference:</strong></p>
    <p style="font-style: italic; margin: 10px 0;">Hayes RJ, Moulton LH. (2017). Cluster Randomised Trials. 2nd Edition. CRC Press.</p>
    <p><strong>DOI:</strong> <a href="https://doi.org/10.1201/9781315370286" target="_blank" style="color: #2E86AB;">https://doi.org/10.1201/9781315370286</a></p>
    
    <p style="margin-top: 15px;"><strong>Additional Key References:</strong></p>
    <ul style="margin-top: 10px;">
    <li>Murray DM. (1998). Design and Analysis of Group-Randomized Trials. Oxford University Press.</li>
    <li>Eldridge S, Kerry S. (2012). A Practical Guide to Cluster Randomised Trials in Health Services Research. Wiley.</li>
    <li>Hemming K, Eldridge S, Forbes G, Weijer C, Taljaard M. (2017). How to design efficient cluster randomised trials. BMJ, 358, j3064.</li>
    </ul>
</div>

<div style="background-color: #d1ecf1; padding: 15px; border-radius: 8px; border-left: 4px solid #17a2b8; margin: 15px 0;">
    <h4 style="color: #0c5460; margin-top: 0;">Important Considerations</h4>
    <ul style="color: #0c5460; margin-bottom: 0;">
    <li><strong>Clinical Significance:</strong> Ensure a {risk_diff*100:.1f} percentage point change is meaningful</li>
    <li><strong>Baseline Rate:</strong> MDE depends heavily on the control proportion ({p1*100:.1f}%)</li>
    <li><strong>Rare Events:</strong> {'Consider alternative designs' if p1 < 0.1 else 'Standard methods appropriate'}</li>
    <li><strong>Number Needed to Treat:</strong> NNT = {1/risk_diff:.0f if risk_diff > 0 else 'N/A'}</li>
    </ul>
</div>

<hr style="margin: 20px 0; border: none; border-top: 1px solid #dee2e6;">
<p style="font-size: 0.9em; color: #6c757d; text-align: center; margin: 10px 0;">
Report generated by DesignPower • Cluster RCT Module • Binary Outcome
</p>
</div>
        """
        return report_html
    else:
        return "No specific report template is available for this outcome type."
