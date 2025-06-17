"""
Module for generating Simon's two-stage design report text.

This module provides functions specifically for Simon's two-stage design reports for single arm
trials with binary outcomes.
"""

import textwrap

def generate_simons_report(results, params):
    """
    Generate an enhanced HTML report for Simon's two-stage design results.
    
    Provides comprehensive design overview, operating characteristics,
    and practical execution guidance for two-stage phase II trials.
    
    Parameters
    ----------
    results : dict
        Results from the calculation function, including stage sizes and thresholds
    params : dict
        Input parameters used for the calculation
        
    Returns
    -------
    str
        Formatted HTML report
    """
    # Extract parameters
    p0 = params.get('p0', 0.2)  # Null hypothesis proportion
    p1 = params.get('p', 0.4)   # Alternative hypothesis proportion
    alpha = params.get('alpha', 0.05)
    design_type = results.get('design_type', 'Optimal')
    
    # Extract results
    n1 = results.get('n1', 0)    # First stage sample size
    r1 = results.get('r1', 0)    # First stage rejection threshold
    n = results.get('n', 0)      # Total sample size
    n2 = n - n1                  # Second stage sample size
    r = results.get('r', 0)      # Final rejection threshold
    EN0 = results.get('EN0', 0)  # Expected sample size under H0
    PET0 = results.get('PET0', 0)  # Probability of early termination under H0
    EN1 = results.get('EN1', n)  # Expected sample size under H1 (default to n if not provided)
    PET1 = results.get('PET1', 0)  # Probability of early termination under H1
    actual_alpha = results.get('actual_alpha', alpha)
    actual_power = results.get('actual_power', 0.8)
    
    # Calculate additional metrics
    response_rate_percent_p0 = p0 * 100
    response_rate_percent_p1 = p1 * 100
    improvement = (p1 - p0) / p0 * 100 if p0 > 0 else 0
    savings_potential = (n - EN0) / n * 100 if n > 0 else 0
    
    # Design type explanation
    if design_type.lower() == "optimal":
        design_explanation = "minimizes expected sample size under H₀ (when treatment is ineffective)"
    elif design_type.lower() == "minimax":
        design_explanation = "minimizes maximum sample size (most conservative approach)"
    else:
        design_explanation = "balances between optimal and minimax approaches"
    
    # Generate HTML report
    report_html = f"""
<div style="font-family: Arial, sans-serif; line-height: 1.6; max-width: 800px;">

<h2 style="color: #2E86AB; border-bottom: 2px solid #2E86AB; padding-bottom: 10px;">
Simon's Two-Stage Design - Sample Size Report
</h2>

<h3 style="color: #495057;">Design Overview</h3>
<div style="background-color: #e8f4fd; padding: 20px; border-radius: 8px; border-left: 4px solid #2E86AB;">
    <table style="width: 100%; border-collapse: collapse;">
    <tr>
        <td style="padding: 8px;"><strong>Design Type:</strong></td>
        <td style="padding: 8px; font-size: 1.1em; color: #2E86AB;"><strong>{design_type}</strong></td>
        <td style="padding: 8px;"><strong>Total Sample Size:</strong></td>
        <td style="padding: 8px; font-size: 1.1em; color: #2E86AB;"><strong>{n}</strong></td>
    </tr>
    <tr>
        <td style="padding: 8px;"><strong>Stage 1 Size:</strong></td>
        <td style="padding: 8px;">{n1} patients</td>
        <td style="padding: 8px;"><strong>Stage 2 Size:</strong></td>
        <td style="padding: 8px;">{n2} patients</td>
    </tr>
    <tr>
        <td style="padding: 8px;"><strong>Design Goal:</strong></td>
        <td style="padding: 8px;" colspan="3">{design_explanation}</td>
    </tr>
    </table>
</div>

<h3 style="color: #495057;">Statistical Properties</h3>
<div style="background-color: #f8f9fa; padding: 15px; border-radius: 8px;">
    <p><strong>Null Hypothesis (H₀):</strong> Response rate ≤ {p0:.2f} ({response_rate_percent_p0:.1f}%)</p>
    <p><strong>Alternative Hypothesis (H₁):</strong> Response rate = {p1:.2f} ({response_rate_percent_p1:.1f}%)</p>
    <p><strong>Improvement Over H₀:</strong> {improvement:.1f}%</p>
    <p><strong>Actual Type I Error:</strong> {actual_alpha:.3f} (target: {alpha:.3f})</p>
    <p><strong>Actual Power:</strong> {actual_power:.3f} ({actual_power*100:.1f}%)</p>
</div>

<h3 style="color: #495057;">Decision Rules</h3>
<div style="background-color: #e6f3ff; padding: 15px; border-radius: 8px; border-left: 4px solid #0066cc;">
    <h4 style="color: #0052a3; margin-top: 0;">Stage 1 (n₁ = {n1} patients)</h4>
    <table style="width: 100%; border-collapse: collapse; background-color: white; border-radius: 6px; margin-bottom: 15px;">
    <tr style="background-color: #f0f8ff;">
        <th style="padding: 10px; border: 1px solid #cce7ff; text-align: left;">Responses Observed</th>
        <th style="padding: 10px; border: 1px solid #cce7ff; text-align: left;">Decision</th>
        <th style="padding: 10px; border: 1px solid #cce7ff; text-align: left;">Action</th>
    </tr>
    <tr>
        <td style="padding: 10px; border: 1px solid #cce7ff;">≤ {r1}</td>
        <td style="padding: 10px; border: 1px solid #cce7ff; color: #dc3545;">Stop for futility</td>
        <td style="padding: 10px; border: 1px solid #cce7ff;">Treatment ineffective</td>
    </tr>
    <tr>
        <td style="padding: 10px; border: 1px solid #cce7ff;">≥ {r1 + 1}</td>
        <td style="padding: 10px; border: 1px solid #cce7ff; color: #28a745;">Continue to Stage 2</td>
        <td style="padding: 10px; border: 1px solid #cce7ff;">Enroll {n2} more patients</td>
    </tr>
    </table>
    
    <h4 style="color: #0052a3;">Stage 2 (Total n = {n} patients)</h4>
    <table style="width: 100%; border-collapse: collapse; background-color: white; border-radius: 6px;">
    <tr style="background-color: #f0f8ff;">
        <th style="padding: 10px; border: 1px solid #cce7ff; text-align: left;">Total Responses</th>
        <th style="padding: 10px; border: 1px solid #cce7ff; text-align: left;">Decision</th>
        <th style="padding: 10px; border: 1px solid #cce7ff; text-align: left;">Conclusion</th>
    </tr>
    <tr>
        <td style="padding: 10px; border: 1px solid #cce7ff;">≤ {r}</td>
        <td style="padding: 10px; border: 1px solid #cce7ff;">Fail to reject H₀</td>
        <td style="padding: 10px; border: 1px solid #cce7ff;">Insufficient evidence</td>
    </tr>
    <tr>
        <td style="padding: 10px; border: 1px solid #cce7ff;">≥ {r + 1}</td>
        <td style="padding: 10px; border: 1px solid #cce7ff; color: #28a745; font-weight: bold;">Reject H₀</td>
        <td style="padding: 10px; border: 1px solid #cce7ff; color: #28a745;">Evidence of efficacy</td>
    </tr>
    </table>
</div>

<h3 style="color: #495057;">Operating Characteristics</h3>
<div style="background-color: #fffacd; padding: 15px; border-radius: 8px; border-left: 4px solid #ffd700;">
    <h4 style="color: #856404; margin-top: 0;">Under H₀ (p = {p0:.2f})</h4>
    <table style="width: 100%; margin-bottom: 15px;">
    <tr>
        <td style="padding: 5px;"><strong>Expected Sample Size:</strong></td>
        <td style="padding: 5px;">{EN0:.1f} patients</td>
    </tr>
    <tr>
        <td style="padding: 5px;"><strong>Probability of Early Termination:</strong></td>
        <td style="padding: 5px;">{PET0*100:.1f}%</td>
    </tr>
    <tr>
        <td style="padding: 5px;"><strong>Average Savings:</strong></td>
        <td style="padding: 5px;">{savings_potential:.1f}% of maximum sample size</td>
    </tr>
    </table>
    
    <h4 style="color: #856404;">Under H₁ (p = {p1:.2f})</h4>
    <table style="width: 100%;">
    <tr>
        <td style="padding: 5px;"><strong>Expected Sample Size:</strong></td>
        <td style="padding: 5px;">{EN1:.1f} patients</td>
    </tr>
    <tr>
        <td style="padding: 5px;"><strong>Probability of Early Termination:</strong></td>
        <td style="padding: 5px;">{PET1*100:.1f}%</td>
    </tr>
    </table>
</div>

<div style="background-color: #e6f3ff; padding: 20px; border-radius: 8px; border-left: 4px solid #0066cc; margin: 20px 0;">
    <h4 style="color: #0052a3; margin-top: 0; margin-bottom: 15px;">Methodological Description</h4>
    <div style="background-color: white; padding: 15px; border-radius: 6px; border: 1px solid #cce7ff;">
        <p style="font-style: italic; line-height: 1.8; margin: 0; color: #333;">
        A Simon's {design_type.lower()} two-stage design enrolls {n1} patients in stage 1. If {r1} or fewer 
        responses are observed, the trial stops for futility. Otherwise, {n2} additional patients are 
        enrolled (total n = {n}). The null hypothesis of response rate ≤ {p0:.2f} is rejected if more 
        than {r} total responses are observed. This design provides {actual_power*100:.1f}% power to 
        detect a response rate of {p1:.2f} while controlling the Type I error at {actual_alpha:.3f}. 
        Under H₀, the expected sample size is {EN0:.1f} with {PET0*100:.1f}% probability of early termination.
        </p>
    </div>
    <p style="font-size: 0.9em; color: #666; margin-top: 10px; margin-bottom: 0;">
    <strong>Tip:</strong> Copy the text above directly into your clinical trial protocol.
    </p>
</div>

<h3 style="color: #495057;">Methodological References</h3>
<div style="background-color: #f8f9fa; padding: 15px; border-radius: 8px;">
    <p><strong>Primary Reference:</strong></p>
    <p style="font-style: italic; margin: 10px 0;">Simon R. (1989). Optimal two-stage designs for phase II clinical trials. Controlled Clinical Trials, 10(1), 1-10.</p>
    <p><strong>DOI:</strong> <a href="https://doi.org/10.1016/0197-2456(89)90015-9" target="_blank" style="color: #2E86AB;">https://doi.org/10.1016/0197-2456(89)90015-9</a></p>
    
    <p style="margin-top: 15px;"><strong>Additional Key References:</strong></p>
    <ul style="margin-top: 10px;">
    <li>Jung SH, Lee T, Kim K, George SL. (2004). Admissible two-stage designs for phase II cancer clinical trials. Statistics in Medicine, 23(4), 561-569.</li>
    <li>Chen TT. (1997). Optimal three-stage designs for phase II cancer clinical trials. Statistics in Medicine, 16(23), 2701-2711.</li>
    <li>Ensign LG, Gehan EA, Kamen DS, Thall PF. (1994). An optimal three-stage design for phase II clinical trials. Statistics in Medicine, 13(17), 1727-1736.</li>
    </ul>
</div>

<div style="background-color: #d1ecf1; padding: 15px; border-radius: 8px; border-left: 4px solid #17a2b8; margin: 15px 0;">
    <h4 style="color: #0c5460; margin-top: 0;">Important Considerations</h4>
    <ul style="color: #0c5460; margin-bottom: 0;">
    <li><strong>Interim Analysis:</strong> Must be performed after exactly {n1} patients</li>
    <li><strong>No Skipping:</strong> Cannot skip Stage 1 or modify thresholds based on results</li>
    <li><strong>Complete Follow-up:</strong> All Stage 1 patients must have response assessment before deciding</li>
    <li><strong>Design Selection:</strong> {"Optimal design minimizes patient exposure when treatment ineffective" if design_type.lower() == "optimal" else "Minimax design provides smallest worst-case scenario"}</li>
    <li><strong>Binding Futility:</strong> Stopping for futility at Stage 1 is ethically required</li>
    </ul>
</div>

<hr style="margin: 20px 0; border: none; border-top: 1px solid #dee2e6;">
<p style="font-size: 0.9em; color: #6c757d; text-align: center; margin: 10px 0;">
Report generated by DesignPower • Simon's Two-Stage Design • Binary Outcome
</p>
</div>
    """
    
    return report_html.strip()
