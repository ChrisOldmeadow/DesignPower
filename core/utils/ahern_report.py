"""
Module for generating A'Hern design report text.

This module provides functions specifically for A'Hern design reports for single arm
trials with binary outcomes.
"""

import textwrap

def generate_ahern_report(results, params):
    """
    Generate an enhanced HTML report for A'Hern design results.
    
    Provides comprehensive design overview, operating characteristics,
    and practical interpretation for single-arm phase II trials.
    
    Parameters
    ----------
    results : dict
        Results from the calculation function, including n, r, and actual error rates
    params : dict
        Input parameters used for the calculation
        
    Returns
    -------
    str
        Formatted HTML report
    """
    # Extract parameters
    p = params.get('p', 0.3)  # Alternative hypothesis proportion
    p0 = params.get('p0', 0.5)  # Null hypothesis proportion
    alpha = params.get('alpha', 0.05)
    
    # Extract results
    n = results.get('n', 0)
    r = results.get('r', 0)
    actual_alpha = results.get('actual_alpha', alpha)
    actual_power = results.get('actual_power', 0.8)
    
    # Calculate additional metrics
    response_rate_percent = p * 100
    null_rate_percent = p0 * 100
    min_responses_for_efficacy = r
    decision_boundary_percent = (r / n * 100) if n > 0 else 0
    
    # Interpretation of response rate
    improvement = (p - p0) / p0 * 100 if p0 > 0 else 0
    if improvement > 50:
        improvement_level = "substantial improvement"
    elif improvement > 25:
        improvement_level = "moderate improvement"
    elif improvement > 0:
        improvement_level = "modest improvement"
    else:
        improvement_level = "no improvement"
    
    # Generate HTML report
    report_html = f"""
<div style="font-family: Arial, sans-serif; line-height: 1.6; max-width: 800px;">

<h2 style="color: #2E86AB; border-bottom: 2px solid #2E86AB; padding-bottom: 10px;">
🔬 A'Hern Single-Stage Design - Sample Size Report
</h2>

<h3 style="color: #495057;">✅ Design Parameters</h3>
<div style="background-color: #e8f4fd; padding: 20px; border-radius: 8px; border-left: 4px solid #2E86AB;">
    <table style="width: 100%; border-collapse: collapse;">
    <tr>
        <td style="padding: 8px;"><strong>Sample Size:</strong></td>
        <td style="padding: 8px; font-size: 1.2em; color: #2E86AB;"><strong>{n}</strong></td>
        <td style="padding: 8px;"><strong>Success Threshold:</strong></td>
        <td style="padding: 8px; font-size: 1.2em; color: #2E86AB;"><strong>≥ {r} responses</strong></td>
    </tr>
    <tr>
        <td style="padding: 8px;"><strong>Decision Boundary:</strong></td>
        <td style="padding: 8px;">{decision_boundary_percent:.1f}%</td>
        <td style="padding: 8px;"><strong>One-sided Test:</strong></td>
        <td style="padding: 8px;">Yes (standard for Phase II)</td>
    </tr>
    </table>
</div>

<h3 style="color: #495057;">📊 Statistical Properties</h3>
<div style="background-color: #f8f9fa; padding: 15px; border-radius: 8px;">
    <p><strong>Null Hypothesis (H₀):</strong> Response rate ≤ {p0:.2f} ({null_rate_percent:.1f}%)</p>
    <p><strong>Alternative Hypothesis (H₁):</strong> Response rate = {p:.2f} ({response_rate_percent:.1f}%)</p>
    <p><strong>Actual Type I Error:</strong> {actual_alpha:.3f} (target: {alpha:.3f})</p>
    <p><strong>Actual Power:</strong> {actual_power:.3f} ({actual_power*100:.1f}%)</p>
    <p><strong>Improvement Over H₀:</strong> {improvement:.1f}% ({improvement_level})</p>
</div>

<h3 style="color: #495057;">🎯 Decision Rule</h3>
<div style="background-color: #e6f3ff; padding: 15px; border-radius: 8px; border-left: 4px solid #0066cc;">
    <p style="font-size: 1.1em; margin-bottom: 15px;"><strong>Reject H₀ if ≥ {r} of {n} patients respond</strong></p>
    <table style="width: 100%; border-collapse: collapse; background-color: white; border-radius: 6px;">
    <tr style="background-color: #f0f8ff;">
        <th style="padding: 10px; border: 1px solid #cce7ff; text-align: left;">Observed Responses</th>
        <th style="padding: 10px; border: 1px solid #cce7ff; text-align: left;">Decision</th>
        <th style="padding: 10px; border: 1px solid #cce7ff; text-align: left;">Conclusion</th>
    </tr>
    <tr>
        <td style="padding: 10px; border: 1px solid #cce7ff;">0 to {r-1}</td>
        <td style="padding: 10px; border: 1px solid #cce7ff;">Fail to reject H₀</td>
        <td style="padding: 10px; border: 1px solid #cce7ff;">Insufficient evidence of efficacy</td>
    </tr>
    <tr>
        <td style="padding: 10px; border: 1px solid #cce7ff;">{r} to {n}</td>
        <td style="padding: 10px; border: 1px solid #cce7ff; color: #28a745; font-weight: bold;">Reject H₀</td>
        <td style="padding: 10px; border: 1px solid #cce7ff; color: #28a745;">Evidence of efficacy</td>
    </tr>
    </table>
</div>

<div style="background-color: #e6f3ff; padding: 20px; border-radius: 8px; border-left: 4px solid #0066cc; margin: 20px 0;">
    <h4 style="color: #0052a3; margin-top: 0; margin-bottom: 15px;">📝 Methodological Description</h4>
    <div style="background-color: white; padding: 15px; border-radius: 6px; border: 1px solid #cce7ff;">
        <p style="font-style: italic; line-height: 1.8; margin: 0; color: #333;">
        A single-arm study with {n} patients will have {actual_power*100:.1f}% power to detect a 
        response rate of {p:.2f} ({response_rate_percent:.1f}%) when the true response rate under the null hypothesis 
        is {p0:.2f} ({null_rate_percent:.1f}%). The null hypothesis will be rejected if {r} or more responses 
        are observed among the {n} patients. This A'Hern design controls the Type I error rate at 
        {actual_alpha:.3f} using exact binomial calculations.
        </p>
    </div>
    <p style="font-size: 0.9em; color: #666; margin-top: 10px; margin-bottom: 0;">
    <strong>Tip:</strong> Copy the text above directly into your clinical trial protocol.
    </p>
</div>

<div style="background-color: #fffacd; padding: 15px; border-radius: 8px; border-left: 4px solid #ffd700; margin: 15px 0;">
    <h4 style="color: #856404; margin-top: 0;">💡 Design Characteristics</h4>
    <p style="color: #856404; margin-bottom: 10px;">
    The A'Hern design is a single-stage exact design for phase II trials:
    </p>
    <ul style="color: #856404; margin-bottom: 0;">
    <li><strong>Fixed sample size:</strong> All {n} patients enrolled regardless of outcomes</li>
    <li><strong>Exact calculations:</strong> Uses binomial distribution, not normal approximation</li>
    <li><strong>Conservative:</strong> Actual Type I error ({actual_alpha:.3f}) ≤ nominal ({alpha:.3f})</li>
    <li><strong>Simple execution:</strong> No interim analyses or stopping rules</li>
    </ul>
</div>

<h3 style="color: #495057;">📚 Methodological Reference</h3>
<div style="background-color: #f8f9fa; padding: 15px; border-radius: 8px;">
    <p><strong>Primary Reference:</strong></p>
    <p style="font-style: italic; margin: 10px 0;">A'Hern RP. (2001). Sample size tables for exact single-stage phase II designs. Statistics in Medicine, 20(6), 859-866.</p>
    <p><strong>DOI:</strong> <a href="https://doi.org/10.1002/sim.721" target="_blank" style="color: #2E86AB;">https://doi.org/10.1002/sim.721</a></p>
    
    <p style="margin-top: 15px;"><strong>Additional Key References:</strong></p>
    <ul style="margin-top: 10px;">
    <li>Fleming TR. (1982). One-sample multiple testing procedure for phase II clinical trials. Biometrics, 38(1), 143-151.</li>
    <li>Machin D, Campbell MJ, Tan SB, Tan SH. (2018). Sample Size Tables for Clinical Studies. 3rd Edition. Wiley-Blackwell.</li>
    <li>Jung SH. (2013). Randomized phase II cancer clinical trials. CRC Press.</li>
    </ul>
</div>

<div style="background-color: #d1ecf1; padding: 15px; border-radius: 8px; border-left: 4px solid #17a2b8; margin: 15px 0;">
    <h4 style="color: #0c5460; margin-top: 0;">⚠️ Important Considerations</h4>
    <ul style="color: #0c5460; margin-bottom: 0;">
    <li><strong>Response definition:</strong> Clearly define what constitutes a "response" before starting</li>
    <li><strong>Patient selection:</strong> Results only apply to similar patient populations</li>
    <li><strong>Early stopping:</strong> No provision for early termination (consider Simon's design if needed)</li>
    <li><strong>Historical control:</strong> Ensure p₀ = {p0:.2f} is well-established from prior data</li>
    </ul>
</div>

<hr style="margin: 20px 0; border: none; border-top: 1px solid #dee2e6;">
<p style="font-size: 0.9em; color: #6c757d; text-align: center; margin: 10px 0;">
Report generated by DesignPower • A'Hern Single-Stage Design • Binary Outcome
</p>
</div>
    """
    
    return report_html.strip()
