"""
Module for generating Simon's two-stage design report text.

This module provides functions specifically for Simon's two-stage design reports for single arm
trials with binary outcomes.
"""

import textwrap

def generate_simons_report(results, params):
    """
    Generate a human-readable report for Simon's two-stage design results.
    
    Parameters
    ----------
    results : dict
        Results from the calculation function, including stage sizes and thresholds
    params : dict
        Input parameters used for the calculation
        
    Returns
    -------
    str
        Formatted text report
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
    r = results.get('r', 0)      # Final rejection threshold
    EN0 = results.get('EN0', 0)  # Expected sample size under H0
    PET0 = results.get('PET0', 0)  # Probability of early termination under H0
    actual_alpha = results.get('actual_alpha', alpha)
    actual_power = results.get('actual_power', 0.8)
    
    # Reference for Simon's design
    citation = "Simon R. (1989). Optimal two-stage designs for phase II clinical trials. Controlled Clinical Trials, 10(1), 1-10"
    doi = "https://doi.org/10.1016/0197-2456(89)90015-9"
    
    # Generate report text
    report_text = textwrap.dedent(f"""
    Sample Size Calculation Report (Simon's Two-Stage Design):
    
    A {design_type.lower()} two-stage design was calculated with the following characteristics:
    
    Stage 1: Enroll {n1} patients. If {r1} or fewer responses are observed, stop the trial for futility.
    Stage 2: If continuing, enroll {n - n1} additional patients (total n = {n}).
    
    The null hypothesis of response rate ≤ {p0:.2f} will be rejected if more than {r} total responses 
    are observed. This design provides {actual_power * 100:.1f}% power to detect a true response rate 
    of {p1:.2f} while controlling the Type I error rate at {actual_alpha * 100:.2f}%.
    
    The probability of early termination under the null hypothesis is {PET0 * 100:.1f}%, with an 
    expected sample size of {EN0:.1f} patients under the null hypothesis.
    
    Simon's two-stage design allows for early stopping for futility, potentially saving resources 
    if the treatment is not effective.
    
    Reference: {citation}
    DOI: {doi}
    """)
    
    return report_text.strip()
