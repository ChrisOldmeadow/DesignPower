"""
Module for generating A'Hern design report text.

This module provides functions specifically for A'Hern design reports for single arm
trials with binary outcomes.
"""

import textwrap

def generate_ahern_report(results, params):
    """
    Generate a human-readable report for A'Hern design results.
    
    Parameters
    ----------
    results : dict
        Results from the calculation function, including n, r, and actual error rates
    params : dict
        Input parameters used for the calculation
        
    Returns
    -------
    str
        Formatted text report
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
    
    # Reference for A'Hern design
    citation = "A'Hern RP. (2001). Sample size tables for exact single-stage phase II designs. Statistics in Medicine, 20(6), 859-866"
    doi = "https://doi.org/10.1002/sim.721"
    
    # Generate report text
    report_text = textwrap.dedent(f"""
    Sample Size Calculation Report (A'Hern Design):
    
    A sample size of {n} participants is required with a rejection threshold of {r} responses.
    This design provides {actual_power * 100:.1f}% power to detect a response rate of {p:.2f} 
    when the null hypothesis response rate is {p0:.2f}. The null hypothesis will be rejected 
    if {r} or more responses are observed. This design controls the Type I error rate at 
    {actual_alpha * 100:.2f}%.
    
    A'Hern's design uses exact binomial probabilities rather than normal approximations, making 
    it particularly suitable for small-to-moderate sample sizes in phase II trials.
    
    Reference: {citation}
    DOI: {doi}
    """)
    
    return report_text.strip()
