"""
Module for generating reports specifically for Cluster RCT designs.
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
        Formatted text report
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
    Generate a report for Cluster RCT sample size calculations.
    """
    # Extract shared parameters
    n_clusters = results.get('n_clusters', 0)
    cluster_size = params.get('cluster_size', 0)
    total_n = n_clusters * 2 * cluster_size
    icc = params.get('icc', 0)
    design_effect = results.get('design_effect', 1 + (cluster_size - 1) * icc)
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
        
        return textwrap.dedent(f"""
        Sample Size Calculation Report for Cluster Randomized Controlled Trial:
        
        A design with {n_clusters} clusters per arm and an average of {cluster_size} 
        individuals per cluster (total N = {total_n}) will provide {power * 100:.0f}% power 
        to detect a difference in means of {abs(mean2 - mean1):.2f} between arms, 
        assuming a standard deviation of {std_dev:.2f}.
        
        With an intracluster correlation coefficient (ICC) of {icc:.3f}, this yields 
        a design effect of {design_effect:.2f}, {method_text}, with a Type I error rate 
        of {alpha * 100:.0f}%.
        
        Reference: Donner A, Klar N. (2000). Design and Analysis of Cluster Randomization Trials in Health Research.
        DOI: https://doi.org/10.1002/sim.836
        """).strip()
    elif "Binary" in outcome_type:
        # Extract binary-specific parameters
        p1 = params.get('p1', 0)
        p2 = params.get('p2', 0)
        
        return textwrap.dedent(f"""
        Sample Size Calculation Report for Cluster Randomized Controlled Trial:
        
        A design with {n_clusters} clusters per arm and an average of {cluster_size} 
        individuals per cluster (total N = {total_n}) will provide {power * 100:.0f}% power 
        to detect a difference in proportions from {p1:.2f} in the control arm to {p2:.2f} 
        in the intervention arm (risk difference = {abs(p2 - p1):.2f}).
        
        With an intracluster correlation coefficient (ICC) of {icc:.3f}, this yields 
        a design effect of {design_effect:.2f}, {method_text}, with a Type I error rate 
        of {alpha * 100:.0f}%.
        
        Reference: Hayes RJ, Moulton LH. (2017). Cluster Randomised Trials. CRC Press
        DOI: https://doi.org/10.1201/9781315370286
        """).strip()
    else:
        return "No specific report template is available for this outcome type."

def generate_cluster_power_report(results, params, outcome_type):
    """
    Generate a report for Cluster RCT power calculations.
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
        
        return textwrap.dedent(f"""
        Power Calculation Report for Cluster Randomized Controlled Trial:
        
        With {n_clusters} clusters per arm and an average of {cluster_size} individuals per cluster 
        (total N = {total_n}), the study will have {power * 100:.1f}% power to detect a difference 
        in means of {abs(mean2 - mean1):.2f} between arms, assuming a standard deviation of {std_dev:.2f}.
        
        With an intracluster correlation coefficient (ICC) of {icc:.3f}, this yields a design effect 
        of {design_effect:.2f}, {method_text}, with a Type I error rate of {alpha * 100:.0f}%.
        
        Reference: Donner A, Klar N. (2000). Design and Analysis of Cluster Randomization Trials in Health Research.
        DOI: https://doi.org/10.1002/sim.836
        """).strip()
    elif "Binary" in outcome_type:
        # Extract binary-specific parameters
        p1 = params.get('p1', 0)
        p2 = params.get('p2', 0)
        
        return textwrap.dedent(f"""
        Power Calculation Report for Cluster Randomized Controlled Trial:
        
        With {n_clusters} clusters per arm and an average of {cluster_size} individuals per cluster 
        (total N = {total_n}), the study will have {power * 100:.1f}% power to detect a difference 
        in proportions from {p1:.2f} in the control arm to {p2:.2f} in the intervention arm 
        (risk difference = {abs(p2 - p1):.2f}).
        
        With an intracluster correlation coefficient (ICC) of {icc:.3f}, this yields a design effect 
        of {design_effect:.2f}, {method_text}, with a Type I error rate of {alpha * 100:.0f}%.
        
        Reference: Hayes RJ, Moulton LH. (2017). Cluster Randomised Trials. CRC Press
        DOI: https://doi.org/10.1201/9781315370286
        """).strip()
    else:
        return "No specific report template is available for this outcome type."

def generate_cluster_mde_report(results, params, outcome_type):
    """
    Generate a report for Cluster RCT minimum detectable effect calculations.
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
    
    # Method text based on analytical or simulation
    if method == "simulation":
        nsim = params.get("nsim", 1000)
        seed = params.get("seed", 42)
        method_text = f"using Monte Carlo simulation ({nsim:,} simulations, random seed {seed})"
    else:
        method_text = "using analytical methods"
    
    if "Continuous" in outcome_type:
        # Extract continuous-specific parameters
        std_dev = params.get('std_dev', 0)
        mde = results.get('mde', 0)
        effect_size = results.get('effect_size', mde/std_dev if std_dev != 0 else 0)
        
        return textwrap.dedent(f"""
        Minimum Detectable Effect Report for Cluster Randomized Controlled Trial:
        
        With {n_clusters} clusters per arm and an average of {cluster_size} individuals per cluster 
        (total N = {total_n}) and {power * 100:.0f}% power, the minimum detectable difference in means 
        is {mde:.2f} (effect size d = {effect_size:.2f}), assuming a standard deviation of {std_dev:.2f}.
        
        With an intracluster correlation coefficient (ICC) of {icc:.3f}, this yields a design effect 
        of {design_effect:.2f}, {method_text}, with a Type I error rate of {alpha * 100:.0f}%.
        
        Reference: Donner A, Klar N. (2000). Design and Analysis of Cluster Randomization Trials in Health Research.
        DOI: https://doi.org/10.1002/sim.836
        """).strip()
    elif "Binary" in outcome_type:
        # Extract binary-specific parameters
        p1 = params.get('p1', 0)
        p2 = results.get('p2', 0)
        
        return textwrap.dedent(f"""
        Minimum Detectable Effect Report for Cluster Randomized Controlled Trial:
        
        With {n_clusters} clusters per arm and an average of {cluster_size} individuals per cluster 
        (total N = {total_n}) and {power * 100:.0f}% power, the minimum detectable proportion 
        in the intervention arm is {p2:.2f} (risk difference = {abs(p2 - p1):.2f}), given a proportion 
        of {p1:.2f} in the control arm.
        
        With an intracluster correlation coefficient (ICC) of {icc:.3f}, this yields a design effect 
        of {design_effect:.2f}, {method_text}, with a Type I error rate of {alpha * 100:.0f}%.
        
        Reference: Hayes RJ, Moulton LH. (2017). Cluster Randomised Trials. CRC Press
        DOI: https://doi.org/10.1201/9781315370286
        """).strip()
    else:
        return "No specific report template is available for this outcome type."
