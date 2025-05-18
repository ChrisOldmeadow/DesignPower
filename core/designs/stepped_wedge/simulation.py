"""
Simulation-based methods for stepped wedge cluster randomized trials.

This module provides functions for power analysis and sample size calculation
for stepped wedge designs using simulation-based approaches.
"""
import numpy as np
from scipy import stats


def simulate_continuous(clusters, steps, individuals_per_cluster,
                       icc, treatment_effect, std_dev, nsim=1000, alpha=0.05):
    """
    Simulate a stepped wedge cluster RCT with continuous outcome and estimate power.
    
    Parameters
    ----------
    clusters : int
        Number of clusters
    steps : int
        Number of time steps (including baseline)
    individuals_per_cluster : int
        Number of individuals per cluster per time step
    icc : float
        Intracluster correlation coefficient
    treatment_effect : float
        Effect size of the intervention
    std_dev : float
        Total standard deviation of outcome
    nsim : int, optional
        Number of simulations, by default 1000
    alpha : float, optional
        Significance level, by default 0.05
    
    Returns
    -------
    dict
        Dictionary containing the estimated power and simulation details
    """
    # Calculate between-cluster and within-cluster variance
    var_between = icc * std_dev**2
    var_within = (1 - icc) * std_dev**2
    
    # Initialize counter for significant results
    sig_count = 0
    
    # Store p-values for all simulations
    p_values = []
    
    # Run simulations
    for _ in range(nsim):
        # Create dataset
        cluster_ids = []
        time_ids = []
        treatment = []
        outcomes = []
        
        # Generate cluster random effects
        cluster_effects = np.random.normal(0, np.sqrt(var_between), clusters)
        
        # Assign clusters to steps (excluding baseline)
        cluster_step_assignment = np.array(range(clusters)) % (steps - 1) + 1
        np.random.shuffle(cluster_step_assignment)
        
        # Generate data
        for c in range(clusters):
            # Determine when this cluster receives intervention
            step_of_intervention = cluster_step_assignment[c]
            
            # Generate cluster effect
            cluster_effect = cluster_effects[c]
            
            for t in range(steps):
                # Determine if cluster is under treatment at this time step
                is_treated = t >= step_of_intervention
                
                # Generate outcomes for individuals in this cluster at this time step
                for _ in range(individuals_per_cluster):
                    # Calculate mean for this individual
                    individual_mean = 0 + (treatment_effect if is_treated else 0) + cluster_effect
                    
                    # Generate outcome
                    outcome = np.random.normal(individual_mean, np.sqrt(var_within))
                    
                    # Add to dataset
                    cluster_ids.append(c)
                    time_ids.append(t)
                    treatment.append(is_treated)
                    outcomes.append(outcome)
        
        # Convert to numpy arrays
        cluster_ids = np.array(cluster_ids)
        time_ids = np.array(time_ids)
        treatment = np.array(treatment)
        outcomes = np.array(outcomes)
        
        # Perform simple t-test (this is a simplification; in practice a mixed model would be used)
        treated_outcomes = outcomes[treatment == 1]
        control_outcomes = outcomes[treatment == 0]
        t_stat, p_val = stats.ttest_ind(treated_outcomes, control_outcomes, equal_var=True)
        
        # Store p-value
        p_values.append(p_val)
        
        # Check if result is significant
        if p_val < alpha:
            sig_count += 1
    
    # Calculate power
    power = sig_count / nsim
    
    return {
        "power": power,
        "mean_p_value": np.mean(p_values),
        "median_p_value": np.median(p_values),
        "nsim": nsim,
        "parameters": {
            "clusters": clusters,
            "steps": steps,
            "individuals_per_cluster": individuals_per_cluster,
            "total_n": clusters * steps * individuals_per_cluster,
            "icc": icc,
            "treatment_effect": treatment_effect,
            "std_dev": std_dev,
            "alpha": alpha
        }
    }


def simulate_binary(clusters, steps, individuals_per_cluster, 
                  icc, p_control, p_intervention, nsim=1000, alpha=0.05):
    """
    Simulate a stepped wedge cluster RCT with binary outcome and estimate power.
    
    Parameters
    ----------
    clusters : int
        Number of clusters
    steps : int
        Number of time steps (including baseline)
    individuals_per_cluster : int
        Number of individuals per cluster per time step
    icc : float
        Intracluster correlation coefficient
    p_control : float
        Proportion in control condition
    p_intervention : float
        Proportion in intervention condition
    nsim : int, optional
        Number of simulations, by default 1000
    alpha : float, optional
        Significance level, by default 0.05
    
    Returns
    -------
    dict
        Dictionary containing the estimated power and simulation details
    """
    # Calculate beta distribution parameters for control
    mu1 = p_control
    if icc > 0:
        kappa1 = (1 - icc) / icc
        alpha1 = mu1 * kappa1
        beta1 = (1 - mu1) * kappa1
    
    # Calculate beta distribution parameters for intervention
    mu2 = p_intervention
    if icc > 0:
        kappa2 = (1 - icc) / icc
        alpha2 = mu2 * kappa2
        beta2 = (1 - mu2) * kappa2
    
    # Initialize counter for significant results
    sig_count = 0
    
    # Store p-values for all simulations
    p_values = []
    
    # Run simulations
    for _ in range(nsim):
        # Create dataset
        cluster_ids = []
        time_ids = []
        treatment = []
        outcomes = []
        
        # Generate cluster-specific probabilities
        cluster_probs_control = np.random.beta(alpha1, beta1, clusters) if icc > 0 else np.ones(clusters) * p_control
        cluster_probs_intervention = np.random.beta(alpha2, beta2, clusters) if icc > 0 else np.ones(clusters) * p_intervention
        
        # Assign clusters to steps (excluding baseline)
        cluster_step_assignment = np.array(range(clusters)) % (steps - 1) + 1
        np.random.shuffle(cluster_step_assignment)
        
        # Generate data
        for c in range(clusters):
            # Determine when this cluster receives intervention
            step_of_intervention = cluster_step_assignment[c]
            
            # Get cluster-specific probabilities
            prob_control = cluster_probs_control[c]
            prob_intervention = cluster_probs_intervention[c]
            
            for t in range(steps):
                # Determine if cluster is under treatment at this time step
                is_treated = t >= step_of_intervention
                
                # Select appropriate probability for this cluster and condition
                prob = prob_intervention if is_treated else prob_control
                
                # Generate outcomes for individuals in this cluster at this time step
                for _ in range(individuals_per_cluster):
                    # Generate binary outcome
                    outcome = np.random.binomial(1, prob)
                    
                    # Add to dataset
                    cluster_ids.append(c)
                    time_ids.append(t)
                    treatment.append(is_treated)
                    outcomes.append(outcome)
        
        # Convert to numpy arrays
        cluster_ids = np.array(cluster_ids)
        time_ids = np.array(time_ids)
        treatment = np.array(treatment)
        outcomes = np.array(outcomes)
        
        # Calculate proportions
        treated_outcomes = outcomes[treatment == 1]
        control_outcomes = outcomes[treatment == 0]
        
        treated_prop = np.mean(treated_outcomes)
        control_prop = np.mean(control_outcomes)
        
        # Calculate pooled proportion
        p_pooled = (sum(treated_outcomes) + sum(control_outcomes)) / len(outcomes)
        
        # Calculate standard error (simplified approach)
        n_treated = len(treated_outcomes)
        n_control = len(control_outcomes)
        
        # Design effect (simplified)
        deff = 1 + (individuals_per_cluster - 1) * icc
        
        # Standard error with design effect
        se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n_treated + 1/n_control) * deff)
        
        # Calculate test statistic
        if se > 0:
            z_stat = (treated_prop - control_prop) / se
            # Calculate p-value
            p_val = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        else:
            # If standard error is 0, set p-value to 1
            p_val = 1.0
        
        # Store p-value
        p_values.append(p_val)
        
        # Check if result is significant
        if p_val < alpha:
            sig_count += 1
    
    # Calculate power
    power = sig_count / nsim
    
    return {
        "power": power,
        "mean_p_value": np.mean(p_values),
        "median_p_value": np.median(p_values),
        "nsim": nsim,
        "parameters": {
            "clusters": clusters,
            "steps": steps,
            "individuals_per_cluster": individuals_per_cluster,
            "total_n": clusters * steps * individuals_per_cluster,
            "icc": icc,
            "p_control": p_control,
            "p_intervention": p_intervention,
            "alpha": alpha
        }
    }
