"""
Simulation-based estimation utilities for complex study designs.

This module provides functions to simulate various study designs and estimate
power, sample size requirements, and minimum detectable effects through simulation.
"""
import numpy as np
from scipy import stats


def simulate_parallel_rct(n1, n2, mean1, mean2, std_dev, nsim=1000, alpha=0.05):
    """
    Simulate a parallel RCT with continuous outcome and estimate power.
    
    Parameters
    ----------
    n1 : int
        Sample size for group 1
    n2 : int
        Sample size for group 2
    mean1 : float
        Mean outcome in group 1
    mean2 : float
        Mean outcome in group 2
    std_dev : float
        Standard deviation of outcome (assumed equal in both groups)
    nsim : int, optional
        Number of simulations, by default 1000
    alpha : float, optional
        Significance level, by default 0.05
    
    Returns
    -------
    dict
        Dictionary containing the estimated power and simulation details
    """
    # Initialize counter for significant results
    sig_count = 0
    
    # Store p-values for all simulations
    p_values = []
    
    # Run simulations
    for _ in range(nsim):
        # Generate data for both groups
        group1 = np.random.normal(mean1, std_dev, n1)
        group2 = np.random.normal(mean2, std_dev, n2)
        
        # Perform t-test
        t_stat, p_val = stats.ttest_ind(group1, group2, equal_var=True)
        
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
            "n1": n1,
            "n2": n2,
            "mean1": mean1,
            "mean2": mean2,
            "std_dev": std_dev,
            "alpha": alpha
        }
    }


def simulate_cluster_rct(n_clusters, cluster_size, icc, mean1, mean2, 
                         std_dev, nsim=1000, alpha=0.05):
    """
    Simulate a cluster RCT with continuous outcome and estimate power.
    
    Parameters
    ----------
    n_clusters : int
        Number of clusters per arm
    cluster_size : int
        Number of individuals per cluster
    icc : float
        Intracluster correlation coefficient
    mean1 : float
        Mean outcome in control arm
    mean2 : float
        Mean outcome in intervention arm
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
        # Generate data for control arm
        control_data = []
        for _ in range(n_clusters):
            # Generate cluster effect
            cluster_effect = np.random.normal(0, np.sqrt(var_between))
            # Generate individual outcomes within cluster
            cluster_outcomes = np.random.normal(mean1 + cluster_effect, np.sqrt(var_within), cluster_size)
            control_data.extend(cluster_outcomes)
        
        # Generate data for intervention arm
        intervention_data = []
        for _ in range(n_clusters):
            # Generate cluster effect
            cluster_effect = np.random.normal(0, np.sqrt(var_between))
            # Generate individual outcomes within cluster
            cluster_outcomes = np.random.normal(mean2 + cluster_effect, np.sqrt(var_within), cluster_size)
            intervention_data.extend(cluster_outcomes)
        
        # Perform t-test
        t_stat, p_val = stats.ttest_ind(control_data, intervention_data, equal_var=True)
        
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
            "n_clusters": n_clusters,
            "cluster_size": cluster_size,
            "total_n": n_clusters * 2 * cluster_size,
            "icc": icc,
            "mean1": mean1,
            "mean2": mean2,
            "std_dev": std_dev,
            "alpha": alpha
        }
    }


def simulate_stepped_wedge(clusters, steps, individuals_per_cluster,
                          icc, treatment_effect, std_dev, nsim=1000, alpha=0.05):
    """
    Simulate a stepped wedge cluster RCT and estimate power.
    
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


def simulate_binary_cluster_rct(n_clusters, cluster_size, icc, p1, p2, 
                              nsim=1000, alpha=0.05):
    """
    Simulate a cluster RCT with binary outcome using the beta-binomial model.
    
    Parameters
    ----------
    n_clusters : int
        Number of clusters per arm
    cluster_size : int
        Number of individuals per cluster
    icc : float
        Intracluster correlation coefficient
    p1 : float
        Proportion in control arm
    p2 : float
        Proportion in intervention arm
    nsim : int, optional
        Number of simulations, by default 1000
    alpha : float, optional
        Significance level, by default 0.05
    
    Returns
    -------
    dict
        Dictionary containing the estimated power and simulation details
    """
    # Calculate beta distribution parameters for control arm
    mu1 = p1
    if icc > 0:
        kappa1 = (1 - icc) / icc
        alpha1 = mu1 * kappa1
        beta1 = (1 - mu1) * kappa1
    
    # Calculate beta distribution parameters for intervention arm
    mu2 = p2
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
        # Generate data for control arm
        control_successes = []
        for _ in range(n_clusters):
            if icc > 0:
                # Draw cluster-specific probability from beta distribution
                p_cluster = np.random.beta(alpha1, beta1)
            else:
                p_cluster = p1
            
            # Draw number of successes from binomial distribution
            successes = np.random.binomial(cluster_size, p_cluster)
            control_successes.append(successes)
        
        # Generate data for intervention arm
        intervention_successes = []
        for _ in range(n_clusters):
            if icc > 0:
                # Draw cluster-specific probability from beta distribution
                p_cluster = np.random.beta(alpha2, beta2)
            else:
                p_cluster = p2
            
            # Draw number of successes from binomial distribution
            successes = np.random.binomial(cluster_size, p_cluster)
            intervention_successes.append(successes)
        
        # Calculate proportions
        control_prop = sum(control_successes) / (n_clusters * cluster_size)
        intervention_prop = sum(intervention_successes) / (n_clusters * cluster_size)
        
        # Calculate standard error under null hypothesis
        p_pooled = (sum(control_successes) + sum(intervention_successes)) / (2 * n_clusters * cluster_size)
        deff = 1 + (cluster_size - 1) * icc
        se = np.sqrt(2 * p_pooled * (1 - p_pooled) * deff / (n_clusters * cluster_size))
        
        # Calculate test statistic
        z_stat = (intervention_prop - control_prop) / se
        
        # Calculate p-value
        p_val = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        
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
            "n_clusters": n_clusters,
            "cluster_size": cluster_size,
            "total_n": n_clusters * 2 * cluster_size,
            "icc": icc,
            "p1": p1,
            "p2": p2,
            "alpha": alpha
        }
    }
