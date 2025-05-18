"""
Simulation-based methods for parallel group randomized controlled trials.

This module provides functions for power analysis and sample size calculation
for parallel group RCTs using simulation-based approaches.
"""
import numpy as np
import math
from scipy import stats
from scipy import optimize


def simulate_continuous(n1, n2, mean1, mean2, std_dev, nsim=1000, alpha=0.05, seed=None, repeated_measures=False, correlation=0.5, method="change_score"):
    """
    Simulate a parallel group RCT with continuous outcome.
    
    Parameters
    ----------
    n1 : int
        Sample size in group 1
    n2 : int
        Sample size in group 2
    mean1 : float
        Mean in group 1
    mean2 : float
        Mean in group 2
    std_dev : float
        Standard deviation of the outcome (assumed equal in both groups)
    nsim : int, optional
        Number of simulations, by default 1000
    alpha : float, optional
        Significance level, by default 0.05
    seed : int, optional
        Random seed for reproducibility, by default None
    repeated_measures : bool, optional
        Whether to simulate repeated measures design with baseline and follow-up measurements,
        by default False
    correlation : float, optional
        Correlation between baseline and follow-up measurements, by default 0.5
    method : str, optional
        Analysis method for repeated measures: "change_score" or "ancova", by default "change_score"
    
    Returns
    -------
    dict
        Dictionary containing power and other simulation results
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Store p-values from each simulation
    p_values = np.zeros(nsim)
    
    # If repeated measures, we'll simulate baseline and follow-up measurements
    if repeated_measures:
        # Generate baseline and follow-up data
        for sim in range(nsim):
            # Create baseline data with same mean in both groups
            baseline1 = np.random.normal(mean1, std_dev, n1)
            baseline2 = np.random.normal(mean1, std_dev, n2)
            
            # Create follow-up data considering correlation and treatment effect
            # We use a multivariate normal to account for correlation between baseline and follow-up
            # Covariance = correlation * std_dev^2
            covariance = correlation * std_dev**2
            
            # Group 1 follow-up (control group, no effect)
            error1 = np.random.normal(0, std_dev * np.sqrt(1 - correlation**2), n1)
            followup1 = baseline1 * correlation + mean1 * (1 - correlation) + error1
            
            # Group 2 follow-up (treatment group, with effect delta = mean2 - mean1)
            error2 = np.random.normal(0, std_dev * np.sqrt(1 - correlation**2), n2)
            followup2 = baseline2 * correlation + mean2 * (1 - correlation) + error2
            
            if method == "change_score":
                # Analysis using change score
                change1 = followup1 - baseline1
                change2 = followup2 - baseline2
                
                # Perform t-test on change scores
                t_stat, p_value = stats.ttest_ind(change2, change1, equal_var=True)
                p_values[sim] = p_value
                
            elif method == "ancova":
                # Simplified ANCOVA analysis using linear regression
                # Combine data for regression
                y = np.concatenate([followup1, followup2])
                x_baseline = np.concatenate([baseline1, baseline2])
                x_group = np.concatenate([np.zeros(n1), np.ones(n2)])  # 0 for control, 1 for treatment
                
                # Create design matrix for regression
                X = np.column_stack([np.ones(n1 + n2), x_baseline, x_group])
                
                # Fit linear regression using OLS
                # y = b0 + b1*baseline + b2*group
                beta = np.linalg.lstsq(X, y, rcond=None)[0]
                
                # Extract residuals
                residuals = y - X @ beta
                
                # Calculate MSE
                mse = np.sum(residuals**2) / (n1 + n2 - 3)  # df = n1 + n2 - 3 parameters
                
                # Calculate standard error of the treatment effect (beta[2])
                X_transposed_X_inv = np.linalg.inv(X.T @ X)
                se_beta2 = np.sqrt(mse * X_transposed_X_inv[2, 2])
                
                # Calculate t-statistic and p-value for treatment effect
                t_stat = beta[2] / se_beta2
                p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n1 + n2 - 3))
                p_values[sim] = p_value
    else:
        # Standard parallel design without repeated measures
        for sim in range(nsim):
            # Generate data for each group
            data1 = np.random.normal(mean1, std_dev, n1)
            data2 = np.random.normal(mean2, std_dev, n2)
            
            # Perform t-test
            t_stat, p_value = stats.ttest_ind(data2, data1, equal_var=True)
            p_values[sim] = p_value
    
    # Calculate power
    power = np.mean(p_values < alpha)
    
    return {
        "power": power,
        "parameters": {
            "n1": n1,
            "n2": n2,
            "mean1": mean1,
            "mean2": mean2,
            "std_dev": std_dev,
            "nsim": nsim,
            "alpha": alpha,
            "repeated_measures": repeated_measures,
            "correlation": correlation if repeated_measures else None,
            "method": method if repeated_measures else None
        }
    }


def sample_size_continuous(delta, std_dev, power=0.8, alpha=0.05, allocation_ratio=1.0, nsim=1000, min_n=10, max_n=1000, step=10, repeated_measures=False, correlation=0.5, method="change_score"):
    """
    Calculate sample size required for detecting a difference in means using simulation.
    
    Parameters
    ----------
    delta : float
        Minimum detectable effect (difference between means)
    std_dev : float
        Standard deviation of the outcome (assumed equal in both groups)
    power : float, optional
        Desired statistical power (1 - beta), by default 0.8
    alpha : float, optional
        Significance level, by default 0.05
    allocation_ratio : float, optional
        Ratio of sample sizes (n2/n1), by default 1.0
    nsim : int, optional
        Number of simulations per sample size, by default 1000
    min_n : int, optional
        Minimum sample size to try for group 1, by default 10
    max_n : int, optional
        Maximum sample size to try for group 1, by default 1000
    step : int, optional
        Step size for incrementing sample size, by default 10
    repeated_measures : bool, optional
        Whether to simulate repeated measures design with baseline and follow-up measurements,
        by default False
    correlation : float, optional
        Correlation between baseline and follow-up measurements, by default 0.5
    method : str, optional
        Analysis method for repeated measures: "change_score" or "ancova", by default "change_score"
    
    Returns
    -------
    dict
        Dictionary containing the required sample sizes for each group and total
    
    Notes
    -----
    This function uses simulation to find the smallest sample size that achieves
    the desired power. It tries sample sizes from min_n to max_n in steps of step.
    """
    # Simple iterative approach trying increasing sample sizes
    # No fancy optimizations, just brute-force search with early stopping
    print(f"Starting sample size search with min_n={min_n}, max_n={max_n}, step={step}")
    
    # Enforce max_n constraint 
    if max_n > 1000:
        print(f"Warning: max_n={max_n} is very large, capping at 1000 for efficiency")
        max_n = 1000
        
    # Store results for each sample size
    results = []
    
    # Try different sample sizes starting from the minimum
    for n1 in range(min_n, max_n + 1, step):
        n2 = math.ceil(n1 * allocation_ratio)
        print(f"Testing n1={n1}, n2={n2}...")
        
        # For each sample size, run nsim simulations
        significant_count = 0
        
        for sim in range(nsim):
            # Generate data based on whether we're doing repeated measures or not
            if repeated_measures:
                # Simulate correlated baseline and follow-up using multivariate normal distribution
                # Control group (no treatment effect)
                mean1 = [0, 0]  # Same mean at baseline and follow-up
                cov1 = [
                    [std_dev**2, correlation * std_dev**2],
                    [correlation * std_dev**2, std_dev**2]
                ]
                data1 = np.random.multivariate_normal(mean1, cov1, size=n1)
                baseline1 = data1[:, 0]
                followup1 = data1[:, 1]
                
                # Treatment group (with treatment effect at follow-up)
                mean2 = [0, delta]  # Mean increases by delta at follow-up
                cov2 = [
                    [std_dev**2, correlation * std_dev**2],
                    [correlation * std_dev**2, std_dev**2]
                ]
                data2 = np.random.multivariate_normal(mean2, cov2, size=n2)
                baseline2 = data2[:, 0]
                followup2 = data2[:, 1]
                
                # Analyze according to specified method
                if method == "change_score":
                    # Paired t-test on change scores
                    change1 = followup1 - baseline1
                    change2 = followup2 - baseline2
                    _, p_value = stats.ttest_ind(change2, change1)
                else:  # ANCOVA
                    # Simple ANCOVA implementation via regression
                    y = np.concatenate([followup1, followup2])
                    x_baseline = np.concatenate([baseline1, baseline2])
                    x_group = np.concatenate([np.zeros(n1), np.ones(n2)])
                    X = np.column_stack([np.ones(n1 + n2), x_baseline, x_group])
                    beta = np.linalg.lstsq(X, y, rcond=None)[0]
                    residuals = y - X @ beta
                    mse = np.sum(residuals**2) / (n1 + n2 - 3)
                    X_inv = np.linalg.inv(X.T @ X)
                    se_beta = np.sqrt(mse * X_inv[2, 2])
                    t_stat = beta[2] / se_beta
                    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n1 + n2 - 3))
            else:
                # Standard parallel groups t-test
                group1 = np.random.normal(0, std_dev, n1)
                group2 = np.random.normal(delta, std_dev, n2)
                _, p_value = stats.ttest_ind(group2, group1)
            
            # Count significant results
            if p_value < alpha:
                significant_count += 1
        
        # Calculate empirical power
        current_power = significant_count / nsim
        print(f"Power for n1={n1}, n2={n2}: {current_power:.4f}")
        
        # Store this result
        results.append({
            "n1": n1,
            "n2": n2,
            "power": current_power
        })
        
        # If we've achieved the desired power, we can stop
        if current_power >= power:
            print(f"Found sufficient power ({current_power:.4f} >= {power}) at n1={n1}, n2={n2}")
            break
    
    # If we didn't find a solution, use the last tried sample size
    if not results or results[-1]["power"] < power:
        if results:
            n1 = results[-1]["n1"]
            n2 = results[-1]["n2"]
            achieved_power = results[-1]["power"]
            print(f"WARNING: Reached max_n={max_n} but only achieved power {achieved_power:.4f} < {power}")
        else:
            n1 = max_n
            n2 = math.ceil(max_n * allocation_ratio)
            achieved_power = 0.0
            print(f"WARNING: No valid results found, returning max_n={max_n}")
        
        return {
            "n1": n1,
            "n2": n2,
            "total_n": n1 + n2,
            "warning": f"Maximum sample size {max_n} reached but power only {achieved_power:.3f} vs target {power:.3f}",
            "parameters": {
                "delta": delta,
                "std_dev": std_dev,
                "power": power,
                "achieved_power": achieved_power,
                "alpha": alpha,
                "allocation_ratio": allocation_ratio,
                "nsim": nsim,
                "repeated_measures": repeated_measures,
                "correlation": correlation if repeated_measures else None,
                "method": method if repeated_measures else None,
                "max_n_limit_reached": True
            }
        }
    
    # Return the result with the smallest sample size that achieved the desired power
    for result in results:
        if result["power"] >= power:
            print(f"SUCCESS: Found sample size n1={result['n1']}, n2={result['n2']} with power {result['power']:.4f} >= {power}")
            return {
                "n1": result["n1"],
                "n2": result["n2"],
                "total_n": result["n1"] + result["n2"],
                "parameters": {
                    "delta": delta,
                    "std_dev": std_dev,
                    "power": power,
                    "achieved_power": result["power"],
                    "alpha": alpha,
                    "allocation_ratio": allocation_ratio,
                    "nsim": nsim,
                    "repeated_measures": repeated_measures,
                    "correlation": correlation if repeated_measures else None,
                    "method": method if repeated_measures else None,
                    "max_n_limit_reached": False
                }
            }


def min_detectable_effect_continuous(n1, n2, std_dev, power=0.8, nsim=1000, alpha=0.05, precision=0.01, repeated_measures=False, correlation=0.5, method="change_score"):
    """
    Calculate minimum detectable effect size using simulation-based approach and optimization.
    
    Parameters
    ----------
    n1 : int
        Sample size for group 1
    n2 : int
        Sample size for group 2
    std_dev : float
        Standard deviation of the outcome
    power : float, optional
        Desired statistical power (1 - beta), by default 0.8
    nsim : int, optional
        Number of simulations, by default 1000
    alpha : float, optional
        Significance level, by default 0.05
    precision : float, optional
        Desired precision for the effect size, by default 0.01
    repeated_measures : bool, optional
        Whether to simulate repeated measures design with baseline and follow-up measurements,
        by default False
    correlation : float, optional
        Correlation between baseline and follow-up measurements, by default 0.5
    method : str, optional
        Analysis method for repeated measures: "change_score" or "ancova", by default "change_score"
    
    Returns
    -------
    dict
        Dictionary containing the minimum detectable effect and input parameters
    """
    # Function to evaluate the distance from target power
    def power_distance(delta):
        sim_result = simulate_continuous(n1=n1, n2=n2, mean1=0, mean2=delta, std_dev=std_dev, nsim=nsim, alpha=alpha,
                                    repeated_measures=repeated_measures, correlation=correlation, method=method)
        return abs(sim_result["power"] - power)
    
    # Initial guess - use analytical formula as starting point
    z_alpha = stats.norm.ppf(1 - alpha/2)
    z_beta = stats.norm.ppf(power)
    initial_guess = (z_alpha + z_beta) * std_dev * np.sqrt(1/n1 + 1/n2)
    
    # Run optimization
    result = optimize.minimize_scalar(power_distance, 
                                     bounds=[initial_guess/10, initial_guess*5], 
                                     method="bounded",
                                     options={"xatol": precision})
    
    # Get final simulation to report detailed results
    final_sim = simulate_continuous(n1=n1, n2=n2, mean1=0, mean2=result.x, std_dev=std_dev, nsim=nsim, alpha=alpha,
                                repeated_measures=repeated_measures, correlation=correlation, method=method)
    final_power = final_sim["power"]
    
    return {
        "mde": result.x,
        "achieved_power": final_power,
        "parameters": {
            "n1": n1,
            "n2": n2,
            "std_dev": std_dev,
            "power": power,
            "nsim": nsim,
            "alpha": alpha,
            "repeated_measures": repeated_measures,
            "correlation": correlation if repeated_measures else None,
            "method": method if repeated_measures else None
        },
        "optimization_info": {
            "initial_guess": initial_guess,
            "n_iterations": result.nit,
            "n_function_calls": result.nfev,
            "success": result.success
        }
    }


def simulate_binary(n1, n2, p1, p2, nsim=1000, alpha=0.05):
    """
    Simulate a parallel RCT with binary outcome and estimate power.
    
    Parameters
    ----------
    n1 : int
        Sample size for group 1
    n2 : int
        Sample size for group 2
    p1 : float
        Proportion in group 1 (between 0 and 1)
    p2 : float
        Proportion in group 2 (between 0 and 1)
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
        group1 = np.random.binomial(1, p1, n1)
        group2 = np.random.binomial(1, p2, n2)
        
        # Calculate proportions
        prop1 = np.mean(group1)
        prop2 = np.mean(group2)
        
        # Calculate standard error under null hypothesis
        p_pooled = (sum(group1) + sum(group2)) / (n1 + n2)
        se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n1 + 1/n2))
        
        # Calculate z-statistic
        if se > 0:
            z_stat = (prop2 - prop1) / se
            # Calculate p-value
            p_val = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        else:
            # If standard error is 0 (e.g., both groups have same outcome), set p-value to 1
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
            "n1": n1,
            "n2": n2,
            "p1": p1,
            "p2": p2,
            "alpha": alpha
        }
    }
