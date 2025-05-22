# Simulation Methods for Study Design

This document details the simulation-based methodology implemented in DesignPower for calculating sample size, power, and minimum detectable effect across various study designs.

## Background

While analytical formulas provide exact results under specific assumptions, simulation methods offer several advantages:

1. **Flexibility**: Can handle complex study designs that may not have closed-form solutions
2. **Realistic scenarios**: Can incorporate real-world complexities such as missing data, non-normal distributions, and heterogeneous treatment effects
3. **Robustness**: Can evaluate the impact of violations of standard assumptions
4. **Validation**: Can verify analytical results and provide confidence intervals for power estimates

## General Simulation Framework

DesignPower implements a Monte Carlo simulation framework that follows these general steps:

1. Generate synthetic data according to the study design and parameter specifications
2. Apply the planned statistical analysis to the synthetic data
3. Record whether the null hypothesis is rejected (power) or the parameter estimate (precision)
4. Repeat steps 1-3 many times to estimate the probability of rejecting the null hypothesis or the expected precision

## Implementation by Study Design

### Parallel RCT with Continuous Outcomes

For continuous outcomes, the simulation:

1. Generates individual-level data from normal distributions with specified means and standard deviations
2. Performs a two-sample t-test or specified alternative test
3. Records whether the p-value is less than the significance threshold

```python
def simulate_parallel_continuous(n1, n2, mean1, mean2, sd1, sd2, alpha=0.05, nsim=1000):
    # n1, n2: sample sizes
    # mean1, mean2: group means
    # sd1, sd2: group standard deviations
    # alpha: significance level
    # nsim: number of simulations
    
    reject_count = 0
    
    for _ in range(nsim):
        # Generate data
        group1 = np.random.normal(mean1, sd1, n1)
        group2 = np.random.normal(mean2, sd2, n2)
        
        # Perform t-test
        t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=(sd1 == sd2))
        
        # Record rejection
        if p_value < alpha:
            reject_count += 1
    
    # Calculate power
    power = reject_count / nsim
    return power
```

### Parallel RCT with Binary Outcomes

For binary outcomes, the simulation:

1. Generates binary outcomes using Bernoulli random variables with specified probabilities
2. Performs a chi-square test, Fisher's exact test, or other specified test
3. Records whether the p-value is less than the significance threshold

### Cluster RCT

For cluster randomized trials, the simulation:

1. Generates cluster-level random effects to account for intracluster correlation
2. Generates individual-level outcomes within clusters
3. Performs the appropriate analysis (e.g., mixed-effects model)
4. Records whether the p-value is less than the significance threshold

```python
def simulate_cluster_continuous(k, m, mean1, mean2, sigma_b, sigma_w, alpha=0.05, nsim=1000):
    # k: number of clusters per arm
    # m: number of individuals per cluster
    # mean1, mean2: group means
    # sigma_b: between-cluster standard deviation
    # sigma_w: within-cluster standard deviation
    # alpha: significance level
    # nsim: number of simulations
    
    reject_count = 0
    icc = sigma_b**2 / (sigma_b**2 + sigma_w**2)
    
    for _ in range(nsim):
        # Generate cluster-level random effects
        cluster_effects1 = np.random.normal(0, sigma_b, k)
        cluster_effects2 = np.random.normal(0, sigma_b, k)
        
        # Generate data for each cluster
        data = []
        group = []
        cluster_id = []
        
        for i in range(k):
            # Control group clusters
            for j in range(m):
                y = mean1 + cluster_effects1[i] + np.random.normal(0, sigma_w)
                data.append(y)
                group.append(0)
                cluster_id.append(i)
            
            # Treatment group clusters
            for j in range(m):
                y = mean2 + cluster_effects2[i] + np.random.normal(0, sigma_w)
                data.append(y)
                group.append(1)
                cluster_id.append(i + k)
        
        # Create DataFrame for analysis
        df = pd.DataFrame({'y': data, 'group': group, 'cluster': cluster_id})
        
        # Fit mixed-effects model
        model = sm.MixedLM(df['y'], sm.add_constant(df['group']), groups=df['cluster'])
        result = model.fit()
        
        # Record rejection
        p_value = result.pvalues[1]  # p-value for the group effect
        if p_value < alpha:
            reject_count += 1
    
    # Calculate power
    power = reject_count / nsim
    return power
```

### Repeated Measures Designs

For repeated measures designs, the simulation:

1. Generates correlated measurements for each subject using multivariate normal distributions with the specified correlation structure
2. Applies the treatment effect according to the design
3. Performs the appropriate analysis (e.g., ANCOVA, mixed model)
4. Records whether the p-value is less than the significance threshold

## Specialized Simulation Methods

### Sample Size Calculation

For sample size calculation, DesignPower uses an iterative approach:

1. Start with an initial sample size guess
2. Run simulations to estimate power at that sample size
3. If the estimated power is too low, increase the sample size; if too high, decrease it
4. Repeat until the desired power is achieved within a specified tolerance

To make this process more efficient, DesignPower implements:
- Binary search for faster convergence
- Interpolation between sample sizes
- Early stopping rules to avoid unnecessary simulations

### Minimum Detectable Effect Calculation

For minimum detectable effect calculation, a similar iterative approach is used:

1. Start with an initial effect size guess
2. Run simulations to estimate power at that effect size
3. If the estimated power is too low, increase the effect size; if too high, decrease it
4. Repeat until the desired power is achieved within a specified tolerance

### Precision-Based Sample Size Calculation

In addition to power-based calculations, DesignPower can perform precision-based sample size calculations through simulation:

1. Generate data and calculate the parameter estimate (e.g., mean difference, odds ratio)
2. Record the confidence interval width
3. Increase or decrease the sample size until the desired precision is achieved

## Advanced Features

### Non-normal Distributions

DesignPower can simulate from non-normal distributions to evaluate robustness to distributional assumptions:

1. Skewed distributions (e.g., log-normal, gamma)
2. Heavy-tailed distributions (e.g., t-distribution)
3. Mixture distributions

### Missing Data

DesignPower can simulate various missing data mechanisms:

1. Missing completely at random (MCAR)
2. Missing at random (MAR)
3. Missing not at random (MNAR)

The impact of these mechanisms on power and sample size can be evaluated under different analysis approaches:
- Complete case analysis
- Multiple imputation
- Mixed models

### Heterogeneous Treatment Effects

DesignPower can simulate heterogeneous treatment effects across:
- Subgroups defined by baseline characteristics
- Clusters in cluster randomized trials
- Time points in longitudinal studies

## Validation and Diagnostics

DesignPower includes several validation features for simulation-based calculations:

1. **Comparison with analytical formulas**: When analytical formulas are available, simulation results are compared to validate the approach
2. **Convergence diagnostics**: Monitoring of Monte Carlo error and convergence across iterations
3. **Sensitivity analyses**: Evaluation of how sensitive the results are to changes in key parameters

## Practical Recommendations

### Number of Simulations

The precision of power estimates depends on the number of simulations. DesignPower provides:
- Default simulation counts that balance precision and computational time
- Guidelines for increasing simulation counts for final calculations
- Confidence intervals for power estimates based on the binomial distribution

### Seed Setting

For reproducibility, DesignPower allows setting the random seed for simulations:
- Each simulation run with the same seed produces identical results
- Different seeds should be used for sensitivity analyses

### Computational Considerations

Simulation-based calculations can be computationally intensive. DesignPower implements:
- Parallel processing for faster calculations
- Progressive refinement to focus simulations where they provide the most information
- Caching of intermediate results

## References

1. Burton A, Altman DG, Royston P, Holder RL. The design of simulation studies in medical statistics. Stat Med. 2006;25(24):4279-4292.

2. Morris TP, White IR, Crowther MJ. Using simulation studies to evaluate statistical methods. Stat Med. 2019;38(11):2074-2102.

3. Landau S, Stahl D. Sample size and power calculations for medical studies by simulation when closed form expressions are not available. Stat Methods Med Res. 2013;22(3):324-345.

4. Feiveson AH. Power by simulation. Stata J. 2002;2(2):107-124.
