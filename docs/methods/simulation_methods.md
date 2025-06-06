# Simulation Methods for Study Design

This document details the comprehensive simulation-based methodology implemented in DesignPower for calculating sample size, power, and minimum detectable effect across various study designs, including classical Monte Carlo methods and modern Bayesian inference approaches.

## Background

While analytical formulas provide exact results under specific assumptions, simulation methods offer several advantages:

1. **Flexibility**: Can handle complex study designs that may not have closed-form solutions
2. **Realistic scenarios**: Can incorporate real-world complexities such as missing data, non-normal distributions, and heterogeneous treatment effects
3. **Robustness**: Can evaluate the impact of violations of standard assumptions
4. **Validation**: Can verify analytical results and provide confidence intervals for power estimates
5. **Bayesian Inference**: Provides full uncertainty quantification through posterior distributions

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

## Bayesian Simulation Methods

DesignPower implements state-of-the-art Bayesian inference for power analysis, offering both full MCMC and fast approximate methods. Bayesian approaches provide complete uncertainty quantification and natural incorporation of prior information.

### Hierarchical Bayesian Model for Cluster RCTs

The Bayesian framework models cluster RCTs using hierarchical models that explicitly account for clustering:

```
Level 1 (Individual): y_ij ~ Normal(μ_ij, σ_e²)
Level 2 (Cluster):    μ_ij = α + β×treatment_j + u_j
Level 3 (Population): u_j ~ Normal(0, σ_u²)

Priors:
- α ~ Normal(0, 10²)     # Intercept
- β ~ Normal(0, 10²)     # Treatment effect
- σ_u ~ HalfStudentT(3, 0, 2.5)  # Between-cluster SD
- σ_e ~ HalfStudentT(3, 0, 2.5)  # Within-cluster SD
```

### Bayesian Inference Backends

#### 1. Stan (CmdStanPy) - Full MCMC
- **Method**: Hamiltonian Monte Carlo with NUTS sampler
- **Strengths**: Gold standard for Bayesian inference, excellent convergence
- **Use Case**: Research-quality analysis, final results
- **Requirements**: `pip install cmdstanpy`

```python
# Example usage
results = simulation_continuous.power_continuous_sim(
    n_clusters=10, cluster_size=20, icc=0.05,
    mean1=3.0, mean2=3.5, std_dev=1.2,
    analysis_model="bayes", bayes_backend="stan",
    bayes_draws=1000, bayes_warmup=1000
)
```

#### 2. PyMC - Full MCMC
- **Method**: NUTS sampler with pure Python implementation
- **Strengths**: Easier installation, excellent Python integration
- **Use Case**: Development, exploration, Python-native workflows
- **Requirements**: `pip install pymc`

```python
# PyMC backend
results = simulation_continuous.power_continuous_sim(
    n_clusters=10, cluster_size=20, icc=0.05,
    mean1=3.0, mean2=3.5, std_dev=1.2,
    analysis_model="bayes", bayes_backend="pymc",
    bayes_draws=1000, bayes_warmup=1000
)
```

#### 3. Variational Bayes - Fast Approximation
- **Method**: Laplace approximation using MAP estimation + Hessian
- **Speed**: 10-100x faster than MCMC
- **Accuracy**: Good approximation for well-behaved posteriors
- **Use Case**: Rapid exploration, parameter tuning, CI/CD testing
- **Requirements**: scipy (always available)

**Algorithm**:
1. Find Maximum A Posteriori (MAP) estimate via optimization
2. Compute Hessian matrix at MAP for uncertainty quantification
3. Approximate posterior as multivariate normal
4. Sample from approximate posterior

```python
# Variational approximation
results = simulation_continuous.power_continuous_sim(
    n_clusters=10, cluster_size=20, icc=0.05,
    mean1=3.0, mean2=3.5, std_dev=1.2,
    analysis_model="bayes", bayes_backend="variational",
    bayes_draws=1000  # Number of approximate samples
)
```

#### 4. Approximate Bayesian Computation (ABC) - Ultra-Lightweight
- **Method**: Simulation-based likelihood-free inference
- **Speed**: Very fast, minimal memory footprint
- **Use Case**: Web deployment, free hosting, educational demos
- **Requirements**: scipy (always available)

**Algorithm**:
1. Define summary statistics from observed data
2. Sample parameters from priors
3. Simulate data using sampled parameters
4. Accept parameters if simulated statistics ≈ observed statistics
5. Repeat until sufficient samples collected

```python
# ABC for web deployment
results = simulation_continuous.power_continuous_sim(
    n_clusters=10, cluster_size=20, icc=0.05,
    mean1=3.0, mean2=3.5, std_dev=1.2,
    analysis_model="bayes", bayes_backend="abc",
    bayes_draws=1000  # Number of ABC samples
)
```

### Bayesian Inference Methods

All Bayesian backends support three inference approaches for determining statistical significance:

#### 1. Credible Intervals (Default)
- **Method**: 95% posterior credible interval excludes zero
- **Interpretation**: "95% probability that true effect is non-zero"
- **Advantage**: Most interpretable, standard Bayesian approach

```python
significant = (ci_lower > 0) or (ci_upper < 0)
```

#### 2. Posterior Probability
- **Method**: Probability that effect is in favorable direction
- **Criterion**: P(β > 0) > 97.5% or P(β > 0) < 2.5%
- **Interpretation**: Direct probability of treatment benefit

```python
prob_positive = (beta_samples > 0).mean()
significant = prob_positive > 0.975 or prob_positive < 0.025
```

#### 3. Region of Practical Equivalence (ROPE)
- **Method**: Test if effect is practically significant
- **Criterion**: P(|β| < δ) < 5% where δ is equivalence threshold
- **Advantage**: Incorporates practical significance, not just statistical

```python
rope_half_width = 0.1 * std_dev  # 10% of SD as ROPE
prob_rope = ((beta_samples > -rope_half_width) & 
             (beta_samples < rope_half_width)).mean()
significant = prob_rope < 0.05
```

### Smart Resource Management

DesignPower automatically manages computational resources based on environment constraints:

#### Environment Detection
```python
def _detect_resource_constraints():
    """Detect resource-constrained environments."""
    memory_gb = psutil.virtual_memory().available / (1024**3)
    cpu_count = psutil.cpu_count()
    
    # Suggest lightweight methods for constrained environments
    is_constrained = memory_gb < 2.0 or cpu_count <= 1
    return is_constrained
```

#### Automatic Fallback Hierarchy
1. **Primary**: User-selected backend (Stan/PyMC/Variational/ABC)
2. **Fallback 1**: If primary unavailable → Variational Bayes
3. **Fallback 2**: If scipy unavailable → Classical t-test
4. **User notification**: Clear warnings about fallback usage

### Performance Comparison

| Backend | Speed | Memory | Accuracy | Convergence | Dependencies |
|---------|-------|--------|----------|-------------|--------------|
| **Stan** | ⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Heavy |
| **PyMC** | ⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Medium |
| **Variational** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | Light |
| **ABC** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | Light |

### Convergence Diagnostics

#### MCMC Diagnostics (Stan/PyMC)
- **R-hat**: Potential Scale Reduction Factor < 1.1
- **Effective Sample Size**: Independent samples from chains
- **Trace plots**: Visual inspection of chain mixing
- **Energy diagnostics**: HMC-specific convergence checks

#### Approximate Method Diagnostics
- **Variational**: Hessian conditioning, optimization success
- **ABC**: Acceptance rate, tolerance sensitivity
- **Warning system**: Automatic detection of poor approximations

### Deployment Considerations

#### Local Development
- **Recommended**: Stan or PyMC for full accuracy
- **Fallback**: Variational for rapid iteration

#### Free Web Hosting (Streamlit Cloud, Heroku, etc.)
- **Recommended**: ABC or Variational
- **Avoid**: Full MCMC (resource constraints)

#### Production Research
- **Recommended**: Stan with multiple chains
- **Validation**: Compare with classical methods

#### Educational/Demo Use
- **Recommended**: ABC with clear limitation warnings
- **Advantage**: Works everywhere, minimal dependencies

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
