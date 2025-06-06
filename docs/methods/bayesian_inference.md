# Bayesian Inference Methods in DesignPower

This document provides comprehensive details on the Bayesian inference capabilities implemented in DesignPower for clinical trial power analysis, including theoretical foundations, computational methods, and practical applications.

## Overview

DesignPower implements state-of-the-art Bayesian inference for power analysis in clinical trials, with particular emphasis on cluster randomized controlled trials (cluster RCTs). The framework provides:

- **Multiple backends**: Stan, PyMC, Variational Bayes, and ABC
- **Flexible inference**: Credible intervals, posterior probability, and ROPE testing
- **Resource awareness**: Automatic selection based on computational constraints
- **Production ready**: Validated against classical methods with comprehensive diagnostics

## Theoretical Foundation

### Bayesian Hierarchical Model for Cluster RCTs

The Bayesian approach models cluster RCTs using a three-level hierarchical structure that explicitly accounts for the nested nature of the data:

```
Level 1 (Individual):  y_ij | μ_ij, σ_e² ~ Normal(μ_ij, σ_e²)
Level 2 (Cluster):     μ_ij = α + β × treatment_j + u_j
Level 3 (Population):  u_j | σ_u² ~ Normal(0, σ_u²)
```

Where:
- `y_ij` is the outcome for individual `i` in cluster `j`
- `α` is the overall intercept
- `β` is the treatment effect (primary parameter of interest)
- `u_j` is the random effect for cluster `j`
- `σ_e²` is the within-cluster variance
- `σ_u²` is the between-cluster variance

### Prior Specifications

DesignPower uses weakly informative priors that regularize the model while allowing the data to dominate:

```
α ~ Normal(0, 10²)                    # Intercept prior
β ~ Normal(0, 10²)                    # Treatment effect prior  
σ_u ~ HalfStudentT(3, 0, 2.5)        # Between-cluster SD prior
σ_e ~ HalfStudentT(3, 0, 2.5)        # Within-cluster SD prior
```

These priors are chosen to be:
- **Weakly informative**: Don't unduly influence results but prevent extreme values
- **Proper**: Ensure proper posterior distributions
- **Scale-appropriate**: HalfStudentT priors for variance parameters
- **Robust**: Student-t distributions handle outliers better than normal priors

### Intracluster Correlation (ICC)

The ICC emerges naturally from the hierarchical model:

```
ICC = σ_u² / (σ_u² + σ_e²)
```

This formulation automatically handles:
- **ICC uncertainty**: Full posterior distribution for ICC
- **Boundary conditions**: Proper behavior at ICC ≈ 0 or ICC ≈ 1
- **Identifiability**: Constraint through proper priors

## Computational Methods

### 1. Stan Backend (CmdStanPy)

#### Implementation
Stan uses Hamiltonian Monte Carlo (HMC) with the No-U-Turn Sampler (NUTS) for efficient exploration of the posterior distribution.

**Stan Model Code**:
```stan
data {
    int<lower=1> N;                           // Total observations
    int<lower=1> J;                           // Number of clusters
    array[N] int<lower=1, upper=J> cluster;   // Cluster indicators
    vector[N] y;                              // Outcomes
    vector[N] treat;                          // Treatment indicators
}
parameters {
    real alpha;                               // Intercept
    real beta;                                // Treatment effect
    vector[J] u_raw;                         // Raw cluster effects
    real<lower=0> sigma_u;                   // Between-cluster SD
    real<lower=0> sigma_e;                   // Within-cluster SD
}
transformed parameters {
    vector[J] u = u_raw * sigma_u;           // Non-centered parameterization
}
model {
    // Priors
    alpha ~ normal(0, 10);
    beta ~ normal(0, 10);
    u_raw ~ normal(0, 1);                    // Non-centered parameterization
    sigma_u ~ student_t(3, 0, 2.5);
    sigma_e ~ student_t(3, 0, 2.5);
    
    // Likelihood
    for (n in 1:N)
        y[n] ~ normal(alpha + beta * treat[n] + u[cluster[n]], sigma_e);
}
```

#### Advantages
- **Gold standard**: Most accurate Bayesian inference
- **Robust convergence**: Excellent diagnostic capabilities
- **Scalable**: Handles complex models efficiently
- **Well-tested**: Extensively validated in research

#### Diagnostics
- **R-hat**: Potential Scale Reduction Factor < 1.1
- **Effective Sample Size**: Independent samples > 400
- **Energy diagnostics**: HMC-specific convergence checks
- **Divergences**: Indication of sampling problems

### 2. PyMC Backend

#### Implementation
PyMC provides a pure Python implementation using the same NUTS sampler as Stan, with ArviZ for convergence diagnostics.

```python
with pm.Model() as model:
    # Priors
    alpha = pm.Normal("alpha", mu=0, sigma=10)
    beta = pm.Normal("beta", mu=0, sigma=10)
    
    # Cluster random effects
    sigma_u = pm.HalfStudentT("sigma_u", nu=3, sigma=2.5)
    u_raw = pm.Normal("u_raw", mu=0, sigma=1, shape=n_clusters)
    u = pm.Deterministic("u", u_raw * sigma_u)
    
    # Individual-level variance
    sigma_e = pm.HalfStudentT("sigma_e", nu=3, sigma=2.5)
    
    # Likelihood
    mu = alpha + beta * treatment + u[cluster_ids]
    y = pm.Normal("y", mu=mu, sigma=sigma_e, observed=y_obs)
    
    # Sample
    trace = pm.sample(draws=1000, tune=1000, chains=4)
```

#### Advantages
- **Python native**: Easier installation and debugging
- **ArviZ integration**: Excellent visualization and diagnostics
- **Flexible**: Easy to modify and extend models
- **Active development**: Rapid incorporation of new methods

### 3. Variational Bayes - Fast Approximation

#### Implementation
Variational Bayes uses the Laplace approximation to approximate the posterior distribution around the Maximum A Posteriori (MAP) estimate.

**Algorithm**:
1. **Find MAP**: Optimize the log posterior to find the mode
2. **Compute Hessian**: Second derivatives provide curvature information
3. **Approximate posterior**: Multivariate normal with mean=MAP, cov=inv(Hessian)
4. **Sample**: Draw samples from the approximate posterior

```python
def _fit_variational_bayes(df, n_samples=1000):
    # Negative log posterior for optimization
    def neg_log_posterior(params):
        alpha, beta = params[:2]
        u = params[2:]
        
        # Log likelihood
        mu = alpha + beta * treatment + u[cluster_ids]
        log_lik = -0.5 * np.sum((y - mu)**2) / sigma_e**2
        
        # Log priors
        log_prior = (-0.5 * (alpha**2 + beta**2) / 100 +  # Normal priors
                     -0.5 * np.sum(u**2) / sigma_u**2)     # Cluster effects
        
        return -(log_lik + log_prior)
    
    # Find MAP estimate
    result = minimize(neg_log_posterior, init_params, method='BFGS')
    
    # Approximate Hessian and sample
    hessian = compute_hessian(neg_log_posterior, result.x)
    cov = np.linalg.inv(hessian)
    samples = multivariate_normal.rvs(mean=result.x, cov=cov, size=n_samples)
    
    return samples[:, 1]  # Return treatment effect samples
```

#### Advantages
- **Speed**: 10-100x faster than MCMC
- **Deterministic**: No random seed dependence
- **Memory efficient**: Minimal memory footprint
- **Always available**: Only requires scipy

#### Limitations
- **Approximate**: Not exact posterior samples
- **Gaussian assumption**: May poorly approximate skewed posteriors
- **Local optima**: Optimization might find local rather than global maximum

### 4. Approximate Bayesian Computation (ABC)

#### Implementation
ABC provides likelihood-free inference by matching summary statistics between observed and simulated data.

**Algorithm**:
1. **Choose summary statistics**: Capture essential data features
2. **Sample from priors**: Generate parameter proposals
3. **Simulate data**: Use proposed parameters to generate synthetic data
4. **Compute distance**: Compare summary statistics
5. **Accept/reject**: Keep parameters if distance < tolerance
6. **Iterate**: Repeat until sufficient samples collected

```python
def _fit_abc_bayes(df, n_samples=1000, tolerance=0.1):
    # Observed summary statistics
    obs_stats = compute_summary_stats(df)
    
    accepted_samples = []
    for _ in range(max_attempts):
        # Sample from priors
        alpha = np.random.normal(0, 10)
        beta = np.random.normal(0, 10)  # Target parameter
        sigma_u = np.abs(np.random.normal(0, 2))
        sigma_e = np.abs(np.random.normal(0, 2))
        
        # Simulate data
        sim_data = simulate_cluster_data(alpha, beta, sigma_u, sigma_e, df.structure)
        sim_stats = compute_summary_stats(sim_data)
        
        # Accept if close enough
        distance = np.sqrt(np.sum((obs_stats - sim_stats)**2))
        if distance < tolerance:
            accepted_samples.append(beta)
    
    return np.array(accepted_samples)
```

#### Summary Statistics
For cluster RCTs, ABC uses these summary statistics:
- **Group means**: Mean outcome by treatment group
- **Overall variance**: Total outcome variance
- **Cluster-level variance**: Variance of cluster means
- **ICC estimate**: Sample-based ICC calculation

#### Advantages
- **Likelihood-free**: No need to specify exact likelihood
- **Ultra-lightweight**: Minimal computational requirements
- **Robust**: Works even with model misspecification
- **Web-friendly**: Suitable for resource-constrained environments

#### Limitations
- **Choice of summaries**: Results depend on summary statistic selection
- **Tolerance tuning**: Too strict = low acceptance, too loose = poor approximation
- **Curse of dimensionality**: Efficiency decreases with parameter dimensionality

## Inference Methods

All Bayesian backends support three approaches for determining statistical significance:

### 1. Credible Intervals (Default)

**Method**: Check if 95% posterior credible interval excludes zero

```python
ci_lower = np.percentile(beta_samples, 2.5)
ci_upper = np.percentile(beta_samples, 97.5)
significant = (ci_lower > 0) or (ci_upper < 0)
```

**Interpretation**: "We are 95% confident the true treatment effect is non-zero"

**Advantages**:
- Most interpretable for clinical audiences
- Direct uncertainty quantification
- Standard Bayesian approach

### 2. Posterior Probability

**Method**: Calculate probability that effect is in favorable direction

```python
prob_positive = (beta_samples > 0).mean()
significant = prob_positive > 0.975 or prob_positive < 0.025
```

**Interpretation**: "There is a 97.5% probability the treatment is beneficial"

**Advantages**:
- Direct probability statements
- Natural decision-theoretic interpretation
- Easy to communicate to stakeholders

### 3. Region of Practical Equivalence (ROPE)

**Method**: Test if effect is practically significant, not just statistically significant

```python
rope_half_width = 0.1 * std_dev  # 10% of SD as practical threshold
prob_rope = ((beta_samples > -rope_half_width) & 
             (beta_samples < rope_half_width)).mean()
significant = prob_rope < 0.05  # Less than 5% probability of practical equivalence
```

**Interpretation**: "Less than 5% probability the effect is practically negligible"

**Advantages**:
- Incorporates practical significance
- Addresses "statistical significance but clinical irrelevance"
- More conservative than traditional p-values

## Smart Resource Management

### Environment Detection

DesignPower automatically detects computational constraints and suggests appropriate methods:

```python
def _detect_resource_constraints():
    """Detect resource-constrained environments."""
    try:
        memory_gb = psutil.virtual_memory().available / (1024**3)
        cpu_count = psutil.cpu_count()
        
        # Heuristics for constrained environments
        is_constrained = memory_gb < 2.0 or cpu_count <= 1
        return is_constrained
    except:
        return False  # Conservative default
```

### Automatic Fallback Hierarchy

1. **User Selection**: Stan, PyMC, Variational, or ABC
2. **Availability Check**: Is selected backend available?
3. **Fallback 1**: If Stan/PyMC unavailable → Variational Bayes
4. **Fallback 2**: If scipy unavailable → Classical t-test
5. **User Notification**: Clear warnings about fallback usage

### Resource-Aware Recommendations

| Environment | Recommended | Rationale |
|-------------|-------------|-----------|
| **Local Development** | Stan/PyMC | Full accuracy for exploration |
| **Free Web Hosting** | ABC/Variational | Resource constraints |
| **Production Research** | Stan | Gold standard for publications |
| **CI/CD Testing** | Variational | Fast, deterministic |
| **Educational Demos** | ABC | Works everywhere, minimal deps |

## Validation and Quality Assurance

### Comparison with Classical Methods

All Bayesian implementations are validated against classical approaches:

```python
# Classical power calculation
classical_power = analytical_continuous.power_continuous(
    n_clusters=10, cluster_size=20, icc=0.05,
    mean1=3.0, mean2=3.5, std_dev=1.2
)

# Bayesian power calculation  
bayesian_power = simulation_continuous.power_continuous_sim(
    n_clusters=10, cluster_size=20, icc=0.05,
    mean1=3.0, mean2=3.5, std_dev=1.2,
    analysis_model="bayes", bayes_backend="stan"
)

# Should be approximately equal
assert abs(classical_power - bayesian_power["power"]) < 0.05
```

### Convergence Diagnostics

#### MCMC Diagnostics
- **R-hat < 1.1**: Chains have converged to same distribution
- **ESS > 400**: Sufficient independent samples
- **No divergences**: Sampler explored posterior correctly
- **Energy diagnostics**: HMC-specific checks

#### Approximate Method Diagnostics
- **Optimization success**: MAP finding converged
- **Hessian conditioning**: Posterior approximation quality
- **ABC acceptance rate**: Efficiency of simulation
- **Tolerance sensitivity**: Robustness to parameter choices

### Reproducibility

```python
# Set seeds for reproducibility
np.random.seed(42)

# Stan/PyMC: automatic seed handling
results_stan = power_continuous_sim(..., bayes_backend="stan")

# Variational: deterministic given data
results_var = power_continuous_sim(..., bayes_backend="variational")

# ABC: controlled randomness
results_abc = power_continuous_sim(..., bayes_backend="abc")
```

## Best Practices

### Method Selection Guide

1. **Research Publications**: Use Stan with multiple chains and thorough diagnostics
2. **Exploratory Analysis**: Use Variational Bayes for rapid iteration
3. **Web Deployment**: Use ABC for broad accessibility
4. **Method Comparison**: Compare all methods to ensure consistency

### Diagnostic Checklist

#### Before Analysis
- [ ] Verify prior specifications are appropriate
- [ ] Check data structure and missingness patterns
- [ ] Ensure sufficient cluster/sample sizes

#### During Analysis
- [ ] Monitor convergence diagnostics in real-time
- [ ] Check for warnings or errors
- [ ] Verify reasonable parameter estimates

#### After Analysis
- [ ] Compare with classical methods when available
- [ ] Perform sensitivity analysis on priors
- [ ] Validate using simulation studies

### Reporting Guidelines

When reporting Bayesian analyses, include:

1. **Method specification**: Backend used, convergence diagnostics
2. **Prior justification**: Rationale for prior choices
3. **Inference approach**: Credible interval, posterior probability, or ROPE
4. **Uncertainty quantification**: Full posterior summaries
5. **Sensitivity analysis**: Robustness to modeling choices

## Future Developments

### Planned Enhancements

1. **Additional Models**: Binary outcomes, survival analysis
2. **Advanced Priors**: Informative priors from previous studies
3. **Model Comparison**: Bayesian model selection and averaging
4. **Adaptive Designs**: Bayesian adaptive trial designs
5. **Computational Improvements**: GPU acceleration, approximate inference

### Research Directions

1. **Robustness Studies**: Impact of prior specification
2. **Calibration Studies**: Frequentist properties of Bayesian methods
3. **Practical Guidelines**: When to use which method
4. **Computational Optimization**: Faster approximate inference

## References

1. Gelman, A., et al. (2013). *Bayesian Data Analysis*. CRC press.

2. Carpenter, B., et al. (2017). Stan: A probabilistic programming language. *Journal of Statistical Software*, 76(1).

3. Salvatier, J., et al. (2016). Probabilistic programming in Python using PyMC3. *PeerJ Computer Science*, 2, e55.

4. Blei, D. M., et al. (2017). Variational inference: A review for statisticians. *Journal of the American Statistical Association*, 112(518), 859-877.

5. Beaumont, M. A. (2010). Approximate Bayesian computation in evolution and ecology. *Annual Review of Ecology, Evolution, and Systematics*, 41, 379-406.

6. Kruschke, J. K., & Liddell, T. M. (2018). The Bayesian New Statistics: Hypothesis testing, estimation, meta-analysis, and power analysis from a Bayesian perspective. *Psychonomic Bulletin & Review*, 25(1), 178-206.

7. Spiegelhalter, D. J., et al. (2004). Bayesian measures of model complexity and fit. *Journal of the Royal Statistical Society Series B*, 64(4), 583-639.