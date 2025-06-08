# Cluster RCT Analysis Examples

*Practical implementation examples for different cluster scenarios*

## Overview

This document provides concrete examples of how to analyze cluster RCTs with different characteristics using DesignPower's advanced statistical methods. Each example includes the rationale for method selection and interpretation guidance.

## Example 1: Very Small Cluster Trial (Healthcare Quality Intervention)

### Study Design
- **Intervention**: Electronic health record reminder system
- **Clusters**: 7 primary care clinics per arm (14 total)
- **Outcome**: Binary (guideline adherence rate)
- **Expected baseline rate**: 40%
- **Target improvement**: 15 percentage points

### Recommended Analysis
```python
from core.designs.cluster_rct.simulation_binary import simulate_cluster_binary

params = {
    'n_clusters': 7,           # Very small number
    'cluster_size': 200,       # Large clinic size
    'p1': 0.40,               # Baseline adherence rate
    'p2': 0.55,               # Target improved rate
    'icc': 0.03,              # Moderate clustering
    'analysis_model': 'ttest', # Gold standard for small clusters
    'alpha': 0.05,
    'n_simulations': 1000
}

result = simulate_cluster_binary(params)
```

### Why This Approach?
1. **Small clusters (7 per arm)**: Cluster-level analysis is the gold standard
2. **Robust method**: Not affected by distribution assumptions
3. **Simple interpretation**: Clear degrees of freedom (14 - 2 = 12)
4. **Literature support**: Recommended by Murray (1998), Donner & Klar (2000)

### Expected Output Interpretation
```python
# Typical results:
{
    'power': 0.82,
    'analysis_method': 'ttest',
    'effective_sample_size': 133,  # Adjusted for clustering
    'warnings': ['Total clusters (14) below recommended minimum (40)']
}
```

**Key Points**:
- Power calculation accounts for clustering effect
- Warning acknowledges small cluster limitation
- Method automatically robust to convergence issues

### 🎯 Permutation Test Alternative (Exact Inference)
```python
# For exact inference without distributional assumptions
params_permutation = params.copy()
params_permutation.update({
    'analysis_model': 'permutation',  # Exact permutation test
})

result_permutation = simulate_cluster_binary(params_permutation)
```

**Why use permutation tests?**
1. **Exact inference**: No distributional assumptions required
2. **Very small clusters**: Ideal for ≤10 clusters per arm
3. **Non-normal outcomes**: Robust to any outcome distribution
4. **Conservative**: Provides exact Type I error control

**Expected Results:**
- Exact p-values (not asymptotic approximations)
- Confidence intervals via permutation
- Robust to outliers and non-normality
- Slightly more conservative than t-test

---

## Example 2: Small Cluster Trial (School-Based Intervention)

### Study Design
- **Intervention**: Nutrition education program
- **Clusters**: 12 schools per arm (24 total)  
- **Outcome**: Continuous (BMI change after 6 months)
- **Expected effect**: 0.5 kg/m² reduction
- **Standard deviation**: 2.0 kg/m²

### Recommended Analysis
```python
from core.designs.cluster_rct.simulation_continuous import simulate_cluster_continuous

params = {
    'n_clusters': 12,              # Small but sufficient for GEE
    'cluster_size': 50,            # Average class size
    'mean1': 0.0,                  # Control group (no change)
    'mean2': -0.5,                 # Intervention group (reduction)
    'sd1': 2.0,                    # Standard deviation
    'icc': 0.05,                   # Typical for schools
    'analysis_model': 'gee',       # Optimal for this size
    'use_bias_correction': True,   # Essential for small clusters
    'alpha': 0.05,
    'n_simulations': 1000
}

result = simulate_cluster_continuous(params)
```

### Why This Approach?
1. **Small clusters (12 per arm)**: GEE with bias correction optimal
2. **Bias-corrected sandwich**: Validated for ≥9 clusters per arm (Li et al. 2018)
3. **Robust correlation**: Handles uncertainty in correlation structure
4. **Efficient**: More powerful than cluster-level analysis

### Alternative Approaches
```python
# Conservative alternative
params_conservative = params.copy()
params_conservative.update({
    'analysis_model': 'ttest',     # More conservative
})

# Advanced alternative  
params_advanced = params.copy()
params_advanced.update({
    'analysis_model': 'mixedlm',   # Linear mixed model
    'use_satterthwaite': True,     # Small-sample df correction
    'lmm_reml': True              # REML estimation
})
```

### Method Comparison
```python
# Run all three approaches
results_gee = simulate_cluster_continuous(params)
results_ttest = simulate_cluster_continuous(params_conservative)
results_lmm = simulate_cluster_continuous(params_advanced)

print(f"GEE Power: {results_gee['power']:.3f}")
print(f"T-test Power: {results_ttest['power']:.3f}")  
print(f"LMM Power: {results_lmm['power']:.3f}")
```

**Expected Results**:
- GEE: Highest power (0.85)
- LMM: Similar power (0.84)
- T-test: Lower power (0.78)

---

## Example 3: Medium Cluster Trial (Workplace Intervention)

### Study Design
- **Intervention**: Workplace wellness program
- **Clusters**: 20 workplaces per arm (40 total)
- **Outcome**: Binary (smoking cessation at 1 year)
- **Expected baseline rate**: 15%
- **Target improvement**: 8 percentage points

### Recommended Analysis
```python
from core.designs.cluster_rct.simulation_binary import simulate_cluster_binary

params = {
    'n_clusters': 20,              # Medium cluster size
    'cluster_size': 75,            # Average workplace size
    'p1': 0.15,                    # Baseline cessation rate
    'p2': 0.23,                    # Target improved rate
    'icc': 0.02,                   # Low clustering for behavior
    'analysis_model': 'mixedlm',   # Standard approach
    'use_satterthwaite': True,     # Small-sample correction
    'lmm_reml': True,             # REML estimation
    'alpha': 0.05,
    'n_simulations': 1000
}

result = simulate_cluster_binary(params)
```

### Why This Approach?
1. **Medium clusters (20 per arm)**: Mixed models with corrections optimal
2. **Satterthwaite approximation**: Better degrees of freedom than normal theory
3. **REML estimation**: Unbiased variance component estimates
4. **Standard practice**: Widely accepted in literature

### Enhanced Analysis with Multiple Methods
```python
# Compare methods for robustness
methods = ['mixedlm', 'gee', 'ttest']
corrections = [True, True, False]  # Satterthwaite, bias correction, N/A

results = {}
for method, correction in zip(methods, corrections):
    params_method = params.copy()
    params_method.update({
        'analysis_model': method,
        'use_satterthwaite': correction if method == 'mixedlm' else False,
        'use_bias_correction': correction if method == 'gee' else False
    })
    results[method] = simulate_cluster_binary(params_method)

# Compare power estimates
for method, result in results.items():
    print(f"{method.upper()}: Power = {result['power']:.3f}")
```

**Expected Convergence**:
- All methods should give similar power estimates (±0.02)
- Validates robustness of findings

---

## Example 4: Large Cluster Trial (Community Intervention)

### Study Design
- **Intervention**: Community diabetes prevention program
- **Clusters**: 35 communities per arm (70 total)
- **Outcome**: Continuous (HbA1c change)
- **Expected effect**: 0.3% reduction
- **Standard deviation**: 1.2%

### Recommended Analysis
```python
from core.designs.cluster_rct.simulation_continuous import simulate_cluster_continuous

params = {
    'n_clusters': 35,              # Large cluster trial
    'cluster_size': 100,           # Community participation
    'mean1': 7.5,                  # Baseline HbA1c
    'mean2': 7.2,                  # Target reduced HbA1c
    'sd1': 1.2,                    # Standard deviation
    'icc': 0.01,                   # Low community clustering
    'analysis_model': 'mixedlm',   # Standard mixed model
    'lmm_reml': True,             # REML estimation
    'alpha': 0.05,
    'n_simulations': 1000
}

result = simulate_cluster_continuous(params)
```

### Why This Approach?
1. **Large clusters (35 per arm)**: Large-sample theory applies
2. **Standard mixed model**: Maximum efficiency
3. **No special corrections needed**: Asymptotic methods valid
4. **REML for variance**: Unbiased variance component estimation

### Advanced Features Available
```python
# Can use advanced features with large clusters
params_advanced = params.copy()
params_advanced.update({
    'analysis_model': 'bayes',     # Bayesian analysis
    'bayes_backend': 'stan',       # Full MCMC
    'bayes_prior_icc': 'normal',   # Weakly informative prior
})

result_bayes = simulate_cluster_continuous(params_advanced)

# Compare frequentist vs Bayesian
print(f"Frequentist Power: {result['power']:.3f}")
print(f"Bayesian Power: {result_bayes['power']:.3f}")
```

---

## Example 5: Bayesian Analysis (Informative Priors)

### Study Design
- **Intervention**: Digital health intervention for diabetes management
- **Clusters**: 10 primary care practices per arm
- **Outcome**: Continuous (HbA1c reduction)
- **Previous studies**: Meta-analysis provides informative priors
- **Expected effect**: 0.4% reduction (95% CI: 0.2-0.6%)

### Bayesian Analysis with Informative Priors
```python
from core.designs.cluster_rct.simulation_continuous import simulate_cluster_continuous

# Primary Bayesian approach using prior knowledge
params_bayes = {
    'n_clusters': 10,              # Small-medium cluster trial
    'cluster_size': 80,            # Practice size
    'mean1': 8.0,                  # Baseline HbA1c
    'mean2': 7.6,                  # Target reduced HbA1c (0.4% reduction)
    'sd1': 1.5,                    # HbA1c standard deviation
    'icc': 0.02,                   # Low clustering for clinical measures
    'analysis_model': 'bayes',     # Primary choice for informed analysis
    'bayes_backend': 'pymc',       # Good for complex priors
    'bayes_prior_effect': 'normal', # Use prior knowledge
    'alpha': 0.05,
    'n_simulations': 1000
}

result_bayes = simulate_cluster_continuous(params_bayes)

# Compare with frequentist approach
params_freq = params_bayes.copy()
params_freq.update({
    'analysis_model': 'gee',
    'use_bias_correction': True
})

result_freq = simulate_cluster_continuous(params_freq)

print(f"Bayesian Power: {result_bayes['power']:.3f}")
print(f"Frequentist Power: {result_freq['power']:.3f}")
```

### Why Bayesian Approach?
1. **Incorporates prior knowledge**: Meta-analysis provides effect size distribution
2. **Uncertainty quantification**: Full posterior distribution available
3. **Decision framework**: Can incorporate costs and utilities
4. **Small cluster robustness**: Handles convergence issues automatically

### Interpretation Advantages
```python
# Bayesian provides richer information
{
    'power': 0.87,
    'power_ci': [0.84, 0.90],              # Credible interval for power
    'effect_posterior_mean': 0.42,         # Posterior mean effect
    'effect_posterior_ci': [0.35, 0.49],   # Effect uncertainty
    'prob_clinically_meaningful': 0.94,    # P(effect > 0.3%)
    'prob_superiority': 0.99               # P(effect > 0)
}
```

---

## Example 6: Complex Scenario (Convergence Issues)

### Study Design
- **Intervention**: Mental health screening protocol
- **Clusters**: 8 emergency departments per arm
- **Outcome**: Binary (screening completion rate)
- **Baseline rate**: 25%
- **Complex issue**: Rare outcome, small clusters, potential convergence problems

### Robust Analysis Strategy
```python
from core.designs.cluster_rct.simulation_binary import simulate_cluster_binary

# Primary approach: Conservative cluster-level analysis
params_primary = {
    'n_clusters': 8,               # Very small clusters
    'cluster_size': 150,           # Large ED volume
    'p1': 0.25,                    # Baseline screening rate
    'p2': 0.40,                    # Target improved rate
    'icc': 0.04,                   # Moderate clustering
    'analysis_model': 'ttest',     # Gold standard for small clusters
    'alpha': 0.05,
    'n_simulations': 1000
}

# Backup approach: Bayesian analysis
params_bayes = params_primary.copy()
params_bayes.update({
    'analysis_model': 'bayes',
    'bayes_backend': 'stan',       # Most robust for convergence
    'bayes_prior_icc': 'uniform',  # Conservative prior
})

# Run both analyses
result_primary = simulate_cluster_binary(params_primary)
result_bayes = simulate_cluster_binary(params_bayes)

print(f"Cluster t-test Power: {result_primary['power']:.3f}")
print(f"Bayesian Power: {result_bayes['power']:.3f}")
```

### Why This Strategy?
1. **Primary analysis**: Robust, simple, interpretable
2. **Bayesian backup**: Handles convergence issues automatically
3. **Convergence-proof**: Both methods work regardless of data characteristics
4. **Conservative**: Appropriate for very small cluster scenario

---

## Example 6: Method Selection Based on ICC

### High ICC Scenario (Classroom Intervention)
```python
# High ICC requires careful method selection
params_high_icc = {
    'n_clusters': 15,              # Moderate cluster number
    'cluster_size': 25,            # Classroom size
    'mean1': 75,                   # Baseline test score
    'mean2': 82,                   # Target improved score
    'sd1': 15,                     # Score standard deviation
    'icc': 0.15,                   # High clustering (classrooms)
    'analysis_model': 'mixedlm',   # Best for high ICC
    'use_satterthwaite': True,     # Essential for small clusters
    'lmm_reml': True,             # Unbiased variance estimation
    'alpha': 0.05
}

result = simulate_cluster_continuous(params_high_icc)
```

### Low ICC Scenario (Individual-Level Intervention)
```python
# Low ICC - boundary condition possible
params_low_icc = {
    'n_clusters': 15,
    'cluster_size': 50,
    'mean1': 50,
    'mean2': 53,
    'sd1': 10,
    'icc': 0.005,                  # Very low clustering
    'analysis_model': 'mixedlm',   # Will auto-fallback if needed
    'alpha': 0.05
}

result = simulate_cluster_continuous(params_low_icc)

# DesignPower automatically detects ICC ≈ 0 and uses appropriate method
if 'boundary_condition' in result.get('warnings', []):
    print("Automatic fallback to individual-level analysis")
```

---

## Interpretation Guidelines

### Power Interpretation
```python
# Interpreting power results
if result['power'] >= 0.80:
    print("✓ Adequate power for detecting specified effect")
elif result['power'] >= 0.70:
    print("⚠ Marginal power - consider increasing sample size")
else:
    print("❌ Insufficient power - redesign study")
```

### Warning Messages
```python
# Common warnings and their meanings
warnings = result.get('warnings', [])

for warning in warnings:
    if 'clusters' in warning and '40' in warning:
        print("General caution: Consider small-sample methods (already applied)")
    elif 'clusters' in warning and '30' in warning:
        print("Moderate concern: Results reliable with current methods")
    elif 'clusters' in warning and '20' in warning:
        print("High concern: Interpret results cautiously")
    elif 'boundary' in warning.lower():
        print("ICC near zero: Individual-level analysis more appropriate")
```

### Method Validation
```python
# Check convergence and reliability
if 'converged' in result and result['converged']:
    print("✓ Analysis converged successfully")
else:
    print("⚠ Convergence issues - automatic fallback used")

if 'fallback_method' in result:
    print(f"Note: Used fallback method: {result['fallback_method']}")
```

## Summary Best Practices

### 1. Choose Primary Method Based on Scenario
**Frequentist (Standard scenarios):**
- **5-8 clusters/arm**: `analysis_model="ttest"`
- **9-15 clusters/arm**: `analysis_model="gee"` + `use_bias_correction=True`
- **16+ clusters/arm**: `analysis_model="mixedlm"` + `use_satterthwaite=True`

**Exact Methods (Distribution-free scenarios):**
- **5-10 clusters/arm**: `analysis_model="permutation"`
- **Non-normal outcomes**: `analysis_model="permutation"`
- **Conservative inference**: `analysis_model="permutation"`

**Bayesian (Complex/challenging scenarios):**
- **5-8 clusters/arm**: `analysis_model="bayes"` + `bayes_backend="stan"`
- **9-15 clusters/arm**: `analysis_model="bayes"` + `bayes_backend="stan"`  
- **16+ clusters/arm**: `analysis_model="bayes"` + `bayes_backend="pymc"`

### 2. Use Built-in Safeguards
- DesignPower automatically handles convergence issues
- Trust the validation warnings and automatic corrections
- Boundary conditions are detected and handled appropriately

### 3. Consider Multiple Approaches
- Run sensitivity analyses with different methods
- Compare results for robustness
- Use Bayesian methods for complex scenarios

### 4. Document Your Approach
- Justify method selection based on cluster size
- Acknowledge limitations in analysis plan
- Report any fallback methods used

---

*These examples demonstrate DesignPower's advanced cluster RCT capabilities and provide practical templates for real-world analysis scenarios.*