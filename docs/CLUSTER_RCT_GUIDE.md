# Cluster RCT Analysis Guide

*Comprehensive guide for cluster randomized controlled trials in DesignPower*

## Quick Decision Tree

```
How many clusters per arm do you have?

┌─ 5-8 clusters
│  ├─ Primary: analysis_model="ttest"
│  │   ✓ Gold standard for very small clusters
│  │   ✓ Robust, simple, literature-backed
│  ├─ Exact: analysis_model="permutation"
│  │   ✓ Exact inference without distributional assumptions
│  │   ✓ Recommended when ≤10 clusters per arm
│  └─ Alternative: analysis_model="bayes", bayes_backend="stan"
│     ✓ Use for convergence issues or informative priors
│
├─ 9-15 clusters  
│  ├─ Primary: analysis_model="gee", use_bias_correction=True
│  │   ✓ Bias-corrected sandwich estimators
│  │   ✓ Efficient, handles correlation uncertainty
│  └─ Alternative: analysis_model="bayes", bayes_backend="stan"
│     ✓ Use for rare outcomes or complex structures
│
├─ 16-30 clusters
│  ├─ Primary: analysis_model="mixedlm", use_satterthwaite=True
│  │   ✓ Small-sample degree of freedom corrections
│  │   ✓ Standard approach with adjustments
│  └─ Alternative: analysis_model="bayes", bayes_backend="pymc"
│     ✓ Use for non-standard distributions
│
└─ 30+ clusters
   ├─ Primary: analysis_model="mixedlm", lmm_reml=True
   │   ✓ Standard mixed model analysis
   │   ✓ Large-sample theory applies
   └─ Alternative: analysis_model="bayes", bayes_backend="variational"
      ✓ Use for decision-theoretic frameworks
```

## Method Parameters

### Cluster-Level Analysis (5-8 clusters/arm)
```python
analysis_model="ttest"
# ✓ No additional parameters needed
# ✓ Automatic fallback for convergence issues
# ✓ Handles any ICC value robustly
```

### Permutation Tests (5-10 clusters/arm)
```python
analysis_model="permutation"
# ✓ Exact inference without distributional assumptions
# ✓ Automatic exact vs Monte Carlo selection
# ✓ Confidence intervals via permutation
```

### Bias-Corrected GEE (9-15 clusters/arm)
```python
analysis_model="gee"
use_bias_correction=True  # Essential for small clusters
# ✓ Exchangeable correlation automatically used
# ✓ Robust to correlation mis-specification
```

### Mixed Models (16+ clusters/arm)
```python
analysis_model="mixedlm"
use_satterthwaite=True    # Better degrees of freedom
lmm_reml=True            # Unbiased variance estimates
```

### Bayesian Analysis (Complex scenarios)
```python
analysis_model="bayes"
bayes_backend="stan"         # Most robust
bayes_backend="pymc"         # Modern interface  
bayes_backend="variational"  # Fast approximation
```

## Practical Examples

### Example 1: Very Small Cluster Trial
**Scenario**: 7 primary care clinics per arm, binary outcome

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

**Why this approach?**
- Cluster-level analysis is the gold standard for very small clusters
- Robust method not affected by distribution assumptions
- Simple interpretation with clear degrees of freedom (14 - 2 = 12)

### Example 2: Small Cluster Trial  
**Scenario**: 12 schools per arm, continuous outcome

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

**Why this approach?**
- GEE with bias correction optimal for 9-15 clusters
- Validated for small clusters by Li et al. (2018)
- More efficient than cluster-level analysis

### Example 3: Medium Cluster Trial
**Scenario**: 20 workplaces per arm, binary outcome

```python
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

**Why this approach?**
- Mixed models with corrections optimal for medium clusters
- Satterthwaite approximation provides better degrees of freedom
- Standard practice widely accepted in literature

## Warning Interpretation

| Warning Message | Meaning | Action |
|----------------|---------|--------|
| **"< 40 clusters total"** | General caution | Use recommended corrections (built-in) |
| **"< 30 clusters total"** | Consider exact methods | Methods automatically adjusted |
| **"< 20 clusters total"** | High Type I error risk | Consider increasing sample size |
| **"Boundary condition detected"** | ICC ≈ 0 | Automatic fallback to cluster-robust OLS |
| **"Convergence issues"** | Optimization failed | Automatic fallback hierarchy activated |

## Special Scenarios

### Binary Outcomes
```python
# Rare events (p < 0.1) with small clusters
analysis_model="ttest"           # Avoid convergence issues

# Common events with bias correction  
analysis_model="gee"
use_bias_correction=True
```

### High ICC (> 0.10)
```python
analysis_model="mixedlm"
use_satterthwaite=True
lmm_reml=True
# Better handling of large cluster effects
```

### Convergence Problems
```python
# Automatic fallback sequence:
# 1. Multiple optimizers tried
# 2. Cluster-robust OLS
# 3. Simple cluster t-test

# Manual override:
analysis_model="bayes"
bayes_backend="stan"
```

## Method Comparison Summary

| Method | Clusters/Arm | Pros | Cons | When to Use |
|--------|--------------|------|------|-------------|
| **ttest** | 5-8 | Gold standard, robust | Less efficient | Very small clusters |
| **permutation** | 5-10 | Exact inference, no assumptions | Computationally intensive | Very small clusters, non-normal data |
| **GEE + bias correction** | 9-15 | Efficient, robust correlation | Requires ≥9 clusters | Small clusters |
| **Mixed models + Satterthwaite** | 16-30 | Standard approach, flexible | Distributional assumptions | Medium clusters |
| **Standard mixed models** | 30+ | Maximum efficiency | Large-sample theory | Large clusters |
| **Bayesian (Stan)** | 5-8 | Handles convergence issues | Slower computation | Challenging small clusters |
| **Bayesian (PyMC)** | 9-30 | Full uncertainty quantification | Complex interpretation | Non-standard scenarios |

## Troubleshooting

| Problem | Quick Fix |
|---------|-----------|
| **"Model didn't converge"** | Trust automatic fallback OR use `analysis_model="bayes"` |
| **"ICC estimate is 0"** | Normal - DesignPower handles automatically |
| **"Very large standard errors"** | Check for data issues OR use `analysis_model="ttest"` |
| **"Unrealistic results"** | Try `analysis_model="bayes"` with `bayes_backend="stan"` |
| **"Rare outcome convergence"** | Use `analysis_model="bayes"` with `bayes_backend="stan"` |

## Quality Checklist

- [ ] **Check cluster count**: Use appropriate method for cluster size
- [ ] **Review warnings**: Address any methodological concerns  
- [ ] **Validate ICC**: Ensure estimate is plausible for context
- [ ] **Check convergence**: Verify successful model fitting
- [ ] **Consider alternatives**: Try multiple approaches for robustness

## Key Principles

1. **Conservative is better**: Especially with small clusters
2. **Use built-in corrections**: DesignPower handles small-sample issues
3. **Trust the validation**: Built-in warnings guide appropriate use
4. **Document approach**: Justify method selection in analysis plan

---

*DesignPower automatically handles most complexity - these guidelines help optimize your analysis choice*