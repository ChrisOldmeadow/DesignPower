# Cluster RCT Quick Reference Card

*Fast method selection for cluster randomized controlled trials*

## 🚀 Quick Decision Tree

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

## 📋 Method Parameters Cheat Sheet

### Cluster-Level Analysis (Very Small: 5-8 clusters/arm)
```python
analysis_model="ttest"
# ✓ No additional parameters needed
# ✓ Automatic fallback for convergence issues
# ✓ Handles any ICC value robustly
```

### Exact Permutation Tests (Very Small: 5-10 clusters/arm)
```python
analysis_model="permutation"
# ✓ Exact inference without distributional assumptions
# ✓ Automatic exact vs Monte Carlo selection
# ✓ Confidence intervals via permutation
# ✓ Handles any outcome distribution
```

### Bias-Corrected GEE (Small: 9-15 clusters/arm)
```python
analysis_model="gee"
use_bias_correction=True  # Essential for small clusters
# ✓ Exchangeable correlation automatically used
# ✓ Robust to correlation mis-specification
```

### Mixed Models with Corrections (Medium: 16-30 clusters/arm)
```python
analysis_model="mixedlm"
use_satterthwaite=True    # Better degrees of freedom
lmm_reml=True            # Unbiased variance estimates
# ✓ Optimal efficiency with small-sample awareness
```

### Standard Mixed Models (Large: 30+ clusters/arm)
```python
analysis_model="mixedlm"
lmm_reml=True            # REML for variance estimation
# ✓ Standard approach, all corrections optional
```

### Bayesian Analysis (Complex scenarios)
```python
analysis_model="bayes"
bayes_backend="stan"         # Most robust - use for challenging scenarios
bayes_backend="pymc"         # Modern interface - good diagnostics  
bayes_backend="variational"  # Fast approximation - large studies
bayes_prior_icc="normal"     # Weakly informative (default)
# ✓ Use when: convergence issues, informative priors, complex structures
```

## ⚠️ Warning Interpretation Guide

| Warning Message | Meaning | Action |
|----------------|---------|--------|
| **"< 40 clusters total"** | General caution | Use recommended corrections (already built-in) |
| **"< 30 clusters total"** | Consider exact methods | Methods automatically adjusted |
| **"< 20 clusters total"** | High Type I error risk | Consider increasing sample size |
| **"Boundary condition detected"** | ICC ≈ 0 | Automatic fallback to cluster-robust OLS |
| **"Convergence issues"** | Optimization failed | Automatic fallback hierarchy activated |

## 🎯 Special Scenarios

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

## 🔧 Troubleshooting Quick Fixes

| Problem | Quick Fix |
|---------|-----------|
| **"Model didn't converge"** | Trust automatic fallback OR use `analysis_model="bayes"` |
| **"ICC estimate is 0"** | Normal - DesignPower handles automatically |
| **"Very large standard errors"** | Check for data issues OR use `analysis_model="ttest"` |
| **"Unrealistic results"** | Try `analysis_model="bayes"` with `bayes_backend="stan"` |
| **"Rare outcome convergence"** | Use `analysis_model="bayes"` with `bayes_backend="stan"` |
| **"Complex correlation structure"** | Use `analysis_model="bayes"` with hierarchical modeling |

## 📊 Method Comparison Summary

| Method | Clusters/Arm | Pros | Cons | When to Use |
|--------|--------------|------|------|-------------|
| **ttest** | 5-8 | Gold standard, robust | Less efficient | Very small clusters |
| **permutation** | 5-10 | Exact inference, no assumptions | Computationally intensive | Very small clusters, non-normal data |
| **GEE + bias correction** | 9-15 | Efficient, robust correlation | Requires ≥9 clusters | Small clusters |
| **Mixed models + Satterthwaite** | 16-30 | Standard approach, flexible | Distributional assumptions | Medium clusters |
| **Standard mixed models** | 30+ | Maximum efficiency | Large-sample theory | Large clusters |
| **Bayesian (Stan)** | 5-8 | Handles convergence issues | Slower computation | Challenging small clusters |
| **Bayesian (PyMC)** | 9-30 | Full uncertainty quantification | Complex interpretation | Non-standard scenarios |
| **Bayesian (Variational)** | 30+ | Fast approximation | Less accurate | Decision theory contexts |

## ✅ Quality Checklist

- [ ] **Check cluster count**: Use appropriate method for cluster size
- [ ] **Review warnings**: Address any methodological concerns  
- [ ] **Validate ICC**: Ensure estimate is plausible for context
- [ ] **Check convergence**: Verify successful model fitting
- [ ] **Consider alternatives**: Try multiple approaches for robustness

## 🔗 Full Documentation

- **Complete guide**: [Method Selection Guide](CLUSTER_RCT_METHOD_SELECTION_GUIDE.md)
- **Technical details**: [Small Clusters Analysis](CLUSTER_RCT_SMALL_CLUSTERS_ANALYSIS.md)
- **Implementation plan**: [Action Plan](CLUSTER_RCT_ACTION_PLAN.md)
- **Validation status**: [Roadmap Tracking](../VALIDATION_ROADMAP_TRACKING.md)

---

*DesignPower automatically handles most complexity - these guidelines help optimize your analysis choice*