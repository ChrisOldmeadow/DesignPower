# Cluster RCT Method Selection Guide

*A practical guide for choosing optimal statistical analysis methods in DesignPower*

## Quick Reference Decision Tree

```
Number of clusters per arm?
├─ 5-8 clusters → Use cluster-level analysis (t-test) OR Bayesian (Stan)
├─ 9-15 clusters → Use bias-corrected GEE OR Bayesian for complex scenarios
├─ 16-30 clusters → Use mixed models with corrections OR Bayesian (PyMC)
└─ 30+ clusters → Use standard mixed models OR Bayesian (variational)

Convergence issues or complex scenarios?
├─ YES → Consider Bayesian methods first
│  ├─ Rare outcomes → bayes + stan
│  ├─ Informative priors → bayes + pymc
│  └─ Complex structures → bayes + stan
└─ NO → Use frequentist methods as primary choice
```

## Detailed Method Selection

### Based on Number of Clusters Per Arm

#### **Very Small Clusters (5-8 per arm)**
**🥇 Recommended: Cluster-level analysis**
```python
analysis_model="ttest"  # Gold standard for very small clusters
```

**Why this method?**
- **Theoretically sound**: No distributional assumptions about individual-level data
- **Robust**: Not affected by within-cluster correlation structure
- **Literature-backed**: Recommended by Murray (1998), Donner & Klar (2000)
- **Simple interpretation**: Clear degrees of freedom (n_clusters - 2)

**🎯 Bayesian alternative (Recommended for challenging scenarios):**
```python
analysis_model="bayes"
bayes_backend="stan"           # Most robust for very small clusters
bayes_prior_icc="normal"       # Weakly informative prior
```

**When to choose Bayesian over t-test:**
- Convergence issues with rare outcomes
- Informative priors from previous studies
- Complex hierarchical structures (e.g., nested clustering)
- Decision-theoretic framework needed

---

#### **Small Clusters (9-15 per arm)**
**🥇 Recommended: Bias-corrected GEE**
```python
analysis_model="gee"
use_bias_correction=True  # Essential for small clusters
```

**Why this method?**
- **Small-sample optimized**: Bias-corrected sandwich estimators (Li et al. 2018)
- **Robust correlation**: Doesn't require correct correlation structure
- **Efficient**: More powerful than cluster-level analysis
- **Validated**: Performs well with ≥9 clusters per arm

**🎯 Bayesian alternative (For complex scenarios):**
```python
analysis_model="bayes"
bayes_backend="stan"           # Handle rare outcomes and complex structures
bayes_prior_icc="normal"       # Weakly informative prior
```

**When to choose Bayesian over GEE:**
- Rare binary outcomes causing convergence issues
- Very unbalanced cluster sizes
- Need for uncertainty quantification
- Informative priors available

**Alternative frequentist option:**
```python
analysis_model="mixedlm"
use_satterthwaite=True     # Enhanced degrees of freedom
lmm_reml=True             # REML for unbiased variance estimates
```

---

#### **Medium Clusters (16-30 per arm)**
**🥇 Recommended: Mixed models with corrections**
```python
analysis_model="mixedlm"
use_satterthwaite=True     # Better degrees of freedom approximation
lmm_reml=True             # Recommended for variance estimation
```

**Why this method?**
- **Optimal efficiency**: Maximum likelihood framework
- **Flexible modeling**: Can incorporate cluster-level covariates
- **Small-sample aware**: Satterthwaite approximation for df
- **Standard approach**: Widely accepted in literature

**🎯 Bayesian alternative (For advanced modeling):**
```python
analysis_model="bayes"
bayes_backend="pymc"           # Modern interface with good diagnostics
bayes_prior_icc="normal"       # Weakly informative prior
```

**When to choose Bayesian:**
- Non-standard distributions (e.g., zero-inflated, heavy-tailed)
- Complex hierarchical structures with multiple levels
- Need for full uncertainty quantification
- Decision-theoretic applications

**Alternative frequentist option:**
```python
analysis_model="gee"       # If correlation structure uncertain
use_bias_correction=True   # Still beneficial for n < 40
```

---

#### **Large Clusters (30+ per arm)**
**🥇 Recommended: Standard mixed models**
```python
analysis_model="mixedlm"   # Default settings typically sufficient
lmm_reml=True             # REML for variance estimation
```

**Why this method?**
- **Asymptotic validity**: Large-sample theory applies
- **Maximum efficiency**: Full likelihood-based inference
- **Standard practice**: Universally accepted approach
- **Rich diagnostics**: Comprehensive model checking available

**🎯 Bayesian alternative (For decision theory):**
```python
analysis_model="bayes"
bayes_backend="variational"    # Fast approximation for large studies
bayes_backend="pymc"          # Full MCMC if speed not critical
bayes_prior_icc="normal"      # Weakly informative prior
```

**When to choose Bayesian:**
- Decision-theoretic framework required
- Need for probability statements about parameters
- Complex outcome distributions
- Sequential or adaptive designs

---

### Based on Study Characteristics

#### **Binary Outcomes**

| Scenario | Recommended Method | Parameters | Rationale |
|----------|-------------------|------------|-----------|
| **Small clusters + rare outcome** | `analysis_model="ttest"` | Default | Avoid convergence issues |
| **Small clusters + common outcome** | `analysis_model="gee"` | `use_bias_correction=True` | Bias-corrected sandwich SE |
| **Large clusters** | `analysis_model="mixedlm"` | `use_satterthwaite=True` | Standard GLMM approach |
| **Very unbalanced clusters** | `analysis_model="bayes"` | `bayes_backend="stan"` | Handles complex structures |

#### **Continuous Outcomes**

| Scenario | Recommended Method | Parameters | Rationale |
|----------|-------------------|------------|-----------|
| **Normal distribution** | `analysis_model="mixedlm"` | `lmm_reml=True` | Standard approach |
| **Non-normal distribution** | `analysis_model="gee"` | `use_bias_correction=True` | Robust to distribution |
| **Missing data** | `analysis_model="mixedlm"` | Default | MAR assumption |
| **Complex correlation** | `analysis_model="bayes"` | `bayes_backend="stan"` | Flexible modeling |

#### **Special Scenarios**

**High ICC (>0.10)**
```python
analysis_model="mixedlm"    # Better handling of large cluster effects
use_satterthwaite=True      # Improved degrees of freedom
lmm_reml=True              # Unbiased variance estimation
```

**Very Low ICC (<0.01)**
- DesignPower automatically detects boundary conditions
- Falls back to cluster-robust OLS when ICC ≈ 0
- No user action required

**Convergence Issues**
- DesignPower tries multiple optimizers automatically
- Falls back to simpler methods if needed
- Consider Bayesian approach for difficult cases

## Advanced Features Guide

### **When to Use Bayesian Methods**

**Scenarios favoring Bayesian analysis:**
```python
analysis_model="bayes"
bayes_backend="stan"        # Full MCMC
# OR
bayes_backend="pymc"        # Modern probabilistic programming
```

1. **Very small clusters** (≤8 per arm) with complex structures
2. **Informative priors** available from previous studies
3. **Non-standard distributions** or complex hierarchical structures
4. **Convergence failures** with frequentist methods
5. **Decision-theoretic framework** required

**Bayesian backend selection:**
- **Stan**: Most robust, handles complex models, slower
- **PyMC**: Modern interface, good diagnostics, medium speed
- **Variational**: Fast approximation, less accurate
- **ABC**: When likelihood intractable, research contexts

### **Parameter Tuning Guidelines**

#### **Mixed Models (`analysis_model="mixedlm"`)**
```python
use_satterthwaite=True      # Always for n_clusters < 30
lmm_reml=True              # Default for variance estimation
lmm_method="lbfgs"         # Usually fastest optimizer
```

#### **GEE (`analysis_model="gee"`)**
```python
use_bias_correction=True    # Always for n_clusters < 40
# Correlation structure automatically set to exchangeable
```

#### **Bayesian (`analysis_model="bayes"`)**
```python
bayes_backend="stan"        # Most reliable
bayes_prior_icc="normal"    # Weakly informative prior
# Chains and iterations automatically optimized
```

## Troubleshooting Common Issues

### **Convergence Problems**

**Symptoms:**
- Warning messages about convergence failure
- Unrealistic parameter estimates
- Very large standard errors

**Solutions:**
1. **Automatic**: DesignPower tries multiple optimizers
2. **Manual**: Try Bayesian approach
```python
analysis_model="bayes"
bayes_backend="stan"
```
3. **Fallback**: Use cluster-level analysis
```python
analysis_model="ttest"
```

### **Boundary Conditions (ICC ≈ 0)**

**Symptoms:**
- ICC estimates near zero
- Warnings about boundary conditions

**Solutions:**
- **Automatic**: DesignPower detects and uses appropriate method
- **Consider**: Whether clustering is meaningful for this outcome
- **Alternative**: Pool data and use individual-level analysis

### **Small Sample Warnings**

**What they mean:**
- `< 40 clusters`: General caution about large-sample theory
- `< 30 clusters`: Consider permutation tests or exact methods
- `< 20 clusters`: High risk of Type I error inflation

**Actions:**
- **Use recommended corrections**: Already built into DesignPower
- **Consider**: Increasing sample size if possible
- **Document**: Acknowledge limitations in analysis plan

## Validation and Quality Checks

### **Built-in Validation**
DesignPower automatically:
- ✅ Checks for minimum cluster requirements
- ✅ Warns about small sample issues
- ✅ Detects boundary conditions
- ✅ Validates convergence
- ✅ Provides appropriate fallbacks

### **Manual Validation Steps**
1. **Check ICC estimate**: Should be plausible for your context
2. **Review warnings**: Address any methodological concerns
3. **Compare methods**: Try multiple approaches for robustness
4. **Check residuals**: Examine model assumptions (when available)

## Method Comparison Examples

### **Example 1: Small Binary Cluster Trial**
**Study**: 12 clusters per arm, binary outcome, ICC ≈ 0.03

**Approach 1: Recommended**
```python
analysis_model="gee"
use_bias_correction=True
```
**Pros**: Bias-corrected, robust, appropriate power
**Cons**: Requires ≥9 clusters per arm

**Approach 2: Conservative**
```python
analysis_model="ttest"
```
**Pros**: Gold standard, simple, robust
**Cons**: Less efficient than GEE

**Approach 3: Advanced**
```python
analysis_model="bayes"
bayes_backend="stan"
```
**Pros**: Full uncertainty quantification
**Cons**: More complex interpretation

### **Example 2: Medium Continuous Cluster Trial**
**Study**: 25 clusters per arm, continuous outcome, normal distribution

**Recommended approach:**
```python
analysis_model="mixedlm"
use_satterthwaite=True
lmm_reml=True
```
**Why**: Standard, efficient, well-validated approach

## Summary Recommendations

### **Quick Decision Rules**

1. **When in doubt**: Use cluster-level analysis (`analysis_model="ttest"`)
2. **For efficiency**: Use GEE with bias correction for 9-30 clusters
3. **For standard practice**: Use mixed models with Satterthwaite for 15+ clusters
4. **For complex scenarios**: Use Bayesian methods with Stan

### **Key Principles**

1. **Conservative is better**: Especially with small clusters
2. **Use built-in corrections**: DesignPower handles small-sample issues
3. **Trust the validation**: Built-in warnings guide appropriate use
4. **Document approach**: Justify method selection in analysis plan

### **Common Mistakes to Avoid**

❌ **Don't ignore warnings** about small clusters  
❌ **Don't use large-sample methods** with <15 clusters per arm  
❌ **Don't assume convergence** without checking diagnostics  
❌ **Don't ignore ICC estimates** that seem implausible  

✅ **Do use recommended corrections** for small samples  
✅ **Do consider multiple approaches** for robustness  
✅ **Do document limitations** in small cluster studies  
✅ **Do validate results** against literature when possible  

---

## Additional Resources

### **Literature References**
- **Murray (1998)**: Foundational cluster analysis methods
- **Donner & Klar (2000)**: Comprehensive cluster RCT methodology
- **Hayes & Moulton (2017)**: Modern cluster trial design and analysis
- **Li et al. (2018)**: Bias-corrected sandwich estimators for small clusters
- **Leyrat et al. (2018)**: Review of small cluster methods

### **Software Comparisons**
- [Technical comparison](CLUSTER_RCT_SMALL_CLUSTERS_ANALYSIS.md) with R/SAS/Stata
- [Implementation details](CLUSTER_RCT_ACTION_PLAN.md) of advanced features
- [Validation roadmap](../VALIDATION_ROADMAP_TRACKING.md) for tested scenarios

### **Getting Help**
- Built-in validation messages provide specific guidance
- Check convergence diagnostics when available
- Consider consulting with biostatistician for complex designs
- Review literature for similar study designs and analysis approaches

---

*This guide is based on DesignPower's state-of-the-art cluster RCT implementation, which exceeds most available statistical software in feature completeness and robustness.*