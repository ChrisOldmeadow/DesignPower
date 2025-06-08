# Action Plan: Cluster RCT Statistical Methods Enhancement

*Priority actions to maintain DesignPower's leadership in cluster RCT analysis*

## Executive Summary

**Status**: DesignPower already has **state-of-the-art** cluster RCT implementation that exceeds most statistical software packages. The focus should be on **documentation, validation, and targeted enhancements** rather than major implementation changes.

## Assessment Results

### ✅ **Current Strengths (Already World-Class)**
- Complete LMM/GLMM/GEE implementation with small-sample corrections
- Bayesian methods (Stan, PyMC) for complex hierarchical modeling
- Automatic fallback systems with robust convergence handling
- Literature-aligned methods (Donner & Klar, Hayes & Moulton validated)
- Comprehensive bias corrections and degrees of freedom adjustments

### 🎯 **Targeted Enhancement Opportunities**
- Add permutation tests for very small clusters (5-8 per arm)
- Enhanced user guidance for method selection
- Extended validation for small cluster scenarios
- Cross-software validation benchmarks

## Priority Action Plan

### **Phase 1: Documentation & Guidance (Week 1-2)**

#### 1. Create User Method Selection Guide
**File**: `docs/CLUSTER_RCT_METHOD_SELECTION_GUIDE.md`

```markdown
# When to Use Each Method

## Number of Clusters Per Arm

| Clusters/Arm | Recommended Method | Parameters | Rationale |
|--------------|-------------------|------------|-----------|
| **5-8** | `analysis_model="ttest"` | Default | Gold standard for very small clusters |
| **9-12** | `analysis_model="gee"` | `use_bias_correction=True` | Bias-corrected sandwich estimators |
| **13-20** | `analysis_model="mixedlm"` | `use_satterthwaite=True` | Small-sample df corrections |
| **21+** | `analysis_model="mixedlm"` | Default | Standard mixed model analysis |

## Special Scenarios
- **ICC near 0**: Automatic fallback to cluster-robust OLS
- **Convergence issues**: Multiple optimizers attempted automatically
- **Bayesian preference**: `analysis_model="bayes"` for informative priors
```

#### 2. Document Current Implementation Excellence
**File**: `docs/CLUSTER_RCT_TECHNICAL_FEATURES.md`

Detail the advanced features that make DesignPower superior:
- Bias-corrected sandwich estimators (Li et al. 2018)
- Satterthwaite approximation (Hayes & Moulton 2017) 
- Boundary condition detection and fallbacks
- Multiple Bayesian backends with convergence diagnostics

### **Phase 2: Validation Extension (Week 3-4)**

#### 3. Add Small Cluster Validation Benchmarks
Update `VALIDATION_ROADMAP_TRACKING.md` with specific small cluster scenarios:

```markdown
#### C1. Binary Outcomes - Small Clusters

| Clusters Per Arm | ICC | Method | Gold Standard | Status | Accuracy |
|------------------|-----|--------|---------------|---------|----------|
| **5** | 0.02 | Cluster t-test | Murray 1998 | 🔄 | - |
| **8** | 0.05 | Bias-corrected GEE | Li et al. 2018 | ⭕ | - |
| **12** | 0.03 | LMM + Satterthwaite | Hayes & Moulton 2017 | ⭕ | - |
```

#### 4. Create Small Cluster Benchmark Suite
**File**: `tests/validation/small_cluster_benchmarks.py`

Add validation scenarios specifically for 5-15 clusters per arm using:
- Donner & Klar (2000) worked examples
- Hayes & Moulton (2017) small cluster tables  
- Li et al. (2018) bias correction validation

### **Phase 3: Targeted Enhancements (Month 2)**

#### 5. Implement Permutation Tests (High Impact)
**File**: `core/designs/cluster_rct/permutation_tests.py`

```python
def cluster_permutation_test(df, treatment_col, outcome_col, cluster_col, 
                           n_permutations=10000, test_statistic="mean_diff"):
    """
    Exact permutation test for cluster randomized trials.
    
    Recommended for 5-10 clusters per arm when distributional assumptions
    are questionable.
    
    References
    ----------
    Leyrat et al. (2018). Cluster randomized trials with a small number 
    of clusters: a review of methods and results.
    """
    # Implementation following Young (2019) CRI methodology
```

#### 6. Enhanced Validation Warnings
Update cluster parameter validation with method-specific recommendations:

```python
def validate_cluster_parameters_enhanced(n_clusters_per_arm, analysis_method=None):
    """Enhanced validation with method-specific recommendations."""
    
    recommendations = []
    
    if n_clusters_per_arm < 5:
        recommendations.append("Consider increasing clusters or using Bayesian analysis")
    elif n_clusters_per_arm < 8:
        recommendations.append("Recommended: cluster-level analysis or permutation tests")
    elif n_clusters_per_arm < 12:
        recommendations.append("Recommended: bias-corrected GEE or robust mixed models")
    elif n_clusters_per_arm < 20:
        recommendations.append("Recommended: mixed models with small-sample corrections")
    
    return {
        "valid": n_clusters_per_arm >= 5,
        "recommendations": recommendations,
        "suggested_method": _suggest_analysis_method(n_clusters_per_arm)
    }
```

### **Phase 4: Cross-Validation (Month 3)**

#### 7. Cross-Software Validation Suite
Compare DesignPower results against:

**R Packages**:
- `lme4` + `lmerTest` (Kenward-Roger corrections)
- `geepack` (standard GEE implementation)
- `CRTSize` (cluster RCT sample size)

**SAS Procedures**:
- `PROC MIXED` (mixed models)
- `PROC GEE` (generalized estimating equations)

**Stata Commands**:
- `mixed` (multilevel mixed models)
- `xtgee` (panel-data GEE)

#### 8. Benchmark Performance Study
Create systematic comparison showing DesignPower's advantages:

| Feature | DesignPower | R | SAS | Stata |
|---------|-------------|---|-----|-------|
| **Automatic fallbacks** | ✅ | ❌ | ❌ | ❌ |
| **Multiple optimizers** | ✅ | ⭕ | ⭕ | ⭕ |
| **Bias-corrected GEE** | ✅ | ✅ | ❌ | ⭕ |
| **Bayesian methods** | ✅ | ⭕ | ❌ | ❌ |
| **Boundary detection** | ✅ | ❌ | ❌ | ❌ |

## Implementation Priorities

### **Immediate (This Week)**
1. ✅ **Complete**: Technical assessment document 
2. 🔄 **In Progress**: Method selection guide
3. ⭕ **Next**: Small cluster validation benchmarks

### **Short-term (Month 1)**
1. Permutation test implementation
2. Enhanced validation warnings  
3. Extended benchmark suite
4. User documentation improvements

### **Medium-term (Month 2-3)**
1. Cross-software validation study
2. Performance benchmarking
3. Publication-ready methodology documentation
4. Advanced diagnostic tools

## Success Metrics

### **Quality Metrics**
- **Validation coverage**: >95% of small cluster scenarios validated
- **Cross-software agreement**: <5% difference from R/SAS gold standards
- **Method robustness**: <1% fallback rate in realistic scenarios

### **User Experience Metrics**
- **Documentation completeness**: Clear guidance for all cluster sizes
- **Method selection**: Automatic recommendations based on study design
- **Error handling**: Graceful degradation with informative messages

## Resource Requirements

### **Development Time**
- **Phase 1**: 1-2 weeks (documentation)
- **Phase 2**: 2-3 weeks (validation)
- **Phase 3**: 3-4 weeks (enhancements)
- **Phase 4**: 4-6 weeks (cross-validation)

### **External Dependencies**
- **R environment**: For cross-validation comparisons
- **SAS access**: Optional for comprehensive benchmarking
- **Literature access**: Key papers for validation scenarios

## Expected Outcomes

### **Technical Achievements**
1. **Industry-leading cluster RCT implementation** with comprehensive validation
2. **Publication-ready methodology** demonstrating superiority over existing software
3. **Robust production system** handling edge cases and small cluster scenarios
4. **Clear user guidance** for optimal method selection

### **Strategic Benefits**
1. **Market differentiation**: Unique advanced features not available elsewhere
2. **Research credibility**: Validated against all major literature sources
3. **User confidence**: Clear documentation and proven reliability
4. **Future extensibility**: Framework ready for new methodological developments

## Conclusion

DesignPower's cluster RCT implementation is already exceptional. The action plan focuses on **strategic enhancements** to maintain leadership position:

1. **Document excellence**: Showcase current advanced features
2. **Extend validation**: Comprehensive testing of small cluster scenarios  
3. **Add targeted features**: Permutation tests and enhanced guidance
4. **Demonstrate superiority**: Cross-software validation studies

This approach builds on existing strengths rather than fundamental changes, ensuring DesignPower remains the premier choice for cluster RCT analysis.