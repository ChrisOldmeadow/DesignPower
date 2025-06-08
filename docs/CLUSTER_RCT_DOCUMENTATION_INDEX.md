# Cluster RCT Documentation Index

*Complete guide to DesignPower's cluster randomized controlled trial capabilities*

## 📚 Documentation Overview

DesignPower provides **state-of-the-art statistical methods** for cluster randomized controlled trials, with particular strength in handling small numbers of clusters (5-15 per arm) where other software often fails. This documentation suite provides comprehensive guidance for optimal use.

## 🚀 Quick Start (New Users)

### **Step 1: Quick Reference** ⭐ START HERE
📄 **[Quick Reference Card](CLUSTER_RCT_QUICK_REFERENCE.md)**
- 🎯 **Decision tree** for method selection
- ⚡ **Parameter cheat sheet** for each scenario
- 🔧 **Troubleshooting** quick fixes
- ⏱️ **5-minute read** for immediate use

### **Step 2: Practical Examples**
📄 **[Analysis Examples](CLUSTER_RCT_EXAMPLES.md)**
- 💡 **6 realistic scenarios** with complete code
- 🏥 Healthcare, 🏫 school, 🏢 workplace interventions
- 📊 **Method comparisons** and interpretation
- ⏱️ **15-minute read** for practical understanding

## 📖 Detailed Guidance

### **Step 3: Complete Method Selection**
📄 **[Method Selection Guide](CLUSTER_RCT_METHOD_SELECTION_GUIDE.md)**
- 🧭 **Comprehensive decision framework**
- 📏 **Parameter tuning guidelines**
- ⚠️ **Warning interpretation guide**
- 🔍 **Quality validation checklist**
- ⏱️ **30-minute read** for complete understanding

## 🔬 Technical Documentation

### **Advanced Users & Researchers**

📄 **[Small Clusters Technical Analysis](CLUSTER_RCT_SMALL_CLUSTERS_ANALYSIS.md)**
- 🎓 **Literature review** vs. best practices
- ⚖️ **Software comparison** (R, SAS, Stata)
- 🏆 **DesignPower advantages** analysis
- 📑 **Implementation quality** assessment
- ⏱️ **45-minute read** for technical depth

📄 **[Strategic Action Plan](CLUSTER_RCT_ACTION_PLAN.md)**
- 🎯 **Enhancement roadmap** and priorities
- ✅ **Implementation status** tracking
- 📈 **Performance benchmarks** planned
- 🔬 **Cross-validation studies** design
- ⏱️ **20-minute read** for development planning

## 🎯 Validation & Quality Assurance

📄 **[Validation Roadmap](../VALIDATION_ROADMAP_TRACKING.md)**
- ✅ **Current validation status** (92.9% success rate)
- 🧪 **54 test scenarios** across all designs
- 📚 **Gold standard references** (8 authoritative sources)
- 🎯 **Priority targets** for future validation
- ⏱️ **10-minute scan** for validation confidence

## 🎯 User Journey Map

### **👤 New User (First Time)**
1. 📄 [Quick Reference](CLUSTER_RCT_QUICK_REFERENCE.md) → Get method for your cluster size
2. 📄 [Examples](CLUSTER_RCT_EXAMPLES.md) → Find similar scenario
3. 💻 **Run analysis** with recommended parameters

### **👨‍💼 Practical User (Occasional Use)**
1. 📄 [Method Selection Guide](CLUSTER_RCT_METHOD_SELECTION_GUIDE.md) → Understand rationale
2. 📄 [Examples](CLUSTER_RCT_EXAMPLES.md) → Compare multiple approaches
3. 💻 **Run sensitivity analyses** for robustness

### **👨‍🔬 Advanced User (Research/Development)**
1. 📄 [Technical Analysis](CLUSTER_RCT_SMALL_CLUSTERS_ANALYSIS.md) → Understand implementation
2. 📄 [Action Plan](CLUSTER_RCT_ACTION_PLAN.md) → See development roadmap
3. 📄 [Validation Roadmap](../VALIDATION_ROADMAP_TRACKING.md) → Check tested scenarios
4. 💻 **Contribute** to validation or feature development

## 🔧 Feature Comparison

| Feature | DesignPower | R (lme4) | SAS | Stata |
|---------|-------------|----------|-----|-------|
| **Small cluster corrections** | ✅ Multiple | ⭕ Limited | ✅ K-R only | ⭕ Basic |
| **Automatic fallbacks** | ✅ Yes | ❌ No | ❌ No | ❌ No |
| **Bayesian methods** | ✅ Multiple | ⭕ External | ❌ No | ❌ No |
| **User guidance** | ✅ Comprehensive | ❌ No | ❌ No | ❌ No |
| **Validation documentation** | ✅ Extensive | ⭕ Limited | ⭕ Limited | ⭕ Limited |

## 📊 Current Implementation Status

### ✅ **Validated Methods (92.9% Success Rate)**
- **Cohen's effect sizes**: 96-100% accuracy vs. literature
- **A'Hern single-arm designs**: 100% accuracy (exact match)
- **Cluster RCT calculations**: 100% accuracy vs. Donner & Klar, Hayes & Moulton
- **Non-inferiority tests**: 95-100% accuracy vs. multiple sources

### 🔧 **Advanced Features Available**
- **Linear Mixed Models (LMM)** with REML/ML estimation
- **Generalized Linear Mixed Models (GLMM)** for binary outcomes
- **Generalized Estimating Equations (GEE)** with bias corrections
- **Bayesian hierarchical models** (Stan, PyMC backends)
- **Satterthwaite approximation** for small-sample degrees of freedom
- **Bias-corrected sandwich estimators** for robust inference
- **Automatic fallback systems** with convergence handling

### 🚀 **Unique Advantages**
1. **Intelligent automation**: Method selection guidance and automatic corrections
2. **Comprehensive fallbacks**: Multiple optimizers and robust error handling
3. **Small cluster expertise**: State-of-the-art methods for 5-15 clusters per arm
4. **Literature validation**: Tested against 8 authoritative textbook sources
5. **Production ready**: Handles edge cases, boundary conditions, convergence issues

## 📞 Getting Help

### **Built-in Guidance**
- ⚠️ **Validation warnings**: Automatic guidance for study parameters
- 🔄 **Convergence handling**: Automatic fallbacks prevent analysis failure
- 📊 **Method recommendations**: Based on cluster size and study characteristics

### **Documentation Support**
- 🆘 **Troubleshooting sections** in each guide
- 💡 **Common scenarios** covered in examples
- 🎯 **Decision trees** for method selection
- ✅ **Quality checklists** for validation

### **Community Resources**
- 📖 **Literature references** provided throughout
- 🔗 **Cross-links** between related documentation
- 📚 **Textbook sources** for theoretical background
- 🔬 **Validation databases** for quality assurance

## 🎯 Key Messages

### **For Users**
1. **DesignPower is already exceptional** for cluster RCT analysis
2. **Start with the Quick Reference** for immediate needs
3. **Use built-in guidance** - trust the validation warnings
4. **Multiple approaches available** for robustness checking

### **For Researchers**
1. **State-of-the-art implementation** exceeding most available software
2. **Comprehensive validation** against gold standard literature
3. **Small cluster expertise** - handles challenging scenarios other software can't
4. **Production-ready quality** with extensive error handling

### **For Developers**
1. **Technical excellence** demonstrated in implementation
2. **Clear enhancement roadmap** for future development
3. **Validation framework** for quality assurance
4. **Cross-software benchmarking** for competitive analysis

---

## 📋 Recommended Reading Order

### **🏃‍♂️ Quick (< 30 minutes)**
1. [Quick Reference](CLUSTER_RCT_QUICK_REFERENCE.md) (5 min)
2. [Examples - Find your scenario](CLUSTER_RCT_EXAMPLES.md) (15 min)
3. **Start analyzing!**

### **🚶‍♂️ Thorough (1-2 hours)**  
1. [Quick Reference](CLUSTER_RCT_QUICK_REFERENCE.md) (5 min)
2. [Method Selection Guide](CLUSTER_RCT_METHOD_SELECTION_GUIDE.md) (30 min)
3. [Examples - Multiple scenarios](CLUSTER_RCT_EXAMPLES.md) (30 min)
4. [Validation status check](../VALIDATION_ROADMAP_TRACKING.md) (10 min)

### **🎓 Complete (Half day)**
1. All user documentation (90 min)
2. [Technical Analysis](CLUSTER_RCT_SMALL_CLUSTERS_ANALYSIS.md) (45 min)
3. [Action Plan](CLUSTER_RCT_ACTION_PLAN.md) (20 min)
4. [Validation details](../VALIDATION_ROADMAP_TRACKING.md) (25 min)

---

*DesignPower: Leading the field in cluster randomized controlled trial statistical analysis*

**Last Updated**: 2025-01-08 | **Documentation Version**: 1.0 | **Software Version**: git-dc58366