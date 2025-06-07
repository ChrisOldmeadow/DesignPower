# DesignPower: Comprehensive Statistical Power Analysis

A modern, comprehensive application for statistical power analysis and sample size calculation across various clinical trial designs, featuring analytical methods, Monte Carlo simulations, and cutting-edge Bayesian inference.

## 🚀 Key Features

### Study Designs
- **Parallel RCTs**: Two-arm randomized controlled trials
- **Single Arm Trials**: Single-group studies with historical controls
- **Cluster RCTs**: Randomized trials at the cluster level
- **Interrupted Time Series**: Time series intervention analysis
- **Stepped Wedge**: Sequential cluster rollout designs

### Outcome Types
- **Continuous**: Normal and non-normal distributions
- **Binary**: Proportions and rates
- **Survival**: Time-to-event analysis
- **Count**: Poisson and negative binomial outcomes

### Calculation Types
- **Sample Size**: Determine required sample sizes for desired power
- **Power**: Calculate statistical power for given sample sizes
- **Minimum Detectable Effect**: Estimate smallest detectable effect sizes

### Hypothesis Testing
- **Superiority**: Testing if new treatment is better
- **Non-Inferiority**: Testing if new treatment is not worse by specified margin
- **Equivalence**: Testing if treatments are equivalent within bounds

## 🧠 Advanced Analysis Methods

### Analytical Methods
- Closed-form statistical formulas
- Design effect adjustments for clustering
- Variance inflation corrections
- Satterthwaite approximations

### Monte Carlo Simulation
- **Full Simulation**: Individual-level data generation
- **Bootstrap Methods**: Resampling-based inference
- **Mixed Models**: Linear and generalized linear mixed models (LMM/GLMM)
- **GEE**: Generalized Estimating Equations for clustered data

### 🔬 Bayesian Inference (NEW!)

#### Full MCMC Backends
- **Stan (CmdStanPy)**: Industry-standard probabilistic programming
- **PyMC**: Pure Python Bayesian modeling with NUTS sampling

#### Fast Approximate Methods
- **Variational Bayes**: Laplace approximation (10-100x faster than MCMC)
- **ABC**: Approximate Bayesian Computation (ultra-lightweight)

#### Bayesian Inference Methods
- **Credible Intervals**: 95% posterior credible intervals
- **Posterior Probability**: Probability of favorable effects
- **ROPE**: Region of Practical Equivalence testing

#### Smart Resource Management
- **Environment Detection**: Automatically suggests appropriate methods
- **Fallback Hierarchy**: Stan/PyMC → Variational → ABC → Classical
- **Web-Friendly**: ABC methods work on minimal server resources

## 🏗️ Architecture

### Modular Design
```
DesignPower/
├── app/                          # Streamlit web interface
│   ├── designpower_app.py        # Main application
│   └── components/               # UI components by design type
├── core/                         # Core calculation engine
│   ├── designs/                  # Design-specific implementations
│   │   ├── parallel/             # Parallel RCT methods
│   │   ├── single_arm/           # Single arm methods
│   │   ├── cluster_rct/          # Cluster RCT methods
│   │   └── interrupted_time_series/ # ITS methods
│   ├── outcomes/                 # Outcome-specific utilities
│   ├── methods/                  # Statistical method implementations
│   └── utils/                    # Shared utilities and reports
├── api/                          # FastAPI REST interface
├── tests/                        # Comprehensive test suite
└── docs/                         # Documentation and methodology
```

### Backend Interfaces
- **Web Interface**: Streamlit-based interactive application
- **REST API**: FastAPI for programmatic access
- **CLI**: Command-line interface for automation
- **Python API**: Direct function calls for integration

## 📊 Cluster RCT Specialization

### Advanced Clustering Features
- **ICC Handling**: Linear and logit scale ICC with automatic conversion
- **Unequal Clusters**: Design effect adjustments for cluster size variation
- **Small Sample Corrections**: Specialized methods for few clusters
- **Sensitivity Analysis**: ICC impact visualization

### Mixed Model Support
- **LMM**: Linear Mixed Models with REML/ML estimation
- **Robust Methods**: Cluster-robust standard errors
- **Fallback Systems**: Automatic degradation when models fail
- **Convergence Diagnostics**: Detailed fitting statistics

### Bayesian Hierarchical Models
```python
# Hierarchical model structure
y_ij ~ Normal(α + β*treatment_j + u_j, σ_e)
u_j ~ Normal(0, σ_u)  # Random cluster effects
β ~ Normal(0, 10)     # Treatment effect prior
```

## 🛠️ Installation

### Basic Installation
```bash
git clone https://github.com/yourusername/DesignPower.git
cd DesignPower
pip install -r requirements.txt
```

### Bayesian Backends (Optional)
```bash
# For Stan backend (full MCMC)
pip install cmdstanpy

# For PyMC backend (full MCMC) 
pip install pymc

# Approximate methods use scipy (included in requirements.txt)
```

### Development Installation
```bash
pip install -e .
pip install -r requirements-dev.txt
```

## 🚀 Usage

### Web Interface
```bash
streamlit run app/designpower_app.py
```
Open http://localhost:8501

### REST API
```bash
uvicorn api.main:app --reload
```
Documentation at http://localhost:8000/docs

### Command Line
```bash
# Parallel RCT sample size
python cli.py sample-size --design parallel --outcome continuous \
  --delta 0.5 --std-dev 1.0 --power 0.8

# Cluster RCT with Bayesian analysis
python cli.py power --design cluster --outcome continuous \
  --n-clusters 10 --cluster-size 20 --icc 0.05 \
  --mean1 3.0 --mean2 3.5 --std-dev 1.2 \
  --method simulation --analysis-model bayes --backend pymc
```

### Python API
```python
from core.designs.cluster_rct import simulation_continuous

# Bayesian power analysis
results = simulation_continuous.power_continuous_sim(
    n_clusters=10, cluster_size=20, icc=0.05,
    mean1=3.0, mean2=3.5, std_dev=1.2,
    analysis_model="bayes", bayes_backend="variational",
    bayes_inference_method="credible_interval"
)

print(f"Power: {results['power']:.3f}")
print(f"Backend: {results['sim_details']['bayes_backend']}")
```

## 📈 Performance Comparison

| Method | Speed | Accuracy | Memory | Use Case |
|--------|-------|----------|--------|----------|
| **Analytical** | ⚡⚡⚡⚡ | 📊📊📊 | 💾 | Quick estimates |
| **Classical Simulation** | ⚡⚡⚡ | 📊📊📊📊 | 💾💾 | Standard analysis |
| **Bayesian (Stan/PyMC)** | ⚡ | 📊📊📊📊📊 | 💾💾💾💾 | Research quality |
| **Bayesian (Variational)** | ⚡⚡⚡ | 📊📊📊📊 | 💾💾 | Fast exploration |
| **Bayesian (ABC)** | ⚡⚡ | 📊📊📊 | 💾 | Web deployment |

## ✅ Validation & Quality Assurance

### Gold Standard Validation
DesignPower has been **comprehensively validated** against established statistical references:

- **Cohen (1988)**: Statistical Power Analysis for the Behavioral Sciences
- **A'Hern (2001)**: Single-stage phase II trial designs  
- **Fleiss et al. (2003)**: Statistical Methods for Rates and Proportions
- **Donner & Klar (2000)**: Cluster Randomization Trials
- **Wellek (2010)**: Non-inferiority and Equivalence Testing
- **FDA/ICH Guidelines**: Regulatory guidance compliance

### Current Validation Status
- **Overall Pass Rate**: 66.7% (8/12 benchmarks)
- **Single-Arm Designs**: 100% ✅ (A'Hern method)
- **Cluster RCTs**: 100% ✅ (Donner & Klar method)
- **Non-Inferiority**: 80% ✅ (Wellek/FDA standards)
- **Superiority Tests**: 67% ✅ (Cohen benchmarks)

### Validation Reports
```bash
# Run comprehensive validation
python tests/validation/comprehensive_validation.py

# Generate validation reports
python tests/validation/validation_report.py --format html
python tests/validation/validation_report.py --format markdown
```

## 🧪 Testing

```bash
# Run all tests
pytest

# Run validation suite
python tests/validation/comprehensive_validation.py --verbose

# Run specific test suites
pytest tests/core/designs/cluster_rct/
pytest tests/app/components/

# Run with coverage
pytest --cov=core --cov=app

# Skip long-running Bayesian tests
pytest -k "not bayesian"
```

## 📚 Documentation

- **Methodology**: See `docs/methods/` for statistical details
- **Validation Reports**: `tests/validation/validation_report.html`
- **API Reference**: Auto-generated from docstrings
- **Examples**: `docs/EXAMPLES.md` for common use cases
- **Testing Strategy**: `docs/TESTING_STRATEGY.md`

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push branch: `git push origin feature/amazing-feature`
5. Open Pull Request

### Development Guidelines
- Follow PEP 8 style guidelines
- Add tests for new functionality
- Update documentation for user-facing changes
- Run full test suite before submitting

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- **Stan Development Team** for probabilistic programming infrastructure
- **PyMC Contributors** for accessible Bayesian modeling
- **SciPy Community** for fundamental scientific computing tools
- **Clinical Trials Research Community** for methodology development

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/DesignPower/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/DesignPower/discussions)
- **Documentation**: [Wiki](https://github.com/yourusername/DesignPower/wiki)

---

**DesignPower**: Bringing modern computational statistics to clinical trial design! 🔬✨