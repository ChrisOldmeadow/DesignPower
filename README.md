# DesignPower: Statistical Power Analysis for Clinical Trials

A comprehensive, validated application for statistical power analysis and sample size calculation across diverse clinical trial designs. Built with rigorous statistical methodology and extensive validation against published gold standards.

## ✨ Features

### Trial Designs Supported
- **Parallel RCTs**: Two-arm randomized controlled trials  
- **Single-Arm Trials**: Phase II designs (A'Hern, Simon's two-stage)
- **Cluster RCTs**: Cluster randomized trials with ICC modeling
- **Interrupted Time Series**: Time series intervention analysis
- **Stepped Wedge**: Sequential cluster rollout designs

### Outcome Types
- **Continuous**: Normal distributions with t-distribution corrections
- **Binary**: Proportions and rates with exact/asymptotic methods
- **Survival**: Time-to-event analysis with censoring
- **Count**: Poisson and overdispersed count outcomes

### Analysis Methods
- **Sample Size**: Calculate required participants for target power
- **Power**: Estimate statistical power for given sample sizes  
- **Minimum Detectable Effect**: Find smallest detectable differences
- **Superiority & Non-Inferiority Testing**: Multiple hypothesis frameworks

## 📊 Validation & Quality

**96.0% validation success rate** against established statistical references:
- Cohen (1988): Statistical Power Analysis for the Behavioral Sciences
- A'Hern (2001) & Simon (1989): Single-stage and two-stage phase II designs  
- Donner & Klar (2000): Cluster Randomization Trials
- Wellek (2010): Non-inferiority and Equivalence Testing
- Hayes & Moulton (2017): Cluster Randomised Trials

### Fully Validated Components (100% accuracy)
- ✅ Single-arm trials (A'Hern and Simon's methods)
- ✅ Cluster RCTs (continuous and binary outcomes)
- ✅ Non-inferiority testing (continuous and binary)
- ✅ Repeated measures designs (ANCOVA and change score)

## 🚀 Installation

### Basic Setup
```bash
git clone https://github.com/yourusername/DesignPower.git
cd DesignPower
pip install -r requirements.txt
```

### System Requirements
- Python 3.8+
- Core dependencies: numpy, scipy, pandas, matplotlib, streamlit
- Optional Bayesian backends: cmdstanpy, pymc

### For Development
```bash
pip install -e .
pytest  # Run test suite
```

## 💻 Usage

### Web Interface (Recommended)
```bash
streamlit run app/designpower_app.py
```
Access at http://localhost:8501

### Command Line Interface
```bash
# Parallel RCT sample size
python cli.py parallel sample-size --outcome continuous --delta 0.5 --std-dev 1.0 --power 0.8

# Cluster RCT power analysis  
python cli.py cluster power --outcome binary --p1 0.3 --p2 0.5 --cluster-size 20 --icc 0.05 --n-clusters 10

# Single-arm A'Hern design
python cli.py single-arm sample-size --design ahern --p 0.3 --p0 0.1 --alpha 0.05 --power 0.8

# Get help for specific designs
python cli.py parallel --help
python cli.py cluster --help
```

### REST API
```bash
uvicorn api.main:app --reload
```
API documentation: http://localhost:8000/docs

### Python API
```python
from core.designs.parallel.analytical_continuous import sample_size_two_sample_t_test

# Two-sample t-test sample size
result = sample_size_two_sample_t_test(
    delta=0.5, std_dev=1.0, power=0.8, alpha=0.05
)
print(f"Required sample size: {result['n1']} per group")

# Cluster RCT with ICC
from core.designs.cluster_rct.analytical_continuous import sample_size_crt_continuous
result = sample_size_crt_continuous(
    delta=0.5, std_dev=1.0, icc=0.05, cluster_size=20, power=0.8
)
print(f"Required clusters: {result['n_clusters']} per arm")
```

## 🧪 Testing & Validation

### Run Tests
```bash
# Full test suite
pytest

# Specific test categories
pytest tests/core/designs/parallel/
pytest tests/core/designs/cluster_rct/
pytest tests/core/designs/single_arm/

# Validation against benchmarks
python tests/validation/comprehensive_validation.py

# Generate validation reports
python tests/validation/validation_report.py --format html
```

### Test Coverage
Current test suite includes:
- **224 unit tests** across statistical functions
- **Validation tests** against 25 published benchmarks  
- **Integration tests** for UI and CLI workflows
- **Edge case testing** for parameter validation

## 📁 Project Structure

```
DesignPower/
├── app/                    # Streamlit web interface
│   ├── designpower_app.py  # Main application entry
│   └── components/         # UI components by design type
├── cli/                    # Modular CLI commands  
│   ├── commands/           # Command implementations
│   └── common/             # Shared CLI utilities
├── core/                   # Statistical calculation engine
│   ├── designs/            # Design-specific methods
│   │   ├── parallel/       # Parallel RCT methods
│   │   ├── single_arm/     # Single-arm methods  
│   │   ├── cluster_rct/    # Cluster RCT methods
│   │   └── stepped_wedge/  # Stepped wedge methods
│   ├── outcomes/           # Outcome-specific utilities
│   └── utils/              # Report generation & validation
├── api/                    # FastAPI REST interface
├── tests/                  # Comprehensive test suite
│   ├── core/               # Unit tests for statistical functions
│   ├── validation/         # Benchmark validation tests
│   └── integration/        # End-to-end workflow tests
└── docs/                   # Documentation and methodology
```

## 🔬 Key Statistical Features

### Advanced Clustering Support
- **ICC modeling**: Linear and logit scale with automatic conversion
- **Unequal cluster sizes**: Design effect adjustments
- **Small sample corrections**: Methods for few clusters (<30)
- **Mixed model integration**: LMM/GLMM with robust standard errors

### Rigorous Implementation
- **t-distribution corrections** for continuous outcomes when n < 30
- **Exact methods** for small samples (Fisher's exact, permutation tests)
- **Design effect calculations** for cluster randomization
- **Satterthwaite approximations** for unequal variances

### Bayesian Methods (Optional)
- **Stan & PyMC backends** for full MCMC analysis
- **Variational approximations** for faster computation
- **Credible intervals** and posterior probabilities
- **ROPE testing** for practical equivalence

## 🤝 Contributing

We welcome contributions! Please:

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/amazing-feature`
3. **Follow** our coding standards (see CLAUDE.md)
4. **Add tests** for new functionality  
5. **Run validation**: `pytest tests/validation/`
6. **Submit** a pull request

### Development Guidelines
- Follow PEP 8 style conventions
- Add comprehensive unit tests
- Validate against published benchmarks when possible
- Keep functions under 50 lines, files under 500 lines
- Document statistical methods with literature citations

### Areas for Contribution
- Additional trial designs (crossover, factorial)
- New outcome types (ordinal, composite endpoints)
- Enhanced Bayesian methods
- Performance optimizations
- Documentation improvements

## 📚 Documentation

- **Statistical Methods**: `docs/methods/` - Detailed methodology
- **Validation Reports**: `tests/validation/validation_report.html`
- **Examples**: `docs/EXAMPLES.md` - Common use cases
- **Testing Guide**: `docs/TESTING_STRATEGY.md`
- **Development Guide**: `CLAUDE.md` - Code standards and principles

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- Statistical methodology based on Cohen (1988), Donner & Klar (2000), and other seminal works
- Validation benchmarks from published clinical trial literature
- Open source scientific Python ecosystem (SciPy, NumPy, pandas)
- Bayesian computing infrastructure (Stan, PyMC)

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/DesignPower/issues)
- **Documentation**: [Project Wiki](https://github.com/yourusername/DesignPower/wiki)  
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/DesignPower/discussions)

---

**Built for researchers, by researchers** - Bringing rigorous statistical methodology to clinical trial design 🔬