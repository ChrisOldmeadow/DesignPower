# DesignPower Validation Framework

This directory contains comprehensive validation tests to ensure DesignPower's accuracy and reliability against established benchmarks.

## Validation Strategy

### 1. Literature Benchmarks (`literature_benchmarks.py`)

Validation against published examples from authoritative sources:

**Parallel RCT:**
- **Fleiss et al. (1973)** - Binary outcome sample sizes
- **Cohen (1988)** - Continuous outcome power analysis  
- **Schoenfeld (1981)** - Survival analysis methods

**Cluster RCT:**
- **Donner & Klar (2000)** - Cluster randomization design
- **Campbell et al. (2004)** - CONSORT cluster extension

**Single Arm:**
- **A'Hern (2001)** - Single-stage phase II designs
- **Simon (1989)** - Two-stage optimal designs

### 2. Software Cross-Validation (`software_comparison.py`)

Comparison against established software packages:

- **R packages:** `pwr`, `clusterPower`, `survival`
- **SAS PROC POWER**
- **G*Power**
- **nQuery/nTerim** (planned)

### 3. Edge Case Testing (`edge_cases.py`)

Testing boundary conditions and edge cases:
- Very small/large sample sizes
- Extreme effect sizes
- Boundary alpha/power values
- Unusual allocation ratios

### 4. Analytical Solutions (`analytical_validation.py`)

Validation against known closed-form solutions:
- Normal approximations
- Exact binomial tests  
- Chi-square tests
- t-test formulas

## Running Validation Tests

### Individual Test Modules

```bash
# Literature benchmarks
python tests/validation/literature_benchmarks.py

# Software comparisons  
python tests/validation/software_comparison.py

# Edge cases
python tests/validation/edge_cases.py
```

### All Validation Tests

```bash
# Run complete validation suite
python tests/validation/run_all.py

# With detailed output
python tests/validation/run_all.py --verbose

# Generate validation report
python tests/validation/run_all.py --report validation_report.html
```

### Integration with pytest

```bash
# Run as part of test suite
pytest tests/validation/ -v

# Run specific validation category
pytest tests/validation/test_literature.py -v
```

## Validation Criteria

### Tolerance Levels

- **Sample Size Calculations:** ±5% (accounts for rounding differences)
- **Power Calculations:** ±2% (more precise)
- **Survival Analysis:** ±10% (inherent variability)
- **Cluster Designs:** ±10% (approximation methods)

### Success Metrics

- **Target:** ≥95% of benchmarks pass validation
- **Minimum Acceptable:** ≥90% pass rate
- **Alert Level:** <85% requires investigation

### Documentation Requirements

Each benchmark must include:
- Source reference (paper, software, manual)
- Page/section number
- Complete parameter specification
- Expected results with precision
- Tolerance justification

## Adding New Benchmarks

### Literature Benchmark Template

```python
LiteratureBenchmark(
    source="Author (Year) - Paper Title",
    page="Chapter X, Example Y",  
    example_name="Descriptive name",
    design_type="parallel|cluster|single_arm",
    outcome_type="binary|continuous|survival",
    parameters={
        # Complete parameter specification
    },
    expected_result={
        # Expected results from literature
    },
    tolerance=0.05  # ±5%
)
```

### Software Comparison Template

```python
SoftwareComparison(
    software="Software Name Version",
    function_call="exact_function_call(params)",
    parameters={
        # DesignPower parameter mapping
    },
    expected_result={
        # Software output results
    },
    tolerance=0.02  # ±2%
)
```

## Validation Reports

Automated validation reports include:

1. **Summary Statistics**
   - Total benchmarks tested
   - Pass/fail rates by category
   - Performance trends over time

2. **Detailed Results** 
   - Individual benchmark results
   - Error analysis for failures
   - Tolerance utilization

3. **Recommendations**
   - Areas needing improvement
   - New benchmarks to add
   - Tolerance adjustments

## Continuous Integration

Validation tests run automatically:

- **Every commit:** Core validation subset (fast benchmarks)
- **Daily:** Complete validation suite  
- **Release:** Full validation + manual review
- **Weekly:** Cross-software validation (requires external tools)

## Troubleshooting Failed Validations

1. **Check tolerance levels** - May need adjustment for approximation methods
2. **Verify parameter mapping** - Ensure correct translation between systems
3. **Review literature source** - Confirm benchmark interpretation
4. **Check for software updates** - External software may have changed
5. **Investigate edge cases** - May reveal implementation improvements needed

---

*This validation framework ensures DesignPower maintains the highest standards of accuracy and reliability for clinical trial design.*