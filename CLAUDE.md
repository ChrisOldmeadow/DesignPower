# CLAUDE.md - DesignPower Project Guidelines

## Project Overview

DesignPower is a comprehensive statistical power analysis application for clinical trial design. It implements various trial designs including parallel group RCTs, single-arm trials, cluster randomized trials, and more advanced designs like stepped wedge and interrupted time series.

## Key Principles

### 1. Statistical Rigor
- **Use established methods only** - Never invent new statistical methods or formulas
- **No arbitrary adjustment factors** - Don't use "fudge factors" to make validation tests pass
- **Document sources** - Always cite the statistical literature or textbook for any formula
- **Match standard software** - Results should align with R, SAS, or other established packages

### 2. Code Quality
- **Keep files under 500 lines** - Split large modules into logical components
- **Test everything** - Always run tests after making changes
- **Use existing structure** - Follow the established directory structure, don't reorganize
- **Clear naming** - Use descriptive function and variable names

### 3. Testing Philosophy
- **Empirical validation** - When in doubt, simulate to verify analytical calculations
- **Accept discrepancies** - Document when our results differ from benchmarks rather than forcing matches
- **Real calculations** - For exact tests (Fisher's, permutation), implement the actual algorithm

## Project Structure

```
DesignPower/
├── app/                      # Streamlit UI application
│   ├── components/          # UI components for each design type
│   └── designpower_app.py   # Main app entry point
├── core/                    # Core statistical implementations
│   ├── designs/            # Design-specific calculations
│   │   ├── cluster_rct/    # Cluster randomized trials
│   │   ├── parallel/       # Parallel group designs
│   │   ├── single_arm/     # Single-arm designs
│   │   └── stepped_wedge/  # Stepped wedge designs
│   ├── simulation.py       # Simulation framework
│   └── utils.py           # Shared utilities
├── tests/                  # Test suite
│   ├── validation/        # Validation against benchmarks
│   └── core/             # Unit tests
└── docs/                  # Documentation
```

## CLI and Dashboard Alignment

### Keep Both Interfaces in Sync
The project has two user interfaces that must remain aligned:
- **CLI** (`cli.py`) - Command-line interface for power calculations
- **Dashboard** (`app/designpower_app.py`) - Streamlit web interface

### When Adding Features
1. **Update both interfaces** - Any new calculation should be available in both CLI and dashboard
2. **Share core logic** - Both should call the same functions in `core/`
3. **Consistent parameters** - Use same parameter names and defaults
4. **Test both paths** - Verify feature works in CLI and dashboard

### Example Workflow
```python
# 1. Add core functionality
# core/designs/parallel/analytical_binary.py
def new_statistical_method(...):
    """Core implementation"""
    pass

# 2. Add to CLI
# cli.py
@cli.command()
def new_method(n1, n2, ...):
    """CLI wrapper"""
    result = new_statistical_method(n1, n2, ...)
    print_results(result)

# 3. Add to Dashboard
# app/components/parallel_rct.py
if method == "new_method":
    result = new_statistical_method(
        n1=st.number_input("Sample size group 1"),
        n2=st.number_input("Sample size group 2"),
        ...
    )
    display_results(result)

# 4. Test both interfaces
# Manual testing: python cli.py new-method --n1 50 --n2 50
# Manual testing: streamlit run app/designpower_app.py
```

## Common Tasks

### Adding a New Statistical Method

1. **Research first** - Find established formulas from statistical literature
2. **Implement in core/** - Add to appropriate module under `core/designs/`
3. **Add tests** - Create unit tests and validation benchmarks
4. **Document sources** - Include citations in docstrings
5. **Update BOTH interfaces** - Add to CLI and Streamlit dashboard
6. **Test both paths** - Ensure consistency between interfaces

### Fixing Validation Discrepancies

1. **Verify the benchmark** - Check if the expected value is correct
2. **Simulate to confirm** - Use Monte Carlo simulation to verify
3. **Check implementation** - Ensure formula matches the source
4. **Document differences** - If discrepancy persists, document why

**DON'T**: Create adjustment factors to force matches
**DO**: Document legitimate differences between methods

### Example of Good Practice

```python
def fishers_exact_power(n1, n2, p1, p2, alpha=0.05):
    """
    Calculate exact power for Fisher's exact test.
    
    Uses complete enumeration of all possible outcomes.
    
    References
    ----------
    Agresti, A. (2007). An Introduction to Categorical Data Analysis, 2nd ed.
    """
    # Actual calculation, not approximation with arbitrary factors
    power = 0.0
    for s1 in range(n1 + 1):
        for s2 in range(n2 + 1):
            prob = binom.pmf(s1, n1, p1) * binom.pmf(s2, n2, p2)
            table = [[s1, n1-s1], [s2, n2-s2]]
            _, p_value = fisher_exact(table)
            if p_value < alpha:
                power += prob
    return power
```

### Example of Bad Practice

```python
def fishers_exact_power_approx(n1, n2, p1, p2, alpha=0.05):
    """AVOID THIS APPROACH"""
    # Don't use arbitrary adjustment factors
    normal_power = calculate_normal_approximation(...)
    
    # BAD: Arbitrary factors to match benchmarks
    if n1 < 50:
        factor = 0.73  # <-- Don't do this!
    else:
        factor = 0.85
    
    return normal_power * factor
```

## Validation Guidelines

### Running Validation Tests
```bash
# Run all validation tests
python tests/validation/run_validation.py

# Run specific validation
python tests/validation/test_fishers_exact_validation.py
```

### When Validation Fails

1. **Check the math** - Verify formula implementation
2. **Check the benchmark** - Benchmarks can have errors too
3. **Simulate** - Use Monte Carlo to find true value
4. **Document** - Record findings in validation notes

## Common Pitfalls to Avoid

1. **Over-engineering** - Keep solutions simple and standard
2. **Breaking existing code** - Always run tests after changes
3. **Reorganizing structure** - Maintain current directory organization
4. **Large files** - Split files exceeding 500 lines
5. **Undocumented formulas** - Always cite sources

## Specific Implementation Notes

### Fisher's Exact Test
- Use exact enumeration for small samples (n ≤ 316)
- Fall back to normal approximation for large samples
- Document computational complexity

### Cluster RCT
- Always account for design effect: DE = 1 + (m-1)*ICC
- Use standard formulas from Donner & Klar or Hayes & Moulton
- Handle ICC edge cases (ICC near 0 or 1)

### Permutation Tests
- Include observed statistic in reference distribution
- Add 1 to numerator and denominator for p-value calculation
- Use exact permutations for small samples, sampling for large

### Simulation Methods
- Set random seeds for reproducibility
- Use sufficient iterations (usually 10,000+)
- Compare simulation results with analytical methods

## Testing Checklist

Before committing changes:
- [ ] Run unit tests: `pytest tests/core/`
- [ ] Run validation tests: `python tests/validation/run_validation.py`
- [ ] Check file sizes: No file over 500 lines
- [ ] Verify calculations: Test with known examples
- [ ] Update documentation: Include sources and formulas
- [ ] Test CLI: Verify new features work in command-line interface
- [ ] Test Dashboard: Verify new features work in Streamlit app
- [ ] Check consistency: Ensure CLI and dashboard give same results

## Resources

### Statistical References
- Chow, S.C. & Liu, J.P. (2008). Design and Analysis of Clinical Trials
- Donner, A. & Klar, N. (2000). Design and Analysis of Cluster Randomization Trials
- Fleiss, J.L. et al. (2003). Statistical Methods for Rates and Proportions
- Hayes, R.J. & Moulton, L.H. (2017). Cluster Randomised Trials

### Software for Validation
- R packages: pwr, clusterPower, blockrand
- Python packages: statsmodels, scipy.stats
- Online calculators: Sealed Envelope, ClinCalc

## Contact for Questions

When working on statistical methods:
1. Consult the literature first
2. Check existing implementations in R/SAS
3. Validate with simulation
4. Document any discrepancies

Remember: **Accuracy over agreement** - It's better to be correct than to match incorrect benchmarks.
