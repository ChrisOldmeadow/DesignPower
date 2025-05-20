# DesignPower Testing Strategy

This document outlines a comprehensive testing strategy for the DesignPower application, focusing on ensuring correctness, reliability, and maintainability of the refactored codebase.

## Testing Levels

### 1. Unit Testing

Unit tests focus on testing individual functions in isolation to verify their correctness.

- [ ] **Setup Testing Environment**
  - [ ] Create a `tests/` directory structure mirroring the core structure
  - [ ] Set up a test runner and configuration
  - [ ] Configure coverage reporting

- [ ] **Binary Outcomes Testing**
  - [ ] Test analytical methods
    - [ ] `power_binary()`
    - [ ] `sample_size_binary()`
    - [ ] `min_detectable_effect_binary()`
    - [ ] `power_binary_non_inferiority()`
    - [ ] `sample_size_binary_non_inferiority()`
  - [ ] Test simulation methods
    - [ ] `power_binary_sim()`
    - [ ] `sample_size_binary_sim()`
    - [ ] `min_detectable_effect_binary_sim()`
    - [ ] `power_binary_non_inferiority_sim()`
    - [ ] `sample_size_binary_non_inferiority_sim()`
    - [ ] `min_detectable_binary_non_inferiority_margin_sim()`
    - [ ] `simulate_binary_trial()`
    - [ ] `simulate_binary_non_inferiority()`

- [ ] **Continuous Outcomes Testing**
  - [ ] Test analytical methods
    - [ ] `power_continuous()`
    - [ ] `sample_size_continuous()`
    - [ ] `min_detectable_effect_continuous()`
    - [ ] `power_continuous_non_inferiority()`
    - [ ] `sample_size_continuous_non_inferiority()`
  - [ ] Test simulation methods
    - [ ] `power_continuous_sim()`
    - [ ] `sample_size_continuous_sim()`
    - [ ] `min_detectable_effect_continuous_sim()`
    - [ ] `power_continuous_non_inferiority_sim()`
    - [ ] `sample_size_continuous_non_inferiority_sim()`
    - [ ] `simulate_continuous_trial()`
    - [ ] `simulate_continuous_non_inferiority()`

- [ ] **Survival Outcomes Testing**
  - [ ] Test analytical methods
    - [ ] `power_survival()`
    - [ ] `sample_size_survival()`
    - [ ] `min_detectable_effect_survival()`
    - [ ] `power_survival_non_inferiority()`
    - [ ] `sample_size_survival_non_inferiority()`
  - [ ] Test simulation methods
    - [ ] `power_survival_sim()`
    - [ ] `sample_size_survival_sim()`
    - [ ] `min_detectable_effect_survival_sim()`
    - [ ] `power_survival_non_inferiority_sim()`
    - [ ] `sample_size_survival_non_inferiority_sim()`
    - [ ] `simulate_survival_trial()`
    - [ ] `simulate_survival_non_inferiority()`

### 2. Integration Testing

Integration tests verify that modules work together correctly.

- [ ] **Module Integration**
  - [ ] Test analytical vs. simulation consistency
    - [ ] Binary outcomes
    - [ ] Continuous outcomes
    - [ ] Survival outcomes
  - [ ] Test legacy compatibility
    - [ ] Binary module compatibility
    - [ ] Continuous module compatibility
    - [ ] Survival module compatibility
  - [ ] Test UI and calculation integration
    - [ ] Binary UI + calculations
    - [ ] Continuous UI + calculations
    - [ ] Survival UI + calculations

### 3. Regression Testing

Regression tests ensure that changes don't break existing functionality.

- [ ] **Benchmark Test Suite**
  - [ ] Create benchmark cases with known outcomes
    - [ ] Binary outcomes benchmarks
    - [ ] Continuous outcomes benchmarks
    - [ ] Survival outcomes benchmarks
  - [ ] Implement automated regression testing
    - [ ] Set up automated comparison with benchmarks
    - [ ] Create regression test runner

### 4. Parameterized Testing

Parameterized tests cover a wide range of inputs to ensure robustness.

- [ ] **Binary Outcomes**
  - [ ] Parameterized tests for `sample_size_binary()`
  - [ ] Parameterized tests for `power_binary()`
  - [ ] Parameterized tests for non-inferiority functions

- [ ] **Continuous Outcomes**
  - [ ] Parameterized tests for `sample_size_continuous()`
  - [ ] Parameterized tests for `power_continuous()`
  - [ ] Parameterized tests for non-inferiority functions

- [ ] **Survival Outcomes**
  - [ ] Parameterized tests for `sample_size_survival()`
  - [ ] Parameterized tests for `power_survival()`
  - [ ] Parameterized tests for non-inferiority functions

### 5. Property-Based Testing

Property-based tests verify that certain properties hold across all valid inputs.

- [ ] **Universal Properties**
  - [ ] Power increases with sample size
  - [ ] Power increases with effect size
  - [ ] Sample size decreases with effect size
  - [ ] Sample size increases with power
  - [ ] Non-inferiority margin affects power/sample size

- [ ] **Simulation Properties**
  - [ ] Simulation results converge with more simulations
  - [ ] Simulation variance decreases with more simulations
  - [ ] Simulation matches analytical results asymptotically

### 6. Edge Case Testing

Edge case tests focus on boundary conditions and extreme inputs.

- [ ] **Binary Outcomes Edge Cases**
  - [ ] Very small proportions (near 0)
  - [ ] Very large proportions (near 1)
  - [ ] Very small effect size
  - [ ] Equal proportions
  - [ ] Extreme sample sizes

- [ ] **Continuous Outcomes Edge Cases**
  - [ ] Very small effect size
  - [ ] Very large effect size
  - [ ] Very small/large standard deviations
  - [ ] Equal means
  - [ ] Extreme sample sizes

- [ ] **Survival Outcomes Edge Cases**
  - [ ] Very short/long median survival times
  - [ ] Very small/large hazard ratios
  - [ ] Equal survival times
  - [ ] Extreme enrollment/follow-up periods
  - [ ] High dropout rates

### 7. Comparison with External Tools

Tests that compare results with established external statistical tools.

- [ ] **Binary Outcomes Comparisons**
  - [ ] Compare with G*Power
  - [ ] Compare with published results from literature
  - [ ] Compare with R packages (e.g., pwr)

- [ ] **Continuous Outcomes Comparisons**
  - [ ] Compare with G*Power
  - [ ] Compare with published results from literature
  - [ ] Compare with R packages (e.g., pwr)

- [ ] **Survival Outcomes Comparisons**
  - [ ] Compare with published survival analysis tools
  - [ ] Compare with R packages (e.g., powerSurvEpi)
  - [ ] Compare with PASS software results

### 8. Simulation Validation

Tests that validate the statistical properties of simulation methods.

- [ ] **Simulation Methodology Tests**
  - [ ] Test simulation convergence
  - [ ] Test simulation variance
  - [ ] Test simulation bias
  - [ ] Test sensitivity to seed values

- [ ] **Binary Simulation Validation**
  - [ ] Validate `simulate_binary_trial()` output distribution
  - [ ] Validate Type I error rate
  - [ ] Validate power estimation accuracy

- [ ] **Continuous Simulation Validation**
  - [ ] Validate `simulate_continuous_trial()` output distribution
  - [ ] Validate Type I error rate
  - [ ] Validate power estimation accuracy

- [ ] **Survival Simulation Validation**
  - [ ] Validate `simulate_survival_trial()` output distribution
  - [ ] Validate Type I error rate
  - [ ] Validate power estimation accuracy

### 9. UI Component Testing

Tests for UI components that ensure they correctly use backend functions.

- [ ] **Binary UI Components**
  - [ ] Test `render_parallel_binary()` 
  - [ ] Test `calculate_parallel_binary()`
  - [ ] Test UI toggle for analytical/simulation methods
  - [ ] Test non-inferiority UI options

- [ ] **Continuous UI Components**
  - [ ] Test `render_parallel_continuous()`
  - [ ] Test `calculate_parallel_continuous()`
  - [ ] Test UI toggle for analytical/simulation methods
  - [ ] Test non-inferiority UI options

- [ ] **Survival UI Components**
  - [ ] Test `render_parallel_survival()`
  - [ ] Test `calculate_parallel_survival()`
  - [ ] Test UI toggle for analytical/simulation methods
  - [ ] Test non-inferiority UI options

### 10. Performance Testing

Tests to ensure computational efficiency, particularly for simulation methods.

- [ ] **Computational Performance**
  - [ ] Benchmark simulation performance
  - [ ] Identify performance bottlenecks
  - [ ] Test with large numbers of simulations
  - [ ] Test with extreme sample sizes

- [ ] **Memory Usage**
  - [ ] Test memory consumption during simulations
  - [ ] Identify memory leaks or inefficiencies
  - [ ] Optimize memory usage for large simulations

## Implementation Phases

### Phase 1: Core Unit Tests

- [ ] Set up testing framework and directory structure
- [ ] Implement unit tests for analytical functions
- [ ] Set up continuous integration for automated testing
- [ ] Achieve minimum code coverage target (80%)

### Phase 2: Integration and Regression Tests

- [ ] Develop integration tests between modules
- [ ] Create regression test suite with benchmark cases
- [ ] Implement property-based tests for key invariants
- [ ] Set up automated regression testing pipeline

### Phase 3: Advanced Testing

- [ ] Add edge case tests and boundary testing
- [ ] Implement simulation validation tests
- [ ] Create comparison tests with external tools
- [ ] Extend code coverage to 90%+

### Phase 4: UI and End-to-End Testing

- [ ] Develop tests for UI components
- [ ] Create end-to-end tests for complete workflows
- [ ] Implement performance testing for simulation methods
- [ ] Set up automated UI testing

## Best Practices

1. **Test Documentation**: Each test should have a clear purpose documented in the test function docstring.
2. **Test Independence**: Tests should not depend on each other's state.
3. **Test Organization**: Organize tests in a logical hierarchy mirroring the code structure.
4. **Realistic Inputs**: Use realistic input values that reflect actual usage.
5. **Test Fixtures**: Use fixtures for common setup/teardown procedures.
6. **Avoid Randomness**: Use fixed seeds for randomized tests to ensure reproducibility.
7. **Test Coverage**: Aim for comprehensive coverage, especially of complex logic branches.
8. **Readable Tests**: Make tests readable and maintainable with clear naming.
9. **Automated Testing**: Run tests automatically on code changes.
10. **Continuous Improvement**: Regularly review and improve the test suite.

## Test Examples

### Unit Test Example

```python
import unittest
from core.designs.parallel import analytical_binary

class TestAnalyticalBinary(unittest.TestCase):
    
    def test_power_binary(self):
        """Test that power calculation gives expected results"""
        result = analytical_binary.power_binary(
            n1=100, n2=100, p1=0.3, p2=0.5, alpha=0.05
        )
        self.assertAlmostEqual(result["power"], 0.94, places=2)
```

### Integration Test Example

```python
def test_analytical_vs_simulation_consistency(self):
    """Test that analytical and simulation methods give consistent results"""
    # Parameters
    p1, p2 = 0.3, 0.5
    n1, n2 = 100, 100
    
    # Get analytical power
    analytical_result = analytical_binary.power_binary(
        n1=n1, n2=n2, p1=p1, p2=p2, alpha=0.05
    )
    
    # Get simulation power with many simulations for accuracy
    sim_result = simulation_binary.power_binary_sim(
        n1=n1, n2=n2, p1=p1, p2=p2, alpha=0.05, nsim=10000
    )
    
    # Results should be within a reasonable margin (e.g., 5%)
    self.assertLess(abs(analytical_result["power"] - sim_result["power"]), 0.05)
```

## Running Tests

To run the tests, use:

```bash
# Run all tests
python -m unittest discover tests

# Run a specific test file
python -m unittest tests.designs.parallel.test_analytical_binary

# Run tests with coverage
coverage run -m unittest discover
coverage report -m
coverage html  # Generate HTML report
```
