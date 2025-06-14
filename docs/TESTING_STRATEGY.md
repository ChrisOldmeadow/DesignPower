# DesignPower Testing Strategy

This document outlines the comprehensive testing strategy for the DesignPower application, focusing on ensuring correctness, reliability, and maintainability across statistical functions, UI components, and user workflows.

## Testing Architecture Overview

The testing strategy follows a three-tier approach:

1. **Unit Tests** - Test individual statistical functions in isolation
2. **Integration Tests** - Test complete user workflows from UI to results
3. **Validation Tests** - Compare results against authoritative benchmarks

## Test Directory Structure

```
tests/
├── core/                           # Unit tests for statistical functions
│   └── designs/
│       ├── parallel/              # Parallel group RCT tests
│       └── cluster_rct/           # Cluster RCT tests
├── integration/                   # End-to-end integration tests
│   ├── test_ui_calculation_flow.py    # Main UI→calculation→result flow
│   └── test_cluster_calculation_flow.py  # Cluster RCT specific flows
├── validation/                    # Benchmark validation tests
│   ├── test_fishers_exact_validation.py
│   ├── test_cluster_rct_icc_validation.py
│   └── authoritative_benchmarks.py
└── app/                          # Legacy component tests (to be phased out)
    └── components/
```

## 1. Unit Testing

**Purpose**: Test individual statistical functions in isolation to verify mathematical correctness.

**Location**: `tests/core/designs/`

**Coverage**:
- ✅ Parallel RCT analytical methods (binary, continuous, survival)
- ✅ Parallel RCT simulation methods (binary, continuous, survival)
- ✅ Cluster RCT analytical methods (binary, continuous)
- ✅ Cluster RCT simulation methods (binary, continuous)
- ✅ Non-inferiority calculations for all designs
- ✅ Permutation tests for cluster RCTs

**Key Test Files**:
- `tests/core/designs/parallel/test_analytical_binary.py` 
- `tests/core/designs/parallel/test_analytical_continuous.py`
- `tests/core/designs/parallel/test_analytical_survival.py`
- `tests/core/designs/parallel/test_simulation_binary.py`
- `tests/core/designs/parallel/test_simulation_continuous.py`
- `tests/core/designs/parallel/test_simulation_survival.py`
- `tests/core/designs/cluster_rct/test_analytical_binary.py`
- `tests/core/designs/cluster_rct/test_analytical_continuous.py`
- `tests/core/designs/cluster_rct/test_simulation_binary.py`
- `tests/core/designs/cluster_rct/test_simulation_continuous.py`

**Best Practices**:
- Test with realistic clinical trial parameters
- Verify mathematical relationships (e.g., power increases with sample size)
- Test edge cases (very small/large effect sizes, extreme sample sizes)
- Use fixed seeds for reproducible simulation tests

## 2. Integration Testing

**Purpose**: Test complete user workflows from UI input through calculation to final results.

**Location**: `tests/integration/`

**Philosophy**: Replace mocked component tests with real end-to-end integration tests that validate the actual user experience.

### 2.1 Main Integration Tests

**File**: `tests/integration/test_ui_calculation_flow.py` (564 lines)

**Coverage**:
- ✅ Binary outcome calculations (analytical & simulation)
- ✅ Continuous outcome calculations (analytical & simulation)
- ✅ Survival outcome calculations (analytical & simulation)
- ✅ Non-inferiority calculations for all outcomes
- ✅ Parameter validation and error handling
- ✅ Consistency between analytical and simulation methods

**Test Classes**:
- `TestBinaryCalculationFlow` - Complete binary calculation workflows
- `TestContinuousCalculationFlow` - Complete continuous calculation workflows
- `TestSurvivalCalculationFlow` - Complete survival calculation workflows
- `TestParameterValidationFlow` - Error handling and validation
- `TestConsistencyBetweenMethods` - Analytical vs simulation consistency

### 2.2 Cluster RCT Integration Tests

**File**: `tests/integration/test_cluster_calculation_flow.py` (334 lines)

**Coverage**:
- ✅ Cluster binary calculations (analytical & simulation)
- ✅ Cluster continuous calculations (analytical & simulation)
- ✅ ICC effects on sample size requirements
- ✅ Parameter validation for cluster designs
- ✅ Consistency between analytical and simulation methods

**Test Classes**:
- `TestClusterBinaryCalculationFlow`
- `TestClusterContinuousCalculationFlow`
- `TestClusterParameterValidation`
- `TestClusterConsistencyChecks`

### Integration Test Benefits

1. **Real Workflow Testing**: Tests actual parameter flow from UI through calculations
2. **Parameter Mapping Validation**: Ensures UI parameters correctly map to function calls
3. **Result Structure Verification**: Validates results are properly formatted for display
4. **Error Handling**: Tests invalid inputs and edge cases
5. **Method Consistency**: Compares analytical vs simulation results

**Current Status**: 23/32 integration tests passing (72% success rate)

## 3. Validation Testing

**Purpose**: Validate statistical calculations against authoritative external benchmarks.

**Location**: `tests/validation/`

**Methodology**: 
- Compare results with established statistical software (R, SAS, PASS)
- Test against published literature examples
- Validate exact methods (Fisher's exact test, permutation tests)

**Key Files**:
- `tests/validation/test_fishers_exact_validation.py` - Fisher's exact test validation
- `tests/validation/test_cluster_rct_icc_validation.py` - Cluster RCT ICC calculations
- `tests/validation/authoritative_benchmarks.py` - Reference values
- `tests/validation/comprehensive_validation.py` - Full validation suite

**Validation Sources**:
- R packages: `pwr`, `clusterPower`, `survival`
- Literature examples from statistical textbooks
- PASS software results
- Hand-calculated exact solutions

## 4. Legacy Test Cleanup

### Files to Phase Out

**Old Mocked Component Tests** (replace with integration tests):
- `tests/app/components/test_parallel_rct.py` (1940 lines) - **Too large, mock-heavy**
- `tests/app/components/test_parallel_rct_binary.py` - **Redundant with integration tests**
- `tests/app/components/test_parallel_rct_survival.py` - **Redundant with integration tests**
- `tests/app/components/test_parallel_rct_fixed.py` - **Temp fix file**
- `tests/app/components/test_parallel_rct_comprehensive_fix.py` - **Temp fix file**

**Old UI Tests** (replace with integration tests):
- `tests/ui/test_ui_integration.py` (1013 lines) - **Too large, mock-heavy**
- `tests/ui/test_survival_ui_integration.py` - **Redundant with integration tests**
- `tests/ui/simplified_ui_tests.py` - **Incomplete/outdated**

### Files to Keep

**Core Unit Tests** - Essential for mathematical validation:
- All files in `tests/core/designs/` - Keep all unit tests

**Validation Tests** - Critical for correctness:
- All files in `tests/validation/` - Keep all validation tests

**Integration Tests** - New comprehensive approach:
- All files in `tests/integration/` - Keep and expand

**Useful Component Tests**:
- `tests/app/components/test_cluster_rct.py` - Keep if still relevant after review

## 5. Running Tests

### Quick Test Commands

```bash
# Run all unit tests
python -m pytest tests/core/ -v

# Run all integration tests  
python -m pytest tests/integration/ -v

# Run all validation tests
python -m pytest tests/validation/ -v

# Run tests with coverage
python -m pytest tests/ --cov=core --cov=app --cov-report=html

# Run specific test classes
python -m pytest tests/integration/test_ui_calculation_flow.py::TestBinaryCalculationFlow -v
```

### Continuous Integration

```bash
# Full test suite (for CI)
python -m pytest tests/core/ tests/integration/ tests/validation/ --cov=core --cov=app
```

## 6. Test Development Guidelines

### Integration Test Design Principles

1. **Test Real Workflows**: Use actual calculation functions, not mocks
2. **Validate Parameter Flow**: Ensure UI parameters correctly reach core functions
3. **Check Result Structure**: Verify results contain expected keys and values
4. **Test Error Handling**: Include tests for invalid inputs and edge cases
5. **Verify Consistency**: Compare analytical vs simulation methods

### Unit Test Best Practices

1. **Isolated Testing**: Each function tested independently
2. **Realistic Parameters**: Use clinically relevant test values
3. **Mathematical Verification**: Test known relationships and properties
4. **Edge Case Coverage**: Test boundary conditions and extremes
5. **Reproducible Results**: Use fixed seeds for randomized tests

### Validation Test Requirements

1. **Authoritative Sources**: Compare against established tools/literature
2. **Documented Benchmarks**: Clear source citation for expected values
3. **Tolerance Specification**: Define acceptable differences for comparisons
4. **Method Documentation**: Document validation methodology

## 7. Current Test Status

### ✅ Completed
- **Unit Tests**: Comprehensive coverage of core statistical functions
- **Integration Tests**: Complete UI→calculation→result flow testing
- **Validation Tests**: Fisher's exact, cluster RCT ICC, literature benchmarks
- **Permutation Tests**: Exact permutation test implementations

### 🔄 In Progress
- **Test Documentation**: Updating strategy documentation
- **Test Cleanup**: Removing obsolete mock-based tests
- **Integration Test Expansion**: Fixing remaining 9 failing integration tests

### 📋 Planned
- **Performance Tests**: Simulation speed and memory usage testing
- **Property-Based Tests**: Automated testing of mathematical relationships
- **UI Component Tests**: Direct Streamlit component testing (if needed)

## 8. Quality Metrics

### Test Coverage Targets
- **Core Functions**: >90% line coverage
- **Integration Workflows**: 100% of user-facing calculation types
- **Validation Coverage**: All major statistical methods validated

### Success Criteria
- **Unit Tests**: All core statistical functions pass mathematical validation
- **Integration Tests**: >95% pass rate for complete user workflows  
- **Validation Tests**: Results within acceptable tolerance of benchmarks
- **Performance**: Simulation tests complete within reasonable time limits

## 9. Maintenance Strategy

### Regular Activities
1. **Test Review**: Monthly review of test failures and coverage
2. **Benchmark Updates**: Annual validation against latest software versions
3. **Documentation Sync**: Keep test docs aligned with code changes
4. **Performance Monitoring**: Track test execution times and optimize slow tests

### Continuous Improvement
1. Add new tests for bug fixes and feature additions
2. Refactor tests to improve readability and maintainability
3. Update validation benchmarks with new authoritative sources
4. Optimize test suite performance for faster CI/CD cycles

---

## Conclusion

This testing strategy emphasizes **real integration testing** over mocked component tests, **comprehensive validation** against authoritative benchmarks, and **maintainable unit tests** for core statistical functions. The approach ensures both mathematical correctness and end-to-end user workflow validation while keeping the test suite fast and reliable.