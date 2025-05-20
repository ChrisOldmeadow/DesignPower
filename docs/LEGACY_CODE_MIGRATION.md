# DesignPower Legacy Code Migration Guide

This document provides specific guidance on migrating from the legacy code structure to the new refactored architecture, with particular focus on preserving the simulation capabilities that have been developed.

## Simulation Capabilities Overview

The DesignPower application includes comprehensive simulation capabilities that must be preserved during refactoring:

1. **User Interface Features**:
   - Toggle between Analytical and Simulation methods in the advanced options
   - Test method selection (Normal Approximation, Likelihood Ratio, Exact Test)
   - Simulation parameters (number of simulations, min/max sample sizes)

2. **Binary Outcome Simulation**:
   - Functions for sample size determination via discrete optimization
   - Power calculation using various statistical tests
   - Minimum detectable effect calculation

3. **Continuous Outcome Simulation**:
   - Sample size calculation via optimization
   - Direct simulation for power analysis
   - Minimum detectable effect calculation

## Migration Strategy

### Phase 1: Maintain Dual Structure (Current)

During the initial refactoring phase, we've maintained both the legacy structure and the new structure:

1. **Legacy Path**:
   ```
   app/power_calculations.py --> core/designs/parallel/binary_simulation.py
                             --> core/designs/parallel/simulation.py
   ```

2. **New Path**:
   ```
   app/components/* --> core/designs/parallel/binary.py
                     --> core/designs/parallel/continuous.py
                     --> core/designs/single_arm/binary.py
                     --> core/designs/single_arm/continuous.py
                     --> core/designs/single_arm/survival.py
   ```

3. **Compatibility Layer**:
   - `app/power_calculations.py` re-exports functions from the new structure
   - Legacy simulation modules are still imported directly

### Phase 2: Complete the Migration (Next Steps)

To complete the migration while preserving all simulation capabilities, we'll implement a clear separation between analytical and simulation methods for each outcome type:

1. **Create Dedicated Analytical and Simulation Modules**:
   - Create separate modules for analytical and simulation methods for each outcome type
   - For example: `analytical_binary.py`, `simulation_binary.py`, `analytical_continuous.py`, `simulation_continuous.py`
   - This clear separation aligns with the UI toggle between analytical and simulation methods

2. **Move Functions to Appropriate Modules**:
   - Move analytical functions from legacy modules to their corresponding analytical modules
   - Move simulation functions from legacy modules to their corresponding simulation modules
   - Ensure all simulation parameters are properly supported in the new structure

3. **Update Component References**:
   - Update all components to use the new module structure
   - Ensure simulation toggle functionality works with the new structural organization

4. **Deprecate Legacy Modules**:
   - Add deprecation warnings to legacy modules
   - Document migration paths for each function

## Function Mapping

### Binary Outcome Functions

| Legacy Function | New Location | Migration Status |
|----------------|--------------|------------------|
| `binary_simulation.sample_size_binary_sim` | `parallel.simulation_binary.sample_size_binary_sim` | To be migrated |
| `binary_simulation.power_binary_sim` | `parallel.simulation_binary.power_binary_sim` | To be migrated |
| `binary_simulation.min_detectable_effect_binary_sim` | `parallel.simulation_binary.min_detectable_effect_binary_sim` | To be migrated |
| `binary_tests.power_binary_with_test` | `parallel.simulation_binary.power_binary_with_test` | To be migrated |
| `analytical.sample_size_binary` | `parallel.analytical_binary.sample_size_binary` | To be migrated |
| `analytical.power_binary` | `parallel.analytical_binary.power_binary` | To be migrated |

### Continuous Outcome Functions

| Legacy Function | New Location | Migration Status |
|----------------|--------------|------------------|
| `simulation.sample_size_continuous_sim` | `parallel.simulation_continuous.sample_size_continuous_sim` | To be migrated |
| `simulation.simulate_continuous` | `parallel.simulation_continuous.simulate_continuous` | To be migrated |
| `simulation.min_detectable_effect_continuous` | `parallel.simulation_continuous.min_detectable_effect_continuous` | To be migrated |
| `analytical.sample_size_continuous` | `parallel.analytical_continuous.sample_size_continuous` | To be migrated |
| `analytical.power_continuous` | `parallel.analytical_continuous.power_continuous` | To be migrated |

## Updating UI Components

When updating UI components to use the new module structure:

1. **Preserve Method Toggle**:
   ```python
   # Example toggle in UI component
   method_type = st.radio("Calculation Method", ["Analytical", "Simulation"])
   use_simulation = method_type == "Simulation"
   
   # Simulation parameters if simulation is selected
   if use_simulation:
       st.subheader("Simulation Parameters")
       nsim = st.number_input("Number of Simulations", min_value=100, value=1000)
       # Additional simulation parameters
   ```

2. **Update Function Calls**:
   ```python
   # Original function call
   if use_simulation:
       result = binary_simulation.sample_size_binary_sim(...)
   else:
       result = analytical.sample_size_binary(...)
   
   # Updated function call (with clear modules for each methodology)
   if use_simulation:
       result = parallel.simulation_binary.sample_size_binary_sim(...)
   else:
       result = parallel.analytical_binary.sample_size_binary(...)
   ```

3. **Ensure Test Method Selection**:
   ```python
   # Preserve test method selection
   test_type = st.selectbox("Test Method", 
                           ["Normal Approximation", "Likelihood Ratio", "Exact Test"])
   
   # Pass to appropriate function
   result = parallel_binary.sample_size_binary(
       # other parameters
       test_type=test_type
   )
   ```

## Testing During Migration

To ensure that the migration preserves all functionality:

1. **Comparison Testing**:
   - Run calculations using both legacy and new functions
   - Compare results to ensure they match

2. **UI Testing**:
   - Test the UI with both analytical and simulation methods
   - Verify that all simulation parameters work correctly

3. **Edge Case Testing**:
   - Test with extreme parameter values
   - Verify that error handling works as expected

## Final Cleanup Checklist

Once the migration is complete:

- [ ] Remove redundant simulation modules
- [ ] Update docstrings to reflect new structure
- [ ] Add deprecation warnings to `app/power_calculations.py`
- [ ] Update all examples and documentation
- [ ] Run full test suite to verify functionality
