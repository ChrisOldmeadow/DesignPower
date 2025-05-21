# DesignPower Refactoring Guide

## Philosophy and Architecture

### Core Design Principles

The refactoring of DesignPower follows these core principles:

1. **Separation of Concerns**: Each module has a specific responsibility with clear boundaries
2. **Logical Hierarchy**: Code organization follows a natural hierarchy based on study design and outcome type
3. **Clean APIs**: Each module exposes a well-defined API through its `__init__.py` file
4. **Backward Compatibility**: Legacy code paths are maintained through wrapper modules
5. **Progressive Enhancement**: New features can be added independently without affecting existing functionality

### Directory Structure

```
DesignPower/
├── app/                           # Frontend application code
│   ├── components/                # UI components for each design type
│   ├── power_calculations.py      # Legacy wrapper for backward compatibility
│   └── designpower_app.py         # Main application entry point
├── core/                          # Core statistical calculations
│   ├── designs/                   # All study design modules
│   │   ├── parallel/              # Parallel group designs
│   │   │   ├── analytical_binary.py      # Analytical methods for binary outcomes
│   │   │   ├── analytical_continuous.py  # Analytical methods for continuous outcomes
│   │   │   ├── analytical_survival.py    # Analytical methods for survival outcomes
│   │   │   ├── simulation_binary.py      # Simulation methods for binary outcomes
│   │   │   ├── simulation_continuous.py  # Simulation methods for continuous outcomes
│   │   │   └── simulation_survival.py    # Simulation methods for survival outcomes
│   │   ├── single_arm/            # Single arm designs
│   │   │   ├── analytical_binary.py      # Analytical methods for binary outcomes
│   │   │   ├── analytical_continuous.py  # Analytical methods for continuous outcomes
│   │   │   ├── analytical_survival.py    # Analytical methods for survival outcomes
│   │   │   ├── simulation_binary.py      # Simulation methods for binary outcomes
│   │   │   ├── simulation_continuous.py  # Simulation methods for continuous outcomes
│   │   │   └── simulation_survival.py    # Simulation methods for survival outcomes
│   │   ├── cluster_rct/           # Cluster randomized controlled trial designs
│   │   │   ├── analytical_binary.py      # Analytical methods for binary outcomes
│   │   │   ├── analytical_continuous.py  # Analytical methods for continuous outcomes
│   │   │   ├── simulation_binary.py      # Simulation methods for binary outcomes
│   │   │   └── simulation_continuous.py  # Simulation methods for continuous outcomes
│   └── utils/                     # Utility functions
│       ├── formatting.py          # Output formatting utilities
│       ├── statistics.py          # General statistical functions
│       └── validation.py          # Input validation
└── docs/                          # Documentation
```

### Module Organization Pattern

The core organization principle is a clear separation between analytical and simulation methods for each outcome type:

1. **Analytical methods**: Modules implementing closed-form formulas and approximations
   - `analytical_binary.py`: Functions for binary/proportion outcomes using analytical methods
   - `analytical_continuous.py`: Functions for continuous outcomes using analytical methods
   - `analytical_survival.py`: Functions for survival outcomes using analytical methods

2. **Simulation methods**: Modules implementing Monte Carlo simulations
   - `simulation_binary.py`: Functions for binary outcomes using simulation methods
   - `simulation_continuous.py`: Functions for continuous outcomes using simulation methods
   - `simulation_survival.py`: Functions for survival outcomes using simulation methods

Within each module, functions are organized by calculation purpose:
- Sample size calculation
- Power calculation
- Minimum detectable effect calculation

### Calculation Methods

This clear separation supports the UI toggle functionality in the app:

1. **Analytical Methods**:
   - Closed-form formulas and approximations
   - Faster computation
   - Results based on theoretical distributions
   - Suitable for standard scenarios with well-established formulas

2. **Simulation Methods**:
   - Monte Carlo simulations for more complex scenarios
   - Configurable simulation parameters (number of simulations, min/max sample sizes)
   - Support for different statistical test methods (Normal Approximation, Likelihood Ratio, Exact Test)
   - More robust for non-standard scenarios and complex designs
   - Allows for better handling of edge cases and unusual parameter combinations

## Simulation Methods Implementation

### Simulation Capabilities

The simulation capabilities are a key feature of DesignPower, allowing users to:
- Toggle between Analytical and Simulation methods in the UI
- Configure simulation parameters (number of simulations, min/max sample sizes, random seed)
- Choose different statistical test methods (Normal Approximation, Fisher's Exact Test, Likelihood Ratio Test)
- Apply optional continuity correction for improved accuracy
- Obtain robust results for scenarios where analytical formulas are unreliable
- Get additional metrics like empirical power and confidence intervals

### Binary Outcome Simulation Details

For binary outcomes, we've implemented comprehensive simulation capabilities with these key features:

1. **Test Type Selection**
   - **Normal Approximation**: Standard chi-square or z-test approach
   - **Fisher's Exact Test**: More conservative, ideal for small sample sizes
   - **Likelihood Ratio Test**: Often more powerful than normal approximation

2. **Calculation Adjustments**
   - Each test type uses different statistical methods that impact the required sample size
   - Fisher's Exact Test typically requires larger sample sizes than normal approximation
   - Likelihood Ratio tests may have higher power for the same sample size

3. **Continuity Correction**
   - Optional correction to improve accuracy of calculations
   - Applied during z-statistic calculation for normal approximation
   - Generally leads to more conservative (larger) sample size estimates

4. **Parameter Mapping**
   - Consistent parameter naming and mapping between UI and calculation functions
   - Proper string normalization to handle case and format differences

### Implementation Challenges and Solutions

1. **String Normalization**
   - Challenge: Parameter names and test types passed between UI and calculation functions needed to be normalized to handle variations in format (spaces, underscores) and case.
   - Solution: Implemented consistent normalization in both the UI components and calculation functions using `.lower()` and `.replace(" ", "_")` patterns.

2. **Type Consistency**
   - Challenge: Different calculation functions expected slightly different formats for the same parameters.
   - Solution: Created mapping dictionaries in UI components to translate between user-friendly names and internal parameter formats.

3. **Result Extraction**
   - Challenge: Different calculation functions returned results with different dictionary key names.
   - Solution: Implemented generic result extraction patterns that check for multiple potential key names when accessing calculation results.

4. **Error Handling**
   - Challenge: Simulation functions could fail with specific kinds of inputs (e.g., very small proportions).
   - Solution: Added robust error handling and appropriate parameter validation.

### UI Integration for Advanced Options

1. **Modular UI Components**
   - Created dedicated functions for rendering advanced options (`render_binary_advanced_options()`, `render_continuous_advanced_options()`)
   - Separated outcome-specific UI logic to improve maintainability

2. **Parameter Collection and Passing**
   - Used dictionary-based parameter collection to flexibly handle varying parameter sets
   - Implemented conditional UI displays based on selected options

3. **Test Type Selection**
   - Added UI controls for selecting statistical test types
   - Implemented info messages to explain the implications of different test choices

4. **Hypothesis Type Handling**
   - Added different UI options based on hypothesis type (Superiority vs. Non-Inferiority)
   - Implemented dynamic calculation of derived parameters (e.g., p2 from p1 and NIM for non-inferiority)

### Function Naming Conventions

A consistent naming pattern is used across all modules:

1. **Analytical Functions**:
   - `power_<outcome_type>`: Calculate power given sample size
   - `sample_size_<outcome_type>`: Calculate sample size given power
   - `min_detectable_effect_<outcome_type>`: Calculate minimum detectable effect

2. **Simulation Functions**:
   - `power_<outcome_type>_sim`: Calculate power using simulation
   - `sample_size_<outcome_type>_sim`: Calculate sample size using simulation
   - `min_detectable_effect_<outcome_type>_sim`: Calculate minimum detectable effect using simulation
   - `simulate_<outcome_type>_trial`: Core simulation function for standard designs
   - `simulate_<outcome_type>_non_inferiority`: Core simulation function for non-inferiority designs

3. **Non-inferiority Variants**:
   - `power_<outcome_type>_non_inferiority`: Power for non-inferiority designs
   - `sample_size_<outcome_type>_non_inferiority`: Sample size for non-inferiority designs
   - `power_<outcome_type>_non_inferiority_sim`: Power for non-inferiority designs (simulation)
   - `sample_size_<outcome_type>_non_inferiority_sim`: Sample size for non-inferiority designs (simulation)

## Implementation Status

The refactoring has been successfully completed for the following outcome types and study designs:

### Parallel Group Designs

| Outcome Type | Analytical Methods | Simulation Methods | Non-inferiority Support |
|-------------|-------------------|-------------------|--------------------------|
| Binary      | ✅ Complete       | ✅ Complete       | ✅ Complete              |
| Continuous  | ✅ Complete       | ✅ Complete       | ✅ Complete              |
| Survival    | ✅ Complete       | ✅ Complete       | ✅ Complete              |

### Single Arm Designs

| Outcome Type | Analytical Methods | Simulation Methods | Non-inferiority Support |
|-------------|-------------------|-------------------|---------------------------|
| Binary      | ✅ Complete       | 🔄 In Progress    | 🔄 In Progress           |
| Continuous  | 🔄 In Progress    | 🔄 In Progress    | 🔄 In Progress           |
| Survival    | 🔄 In Progress    | 🔄 In Progress    | 🔄 In Progress           |

### Cluster RCT Designs

| Outcome Type | Analytical Methods | Simulation Methods | Non-inferiority Support |
|-------------|-------------------|-------------------|---------------------------|
| Binary      | ✅ Complete       | ✅ Complete       | 🔄 In Progress           |
| Continuous  | ✅ Complete       | ✅ Complete       | 🔄 In Progress           |

### UI Integration

All outcome types (binary, continuous, survival) have been updated in the UI to support:

- Toggle between Analytical and Simulation methods
- Advanced simulation parameters (number of simulations, min/max sample sizes)
- Non-inferiority design options
- Test method selection (for applicable outcome types)

## Adding New Functionality

### Adding a New Design Type

To add a new study design (e.g., cross-over, stepped wedge):

1. Create a new directory under `core/designs/`
   ```bash
   mkdir -p core/designs/new_design_type
   ```

2. Create the standard module structure:
   ```bash
   touch core/designs/new_design_type/__init__.py
   touch core/designs/new_design_type/binary.py
   touch core/designs/new_design_type/continuous.py
   ```

3. Implement the `__init__.py` file to expose the API:
   ```python
   """
   [New design type] study designs.
   
   This module provides functions for power analysis and sample size calculation
   for [new design type] studies with various outcome types.
   """
   
   # Import submodules
   from . import continuous
   from . import binary
   
   # Re-export key functions for binary outcomes
   from .binary import (
       # List key functions here
   )
   
   # Re-export key functions for continuous outcomes
   from .continuous import (
       # List key functions here
   )
   
   # Define what gets imported with "from core.designs.new_design_type import *"
   __all__ = [
       # Submodules
       'binary',
       'continuous',
       
       # List key functions here
   ]
   ```

4. Update the parent `core/designs/__init__.py` to include the new design:
   ```python
   from . import new_design_type
   ```

5. Add appropriate UI components in `app/components/new_design_type.py`

6. Implement both analytical and simulation-based methods for the new design to maintain consistency with existing functionality

### Adding a New Outcome Type to Existing Design

To add a new outcome type (e.g., ordinal, count) to an existing design:

1. Create the new outcome module file:
   ```bash
   touch core/designs/existing_design/new_outcome.py
   ```

2. Implement standard functions following the established pattern:
   ```python
   def sample_size_new_outcome(...):
       """
       Calculate sample size for new outcome type.
       
       Parameters
       ----------
       # Document parameters
       
       Returns
       -------
       dict
           Dictionary containing results
       """
       # Implement calculation
       
   def power_new_outcome(...):
       """
       Calculate power for new outcome type.
       
       Parameters
       ----------
       # Document parameters
       
       Returns
       -------
       dict
           Dictionary containing results
       """
       # Implement calculation
       
   def min_detectable_effect_new_outcome(...):
       """
       Calculate minimum detectable effect for new outcome type.
       
       Parameters
       ----------
       # Document parameters
       
       Returns
       -------
       dict
           Dictionary containing results
       """
       # Implement calculation
   ```

3. Implement simulation-based methods for the new outcome type:
   ```python
   def sample_size_new_outcome_sim(...):
       """
       Calculate sample size for new outcome type using simulation.
       
       Parameters
       ----------
       # Document parameters
       # Include simulation parameters
       
       Returns
       -------
       dict
           Dictionary containing results and simulation details
       """
       # Implement simulation-based calculation
   ```

4. Update the design's `__init__.py` to include the new module:
   ```python
   from . import new_outcome
   
   # Re-export key functions
   from .new_outcome import (
       sample_size_new_outcome,
       power_new_outcome,
       min_detectable_effect_new_outcome
   )
   
   # Update __all__ list
   __all__ += [
       'new_outcome',
       'sample_size_new_outcome',
       'power_new_outcome',
       'min_detectable_effect_new_outcome'
   ]
   ```

5. Update UI components to expose the new outcome type, including options for both analytical and simulation methods

### Implementing a New Calculation Method

To add a new calculation method (e.g., Bayesian approach, adaptive design):

1. Create a new module in the appropriate design directory:
   ```bash
   touch core/designs/existing_design/new_method.py
   ```

2. Implement functions following the established pattern for each outcome type:
   ```python
   def sample_size_continuous_new_method(...):
       # Implementation
       
   def power_continuous_new_method(...):
       # Implementation
   ```

3. Update the relevant outcome module to include the new method:
   ```python
   from .new_method import (
       sample_size_continuous_new_method,
       power_continuous_new_method
   )
   ```

4. Add UI components to allow users to choose this new method alongside existing analytical and simulation approaches

## Developer Checklist

When implementing new functionality:

- [ ] **Documentation**: Add comprehensive docstrings with parameters, returns, examples
- [ ] **Type Hints**: Include type hints for better IDE support
- [ ] **Unit Tests**: Create tests for all new functions
- [ ] **Validation**: Include parameter validation in all user-facing functions
- [ ] **Error Handling**: Provide clear error messages for invalid inputs
- [ ] **Backward Compatibility**: Ensure changes don't break existing code
- [ ] **Performance**: Consider computation efficiency for large simulations
- [ ] **Simulation Support**: Implement both analytical and simulation-based methods
- [ ] **UI Integration**: Update UI components to expose new functionality including method toggle options
- [ ] **Examples**: Add example usage in documentation

## Legacy Code Management

### Strategy for Handling Legacy Code

1. **Deprecation Process**:
   - Mark legacy functions with deprecation warnings
   - Provide migration path in warning messages
   - Set timeline for removal

2. **Wrapper Modules**:
   - Keep wrapper modules like `app/power_calculations.py` for backward compatibility
   - Implement using imports from the new structure
   - Document as "legacy" and point to new alternatives

3. **Documentation**:
   - Clearly mark legacy code sections in documentation
   - Provide migration examples
   - Update all documentation to reference new structure

### Cleaning Up Legacy Code

The following modules contain legacy code that should eventually be refactored:

- `app/power_calculations.py` - Replace with direct imports from core modules
- Legacy simulation modules that are not properly integrated into the new structure:
  - `binary_simulation.py` - Consolidate with the binary.py module
  - `binary_tests.py` - Integrate into the binary.py module
- Duplicate calculation functions - Remove in favor of core implementations

### Implementation Status

1. **Phase 1** (Completed): Basic restructuring and refactoring
   - Created new directory structure
   - Moved core functions to appropriate modules
   - Maintained backward compatibility

2. **Phase 2** (Completed): Outcome-specific module separation
   - Separated functions by outcome type (continuous, binary, survival)
   - Created dedicated analytical and simulation modules
   - Updated UI components to use the new modular structure

3. **Phase 3** (Completed): Simulation methods implementation
   - Added simulation toggle in UI for all outcome types
   - Implemented simulation-specific parameter controls
   - Created simulation-based calculation functions

4. **Phase 4** (Completed): Advanced statistical options
   - Added test type selection for binary outcomes
   - Implemented correction options (continuity correction)
   - Enhanced result formatting and extraction
   - Added non-inferiority hypothesis support for all outcome types

5. **Phase 5** (In Progress): Performance optimizations and enhancements
   - Parameter validation and error handling improvements
   - Code documentation enhancements
   - UI/UX improvements

### Future Work

1. **Phase 6** (Planned): Integration of machine learning models
   - Implement machine learning-based sample size calculations
   - Integrate with existing simulation framework
   - Update UI to include machine learning options

2. **Phase 7** (Planned): Advanced visualization tools
   - Implement interactive visualizations for results
   - Add support for custom visualization options
   - Update UI to include visualization components

3. **Phase 8** (Planned): Cloud deployment and scalability
   - Deploy application on cloud infrastructure
   - Implement scalability features for large-scale simulations
   - Update UI to include cloud-specific features

## Migration Timeline

1. **Phase 1** (Completed): Basic restructuring and refactoring
   - Create new directory structure
   - Move core functions to appropriate modules
   - Maintain backward compatibility

2. **Phase 2** (Current): Documentation and stabilization
   - Document new structure
   - Create migration guides
   - Add comprehensive tests

3. **Phase 3** (Future): Deprecation and cleanup
   - Add deprecation warnings to legacy code
   - Update all internal code to use new structure
   - Plan for eventual removal of legacy interfaces

4. **Phase 4** (Future): Complete migration
   - Remove deprecated interfaces
   - Finalize documentation
   - Release new major version

## Best Practices for Extending DesignPower

1. **Use the Hierarchical Structure**:
   - Place new code in the appropriate design/outcome module
   - Follow established naming conventions

2. **Maintain Clean APIs**:
   - Export only necessary functions from modules
   - Use `__all__` to control what's exposed

3. **Consistent Return Values**:
   - Return dictionaries with standardized keys
   - Include input parameters in results for reference

4. **Support Both Calculation Methods**:
   - Implement both analytical and simulation-based approaches
   - Ensure UI components allow toggling between methods

5. **Documentation First**:
   - Write docstrings before implementation
   - Include examples and references

6. **Scientific Rigor**:
   - Include references to statistical literature
   - Validate against established tools when possible
   - Document assumptions and limitations

7. **User Experience**:
   - Consider how the function will be exposed in the UI
   - Provide sensible defaults
   - Include validation to prevent common errors
   - Ensure appropriate UI controls for simulation parameters

## Contribution Guidelines

1. **Code Style**: Follow PEP 8 and existing project conventions
2. **Branch Strategy**: Create feature branches for new functionality
3. **Pull Requests**: Include tests and documentation
4. **Review Process**: Ensure code review by at least one other developer
5. **Version Control**: Use semantic versioning for releases
