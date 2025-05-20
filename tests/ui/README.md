# DesignPower UI Integration Tests

This directory contains integration tests for the DesignPower application's user interface components. These tests verify that the UI components correctly interact with the backend calculation functions and properly process user input.

## Overview

The UI integration tests are designed to validate:

1. The correct conversion of UI inputs to calculation parameters
2. The proper integration between UI components and calculation functions
3. That simulation methods are correctly toggled and configured in the UI
4. That switching between calculation types properly updates the UI
5. The correct handling of non-inferiority and superiority test types

## Test Structure

The tests are organized into:

- **General UI Tests** (`test_ui_integration.py`): Basic tests for UI components across different outcome types
- **Outcome-Specific Tests** (e.g., `test_survival_ui_integration.py`): Tests specific to a particular outcome type
- **UI Testing Utilities** (`ui_test_utils.py`): Common utilities and helper functions for UI testing

## Running Tests

To run all UI integration tests:

```bash
python -m pytest tests/ui
```

To run a specific UI test file:

```bash
python -m pytest tests/ui/test_ui_integration.py
```

## Approach to UI Testing

The UI tests use a combination of:

1. **Mocking Streamlit components**: We use unittest.mock to replace Streamlit UI elements with mock objects that return predefined values
2. **Function patching**: We patch the calculation functions to verify they're called with the correct parameters
3. **Helper classes and utilities**: We provide common utilities to simplify test writing and maintenance

## Current Test Coverage

The current tests verify:

- The integration between UI components and calculation functions for:
  - Binary outcomes (both analytical and simulation methods)
  - Continuous outcomes (both analytical and simulation methods)
  - Survival outcomes (both analytical and simulation methods)
- The correct handling of different calculation types:
  - Sample size calculation
  - Power calculation
  - Minimum detectable effect calculation
- The proper integration of non-inferiority testing parameters
- The correct implementation of simulation toggle and parameters

## Future Test Development

Areas to expand upon:

1. **End-to-end testing**: Implement complete workflow tests from UI input to result rendering
2. **Visual testing**: Validate the correct rendering of UI components and results
3. **Browser automation**: Use tools like Selenium or Playwright for browser-based testing
4. **User workflow testing**: Simulate common user workflows and interactions
5. **Error handling**: Test UI behavior when invalid inputs are provided
6. **Responsiveness testing**: Verify UI behavior across different device sizes

## Best Practices

When adding new UI tests:

1. Use the provided `StreamlitTestCase` base class to leverage common testing utilities
2. Mock only the Streamlit components necessary for the test
3. Verify critical parameters in function calls, not the entire parameter set
4. Organize tests by feature or UI component
5. Include both positive and negative test cases
6. Add comments explaining the purpose of complex test scenarios
