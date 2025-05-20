"""
Utility functions and classes for UI testing.

This module provides common utilities for testing Streamlit applications,
including mocking Streamlit components and session state management.
"""
import unittest
from unittest.mock import patch, MagicMock
import os
import sys
import streamlit as st
from contextlib import contextmanager

# Make sure the project root is in the path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

class StreamlitTestCase(unittest.TestCase):
    """Base test case for Streamlit UI testing."""
    
    def setUp(self):
        """Set up test environment."""
        # Reset Streamlit session state
        st.session_state = {}
    
    def tearDown(self):
        """Clean up after tests."""
        # Clean up any mocks or session state
        st.session_state = {}
    
    @contextmanager
    def mock_streamlit_components(self, return_values=None):
        """
        Context manager that mocks common Streamlit components.
        
        Args:
            return_values (dict, optional): Dictionary mapping component names to return values.
                Example: {'radio': 'Simulation', 'slider': [0.3, 0.5, 0.8]}
        
        Returns:
            dict: Dictionary containing the mock objects for each Streamlit component.
        """
        return_values = return_values or {}
        
        # Define which components to mock and their default return values
        components = {
            'radio': return_values.get('radio', 'Analytical'),
            'slider': return_values.get('slider', 0.5),
            'number_input': return_values.get('number_input', 100),
            'selectbox': return_values.get('selectbox', 0),
            'checkbox': return_values.get('checkbox', False),
            'button': return_values.get('button', False),
            'expander': MagicMock(),
            'columns': MagicMock(),
            'write': MagicMock(),
            'markdown': MagicMock(),
        }
        
        # Create patches for each component
        patches = {}
        mocks = {}
        
        for component, return_value in components.items():
            patches[component] = patch(f'streamlit.{component}')
            mocks[component] = patches[component].start()
            
            # Set return values
            if isinstance(return_value, list) and component not in ['columns']:
                mocks[component].side_effect = return_value
            else:
                mocks[component].return_value = return_value
        
        # Special handling for context manager components
        mocks['expander'].__enter__.return_value = MagicMock()
        mocks['columns'].return_value = [MagicMock() for _ in range(2)]
        
        try:
            yield mocks
        finally:
            # Stop all patches
            for p in patches.values():
                p.stop()
    
    def simulate_render_component(self, component_func, calc_type, hypothesis_type, mock_config=None):
        """
        Helper method to simulate rendering a component with mocked Streamlit elements.
        
        Args:
            component_func: The component render function to test
            calc_type (str): Calculation type ('Sample Size', 'Power', etc.)
            hypothesis_type (str): Hypothesis type ('Superiority', 'Non-Inferiority')
            mock_config (dict, optional): Configuration for mock Streamlit components
        
        Returns:
            dict: The parameters returned by the component render function
        """
        with self.mock_streamlit_components(mock_config) as mocks:
            params = component_func(calc_type, hypothesis_type)
            return params, mocks
    
    def get_calculation_params(self, component_key, calc_type="Sample Size", hypothesis_type="Superiority", use_simulation=False):
        """
        Generate standardized parameter set for a given component.
        
        Args:
            component_key (tuple): Component identifier (design, outcome)
            calc_type (str): Calculation type
            hypothesis_type (str): Hypothesis type
            use_simulation (bool): Whether to use simulation methods
        
        Returns:
            dict: Dictionary of parameters
        """
        design, outcome = component_key
        
        # Base parameters for all components
        params = {
            "calculation_type": calc_type,
            "hypothesis_type": hypothesis_type,
            "use_simulation": use_simulation,
            "alpha": 0.05
        }
        
        # Add outcome-specific parameters
        if outcome == "Binary Outcome":
            params.update({
                "p1": 0.3,
                "p2": 0.5 if hypothesis_type == "Superiority" else None,
                "test_type": "Exact Test" if use_simulation else "Normal Approximation"
            })
            
            if calc_type == "Sample Size":
                params.update({"power": 0.8, "allocation_ratio": 1.0})
            elif calc_type == "Power":
                params.update({"n1": 100, "n2": 100})
            
        elif outcome == "Continuous Outcome":
            params.update({
                "mean1": 10,
                "mean2": 15 if hypothesis_type == "Superiority" else None,
                "sd1": 5,
                "sd2": 5
            })
            
            if calc_type == "Sample Size":
                params.update({"power": 0.8, "allocation_ratio": 1.0})
            elif calc_type == "Power":
                params.update({"n1": 100, "n2": 100})
                
        elif outcome == "Survival Outcome":
            params.update({
                "median1": 10,
                "median2": 15 if hypothesis_type == "Superiority" else None,
                "enrollment_period": 12,
                "follow_up_period": 24,
                "dropout_rate": 0.1
            })
            
            if calc_type == "Sample Size":
                params.update({"power": 0.8, "allocation_ratio": 1.0})
            elif calc_type == "Power":
                params.update({"n1": 100, "n2": 100})
        
        # Add simulation-specific parameters if applicable
        if use_simulation:
            params.update({"nsim": 1000})
            
            if calc_type == "Sample Size":
                params.update({
                    "min_n": 10,
                    "max_n": 500,
                    "step_n": 10
                })
            elif calc_type == "Minimum Detectable Effect":
                params.update({"precision": 0.01})
        
        # Add non-inferiority specific parameters if applicable
        if hypothesis_type == "Non-Inferiority":
            if outcome == "Binary Outcome":
                params.update({
                    "non_inferiority_margin": 0.1,
                    "assumed_difference": 0.0,
                    "direction": "lower"
                })
            elif outcome == "Continuous Outcome":
                params.update({
                    "non_inferiority_margin": 2.0,
                    "assumed_difference": 0.0,
                    "direction": "lower"
                })
            elif outcome == "Survival Outcome":
                params.update({
                    "non_inferiority_margin": 1.3,
                    "assumed_hazard_ratio": 1.0
                })
        
        return params
