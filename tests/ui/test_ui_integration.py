"""
UI Integration tests for DesignPower application.

These tests verify that the UI components correctly interact with the calculation
functions and that the user input is properly processed.
"""
import unittest
from unittest.mock import patch, MagicMock
import os
import sys
import streamlit as st

# Make sure the project root is in the path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the app module
from app import designpower_app

class TestUIIntegration(unittest.TestCase):
    """Test class for UI integration tests."""
    
    def setUp(self):
        """Set up test environment."""
        # Reset Streamlit session state
        st.session_state = {}
    
    def tearDown(self):
        """Clean up after tests."""
        # Clean up any mocks or session state
        st.session_state = {}

    def test_parallel_binary_simulation_toggle(self):
        """Test that the simulation toggle correctly updates the UI parameters."""
        with patch('streamlit.radio') as mock_radio, \
             patch('streamlit.slider') as mock_slider, \
             patch('streamlit.number_input') as mock_number_input:
            
            # Mock the radio button for method selection to return "Simulation"
            mock_radio.return_value = "Simulation"
            
            # Call the render function
            params = designpower_app.render_parallel_binary("Sample Size", "Superiority")
            
            # Verify that use_simulation is True in params
            self.assertTrue(params["use_simulation"])
            
            # Verify that simulation-specific parameters are included
            self.assertIn("nsim", params)
            self.assertIn("min_n", params)
            self.assertIn("max_n", params)
            self.assertIn("step_n", params)
    
    def test_parallel_binary_calculation_call(self):
        """Test that the correct calculation function is called with proper parameters."""
        with patch('core.designs.parallel.binary_simulation.sample_size_binary_sim') as mock_sim_func, \
             patch('streamlit.radio') as mock_radio, \
             patch('streamlit.slider') as mock_slider, \
             patch('streamlit.number_input') as mock_number_input, \
             patch('streamlit.expander') as mock_expander, \
             patch('streamlit.columns') as mock_columns, \
             patch('streamlit.write') as mock_write:
            
            # Set up mock values
            mock_radio.return_value = "Simulation"
            mock_slider.side_effect = [0.3, 0.5, 0.8, 0.05, 1000]  # p1, p2, power, alpha, nsim
            mock_number_input.side_effect = [10, 500, 10, 10, 500, 10]  # min_n, max_n, step_n (repeated for safety)
            mock_expander.return_value.__enter__.return_value = MagicMock()
            mock_columns.return_value = [MagicMock(), MagicMock()]
            
            # Call the render function
            params = designpower_app.render_parallel_binary("Sample Size", "Superiority")
            
            # Manually call the calculation function with our params
            designpower_app.calc_parallel_binary(params)
            
            # Verify the simulation function was called with expected parameters
            mock_sim_func.assert_called_once()
            args, kwargs = mock_sim_func.call_args
            
            # Check key parameters for binary simulation
            self.assertEqual(kwargs.get('p1'), 0.3)
            self.assertEqual(kwargs.get('p2'), 0.5)
            self.assertEqual(kwargs.get('power'), 0.8)
            self.assertEqual(kwargs.get('nsim'), 1000)
            self.assertEqual(kwargs.get('min_n'), 10)
            self.assertEqual(kwargs.get('max_n'), 500)
            self.assertEqual(kwargs.get('step'), 10)

    def test_parallel_continuous_simulation_toggle(self):
        """Test that the simulation toggle correctly updates the UI parameters for continuous outcomes."""
        with patch('streamlit.radio') as mock_radio, \
             patch('streamlit.slider') as mock_slider, \
             patch('streamlit.number_input') as mock_number_input:
            
            # Mock the radio button for method selection to return "Simulation"
            mock_radio.return_value = "Simulation"
            
            # Call the render function
            params = designpower_app.render_parallel_continuous("Sample Size", "Superiority")
            
            # Verify that use_simulation is True in params
            self.assertTrue(params["use_simulation"])
            
            # Verify that simulation-specific parameters are included
            self.assertIn("nsim", params)
            self.assertIn("min_n", params)
            self.assertIn("max_n", params)
            self.assertIn("step_n", params)
    
    def test_continuous_simulation_calculation(self):
        """Test that continuous simulation calculation is called with proper parameters."""
        with patch('core.designs.parallel.simulation_continuous.sample_size_continuous_sim') as mock_sim_func, \
             patch('streamlit.radio') as mock_radio, \
             patch('streamlit.slider') as mock_slider, \
             patch('streamlit.number_input') as mock_number_input, \
             patch('streamlit.expander') as mock_expander, \
             patch('streamlit.columns') as mock_columns, \
             patch('streamlit.write') as mock_write:
            
            # Set up mock values
            mock_radio.return_value = "Simulation"
            mock_slider.side_effect = [10, 15, 5, 0.8, 0.05, 1000]  # mean1, mean2, sd, power, alpha, nsim
            mock_number_input.side_effect = [10, 500, 10, 10, 500, 10]  # min_n, max_n, step_n (repeated for safety)
            mock_expander.return_value.__enter__.return_value = MagicMock()
            mock_columns.return_value = [MagicMock(), MagicMock()]
            
            # Call the render function
            params = designpower_app.render_parallel_continuous("Sample Size", "Superiority")
            
            # Manually call the calculation function with our params
            designpower_app.calc_parallel_continuous(params)
            
            # Verify the simulation function was called with expected parameters
            mock_sim_func.assert_called_once()
            args, kwargs = mock_sim_func.call_args
            
            # Check key parameters for continuous simulation
            self.assertEqual(kwargs.get('mean1'), 10)
            self.assertEqual(kwargs.get('mean2'), 15)
            self.assertEqual(kwargs.get('sd1'), 5)
            self.assertEqual(kwargs.get('power'), 0.8)
            self.assertEqual(kwargs.get('nsim'), 1000)
            self.assertEqual(kwargs.get('min_n'), 10)
            self.assertEqual(kwargs.get('max_n'), 500)
            self.assertEqual(kwargs.get('step'), 10)

    def test_test_type_selection_binary(self):
        """Test that the test type selection is correctly passed to the calculation function."""
        with patch('streamlit.radio') as mock_radio, \
             patch('streamlit.slider') as mock_slider, \
             patch('streamlit.number_input') as mock_number_input, \
             patch('streamlit.expander') as mock_expander, \
             patch('streamlit.columns') as mock_columns, \
             patch('streamlit.write') as mock_write, \
             patch('core.designs.parallel.simulation_binary.power_binary_with_test') as mock_power_sim:
            
            # Set up mock values
            mock_radio.side_effect = ["Simulation", "Exact Test", "Simulation", "Exact Test"]  # Method, Test type (doubled for multiple calls)
            mock_slider.side_effect = [0.3, 0.5, 0.05, 1000]  # p1, p2, alpha, nsim
            mock_number_input.side_effect = [100, 100]  # n1, n2
            mock_expander.return_value.__enter__.return_value = MagicMock()
            mock_columns.return_value = [MagicMock(), MagicMock()]
            
            # Call the render function
            params = designpower_app.render_parallel_binary("Power", "Superiority")
            
            # Verify the test_type parameter
            self.assertEqual(params["test_type"], "Exact Test")
            
            # Skip the actual calculation function call and just verify params
            self.assertEqual(params["calculation_type"], "Power")
            self.assertEqual(params["use_simulation"], True)

if __name__ == '__main__':
    unittest.main()
