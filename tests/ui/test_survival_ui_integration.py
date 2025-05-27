"""
UI Integration tests for survival outcomes in the DesignPower application.

These tests verify that the survival outcome UI components correctly interact with 
the calculation functions and that the user input is properly processed.
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

@patch('app.components.parallel_rct.st') # Target st where parallel_rct UI is built
class TestSurvivalUIIntegration(unittest.TestCase):
    """Test class for survival outcome UI integration tests."""
    
    def setUp(self):
        """Set up test environment."""
        st.session_state = {}
    
    def tearDown(self):
        """Clean up after tests."""
        st.session_state = {}

    @patch('app.components.parallel_rct.analytical_survival.power_survival')
    def test_parallel_survival_analytical_power(self, mock_core_power_func, mock_st_component):
        """Test analytical power calculation for survival superiority via main app flow."""
        # Configure mocks for st components used in parallel_rct.render_parallel_survival
        # Order: hr, median1, (n1, n2 for Power calc), alpha, allocation_ratio, accrual, followup, dropout1, dropout2, method_selectbox
        mock_st_component.number_input.side_effect = [0.7, 10.0, 100, 100, 12.0, 24.0] # hr, median1, n1, n2, accrual, followup
        # Sliders: alpha_slider_survival, dropout_slider_survival. Allocation is not a slider here.
        mock_st_component.slider.side_effect = [0.05, 0.1] # alpha (0.05), dropout (0.1)
        mock_st_component.selectbox.side_effect = ["Analytical"] # method_selectbox
        mock_st_component.expander.return_value.__enter__.return_value = MagicMock()
        mock_st_component.columns.side_effect = lambda x: [MagicMock() for _ in range(x if isinstance(x, int) else len(x))]
        mock_st_component.radio.return_value = "Two-sided" # sides_radio_survival_sup

        # Call the main app's render function for survival
        params = designpower_app.render_parallel_survival("Power", "Superiority")
        
        # Call the main app's calculate function for survival
        designpower_app.calculate_parallel_survival(params)
            
        # Verify st.error was not called
        mock_st_component.error.assert_not_called()

        # Verify the core analytical function was called with expected parameters
        mock_core_power_func.assert_called_once()
        _, kwargs = mock_core_power_func.call_args
        
        self.assertEqual(kwargs.get('n1'), 100)
        self.assertEqual(kwargs.get('n2'), 100)
        self.assertEqual(kwargs.get('median1'), 10.0)
        # median2 is derived, check hr if that's what the UI collects directly
        # For power, the core function expects median1 and median2. The UI collects hr and median1.
        # The calculate_parallel_survival function derives median2 = median1 / hr
        self.assertAlmostEqual(kwargs.get('median2'), 10.0 / 0.7, places=5) 
        self.assertEqual(kwargs.get('enrollment_period'), 12.0)
        self.assertEqual(kwargs.get('follow_up_period'), 24.0)
        self.assertEqual(kwargs.get('dropout_rate'), 0.1)
        self.assertEqual(kwargs.get('alpha'), 0.05)

    @patch('app.components.parallel_rct.simulation_survival.power_survival_sim')
    def test_parallel_survival_simulation_power(self, mock_core_sim_func, mock_st_component):
        """Test simulation power calculation for survival superiority via main app flow."""
        # Number inputs: hr, median1, n1, n2, accrual, followup, nsim, seed
        mock_st_component.number_input.side_effect = [0.6, 12.0, 150, 150, 10.0, 20.0, 1000, 42] # Added seed
        # Sliders: alpha_slider_survival, dropout_slider_survival
        mock_st_component.slider.side_effect = [0.05, 0.05] # alpha (0.05), dropout (0.05)
        mock_st_component.selectbox.side_effect = ["Simulation"]
        mock_st_component.expander.return_value.__enter__.return_value = MagicMock()
        mock_st_component.columns.side_effect = lambda x: [MagicMock() for _ in range(x if isinstance(x, int) else len(x))]
        mock_st_component.radio.return_value = "Two-sided" # sides_radio_survival_sup
            
        params = designpower_app.render_parallel_survival("Power", "Superiority")
        designpower_app.calculate_parallel_survival(params)
            
        # Verify st.error was not called
        mock_st_component.error.assert_not_called()

        # Verify the core simulation function was called with expected parameters
        mock_core_sim_func.assert_called_once()
        _, kwargs = mock_core_sim_func.call_args
                
        self.assertEqual(kwargs.get('n1'), 150)
        self.assertEqual(kwargs.get('n2'), 150)
        self.assertEqual(kwargs.get('median1'), 12.0)
        self.assertAlmostEqual(kwargs.get('median2'), 12.0 / 0.6, places=5)
        self.assertEqual(kwargs.get('enrollment_period'), 10.0)
        self.assertEqual(kwargs.get('follow_up_period'), 20.0)
        self.assertEqual(kwargs.get('dropout_rate'), 0.05)
        self.assertEqual(kwargs.get('alpha'), 0.05)
        self.assertEqual(kwargs.get('nsim'), 1000)

    @patch('app.components.parallel_rct.analytical_survival.sample_size_survival_non_inferiority') 
    def test_survival_analytical_non_inferiority_sample_size(self, mock_core_ss_func, mock_st_component):
        """Test analytical sample size calculation for survival non-inferiority via main app flow."""
        # Number inputs: median1_ni, nim_hr_ni, assumed_hr_ni, accrual, followup
        mock_st_component.number_input.side_effect = [
            15.0,   # median1_input_survival_ni
            1.3,    # nim_hr_input_survival_ni
            1.0,    # assumed_hr_input_survival_ni
            10.0,   # accrual_input_survival
            18.0    # followup_input_survival
        ]
        # Sliders: alpha, power, allocation_ratio, dropout
        mock_st_component.slider.side_effect = [
            0.05,    # alpha_slider_survival (0.05)
            0.90,   # power_slider_survival (0.9)
            1.0,    # allocation_slider_survival_ss (1.0)
            0.1    # dropout_slider_survival (0.1)
        ]
        mock_st_component.selectbox.side_effect = ["Analytical"] # method_survival_selectbox
        mock_st_component.expander.return_value.__enter__.return_value = MagicMock()
        mock_st_component.columns.side_effect = lambda x: [MagicMock() for _ in range(x if isinstance(x, int) else len(x))]
            
        params = designpower_app.render_parallel_survival("Sample Size", "Non-Inferiority")
        designpower_app.calculate_parallel_survival(params)
            
        mock_st_component.error.assert_not_called()
        mock_core_ss_func.assert_called_once()
        _, kwargs = mock_core_ss_func.call_args

        self.assertEqual(params["hypothesis_type"], "Non-Inferiority") 
        self.assertEqual(params["non_inferiority_margin_hr"], 1.3)
        self.assertEqual(params["assumed_true_hr"], 1.0)
    
        self.assertEqual(kwargs.get('median1'), 15.0)
        self.assertEqual(kwargs.get('non_inferiority_margin'), 1.3)
        self.assertEqual(kwargs.get('assumed_hazard_ratio'), 1.0)
        self.assertEqual(kwargs.get('enrollment_period'), 10.0)
        self.assertEqual(kwargs.get('follow_up_period'), 18.0)
        self.assertEqual(kwargs.get('dropout_rate'), 0.1)
        self.assertEqual(kwargs.get('power'), 0.9)
        self.assertEqual(kwargs.get('alpha'), 0.05)

    @patch('app.components.parallel_rct.simulation_survival.power_survival_non_inferiority_sim')
    def test_parallel_survival_simulation_non_inferiority_power_with_seed(self, mock_core_sim_ni_power_func, mock_st_component):
        """Test simulation NI power calculation for survival with seed."""
        # For render_parallel_survival (Non-Inferiority, Power):
        # Basic Params:
        # col1: median1_input_survival_ni, nim_hr_input_survival_ni, assumed_hr_input_survival_ni
        # col2: alpha_slider_survival, n1_input_survival_power, n2_input_survival_power
        # Advanced Params:
        # col_adv1: accrual_input_survival
        # col_adv2: followup_input_survival
        # dropout_slider_survival
        # method_survival_selectbox
        # nsim_survival_input, seed_survival_input (if method is Simulation)

        mock_st_component.number_input.side_effect = [
            12.0,   # median1_input_survival_ni
            1.3,    # nim_hr_input_survival_ni
            1.0,    # assumed_hr_input_survival_ni
            100,    # n1_input_survival_power
            100,    # n2_input_survival_power
            12.0,   # accrual_input_survival
            24.0,   # followup_input_survival
            500,    # nsim_survival_input
            123     # seed_survival_input
        ]
        mock_st_component.slider.side_effect = [
            0.05,   # alpha_slider_survival (0.05)
            0.1     # dropout_slider_survival (0.1)
        ]
        mock_st_component.selectbox.side_effect = ["Simulation"]
        mock_st_component.expander.return_value.__enter__.return_value = MagicMock()
        mock_st_component.columns.side_effect = lambda x: [MagicMock() for _ in range(x if isinstance(x, int) else len(x))]

        params = designpower_app.render_parallel_survival("Power", "Non-Inferiority")
        designpower_app.calculate_parallel_survival(params)

        mock_st_component.error.assert_not_called()
        mock_core_sim_ni_power_func.assert_called_once()
        _, kwargs = mock_core_sim_ni_power_func.call_args

        self.assertEqual(kwargs.get('n1'), 100)
        self.assertEqual(kwargs.get('n2'), 100)
        self.assertEqual(kwargs.get('median1'), 12.0)
        self.assertEqual(kwargs.get('non_inferiority_margin'), 1.3)
        self.assertEqual(kwargs.get('assumed_hazard_ratio'), 1.0)
        self.assertEqual(kwargs.get('enrollment_period'), 12.0)
        self.assertEqual(kwargs.get('follow_up_period'), 24.0)
        self.assertEqual(kwargs.get('dropout_rate'), 0.1)
        self.assertEqual(kwargs.get('alpha'), 0.05)
        self.assertEqual(kwargs.get('nsim'), 500)
        self.assertEqual(kwargs.get('seed'), 123)

    @patch('app.components.parallel_rct.simulation_survival.sample_size_survival_non_inferiority_sim')
    def test_parallel_survival_simulation_non_inferiority_sample_size_with_seed(self, mock_core_sim_ni_ss_func, mock_st_component):
        """Test simulation NI sample size calculation for survival with seed."""
        # For render_parallel_survival (Non-Inferiority, Sample Size):
        # Basic Params:
        # col1: median1_input_survival_ni, nim_hr_input_survival_ni, assumed_hr_input_survival_ni
        # col2: alpha_slider_survival, power_slider_survival, allocation_slider_survival_ss
        # Advanced Params:
        # col_adv1: accrual_input_survival
        # col_adv2: followup_input_survival
        # dropout_slider_survival
        # method_survival_selectbox
        # nsim_survival_input, seed_survival_input
        # min_n_sim_survival, max_n_sim_survival, step_n_sim_survival

        mock_st_component.number_input.side_effect = [
            15.0,   # median1_input_survival_ni
            1.25,   # nim_hr_input_survival_ni
            0.9,    # assumed_hr_input_survival_ni
            10.0,   # accrual_input_survival
            18.0,   # followup_input_survival
            600,    # nsim_survival_input
            456,    # seed_survival_input
            10,     # min_n_sim_survival
            500,    # max_n_sim_survival
            5       # step_n_sim_survival
        ]
        mock_st_component.slider.side_effect = [
            0.05,   # alpha_slider_survival (0.05)
            0.90,   # power_slider_survival (0.9)
            1.0,    # allocation_slider_survival_ss (1.0)
            0.05     # dropout_slider_survival (0.05)
        ]
        mock_st_component.selectbox.side_effect = ["Simulation"]
        mock_st_component.expander.return_value.__enter__.return_value = MagicMock()
        mock_st_component.columns.side_effect = lambda x: [MagicMock() for _ in range(x if isinstance(x, int) else len(x))]

        params = designpower_app.render_parallel_survival("Sample Size", "Non-Inferiority")
        designpower_app.calculate_parallel_survival(params)

        mock_st_component.error.assert_not_called()
        mock_core_sim_ni_ss_func.assert_called_once()
        _, kwargs = mock_core_sim_ni_ss_func.call_args

        self.assertEqual(kwargs.get('median1'), 15.0)
        self.assertEqual(kwargs.get('non_inferiority_margin'), 1.25)
        self.assertEqual(kwargs.get('assumed_hazard_ratio'), 0.9)
        self.assertEqual(kwargs.get('enrollment_period'), 10.0)
        self.assertEqual(kwargs.get('follow_up_period'), 18.0)
        self.assertEqual(kwargs.get('dropout_rate'), 0.05)
        self.assertEqual(kwargs.get('power'), 0.9)
        self.assertEqual(kwargs.get('alpha'), 0.05)
        self.assertEqual(kwargs.get('allocation_ratio'), 1.0)
        self.assertEqual(kwargs.get('nsim'), 600)
        self.assertEqual(kwargs.get('seed'), 456)
        self.assertEqual(kwargs.get('min_n'), 10)
        self.assertEqual(kwargs.get('max_n'), 500)
        self.assertEqual(kwargs.get('step'), 5)

    @patch('app.components.parallel_rct.analytical_survival.min_detectable_effect_survival')
    def test_parallel_survival_analytical_mde(self, mock_core_mde_func, mock_st_component):
        """Test analytical MDE calculation for survival superiority via main app flow."""
        # Configure the mock for the core MDE function to return a sample result
        mock_core_mde_func.return_value = {'hr': 0.75, 'events': 50} # Example return

        # Number inputs: hr_sup, median1_sup, accrual, followup, n1_mde, n2_mde
        mock_st_component.number_input.side_effect = [
            0.7,   # hr_input_survival_sup
            12.0,  # median1_input_survival_sup
            12.0,  # accrual_input_survival
            18.0,  # followup_input_survival
            100,   # n1_input_survival_mde
            100    # n2_input_survival_mde
        ]
        # Sliders: alpha, power, dropout
        mock_st_component.slider.side_effect = [0.05, 0.80, 0.1] # alpha (0.05), power (0.8), dropout (0.1)
        mock_st_component.selectbox.side_effect = ["Analytical"]
        mock_st_component.expander.return_value.__enter__.return_value = MagicMock()
        mock_st_component.columns.side_effect = lambda x: [MagicMock() for _ in range(x if isinstance(x, int) else len(x))]
        mock_st_component.radio.return_value = "Two-sided" # sides_radio_survival_sup

        params = designpower_app.render_parallel_survival("Minimum Detectable Effect", "Superiority")
        designpower_app.calculate_parallel_survival(params)

        mock_st_component.error.assert_not_called()
        mock_core_mde_func.assert_called_once()
        _, kwargs = mock_core_mde_func.call_args

        self.assertEqual(kwargs.get('n1'), 100)
        self.assertEqual(kwargs.get('n2'), 100)
        self.assertEqual(kwargs.get('median1'), 12.0)
        self.assertEqual(kwargs.get('enrollment_period'), 12.0)
        self.assertEqual(kwargs.get('follow_up_period'), 18.0)
        self.assertEqual(kwargs.get('dropout_rate'), 0.1)
        self.assertEqual(kwargs.get('power'), 0.8)
        self.assertEqual(kwargs.get('alpha'), 0.05)
        self.assertEqual(kwargs.get('sides'), 2)

if __name__ == '__main__':
    unittest.main()
