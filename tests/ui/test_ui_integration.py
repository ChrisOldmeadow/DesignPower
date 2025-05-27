"""
UI Integration tests for DesignPower application.

These tests verify that the UI components correctly interact with the calculation
functions and that the user input is properly processed.
"""
import unittest
from unittest.mock import patch, MagicMock, ANY
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
    
    @patch('core.designs.parallel.simulation_binary.sample_size_binary_sim')
    @patch('app.components.parallel_rct.st')
    @patch('app.designpower_app.st') 
    def test_parallel_binary_calculation_call(self, mock_st_designpower_app, mock_st_component, mock_sim_func):
        """Test that the correct calculation function is called with proper parameters."""
        # Setup mock return values for UI elements on the component's 'st' mock
        mock_st_component.radio.return_value = "Simulation"  # For method selection
        mock_st_component.slider.side_effect = [0.3, 0.5, 0.05, 0.8, 1.0]  # p1, p2, alpha, power, allocation_ratio
        mock_st_component.number_input.side_effect = [1000, 42] # nsim, seed
        mock_st_component.selectbox.return_value = "Normal Approximation" # For test_type
        mock_st_component.checkbox.return_value = False # For continuity correction
        mock_st_component.expander.return_value.__enter__.return_value = MagicMock()
        mock_st_component.columns.return_value = [MagicMock(), MagicMock()]

        # Mock any st calls made directly by designpower_app if necessary (e.g., session_state)
        # For this test, render_parallel_binary is called directly, which uses app.components.parallel_rct.st

        # Debug: Verify the patch is correctly applied to the simulation function
        from app.components import parallel_rct
        self.assertIs(parallel_rct.simulation_binary.sample_size_binary_sim, mock_sim_func,
                      "Patch on simulation_binary.sample_size_binary_sim failed.")

        # Call the render function using designpower_app (which should use the patched st component internally)
        # Note: designpower_app.render_parallel_binary itself is not mocked, it will run.
        # It's expected to call app.components.parallel_rct.render_parallel_binary which is also not mocked.
        # The functions from app.components.parallel_rct will use the mock_st_component.
        params = designpower_app.render_parallel_binary("Sample Size", "Superiority")

        # Manually call the calculation function with our params
        # designpower_app.calculate_parallel_binary will call app.components.parallel_rct.calculate_parallel_binary
        # which in turn calls the (patched) simulation_binary.sample_size_binary_sim
        designpower_app.calculate_parallel_binary(params)

        # Verify the simulation function was called with expected parameters
        mock_sim_func.assert_called_once()
        args, kwargs = mock_sim_func.call_args
        
        # Check key parameters for binary simulation
        self.assertEqual(kwargs.get('p1'), 0.3)
        self.assertEqual(kwargs.get('p2'), 0.5)
        self.assertEqual(kwargs.get('alpha'), 0.05)
        self.assertEqual(kwargs.get('power'), 0.8)
        self.assertEqual(kwargs.get('allocation_ratio'), 1.0)
        self.assertEqual(kwargs.get('nsim'), 1000)
        # self.assertEqual(kwargs.get('seed'), 42) # Seed is not passed to sample_size_binary_sim
        self.assertEqual(kwargs.get('test_type'), 'normal_approximation')

    @patch('core.designs.parallel.simulation_binary.sample_size_binary_sim')
    @patch('app.components.parallel_rct.st')
    @patch('app.designpower_app.st') 
    def test_parallel_binary_simulation_toggle(self, mock_st_designpower_app, mock_st_component, mock_sim_func):
        """Test that simulation method is correctly chosen when toggled."""
        # Method selection: Simulation
        mock_st_component.radio.side_effect = ["Simulation", "Normal Approximation"] # Method, Test Type
        mock_st_component.slider.side_effect = [0.3, 0.5, 0.05, 0.8, 1.0]  # p1, p2, alpha, power, allocation_ratio
        mock_st_component.number_input.side_effect = [1000, 42] # nsim, seed
        mock_st_component.checkbox.return_value = False # Continuity correction
        mock_st_component.expander.return_value.__enter__.return_value = MagicMock()
        mock_st_component.columns.return_value = [MagicMock(), MagicMock()]

        params = designpower_app.render_parallel_binary("Sample Size", "Superiority")
        self.assertEqual(params.get("method"), "simulation")
        designpower_app.calculate_parallel_binary(params)
        mock_sim_func.assert_called_once()

    @patch('core.designs.parallel.analytical_binary.sample_size_binary') # Patch analytical for this test
    @patch('app.components.parallel_rct.st')
    @patch('app.designpower_app.st') 
    def test_parallel_binary_analytical_toggle(self, mock_st_designpower_app, mock_st_component, mock_analytical_func):
        """Test that analytical method is correctly chosen when toggled."""
        # Method selection: Analytical
        mock_st_component.radio.side_effect = ["Analytical", "Normal Approximation"] # Method, Test Type
        mock_st_component.slider.side_effect = [0.3, 0.5, 0.05, 0.8, 1.0]  # p1, p2, alpha, power, allocation_ratio
        # No number_input calls expected for nsim/seed in analytical
        mock_st_component.checkbox.return_value = False # Continuity correction
        mock_st_component.expander.return_value.__enter__.return_value = MagicMock()
        mock_st_component.columns.return_value = [MagicMock(), MagicMock()]

        params = designpower_app.render_parallel_binary("Sample Size", "Superiority")
        self.assertEqual(params.get("method"), "analytical")
        designpower_app.calculate_parallel_binary(params)
        mock_analytical_func.assert_called_once()


    @patch('core.designs.parallel.simulation_binary.sample_size_binary_sim')
    @patch('app.components.parallel_rct.st')
    @patch('app.designpower_app.st') 
    def test_test_type_selection_binary(self, mock_st_designpower_app, mock_st_component, mock_sim_func):
        """Test that the test type selection is correctly passed to the calculation function."""
        
        test_cases = [
            ("Normal Approximation", "normal_approximation"),
            ("Fisher's Exact Test", "fishers_exact"),
            ("Likelihood Ratio Test", "likelihood_ratio")
        ]

        for ui_test_type, expected_sim_test_type in test_cases:
            with self.subTest(ui_test_type=ui_test_type):
                mock_sim_func.reset_mock() # Reset for each run
                # Configure mocks for each run - CRITICAL to set side_effects here for each subtest
                mock_st_component.radio.return_value = "Simulation" # Method is Simulation
                mock_st_component.selectbox.return_value = ui_test_type # Test Type selection
                # For calc_type="Power": p1, p2, alpha, allocation_ratio (No power slider)
                mock_st_component.slider.side_effect = [0.2, 0.4, 0.05, 1.0] 
                # For calc_type="Power": n1, n2 (from render_parallel_binary) then nsim, seed (from render_binary_advanced_options)
                mock_st_component.number_input.side_effect = [500, 123, 1000, 42] 
                # For binary, the checkbox is for continuity correction.
                mock_st_component.checkbox.return_value = True # Continuity correction

                mock_st_component.expander.return_value.__enter__.return_value = MagicMock()
                # Update st.columns mocking to be consistent and robust
                mock_st_component.columns.side_effect = lambda x: [MagicMock() for _ in range(x if isinstance(x, int) else len(x))]

                # The test calls render_parallel_binary with calc_type = "Power"
                params = designpower_app.render_parallel_binary("Power", "Superiority")
                self.assertEqual(params.get("method"), "simulation")
                self.assertEqual(params.get("test_type"), ui_test_type)
                
                designpower_app.calculate_parallel_binary(params)
                
                mock_sim_func.assert_called_once()
                _, kwargs = mock_sim_func.call_args
                self.assertEqual(kwargs.get('test_type'), expected_sim_test_type)


    @patch('core.designs.parallel.simulation_continuous.sample_size_continuous_sim') # Target the actual function
    @patch('app.components.parallel_rct.st') # Mock st used in the component
    @patch('app.designpower_app.st') # Mock st used in the main app
    def test_continuous_simulation_calculation(self, mock_st_designpower_app, mock_st_component, mock_sim_func):
        """Test that continuous simulation calculation is called with proper parameters."""
        
        # Configure the component's st mock
        mock_st_component.radio.return_value = "Simulation"  # Method selection
        mock_st_component.slider.side_effect = [0.05, 0.8, 1.0]  # alpha, power, allocation_ratio
        # mean1, sd (std_dev), mean2, nsim, seed, min_n, max_n, step_n
        mock_st_component.number_input.side_effect = [10, 5, 15, 1000, 42, 10, 500, 10] 
        mock_st_component.expander.return_value.__enter__.return_value = MagicMock()
        # Correctly mock st.columns to return a list of mocks based on the input integer
        # or the length of the input list (for column ratios)
        mock_st_component.columns.side_effect = lambda x: [MagicMock() for _ in range(x if isinstance(x, int) else len(x))]
        # Ensure checkboxes for unequal variances and repeated measures are False
        mock_st_component.checkbox.side_effect = [False, False] # Unequal variances, Repeated measures
        
        # Call the render function (which internally calls render_continuous_advanced_options)
        params = designpower_app.render_parallel_continuous("Sample Size", "Superiority")
        
        # Manually call the calculation function with our params
        designpower_app.calculate_parallel_continuous(params)
        
        # Verify the simulation function was called with expected parameters
        mock_sim_func.assert_called_once()
        args, kwargs = mock_sim_func.call_args
        
        # Assertions for continuous simulation
        self.assertEqual(kwargs.get('delta'), 5)  # mean2 (15) - mean1 (10)
        self.assertEqual(kwargs.get('std_dev'), 5)
        self.assertEqual(kwargs.get('power'), 0.8)
        self.assertEqual(kwargs.get('alpha'), 0.05)
        self.assertEqual(kwargs.get('allocation_ratio'), 1.0)
        self.assertEqual(kwargs.get('nsim'), 1000)
        self.assertEqual(kwargs.get('seed'), 42)
        self.assertEqual(kwargs.get('min_n'), 10)
        self.assertEqual(kwargs.get('max_n'), 500)
        self.assertEqual(kwargs.get('step'), 10) # Changed from 'step_n' to 'step'

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
    
    @patch('core.designs.parallel.simulation_binary.sample_size_binary_sim')
    @patch('app.components.parallel_rct.st')
    @patch('app.designpower_app.st') 
    def test_test_type_selection_binary(self, mock_st_designpower_app, mock_st_component, mock_sim_func):
        """Test that the test type selection is correctly passed to the calculation function."""
        
        test_cases = [
            ("Normal Approximation", "normal_approximation"),
            ("Fisher's Exact Test", "fishers_exact"),
            ("Likelihood Ratio Test", "likelihood_ratio")
        ]

        for ui_test_type, expected_sim_test_type in test_cases:
            with self.subTest(ui_test_type=ui_test_type):
                mock_sim_func.reset_mock() # Reset for each run
                # Configure mocks for each run - CRITICAL to set side_effects here for each subtest
                mock_st_component.radio.return_value = "Simulation" # Method is Simulation
                mock_st_component.selectbox.return_value = ui_test_type # Test Type selection
                # For calc_type="Power": p1, p2, alpha, allocation_ratio (No power slider)
                mock_st_component.slider.side_effect = [0.2, 0.4, 0.05, 1.0] 
                # For calc_type="Power": n1, n2 (from render_parallel_binary) then nsim, seed (from render_binary_advanced_options)
                mock_st_component.number_input.side_effect = [500, 123, 1000, 42] 
                # For binary, the checkbox is for continuity correction.
                mock_st_component.checkbox.return_value = True # Continuity correction

                mock_st_component.expander.return_value.__enter__.return_value = MagicMock()
                # Update st.columns mocking to be consistent and robust
                mock_st_component.columns.side_effect = lambda x: [MagicMock() for _ in range(x if isinstance(x, int) else len(x))]

                # The test calls render_parallel_binary with calc_type = "Power"
                params = designpower_app.render_parallel_binary("Power", "Superiority")
                self.assertEqual(params.get("method"), "simulation")
                self.assertEqual(params.get("test_type"), ui_test_type)
                
                designpower_app.calculate_parallel_binary(params)
                
                mock_sim_func.assert_called_once()
                _, kwargs = mock_sim_func.call_args
                self.assertEqual(kwargs.get('test_type'), expected_sim_test_type)


    @patch('core.designs.cluster_rct.simulation_continuous.sample_size_continuous_sim')
    @patch('app.components.cluster_rct.st', new_callable=MagicMock) # Mock st used by the component
    @patch('app.designpower_app.st')        # Mock st used in the main app (for session_state mainly)
    def test_cluster_rct_continuous_bayesian_sample_size_converged(self, mock_st_main_app, mock_st_component, mock_core_sim_func):
        """Test Cluster RCT, Continuous, Bayesian, Sample Size - Converged scenario."""
        # 1. Set up st.session_state via the main app's mocked st object
        mock_st_main_app.session_state = {
            "design_type": "Cluster RCT",
            "outcome_type": "Continuous Outcome",
            "calculation_type": "Sample Size",
            "hypothesis_type": "Superiority",
            "results": None
        }

        # 2. Configure mock_st_component return values
        mock_st_component.radio.side_effect = ["Simulation"] # Method selection
        mock_st_component.selectbox.side_effect = ["Bayesian (Stan)"] # Analysis model

        # Order of keyless st.number_input calls in render_cluster_continuous:
        # 1. cluster_size, 2. icc, 3. mean1, 4. mean2, 5. std_dev
        keyless_ni_values = [
            20,    # cluster_size
            0.05,  # icc
            10.0,  # mean1
            8.0,   # mean2
            2.0    # std_dev
        ]
        keyless_ni_call_count = 0

        def number_input_side_effect(*args, **kwargs):
            nonlocal keyless_ni_call_count
            key = kwargs.get('key')
            if key: # Inputs from advanced_options have keys
                if key == 'cluster_continuous_nsim': return 1000
                if key == 'cluster_continuous_seed': return 123
                if key == 'cluster_continuous_bayes_draws': return 600
                if key == 'cluster_continuous_bayes_warmup': return 400
                return MagicMock()
            else: # Keyless inputs from render_cluster_continuous
                if keyless_ni_call_count < len(keyless_ni_values):
                    value = keyless_ni_values[keyless_ni_call_count]
                    keyless_ni_call_count += 1
                    return value
                return MagicMock()
        mock_st_component.number_input.side_effect = number_input_side_effect

        # render_cluster_continuous for Sample Size calls st.slider for power
        mock_st_component.slider.return_value = 0.8  # Power (1-β)
        # render_cluster_continuous calls st.select_slider for alpha
        mock_st_component.select_slider.return_value = 0.05  # Alpha

        mock_st_component.expander.return_value.__enter__.return_value = MagicMock()
        mock_st_component.columns.return_value = [MagicMock(), MagicMock()]
        mock_st_component.checkbox.return_value = False

        # 3. Call render function
        render_func = designpower_app.COMPONENTS[("Cluster RCT", "Continuous Outcome")]["render"]
        params = render_func(
            mock_st_main_app.session_state["calculation_type"],
            mock_st_main_app.session_state["hypothesis_type"]
        )

        # 4. Assert params
        self.assertEqual(params.get("method"), "simulation")
        self.assertEqual(params.get("analysis_model"), "bayes")
        self.assertEqual(params.get("cluster_size"), 20)
        self.assertEqual(params.get("icc"), 0.05)
        self.assertEqual(params.get("mean1"), 10.0)
        self.assertEqual(params.get("mean2"), 8.0)
        self.assertEqual(params.get("std_dev"), 2.0)
        self.assertEqual(params.get("power"), 0.8)
        self.assertEqual(params.get("alpha"), 0.05)
        self.assertEqual(params.get("nsim"), 1000)
        self.assertEqual(params.get("seed"), 123)
        self.assertEqual(params.get("bayes_draws"), 600)
        self.assertEqual(params.get("bayes_warmup"), 400)

        # 5. Configure mock_core_sim_func.return_value
        expected_sim_results = {
            'n_clusters': 22,
            'power': 0.805,
            'details_bayes': {'converged': True, 'summary': 'Mocked Stan Summary', 'r_hat_max': 1.01},
            'design_method': 'Cluster RCT'
        }
        mock_core_sim_func.return_value = expected_sim_results

        # 6. Call calculate function
        calculate_func = designpower_app.COMPONENTS[("Cluster RCT", "Continuous Outcome")]["calculate"]
        results = calculate_func(params)

        # 7. Assert mock_core_sim_func was called correctly
        mock_core_sim_func.assert_called_once_with(
            mean1=10.0,
            mean2=8.0,
            std_dev=2.0,
            icc=0.05,
            cluster_size=20,
            power=0.8,
            alpha=0.05,
            nsim=1000,
            seed=123,
            analysis_model='bayes',
            use_satterthwaite=params.get("use_satterthwaite", False),
            use_bias_correction=params.get("use_bias_correction", False),
            bayes_draws=600,
            bayes_warmup=400,
            lmm_method=params.get("lmm_method", "auto"),
            lmm_reml=params.get("lmm_reml", True)
        )

        # 8. Assert results
        self.assertEqual(results, expected_sim_results)

    @patch('core.designs.cluster_rct.simulation_continuous.sample_size_continuous_sim')
    @patch('app.components.cluster_rct.st', new_callable=MagicMock) # Mock st used by the component
    @patch('app.designpower_app.st')        # Mock st used in the main app (for session_state mainly)
    def test_cluster_rct_continuous_bayesian_sample_size_fallback(self, mock_st_main_app, mock_st_component, mock_core_sim_func):
        """Test Cluster RCT, Continuous, Bayesian, Sample Size - Fallback/Non-Converged scenario."""
        # 1. Set up st.session_state via the main app's mocked st object
        mock_st_main_app.session_state = {
            "design_type": "Cluster RCT",
            "outcome_type": "Continuous Outcome",
            "calculation_type": "Sample Size",
            "hypothesis_type": "Superiority",
            "results": None
        }

        # 2. Configure mock_st_component return values (same as converged test)
        mock_st_component.radio.side_effect = ["Simulation"] # Method selection
        mock_st_component.selectbox.side_effect = ["Bayesian (Stan)"] # Analysis model

        keyless_ni_values = [20, 0.05, 10.0, 8.0, 2.0] # cluster_size, icc, mean1, mean2, std_dev
        keyless_ni_call_count = 0

        def number_input_side_effect(*args, **kwargs):
            nonlocal keyless_ni_call_count
            key = kwargs.get('key')
            if key:
                if key == 'cluster_continuous_nsim': return 1000
                if key == 'cluster_continuous_seed': return 123
                if key == 'cluster_continuous_bayes_draws': return 600
                if key == 'cluster_continuous_bayes_warmup': return 400
                return MagicMock()
            else:
                if keyless_ni_call_count < len(keyless_ni_values):
                    value = keyless_ni_values[keyless_ni_call_count]
                    keyless_ni_call_count += 1
                    return value
                return MagicMock()
        mock_st_component.number_input.side_effect = number_input_side_effect
        mock_st_component.slider.return_value = 0.8  # Power
        mock_st_component.select_slider.return_value = 0.05  # Alpha
        mock_st_component.expander.return_value.__enter__.return_value = MagicMock()
        mock_st_component.columns.return_value = [MagicMock(), MagicMock()]
        mock_st_component.checkbox.return_value = False

        # 3. Call render function
        render_func = designpower_app.COMPONENTS[("Cluster RCT", "Continuous Outcome")]["render"]
        params = render_func(
            mock_st_main_app.session_state["calculation_type"],
            mock_st_main_app.session_state["hypothesis_type"]
        )

        # 4. Assert params (same as converged test)
        self.assertEqual(params.get("method"), "simulation")
        self.assertEqual(params.get("analysis_model"), "bayes")
        self.assertEqual(params.get("cluster_size"), 20)
        self.assertEqual(params.get("icc"), 0.05)
        self.assertEqual(params.get("mean1"), 10.0)
        self.assertEqual(params.get("mean2"), 8.0)
        self.assertEqual(params.get("std_dev"), 2.0)
        self.assertEqual(params.get("power"), 0.8)
        self.assertEqual(params.get("alpha"), 0.05)
        self.assertEqual(params.get("nsim"), 1000)
        self.assertEqual(params.get("seed"), 123)
        self.assertEqual(params.get("bayes_draws"), 600)
        self.assertEqual(params.get("bayes_warmup"), 400)

        # 5. Configure mock_core_sim_func.return_value for non-convergence
        fallback_sim_results = {
            'n_clusters': None, # Or some indicator like 'N/A'
            'power': None,      # Or some indicator like 'N/A'
            'message': 'Bayesian simulation did not converge. Check Bayesian Model Details.',
            'details_bayes': {
                'converged': False,
                'summary': 'Model did not converge due to high R-hat values and low ESS.',
                'r_hat_max': 1.25,
                'ess_min': 50,
                'fallback_method_used': None # Or e.g., 'analytical_approximation'
            },
            'design_method': 'Cluster RCT'
        }
        mock_core_sim_func.return_value = fallback_sim_results

        # 6. Call calculate function
        calculate_func = designpower_app.COMPONENTS[("Cluster RCT", "Continuous Outcome")]["calculate"]
        results = calculate_func(params)

        # 7. Assert mock_core_sim_func was called correctly (same as converged test)
        mock_core_sim_func.assert_called_once_with(
            mean1=10.0,
            mean2=8.0,
            std_dev=2.0,
            icc=0.05,
            cluster_size=20,
            power=0.8,
            alpha=0.05,
            nsim=1000,
            seed=123,
            analysis_model='bayes',
            use_satterthwaite=params.get("use_satterthwaite", False),
            use_bias_correction=params.get("use_bias_correction", False),
            bayes_draws=600,
            bayes_warmup=400,
            lmm_method=params.get("lmm_method", "auto"),
            lmm_reml=params.get("lmm_reml", True)
        )

        # 8. Assert results dictionary reflects non-convergence
        self.assertEqual(results, fallback_sim_results)
        self.assertIsNone(results.get('n_clusters'))
        self.assertFalse(results['details_bayes']['converged'])
        self.assertIn('high r-hat values', results['details_bayes']['summary'].lower())

    @patch('core.designs.cluster_rct.simulation_continuous.power_continuous_sim')
    @patch('app.components.cluster_rct.st', new_callable=MagicMock) # Mock st used by the component
    @patch('app.designpower_app.st')        # Mock st used in the main app (for session_state mainly)
    def test_cluster_rct_continuous_bayesian_power_converged(
        self, mock_st_main_app, mock_st_component, mock_core_sim_func
    ):
        """Test Cluster RCT, Continuous, Bayesian, Power - Converged scenario."""
        # 1. Set up st.session_state
        mock_st_main_app.session_state = {
            "design_type": "Cluster RCT",
            "outcome_type": "Continuous Outcome",
            "calculation_type": "Power", # Changed to Power
            "hypothesis_type": "Superiority",
            "results": None
        }

        # 2. Configure mock_st_component return values
        mock_st_component.radio.side_effect = ["Simulation"] # Method selection
        mock_st_component.selectbox.side_effect = ["Bayesian (Stan)"] # Analysis model

        # Order for Power calc: cluster_size, icc, n_clusters, mean1, mean2, std_dev
        keyless_ni_values = [20, 0.05, 15, 10.0, 8.0, 2.0] 
        keyless_ni_call_count = 0

        def number_input_side_effect(*args, **kwargs):
            nonlocal keyless_ni_call_count
            key = kwargs.get('key')
            if key:
                if key == 'cluster_continuous_nsim': return 1000
                if key == 'cluster_continuous_seed': return 123
                if key == 'cluster_continuous_bayes_draws': return 600
                if key == 'cluster_continuous_bayes_warmup': return 400
                return MagicMock()
            else:
                if keyless_ni_call_count < len(keyless_ni_values):
                    value = keyless_ni_values[keyless_ni_call_count]
                    keyless_ni_call_count += 1
                    return value
                return MagicMock()
        mock_st_component.number_input.side_effect = number_input_side_effect
        # No power slider for Power calculation type
        mock_st_component.select_slider.return_value = 0.05  # Alpha
        mock_st_component.expander.return_value.__enter__.return_value = MagicMock()
        mock_st_component.columns.return_value = [MagicMock(), MagicMock()]
        mock_st_component.checkbox.return_value = False

        # 3. Call render function
        render_func = designpower_app.COMPONENTS[("Cluster RCT", "Continuous Outcome")]["render"]
        params = render_func(
            mock_st_main_app.session_state["calculation_type"],
            mock_st_main_app.session_state["hypothesis_type"]
        )

        # 4. Assert params
        self.assertEqual(params.get("method"), "simulation")
        self.assertEqual(params.get("analysis_model"), "bayes")
        self.assertEqual(params.get("cluster_size"), 20)
        self.assertEqual(params.get("icc"), 0.05)
        self.assertEqual(params.get("n_clusters"), 15) # n_clusters is input
        self.assertEqual(params.get("mean1"), 10.0)
        self.assertEqual(params.get("mean2"), 8.0)
        self.assertEqual(params.get("std_dev"), 2.0)
        self.assertEqual(params.get("alpha"), 0.05)
        self.assertEqual(params.get("nsim"), 1000)
        self.assertEqual(params.get("seed"), 123)
        self.assertEqual(params.get("bayes_draws"), 600)
        self.assertEqual(params.get("bayes_warmup"), 400)

        # 5. Configure mock_core_sim_func.return_value for convergence
        expected_sim_results = {
            'power': 0.825, # Calculated power
            'details_bayes': {'converged': True, 'summary': 'Mocked Stan Summary for Power', 'r_hat_max': 1.02},
            'design_method': 'Cluster RCT'
        }
        mock_core_sim_func.return_value = expected_sim_results

        # 6. Call calculate function
        calculate_func = designpower_app.COMPONENTS[("Cluster RCT", "Continuous Outcome")]["calculate"]
        results = calculate_func(params)

        # 7. Assert mock_core_sim_func was called correctly
        mock_core_sim_func.assert_called_once_with(
            mean1=10.0,
            mean2=8.0,
            std_dev=2.0,
            icc=0.05,
            cluster_size=20,
            n_clusters=15, # n_clusters is an argument
            alpha=0.05,
            nsim=1000,
            seed=123,
            analysis_model='bayes',
            use_satterthwaite=params.get("use_satterthwaite", False),
            use_bias_correction=params.get("use_bias_correction", False),
            bayes_draws=600,
            bayes_warmup=400,
            lmm_method=params.get("lmm_method", "auto"),
            lmm_reml=params.get("lmm_reml", True),
            progress_callback=ANY # Account for progress_callback
        )

        # 8. Assert results
        self.assertEqual(results, expected_sim_results)
        self.assertEqual(results.get('power'), 0.825)
        self.assertTrue(results['details_bayes']['converged'])

    @patch('core.designs.cluster_rct.simulation_continuous.power_continuous_sim')
    @patch('app.components.cluster_rct.st', new_callable=MagicMock)
    @patch('app.designpower_app.st')
    def test_cluster_rct_continuous_bayesian_power_fallback(
        self, mock_st_main_app, mock_st_component, mock_core_sim_func
    ):
        """Test Cluster RCT, Continuous, Bayesian, Power - Fallback scenario."""
        # 1. Set up st.session_state
        mock_st_main_app.session_state = {
            "design_type": "Cluster RCT",
            "outcome_type": "Continuous Outcome",
            "calculation_type": "Power",
            "hypothesis_type": "Superiority",
            "results": None
        }

        # 2. Configure mock_st_component return values (same as power_converged)
        mock_st_component.radio.side_effect = ["Simulation"]
        mock_st_component.selectbox.side_effect = ["Bayesian (Stan)"]
        keyless_ni_values = [20, 0.05, 15, 10.0, 8.0, 2.0] # cluster_size, icc, n_clusters, mean1, mean2, std_dev
        keyless_ni_call_count = 0
        def number_input_side_effect(*args, **kwargs):
            nonlocal keyless_ni_call_count
            key = kwargs.get('key')
            if key:
                if key == 'cluster_continuous_nsim': return 1000
                if key == 'cluster_continuous_seed': return 123
                if key == 'cluster_continuous_bayes_draws': return 600
                if key == 'cluster_continuous_bayes_warmup': return 400
                return MagicMock()
            else:
                if keyless_ni_call_count < len(keyless_ni_values):
                    value = keyless_ni_values[keyless_ni_call_count]
                    keyless_ni_call_count += 1
                    return value
                return MagicMock()
        mock_st_component.number_input.side_effect = number_input_side_effect
        mock_st_component.select_slider.return_value = 0.05  # Alpha
        mock_st_component.expander.return_value.__enter__.return_value = MagicMock()
        mock_st_component.columns.return_value = [MagicMock(), MagicMock()]
        mock_st_component.checkbox.return_value = False

        # 3. Call render function
        render_func = designpower_app.COMPONENTS[("Cluster RCT", "Continuous Outcome")]["render"]
        params = render_func(
            mock_st_main_app.session_state["calculation_type"],
            mock_st_main_app.session_state["hypothesis_type"]
        )

        # 4. Assert params (same as power_converged)
        self.assertEqual(params.get("method"), "simulation")
        self.assertEqual(params.get("analysis_model"), "bayes")
        self.assertEqual(params.get("n_clusters"), 15)

        # 5. Configure mock_core_sim_func.return_value for FALLBACK
        fallback_sim_results = {
            'power': None, # Power is None due to non-convergence
            'details_bayes': {'converged': False, 'summary': 'model did not converge due to high r-hat values and low ess for power.', 'r_hat_max': 1.5},
            'design_method': 'Cluster RCT'
        }
        mock_core_sim_func.return_value = fallback_sim_results

        # 6. Call calculate function
        calculate_func = designpower_app.COMPONENTS[("Cluster RCT", "Continuous Outcome")]["calculate"]
        results = calculate_func(params)

        # 7. Assert mock_core_sim_func was called correctly (same as power_converged)
        mock_core_sim_func.assert_called_once_with(
            mean1=params["mean1"],
            mean2=params["mean2"],
            std_dev=params["std_dev"],
            icc=params["icc"],
            cluster_size=params["cluster_size"],
            n_clusters=params["n_clusters"],
            alpha=params["alpha"],
            nsim=params["nsim"],
            seed=params["seed"],
            analysis_model='bayes',
            use_satterthwaite=params.get("use_satterthwaite", False),
            use_bias_correction=params.get("use_bias_correction", False),
            bayes_draws=params["bayes_draws"],
            bayes_warmup=params["bayes_warmup"],
            lmm_method=params.get("lmm_method", "auto"),
            lmm_reml=params.get("lmm_reml", True),
            progress_callback=ANY
        )

        # 8. Assert results dictionary reflects non-convergence
        self.assertEqual(results, fallback_sim_results)
        self.assertIsNone(results.get('power'))
        self.assertFalse(results['details_bayes']['converged'])
        self.assertIn('high r-hat values', results['details_bayes']['summary'].lower())

    @patch('core.designs.cluster_rct.simulation_continuous.min_detectable_effect_continuous_sim')
    @patch('app.components.cluster_rct.st', new_callable=MagicMock)
    @patch('app.designpower_app.st')
    def test_cluster_rct_continuous_bayesian_mde_converged(
        self, mock_st_main_app, mock_st_component, mock_core_sim_func
    ):
        """Test Cluster RCT, Continuous, Bayesian, MDE - Converged scenario."""
        # 1. Set up st.session_state
        mock_st_main_app.session_state = {
            "design_type": "Cluster RCT",
            "outcome_type": "Continuous Outcome",
            "calculation_type": "Minimum Detectable Effect", # Changed to MDE
            "hypothesis_type": "Superiority",
            "results": None
        }

        # 2. Configure mock_st_component return values
        mock_st_component.radio.side_effect = ["Simulation"]
        mock_st_component.selectbox.side_effect = ["Bayesian (Stan)"]

        # Order for MDE calc: cluster_size, icc, n_clusters, mean1, std_dev
        keyless_ni_values = [25, 0.04, 10, 12.0, 2.5] 
        keyless_ni_call_count = 0

        def number_input_side_effect(*args, **kwargs):
            nonlocal keyless_ni_call_count
            key = kwargs.get('key')
            if key:
                if key == 'cluster_continuous_nsim': return 1200
                if key == 'cluster_continuous_seed': return 456
                if key == 'cluster_continuous_bayes_draws': return 700
                if key == 'cluster_continuous_bayes_warmup': return 500
                return MagicMock()
            else:
                if keyless_ni_call_count < len(keyless_ni_values):
                    value = keyless_ni_values[keyless_ni_call_count]
                    keyless_ni_call_count += 1
                    return value
                return MagicMock()
        mock_st_component.number_input.side_effect = number_input_side_effect
        # Power and Alpha are sliders for MDE
        mock_st_component.slider.return_value = 0.80  # Power input for MDE
        mock_st_component.select_slider.return_value = 0.05  # Alpha input
        mock_st_component.expander.return_value.__enter__.return_value = MagicMock()
        mock_st_component.columns.return_value = [MagicMock(), MagicMock()]
        mock_st_component.checkbox.return_value = False

        # 3. Call render function
        render_func = designpower_app.COMPONENTS[("Cluster RCT", "Continuous Outcome")]["render"]
        params = render_func(
            mock_st_main_app.session_state["calculation_type"],
            mock_st_main_app.session_state["hypothesis_type"]
        )

        # 4. Assert params
        self.assertEqual(params.get("method"), "simulation")
        self.assertEqual(params.get("analysis_model"), "bayes")
        self.assertEqual(params.get("cluster_size"), 25)
        self.assertEqual(params.get("icc"), 0.04)
        self.assertEqual(params.get("n_clusters"), 10)
        self.assertEqual(params.get("mean1"), 12.0)
        self.assertNotIn("mean2", params) # mean2 is not an input for MDE
        self.assertEqual(params.get("std_dev"), 2.5)
        self.assertEqual(params.get("power"), 0.80)
        self.assertEqual(params.get("alpha"), 0.05)
        self.assertEqual(params.get("nsim"), 1200)
        self.assertEqual(params.get("seed"), 456)
        self.assertEqual(params.get("bayes_draws"), 700)
        self.assertEqual(params.get("bayes_warmup"), 500)

        # 5. Configure mock_core_sim_func.return_value for convergence
        expected_sim_results = {
            'mde': 1.5, 
            'mean2_mde': 10.5, # mean1 (12.0) - mde (1.5)
            'details_bayes': {'converged': True, 'summary': 'Mocked Stan Summary for MDE', 'r_hat_max': 1.01},
            'design_method': 'Cluster RCT'
        }
        mock_core_sim_func.return_value = expected_sim_results

        # 6. Call calculate function
        calculate_func = designpower_app.COMPONENTS[("Cluster RCT", "Continuous Outcome")]["calculate"]
        returned_results_from_calculate = calculate_func(params)
        
        # Simulate main app assigning the returned results to session_state
        mock_st_main_app.session_state.results = returned_results_from_calculate

        results_from_session_state_after_calc = mock_st_main_app.session_state.results # Re-fetch after assignment for clarity

        # Assertions for parameters passed to the calculate function
        self.assertEqual(params.get("calc_type"), "Minimum Detectable Effect")
        self.assertEqual(params.get("method"), "simulation")
        self.assertEqual(params.get("analysis_model"), "bayes")
        self.assertEqual(params.get("cluster_size"), 25)
        self.assertEqual(params.get("icc"), 0.04)
        self.assertEqual(params.get("n_clusters"), 10)
        self.assertEqual(params.get("mean1"), 12.0)
        self.assertEqual(params.get("std_dev"), 2.5)
        self.assertEqual(params.get("power"), 0.80)
        self.assertEqual(params.get("alpha"), 0.05)
        self.assertNotIn("mean2", params) # mean2 is not an input for MDE

        # Assert that the core simulation function was called correctly
        mock_core_sim_func.assert_called_once_with(
            mean1=12.0,
            std_dev=2.5,
            icc=0.04,
            cluster_size=25,
            n_clusters=10,
            power=0.80,
            alpha=0.05,
            nsim=1200,
            seed=456,
            analysis_model='bayes',
            use_satterthwaite=params.get("use_satterthwaite", False),
            use_bias_correction=params.get("use_bias_correction", False),
            bayes_draws=700,
            bayes_warmup=500,
            lmm_method=params.get("lmm_method", "auto"),
            lmm_reml=params.get("lmm_reml", True),
            progress_callback=ANY
        )

        # Manually trigger the error display logic that would occur in designpower_app.py
        if results_from_session_state_after_calc is not None:
            if isinstance(results_from_session_state_after_calc, dict) and "error" in results_from_session_state_after_calc:
                mock_st_main_app.error(results_from_session_state_after_calc["error"])
        
        # Check that st.error was called with the expected fallback message
        expected_error_message = "Simulation did not converge. Using fallback MDE."
        mock_st_main_app.error.assert_any_call(expected_error_message)

        # Verify the results dictionary in session state
        self.assertIsNotNone(results_from_session_state_after_calc) # Ensure results were set
        self.assertIn("error", results_from_session_state_after_calc)
        # The error message in results should match the mock core sim func return
        self.assertEqual(results_from_session_state_after_calc.get("error"), "Simulation did not converge. Using fallback MDE.") 
        self.assertEqual(results_from_session_state_after_calc.get("fallback_mde"), 1.8)
        self.assertEqual(results_from_session_state_after_calc.get("fallback_mean2_mde"), 13.8) # 12.0 (mean1) + 1.8 (fallback_mde)
        self.assertFalse(results_from_session_state_after_calc.get("converged"))

    @patch("app.designpower_app.st")
    @patch("app.components.cluster_rct.st")
    @patch("app.components.cluster_rct.render_continuous_advanced_options")
    @patch("core.designs.cluster_rct.simulation_continuous.min_detectable_effect_continuous_sim")
    def test_cluster_rct_continuous_bayesian_mde_fallback(
        self, 
        mock_core_sim_func, 
        mock_render_advanced, 
        mock_st_component, 
        mock_st_app
    ):
        """Test Cluster RCT, Continuous, Bayesian, MDE - Fallback scenario."""
        # Reset session state
        mock_st_app.session_state = MagicMock()
        mock_st_component.session_state = mock_st_app.session_state # Link session states

        mock_st_app.session_state.results = None
        mock_st_app.session_state.calculation_type = "Minimum Detectable Effect"
        mock_st_app.session_state.hypothesis_type = "Superiority"

        # Mock UI elements and their return values
        # mock_st_app.session_state and mock_st_component.session_state are already set up

        # Mock return values for render_continuous_advanced_options
        mock_render_advanced.return_value = {
            "method": "simulation",
            "analysis_model": "bayes",
            "nsim": 1200,
            "seed": 456,
            "use_satterthwaite": False,
            "use_bias_correction": False,
            "bayes_draws": 700,
            "bayes_warmup": 500,
            "lmm_method": "auto",
            "lmm_reml": True,
        }

        # Mock st.number_input calls
        # Order: cluster_size, icc, n_clusters, mean1, std_dev (for MDE)
        # These are keyless in render_cluster_continuous for these specific inputs
        keyless_ni_values = [25, 0.04, 10, 12.0, 2.5] 
        # For keyed inputs in render_continuous_advanced_options
        keyed_ni_values = {
            "cluster_continuous_nsim": 1200,
            "cluster_continuous_seed": 456,
            "cluster_continuous_bayes_draws": 700,
            "cluster_continuous_bayes_warmup": 500,
        }

        def number_input_side_effect(label, value=None, min_value=None, max_value=None, step=None, format=None, key=None, help=None):
            if key:
                if key in keyed_ni_values:
                    return keyed_ni_values[key]
            elif keyless_ni_values: # Check if list is not empty
                return keyless_ni_values.pop(0) # Return first item for non-keyed inputs
            # Fallback for any other number_input calls not explicitly handled
            if "Mean Outcome in Control Group" in label: return 12.0
            if "Standard Deviation" in label: return 2.5
            if "Average Cluster Size" in label: return 25
            if "Intracluster Correlation Coefficient (ICC)" in label: return 0.04
            if "Number of Clusters per Arm" in label: return 10
            if "Number of Simulations" in label: return 1200
            if "Random Seed" in label: return 456
            if "Posterior draws" in label: return 700
            if "Warm-up iterations" in label: return 500
            return MagicMock()
        mock_st_component.number_input.side_effect = number_input_side_effect
        
        # Power is st.slider, Alpha is st.select_slider for MDE
        mock_st_component.slider.return_value = 0.80  # Power input for MDE
        mock_st_component.select_slider.return_value = 0.05  # Alpha input
        mock_st_component.expander.return_value.__enter__.return_value = MagicMock()
        mock_st_component.columns.return_value = [MagicMock(), MagicMock()]
        mock_st_component.checkbox.return_value = False
        mock_st_component.radio.return_value = "Simulation" # Calculation Method
        mock_st_component.selectbox.return_value = "Bayesian (Stan)" # Statistical Model

        # Mock the core simulation function to return fallback results
        fallback_results = {
            "error": "Simulation did not converge. Using fallback MDE.",
            "fallback_mde": 1.8,
            "fallback_mean2_mde": 13.8, # 12.0 (mean1) + 1.8 (fallback_mde)
            "converged": False
        }
        mock_core_sim_func.return_value = fallback_results

        # Call the render and calculate functions
        params = designpower_app.COMPONENTS[("Cluster RCT", "Continuous Outcome")]["render"](
            mock_st_app.session_state.calculation_type, mock_st_app.session_state.hypothesis_type
        )
        
        # Update params with mean1 for fallback_mean2_mde calculation if needed by test logic
        # (though the mock directly provides fallback_mean2_mde)
        # Here, we ensure the mean1 used in the test matches what the mock would expect if it calculated it.
        # The actual mean1 passed to the core sim comes from the number_input mock.
        expected_mean1_for_fallback_calc = 10.0 # This is just for calculating the expected fallback_mean2_mde
        fallback_results["fallback_mean2_mde"] = expected_mean1_for_fallback_calc + fallback_results["fallback_mde"]

        # Re-mock the core simulation function with the updated fallback_results (if mean1 dependency was real)
        # In this case, it's fine as the mock is static. We'll use the initially set mean1 (12.0) for assertions.
        mock_core_sim_func.return_value = {
            "error": "Simulation did not converge. Using fallback MDE.",
            "fallback_mde": 1.8,
            "fallback_mean2_mde": 12.0 + 1.8, # Based on mean1=12.0 from number_input
            "converged": False
        }

        returned_results_from_calculate = designpower_app.COMPONENTS[("Cluster RCT", "Continuous Outcome")]["calculate"](params)

        # Simulate main app assigning the returned results to session_state
        mock_st_app.session_state.results = returned_results_from_calculate

        results_from_session_state_after_calc = mock_st_app.session_state.results # Use the mock's session_state

        # Assertions for parameters passed to the calculate function
        self.assertEqual(params.get("calc_type"), "Minimum Detectable Effect")
        self.assertEqual(params.get("method"), "simulation")
        self.assertEqual(params.get("analysis_model"), "bayes")
        self.assertEqual(params.get("cluster_size"), 25)
        self.assertEqual(params.get("icc"), 0.04)
        self.assertEqual(params.get("n_clusters"), 10)
        self.assertEqual(params.get("mean1"), 12.0)
        self.assertEqual(params.get("std_dev"), 2.5)
        self.assertEqual(params.get("power"), 0.80)
        self.assertEqual(params.get("alpha"), 0.05)
        self.assertNotIn("mean2", params) # mean2 is not an input for MDE

        # Assert that the core simulation function was called correctly
        mock_core_sim_func.assert_called_once_with(
            mean1=12.0,
            std_dev=2.5,
            icc=0.04,
            cluster_size=25,
            n_clusters=10,
            power=0.80,
            alpha=0.05,
            nsim=1200,
            seed=456,
            analysis_model='bayes',
            use_satterthwaite=False, # Default for Bayesian
            use_bias_correction=False, # Default for Bayesian
            bayes_draws=700,
            bayes_warmup=500,
            lmm_method='auto', # Default, not strictly used by Bayesian
            lmm_reml=True, # Default, not strictly used by Bayesian
            progress_callback=ANY
        )

        # Manually trigger the error display logic that would occur in designpower_app.py
        if results_from_session_state_after_calc is not None:
            if isinstance(results_from_session_state_after_calc, dict) and "error" in results_from_session_state_after_calc:
                mock_st_app.error(results_from_session_state_after_calc["error"])
        
        # Check that st.error was called with the expected fallback message
        expected_error_message = "Simulation did not converge. Using fallback MDE."
        mock_st_app.error.assert_any_call(expected_error_message)

        # Verify the results dictionary in session state
        self.assertIsNotNone(results_from_session_state_after_calc) # Ensure results were set
        self.assertIn("error", results_from_session_state_after_calc)
        # The error message in results should match the mock core sim func return
        self.assertEqual(results_from_session_state_after_calc.get("error"), "Simulation did not converge. Using fallback MDE.") 
        self.assertEqual(results_from_session_state_after_calc.get("fallback_mde"), 1.8)
        self.assertEqual(results_from_session_state_after_calc.get("fallback_mean2_mde"), 13.8) # 12.0 (mean1) + 1.8 (fallback_mde)
        self.assertFalse(results_from_session_state_after_calc.get("converged"))


if __name__ == '__main__':
    unittest.main()
