import unittest
from unittest.mock import patch, MagicMock
import sys
import os

# Add project root to sys.path to allow for app module imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from app.components import cluster_rct
# Corrected import paths for core modules
from core.designs.cluster_rct import analytical_continuous, simulation_continuous
from core.designs.cluster_rct import analytical_binary, simulation_binary

class TestClusterRCTComponent(unittest.TestCase):

    def setUp(self):
        """Set up for test methods."""
        # Common setup if needed, e.g., default parameters
        self.default_params_continuous = {
            "calc_type": "Power",
            "method": "analytical",
            "n_clusters": 10,
            "cluster_size": 20,
            "icc": 0.05,
            "mean1": 10,
            "mean2": 12,
            "std_dev": 2,
            "alpha": 0.05,
            "hypothesis_type": "Superiority"
        }
        self.default_params_binary = {
            "calc_type": "Power",
            "method": "analytical",
            "n_clusters": 20,
            "cluster_size": 50,
            "icc": 0.02,
            "p1": 0.30, # Control group proportion
            "p2": 0.20, # Treatment group proportion
            "alpha": 0.05,
            "power": 0.80, # For sample size and MDE calcs
            "cv_cluster_size": 0.1,
            "icc_scale": "Linear",
            "effect_measure": "risk_difference",
            "run_sensitivity": False,
            "nsim": 1000, # For simulation
            "seed": 42    # For simulation
        }

    def test_example_placeholder(self):
        """Placeholder test to ensure the file is runnable."""
        self.assertTrue(True)

    @patch('app.components.cluster_rct.analytical_continuous.power_continuous')
    def test_calculate_cluster_continuous_analytical_power(self, mock_power_continuous):
        """Test calculate_cluster_continuous for analytical power calculation."""
        # Arrange
        params = self.default_params_continuous.copy()
        params["calc_type"] = "Power"
        params["method"] = "analytical"

        expected_core_result = {"power": 0.85, "n_total": 400}
        mock_power_continuous.return_value = expected_core_result

        # Act
        result = cluster_rct.calculate_cluster_continuous(params)

        # Assert
        mock_power_continuous.assert_called_once_with(
            n_clusters=params["n_clusters"],
            cluster_size=params["cluster_size"],
            icc=params["icc"],
            mean1=params["mean1"],
            mean2=params["mean2"],
            std_dev=params["std_dev"],
            alpha=params["alpha"]
        )
        self.assertEqual(result, expected_core_result)

    @patch('app.components.cluster_rct.analytical_continuous.sample_size_continuous')
    def test_calculate_cluster_continuous_analytical_sample_size(self, mock_sample_size_continuous):
        """Test calculate_cluster_continuous for analytical sample size calculation."""
        # Arrange
        params = self.default_params_continuous.copy()
        params["calc_type"] = "Sample Size"
        params["method"] = "analytical"
        params["power"] = 0.80 # Add power as it's an input for sample size
        # Remove n_clusters as it's an output for sample size, not input
        if "n_clusters" in params: # Make deletion robust
            del params["n_clusters"]

        expected_core_result = {"n_clusters": 12, "n_total": 240}
        mock_sample_size_continuous.return_value = expected_core_result

        # Act
        result = cluster_rct.calculate_cluster_continuous(params)

        # Assert
        mock_sample_size_continuous.assert_called_once_with(
            mean1=params["mean1"],
            mean2=params["mean2"],
            std_dev=params["std_dev"],
            icc=params["icc"],
            cluster_size=params["cluster_size"],
            power=params["power"],
            alpha=params["alpha"]
        )
        self.assertEqual(result, expected_core_result)

    @patch('app.components.cluster_rct.analytical_continuous.min_detectable_effect_continuous')
    def test_calculate_cluster_continuous_analytical_mde(self, mock_mde_continuous):
        """Test calculate_cluster_continuous for analytical MDE calculation."""
        # Arrange
        params = self.default_params_continuous.copy()
        params["calc_type"] = "Minimum Detectable Effect"
        params["method"] = "analytical"
        params["power"] = 0.80 # MDE needs power as input
        # Remove mean2 as it's an output (derived from MDE), not input
        if "mean2" in params:
            del params["mean2"]
        # n_clusters is an input for MDE, ensure it's present from default_params

        expected_core_result = {"mde": 0.5, "mean2_mde": 10.5}
        mock_mde_continuous.return_value = expected_core_result

        # Act
        result = cluster_rct.calculate_cluster_continuous(params)

        # Assert
        mock_mde_continuous.assert_called_once_with(
            n_clusters=params["n_clusters"],
            cluster_size=params["cluster_size"],
            icc=params["icc"],
            std_dev=params["std_dev"],
            power=params["power"],
            alpha=params["alpha"]
        )
        self.assertEqual(result, expected_core_result)

    @patch('app.components.cluster_rct.st') # Mock streamlit for progress bar
    @patch('app.components.cluster_rct.simulation_continuous.power_continuous_sim')
    def test_calculate_cluster_continuous_simulation_power(self, mock_power_sim, mock_st):
        """Test calculate_cluster_continuous for simulation-based power calculation."""
        # Arrange
        params = self.default_params_continuous.copy()
        params["method"] = "simulation"
        params["calc_type"] = "Power" # Already default, but explicit
        # Add simulation-specific parameters
        params["nsim"] = 1000
        params["seed"] = 42
        params["analysis_model"] = "ttest"
        params["use_satterthwaite"] = False
        params["use_bias_correction"] = False
        params["bayes_draws"] = 500 # Will be passed but might not be used by ttest sim
        params["bayes_warmup"] = 500 # Similar to bayes_draws
        params["lmm_method"] = "auto"
        params["lmm_reml"] = True

        # Mock the progress bar behavior
        mock_progress_bar = MagicMock()
        mock_st.progress.return_value = mock_progress_bar

        expected_core_result = {"power": 0.82, "n_total": 400, "nsim": 1000}
        mock_power_sim.return_value = expected_core_result

        # Act
        result = cluster_rct.calculate_cluster_continuous(params)

        # Assert
        mock_st.progress.assert_called_once_with(0.0)
        mock_power_sim.assert_called_once_with(
            n_clusters=params["n_clusters"],
            cluster_size=params["cluster_size"],
            icc=params["icc"],
            mean1=params["mean1"],
            mean2=params["mean2"],
            std_dev=params["std_dev"],
            alpha=params["alpha"],
            nsim=params["nsim"],
            seed=params["seed"],
            analysis_model=params["analysis_model"],
            use_satterthwaite=params["use_satterthwaite"],
            use_bias_correction=params["use_bias_correction"],
            bayes_draws=params["bayes_draws"],
            bayes_warmup=params["bayes_warmup"],
            lmm_method=params["lmm_method"],
            lmm_reml=params["lmm_reml"],
            progress_callback=unittest.mock.ANY # Callback is internally defined
        )
        mock_progress_bar.empty.assert_called_once()
        self.assertEqual(result, expected_core_result)

    @patch('app.components.cluster_rct.simulation_continuous.sample_size_continuous_sim')
    def test_calculate_cluster_continuous_simulation_sample_size(self, mock_sample_size_sim):
        """Test calculate_cluster_continuous for simulation-based sample size calculation."""
        # Arrange
        params = self.default_params_continuous.copy()
        params["method"] = "simulation"
        params["calc_type"] = "Sample Size"
        params["power"] = 0.80 # Sample size needs power
        if "n_clusters" in params: # n_clusters is an output
            del params["n_clusters"]

        # Add simulation-specific parameters
        params["nsim"] = 1200
        params["seed"] = 43
        params["analysis_model"] = "mixedlm"
        params["use_satterthwaite"] = True
        params["use_bias_correction"] = True
        params["bayes_draws"] = 600
        params["bayes_warmup"] = 400
        params["lmm_method"] = "powell"
        params["lmm_reml"] = False

        expected_core_result = {"n_clusters": 15, "n_total": 300, "nsim": 1200}
        mock_sample_size_sim.return_value = expected_core_result

        # Act
        result = cluster_rct.calculate_cluster_continuous(params)

        # Assert
        mock_sample_size_sim.assert_called_once_with(
            mean1=params["mean1"],
            mean2=params["mean2"],
            std_dev=params["std_dev"],
            icc=params["icc"],
            cluster_size=params["cluster_size"],
            power=params["power"],
            alpha=params["alpha"],
            nsim=params["nsim"],
            seed=params["seed"],
            analysis_model=params["analysis_model"],
            use_satterthwaite=params["use_satterthwaite"],
            use_bias_correction=params["use_bias_correction"],
            bayes_draws=params["bayes_draws"],
            bayes_warmup=params["bayes_warmup"],
            lmm_method=params["lmm_method"],
            lmm_reml=params["lmm_reml"]
            # No progress_callback for sample size sim in component
        )
        self.assertEqual(result, expected_core_result)

    @patch('app.components.cluster_rct.st') # Mock streamlit for progress bar
    @patch('app.components.cluster_rct.simulation_continuous.min_detectable_effect_continuous_sim')
    def test_calculate_cluster_continuous_simulation_mde_non_bayesian(self, mock_mde_sim, mock_st):
        """Test calculate_cluster_continuous for non-Bayesian simulation-based MDE."""
        # Arrange
        params = self.default_params_continuous.copy()
        params["method"] = "simulation"
        params["calc_type"] = "Minimum Detectable Effect"
        params["power"] = 0.85 # MDE needs power
        if "mean2" in params: # mean2 is an output
            del params["mean2"]

        params["nsim"] = 1100
        params["seed"] = 44
        params["analysis_model"] = "ttest" # Non-Bayesian model
        params["use_satterthwaite"] = False
        params["use_bias_correction"] = False
        params["bayes_draws"] = 500
        params["bayes_warmup"] = 500
        params["lmm_method"] = "auto"
        params["lmm_reml"] = True

        # Mock the progress bar behavior
        mock_progress_bar = MagicMock()
        mock_st.progress.return_value = mock_progress_bar

        expected_core_result = {"mde": 0.45, "mean2_mde": 10.45, "nsim": 1100}
        mock_mde_sim.return_value = expected_core_result

        # Act
        result = cluster_rct.calculate_cluster_continuous(params)

        # Assert
        mock_st.progress.assert_called_once_with(0.0)
        mock_mde_sim.assert_called_once_with(
            n_clusters=params["n_clusters"],
            cluster_size=params["cluster_size"],
            icc=params["icc"],
            std_dev=params["std_dev"],
            mean1=params["mean1"],
            power=params["power"],
            alpha=params["alpha"],
            nsim=params["nsim"],
            seed=params["seed"],
            analysis_model=params["analysis_model"],
            use_satterthwaite=params["use_satterthwaite"],
            use_bias_correction=params["use_bias_correction"],
            bayes_draws=params["bayes_draws"],
            bayes_warmup=params["bayes_warmup"],
            lmm_method=params["lmm_method"],
            lmm_reml=params["lmm_reml"],
            progress_callback=unittest.mock.ANY
        )
        mock_progress_bar.empty.assert_called_once()
        self.assertEqual(result, expected_core_result)

    @patch('app.components.cluster_rct.st') # Mock streamlit for progress bar
    @patch('app.components.cluster_rct.simulation_continuous.min_detectable_effect_continuous_sim')
    def test_calculate_cluster_continuous_simulation_mde_bayesian_direct(self, mock_mde_sim, mock_st):
        """Test calculate_cluster_continuous for Bayesian simulation-based MDE (direct call)."""
        # Arrange
        params = self.default_params_continuous.copy()
        params["method"] = "simulation"
        params["calc_type"] = "Minimum Detectable Effect"
        params["power"] = 0.90 # MDE needs power
        if "mean2" in params: # mean2 is an output
            del params["mean2"]

        params["nsim"] = 1300
        params["seed"] = 45
        params["analysis_model"] = "bayes" # Bayesian model
        params["use_satterthwaite"] = False # Not typically used with Bayes
        params["use_bias_correction"] = False # Not typically used with Bayes
        params["bayes_draws"] = 700
        params["bayes_warmup"] = 300
        params["lmm_method"] = "auto" # Not used by Bayes
        params["lmm_reml"] = True    # Not used by Bayes

        # Mock the progress bar behavior
        mock_progress_bar = MagicMock()
        mock_st.progress.return_value = mock_progress_bar

        expected_core_result = {"mde": 0.40, "mean2_mde": 10.40, "nsim": 1300}
        mock_mde_sim.return_value = expected_core_result

        # Act
        result = cluster_rct.calculate_cluster_continuous(params)

        # Assert
        mock_st.progress.assert_called_once_with(0.0)
        mock_mde_sim.assert_called_once_with(
            n_clusters=params["n_clusters"],
            cluster_size=params["cluster_size"],
            icc=params["icc"],
            std_dev=params["std_dev"],
            mean1=params["mean1"],
            power=params["power"],
            alpha=params["alpha"],
            nsim=params["nsim"],
            seed=params["seed"],
            analysis_model="bayes", # Ensure 'bayes' is passed
            use_satterthwaite=params["use_satterthwaite"],
            use_bias_correction=params["use_bias_correction"],
            bayes_draws=params["bayes_draws"],
            bayes_warmup=params["bayes_warmup"],
            lmm_method=params["lmm_method"],
            lmm_reml=params["lmm_reml"],
            progress_callback=unittest.mock.ANY
        )
        mock_progress_bar.empty.assert_called_once()
        self.assertEqual(result, expected_core_result)

    @patch('app.components.cluster_rct.analytical_continuous.min_detectable_effect_continuous')
    @patch('app.components.cluster_rct.simulation_continuous.min_detectable_effect_continuous_sim')
    @patch('app.components.cluster_rct.st')
    def test_calculate_cluster_continuous_simulation_mde_bayesian_fallback(
        self, mock_st, mock_mde_sim, mock_mde_analytical
    ):
        """Test Bayesian MDE simulation fallback to analytical when sim fails."""
        # Arrange
        params = self.default_params_continuous.copy()
        params["method"] = "simulation"
        params["calc_type"] = "Minimum Detectable Effect"
        params["power"] = 0.90
        if "mean2" in params:
            del params["mean2"]

        params["nsim"] = 1300
        params["seed"] = 45
        params["analysis_model"] = "bayes"
        params["bayes_draws"] = 700
        params["bayes_warmup"] = 300
        # Other sim params that would be passed to the initial sim attempt
        params["use_satterthwaite"] = False
        params["use_bias_correction"] = False
        params["lmm_method"] = "auto"
        params["lmm_reml"] = True

        # Mock simulation failure (e.g., non-convergence)
        sim_failure_result = {"mde": None, "nsim": params["nsim"]} # or an error key
        mock_mde_sim.return_value = sim_failure_result

        # Expected analytical fallback result
        analytical_fallback_result = {"mde": 0.42, "mean2_mde": 10.42}
        mock_mde_analytical.return_value = analytical_fallback_result

        mock_progress_bar = MagicMock()
        mock_st.progress.return_value = mock_progress_bar
        
        expected_warning_message = (
            "Bayesian MDE simulation failed to converge or returned an error. "
            "Falling back to analytical MDE calculation. "
            "The analytical result may differ and does not account for "
            "Bayesian posterior uncertainty."
        )

        # Act
        result = cluster_rct.calculate_cluster_continuous(params)

        # Assert
        # 1. Simulation was attempted
        mock_st.progress.assert_called_once_with(0.0)
        mock_mde_sim.assert_called_once_with(
            n_clusters=params["n_clusters"],
            cluster_size=params["cluster_size"],
            icc=params["icc"],
            std_dev=params["std_dev"],
            mean1=params["mean1"],
            power=params["power"],
            alpha=params["alpha"],
            nsim=params["nsim"],
            seed=params["seed"],
            analysis_model="bayes",
            use_satterthwaite=params["use_satterthwaite"],
            use_bias_correction=params["use_bias_correction"],
            bayes_draws=params["bayes_draws"],
            bayes_warmup=params["bayes_warmup"],
            lmm_method=params["lmm_method"],
            lmm_reml=params["lmm_reml"],
            progress_callback=unittest.mock.ANY
        )
        mock_progress_bar.empty.assert_called_once()

        # 2. Warning was issued
        mock_st.warning.assert_called_once_with(expected_warning_message)

        # 3. Analytical fallback was called
        mock_mde_analytical.assert_called_once_with(
            n_clusters=params["n_clusters"],
            cluster_size=params["cluster_size"],
            icc=params["icc"],
            std_dev=params["std_dev"],
            power=params["power"],
            alpha=params["alpha"]
        )

        # 4. Final result is analytical + warning + nsim from sim attempt
        expected_final_result = analytical_fallback_result.copy()
        expected_final_result["warning"] = expected_warning_message
        expected_final_result["nsim"] = params["nsim"] # nsim from sim attempt is preserved
        expected_final_result["design_method"] = "Cluster RCT" # Added by component
        
        self.assertEqual(result, expected_final_result)

    def test_calculate_cluster_continuous_missing_params(self):
        """Test error handling for missing required parameters in calculate_cluster_continuous."""
        base_params_power = {
            "calc_type": "Power", "method": "analytical",
            "n_clusters": 10, "cluster_size": 20, "icc": 0.05,
            "mean1": 10, "mean2": 12, "std_dev": 2, "alpha": 0.05
        }
        required_for_power = ["n_clusters", "cluster_size", "icc", "mean1", "mean2", "std_dev", "alpha"]

        base_params_ss = {
            "calc_type": "Sample Size", "method": "analytical",
            "mean1": 10, "mean2": 12, "std_dev": 2, "icc": 0.05,
            "cluster_size": 20, "power": 0.8, "alpha": 0.05
        }
        required_for_ss = ["mean1", "mean2", "std_dev", "icc", "cluster_size", "power", "alpha"]

        base_params_mde = {
            "calc_type": "Minimum Detectable Effect", "method": "analytical",
            "n_clusters": 10, "cluster_size": 20, "icc": 0.05,
            "mean1": 10, "std_dev": 2, "power": 0.8, "alpha": 0.05
        }
        required_for_mde = ["n_clusters", "cluster_size", "icc", "mean1", "std_dev", "power", "alpha"]

        scenarios = [
            (base_params_power, required_for_power),
            (base_params_ss, required_for_ss),
            (base_params_mde, required_for_mde)
        ]

        for base_params, required_list in scenarios:
            calc_type = base_params["calc_type"]
            for req_param in required_list:
                with self.subTest(calc_type=calc_type, missing_param=req_param):
                    params = base_params.copy()
                    del params[req_param] # Remove the required parameter
                    
                    result = cluster_rct.calculate_cluster_continuous(params)
                    self.assertIn("error", result, f"Error key missing for {calc_type} missing {req_param}")
                    self.assertEqual(result["error"], f"Missing required parameter: {req_param}")

    @patch('app.components.cluster_rct.analytical_binary.power_binary')
    @patch('core.designs.cluster_rct.cluster_utils.convert_icc_logit_to_linear') # Corrected mock path to source
    def test_calculate_cluster_binary_analytical_power(self, mock_convert_icc, mock_power_analytical):
        """Test calculate_cluster_binary for analytical power calculation."""
        # Arrange
        params = self.default_params_binary.copy()
        params["calc_type"] = "Power"
        params["method"] = "analytical"
        params["icc_scale"] = "Linear" # Ensure no conversion for this test
        params["run_sensitivity"] = False
        # p2 is required for power, remove if it's an output for other types
        if "mde" in params: del params["mde"] 

        expected_core_result = {"power": 0.85, "design_effect": 1.98}
        mock_power_analytical.return_value = expected_core_result

        # Act
        result = cluster_rct.calculate_cluster_binary(params)

        # Assert
        mock_convert_icc.assert_not_called() # ICC scale is Linear
        mock_power_analytical.assert_called_once_with(
            n_clusters=params["n_clusters"],
            cluster_size=params["cluster_size"],
            icc=params["icc"],
            p1=params["p1"],
            p2=params["p2"],
            alpha=params["alpha"],
            cv_cluster_size=params["cv_cluster_size"]
        )
        
        expected_final_result = expected_core_result.copy()
        expected_final_result["design_method"] = "Cluster RCT"
        self.assertEqual(result, expected_final_result)

    @patch('app.components.cluster_rct.analytical_binary.sample_size_binary')
    @patch('core.designs.cluster_rct.cluster_utils.convert_icc_logit_to_linear')
    def test_calculate_cluster_binary_analytical_sample_size(self, mock_convert_icc, mock_sample_size_analytical):
        """Test calculate_cluster_binary for analytical sample size calculation."""
        # Arrange
        params = self.default_params_binary.copy()
        params["calc_type"] = "Sample Size"
        params["method"] = "analytical"
        params["icc_scale"] = "Linear"
        params["run_sensitivity"] = False
        # n_clusters is an output for sample size, remove if present from default
        if "n_clusters" in params: 
            original_n_clusters = params.pop("n_clusters") # Keep for later if needed, but not for this call

        expected_core_result = {"n_clusters": 30, "total_n": 1500, "design_effect": 1.98}
        mock_sample_size_analytical.return_value = expected_core_result

        # Act
        result = cluster_rct.calculate_cluster_binary(params)

        # Assert
        mock_convert_icc.assert_not_called() # ICC scale is Linear
        mock_sample_size_analytical.assert_called_once_with(
            p1=params["p1"],
            p2=params["p2"],
            icc=params["icc"],
            cluster_size=params["cluster_size"],
            power=params["power"],
            alpha=params["alpha"],
            cv_cluster_size=params["cv_cluster_size"],
            effect_measure=None # Component passes None when p2 is given
        )
        
        expected_final_result = expected_core_result.copy()
        expected_final_result["design_method"] = "Cluster RCT"
        self.assertEqual(result, expected_final_result)

    @patch('app.components.cluster_rct.analytical_binary.min_detectable_effect_binary')
    @patch('core.designs.cluster_rct.cluster_utils.convert_icc_logit_to_linear')
    def test_calculate_cluster_binary_analytical_mde(self, mock_convert_icc, mock_mde_analytical):
        """Test calculate_cluster_binary for analytical MDE calculation."""
        # Arrange
        params = self.default_params_binary.copy()
        params["calc_type"] = "Minimum Detectable Effect"
        params["method"] = "analytical"
        params["icc_scale"] = "Linear"
        params["run_sensitivity"] = False
        params["effect_measure"] = "risk_ratio" # Test with a specific effect measure
        # p2 is an output for MDE, remove if present from default
        if "p2" in params: 
            del params["p2"]

        expected_core_result = {"mde": 0.08, "p2_mde": 0.22, "design_effect": 1.98}
        mock_mde_analytical.return_value = expected_core_result

        # Act
        result = cluster_rct.calculate_cluster_binary(params)

        # Assert
        mock_convert_icc.assert_not_called() # ICC scale is Linear
        mock_mde_analytical.assert_called_once_with(
            n_clusters=params["n_clusters"],
            cluster_size=params["cluster_size"],
            icc=params["icc"],
            p1=params["p1"],
            power=params["power"],
            alpha=params["alpha"],
            cv_cluster_size=params["cv_cluster_size"],
            effect_measure=params["effect_measure"]
        )
        
        expected_final_result = expected_core_result.copy()
        expected_final_result["design_method"] = "Cluster RCT"
        self.assertEqual(result, expected_final_result)

    @patch('app.components.cluster_rct.analytical_binary.power_binary')
    @patch('core.designs.cluster_rct.cluster_utils.convert_icc_logit_to_linear')
    def test_calculate_cluster_binary_icc_logit_conversion(self, mock_convert_icc, mock_power_analytical):
        """Test ICC conversion from logit to linear scale in calculate_cluster_binary."""
        # Arrange
        params = self.default_params_binary.copy()
        params["calc_type"] = "Power"
        params["method"] = "analytical"
        params["icc_scale"] = "Logit"  # Key for this test
        params["run_sensitivity"] = False
        original_logit_icc = params["icc"] # e.g., 0.02 (as logit)

        converted_linear_icc = 0.018 # Example converted value
        mock_convert_icc.return_value = converted_linear_icc

        expected_core_result = {"power": 0.82, "design_effect": 1.88}
        mock_power_analytical.return_value = expected_core_result

        # Act
        result = cluster_rct.calculate_cluster_binary(params)

        # Assert
        mock_convert_icc.assert_called_once_with(original_logit_icc, params["p1"])
        mock_power_analytical.assert_called_once_with(
            n_clusters=params["n_clusters"],
            cluster_size=params["cluster_size"],
            icc=converted_linear_icc,  # Ensure the converted ICC is used
            p1=params["p1"],
            p2=params["p2"],
            alpha=params["alpha"],
            cv_cluster_size=params["cv_cluster_size"]
        )
        
        expected_final_result = expected_core_result.copy()
        expected_final_result["design_method"] = "Cluster RCT"
        self.assertEqual(result, expected_final_result)

    # --- Simulation Tests for Binary --- 

    @patch('app.components.cluster_rct.simulation_binary.power_binary_sim')
    @patch('core.designs.cluster_rct.cluster_utils.convert_icc_logit_to_linear')
    def test_calculate_cluster_binary_simulation_power(self, mock_convert_icc, mock_power_sim):
        """Test calculate_cluster_binary for simulation-based power calculation."""
        # Arrange
        params = self.default_params_binary.copy()
        params["calc_type"] = "Power"
        params["method"] = "simulation"
        params["icc_scale"] = "Linear" 
        params["run_sensitivity"] = False
        params["nsim"] = 1200 # Specific nsim for this test
        params["seed"] = 50   # Specific seed

        expected_core_result = {"power": 0.83, "design_effect": 1.95, "nsim": 1200}
        mock_power_sim.return_value = expected_core_result

        # Act
        result = cluster_rct.calculate_cluster_binary(params)

        # Assert
        mock_convert_icc.assert_not_called()
        mock_power_sim.assert_called_once_with(
            n_clusters=params["n_clusters"],
            cluster_size=params["cluster_size"],
            icc=params["icc"],
            p1=params["p1"],
            p2=params["p2"],
            alpha=params["alpha"],
            nsim=params["nsim"],
            seed=params["seed"],
            cv_cluster_size=params["cv_cluster_size"]
        )
        
        expected_final_result = expected_core_result.copy()
        expected_final_result["design_method"] = "Cluster RCT"
        self.assertEqual(result, expected_final_result)

    @patch('app.components.cluster_rct.simulation_binary.sample_size_binary_sim')
    @patch('core.designs.cluster_rct.cluster_utils.convert_icc_logit_to_linear')
    def test_calculate_cluster_binary_simulation_sample_size(self, mock_convert_icc, mock_sample_size_sim):
        """Test calculate_cluster_binary for simulation-based sample size calculation."""
        # Arrange
        params = self.default_params_binary.copy()
        params["calc_type"] = "Sample Size"
        params["method"] = "simulation"
        params["icc_scale"] = "Linear"
        params["run_sensitivity"] = False
        params["nsim"] = 1250
        params["seed"] = 55
        # n_clusters is an output, remove if present
        if "n_clusters" in params: del params["n_clusters"]

        expected_core_result = {"n_clusters": 28, "total_n": 1400, "design_effect": 1.90, "nsim": 1250}
        mock_sample_size_sim.return_value = expected_core_result

        # Act
        result = cluster_rct.calculate_cluster_binary(params)

        # Assert
        mock_convert_icc.assert_not_called()
        mock_sample_size_sim.assert_called_once_with(
            p1=params["p1"],
            p2=params["p2"],
            icc=params["icc"],
            cluster_size=params["cluster_size"],
            power=params["power"],
            alpha=params["alpha"],
            nsim=params["nsim"],
            seed=params["seed"],
            cv_cluster_size=params["cv_cluster_size"]
        )
        
        expected_final_result = expected_core_result.copy()
        expected_final_result["design_method"] = "Cluster RCT"
        self.assertEqual(result, expected_final_result)

    @patch('app.components.cluster_rct.simulation_binary.min_detectable_effect_binary_sim')
    @patch('core.designs.cluster_rct.cluster_utils.convert_icc_logit_to_linear')
    def test_calculate_cluster_binary_simulation_mde(self, mock_convert_icc, mock_mde_sim):
        """Test calculate_cluster_binary for simulation-based MDE calculation."""
        # Arrange
        params = self.default_params_binary.copy()
        params["calc_type"] = "Minimum Detectable Effect"
        params["method"] = "simulation"
        params["icc_scale"] = "Linear"
        params["run_sensitivity"] = False
        params["effect_measure"] = "odds_ratio"
        params["nsim"] = 1300
        params["seed"] = 60
        # p2 is an output, remove if present
        if "p2" in params: del params["p2"]

        expected_core_result = {"mde": 0.75, "p2_mde": 0.25, "design_effect": 1.85, "nsim": 1300}
        mock_mde_sim.return_value = expected_core_result

        # Act
        result = cluster_rct.calculate_cluster_binary(params)

        # Assert
        mock_convert_icc.assert_not_called()
        mock_mde_sim.assert_called_once_with(
            n_clusters=params["n_clusters"],
            cluster_size=params["cluster_size"],
            icc=params["icc"],
            p1=params["p1"],
            power=params["power"],
            alpha=params["alpha"],
            nsim=params["nsim"],
            seed=params["seed"],
            cv_cluster_size=params["cv_cluster_size"],
            effect_measure=params["effect_measure"]
        )
        
        expected_final_result = expected_core_result.copy()
        expected_final_result["design_method"] = "Cluster RCT"
        self.assertEqual(result, expected_final_result)

    # --- Parameter Validation Tests for Binary --- 

    def test_calculate_cluster_binary_missing_params(self):
        """Test calculate_cluster_binary for missing required parameters."""
        required_params_map = {
            "Power": ["n_clusters", "cluster_size", "icc", "p1", "p2", "alpha"],
            "Sample Size": ["cluster_size", "icc", "p1", "p2", "power", "alpha"],
            "Minimum Detectable Effect": ["n_clusters", "cluster_size", "icc", "p1", "power", "alpha", "effect_measure"],
        }

        for calc_type, req_params_list in required_params_map.items():
            for method in ["analytical", "simulation"]:
                for req_param in req_params_list:
                    with self.subTest(calc_type=calc_type, method=method, missing_param=req_param):
                        params = self.default_params_binary.copy()
                        params["calc_type"] = calc_type
                        params["method"] = method
                        params["run_sensitivity"] = False # Ensure sensitivity doesn't interfere
                        
                        # Remove the required parameter to test for its absence
                        if req_param in params: # It might have been removed by default_params setup for other calc_types
                            del params[req_param]
                        elif req_param == "effect_measure" and calc_type == "Minimum Detectable Effect":
                            # effect_measure might not be in default_params_binary, ensure it's considered missing
                            pass # It's fine if it's not there, it's the one we are testing for missing
                        elif req_param not in params:
                            # This case should ideally not happen if default_params is comprehensive enough
                            # or if the req_param is specific to a calc_type not in default_params' initial setup.
                            # For MDE, 'p2' is removed from default, for SS, 'n_clusters' is removed.
                            # If req_param is one of these, it's already 'missing' for the test.
                            pass 

                        result = cluster_rct.calculate_cluster_binary(params)
                        self.assertIn("error", result)
                        self.assertEqual(result["error"], f"Missing required parameter: {req_param}")

if __name__ == '__main__':
    unittest.main()
