import unittest
from unittest.mock import patch, MagicMock
import pytest # Keep for pytest.approx if used, or remove if all assertions are unittest based

from app.components import parallel_rct # Import the module to test

class TestParallelRCTComponent(unittest.TestCase):

    def setUp(self):
        """Set up default parameters for each outcome type."""
        self.default_params_continuous = {
            "n1": 100, "n2": 100, "allocation_ratio": 1.0,
            "mean1": 10, "mean2": 8, "sd1": 3, "sd2": 3,
            "alpha": 0.05, "power": 0.8, "sides": 2,
            "non_inferiority_margin": 1.5, "assumed_true_mean_diff": 0,
            "nsim": 100, "seed": 42, # Reduced nsim for tests
            "bayesian_method": "direct_simulation",
            "credible_interval": 0.95, "probability_threshold": 0.975,
            "prior_type": "non_informative",
            "prior_mean1_mean": 10, "prior_mean1_sd": 100,
            "prior_mean2_mean": 8, "prior_mean2_sd": 100,
            "prior_sd1_max": 10, "prior_sd2_max": 10,
        }

        self.default_params_binary = {
            "design_type": "Parallel RCT",
            "outcome_type": "Binary",
            "calculation_type": "Power",
            "method": "Analytical",
            "hypothesis_type": "Superiority",
            "p1": 0.3,  # Proportion in group 1 (control)
            "p2": 0.2,  # Proportion in group 2 (treatment)
            "n1": None, # Sample size group 1
            "n2": None, # Sample size group 2
            "power": None,
            "alpha": 0.05,
            "sides": 2,
            "allocation_ratio": 1.0,
            "effect_measure": "Risk Difference", # Could also be "Risk Ratio", "Odds Ratio"
            "non_inferiority_margin": None,
            "nsim": 1000, # Default for simulation tests
            "seed": 42
        }

        self.default_params_survival = {
            "n1": 100, "n2": 100, "allocation_ratio": 1.0,
            "median_survival1": 12, "hr": 0.75,
            "accrual_time": 12, "follow_up_time": 18, "dropout_rate": 0.1,
            "alpha": 0.05, "power": 0.8, "sides": 2,
            "non_inferiority_margin_hr": 1.3, "assumed_true_hr": 1.0,
            "nsim": 100, "seed": 42 # Reduced nsim for tests
        }

    # --- Tests for calculate_parallel_survival --- 
    # (Existing pytest functions will be converted and placed here)

    def test_calculate_parallel_survival_analytical_superiority_power_two_sided(self):
        """Test analytical power calculation for survival outcome, superiority, two-sided."""
        params = self.default_params_survival.copy()
        params["calculation_type"] = "Power"
        params["method"] = "Analytical"
        params["hypothesis_type"] = "Superiority"
        # Specific params for this test if different from default
        params["median_survival1"] = 12
        params["hr"] = 0.75
        params["accrual_time"] = 12
        params["follow_up_time"] = 18
        params["dropout_rate"] = 0.1
        params["alpha"] = 0.05
        params["n1"] = 100
        params["n2"] = 100
        params["sides"] = 2

        # Mock the core analytical function
        with patch('app.components.parallel_rct.analytical_survival.power_survival_analytical') as mock_power_analytical:
            expected_core_result = {"power": 0.85, "events": 50}
            mock_power_analytical.return_value = expected_core_result
            
            result = parallel_rct.calculate_parallel_survival(params)

            mock_power_analytical.assert_called_once_with(
                median_survival1=params["median_survival1"],
                hr=params["hr"],
                n1=params["n1"],
                n2=params["n2"],
                accrual_time=params["accrual_time"],
                follow_up_time=params["follow_up_time"],
                dropout_rate=params["dropout_rate"],
                alpha=params["alpha"],
                sides=params["sides"]
            )
            self.assertIn("power", result)
            self.assertEqual(result["power"], expected_core_result["power"])
            self.assertIn("events", result)
            self.assertEqual(result["events"], expected_core_result["events"])
            self.assertNotIn("error", result, f"Unexpected error: {result.get('error')}")
            self.assertEqual(result.get("median_survival1_param"), params["median_survival1"])
            self.assertEqual(result.get("hr_param"), params["hr"])
            # Using self.assertAlmostEqual for floating point comparison from pytest.approx
            self.assertAlmostEqual(result.get("median_survival2_derived"), params["median_survival1"] / params["hr"])
            self.assertEqual(result.get("method_param"), params["method"])
            self.assertEqual(result.get("hypothesis_type_param"), params["hypothesis_type"])
            self.assertEqual(result.get("calculation_type_param"), params["calculation_type"])
            self.assertEqual(result.get("design_method"), "Parallel RCT")

    def test_calculate_parallel_survival_analytical_non_inferiority_power(self):
        """Test analytical power calculation for survival outcome, non-inferiority."""
        params = self.default_params_survival.copy()
        params["calculation_type"] = "Power"
        params["method"] = "Analytical"
        params["hypothesis_type"] = "Non-Inferiority"
        params["median_survival1"] = 15
        params["non_inferiority_margin_hr"] = 1.3
        params["assumed_true_hr"] = 1.0
        params["accrual_time"] = 12
        params["follow_up_time"] = 24
        params["dropout_rate"] = 0.05
        params["alpha"] = 0.025 # One-sided alpha for NI
        params["n1"] = 150
        params["n2"] = 150
        # 'sides' is effectively 1 for NI analytical, component should handle this

        with patch('app.components.parallel_rct.analytical_survival.power_survival_non_inferiority_analytical') as mock_power_ni_analytical:
            expected_core_result = {"power": 0.90, "events": 70}
            mock_power_ni_analytical.return_value = expected_core_result

            result = parallel_rct.calculate_parallel_survival(params)

            mock_power_ni_analytical.assert_called_once_with(
                median_survival1=params["median_survival1"],
                non_inferiority_margin_hr=params["non_inferiority_margin_hr"],
                assumed_true_hr=params["assumed_true_hr"],
                n1=params["n1"],
                n2=params["n2"],
                accrual_time=params["accrual_time"],
                follow_up_time=params["follow_up_time"],
                dropout_rate=params["dropout_rate"],
                alpha=params["alpha"]
                # sides is not passed to NI core function
            )
            self.assertIn("power", result)
            self.assertEqual(result["power"], expected_core_result["power"])
            self.assertIn("events", result)
            self.assertEqual(result["events"], expected_core_result["events"])
            self.assertNotIn("error", result, f"Unexpected error: {result.get('error')}")
            self.assertEqual(result.get("median_survival1_param"), params["median_survival1"])
            self.assertEqual(result.get("non_inferiority_margin_hr_param"), params["non_inferiority_margin_hr"])
            self.assertEqual(result.get("assumed_true_hr_param"), params["assumed_true_hr"])
            self.assertAlmostEqual(result.get("median_survival2_derived"), params["median_survival1"] / params["assumed_true_hr"])
            self.assertEqual(result.get("method_param"), params["method"])
            self.assertEqual(result.get("hypothesis_type_param"), params["hypothesis_type"])
            self.assertEqual(result.get("calculation_type_param"), params["calculation_type"])
            self.assertEqual(result.get("alpha_param"), params["alpha"])
            self.assertEqual(result.get("design_method"), "Parallel RCT")

    def test_calculate_parallel_survival_analytical_non_inferiority_sample_size(self):
        """Test analytical sample size calculation for survival outcome, non-inferiority."""
        params = self.default_params_survival.copy()
        params["calculation_type"] = "Sample Size"
        params["method"] = "Analytical"
        params["hypothesis_type"] = "Non-Inferiority"
        params["median_survival1"] = 20
        params["non_inferiority_margin_hr"] = 1.25
        params["assumed_true_hr"] = 0.95
        params["accrual_time"] = 18
        params["follow_up_time"] = 30
        params["dropout_rate"] = 0.1
        params["alpha"] = 0.025
        params["power"] = 0.9
        # 'sides' is effectively 1 for NI analytical

        with patch('app.components.parallel_rct.analytical_survival.sample_size_survival_non_inferiority_analytical') as mock_ss_ni_analytical:
            expected_core_result = {"n1": 200, "n2": 200, "total_n": 400, "events": 150}
            mock_ss_ni_analytical.return_value = expected_core_result

            result = parallel_rct.calculate_parallel_survival(params)

            mock_ss_ni_analytical.assert_called_once_with(
                median_survival1=params["median_survival1"],
                non_inferiority_margin_hr=params["non_inferiority_margin_hr"],
                assumed_true_hr=params["assumed_true_hr"],
                power=params["power"],
                alpha=params["alpha"],
                allocation_ratio=params["allocation_ratio"],
                accrual_time=params["accrual_time"],
                follow_up_time=params["follow_up_time"],
                dropout_rate=params["dropout_rate"]
            )
            self.assertIn("n1", result)
            self.assertEqual(result["n1"], expected_core_result["n1"])
            self.assertIn("n2", result)
            self.assertEqual(result["n2"], expected_core_result["n2"])
            self.assertIn("total_n", result)
            self.assertEqual(result["total_n"], expected_core_result["total_n"])
            self.assertIn("events", result)
            self.assertEqual(result["events"], expected_core_result["events"])
            self.assertNotIn("error", result, f"Unexpected error: {result.get('error')}")
            self.assertEqual(result.get("median_survival1_param"), params["median_survival1"])
            self.assertEqual(result.get("non_inferiority_margin_hr_param"), params["non_inferiority_margin_hr"])
            self.assertEqual(result.get("assumed_true_hr_param"), params["assumed_true_hr"])
            self.assertEqual(result.get("power_param"), params["power"])
            self.assertEqual(result.get("method_param"), params["method"])
            self.assertEqual(result.get("hypothesis_type_param"), params["hypothesis_type"])
            self.assertEqual(result.get("calculation_type_param"), params["calculation_type"])
            self.assertEqual(result.get("alpha_param"), params["alpha"])
            self.assertEqual(result.get("design_method"), "Parallel RCT")

    def test_calculate_parallel_survival_analytical_superiority_sample_size(self):
        """Test analytical sample size calculation for survival outcome, superiority, two-sided."""
        params = self.default_params_survival.copy()
        params["calculation_type"] = "Sample Size"
        params["method"] = "Analytical"
        params["hypothesis_type"] = "Superiority"
        params["median_survival1"] = 12
        params["hr"] = 0.75
        params["accrual_time"] = 12
        params["follow_up_time"] = 18
        params["dropout_rate"] = 0.1
        params["alpha"] = 0.05
        params["power"] = 0.8
        params["sides"] = 2

        with patch('app.components.parallel_rct.analytical_survival.sample_size_survival_analytical') as mock_ss_analytical:
            expected_core_result = {"n1": 95, "n2": 95, "total_n": 190, "events": 48}
            mock_ss_analytical.return_value = expected_core_result

            result = parallel_rct.calculate_parallel_survival(params)

            mock_ss_analytical.assert_called_once_with(
                median_survival1=params["median_survival1"],
                hr=params["hr"],
                power=params["power"],
                alpha=params["alpha"],
                allocation_ratio=params["allocation_ratio"],
                accrual_time=params["accrual_time"],
                follow_up_time=params["follow_up_time"],
                dropout_rate=params["dropout_rate"],
                sides=params["sides"]
            )
            self.assertIn("n1", result)
            self.assertEqual(result["n1"], expected_core_result["n1"])
            self.assertIn("n2", result)
            self.assertEqual(result["n2"], expected_core_result["n2"])
            self.assertIn("total_n", result)
            self.assertEqual(result["total_n"], expected_core_result["total_n"])
            self.assertIn("events", result)
            self.assertEqual(result["events"], expected_core_result["events"])
            self.assertNotIn("error", result, f"Unexpected error: {result.get('error')}")
            self.assertEqual(result.get("median_survival1_param"), params["median_survival1"])
            self.assertEqual(result.get("hr_param"), params["hr"])
            self.assertAlmostEqual(result.get("median_survival2_derived"), params["median_survival1"] / params["hr"])
            self.assertEqual(result.get("power_param"), params["power"])
            self.assertEqual(result.get("method_param"), params["method"])
            self.assertEqual(result.get("hypothesis_type_param"), params["hypothesis_type"])
            self.assertEqual(result.get("calculation_type_param"), params["calculation_type"])
            self.assertEqual(result.get("design_method"), "Parallel RCT")

    def test_calculate_parallel_survival_analytical_superiority_mde(self):
        """Test analytical MDE calculation for survival outcome, superiority, two-sided."""
        params = self.default_params_survival.copy()
        params["calculation_type"] = "Minimum Detectable Effect"
        params["method"] = "Analytical"
        params["hypothesis_type"] = "Superiority"
        params["median_survival1"] = 12
        params["n1"] = 100
        params["n2"] = 100
        params["accrual_time"] = 12
        params["follow_up_time"] = 18
        params["dropout_rate"] = 0.1
        params["alpha"] = 0.05
        params["power"] = 0.8
        params["sides"] = 2
        # 'hr' is what we are solving for, so it's not an input for MDE
        if "hr" in params: del params["hr"]

        with patch('app.components.parallel_rct.analytical_survival.min_detectable_effect_survival_analytical') as mock_mde_analytical:
            expected_core_result = {"mde_hr": 0.72, "events": 52}
            mock_mde_analytical.return_value = expected_core_result

            result = parallel_rct.calculate_parallel_survival(params)

            mock_mde_analytical.assert_called_once_with(
                median_survival1=params["median_survival1"],
                n1=params["n1"],
                n2=params["n2"],
                power=params["power"],
                alpha=params["alpha"],
                allocation_ratio=params["allocation_ratio"],
                accrual_time=params["accrual_time"],
                follow_up_time=params["follow_up_time"],
                dropout_rate=params["dropout_rate"],
                sides=params["sides"]
            )
            self.assertIn("mde_hr", result)
            self.assertEqual(result["mde_hr"], expected_core_result["mde_hr"])
            self.assertIn("events", result)
            self.assertEqual(result["events"], expected_core_result["events"])
            self.assertNotIn("error", result, f"Unexpected error: {result.get('error')}")
            self.assertEqual(result.get("median_survival1_param"), params["median_survival1"])
            self.assertAlmostEqual(result.get("median_survival2_derived"), params["median_survival1"] / expected_core_result["mde_hr"])
            self.assertEqual(result.get("power_param"), params["power"])
            self.assertEqual(result.get("method_param"), params["method"])
            self.assertEqual(result.get("hypothesis_type_param"), params["hypothesis_type"])
            self.assertEqual(result.get("calculation_type_param"), params["calculation_type"])
            self.assertEqual(result.get("design_method"), "Parallel RCT")

    @patch('app.components.parallel_rct.simulation_survival.power_survival_non_inferiority_sim')
    def test_calculate_parallel_survival_simulation_non_inferiority_power(self, mock_power_ni_sim):
        """Test simulation power calculation for survival outcome, non-inferiority."""
        params = self.default_params_survival.copy()
        params["calculation_type"] = "Power"
        params["method"] = "Simulation"
        params["hypothesis_type"] = "Non-Inferiority"
        params["median_survival1"] = 15
        params["non_inferiority_margin_hr"] = 1.3
        params["assumed_true_hr"] = 1.0
        params["accrual_time"] = 12
        params["follow_up_time"] = 24
        params["dropout_rate"] = 0.05
        params["alpha"] = 0.025
        params["n1"] = 200
        params["n2"] = 200
        params["nsim"] = 100 # Using default_params_survival value
        params["seed"] = 42  # Using default_params_survival value
        # sides is not relevant for NI sim as it's handled internally

        expected_core_result = {"power": 0.88, "events": 80, "nsim": params["nsim"]}
        mock_power_ni_sim.return_value = expected_core_result

        result = parallel_rct.calculate_parallel_survival(params)

        mock_power_ni_sim.assert_called_once_with(
            median_survival1=params["median_survival1"],
            non_inferiority_margin_hr=params["non_inferiority_margin_hr"],
            assumed_true_hr=params["assumed_true_hr"],
            n1=params["n1"],
            n2=params["n2"],
            accrual_time=params["accrual_time"],
            follow_up_time=params["follow_up_time"],
            dropout_rate=params["dropout_rate"],
            alpha=params["alpha"],
            nsim=params["nsim"],
            seed=params["seed"]
        )
        self.assertIn("power", result)
        self.assertEqual(result["power"], expected_core_result["power"])
        self.assertIn("events", result)
        self.assertEqual(result["events"], expected_core_result["events"])
        self.assertNotIn("error", result, f"Unexpected error: {result.get('error')}")
        self.assertEqual(result.get("median_survival1_param"), params["median_survival1"])
        self.assertEqual(result.get("non_inferiority_margin_hr_param"), params["non_inferiority_margin_hr"])
        self.assertEqual(result.get("assumed_true_hr_param"), params["assumed_true_hr"])
        self.assertAlmostEqual(result.get("median_survival2_derived"), params["median_survival1"] / params["assumed_true_hr"])
        self.assertEqual(result.get("method_param"), params["method"])
        self.assertEqual(result.get("hypothesis_type_param"), params["hypothesis_type"])
        self.assertEqual(result.get("calculation_type_param"), params["calculation_type"])
        self.assertEqual(result.get("alpha_param"), params["alpha"])
        self.assertEqual(result.get("nsim"), params["nsim"])
        self.assertEqual(result.get("design_method"), "Parallel RCT")

    @patch('app.components.parallel_rct.simulation_survival.power_survival_superiority_sim')
    def test_calculate_parallel_survival_simulation_superiority_power(self, mock_power_sup_sim):
        """Test simulation power calculation for survival outcome, superiority, two-sided."""
        params = self.default_params_survival.copy()
        params["calculation_type"] = "Power"
        params["method"] = "Simulation"
        params["hypothesis_type"] = "Superiority"
        params["median_survival1"] = 12
        params["hr"] = 0.75
        params["n1"] = 100
        params["n2"] = 100
        params["accrual_time"] = 12
        params["follow_up_time"] = 18
        params["dropout_rate"] = 0.1
        params["alpha"] = 0.05
        params["sides"] = 2
        params["nsim"] = self.default_params_survival["nsim"]
        params["seed"] = self.default_params_survival["seed"]

        expected_core_result = {"power": 0.83, "events": 51, "nsim": params["nsim"]}
        mock_power_sup_sim.return_value = expected_core_result

        result = parallel_rct.calculate_parallel_survival(params)

        mock_power_sup_sim.assert_called_once_with(
            median_survival1=params["median_survival1"],
            hr=params["hr"],
            n1=params["n1"],
            n2=params["n2"],
            accrual_time=params["accrual_time"],
            follow_up_time=params["follow_up_time"],
            dropout_rate=params["dropout_rate"],
            alpha=params["alpha"],
            sides=params["sides"],
            nsim=params["nsim"],
            seed=params["seed"]
        )
        self.assertIn("power", result)
        self.assertEqual(result["power"], expected_core_result["power"])
        self.assertIn("events", result)
        self.assertEqual(result["events"], expected_core_result["events"])
        self.assertNotIn("error", result, f"Unexpected error: {result.get('error')}")
        self.assertEqual(result.get("median_survival1_param"), params["median_survival1"])
        self.assertEqual(result.get("hr_param"), params["hr"])
        self.assertAlmostEqual(result.get("median_survival2_derived"), params["median_survival1"] / params["hr"])
        self.assertEqual(result.get("method_param"), params["method"])
        self.assertEqual(result.get("hypothesis_type_param"), params["hypothesis_type"])
        self.assertEqual(result.get("calculation_type_param"), params["calculation_type"])
        self.assertEqual(result.get("nsim"), params["nsim"])
        self.assertEqual(result.get("design_method"), "Parallel RCT")

    @patch('app.components.parallel_rct.simulation_survival.min_detectable_effect_survival_superiority_sim')
    def test_calculate_parallel_survival_simulation_superiority_mde(self, mock_mde_sup_sim):
        """Test simulation MDE calculation for survival outcome, superiority, two-sided."""
        params = self.default_params_survival.copy()
        params["calculation_type"] = "Minimum Detectable Effect"
        params["method"] = "Simulation"
        params["hypothesis_type"] = "Superiority"
        params["median_survival1"] = 12
        params["n1"] = 100
        params["n2"] = 100
        params["accrual_time"] = 12
        params["follow_up_time"] = 18
        params["dropout_rate"] = 0.1
        params["alpha"] = 0.05
        params["power"] = 0.8
        params["sides"] = 2
        params["nsim"] = self.default_params_survival["nsim"]
        params["seed"] = self.default_params_survival["seed"]
        if "hr" in params: del params["hr"] # hr is solved for

        expected_core_result = {"mde_hr": 0.71, "events": 53, "nsim": params["nsim"]}
        mock_mde_sup_sim.return_value = expected_core_result

        result = parallel_rct.calculate_parallel_survival(params)

        mock_mde_sup_sim.assert_called_once_with(
            median_survival1=params["median_survival1"],
            n1=params["n1"],
            n2=params["n2"],
            power=params["power"],
            alpha=params["alpha"],
            allocation_ratio=params["allocation_ratio"],
            accrual_time=params["accrual_time"],
            follow_up_time=params["follow_up_time"],
            dropout_rate=params["dropout_rate"],
            sides=params["sides"],
            nsim=params["nsim"],
            seed=params["seed"]
        )
        self.assertIn("mde_hr", result)
        self.assertEqual(result["mde_hr"], expected_core_result["mde_hr"])
        self.assertIn("events", result)
        self.assertEqual(result["events"], expected_core_result["events"])
        self.assertNotIn("error", result, f"Unexpected error: {result.get('error')}")
        self.assertEqual(result.get("median_survival1_param"), params["median_survival1"])
        self.assertAlmostEqual(result.get("median_survival2_derived"), params["median_survival1"] / expected_core_result["mde_hr"])
        self.assertEqual(result.get("power_param"), params["power"])
        self.assertEqual(result.get("method_param"), params["method"])
        self.assertEqual(result.get("hypothesis_type_param"), params["hypothesis_type"])
        self.assertEqual(result.get("calculation_type_param"), params["calculation_type"])
        self.assertEqual(result.get("nsim"), params["nsim"])
        self.assertEqual(result.get("design_method"), "Parallel RCT")

    # --- Tests for calculate_parallel_continuous --- #
    # ... (rest of the code remains the same)

    # --- Tests for calculate_parallel_binary --- #

    # --- Analytical tests for calculate_parallel_binary --- #
    @patch('app.components.parallel_rct.analytical_binary.power_binary_analytical')
    def test_calculate_parallel_binary_analytical_superiority_power_risk_difference(self, mock_power_analytical_binary):
        """Test analytical power calculation for binary outcome, superiority, risk difference."""
        params = self.default_params_binary.copy()
        params["calculation_type"] = "Power"
        params["method"] = "Analytical"
        params["hypothesis_type"] = "Superiority"
        # Specific params for this test
        params["p1"] = 0.6
        params["p2"] = 0.4
        params["n1"] = 100
        params["n2"] = 100
        params["alpha"] = 0.05
        params["sides"] = 2
        params["effect_measure"] = "Risk Difference"

        expected_core_result = {"power": 0.85}
        mock_power_analytical_binary.return_value = expected_core_result

        result = parallel_rct.calculate_parallel_binary(params)

        mock_power_analytical_binary.assert_called_once_with(
            p1=params["p1"],
            p2=params["p2"],
            n1=params["n1"],
            n2=params["n2"],
            alpha=params["alpha"],
            sides=params["sides"],
            effect_measure=params["effect_measure"]
        )

        self.assertIn("power", result)
        self.assertEqual(result["power"], expected_core_result["power"])
        self.assertNotIn("error", result, f"Unexpected error: {result.get('error')}")
        self.assertEqual(result.get("p1_param"), params["p1"])
        self.assertEqual(result.get("p2_param"), params["p2"])
        self.assertEqual(result.get("n1_param"), params["n1"])
        self.assertEqual(result.get("n2_param"), params["n2"])
        self.assertEqual(result.get("alpha_param"), params["alpha"])
        self.assertEqual(result.get("sides_param"), params["sides"])
        self.assertEqual(result.get("effect_measure_param"), params["effect_measure"])
        self.assertEqual(result.get("method_param"), params["method"])
        self.assertEqual(result.get("hypothesis_type_param"), params["hypothesis_type"])
        self.assertEqual(result.get("calculation_type_param"), params["calculation_type"])
        self.assertEqual(result.get("design_method"), "Parallel RCT")
        self.assertEqual(result.get("outcome_type_param"), "Binary")

    @patch('app.components.parallel_rct.analytical_binary.power_binary_analytical')
    def test_calculate_parallel_binary_analytical_superiority_power_risk_ratio(self, mock_power_analytical_binary):
        """Test analytical power calculation for binary outcome, superiority, risk ratio."""
        params = self.default_params_binary.copy()
        params["calculation_type"] = "Power"
        params["method"] = "Analytical"
        params["hypothesis_type"] = "Superiority"
        params["p1"] = 0.6
        params["p2"] = 0.3 # p2 such that RR = 0.3/0.6 = 0.5
        params["n1"] = 80
        params["n2"] = 80
        params["alpha"] = 0.05
        params["sides"] = 2
        params["effect_measure"] = "Risk Ratio"

        expected_core_result = {"power": 0.95}
        mock_power_analytical_binary.return_value = expected_core_result

        result = parallel_rct.calculate_parallel_binary(params)

        mock_power_analytical_binary.assert_called_once_with(
            p1=params["p1"],
            p2=params["p2"],
            n1=params["n1"],
            n2=params["n2"],
            alpha=params["alpha"],
            sides=params["sides"],
            effect_measure=params["effect_measure"]
        )

        self.assertIn("power", result)
        self.assertEqual(result["power"], expected_core_result["power"])
        self.assertNotIn("error", result, f"Unexpected error: {result.get('error')}")
        self.assertEqual(result.get("p1_param"), params["p1"])
        self.assertEqual(result.get("p2_param"), params["p2"])
        self.assertEqual(result.get("n1_param"), params["n1"])
        self.assertEqual(result.get("n2_param"), params["n2"])
        self.assertEqual(result.get("alpha_param"), params["alpha"])
        self.assertEqual(result.get("sides_param"), params["sides"])
        self.assertEqual(result.get("effect_measure_param"), params["effect_measure"])
        self.assertEqual(result.get("method_param"), params["method"])
        self.assertEqual(result.get("hypothesis_type_param"), params["hypothesis_type"])
        self.assertEqual(result.get("calculation_type_param"), params["calculation_type"])
        self.assertEqual(result.get("design_method"), "Parallel RCT")
        self.assertEqual(result.get("outcome_type_param"), "Binary")

    @patch('app.components.parallel_rct.analytical_binary.power_binary_analytical')
    def test_calculate_parallel_binary_analytical_superiority_power_odds_ratio(self, mock_power_analytical_binary):
        """Test analytical power calculation for binary outcome, superiority, odds ratio."""
        params = self.default_params_binary.copy()
        params["calculation_type"] = "Power"
        params["method"] = "Analytical"
        params["hypothesis_type"] = "Superiority"
        params["p1"] = 0.4 # Control proportion
        params["p2"] = 0.2 # Treatment proportion, implies OR = (0.2/(1-0.2)) / (0.4/(1-0.4)) = (0.2/0.8) / (0.4/0.6) = 0.25 / 0.666... = 0.375
        params["n1"] = 150
        params["n2"] = 150
        params["alpha"] = 0.05
        params["sides"] = 2
        params["effect_measure"] = "Odds Ratio"

        expected_core_result = {"power": 0.88}
        mock_power_analytical_binary.return_value = expected_core_result

        result = parallel_rct.calculate_parallel_binary(params)

        mock_power_analytical_binary.assert_called_once_with(
            p1=params["p1"],
            p2=params["p2"],
            n1=params["n1"],
            n2=params["n2"],
            alpha=params["alpha"],
            sides=params["sides"],
            effect_measure=params["effect_measure"]
        )

        self.assertIn("power", result)
        self.assertEqual(result["power"], expected_core_result["power"])
        self.assertNotIn("error", result, f"Unexpected error: {result.get('error')}")
        self.assertEqual(result.get("p1_param"), params["p1"])
        self.assertEqual(result.get("p2_param"), params["p2"])
        self.assertEqual(result.get("n1_param"), params["n1"])
        self.assertEqual(result.get("n2_param"), params["n2"])
        self.assertEqual(result.get("alpha_param"), params["alpha"])
        self.assertEqual(result.get("sides_param"), params["sides"])
        self.assertEqual(result.get("effect_measure_param"), params["effect_measure"])
        self.assertEqual(result.get("method_param"), params["method"])
        self.assertEqual(result.get("hypothesis_type_param"), params["hypothesis_type"])
        self.assertEqual(result.get("calculation_type_param"), params["calculation_type"])
        self.assertEqual(result.get("design_method"), "Parallel RCT")
        self.assertEqual(result.get("outcome_type_param"), "Binary")

    @patch('app.components.parallel_rct.analytical_binary.sample_size_binary_analytical')
    def test_calculate_parallel_binary_analytical_superiority_sample_size_risk_difference(self, mock_ss_analytical_binary):
        """Test analytical sample size calculation for binary outcome, superiority, risk difference."""
        params = self.default_params_binary.copy()
        params["calculation_type"] = "Sample Size"
        params["method"] = "Analytical"
        params["hypothesis_type"] = "Superiority"
        params["p1"] = 0.7
        params["p2"] = 0.5
        params["power"] = 0.90
        params["alpha"] = 0.05
        params["sides"] = 2
        params["allocation_ratio"] = 1.0
        params["effect_measure"] = "Risk Difference"

        expected_core_result = {"n1": 130, "n2": 130, "total_n": 260}
        mock_ss_analytical_binary.return_value = expected_core_result

        result = parallel_rct.calculate_parallel_binary(params)

        mock_ss_analytical_binary.assert_called_once_with(
            p1=params["p1"],
            p2=params["p2"],
            power=params["power"],
            alpha=params["alpha"],
            allocation_ratio=params["allocation_ratio"],
            sides=params["sides"],
            effect_measure=params["effect_measure"]
        )

        self.assertIn("n1", result)
        self.assertEqual(result["n1"], expected_core_result["n1"])
        self.assertIn("n2", result)
        self.assertEqual(result["n2"], expected_core_result["n2"])
        self.assertIn("total_n", result)
        self.assertEqual(result["total_n"], expected_core_result["total_n"])
        self.assertNotIn("error", result, f"Unexpected error: {result.get('error')}")
        self.assertEqual(result.get("p1_param"), params["p1"])
        self.assertEqual(result.get("p2_param"), params["p2"])
        self.assertEqual(result.get("power_param"), params["power"])
        self.assertEqual(result.get("alpha_param"), params["alpha"])
        self.assertEqual(result.get("sides_param"), params["sides"])
        self.assertEqual(result.get("allocation_ratio_param"), params["allocation_ratio"])
        self.assertEqual(result.get("effect_measure_param"), params["effect_measure"])
        self.assertEqual(result.get("method_param"), params["method"])
        self.assertEqual(result.get("hypothesis_type_param"), params["hypothesis_type"])
        self.assertEqual(result.get("calculation_type_param"), params["calculation_type"])
        self.assertEqual(result.get("design_method"), "Parallel RCT")
        self.assertEqual(result.get("outcome_type_param"), "Binary")

    @patch('app.components.parallel_rct.analytical_binary.sample_size_binary_analytical')
    def test_calculate_parallel_binary_analytical_superiority_sample_size_risk_ratio(self, mock_ss_analytical_binary):
        """Test analytical sample size calculation for binary outcome, superiority, risk ratio."""
        params = self.default_params_binary.copy()
        params["calculation_type"] = "Sample Size"
        params["method"] = "Analytical"
        params["hypothesis_type"] = "Superiority"
        params["p1"] = 0.5
        params["p2"] = 0.25 # RR = 0.25/0.5 = 0.5
        params["power"] = 0.85
        params["alpha"] = 0.05
        params["sides"] = 2
        params["allocation_ratio"] = 1.0
        params["effect_measure"] = "Risk Ratio"

        expected_core_result = {"n1": 75, "n2": 75, "total_n": 150}
        mock_ss_analytical_binary.return_value = expected_core_result

        result = parallel_rct.calculate_parallel_binary(params)

        mock_ss_analytical_binary.assert_called_once_with(
            p1=params["p1"],
            p2=params["p2"],
            power=params["power"],
            alpha=params["alpha"],
            allocation_ratio=params["allocation_ratio"],
            sides=params["sides"],
            effect_measure=params["effect_measure"]
        )

        self.assertIn("n1", result)
        self.assertEqual(result["n1"], expected_core_result["n1"])
        self.assertIn("n2", result)
        self.assertEqual(result["n2"], expected_core_result["n2"])
        self.assertIn("total_n", result)
        self.assertEqual(result["total_n"], expected_core_result["total_n"])
        self.assertNotIn("error", result, f"Unexpected error: {result.get('error')}")
        self.assertEqual(result.get("p1_param"), params["p1"])
        self.assertEqual(result.get("p2_param"), params["p2"])
        self.assertEqual(result.get("power_param"), params["power"])
        self.assertEqual(result.get("alpha_param"), params["alpha"])
        self.assertEqual(result.get("sides_param"), params["sides"])
        self.assertEqual(result.get("allocation_ratio_param"), params["allocation_ratio"])
        self.assertEqual(result.get("effect_measure_param"), params["effect_measure"])
        self.assertEqual(result.get("method_param"), params["method"])
        self.assertEqual(result.get("hypothesis_type_param"), params["hypothesis_type"])
        self.assertEqual(result.get("calculation_type_param"), params["calculation_type"])
        self.assertEqual(result.get("design_method"), "Parallel RCT")
        self.assertEqual(result.get("outcome_type_param"), "Binary")

    @patch('app.components.parallel_rct.analytical_binary.sample_size_binary_analytical')
    def test_calculate_parallel_binary_analytical_superiority_sample_size_odds_ratio(self, mock_ss_analytical_binary):
        """Test analytical sample size calculation for binary outcome, superiority, odds ratio."""
        params = self.default_params_binary.copy()
        params["calculation_type"] = "Sample Size"
        params["method"] = "Analytical"
        params["hypothesis_type"] = "Superiority"
        params["p1"] = 0.35
        params["p2"] = 0.15 # OR = (0.15/0.85) / (0.35/0.65) = 0.176 / 0.538 = 0.327
        params["power"] = 0.80
        params["alpha"] = 0.05
        params["sides"] = 2
        params["allocation_ratio"] = 1.0
        params["effect_measure"] = "Odds Ratio"

        expected_core_result = {"n1": 95, "n2": 95, "total_n": 190}
        mock_ss_analytical_binary.return_value = expected_core_result

        result = parallel_rct.calculate_parallel_binary(params)

        mock_ss_analytical_binary.assert_called_once_with(
            p1=params["p1"],
            p2=params["p2"],
            power=params["power"],
            alpha=params["alpha"],
            allocation_ratio=params["allocation_ratio"],
            sides=params["sides"],
            effect_measure=params["effect_measure"]
        )

        self.assertIn("n1", result)
        self.assertEqual(result["n1"], expected_core_result["n1"])
        self.assertIn("n2", result)
        self.assertEqual(result["n2"], expected_core_result["n2"])
        self.assertIn("total_n", result)
        self.assertEqual(result["total_n"], expected_core_result["total_n"])
        self.assertNotIn("error", result, f"Unexpected error: {result.get('error')}")
        self.assertEqual(result.get("p1_param"), params["p1"])
        self.assertEqual(result.get("p2_param"), params["p2"])
        self.assertEqual(result.get("power_param"), params["power"])
        self.assertEqual(result.get("alpha_param"), params["alpha"])
        self.assertEqual(result.get("sides_param"), params["sides"])
        self.assertEqual(result.get("allocation_ratio_param"), params["allocation_ratio"])
        self.assertEqual(result.get("effect_measure_param"), params["effect_measure"])
        self.assertEqual(result.get("method_param"), params["method"])
        self.assertEqual(result.get("hypothesis_type_param"), params["hypothesis_type"])
        self.assertEqual(result.get("calculation_type_param"), params["calculation_type"])
        self.assertEqual(result.get("design_method"), "Parallel RCT")
        self.assertEqual(result.get("outcome_type_param"), "Binary")

    @patch('app.components.parallel_rct.analytical_binary.min_detectable_effect_binary_analytical')
    def test_calculate_parallel_binary_analytical_superiority_mde_risk_difference(self, mock_mde_analytical_binary):
        """Test analytical MDE calculation for binary outcome, superiority, risk difference."""
        params = self.default_params_binary.copy()
        params["calculation_type"] = "Minimum Detectable Effect"
        params["method"] = "Analytical"
        params["hypothesis_type"] = "Superiority"
        params["p1"] = 0.65
        params["n1"] = 100
        params["n2"] = 100
        params["power"] = 0.80
        params["alpha"] = 0.05
        params["sides"] = 2
        params["effect_measure"] = "Risk Difference"

        expected_core_result = {"mde": 0.15, "p2_lower": 0.50, "p2_upper": 0.80} # Assuming p2_upper is not relevant for superiority with RD here or is handled by core
        mock_mde_analytical_binary.return_value = expected_core_result

        result = parallel_rct.calculate_parallel_binary(params)

        mock_mde_analytical_binary.assert_called_once_with(
            p1=params["p1"],
            n1=params["n1"],
            n2=params["n2"],
            power=params["power"],
            alpha=params["alpha"],
            sides=params["sides"],
            effect_measure=params["effect_measure"]
        )
        self.assertIn("mde", result)
        self.assertEqual(result["mde"], expected_core_result["mde"])
        self.assertIn("p2_lower", result)
        self.assertEqual(result["p2_lower"], expected_core_result["p2_lower"])
        # self.assertIn("p2_upper", result) # Depending on core function's return for RD MDE
        # self.assertEqual(result["p2_upper"], expected_core_result["p2_upper"])
        self.assertNotIn("error", result, f"Unexpected error: {result.get('error')}")
        self.assertEqual(result.get("p1_param"), params["p1"])
        self.assertEqual(result.get("n1_param"), params["n1"])
        self.assertEqual(result.get("n2_param"), params["n2"])
        self.assertEqual(result.get("power_param"), params["power"])
        self.assertEqual(result.get("alpha_param"), params["alpha"])
        self.assertEqual(result.get("sides_param"), params["sides"])
        self.assertEqual(result.get("effect_measure_param"), params["effect_measure"])
        self.assertEqual(result.get("method_param"), params["method"])
        self.assertEqual(result.get("hypothesis_type_param"), params["hypothesis_type"])
        self.assertEqual(result.get("calculation_type_param"), params["calculation_type"])
        self.assertEqual(result.get("design_method"), "Parallel RCT")
        self.assertEqual(result.get("outcome_type_param"), "Binary")

    @patch('app.components.parallel_rct.analytical_binary.min_detectable_effect_binary_analytical')
    def test_calculate_parallel_binary_analytical_superiority_mde_risk_ratio(self, mock_mde_analytical_binary):
        """Test analytical MDE calculation for binary outcome, superiority, risk ratio."""
        params = self.default_params_binary.copy()
        params["calculation_type"] = "Minimum Detectable Effect"
        params["method"] = "Analytical"
        params["hypothesis_type"] = "Superiority"
        params["p1"] = 0.50
        params["n1"] = 90
        params["n2"] = 90
        params["power"] = 0.80
        params["alpha"] = 0.05
        params["sides"] = 2
        params["effect_measure"] = "Risk Ratio"

        # For RR, MDE is a ratio. p2_lower/upper would be p1 * MDE_lower and p1 * MDE_upper or p1 / MDE_upper etc.
        # Let's assume the core function returns the MDE (as a ratio) and the corresponding p2 values.
        # If p1=0.5, and MDE (RR) is 0.6, then p2_lower would be approx 0.3.
        # If MDE (RR) is 1.67 (for increase), p2_upper = 0.5 * 1.67 (but we are testing superiority, so expect p2 < p1)
        expected_core_result = {"mde_as_ratio": 0.6, "p2_lower": 0.30, "p2_upper": 0.75} # Example values
        mock_mde_analytical_binary.return_value = expected_core_result

        result = parallel_rct.calculate_parallel_binary(params)

        mock_mde_analytical_binary.assert_called_once_with(
            p1=params["p1"],
            n1=params["n1"],
            n2=params["n2"],
            power=params["power"],
            alpha=params["alpha"],
            sides=params["sides"],
            effect_measure=params["effect_measure"]
        )

        self.assertIn("mde_as_ratio", result) # Key might be 'mde' or 'mde_as_ratio' depending on core impl.
        self.assertEqual(result["mde_as_ratio"], expected_core_result["mde_as_ratio"])
        self.assertIn("p2_lower", result)
        self.assertEqual(result["p2_lower"], expected_core_result["p2_lower"])
        self.assertIn("p2_upper", result)
        self.assertEqual(result["p2_upper"], expected_core_result["p2_upper"])
        self.assertNotIn("error", result, f"Unexpected error: {result.get('error')}")
        self.assertEqual(result.get("p1_param"), params["p1"])
        self.assertEqual(result.get("n1_param"), params["n1"])
        self.assertEqual(result.get("n2_param"), params["n2"])
        self.assertEqual(result.get("power_param"), params["power"])
        self.assertEqual(result.get("alpha_param"), params["alpha"])
        self.assertEqual(result.get("sides_param"), params["sides"])
        self.assertEqual(result.get("effect_measure_param"), params["effect_measure"])
        self.assertEqual(result.get("method_param"), params["method"])
        self.assertEqual(result.get("hypothesis_type_param"), params["hypothesis_type"])
        self.assertEqual(result.get("calculation_type_param"), params["calculation_type"])
        self.assertEqual(result.get("design_method"), "Parallel RCT")
        self.assertEqual(result.get("outcome_type_param"), "Binary")

    @patch('app.components.parallel_rct.analytical_binary.min_detectable_effect_binary_analytical')
    def test_calculate_parallel_binary_analytical_superiority_mde_odds_ratio(self, mock_mde_analytical_binary):
        """Test analytical MDE calculation for binary outcome, superiority, odds ratio."""
        params = self.default_params_binary.copy()
        params["calculation_type"] = "Minimum Detectable Effect"
        params["method"] = "Analytical"
        params["hypothesis_type"] = "Superiority"
        params["p1"] = 0.40 # Control group proportion
        params["n1"] = 120
        params["n2"] = 120
        params["power"] = 0.80
        params["alpha"] = 0.05
        params["sides"] = 2
        params["effect_measure"] = "Odds Ratio"

        # For OR, MDE is a ratio. p2_lower/upper are derived from p1 and the MDE OR.
        # Example: if p1=0.4, and detectable OR (mde_as_ratio) is 0.45 (meaning p2 is lower, better)
        # then p2_lower would be approx 0.23.
        # p2_upper would correspond to an OR of 1/0.45 = 2.22, giving p2 approx 0.59.
        expected_core_result = {"mde_as_ratio": 0.45, "p2_lower": 0.23, "p2_upper": 0.59} # Example values
        mock_mde_analytical_binary.return_value = expected_core_result

        result = parallel_rct.calculate_parallel_binary(params)

        mock_mde_analytical_binary.assert_called_once_with(
            p1=params["p1"],
            n1=params["n1"],
            n2=params["n2"],
            power=params["power"],
            alpha=params["alpha"],
            sides=params["sides"],
            effect_measure=params["effect_measure"]
        )

        self.assertIn("mde_as_ratio", result) # Key for MDE when it's a ratio (RR or OR)
        self.assertEqual(result["mde_as_ratio"], expected_core_result["mde_as_ratio"])
        self.assertIn("p2_lower", result)
        self.assertEqual(result["p2_lower"], expected_core_result["p2_lower"])
        self.assertIn("p2_upper", result)
        self.assertEqual(result["p2_upper"], expected_core_result["p2_upper"])
        self.assertNotIn("error", result, f"Unexpected error: {result.get('error')}")
        self.assertEqual(result.get("p1_param"), params["p1"])
        self.assertEqual(result.get("n1_param"), params["n1"])
        self.assertEqual(result.get("n2_param"), params["n2"])
        self.assertEqual(result.get("power_param"), params["power"])
        self.assertEqual(result.get("alpha_param"), params["alpha"])
        self.assertEqual(result.get("sides_param"), params["sides"])
        self.assertEqual(result.get("effect_measure_param"), params["effect_measure"])
        self.assertEqual(result.get("method_param"), params["method"])
        self.assertEqual(result.get("hypothesis_type_param"), params["hypothesis_type"])
        self.assertEqual(result.get("calculation_type_param"), params["calculation_type"])
        self.assertEqual(result.get("design_method"), "Parallel RCT")
        self.assertEqual(result.get("outcome_type_param"), "Binary")

    # --- Analytical Non-Inferiority Tests for Binary Outcomes ---
    @patch('app.components.parallel_rct.analytical_binary.power_binary_ni_analytical')
    def test_calculate_parallel_binary_analytical_non_inferiority_power_risk_difference(self, mock_power_ni_analytical_binary):
        """Test analytical power for non-inferiority, binary outcome, risk difference."""
        params = self.default_params_binary.copy()
        params["calculation_type"] = "Power"
        params["method"] = "Analytical"
        params["hypothesis_type"] = "Non-Inferiority"
        params["p1"] = 0.60  # Proportion in control group (e.g., success rate)
        params["p2"] = 0.55  # Proportion in treatment group (e.g., success rate)
        params["n1"] = 200
        params["n2"] = 200
        params["alpha"] = 0.025 # Typically one-sided alpha for NI
        params["sides"] = 1 # Non-inferiority is usually one-sided
        params["effect_measure"] = "Risk Difference"
        params["non_inferiority_margin"] = 0.10 # Treatment is non-inferior if p2 is not worse than p1 by more than 0.10

        expected_core_result = {"power": 0.85}
        mock_power_ni_analytical_binary.return_value = expected_core_result

        result = parallel_rct.calculate_parallel_binary(params)

        mock_power_ni_analytical_binary.assert_called_once_with(
            p1=params["p1"],
            p2=params["p2"],
            n1=params["n1"],
            n2=params["n2"],
            alpha=params["alpha"],
            sides=params["sides"], # Core function might always assume 1 for NI, or use this value
            non_inferiority_margin=params["non_inferiority_margin"],
            effect_measure=params["effect_measure"]
        )

        self.assertIn("power", result)
        self.assertEqual(result["power"], expected_core_result["power"])
        self.assertNotIn("error", result, f"Unexpected error: {result.get('error')}")
        self.assertEqual(result.get("p1_param"), params["p1"])
        self.assertEqual(result.get("p2_param"), params["p2"])
        self.assertEqual(result.get("n1_param"), params["n1"])
        self.assertEqual(result.get("n2_param"), params["n2"])
        self.assertEqual(result.get("alpha_param"), params["alpha"])
        self.assertEqual(result.get("sides_param"), params["sides"])
        self.assertEqual(result.get("effect_measure_param"), params["effect_measure"])
        self.assertEqual(result.get("non_inferiority_margin_param"), params["non_inferiority_margin"])
        self.assertEqual(result.get("method_param"), params["method"])
        self.assertEqual(result.get("hypothesis_type_param"), params["hypothesis_type"])
        self.assertEqual(result.get("calculation_type_param"), params["calculation_type"])
        self.assertEqual(result.get("design_method"), "Parallel RCT")
        self.assertEqual(result.get("outcome_type_param"), "Binary")

    @patch('app.components.parallel_rct.analytical_binary.power_binary_ni_analytical')
    def test_calculate_parallel_binary_analytical_non_inferiority_power_risk_ratio(self, mock_power_ni_analytical_binary):
        """Test analytical power for non-inferiority, binary outcome, risk ratio."""
        params = self.default_params_binary.copy()
        params["calculation_type"] = "Power"
        params["method"] = "Analytical"
        params["hypothesis_type"] = "Non-Inferiority"
        params["p1"] = 0.70  # Control success rate
        params["p2"] = 0.68  # Treatment success rate (p2/p1 = 0.68/0.70 = 0.971)
        params["n1"] = 300
        params["n2"] = 300
        params["alpha"] = 0.025
        params["sides"] = 1
        params["effect_measure"] = "Risk Ratio"
        # Treatment is non-inferior if p2/p1 is not less than this margin (e.g. RR_p2/p1 >= margin)
        # Or, if margin is for p1/p2 (how much better p1 can be), margin would be > 1.
        # Assuming margin is lower bound for p2/p1, so margin < 1.
        params["non_inferiority_margin"] = 0.9  # e.g. p2/p1 must be >= 0.9

        expected_core_result = {"power": 0.92}
        mock_power_ni_analytical_binary.return_value = expected_core_result

        result = parallel_rct.calculate_parallel_binary(params)

        mock_power_ni_analytical_binary.assert_called_once_with(
            p1=params["p1"],
            p2=params["p2"],
            n1=params["n1"],
            n2=params["n2"],
            alpha=params["alpha"],
            sides=params["sides"],
            non_inferiority_margin=params["non_inferiority_margin"],
            effect_measure=params["effect_measure"]
        )

        self.assertIn("power", result)
        self.assertEqual(result["power"], expected_core_result["power"])
        self.assertNotIn("error", result, f"Unexpected error: {result.get('error')}")
        self.assertEqual(result.get("p1_param"), params["p1"])
        self.assertEqual(result.get("p2_param"), params["p2"])
        self.assertEqual(result.get("n1_param"), params["n1"])
        self.assertEqual(result.get("n2_param"), params["n2"])
        self.assertEqual(result.get("alpha_param"), params["alpha"])
        self.assertEqual(result.get("sides_param"), params["sides"])
        self.assertEqual(result.get("effect_measure_param"), params["effect_measure"])
        self.assertEqual(result.get("non_inferiority_margin_param"), params["non_inferiority_margin"])
        self.assertEqual(result.get("method_param"), params["method"])
        self.assertEqual(result.get("hypothesis_type_param"), params["hypothesis_type"])
        self.assertEqual(result.get("calculation_type_param"), params["calculation_type"])
        self.assertEqual(result.get("design_method"), "Parallel RCT")
        self.assertEqual(result.get("outcome_type_param"), "Binary")

    @patch('app.components.parallel_rct.analytical_binary.power_binary_ni_analytical')
    def test_calculate_parallel_binary_analytical_non_inferiority_power_odds_ratio(self, mock_power_ni_analytical_binary):
        """Test analytical power for non-inferiority, binary outcome, odds ratio."""
        params = self.default_params_binary.copy()
        params["calculation_type"] = "Power"
        params["method"] = "Analytical"
        params["hypothesis_type"] = "Non-Inferiority"
        params["p1"] = 0.40  # Control success rate. Odds = 0.4/0.6 = 0.667
        params["p2"] = 0.38  # Treatment success rate. Odds = 0.38/0.62 = 0.613. OR_p2/p1 = 0.613/0.667 = 0.919
        params["n1"] = 500
        params["n2"] = 500
        params["alpha"] = 0.025
        params["sides"] = 1
        params["effect_measure"] = "Odds Ratio"
        params["non_inferiority_margin"] = 0.7 # OR_t/c must be >= 0.7

        expected_core_result = {"power": 0.82}
        mock_power_ni_analytical_binary.return_value = expected_core_result

        result = parallel_rct.calculate_parallel_binary(params)

        mock_power_ni_analytical_binary.assert_called_once_with(
            p1=params["p1"],
            p2=params["p2"],
            n1=params["n1"],
            n2=params["n2"],
            alpha=params["alpha"],
            sides=params["sides"],
            non_inferiority_margin=params["non_inferiority_margin"],
            effect_measure=params["effect_measure"]
        )

        self.assertIn("power", result)
        self.assertEqual(result["power"], expected_core_result["power"])
        self.assertNotIn("error", result, f"Unexpected error: {result.get('error')}")
        self.assertEqual(result.get("p1_param"), params["p1"])
        self.assertEqual(result.get("p2_param"), params["p2"])
        self.assertEqual(result.get("n1_param"), params["n1"])
        self.assertEqual(result.get("n2_param"), params["n2"])
        self.assertEqual(result.get("alpha_param"), params["alpha"])
        self.assertEqual(result.get("sides_param"), params["sides"])
        self.assertEqual(result.get("effect_measure_param"), params["effect_measure"])
        self.assertEqual(result.get("non_inferiority_margin_param"), params["non_inferiority_margin"])
        self.assertEqual(result.get("method_param"), params["method"])
        self.assertEqual(result.get("hypothesis_type_param"), params["hypothesis_type"])
        self.assertEqual(result.get("calculation_type_param"), params["calculation_type"])
        self.assertEqual(result.get("design_method"), "Parallel RCT")
        self.assertEqual(result.get("outcome_type_param"), "Binary")

    @patch('app.components.parallel_rct.analytical_binary.sample_size_binary_ni_analytical')
    def test_calculate_parallel_binary_analytical_non_inferiority_sample_size_risk_difference(self, mock_ss_ni_analytical_binary):
        """Test analytical sample size for non-inferiority, binary outcome, risk difference."""
        params = self.default_params_binary.copy()
        params["calculation_type"] = "Sample Size"
        params["method"] = "Analytical"
        params["hypothesis_type"] = "Non-Inferiority"
        params["p1"] = 0.80 # Control success rate
        params["p2"] = 0.78 # Expected treatment success rate
        params["power"] = 0.90
        params["alpha"] = 0.025
        params["sides"] = 1
        params["allocation_ratio"] = 1.0
        params["effect_measure"] = "Risk Difference"
        params["non_inferiority_margin"] = 0.05 # Max acceptable difference p1-p2

        expected_core_result = {"n1": 1050, "n2": 1050, "total_n": 2100}
        mock_ss_ni_analytical_binary.return_value = expected_core_result

        result = parallel_rct.calculate_parallel_binary(params)

        mock_ss_ni_analytical_binary.assert_called_once_with(
            p1=params["p1"],
            p2=params["p2"],
            power=params["power"],
            alpha=params["alpha"],
            allocation_ratio=params["allocation_ratio"],
            sides=params["sides"],
            non_inferiority_margin=params["non_inferiority_margin"],
            effect_measure=params["effect_measure"]
        )

        self.assertIn("n1", result)
        self.assertEqual(result["n1"], expected_core_result["n1"])
        self.assertIn("n2", result)
        self.assertEqual(result["n2"], expected_core_result["n2"])
        self.assertIn("total_n", result)
        self.assertEqual(result["total_n"], expected_core_result["total_n"])
        self.assertNotIn("error", result, f"Unexpected error: {result.get('error')}")
        self.assertEqual(result.get("p1_param"), params["p1"])
        self.assertEqual(result.get("p2_param"), params["p2"])
        self.assertEqual(result.get("power_param"), params["power"])
        self.assertEqual(result.get("alpha_param"), params["alpha"])
        self.assertEqual(result.get("sides_param"), params["sides"])
        self.assertEqual(result.get("allocation_ratio_param"), params["allocation_ratio"])
        self.assertEqual(result.get("effect_measure_param"), params["effect_measure"])
        self.assertEqual(result.get("non_inferiority_margin_param"), params["non_inferiority_margin"])
        self.assertEqual(result.get("method_param"), params["method"])
        self.assertEqual(result.get("hypothesis_type_param"), params["hypothesis_type"])
        self.assertEqual(result.get("calculation_type_param"), params["calculation_type"])
        self.assertEqual(result.get("design_method"), "Parallel RCT")
        self.assertEqual(result.get("outcome_type_param"), "Binary")

    @patch('app.components.parallel_rct.analytical_binary.sample_size_binary_ni_analytical')
    def test_calculate_parallel_binary_analytical_non_inferiority_sample_size_risk_ratio(self, mock_ss_ni_analytical_binary):
        """Test analytical sample size for non-inferiority, binary outcome, risk ratio."""
        params = self.default_params_binary.copy()
        params["calculation_type"] = "Sample Size"
        params["method"] = "Analytical"
        params["hypothesis_type"] = "Non-Inferiority"
        params["p1"] = 0.65 # Control success rate
        params["p2"] = 0.60 # Expected treatment success rate (p2/p1 = 0.60/0.65 = 0.923)
        params["power"] = 0.80
        params["alpha"] = 0.025
        params["sides"] = 1
        params["allocation_ratio"] = 1.0
        params["effect_measure"] = "Risk Ratio"
        params["non_inferiority_margin"] = 0.85 # p2/p1 must be >= 0.85

        expected_core_result = {"n1": 450, "n2": 450, "total_n": 900}
        mock_ss_ni_analytical_binary.return_value = expected_core_result

        result = parallel_rct.calculate_parallel_binary(params)

        mock_ss_ni_analytical_binary.assert_called_once_with(
            p1=params["p1"],
            p2=params["p2"],
            power=params["power"],
            alpha=params["alpha"],
            allocation_ratio=params["allocation_ratio"],
            sides=params["sides"],
            non_inferiority_margin=params["non_inferiority_margin"],
            effect_measure=params["effect_measure"]
        )

        self.assertIn("n1", result)
        self.assertEqual(result["n1"], expected_core_result["n1"])
        self.assertIn("n2", result)
        self.assertEqual(result["n2"], expected_core_result["n2"])
        self.assertIn("total_n", result)
        self.assertEqual(result["total_n"], expected_core_result["total_n"])
        self.assertNotIn("error", result, f"Unexpected error: {result.get('error')}")
        self.assertEqual(result.get("p1_param"), params["p1"])
        self.assertEqual(result.get("p2_param"), params["p2"])
        self.assertEqual(result.get("power_param"), params["power"])
        self.assertEqual(result.get("alpha_param"), params["alpha"])
        self.assertEqual(result.get("sides_param"), params["sides"])
        self.assertEqual(result.get("allocation_ratio_param"), params["allocation_ratio"])
        self.assertEqual(result.get("effect_measure_param"), params["effect_measure"])
        self.assertEqual(result.get("non_inferiority_margin_param"), params["non_inferiority_margin"])
        self.assertEqual(result.get("method_param"), params["method"])
        self.assertEqual(result.get("hypothesis_type_param"), params["hypothesis_type"])
        self.assertEqual(result.get("calculation_type_param"), params["calculation_type"])
        self.assertEqual(result.get("design_method"), "Parallel RCT")
        self.assertEqual(result.get("outcome_type_param"), "Binary")

    @patch('app.components.parallel_rct.analytical_binary.sample_size_binary_ni_analytical')
    def test_calculate_parallel_binary_analytical_non_inferiority_sample_size_odds_ratio(self, mock_ss_ni_analytical_binary):
        """Test analytical sample size for non-inferiority, binary outcome, odds ratio."""
        params = self.default_params_binary.copy()
        params["calculation_type"] = "Sample Size"
        params["method"] = "Analytical"
        params["hypothesis_type"] = "Non-Inferiority"
        params["p1"] = 0.30 # Control success rate. Odds_c = 0.3/0.7 = 0.428
        params["p2"] = 0.32 # Expected treatment success rate. Odds_t = 0.32/0.68 = 0.470. OR_t/c = 1.098
        params["power"] = 0.85
        params["alpha"] = 0.025
        params["sides"] = 1
        params["allocation_ratio"] = 1.0
        params["effect_measure"] = "Odds Ratio"
        params["non_inferiority_margin"] = 0.7 # OR_t/c must be >= 0.7

        expected_core_result = {"n1": 380, "n2": 380, "total_n": 760}
        mock_ss_ni_analytical_binary.return_value = expected_core_result

        result = parallel_rct.calculate_parallel_binary(params)

        mock_ss_ni_analytical_binary.assert_called_once_with(
            p1=params["p1"],
            p2=params["p2"],
            power=params["power"],
            alpha=params["alpha"],
            allocation_ratio=params["allocation_ratio"],
            sides=params["sides"],
            non_inferiority_margin=params["non_inferiority_margin"],
            effect_measure=params["effect_measure"]
        )

        self.assertIn("n1", result)
        self.assertEqual(result["n1"], expected_core_result["n1"])
        self.assertIn("n2", result)
        self.assertEqual(result["n2"], expected_core_result["n2"])
        self.assertIn("total_n", result)
        self.assertEqual(result["total_n"], expected_core_result["total_n"])
        self.assertNotIn("error", result, f"Unexpected error: {result.get('error')}")
        self.assertEqual(result.get("p1_param"), params["p1"])
        self.assertEqual(result.get("p2_param"), params["p2"])
        self.assertEqual(result.get("power_param"), params["power"])
        self.assertEqual(result.get("alpha_param"), params["alpha"])
        self.assertEqual(result.get("sides_param"), params["sides"])
        self.assertEqual(result.get("allocation_ratio_param"), params["allocation_ratio"])
        self.assertEqual(result.get("effect_measure_param"), params["effect_measure"])
        self.assertEqual(result.get("non_inferiority_margin_param"), params["non_inferiority_margin"])
        self.assertEqual(result.get("method_param"), params["method"])
        self.assertEqual(result.get("hypothesis_type_param"), params["hypothesis_type"])
        self.assertEqual(result.get("calculation_type_param"), params["calculation_type"])
        self.assertEqual(result.get("design_method"), "Parallel RCT")
        self.assertEqual(result.get("outcome_type_param"), "Binary")

    @patch('app.components.parallel_rct.analytical_binary.min_detectable_margin_binary_ni_analytical') # Assuming a specific NI margin function
    def test_calculate_parallel_binary_analytical_non_inferiority_mde_risk_difference(self, mock_mde_ni_analytical_binary):
        """Test analytical MDE (margin) for non-inferiority, binary outcome, risk difference."""
        params = self.default_params_binary.copy()
        params["calculation_type"] = "Minimum Detectable Effect" # Or "Non-Inferiority Margin"
        params["method"] = "Analytical"
        params["hypothesis_type"] = "Non-Inferiority"
        params["p1"] = 0.75 # Control success rate
        params["p2"] = 0.70 # Assumed treatment success rate (for context, not direct input for margin calc usually)
        params["n1"] = 250
        params["n2"] = 250
        params["power"] = 0.80
        params["alpha"] = 0.025
        params["sides"] = 1
        params["effect_measure"] = "Risk Difference"

        # The result here is the largest acceptable risk difference (p1-p2) to still claim NI
        expected_core_result = {"non_inferiority_margin": 0.08}
        mock_mde_ni_analytical_binary.return_value = expected_core_result

        # For MDE/Margin calculation in NI, p2 is often not a direct input to the core margin calculation function.
        # The function calculates the margin based on p1, N, power, alpha.
        # We'll remove p2 if the core function doesn't expect it for margin calculation.
        core_params_call = {
            "p1": params["p1"],
            "n1": params["n1"],
            "n2": params["n2"],
            "power": params["power"],
            "alpha": params["alpha"],
            "sides": params["sides"],
            "effect_measure": params["effect_measure"]
        }
        # If the core function for NI margin *does* use p2 (e.g. to calculate margin around an expected p2),
        # then p2 should be in core_params_call. For now, assume it doesn't for RD margin.
        # params.pop("p2", None) # remove p2 if not needed by core for this calc type

        result = parallel_rct.calculate_parallel_binary(params)

        mock_mde_ni_analytical_binary.assert_called_once_with(**core_params_call)

        self.assertIn("non_inferiority_margin", result)
        self.assertEqual(result["non_inferiority_margin"], expected_core_result["non_inferiority_margin"])
        self.assertNotIn("error", result, f"Unexpected error: {result.get('error')}")
        self.assertEqual(result.get("p1_param"), params["p1"])
        # self.assertEqual(result.get("p2_param"), params.get("p2")) # p2 might not be a primary output here
        self.assertEqual(result.get("n1_param"), params["n1"])
        self.assertEqual(result.get("n2_param"), params["n2"])
        self.assertEqual(result.get("power_param"), params["power"])
        self.assertEqual(result.get("alpha_param"), params["alpha"])
        self.assertEqual(result.get("sides_param"), params["sides"])
        self.assertEqual(result.get("effect_measure_param"), params["effect_measure"])
        self.assertEqual(result.get("method_param"), params["method"])
        self.assertEqual(result.get("hypothesis_type_param"), params["hypothesis_type"])
        self.assertEqual(result.get("calculation_type_param"), params["calculation_type"])
        self.assertEqual(result.get("design_method"), "Parallel RCT")
        self.assertEqual(result.get("outcome_type_param"), "Binary")

    @patch('app.components.parallel_rct.analytical_binary.min_detectable_margin_binary_ni_analytical')
    def test_calculate_parallel_binary_analytical_non_inferiority_mde_risk_ratio(self, mock_mde_ni_analytical_binary):
        """Test analytical MDE (margin) for non-inferiority, binary outcome, risk ratio."""
        params = self.default_params_binary.copy()
        params["calculation_type"] = "Minimum Detectable Effect"
        params["method"] = "Analytical"
        params["hypothesis_type"] = "Non-Inferiority"
        params["p1"] = 0.80 # Control success rate
        # p2 is not typically a direct input for margin calculation but included for context
        params["p2"] = 0.78 # Assumed treatment success rate for context
        params["n1"] = 300
        params["n2"] = 300
        params["power"] = 0.90
        params["alpha"] = 0.025
        params["sides"] = 1
        params["effect_measure"] = "Risk Ratio"

        # The result is the smallest acceptable p2/p1 ratio to still claim NI
        expected_core_result = {"non_inferiority_margin": 0.92} # Example: p2/p1 must be >= 0.92
        mock_mde_ni_analytical_binary.return_value = expected_core_result

        core_params_call = {
            "p1": params["p1"],
            "n1": params["n1"],
            "n2": params["n2"],
            "power": params["power"],
            "alpha": params["alpha"],
            "sides": params["sides"],
            "effect_measure": params["effect_measure"]
        }
        # params.pop("p2", None) # p2 is not passed to the core margin calculation function

        result = parallel_rct.calculate_parallel_binary(params)

        mock_mde_ni_analytical_binary.assert_called_once_with(**core_params_call)

        self.assertIn("non_inferiority_margin", result)
        self.assertEqual(result["non_inferiority_margin"], expected_core_result["non_inferiority_margin"])
        self.assertNotIn("error", result, f"Unexpected error: {result.get('error')}")
        self.assertEqual(result.get("p1_param"), params["p1"])
        # self.assertEqual(result.get("p2_param"), params.get("p2")) # p2 is context, not primary output of margin calc
        self.assertEqual(result.get("n1_param"), params["n1"])
        self.assertEqual(result.get("n2_param"), params["n2"])
        self.assertEqual(result.get("power_param"), params["power"])
        self.assertEqual(result.get("alpha_param"), params["alpha"])
        self.assertEqual(result.get("sides_param"), params["sides"])
        self.assertEqual(result.get("effect_measure_param"), params["effect_measure"])
        self.assertEqual(result.get("method_param"), params["method"])
        self.assertEqual(result.get("hypothesis_type_param"), params["hypothesis_type"])
        self.assertEqual(result.get("calculation_type_param"), params["calculation_type"])
        self.assertEqual(result.get("design_method"), "Parallel RCT")
        self.assertEqual(result.get("outcome_type_param"), "Binary")

    @patch('app.components.parallel_rct.analytical_binary.min_detectable_margin_binary_ni_analytical')
    def test_calculate_parallel_binary_analytical_non_inferiority_mde_odds_ratio(self, mock_mde_ni_analytical_binary):
        """Test analytical MDE (margin) for non-inferiority, binary outcome, odds ratio."""
        params = self.default_params_binary.copy()
        params["calculation_type"] = "Minimum Detectable Effect"
        params["method"] = "Analytical"
        params["hypothesis_type"] = "Non-Inferiority"
        params["p1"] = 0.40 # Control success rate
        # p2 is not typically a direct input for margin calculation but included for context
        params["p2"] = 0.38 # Assumed treatment success rate for context
        params["n1"] = 400
        params["n2"] = 400
        params["power"] = 0.80
        params["alpha"] = 0.05 # Using 0.05 for variety
        params["sides"] = 1
        params["effect_measure"] = "Odds Ratio"

        # The result is the smallest acceptable OR_t/c to still claim NI
        expected_core_result = {"non_inferiority_margin": 0.75} # Example: OR_t/c must be >= 0.75
        mock_mde_ni_analytical_binary.return_value = expected_core_result

        core_params_call = {
            "p1": params["p1"],
            "n1": params["n1"],
            "n2": params["n2"],
            "power": params["power"],
            "alpha": params["alpha"],
            "sides": params["sides"],
            "effect_measure": params["effect_measure"]
        }
        # params.pop("p2", None) # p2 is not passed to the core margin calculation function

        result = parallel_rct.calculate_parallel_binary(params)

        mock_mde_ni_analytical_binary.assert_called_once_with(**core_params_call)

        self.assertIn("non_inferiority_margin", result)
        self.assertEqual(result["non_inferiority_margin"], expected_core_result["non_inferiority_margin"])
        self.assertNotIn("error", result, f"Unexpected error: {result.get('error')}")
        self.assertEqual(result.get("p1_param"), params["p1"])
        # self.assertEqual(result.get("p2_param"), params.get("p2")) # p2 is context, not primary output
        self.assertEqual(result.get("n1_param"), params["n1"])
        self.assertEqual(result.get("n2_param"), params["n2"])
        self.assertEqual(result.get("power_param"), params["power"])
        self.assertEqual(result.get("alpha_param"), params["alpha"])
        self.assertEqual(result.get("sides_param"), params["sides"])
        self.assertEqual(result.get("effect_measure_param"), params["effect_measure"])
        self.assertEqual(result.get("method_param"), params["method"])
        self.assertEqual(result.get("hypothesis_type_param"), params["hypothesis_type"])
        self.assertEqual(result.get("calculation_type_param"), params["calculation_type"])
        self.assertEqual(result.get("design_method"), "Parallel RCT")
        self.assertEqual(result.get("outcome_type_param"), "Binary")

    # --- Simulation Superiority Tests --- #
    @patch('app.components.parallel_rct.simulation_binary.power_binary_sim')
    def test_calculate_parallel_binary_simulation_superiority_power_risk_difference(self, mock_power_sim_binary):
        """Test simulation power for superiority, binary outcome, risk difference."""
        params = self.default_params_binary.copy()
        params["calculation_type"] = "Power"
        params["method"] = "Simulation"
        params["hypothesis_type"] = "Superiority"
        params["p1"] = 0.20
        params["p2"] = 0.30
        params["n1"] = 150
        params["n2"] = 150
        params["alpha"] = 0.05
        params["sides"] = 2
        params["effect_measure"] = "Risk Difference"
        params["num_simulations"] = 1000
        params["seed"] = 42

        expected_core_result = {"power": 0.88}
        mock_power_sim_binary.return_value = expected_core_result

        result = parallel_rct.calculate_parallel_binary(params)

        mock_power_sim_binary.assert_called_once_with(
            p1=params["p1"],
            p2=params["p2"],
            n1=params["n1"],
            n2=params["n2"],
            alpha=params["alpha"],
            sides=params["sides"],
            effect_measure=params["effect_measure"],
            num_simulations=params["num_simulations"],
            seed=params["seed"]
        )

        self.assertIn("power", result)
        self.assertEqual(result["power"], expected_core_result["power"])
        self.assertNotIn("error", result, f"Unexpected error: {result.get('error')}")
        self.assertEqual(result.get("p1_param"), params["p1"])
        self.assertEqual(result.get("p2_param"), params["p2"])
        self.assertEqual(result.get("n1_param"), params["n1"])
        self.assertEqual(result.get("n2_param"), params["n2"])
        self.assertEqual(result.get("alpha_param"), params["alpha"])
        self.assertEqual(result.get("sides_param"), params["sides"])
        self.assertEqual(result.get("effect_measure_param"), params["effect_measure"])
        self.assertEqual(result.get("num_simulations_param"), params["num_simulations"])
        self.assertEqual(result.get("seed_param"), params["seed"])
        self.assertEqual(result.get("method_param"), params["method"])
        self.assertEqual(result.get("hypothesis_type_param"), params["hypothesis_type"])
        self.assertEqual(result.get("calculation_type_param"), params["calculation_type"])
        self.assertEqual(result.get("design_method"), "Parallel RCT")
        self.assertEqual(result.get("outcome_type_param"), "Binary")

    @patch('app.components.parallel_rct.simulation_binary.power_binary_sim')
    def test_calculate_parallel_binary_simulation_superiority_power_risk_ratio(self, mock_power_sim_binary):
        """Test simulation power for superiority, binary outcome, risk ratio."""
        params = self.default_params_binary.copy()
        params["calculation_type"] = "Power"
        params["method"] = "Simulation"
        params["hypothesis_type"] = "Superiority"
        params["p1"] = 0.15 # Control event rate
        params["p2"] = 0.25 # Treatment event rate (p2/p1 = 0.25/0.15 = 1.667)
        params["n1"] = 200
        params["n2"] = 200
        params["alpha"] = 0.05
        params["sides"] = 2
        params["effect_measure"] = "Risk Ratio"
        params["num_simulations"] = 1200
        params["seed"] = 123

        expected_core_result = {"power": 0.91}
        mock_power_sim_binary.return_value = expected_core_result

        result = parallel_rct.calculate_parallel_binary(params)

        mock_power_sim_binary.assert_called_once_with(
            p1=params["p1"],
            p2=params["p2"],
            n1=params["n1"],
            n2=params["n2"],
            alpha=params["alpha"],
            sides=params["sides"],
            effect_measure=params["effect_measure"],
            num_simulations=params["num_simulations"],
            seed=params["seed"]
        )

        self.assertIn("power", result)
        self.assertEqual(result["power"], expected_core_result["power"])
        self.assertNotIn("error", result, f"Unexpected error: {result.get('error')}")
        self.assertEqual(result.get("p1_param"), params["p1"])
        self.assertEqual(result.get("p2_param"), params["p2"])
        self.assertEqual(result.get("n1_param"), params["n1"])
        self.assertEqual(result.get("n2_param"), params["n2"])
        self.assertEqual(result.get("alpha_param"), params["alpha"])
        self.assertEqual(result.get("sides_param"), params["sides"])
        self.assertEqual(result.get("effect_measure_param"), params["effect_measure"])
        self.assertEqual(result.get("num_simulations_param"), params["num_simulations"])
        self.assertEqual(result.get("seed_param"), params["seed"])
        self.assertEqual(result.get("method_param"), params["method"])
        self.assertEqual(result.get("hypothesis_type_param"), params["hypothesis_type"])
        self.assertEqual(result.get("calculation_type_param"), params["calculation_type"])
        self.assertEqual(result.get("design_method"), "Parallel RCT")
        self.assertEqual(result.get("outcome_type_param"), "Binary")

    @patch('app.components.parallel_rct.simulation_binary.power_binary_sim')
    def test_calculate_parallel_binary_simulation_superiority_power_odds_ratio(self, mock_power_sim_binary):
        """Test simulation power for superiority, binary outcome, odds ratio."""
        params = self.default_params_binary.copy()
        params["calculation_type"] = "Power"
        params["method"] = "Simulation"
        params["hypothesis_type"] = "Superiority"
        params["p1"] = 0.40 # Control event rate. Odds_c = 0.4/0.6 = 0.667
        params["p2"] = 0.55 # Treatment event rate. Odds_t = 0.55/0.45 = 1.222. OR_t/c = 1.222/0.667 = 1.832
        params["n1"] = 100
        params["n2"] = 100
        params["alpha"] = 0.05
        params["sides"] = 2
        params["effect_measure"] = "Odds Ratio"
        params["num_simulations"] = 1500
        params["seed"] = 500

        expected_core_result = {"power": 0.79}
        mock_power_sim_binary.return_value = expected_core_result

        result = parallel_rct.calculate_parallel_binary(params)

        mock_power_sim_binary.assert_called_once_with(
            p1=params["p1"],
            p2=params["p2"],
            n1=params["n1"],
            n2=params["n2"],
            alpha=params["alpha"],
            sides=params["sides"],
            effect_measure=params["effect_measure"],
            num_simulations=params["num_simulations"],
            seed=params["seed"]
        )

        self.assertIn("power", result)
        self.assertEqual(result["power"], expected_core_result["power"])
        self.assertNotIn("error", result, f"Unexpected error: {result.get('error')}")
        self.assertEqual(result.get("p1_param"), params["p1"])
        self.assertEqual(result.get("p2_param"), params["p2"])
        self.assertEqual(result.get("n1_param"), params["n1"])
        self.assertEqual(result.get("n2_param"), params["n2"])
        self.assertEqual(result.get("alpha_param"), params["alpha"])
        self.assertEqual(result.get("sides_param"), params["sides"])
        self.assertEqual(result.get("effect_measure_param"), params["effect_measure"])
        self.assertEqual(result.get("num_simulations_param"), params["num_simulations"])
        self.assertEqual(result.get("seed_param"), params["seed"])
        self.assertEqual(result.get("method_param"), params["method"])
        self.assertEqual(result.get("hypothesis_type_param"), params["hypothesis_type"])
        self.assertEqual(result.get("calculation_type_param"), params["calculation_type"])
        self.assertEqual(result.get("design_method"), "Parallel RCT")
        self.assertEqual(result.get("outcome_type_param"), "Binary")

    @patch('app.components.parallel_rct.simulation_binary.sample_size_binary_sim')
    def test_calculate_parallel_binary_simulation_superiority_sample_size_risk_difference(self, mock_ss_sim_binary):
        """Test simulation sample size for superiority, binary outcome, risk difference."""
        params = self.default_params_binary.copy()
        params["calculation_type"] = "Sample Size"
        params["method"] = "Simulation"
        params["hypothesis_type"] = "Superiority"
        params["p1"] = 0.25
        params["p2"] = 0.35
        params["power"] = 0.80
        params["alpha"] = 0.05
        params["sides"] = 2
        params["allocation_ratio"] = 1.0
        params["effect_measure"] = "Risk Difference"
        params["num_simulations"] = 1000
        params["seed"] = 42

        expected_core_result = {"n1": 190, "n2": 190, "total_n": 380}
        mock_ss_sim_binary.return_value = expected_core_result

        result = parallel_rct.calculate_parallel_binary(params)

        mock_ss_sim_binary.assert_called_once_with(
            p1=params["p1"],
            p2=params["p2"],
            power=params["power"],
            alpha=params["alpha"],
            sides=params["sides"],
            allocation_ratio=params["allocation_ratio"],
            effect_measure=params["effect_measure"],
            num_simulations=params["num_simulations"],
            seed=params["seed"]
        )

        self.assertIn("n1", result)
        self.assertEqual(result["n1"], expected_core_result["n1"])
        self.assertIn("n2", result)
        self.assertEqual(result["n2"], expected_core_result["n2"])
        self.assertIn("total_n", result)
        self.assertEqual(result["total_n"], expected_core_result["total_n"])
        self.assertNotIn("error", result, f"Unexpected error: {result.get('error')}")
        self.assertEqual(result.get("p1_param"), params["p1"])
        self.assertEqual(result.get("p2_param"), params["p2"])
        self.assertEqual(result.get("power_param"), params["power"])
        self.assertEqual(result.get("alpha_param"), params["alpha"])
        self.assertEqual(result.get("sides_param"), params["sides"])
        self.assertEqual(result.get("allocation_ratio_param"), params["allocation_ratio"])
        self.assertEqual(result.get("effect_measure_param"), params["effect_measure"])
        self.assertEqual(result.get("num_simulations_param"), params["num_simulations"])
        self.assertEqual(result.get("seed_param"), params["seed"])
        self.assertEqual(result.get("method_param"), params["method"])
        self.assertEqual(result.get("hypothesis_type_param"), params["hypothesis_type"])
        self.assertEqual(result.get("calculation_type_param"), params["calculation_type"])
        self.assertEqual(result.get("design_method"), "Parallel RCT")
        self.assertEqual(result.get("outcome_type_param"), "Binary")

    @patch('app.components.parallel_rct.simulation_binary.sample_size_binary_sim')
    def test_calculate_parallel_binary_simulation_superiority_sample_size_risk_ratio(self, mock_ss_sim_binary):
        """Test simulation sample size for superiority, binary outcome, risk ratio."""
        params = self.default_params_binary.copy()
        params["calculation_type"] = "Sample Size"
        params["method"] = "Simulation"
        params["hypothesis_type"] = "Superiority"
        params["p1"] = 0.10 # Control event rate
        params["p2"] = 0.18 # Treatment event rate (p2/p1 = 1.8)
        params["power"] = 0.85
        params["alpha"] = 0.05
        params["sides"] = 2
        params["allocation_ratio"] = 1.0
        params["effect_measure"] = "Risk Ratio"
        params["num_simulations"] = 1200
        params["seed"] = 123

        expected_core_result = {"n1": 250, "n2": 250, "total_n": 500}
        mock_ss_sim_binary.return_value = expected_core_result

        result = parallel_rct.calculate_parallel_binary(params)

        mock_ss_sim_binary.assert_called_once_with(
            p1=params["p1"],
            p2=params["p2"],
            power=params["power"],
            alpha=params["alpha"],
            sides=params["sides"],
            allocation_ratio=params["allocation_ratio"],
            effect_measure=params["effect_measure"],
            num_simulations=params["num_simulations"],
            seed=params["seed"]
        )

        self.assertIn("n1", result)
        self.assertEqual(result["n1"], expected_core_result["n1"])
        self.assertIn("n2", result)
        self.assertEqual(result["n2"], expected_core_result["n2"])
        self.assertIn("total_n", result)
        self.assertEqual(result["total_n"], expected_core_result["total_n"])
        self.assertNotIn("error", result, f"Unexpected error: {result.get('error')}")
        self.assertEqual(result.get("p1_param"), params["p1"])
        self.assertEqual(result.get("p2_param"), params["p2"])
        self.assertEqual(result.get("power_param"), params["power"])
        self.assertEqual(result.get("alpha_param"), params["alpha"])
        self.assertEqual(result.get("sides_param"), params["sides"])
        self.assertEqual(result.get("allocation_ratio_param"), params["allocation_ratio"])
        self.assertEqual(result.get("effect_measure_param"), params["effect_measure"])
        self.assertEqual(result.get("num_simulations_param"), params["num_simulations"])
        self.assertEqual(result.get("seed_param"), params["seed"])
        self.assertEqual(result.get("method_param"), params["method"])
        self.assertEqual(result.get("hypothesis_type_param"), params["hypothesis_type"])
        self.assertEqual(result.get("calculation_type_param"), params["calculation_type"])
        self.assertEqual(result.get("design_method"), "Parallel RCT")
        self.assertEqual(result.get("outcome_type_param"), "Binary")

    @patch('app.components.parallel_rct.simulation_binary.sample_size_binary_sim')
    def test_calculate_parallel_binary_simulation_superiority_sample_size_odds_ratio(self, mock_ss_sim_binary):
        """Test simulation sample size for superiority, binary outcome, odds ratio."""
        params = self.default_params_binary.copy()
        params["calculation_type"] = "Sample Size"
        params["method"] = "Simulation"
        params["hypothesis_type"] = "Superiority"
        params["p1"] = 0.30 # Control event rate. Odds_c = 0.3/0.7 = 0.428
        params["p2"] = 0.45 # Treatment event rate. Odds_t = 0.45/0.55 = 0.818. OR_t/c = 0.818/0.428 = 1.91
        params["power"] = 0.90
        params["alpha"] = 0.05
        params["sides"] = 2
        params["allocation_ratio"] = 1.0
        params["effect_measure"] = "Odds Ratio"
        params["num_simulations"] = 1500
        params["seed"] = 500

        expected_core_result = {"n1": 180, "n2": 180, "total_n": 360}
        mock_ss_sim_binary.return_value = expected_core_result

        result = parallel_rct.calculate_parallel_binary(params)

        mock_ss_sim_binary.assert_called_once_with(
            p1=params["p1"],
            p2=params["p2"],
            power=params["power"],
            alpha=params["alpha"],
            sides=params["sides"],
            allocation_ratio=params["allocation_ratio"],
            effect_measure=params["effect_measure"],
            num_simulations=params["num_simulations"],
            seed=params["seed"]
        )

        self.assertIn("n1", result)
        self.assertEqual(result["n1"], expected_core_result["n1"])
        self.assertIn("n2", result)
        self.assertEqual(result["n2"], expected_core_result["n2"])
        self.assertIn("total_n", result)
        self.assertEqual(result["total_n"], expected_core_result["total_n"])
        self.assertNotIn("error", result, f"Unexpected error: {result.get('error')}")
        self.assertEqual(result.get("p1_param"), params["p1"])
        self.assertEqual(result.get("p2_param"), params["p2"])
        self.assertEqual(result.get("power_param"), params["power"])
        self.assertEqual(result.get("alpha_param"), params["alpha"])
        self.assertEqual(result.get("sides_param"), params["sides"])
        self.assertEqual(result.get("allocation_ratio_param"), params["allocation_ratio"])
        self.assertEqual(result.get("effect_measure_param"), params["effect_measure"])
        self.assertEqual(result.get("num_simulations_param"), params["num_simulations"])
        self.assertEqual(result.get("seed_param"), params["seed"])
        self.assertEqual(result.get("method_param"), params["method"])
        self.assertEqual(result.get("hypothesis_type_param"), params["hypothesis_type"])
        self.assertEqual(result.get("calculation_type_param"), params["calculation_type"])
        self.assertEqual(result.get("design_method"), "Parallel RCT")
        self.assertEqual(result.get("outcome_type_param"), "Binary")

    @patch('app.components.parallel_rct.simulation_binary.min_detectable_effect_binary_sim')
    def test_calculate_parallel_binary_simulation_superiority_mde_risk_difference(self, mock_mde_sim_binary):
        """Test simulation MDE for superiority, binary outcome, risk difference."""
        params = self.default_params_binary.copy()
        params["calculation_type"] = "Minimum Detectable Effect"
        params["method"] = "Simulation"
        params["hypothesis_type"] = "Superiority"
        params["p1"] = 0.60
        params["n1"] = 100
        params["n2"] = 100
        params["power"] = 0.80
        params["alpha"] = 0.05
        params["sides"] = 2
        params["effect_measure"] = "Risk Difference"
        params["num_simulations"] = 1000
        params["seed"] = 42

        expected_core_result = {"mde": 0.15, "p2_lower": 0.75, "p2_upper": 0.45} # p2_upper if p1 is higher
        mock_mde_sim_binary.return_value = expected_core_result

        result = parallel_rct.calculate_parallel_binary(params)

        mock_mde_sim_binary.assert_called_once_with(
            p1=params["p1"],
            n1=params["n1"],
            n2=params["n2"],
            power=params["power"],
            alpha=params["alpha"],
            sides=params["sides"],
            effect_measure=params["effect_measure"],
            num_simulations=params["num_simulations"],
            seed=params["seed"]
        )

        self.assertIn("mde", result)
        self.assertEqual(result["mde"], expected_core_result["mde"])
        self.assertIn("p2_lower", result)
        self.assertEqual(result["p2_lower"], expected_core_result["p2_lower"])
        self.assertIn("p2_upper", result)
        self.assertEqual(result["p2_upper"], expected_core_result["p2_upper"])
        self.assertNotIn("error", result, f"Unexpected error: {result.get('error')}")
        self.assertEqual(result.get("p1_param"), params["p1"])
        self.assertEqual(result.get("n1_param"), params["n1"])
        self.assertEqual(result.get("n2_param"), params["n2"])
        self.assertEqual(result.get("power_param"), params["power"])
        self.assertEqual(result.get("alpha_param"), params["alpha"])
        self.assertEqual(result.get("sides_param"), params["sides"])
        self.assertEqual(result.get("effect_measure_param"), params["effect_measure"])
        self.assertEqual(result.get("num_simulations_param"), params["num_simulations"])
        self.assertEqual(result.get("seed_param"), params["seed"])
        self.assertEqual(result.get("method_param"), params["method"])
        self.assertEqual(result.get("hypothesis_type_param"), params["hypothesis_type"])
        self.assertEqual(result.get("calculation_type_param"), params["calculation_type"])
        self.assertEqual(result.get("design_method"), "Parallel RCT")
        self.assertEqual(result.get("outcome_type_param"), "Binary")

    @patch('app.components.parallel_rct.simulation_binary.min_detectable_effect_binary_sim')
    def test_calculate_parallel_binary_simulation_superiority_mde_risk_ratio(self, mock_mde_sim_binary):
        """Test simulation MDE for superiority, binary outcome, risk ratio."""
        params = self.default_params_binary.copy()
        params["calculation_type"] = "Minimum Detectable Effect"
        params["method"] = "Simulation"
        params["hypothesis_type"] = "Superiority"
        params["p1"] = 0.20 # Control event rate
        params["n1"] = 150
        params["n2"] = 150
        params["power"] = 0.80
        params["alpha"] = 0.05
        params["sides"] = 2
        params["effect_measure"] = "Risk Ratio"
        params["num_simulations"] = 1200
        params["seed"] = 123

        # MDE is RR, p2_lower is p1*RR, p2_upper is p1/RR (or p1 if RR < 1 and p2 is better if higher)
        expected_core_result = {"mde": 1.5, "p2_lower": 0.30, "p2_upper": 0.1333} # Example values
        mock_mde_sim_binary.return_value = expected_core_result

        result = parallel_rct.calculate_parallel_binary(params)

        mock_mde_sim_binary.assert_called_once_with(
            p1=params["p1"],
            n1=params["n1"],
            n2=params["n2"],
            power=params["power"],
            alpha=params["alpha"],
            sides=params["sides"],
            effect_measure=params["effect_measure"],
            num_simulations=params["num_simulations"],
            seed=params["seed"]
        )

        self.assertIn("mde", result)
        self.assertAlmostEqual(result["mde"], expected_core_result["mde"], places=4)
        self.assertIn("p2_lower", result)
        self.assertAlmostEqual(result["p2_lower"], expected_core_result["p2_lower"], places=4)
        self.assertIn("p2_upper", result)
        self.assertAlmostEqual(result["p2_upper"], expected_core_result["p2_upper"], places=4)
        self.assertNotIn("error", result, f"Unexpected error: {result.get('error')}")
        self.assertEqual(result.get("p1_param"), params["p1"])
        self.assertEqual(result.get("n1_param"), params["n1"])
        self.assertEqual(result.get("n2_param"), params["n2"])
        self.assertEqual(result.get("power_param"), params["power"])
        self.assertEqual(result.get("alpha_param"), params["alpha"])
        self.assertEqual(result.get("sides_param"), params["sides"])
        self.assertEqual(result.get("effect_measure_param"), params["effect_measure"])
        self.assertEqual(result.get("num_simulations_param"), params["num_simulations"])
        self.assertEqual(result.get("seed_param"), params["seed"])
        self.assertEqual(result.get("method_param"), params["method"])
        self.assertEqual(result.get("hypothesis_type_param"), params["hypothesis_type"])
        self.assertEqual(result.get("calculation_type_param"), params["calculation_type"])
        self.assertEqual(result.get("design_method"), "Parallel RCT")
        self.assertEqual(result.get("outcome_type_param"), "Binary")

    @patch('app.components.parallel_rct.simulation_binary.min_detectable_effect_binary_sim')
    def test_calculate_parallel_binary_simulation_superiority_mde_odds_ratio(self, mock_mde_sim_binary):
        """Test simulation MDE for superiority, binary outcome, odds ratio."""
        params = self.default_params_binary.copy()
        params["calculation_type"] = "Minimum Detectable Effect"
        params["method"] = "Simulation"
        params["hypothesis_type"] = "Superiority"
        params["p1"] = 0.30 # Control event rate
        params["n1"] = 120
        params["n2"] = 120
        params["power"] = 0.85
        params["alpha"] = 0.05
        params["sides"] = 2
        params["effect_measure"] = "Odds Ratio"
        params["num_simulations"] = 1500
        params["seed"] = 500

        # MDE is OR, p2_lower/upper are event rates corresponding to that OR
        expected_core_result = {"mde": 2.0, "p2_lower": 0.4615, "p2_upper": 0.1765} # Example values
        mock_mde_sim_binary.return_value = expected_core_result

        result = parallel_rct.calculate_parallel_binary(params)

        mock_mde_sim_binary.assert_called_once_with(
            p1=params["p1"],
            n1=params["n1"],
            n2=params["n2"],
            power=params["power"],
            alpha=params["alpha"],
            sides=params["sides"],
            effect_measure=params["effect_measure"],
            num_simulations=params["num_simulations"],
            seed=params["seed"]
        )

        self.assertIn("mde", result)
        self.assertAlmostEqual(result["mde"], expected_core_result["mde"], places=4)
        self.assertIn("p2_lower", result)
        self.assertAlmostEqual(result["p2_lower"], expected_core_result["p2_lower"], places=4)
        self.assertIn("p2_upper", result)
        self.assertAlmostEqual(result["p2_upper"], expected_core_result["p2_upper"], places=4)
        self.assertNotIn("error", result, f"Unexpected error: {result.get('error')}")
        self.assertEqual(result.get("p1_param"), params["p1"])
        self.assertEqual(result.get("n1_param"), params["n1"])
        self.assertEqual(result.get("n2_param"), params["n2"])
        self.assertEqual(result.get("power_param"), params["power"])
        self.assertEqual(result.get("alpha_param"), params["alpha"])
        self.assertEqual(result.get("sides_param"), params["sides"])
        self.assertEqual(result.get("effect_measure_param"), params["effect_measure"])
        self.assertEqual(result.get("num_simulations_param"), params["num_simulations"])
        self.assertEqual(result.get("seed_param"), params["seed"])
        self.assertEqual(result.get("method_param"), params["method"])
        self.assertEqual(result.get("hypothesis_type_param"), params["hypothesis_type"])
        self.assertEqual(result.get("calculation_type_param"), params["calculation_type"])
        self.assertEqual(result.get("design_method"), "Parallel RCT")
        self.assertEqual(result.get("outcome_type_param"), "Binary")

    # --- Simulation Non-Inferiority Tests for Binary Outcomes ---
    @patch('app.components.parallel_rct.simulation_binary.power_binary_ni_sim')
    def test_calculate_parallel_binary_simulation_non_inferiority_power_risk_difference(self, mock_power_ni_sim_binary):
        """Test simulation power for non-inferiority, binary outcome, risk difference."""
        params = self.default_params_binary.copy()
        params["calculation_type"] = "Power"
        params["method"] = "Simulation"
        params["hypothesis_type"] = "Non-Inferiority"
        params["p1"] = 0.40  # Control event rate
        params["p2"] = 0.42  # Assumed treatment event rate (slightly better or same)
        params["n1"] = 500
        params["n2"] = 500
        params["alpha"] = 0.025 # Typically one-sided for NI
        params["non_inferiority_margin"] = 0.10 # NI margin for Risk Difference
        params["effect_measure"] = "Risk Difference"
        params["num_simulations"] = 1000
        params["seed"] = 42
        # sides is not explicitly passed for NI to core sim, alpha is adjusted

        expected_core_result = {"power": 0.85}
        mock_power_ni_sim_binary.return_value = expected_core_result

        result = parallel_rct.calculate_parallel_binary(params)

        mock_power_ni_sim_binary.assert_called_once_with(
            p1=params["p1"],
            p2_assumed=params["p2"], # Core function might take p2_assumed
            n1=params["n1"],
            n2=params["n2"],
            alpha=params["alpha"],
            non_inferiority_margin=params["non_inferiority_margin"],
            effect_measure=params["effect_measure"],
            num_simulations=params["num_simulations"],
            seed=params["seed"]
        )

        self.assertIn("power", result)
        self.assertEqual(result["power"], expected_core_result["power"])
        self.assertNotIn("error", result, f"Unexpected error: {result.get('error')}")
        self.assertEqual(result.get("p1_param"), params["p1"])
        self.assertEqual(result.get("p2_param"), params["p2"])
        self.assertEqual(result.get("n1_param"), params["n1"])
        self.assertEqual(result.get("n2_param"), params["n2"])
        self.assertEqual(result.get("alpha_param"), params["alpha"])
        self.assertEqual(result.get("non_inferiority_margin_param"), params["non_inferiority_margin"])
        self.assertEqual(result.get("effect_measure_param"), params["effect_measure"])
        self.assertEqual(result.get("num_simulations_param"), params["num_simulations"])
        self.assertEqual(result.get("seed_param"), params["seed"])
        self.assertEqual(result.get("method_param"), params["method"])
        self.assertEqual(result.get("hypothesis_type_param"), params["hypothesis_type"])
        self.assertEqual(result.get("calculation_type_param"), params["calculation_type"])
        self.assertEqual(result.get("design_method"), "Parallel RCT")
        self.assertEqual(result.get("outcome_type_param"), "Binary")

if __name__ == '__main__':
    unittest.main()
