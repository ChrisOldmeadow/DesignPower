import unittest
from unittest.mock import patch, MagicMock

from app.components import parallel_rct # Import the module to test

class TestParallelRCTSurvivalComponent(unittest.TestCase):

    def setUp(self):
        """Set up default parameters for survival outcome type."""
        self.default_params_survival = {
            "n1": 100, "n2": 100, "allocation_ratio": 1.0,
            "median_survival1": 12, "hr": 0.75,
            "accrual_time": 12, "follow_up_time": 18, "dropout_rate": 0.1,
            "alpha": 0.05, "power": 0.8, "sides": 2,
            "non_inferiority_margin_hr": 1.3, "assumed_true_hr": 1.0,
            "nsim": 100, "seed": 42 # Reduced nsim for tests
        }

    # --- Tests for calculate_parallel_survival --- 

    def test_calculate_parallel_survival_analytical_superiority_power_two_sided(self):
        """Test analytical power calculation for survival outcome, superiority, two-sided."""
        params = self.default_params_survival.copy()
        params["calculation_type"] = "Power"
        params["method"] = "Analytical"
        params["hypothesis_type"] = "Superiority"
        params["median_survival1"] = 12
        params["hr"] = 0.75
        params["accrual_time"] = 12
        params["follow_up_time"] = 18
        params["dropout_rate"] = 0.1
        params["alpha"] = 0.05
        params["n1"] = 100
        params["n2"] = 100
        params["sides"] = 2

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
            self.assertAlmostEqual(result.get("median_survival2_derived"), params["median_survival1"] / params["hr"])
            self.assertEqual(result.get("method_param"), params["method"])
            self.assertEqual(result.get("hypothesis_type_param"), params["hypothesis_type"])
            self.assertEqual(result.get("calculation_type_param"), params["calculation_type"])
            self.assertEqual(result.get("design_method"), "Parallel RCT")
            self.assertEqual(result.get("outcome_type_param"), "Survival")

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
        params["alpha"] = 0.025
        params["n1"] = 150
        params["n2"] = 150

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
            self.assertEqual(result.get("outcome_type_param"), "Survival")

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
            self.assertEqual(result.get("design_method"), "Parallel RCT")
            self.assertEqual(result.get("outcome_type_param"), "Survival")

    def test_calculate_parallel_survival_analytical_superiority_sample_size(self):
        """Test analytical sample size calculation for survival outcome, superiority, two-sided."""
        params = self.default_params_survival.copy()
        params["calculation_type"] = "Sample Size"
        params["method"] = "Analytical"
        params["hypothesis_type"] = "Superiority"
        params["median_survival1"] = 10
        params["hr"] = 0.7
        params["accrual_time"] = 10
        params["follow_up_time"] = 15
        params["dropout_rate"] = 0.05
        params["alpha"] = 0.05
        params["power"] = 0.8
        params["sides"] = 2

        with patch('app.components.parallel_rct.analytical_survival.sample_size_survival_analytical') as mock_ss_analytical:
            expected_core_result = {"n1": 150, "n2": 150, "total_n": 300, "events": 100}
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
            self.assertEqual(result.get("power_param"), params["power"])
            self.assertEqual(result.get("method_param"), params["method"])
            self.assertEqual(result.get("hypothesis_type_param"), params["hypothesis_type"])
            self.assertEqual(result.get("calculation_type_param"), params["calculation_type"])
            self.assertEqual(result.get("design_method"), "Parallel RCT")
            self.assertEqual(result.get("outcome_type_param"), "Survival")

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

        with patch('app.components.parallel_rct.analytical_survival.min_detectable_effect_survival_analytical') as mock_mde_analytical:
            expected_core_result = {"mde_hr": 0.70, "events": 60} # Example MDE hazard ratio
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
            self.assertEqual(result.get("power_param"), params["power"])
            self.assertEqual(result.get("method_param"), params["method"])
            self.assertEqual(result.get("hypothesis_type_param"), params["hypothesis_type"])
            self.assertEqual(result.get("calculation_type_param"), params["calculation_type"])
            self.assertEqual(result.get("design_method"), "Parallel RCT")
            self.assertEqual(result.get("outcome_type_param"), "Survival")

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
        params["nsim"] = self.default_params_survival["nsim"]
        params["seed"] = self.default_params_survival["seed"]

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
        self.assertEqual(result.get("outcome_type_param"), "Survival")

    @patch('app.components.parallel_rct.simulation_survival.power_survival_superiority_sim')
    def test_calculate_parallel_survival_simulation_superiority_power(self, mock_power_sup_sim):
        """Test simulation power calculation for survival outcome, superiority, two-sided."""
        params = self.default_params_survival.copy()
        params["calculation_type"] = "Power"
        params["method"] = "Simulation"
        params["hypothesis_type"] = "Superiority"
        params["median_survival1"] = 12
        params["hr"] = 0.65
        params["accrual_time"] = 10
        params["follow_up_time"] = 20
        params["dropout_rate"] = 0.1
        params["alpha"] = 0.05
        params["n1"] = 90
        params["n2"] = 90
        params["sides"] = 2
        params["nsim"] = self.default_params_survival["nsim"]
        params["seed"] = self.default_params_survival["seed"]

        expected_core_result = {"power": 0.92, "events": 75, "nsim": params["nsim"]}
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
        self.assertEqual(result.get("outcome_type_param"), "Survival")

    @patch('app.components.parallel_rct.simulation_survival.min_detectable_effect_survival_superiority_sim')
    def test_calculate_parallel_survival_simulation_superiority_mde(self, mock_mde_sup_sim):
        """Test simulation MDE calculation for survival outcome, superiority, two-sided."""
        params = self.default_params_survival.copy()
        params["calculation_type"] = "Minimum Detectable Effect"
        params["method"] = "Simulation"
        params["hypothesis_type"] = "Superiority"
        params["median_survival1"] = 10
        params["accrual_time"] = 10
        params["follow_up_time"] = 15
        params["dropout_rate"] = 0.05
        params["alpha"] = 0.05
        params["power"] = 0.8
        params["sides"] = 2
        params["n1"] = 120
        params["n2"] = 120
        params["nsim"] = self.default_params_survival["nsim"]
        params["seed"] = self.default_params_survival["seed"]

        expected_core_result = {"mde_hr": 0.70, "events": 45, "nsim": params["nsim"]}
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
        self.assertEqual(result.get("power_param"), params["power"])
        self.assertEqual(result.get("method_param"), params["method"])
        self.assertEqual(result.get("hypothesis_type_param"), params["hypothesis_type"])
        self.assertEqual(result.get("calculation_type_param"), params["calculation_type"])
        self.assertEqual(result.get("nsim"), params["nsim"])
        self.assertEqual(result.get("design_method"), "Parallel RCT")
        self.assertEqual(result.get("outcome_type_param"), "Survival")

if __name__ == '__main__':
    unittest.main()
