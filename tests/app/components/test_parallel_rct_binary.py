import unittest
from unittest.mock import patch, MagicMock
import pytest # Keep for pytest.approx if used, or remove if all assertions are unittest based

from app.components import parallel_rct # Import the module to test

class TestParallelRCTBinaryComponent(unittest.TestCase):

    def setUp(self):
        """Set up default parameters for binary outcome tests."""
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

    @patch('app.components.parallel_rct.analytical_binary.power_binary')
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
            test_type="normal approximation",
            correction=False
        )

        self.assertIn("power", result)
        self.assertEqual(result["power"], expected_core_result["power"])
        self.assertNotIn("error", result, f"Unexpected error: {result.get('error')}")
        self.assertEqual(result.get("p1"), params["p1"])
        self.assertEqual(result.get("p2"), params["p2"])
        self.assertEqual(result.get("n1"), params["n1"])
        self.assertEqual(result.get("n2"), params["n2"])
        self.assertEqual(result.get("alpha"), params["alpha"])
        self.assertEqual(result.get("method"), params["method"].capitalize())
        self.assertEqual(result.get("hypothesis_type"), params["hypothesis_type"])
        self.assertEqual(result.get("calculation_type"), params["calculation_type"])
        self.assertEqual(result.get("test_type"), params.get("test_type", "Normal Approximation"))
        self.assertEqual(result.get("design_method"), "Parallel RCT")
        self.assertEqual(result.get("outcome_type_param"), "Binary")

    @patch('app.components.parallel_rct.analytical_binary.power_binary')
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
            test_type="normal approximation",
            correction=False
        )

        self.assertIn("power", result)
        self.assertEqual(result["power"], expected_core_result["power"])
        self.assertNotIn("error", result, f"Unexpected error: {result.get('error')}")
        self.assertEqual(result.get("p1"), params["p1"])
        self.assertEqual(result.get("p2"), params["p2"])
        self.assertEqual(result.get("n1"), params["n1"])
        self.assertEqual(result.get("n2"), params["n2"])
        self.assertEqual(result.get("alpha"), params["alpha"])
        self.assertEqual(result.get("method"), params["method"].capitalize())
        self.assertEqual(result.get("hypothesis_type"), params["hypothesis_type"])
        self.assertEqual(result.get("calculation_type"), params["calculation_type"])
        self.assertEqual(result.get("test_type"), params.get("test_type", "Normal Approximation"))
        self.assertEqual(result.get("design_method"), "Parallel RCT")
        self.assertEqual(result.get("outcome_type_param"), "Binary")

    @patch('app.components.parallel_rct.analytical_binary.power_binary')
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
            test_type="normal approximation",
            correction=False
        )

        self.assertIn("power", result)
        self.assertEqual(result["power"], expected_core_result["power"])
        self.assertNotIn("error", result, f"Unexpected error: {result.get('error')}")
        self.assertEqual(result.get("p1"), params["p1"])
        self.assertEqual(result.get("p2"), params["p2"])
        self.assertEqual(result.get("n1"), params["n1"])
        self.assertEqual(result.get("n2"), params["n2"])
        self.assertEqual(result.get("alpha"), params["alpha"])
        self.assertEqual(result.get("method"), params["method"].capitalize())
        self.assertEqual(result.get("hypothesis_type"), params["hypothesis_type"])
        self.assertEqual(result.get("calculation_type"), params["calculation_type"])
        self.assertEqual(result.get("test_type"), params.get("test_type", "Normal Approximation"))
        self.assertEqual(result.get("design_method"), "Parallel RCT")
        self.assertEqual(result.get("outcome_type_param"), "Binary")

    @patch('app.components.parallel_rct.analytical_binary.sample_size_binary')
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
            test_type="normal approximation",
            correction=False
        )

        self.assertIn("n1", result)
        self.assertEqual(result["n1"], expected_core_result["n1"])
        self.assertIn("n2", result)
        self.assertEqual(result["n2"], expected_core_result["n2"])
        self.assertIn("total_n", result)
        self.assertEqual(result["total_n"], expected_core_result["total_n"])
        self.assertNotIn("error", result, f"Unexpected error: {result.get('error')}")
        self.assertEqual(result.get("p1"), params["p1"])
        self.assertEqual(result.get("p2"), params["p2"])
        self.assertEqual(result.get("power"), params["power"])
        self.assertEqual(result.get("alpha"), params["alpha"])
        self.assertEqual(result.get("allocation_ratio"), params["allocation_ratio"])
        self.assertEqual(result.get("method"), params["method"].capitalize())
        self.assertEqual(result.get("hypothesis_type"), params["hypothesis_type"])
        self.assertEqual(result.get("calculation_type"), params["calculation_type"])
        self.assertEqual(result.get("test_type"), params.get("test_type", "Normal Approximation"))
        self.assertEqual(result.get("design_method"), "Parallel RCT")
        self.assertEqual(result.get("outcome_type_param"), "Binary")

    @patch('app.components.parallel_rct.analytical_binary.sample_size_binary')
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
            test_type="normal approximation",
            correction=False
        )

        self.assertIn("n1", result)
        self.assertEqual(result["n1"], expected_core_result["n1"])
        self.assertIn("n2", result)
        self.assertEqual(result["n2"], expected_core_result["n2"])
        self.assertIn("total_n", result)
        self.assertEqual(result["total_n"], expected_core_result["total_n"])
        self.assertNotIn("error", result, f"Unexpected error: {result.get('error')}")
        self.assertEqual(result.get("p1"), params["p1"])
        self.assertEqual(result.get("p2"), params["p2"])
        self.assertEqual(result.get("power"), params["power"])
        self.assertEqual(result.get("alpha"), params["alpha"])
        self.assertEqual(result.get("allocation_ratio"), params["allocation_ratio"])
        self.assertEqual(result.get("method"), params["method"].capitalize())
        self.assertEqual(result.get("hypothesis_type"), params["hypothesis_type"])
        self.assertEqual(result.get("calculation_type"), params["calculation_type"])
        self.assertEqual(result.get("test_type"), params.get("test_type", "Normal Approximation"))
        self.assertEqual(result.get("design_method"), "Parallel RCT")
        self.assertEqual(result.get("outcome_type_param"), "Binary")

    @patch('app.components.parallel_rct.analytical_binary.sample_size_binary')
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
            test_type="normal approximation",
            correction=False
        )

        self.assertIn("n1", result)
        self.assertEqual(result["n1"], expected_core_result["n1"])
        self.assertIn("n2", result)
        self.assertEqual(result["n2"], expected_core_result["n2"])
        self.assertIn("total_n", result)
        self.assertEqual(result["total_n"], expected_core_result["total_n"])
        self.assertNotIn("error", result, f"Unexpected error: {result.get('error')}")
        self.assertEqual(result.get("p1"), params["p1"])
        self.assertEqual(result.get("p2"), params["p2"])
        self.assertEqual(result.get("power"), params["power"])
        self.assertEqual(result.get("alpha"), params["alpha"])
        self.assertEqual(result.get("allocation_ratio"), params["allocation_ratio"])
        self.assertEqual(result.get("method"), params["method"].capitalize())
        self.assertEqual(result.get("hypothesis_type"), params["hypothesis_type"])
        self.assertEqual(result.get("calculation_type"), params["calculation_type"])
        self.assertEqual(result.get("test_type"), params.get("test_type", "Normal Approximation"))
        self.assertEqual(result.get("design_method"), "Parallel RCT")
        self.assertEqual(result.get("outcome_type_param"), "Binary")

    @patch('app.components.parallel_rct.analytical_binary.min_detectable_effect_binary')
    def test_calculate_parallel_binary_analytical_superiority_mde_risk_difference(self, mock_mde_analytical_binary):
        """Test analytical MDE calculation for binary outcome, superiority, risk difference."""
        params = self.default_params_binary.copy()
        params["calculation_type"] = "Minimum Detectable Effect"
        params["method"] = "Analytical"
        params["hypothesis_type"] = "Superiority"
        # p1 is 0.65 from default_params_binary
        params["n1"] = 100
        params["n2"] = 100
        params["power"] = 0.80
        params["alpha"] = 0.05
        params["sides"] = 2 # Will be converted to test_type by component
        params["effect_measure"] = "Risk Difference"

        expected_rd = -0.15  # Expecting p2 to be lower for superiority
        expected_p2_at_mde = params["p1"] + expected_rd # 0.65 - 0.15 = 0.50
        mock_mde_analytical_binary.return_value = {"mde": expected_rd, "p2": expected_p2_at_mde}

        result = parallel_rct.calculate_parallel_binary(params)

        mock_mde_analytical_binary.assert_called_once_with(
            p1=params["p1"],
            n1=params["n1"],
            n2=params["n2"],
            power=params["power"],
            alpha=params["alpha"],
            test_type="normal approximation", # derived from sides=2
            correction=False # default
        )

        self.assertNotIn("error", result, f"Unexpected error: {result.get('error')}")
        self.assertIn("mde", result)
        self.assertAlmostEqual(result["mde"], expected_rd, places=7)
        self.assertIn("p2_mde", result)
        self.assertAlmostEqual(result["p2_mde"], expected_p2_at_mde, places=7)
        self.assertIn("absolute_risk_difference", result)
        # For superiority, mde is directional. absolute_risk_difference in result is also directional from p2-p1.
        self.assertAlmostEqual(result["absolute_risk_difference"], expected_rd, places=7)
        
        # Check other common params
        self.assertEqual(result.get("p1"), params["p1"])
        self.assertEqual(result.get("n1"), params["n1"])
        self.assertEqual(result.get("n2"), params["n2"])
        self.assertEqual(result.get("power"), params["power"])
        self.assertEqual(result.get("alpha"), params["alpha"])
        self.assertEqual(result.get("method"), params["method"].capitalize())
        self.assertEqual(result.get("hypothesis_type"), params["hypothesis_type"])
        self.assertEqual(result.get("calculation_type"), params["calculation_type"])
        self.assertEqual(result.get("test_type"), "Normal Approximation")
        self.assertEqual(result.get("design_method"), "Parallel RCT")
        self.assertEqual(result.get("outcome_type_param"), "Binary")

    @patch('app.components.parallel_rct.analytical_binary.min_detectable_effect_binary')
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

        # Core function returns MDE as Risk Difference
        # Let's say for p1=0.5, the detectable RD is -0.20, so p2_mde = 0.30
        expected_rd_from_core = -0.20
        expected_p2_at_mde = params["p1"] + expected_rd_from_core # 0.50 - 0.20 = 0.30
        mock_mde_analytical_binary.return_value = {"mde": expected_rd_from_core, "p2": expected_p2_at_mde}

        result = parallel_rct.calculate_parallel_binary(params)

        mock_mde_analytical_binary.assert_called_once_with(
            p1=params["p1"],
            n1=params["n1"],
            n2=params["n2"],
            power=params["power"],
            alpha=params["alpha"],
            test_type="normal approximation",
            correction=False
        )
        self.assertNotIn("error", result, f"Unexpected error: {result.get('error')}")
        self.assertIn("p2_mde", result)
        self.assertAlmostEqual(result["p2_mde"], expected_p2_at_mde, places=7)
        self.assertIn("relative_risk", result)
        expected_rr = expected_p2_at_mde / params["p1"] if params["p1"] != 0 else 0 # 0.30 / 0.50 = 0.6
        self.assertAlmostEqual(result["relative_risk"], expected_rr, places=7)
        self.assertIn("mde", result) # This should be the RD from core
        self.assertAlmostEqual(result["mde"], expected_rd_from_core, places=7)

        # Check other common params
        self.assertEqual(result.get("p1"), params["p1"])
        self.assertEqual(result.get("n1"), params["n1"])
        self.assertEqual(result.get("n2"), params["n2"])
        self.assertEqual(result.get("power"), params["power"])
        self.assertEqual(result.get("alpha"), params["alpha"])
        self.assertEqual(result.get("method"), params["method"].capitalize())
        self.assertEqual(result.get("hypothesis_type"), params["hypothesis_type"])
        self.assertEqual(result.get("calculation_type"), params["calculation_type"])
        self.assertEqual(result.get("test_type"), "Normal Approximation")
        self.assertEqual(result.get("design_method"), "Parallel RCT")
        self.assertEqual(result.get("outcome_type_param"), "Binary")

    @patch('app.components.parallel_rct.analytical_binary.min_detectable_effect_binary')
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

        # Core function returns MDE as Risk Difference
        # For p1=0.4, let's say detectable RD is -0.17, so p2_mde = 0.23
        expected_rd_from_core = -0.17
        expected_p2_at_mde = params["p1"] + expected_rd_from_core # 0.40 - 0.17 = 0.23
        mock_mde_analytical_binary.return_value = {"mde": expected_rd_from_core, "p2": expected_p2_at_mde}

        result = parallel_rct.calculate_parallel_binary(params)

        mock_mde_analytical_binary.assert_called_once_with(
            p1=params["p1"],
            n1=params["n1"],
            n2=params["n2"],
            power=params["power"],
            alpha=params["alpha"],
            test_type="normal approximation",
            correction=False
        )
        self.assertNotIn("error", result, f"Unexpected error: {result.get('error')}")
        self.assertIn("p2_mde", result)
        self.assertAlmostEqual(result["p2_mde"], expected_p2_at_mde, places=7)
        self.assertIn("odds_ratio", result)
        
        # Calculate expected odds_ratio carefully to avoid division by zero
        p1_val = params["p1"]
        p2_val = expected_p2_at_mde
        
        expected_or_full_precision = 1.0 # Default or fallback
        if p1_val == 1 or p1_val == 0 or p2_val == 1 or p2_val == 0:
            if p1_val == p2_val : expected_or_full_precision = 1.0
            elif p1_val == 1 and p2_val == 0: expected_or_full_precision = 0.0
            elif p1_val == 0 and p2_val == 1: expected_or_full_precision = float('inf')
            elif (p1_val == 0 and p2_val != 0 and p2_val != 1) or \
                 (p2_val == 1 and p1_val != 0 and p1_val != 1) : expected_or_full_precision = float('inf')
            elif (p1_val == 1 and p2_val != 0 and p2_val != 1) or \
                 (p2_val == 0 and p1_val != 0 and p1_val != 1): expected_or_full_precision = 0.0
            # else: expected_or_full_precision remains 1.0 (already set as default/fallback)
        else:
            odds_p1 = p1_val / (1 - p1_val)
            odds_p2 = p2_val / (1 - p2_val)
            expected_or_full_precision = odds_p2 / odds_p1
        
        # Assume the function under test rounds odds_ratio to 3 decimal places for output
        if expected_or_full_precision == float('inf'):
            expected_value_for_assertion = float('inf')
            self.assertEqual(result["odds_ratio"], expected_value_for_assertion)
        else:
            expected_value_for_assertion = round(expected_or_full_precision, 3)
            # result["odds_ratio"] is expected to be the 3dp rounded value (e.g., 0.448)
            # expected_value_for_assertion is also 0.448 from rounding the full precision value.
            # places=5 ensures that result["odds_ratio"] is very close to this 3dp rounded value.
            self.assertAlmostEqual(result["odds_ratio"], expected_value_for_assertion, places=5)
        
        self.assertIn("mde", result) # This should be the RD from core
        self.assertAlmostEqual(result["mde"], expected_rd_from_core, places=7)

        # Check other common params
        self.assertEqual(result.get("p1"), params["p1"])
        self.assertEqual(result.get("n1"), params["n1"])
        self.assertEqual(result.get("n2"), params["n2"])
        self.assertEqual(result.get("power"), params["power"])
        self.assertEqual(result.get("alpha"), params["alpha"])
        self.assertEqual(result.get("method"), params["method"].capitalize())
        self.assertEqual(result.get("hypothesis_type"), params["hypothesis_type"])
        self.assertEqual(result.get("calculation_type"), params["calculation_type"])
        self.assertEqual(result.get("test_type"), "Normal Approximation")
        self.assertEqual(result.get("design_method"), "Parallel RCT")
        self.assertEqual(result.get("outcome_type_param"), "Binary")

    # --- Analytical Non-Inferiority Tests for Binary Outcomes ---
    @patch('app.components.parallel_rct.analytical_binary.power_binary_non_inferiority')
    def test_calculate_parallel_binary_analytical_non_inferiority_power_risk_difference(self, mock_power_ni_analytical_binary):
        """Test analytical power calculation for non-inferiority, binary outcome, risk difference."""
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
            n1=params["n1"],
            n2=params["n2"],
            p1=params["p1"],
            p2_reference=params["p2"],
            alpha=params["alpha"],
            non_inferiority_margin=params["non_inferiority_margin"],
            direction="Higher is better",
            test_type="normal approximation",
            correction=False
        )

        self.assertIn("power", result)
        self.assertEqual(result["power"], expected_core_result["power"])
        self.assertNotIn("error", result, f"Unexpected error: {result.get('error')}")
        self.assertEqual(result.get("p1"), params["p1"])
        self.assertEqual(result.get("p2"), params["p2"])
        self.assertEqual(result.get("n1"), params["n1"])
        self.assertEqual(result.get("n2"), params["n2"])
        self.assertEqual(result.get("alpha"), params["alpha"])
        self.assertEqual(result.get("non_inferiority_margin"), params["non_inferiority_margin"])
        self.assertEqual(result.get("method"), params["method"].capitalize())
        self.assertEqual(result.get("hypothesis_type"), params["hypothesis_type"])
        self.assertEqual(result.get("calculation_type"), params["calculation_type"])
        self.assertEqual(result.get("test_type"), params.get("test_type", "Normal Approximation"))
        self.assertEqual(result.get("design_method"), "Parallel RCT")
        self.assertEqual(result.get("outcome_type_param"), "Binary")

    @patch('app.components.parallel_rct.analytical_binary.power_binary_non_inferiority')
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
            n1=params["n1"],
            n2=params["n2"],
            p1=params["p1"],
            p2_reference=params["p2"],
            alpha=params["alpha"],
            non_inferiority_margin=params["non_inferiority_margin"],
            direction="Higher is better",
            test_type="normal approximation",
            correction=False
        )

        self.assertIn("power", result)
        self.assertEqual(result["power"], expected_core_result["power"])
        self.assertNotIn("error", result, f"Unexpected error: {result.get('error')}")
        self.assertEqual(result.get("p1"), params["p1"])
        self.assertEqual(result.get("p2"), params["p2"])
        self.assertEqual(result.get("n1"), params["n1"])
        self.assertEqual(result.get("n2"), params["n2"])
        self.assertEqual(result.get("alpha"), params["alpha"])
        self.assertEqual(result.get("non_inferiority_margin"), params["non_inferiority_margin"])
        self.assertEqual(result.get("method"), params["method"].capitalize())
        self.assertEqual(result.get("hypothesis_type"), params["hypothesis_type"])
        self.assertEqual(result.get("calculation_type"), params["calculation_type"])
        self.assertEqual(result.get("test_type"), params.get("test_type", "Normal Approximation"))
        self.assertEqual(result.get("design_method"), "Parallel RCT")
        self.assertEqual(result.get("outcome_type_param"), "Binary")

    @patch('app.components.parallel_rct.analytical_binary.power_binary_non_inferiority')
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
            n1=params["n1"],
            n2=params["n2"],
            p1=params["p1"],
            p2_reference=params["p2"],
            alpha=params["alpha"],
            non_inferiority_margin=params["non_inferiority_margin"],
            direction="Higher is better",
            test_type="normal approximation",
            correction=False
        )

        self.assertIn("power", result)
        self.assertEqual(result["power"], expected_core_result["power"])
        self.assertNotIn("error", result, f"Unexpected error: {result.get('error')}")
        self.assertEqual(result.get("p1"), params["p1"])
        self.assertEqual(result.get("p2"), params["p2"])
        self.assertEqual(result.get("n1"), params["n1"])
        self.assertEqual(result.get("n2"), params["n2"])
        self.assertEqual(result.get("alpha"), params["alpha"])
        self.assertEqual(result.get("non_inferiority_margin"), params["non_inferiority_margin"])
        self.assertEqual(result.get("method"), params["method"].capitalize())
        self.assertEqual(result.get("hypothesis_type"), params["hypothesis_type"])
        self.assertEqual(result.get("calculation_type"), params["calculation_type"])
        self.assertEqual(result.get("test_type"), params.get("test_type", "Normal Approximation"))
        self.assertEqual(result.get("design_method"), "Parallel RCT")
        self.assertEqual(result.get("outcome_type_param"), "Binary")

    @patch('app.components.parallel_rct.analytical_binary.sample_size_binary_non_inferiority')
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
            p2_reference=params["p2"],
            power=params["power"],
            alpha=params["alpha"],
            non_inferiority_margin=params["non_inferiority_margin"],
            allocation_ratio=params["allocation_ratio"],
            direction="Higher is better",
            test_type="normal approximation",
            correction=False
        )

        self.assertIn("n1", result)
        self.assertEqual(result["n1"], expected_core_result["n1"])
        self.assertIn("n2", result)
        self.assertEqual(result["n2"], expected_core_result["n2"])
        self.assertIn("total_n", result)
        self.assertEqual(result["total_n"], expected_core_result["total_n"])
        self.assertNotIn("error", result, f"Unexpected error: {result.get('error')}")
        self.assertEqual(result.get("p1"), params["p1"])
        self.assertEqual(result.get("p2"), params["p2"])
        self.assertEqual(result.get("power"), params["power"])
        self.assertEqual(result.get("alpha"), params["alpha"])
        self.assertEqual(result.get("non_inferiority_margin"), params["non_inferiority_margin"])
        self.assertEqual(result.get("allocation_ratio"), params["allocation_ratio"])
        self.assertEqual(result.get("method"), params["method"].capitalize())
        self.assertEqual(result.get("hypothesis_type"), params["hypothesis_type"])
        self.assertEqual(result.get("calculation_type"), params["calculation_type"])
        self.assertEqual(result.get("test_type"), params.get("test_type", "Normal Approximation"))
        self.assertEqual(result.get("design_method"), "Parallel RCT")
        self.assertEqual(result.get("outcome_type_param"), "Binary")

    @patch('app.components.parallel_rct.analytical_binary.sample_size_binary_non_inferiority')
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
            p2_reference=params["p2"],
            power=params["power"],
            alpha=params["alpha"],
            non_inferiority_margin=params["non_inferiority_margin"],
            allocation_ratio=params["allocation_ratio"],
            direction="Higher is better",
            test_type="normal approximation",
            correction=False
        )

        self.assertIn("n1", result)
        self.assertEqual(result["n1"], expected_core_result["n1"])
        self.assertIn("n2", result)
        self.assertEqual(result["n2"], expected_core_result["n2"])
        self.assertIn("total_n", result)
        self.assertEqual(result["total_n"], expected_core_result["total_n"])
        self.assertNotIn("error", result, f"Unexpected error: {result.get('error')}")
        self.assertEqual(result.get("p1"), params["p1"])
        self.assertEqual(result.get("p2"), params["p2"])
        self.assertEqual(result.get("power"), params["power"])
        self.assertEqual(result.get("alpha"), params["alpha"])
        self.assertEqual(result.get("non_inferiority_margin"), params["non_inferiority_margin"])
        self.assertEqual(result.get("allocation_ratio"), params["allocation_ratio"])
        self.assertEqual(result.get("method"), params["method"].capitalize())
        self.assertEqual(result.get("hypothesis_type"), params["hypothesis_type"])
        self.assertEqual(result.get("calculation_type"), params["calculation_type"])
        self.assertEqual(result.get("test_type"), params.get("test_type", "Normal Approximation"))
        self.assertEqual(result.get("design_method"), "Parallel RCT")
        self.assertEqual(result.get("outcome_type_param"), "Binary")

    @patch('app.components.parallel_rct.analytical_binary.sample_size_binary_non_inferiority')
    def test_calculate_parallel_binary_analytical_non_inferiority_sample_size_odds_ratio(self, mock_ss_ni_analytical_binary):
        """Test analytical sample size for non-inferiority, binary outcome, odds ratio."""
        params = self.default_params_binary.copy()
        params["calculation_type"] = "Sample Size"
        params["method"] = "Analytical"
        params["hypothesis_type"] = "Non-Inferiority"
        params["p1"] = 0.40 # Control success rate. Odds = 0.4/0.6 = 0.667
        params["p2"] = 0.38 # Expected treatment success rate. Odds = 0.38/0.62 = 0.613. OR_p2/p1 = 0.613/0.667 = 0.919
        params["power"] = 0.80
        params["alpha"] = 0.025
        params["sides"] = 1
        params["allocation_ratio"] = 1.0
        params["effect_measure"] = "Odds Ratio"
        params["non_inferiority_margin"] = 0.7 # OR_t/c must be >= 0.7

        expected_core_result = {"n1": 550, "n2": 550, "total_n": 1100}
        mock_ss_ni_analytical_binary.return_value = expected_core_result

        result = parallel_rct.calculate_parallel_binary(params)

        mock_ss_ni_analytical_binary.assert_called_once_with(
            p1=params["p1"],
            p2_reference=params["p2"],
            power=params["power"],
            alpha=params["alpha"],
            non_inferiority_margin=params["non_inferiority_margin"],
            allocation_ratio=params["allocation_ratio"],
            direction="Higher is better",
            test_type="normal approximation",
            correction=False
        )

        self.assertIn("n1", result)
        self.assertEqual(result["n1"], expected_core_result["n1"])
        self.assertIn("n2", result)
        self.assertEqual(result["n2"], expected_core_result["n2"])
        self.assertIn("total_n", result)
        self.assertEqual(result["total_n"], expected_core_result["total_n"])
        self.assertNotIn("error", result, f"Unexpected error: {result.get('error')}")
        self.assertEqual(result.get("p1"), params["p1"])
        self.assertEqual(result.get("p2"), params["p2"])
        self.assertEqual(result.get("power"), params["power"])
        self.assertEqual(result.get("alpha"), params["alpha"])
        self.assertEqual(result.get("non_inferiority_margin"), params["non_inferiority_margin"])
        self.assertEqual(result.get("allocation_ratio"), params["allocation_ratio"])
        self.assertEqual(result.get("method"), params["method"].capitalize())
        self.assertEqual(result.get("hypothesis_type"), params["hypothesis_type"])
        self.assertEqual(result.get("calculation_type"), params["calculation_type"])
        self.assertEqual(result.get("test_type"), params.get("test_type", "Normal Approximation"))
        self.assertEqual(result.get("design_method"), "Parallel RCT")
        self.assertEqual(result.get("outcome_type_param"), "Binary")

    @unittest.skip("Core function for MDE Non-Inferiority (analytical_binary.min_detectable_effect_binary_non_inferiority) is missing")
    @patch('app.components.parallel_rct.analytical_binary.min_detectable_effect_binary_ni_analytical') # This patch target is problematic but skipped
    def test_calculate_parallel_binary_analytical_non_inferiority_mde_risk_difference(self, mock_mde_ni_analytical_binary):
        """Test analytical MDE for non-inferiority, binary outcome, risk difference."""
        params = self.default_params_binary.copy()
        params["calculation_type"] = "Minimum Detectable Effect"
        params["method"] = "Analytical"
        params["hypothesis_type"] = "Non-Inferiority"
        params["p1"] = 0.60  # Control success rate
        params["n1"] = 150
        params["n2"] = 150
        params["power"] = 0.80
        params["alpha"] = 0.025
        params["sides"] = 1
        params["effect_measure"] = "Risk Difference"
        params["non_inferiority_margin"] = -0.10  # p2-p1 >= -0.10

        expected_core_result = {"mde_as_difference": -0.05, "p2_at_mde": 0.55} # p2 = p1 + mde
        mock_mde_ni_analytical_binary.return_value = expected_core_result

        result = parallel_rct.calculate_parallel_binary(params)

        mock_mde_ni_analytical_binary.assert_called_once_with(
            p1=params["p1"],
            n1=params["n1"],
            n2=params["n2"],
            power=params["power"],
            alpha=params["alpha"],
            non_inferiority_margin=params["non_inferiority_margin"],
            direction="Higher is better",
            test_type="normal approximation",
            correction=False
        )

        self.assertIn("mde_as_difference", result)
        self.assertEqual(result["mde_as_difference"], expected_core_result["mde_as_difference"])
        self.assertIn("p2_at_mde", result)
        self.assertEqual(result["p2_at_mde"], expected_core_result["p2_at_mde"])
        self.assertNotIn("error", result, f"Unexpected error: {result.get('error')}")
        self.assertEqual(result.get("p1"), params["p1"])
        self.assertEqual(result.get("n1"), params["n1"])
        self.assertEqual(result.get("n2"), params["n2"])
        self.assertEqual(result.get("power"), params["power"])
        self.assertEqual(result.get("alpha"), params["alpha"])
        self.assertEqual(result.get("non_inferiority_margin"), params["non_inferiority_margin"])
        self.assertEqual(result.get("method"), params["method"].capitalize())
        self.assertEqual(result.get("hypothesis_type"), params["hypothesis_type"])
        self.assertEqual(result.get("calculation_type"), params["calculation_type"])
        self.assertEqual(result.get("test_type"), params.get("test_type", "Normal Approximation"))
        self.assertEqual(result.get("design_method"), "Parallel RCT")
        self.assertEqual(result.get("outcome_type_param"), "Binary")

    @unittest.skip("Core function for MDE Non-Inferiority (analytical_binary.min_detectable_effect_binary_non_inferiority) is missing")
    @patch('app.components.parallel_rct.analytical_binary.min_detectable_effect_binary_ni_analytical') # This patch target is problematic but skipped
    def test_calculate_parallel_binary_analytical_non_inferiority_mde_risk_ratio(self, mock_mde_ni_analytical_binary):
        """Test analytical MDE for non-inferiority, binary outcome, risk ratio."""
        params = self.default_params_binary.copy()
        params["calculation_type"] = "Minimum Detectable Effect"
        params["method"] = "Analytical"
        params["hypothesis_type"] = "Non-Inferiority"
        params["p1"] = 0.30  # Control success rate
        params["n1"] = 200
        params["n2"] = 200
        params["power"] = 0.80
        params["alpha"] = 0.025
        params["sides"] = 1
        params["effect_measure"] = "Risk Ratio"
        params["non_inferiority_margin"] = 0.80  # p2/p1 >= 0.80

        expected_core_result = {"mde_as_ratio": 0.9, "p2_at_mde": 0.27} # p2 = p1 * mde
        mock_mde_ni_analytical_binary.return_value = expected_core_result

        result = parallel_rct.calculate_parallel_binary(params)

        mock_mde_ni_analytical_binary.assert_called_once_with(
            p1=params["p1"],
            n1=params["n1"],
            n2=params["n2"],
            power=params["power"],
            alpha=params["alpha"],
            non_inferiority_margin=params["non_inferiority_margin"],
            direction="Higher is better",
            test_type="normal approximation",
            correction=False
        )

        self.assertIn("mde_as_ratio", result)
        self.assertEqual(result["mde_as_ratio"], expected_core_result["mde_as_ratio"])
        self.assertIn("p2_at_mde", result)
        self.assertEqual(result["p2_at_mde"], expected_core_result["p2_at_mde"])
        self.assertNotIn("error", result, f"Unexpected error: {result.get('error')}")
        self.assertEqual(result.get("p1"), params["p1"])
        self.assertEqual(result.get("n1"), params["n1"])
        self.assertEqual(result.get("n2"), params["n2"])
        self.assertEqual(result.get("power"), params["power"])
        self.assertEqual(result.get("alpha"), params["alpha"])
        self.assertEqual(result.get("non_inferiority_margin"), params["non_inferiority_margin"])
        self.assertEqual(result.get("method"), params["method"].capitalize())
        self.assertEqual(result.get("hypothesis_type"), params["hypothesis_type"])
        self.assertEqual(result.get("calculation_type"), params["calculation_type"])
        self.assertEqual(result.get("test_type"), params.get("test_type", "Normal Approximation"))
        self.assertEqual(result.get("design_method"), "Parallel RCT")
        self.assertEqual(result.get("outcome_type_param"), "Binary")

    @unittest.skip("Core function for MDE Non-Inferiority (analytical_binary.min_detectable_effect_binary_non_inferiority) is missing")
    @patch('app.components.parallel_rct.analytical_binary.min_detectable_effect_binary_ni_analytical') # This patch target is problematic but skipped
    def test_calculate_parallel_binary_analytical_non_inferiority_mde_odds_ratio(self, mock_mde_ni_analytical_binary):
        """Test analytical MDE for non-inferiority, binary outcome, odds ratio."""
        params = self.default_params_binary.copy()
        params["calculation_type"] = "Minimum Detectable Effect"
        params["method"] = "Analytical"
        params["hypothesis_type"] = "Non-Inferiority"
        params["p1"] = 0.25  # Control success rate. Odds1 = 0.25/0.75 = 0.333
        params["n1"] = 250
        params["n2"] = 250
        params["power"] = 0.80
        params["alpha"] = 0.025
        params["sides"] = 1
        params["effect_measure"] = "Odds Ratio"
        params["non_inferiority_margin"] = 0.75  # OR_t/c >= 0.75

        # If MDE OR is 0.85, then p2_odds = 0.333 * 0.85 = 0.283. p2 = 0.283 / (1+0.283) = 0.220
        expected_core_result = {"mde_as_ratio": 0.85, "p2_at_mde": 0.220}
        mock_mde_ni_analytical_binary.return_value = expected_core_result

        result = parallel_rct.calculate_parallel_binary(params)

        mock_mde_ni_analytical_binary.assert_called_once_with(
            p1=params["p1"],
            n1=params["n1"],
            n2=params["n2"],
            power=params["power"],
            alpha=params["alpha"],
            non_inferiority_margin=params["non_inferiority_margin"],
            direction="Higher is better",
            test_type="normal approximation",
            correction=False
        )

        self.assertIn("mde_as_ratio", result)
        self.assertAlmostEqual(result["mde_as_ratio"], expected_core_result["mde_as_ratio"], places=5)
        self.assertIn("p2_at_mde", result)
        self.assertAlmostEqual(result["p2_at_mde"], expected_core_result["p2_at_mde"], places=3)
        self.assertNotIn("error", result, f"Unexpected error: {result.get('error')}")
        self.assertEqual(result.get("p1"), params["p1"])
        self.assertEqual(result.get("n1"), params["n1"])
        self.assertEqual(result.get("n2"), params["n2"])
        self.assertEqual(result.get("power"), params["power"])
        self.assertEqual(result.get("alpha"), params["alpha"])
        self.assertEqual(result.get("non_inferiority_margin"), params["non_inferiority_margin"])
        self.assertEqual(result.get("method"), params["method"].capitalize())
        self.assertEqual(result.get("hypothesis_type"), params["hypothesis_type"])
        self.assertEqual(result.get("calculation_type"), params["calculation_type"])
        self.assertEqual(result.get("test_type"), params.get("test_type", "Normal Approximation"))
        self.assertEqual(result.get("design_method"), "Parallel RCT")
        self.assertEqual(result.get("outcome_type_param"), "Binary")

    # Test methods will be appended here by subsequent edits
