"""
Fixed test file for parallel RCT component - demonstrates correct mocking pattern.

This file shows how to properly mock the actual functions that exist in the codebase
rather than expecting functions with _analytical suffixes that don't exist.
"""
import unittest
from unittest.mock import patch, MagicMock
from app.components import parallel_rct


class TestParallelRCTComponentFixed(unittest.TestCase):
    """Fixed tests that properly mock actual function names."""
    
    def setUp(self):
        """Set up default parameters for testing."""
        self.default_params_survival = {
            "calculation_type": "Power",
            "method": "Analytical",
            "hypothesis_type": "Superiority",
            "median_survival1": 12,
            "hr": 0.75,
            "accrual_time": 12,
            "follow_up_time": 18,
            "dropout_rate": 0.1,
            "alpha": 0.05,
            "power": 0.8,
            "n1": 100,
            "n2": 100,
            "sides": 2,
            "allocation_ratio": 1.0,
            "non_inferiority_margin_hr": 1.3,
            "assumed_true_hr": 1.0,
            "nsim": 100,
            "seed": 42
        }
    
    def test_analytical_superiority_power_FIXED(self):
        """Fixed test for analytical power calculation - superiority."""
        params = self.default_params_survival.copy()
        params["calculation_type"] = "Power"
        params["method"] = "analytical"  # lowercase!
        params["hypothesis_type"] = "Superiority"
        
        # Mock the ACTUAL function name (power_survival, not power_survival_analytical)
        with patch('app.components.parallel_rct.analytical_survival.power_survival') as mock_power:
            expected_result = {
                "power": 0.85,
                "events": 50,
                "n1": params["n1"],
                "n2": params["n2"]
            }
            mock_power.return_value = expected_result
            
            result = parallel_rct.calculate_parallel_survival(params)
            
            # Check the function was called with correct parameters
            # Note: actual parameter names from looking at the implementation
            mock_power.assert_called_once_with(
                n1=params["n1"],
                n2=params["n2"],
                median1=params["median_survival1"],  # Note: median1, not median_survival1
                median2=params["median_survival1"] / params["hr"],  # Calculated from hr
                enrollment_period=params["accrual_time"],  # Note: enrollment_period, not accrual_time
                follow_up_period=params["follow_up_time"],  # Note: follow_up_period, not follow_up_time
                dropout_rate=params["dropout_rate"],
                alpha=params["alpha"],
                sides=params["sides"]
            )
            
            # Verify results
            self.assertIn("power", result)
            self.assertEqual(result["power"], expected_result["power"])
            self.assertNotIn("error", result)
    
    def test_analytical_non_inferiority_power_FIXED(self):
        """Fixed test for analytical power calculation - non-inferiority."""
        params = self.default_params_survival.copy()
        params["calculation_type"] = "Power"
        params["method"] = "analytical"  # lowercase!
        params["hypothesis_type"] = "Non-Inferiority"
        params["alpha"] = 0.025  # One-sided for NI
        
        # Mock the ACTUAL function name
        with patch('app.components.parallel_rct.analytical_survival.power_survival_non_inferiority') as mock_power_ni:
            expected_result = {
                "power": 0.90,
                "events": 70,
                "n1": params["n1"],
                "n2": params["n2"]
            }
            mock_power_ni.return_value = expected_result
            
            result = parallel_rct.calculate_parallel_survival(params)
            
            # Check the function was called with correct parameters
            mock_power_ni.assert_called_once_with(
                n1=params["n1"],
                n2=params["n2"],
                median1=params["median_survival1"],  # Note: median1, not median_survival1
                non_inferiority_margin=params["non_inferiority_margin_hr"],  # Note: different param name
                enrollment_period=params["accrual_time"],
                follow_up_period=params["follow_up_time"],
                dropout_rate=params["dropout_rate"],
                alpha=params["alpha"],
                assumed_hazard_ratio=params["assumed_true_hr"]  # Note: assumed_hazard_ratio, not assumed_true_hr
            )
            
            # Verify results
            self.assertIn("power", result)
            self.assertEqual(result["power"], expected_result["power"])
    
    def test_simulation_superiority_power_FIXED(self):
        """Fixed test for simulation power calculation."""
        params = self.default_params_survival.copy()
        params["calculation_type"] = "Power"
        params["method"] = "simulation"  # lowercase!
        params["hypothesis_type"] = "Superiority"
        
        # Mock the ACTUAL function name (no _superiority suffix)
        with patch('app.components.parallel_rct.simulation_survival.power_survival_sim') as mock_power_sim:
            expected_result = {
                "power": 0.82,
                "events": 48,
                "nsim": params["nsim"]
            }
            mock_power_sim.return_value = expected_result
            
            result = parallel_rct.calculate_parallel_survival(params)
            
            # Check the function was called with correct parameters
            mock_power_sim.assert_called_once_with(
                n1=params["n1"],
                n2=params["n2"],
                median1=params["median_survival1"],
                median2=params["median_survival1"] / params["hr"],
                enrollment_period=params["accrual_time"],
                follow_up_period=params["follow_up_time"],
                dropout_rate=params["dropout_rate"],
                nsim=params["nsim"],
                alpha=params["alpha"],
                seed=params["seed"],
                sides=params["sides"]
            )
            
            # Verify results
            self.assertIn("power", result)
            self.assertEqual(result["power"], expected_result["power"])


if __name__ == '__main__':
    unittest.main()