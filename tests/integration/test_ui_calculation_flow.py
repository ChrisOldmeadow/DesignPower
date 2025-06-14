#!/usr/bin/env python3
"""
Integration tests for UI→calculation→result flow.

These tests verify the complete flow from UI parameters through the calculation
functions to final results, ensuring that:
1. UI parameters are correctly mapped to function calls
2. Functions receive the right parameter names and values
3. Results are properly formatted for UI display
4. All calculation types and methods work correctly

This replaces the old mocked UI component tests with real integration tests.
"""

import unittest
import sys
import os

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

# Import the actual calculation functions that the UI uses
from app.components.parallel_rct.calculations import (
    calculate_parallel_binary,
    calculate_parallel_continuous,
    calculate_parallel_survival
)


class TestBinaryCalculationFlow(unittest.TestCase):
    """Test complete flow for binary outcome calculations."""
    
    def setUp(self):
        """Set up common test parameters."""
        self.base_params = {
            "hypothesis_type": "Superiority",
            "p1": 0.3,
            "p2": 0.5,
            "alpha": 0.05,
            "allocation_ratio": 1.0
        }
    
    def test_binary_analytical_sample_size_flow(self):
        """Test analytical sample size calculation complete flow."""
        params = self.base_params.copy()
        params.update({
            "calculation_type": "Sample Size",
            "power": 0.8,
            "method": "analytical",
            "test_type": "Normal Approximation",
            "correction": False
        })
        
        result = calculate_parallel_binary(params)
        
        # Verify result structure
        self.assertIn("n1", result)
        self.assertIn("n2", result)
        self.assertIn("total_n", result)
        self.assertIn("absolute_risk_difference", result)
        self.assertIn("relative_risk", result)
        self.assertIn("odds_ratio", result)
        
        # Verify result values are reasonable
        self.assertGreater(result["n1"], 0)
        self.assertGreater(result["n2"], 0)
        self.assertEqual(result["total_n"], result["n1"] + result["n2"])
        self.assertAlmostEqual(result["absolute_risk_difference"], 0.2, places=3)
        self.assertAlmostEqual(result["relative_risk"], 1.667, places=2)
        
        # With equal allocation, n1 should equal n2
        self.assertEqual(result["n1"], result["n2"])
    
    def test_binary_analytical_power_flow(self):
        """Test analytical power calculation complete flow."""
        params = self.base_params.copy()
        params.update({
            "calculation_type": "Power",
            "n1": 100,
            "n2": 100,
            "method": "analytical",
            "test_type": "Normal Approximation",
            "correction": False
        })
        
        result = calculate_parallel_binary(params)
        
        # Verify result structure
        self.assertIn("power", result)
        self.assertIn("n1", result)
        self.assertIn("n2", result)
        
        # Verify power is reasonable
        power = result["power"]
        self.assertGreaterEqual(power, 0.0)
        self.assertLessEqual(power, 1.0)
        self.assertGreater(power, 0.7)  # Should be reasonably high power with these parameters
    
    def test_binary_simulation_sample_size_flow(self):
        """Test simulation sample size calculation complete flow."""
        params = self.base_params.copy()
        params.update({
            "calculation_type": "Sample Size",
            "power": 0.8,
            "method": "simulation",
            "test_type": "Normal Approximation",
            "nsim": 100,  # Keep low for test speed
            "seed": 42
        })
        
        result = calculate_parallel_binary(params)
        
        # Verify result structure
        self.assertIn("n1", result)
        self.assertIn("n2", result)
        self.assertIn("total_n", result)
        
        # Verify result values are reasonable
        self.assertGreater(result["n1"], 0)
        self.assertGreater(result["n2"], 0)
        
        # Test reproducibility with same seed
        result2 = calculate_parallel_binary(params)
        self.assertEqual(result["n1"], result2["n1"])
    
    def test_binary_simulation_power_flow(self):
        """Test simulation power calculation complete flow."""
        params = self.base_params.copy()
        params.update({
            "calculation_type": "Power",
            "n1": 50,
            "n2": 50,
            "method": "simulation",
            "test_type": "Normal Approximation",
            "nsim": 100,  # Keep low for test speed
            "seed": 42
        })
        
        result = calculate_parallel_binary(params)
        
        # Verify result structure and values
        self.assertIn("power", result)
        power = result["power"]
        self.assertGreaterEqual(power, 0.0)
        self.assertLessEqual(power, 1.0)
    
    def test_binary_different_test_types(self):
        """Test that different test types work correctly."""
        base_params = self.base_params.copy()
        base_params.update({
            "calculation_type": "Power",
            "n1": 50,
            "n2": 50,
            "method": "analytical"
        })
        
        test_types = ["Normal Approximation", "Fisher's Exact Test", "Likelihood Ratio Test"]
        results = {}
        
        for test_type in test_types:
            params = base_params.copy()
            params["test_type"] = test_type
            
            result = calculate_parallel_binary(params)
            results[test_type] = result["power"]
            
            # Each test type should give a valid power
            self.assertGreaterEqual(result["power"], 0.0)
            self.assertLessEqual(result["power"], 1.0)
        
        # Results should be similar but may differ slightly
        powers = list(results.values())
        self.assertLess(max(powers) - min(powers), 0.2)  # Should be within 20%
    
    def test_binary_non_inferiority_flow(self):
        """Test non-inferiority calculation flow."""
        params = {
            "calculation_type": "Sample Size",
            "hypothesis_type": "Non-Inferiority",
            "p1": 0.7,
            "nim": 0.1,  # Non-inferiority margin
            "direction": "Higher is better",
            "power": 0.8,
            "alpha": 0.05,
            "method": "analytical",
            "allocation_ratio": 1.0
        }
        
        result = calculate_parallel_binary(params)
        
        # Verify result structure
        self.assertIn("n1", result)
        self.assertIn("n2", result)
        self.assertIn("total_n", result)
        
        # Verify reasonable sample size
        self.assertGreater(result["n1"], 50)  # NI typically requires larger samples


class TestContinuousCalculationFlow(unittest.TestCase):
    """Test complete flow for continuous outcome calculations."""
    
    def setUp(self):
        """Set up common test parameters."""
        self.base_params = {
            "hypothesis_type": "Superiority",
            "mean1": 0.0,
            "mean2": 1.0,
            "std_dev": 2.0,
            "alpha": 0.05,
            "allocation_ratio": 1.0
        }
    
    def test_continuous_analytical_sample_size_flow(self):
        """Test analytical sample size calculation complete flow."""
        params = self.base_params.copy()
        params.update({
            "calculation_type": "Sample Size",
            "power": 0.8,
            "method": "analytical"
        })
        
        result = calculate_parallel_continuous(params)
        
        # Verify result structure
        expected_keys = ["n1", "n2", "total_n", "mean_difference"]
        for key in expected_keys:
            self.assertIn(key, result)
        
        # Verify result values
        self.assertGreater(result["n1"], 0)
        self.assertGreater(result["n2"], 0)
        self.assertEqual(result["total_n"], result["n1"] + result["n2"])
        self.assertAlmostEqual(result["mean_difference"], 1.0, places=3)
    
    def test_continuous_analytical_power_flow(self):
        """Test analytical power calculation complete flow."""
        params = self.base_params.copy()
        params.update({
            "calculation_type": "Power",
            "n1": 50,
            "n2": 50,
            "method": "analytical"
        })
        
        result = calculate_parallel_continuous(params)
        
        # Verify result structure and values
        self.assertIn("power", result)
        self.assertIn("n1", result)
        self.assertIn("n2", result)
        
        power = result["power"]
        self.assertGreaterEqual(power, 0.0)
        self.assertLessEqual(power, 1.0)
        self.assertGreater(power, 0.5)  # Should be reasonable power
    
    def test_continuous_simulation_sample_size_flow(self):
        """Test simulation sample size calculation complete flow."""
        params = self.base_params.copy()
        params.update({
            "calculation_type": "Sample Size",
            "power": 0.8,
            "method": "simulation",
            "nsim": 100,  # Keep low for test speed
            "min_n": 10,
            "max_n": 200,
            "step_n": 10,
            "seed": 42
        })
        
        result = calculate_parallel_continuous(params)
        
        # Verify result structure
        self.assertIn("n1", result)
        self.assertIn("n2", result)
        self.assertIn("achieved_power", result)
        
        # Verify values
        self.assertGreater(result["n1"], 0)
        self.assertGreaterEqual(result["achieved_power"], 0.7)  # Should be close to target
    
    def test_continuous_non_inferiority_flow(self):
        """Test non-inferiority calculation flow."""
        params = {
            "calculation_type": "Sample Size",
            "hypothesis_type": "Non-Inferiority",
            "mean1": 10.0,
            "non_inferiority_margin": 2.0,
            "assumed_difference": 0.0,
            "std_dev": 5.0,
            "power": 0.8,
            "alpha": 0.05,
            "method": "analytical",
            "non_inferiority_direction": "lower",
            "allocation_ratio": 1.0
        }
        
        result = calculate_parallel_continuous(params)
        
        # Verify result structure
        self.assertIn("n1", result)
        self.assertIn("n2", result)
        self.assertIn("total_n", result)
        
        # NI typically requires larger samples
        self.assertGreater(result["n1"], 30)
    
    def test_continuous_repeated_measures_flow(self):
        """Test repeated measures calculation flow."""
        params = self.base_params.copy()
        params.update({
            "calculation_type": "Power",
            "n1": 30,
            "n2": 30,
            "method": "analytical",
            "repeated_measures": True,
            "correlation": 0.6,
            "analysis_method": "ANCOVA"
        })
        
        result = calculate_parallel_continuous(params)
        
        # Verify result includes repeated measures info
        self.assertIn("power", result)
        self.assertIn("repeated_measures", result)
        self.assertTrue(result["repeated_measures"])
        
        # Power should be higher due to correlation
        self.assertGreater(result["power"], 0.5)


class TestSurvivalCalculationFlow(unittest.TestCase):
    """Test complete flow for survival outcome calculations."""
    
    def setUp(self):
        """Set up common test parameters."""
        self.base_params = {
            "hypothesis_type": "Superiority",
            "median1": 12,  # months
            "median2": 18,  # months
            "enrollment_period": 12,
            "follow_up_period": 24,
            "dropout_rate": 0.1,
            "alpha": 0.05,
            "allocation_ratio": 1.0
        }
    
    def test_survival_analytical_sample_size_flow(self):
        """Test analytical sample size calculation complete flow."""
        params = self.base_params.copy()
        params.update({
            "calculation_type": "Sample Size",
            "power": 0.8,
            "method": "analytical"
        })
        
        result = calculate_parallel_survival(params)
        
        # Verify result structure
        expected_keys = ["n1", "n2", "total_n", "hazard_ratio", "events"]
        for key in expected_keys:
            self.assertIn(key, result)
        
        # Verify result values
        self.assertGreater(result["n1"], 0)
        self.assertGreater(result["n2"], 0)
        self.assertEqual(result["total_n"], result["n1"] + result["n2"])
        self.assertAlmostEqual(result["hazard_ratio"], 0.667, places=2)  # 12/18
    
    def test_survival_analytical_power_flow(self):
        """Test analytical power calculation complete flow."""
        params = self.base_params.copy()
        params.update({
            "calculation_type": "Power",
            "n1": 100,
            "n2": 100,
            "method": "analytical"
        })
        
        result = calculate_parallel_survival(params)
        
        # Verify result structure and values
        self.assertIn("power", result)
        self.assertIn("events", result)
        
        power = result["power"]
        self.assertGreaterEqual(power, 0.0)
        self.assertLessEqual(power, 1.0)
    
    def test_survival_simulation_flow(self):
        """Test simulation calculation complete flow."""
        params = self.base_params.copy()
        params.update({
            "calculation_type": "Power",
            "n1": 50,
            "n2": 50,
            "method": "simulation",
            "nsim": 100,  # Keep low for test speed
            "seed": 42
        })
        
        result = calculate_parallel_survival(params)
        
        # Verify result structure
        self.assertIn("power", result)
        self.assertIn("n1", result)
        self.assertIn("n2", result)
        
        # Verify power is valid
        power = result["power"]
        self.assertGreaterEqual(power, 0.0)
        self.assertLessEqual(power, 1.0)
    
    def test_survival_non_inferiority_flow(self):
        """Test non-inferiority calculation flow."""
        params = {
            "calculation_type": "Sample Size",
            "hypothesis_type": "Non-Inferiority",
            "median1": 15,
            "non_inferiority_margin": 1.25,  # Hazard ratio margin
            "enrollment_period": 12,
            "follow_up_period": 24,
            "dropout_rate": 0.1,
            "power": 0.8,
            "alpha": 0.05,
            "method": "analytical",
            "allocation_ratio": 1.0,
            "assumed_hazard_ratio": 1.0
        }
        
        result = calculate_parallel_survival(params)
        
        # Verify result structure
        self.assertIn("n1", result)
        self.assertIn("n2", result)
        self.assertIn("total_n", result)
        
        # NI typically requires larger samples
        self.assertGreater(result["n1"], 50)


class TestParameterValidationFlow(unittest.TestCase):
    """Test parameter validation and error handling."""
    
    def test_invalid_proportions_handling(self):
        """Test handling of invalid proportion values."""
        params = {
            "calculation_type": "Power",
            "hypothesis_type": "Superiority",
            "p1": 1.5,  # Invalid: > 1
            "p2": 0.5,
            "n1": 50,
            "n2": 50,
            "alpha": 0.05,
            "method": "analytical"
        }
        
        # Should handle gracefully or raise appropriate error
        with self.assertRaises((ValueError, TypeError)):
            calculate_parallel_binary(params)
    
    def test_missing_required_parameters(self):
        """Test handling of missing required parameters."""
        params = {
            "calculation_type": "Power",
            "hypothesis_type": "Superiority",
            # Missing p1, p2
            "n1": 50,
            "n2": 50,
            "alpha": 0.05,
            "method": "analytical"
        }
        
        # Should handle gracefully or raise appropriate error
        with self.assertRaises((KeyError, ValueError, TypeError)):
            calculate_parallel_binary(params)
    
    def test_zero_sample_sizes(self):
        """Test handling of zero or negative sample sizes."""
        params = {
            "calculation_type": "Power",
            "hypothesis_type": "Superiority",
            "p1": 0.3,
            "p2": 0.5,
            "n1": 0,  # Invalid
            "n2": 50,
            "alpha": 0.05,
            "method": "analytical"
        }
        
        # Should handle gracefully or raise appropriate error
        with self.assertRaises((ValueError, TypeError)):
            calculate_parallel_binary(params)


class TestConsistencyBetweenMethods(unittest.TestCase):
    """Test consistency between analytical and simulation methods."""
    
    def test_binary_analytical_vs_simulation_consistency(self):
        """Test that analytical and simulation give similar results."""
        base_params = {
            "calculation_type": "Power",
            "hypothesis_type": "Superiority",
            "p1": 0.3,
            "p2": 0.5,
            "n1": 100,
            "n2": 100,
            "alpha": 0.05
        }
        
        # Analytical result
        analytical_params = base_params.copy()
        analytical_params.update({"method": "analytical", "test_type": "Normal Approximation"})
        analytical_result = calculate_parallel_binary(analytical_params)
        
        # Simulation result with many simulations for accuracy
        simulation_params = base_params.copy()
        simulation_params.update({"method": "simulation", "nsim": 5000, "seed": 42})
        simulation_result = calculate_parallel_binary(simulation_params)
        
        # Results should be close (within 5%)
        analytical_power = analytical_result["power"]
        simulation_power = simulation_result["power"]
        
        self.assertLess(abs(analytical_power - simulation_power), 0.05,
                       f"Analytical power {analytical_power:.3f} vs simulation power {simulation_power:.3f}")
    
    def test_continuous_analytical_vs_simulation_consistency(self):
        """Test consistency for continuous outcomes."""
        base_params = {
            "calculation_type": "Power",
            "hypothesis_type": "Superiority",
            "mean1": 0.0,
            "mean2": 0.5,
            "std_dev": 1.0,
            "n1": 50,
            "n2": 50,
            "alpha": 0.05
        }
        
        # Analytical result
        analytical_params = base_params.copy()
        analytical_params["method"] = "analytical"
        analytical_result = calculate_parallel_continuous(analytical_params)
        
        # Simulation result
        simulation_params = base_params.copy()
        simulation_params.update({"method": "simulation", "nsim": 2000, "seed": 42})
        simulation_result = calculate_parallel_continuous(simulation_params)
        
        # Results should be close
        analytical_power = analytical_result["power"]
        simulation_power = simulation_result["power"]
        
        self.assertLess(abs(analytical_power - simulation_power), 0.08,
                       f"Analytical power {analytical_power:.3f} vs simulation power {simulation_power:.3f}")


class TestReportAndCLIGeneration(unittest.TestCase):
    """Test HTML report and CLI script generation don't crash."""
    
    def test_html_report_generation_continuous(self):
        """Test HTML report generation for continuous outcomes."""
        from core.utils.report_generator import generate_report
        from app.components.parallel_rct.calculations import calculate_parallel_continuous
        
        params = {
            'calculation_type': 'Power',
            'hypothesis_type': 'Superiority',
            'mean1': 10.0,
            'mean2': 12.0,
            'std_dev': 3.0,
            'n1': 50,
            'n2': 50,
            'alpha': 0.05,
            'allocation_ratio': 1.0,
            'method': 'analytical'
        }
        
        # Get real calculation results
        results = calculate_parallel_continuous(params)
        
        # This should not crash with format string errors
        try:
            report = generate_report(results, params, 'Parallel RCT', 'Continuous Outcome')
            self.assertIsInstance(report, str)
            self.assertGreater(len(report), 0)
            # Should not contain None in formatted output
            self.assertNotIn('None', report)
        except Exception as e:
            self.fail(f"HTML report generation crashed: {e}")


if __name__ == "__main__":
    # Run all tests
    unittest.main(verbosity=2)