"""
Comprehensive fix for parallel RCT component tests.

This test file correctly mocks the actual function names and parameter mappings
used in the parallel_rct component implementation.
"""
import unittest
from unittest.mock import patch, MagicMock
from app.components import parallel_rct


class TestParallelRCTComponentFixed(unittest.TestCase):
    """Fixed tests that match the actual implementation."""
    
    def setUp(self):
        """Set up common test parameters."""
        # Continuous outcome parameters
        self.continuous_params = {
            "calculation_type": "Power",
            "method": "analytical",
            "hypothesis_type": "Superiority",
            "mean1": 10.0,
            "mean2": 12.0,
            "std_dev": 5.0,
            "n1": 50,
            "n2": 50,
            "alpha": 0.05,
            "allocation_ratio": 1.0
        }
        
        # Binary outcome parameters
        self.binary_params = {
            "calculation_type": "Sample Size",
            "method": "analytical",
            "hypothesis_type": "Superiority",
            "p1": 0.3,
            "p2": 0.5,
            "power": 0.8,
            "alpha": 0.05,
            "allocation_ratio": 1.0,
            "test_type": "Normal Approximation",
            "correction": False
        }
        
        # Survival outcome parameters
        self.survival_params = {
            "calculation_type": "Power",
            "method": "analytical",
            "hypothesis_type": "Superiority",
            "median_survival1": 12.0,
            "hr": 0.75,
            "accrual_time": 12.0,
            "follow_up_time": 18.0,
            "dropout_rate": 0.1,
            "alpha": 0.05,
            "n1": 100,
            "n2": 100,
            "sides": 2,
            "allocation_ratio": 1.0
        }
    
    # ==================== CONTINUOUS OUTCOME TESTS ====================
    
    def test_continuous_analytical_power(self):
        """Test analytical power calculation for continuous outcomes."""
        params = self.continuous_params.copy()
        
        with patch('app.components.parallel_rct.analytical_continuous.power_continuous') as mock_power:
            expected_result = {
                "power": 0.642,
                "effect_size": 0.4,
                "n1": 50,
                "n2": 50
            }
            mock_power.return_value = expected_result
            
            result = parallel_rct.calculate_parallel_continuous(params)
            
            # Verify the mock was called with correct parameters
            mock_power.assert_called_once_with(
                n1=50,
                n2=50,
                mean1=10.0,
                mean2=12.0,
                sd1=5.0,
                sd2=5.0,  # Default when unequal_var is False
                alpha=0.05
            )
            
            # Verify results
            self.assertIn("power", result)
            self.assertEqual(result["power"], expected_result["power"])
    
    def test_continuous_simulation_sample_size(self):
        """Test simulation sample size calculation for continuous outcomes."""
        params = self.continuous_params.copy()
        params["calculation_type"] = "Sample Size"
        params["method"] = "simulation"
        params["use_simulation"] = True  # Required flag
        params["power"] = 0.8
        params["nsim"] = 1000
        params["seed"] = 42
        
        with patch('app.components.parallel_rct.simulation_continuous.sample_size_continuous_sim') as mock_ss:
            expected_result = {
                "n1": 64,
                "n2": 64,
                "sample_size_1": 64,
                "sample_size_2": 64,
                "total_sample_size": 128,
                "delta": 2.0
            }
            mock_ss.return_value = expected_result
            
            result = parallel_rct.calculate_parallel_continuous(params)
            
            # Verify the mock was called with correct parameters
            mock_ss.assert_called_once()
            call_args = mock_ss.call_args[1]
            self.assertEqual(call_args["delta"], 2.0)  # |12-10|
            self.assertEqual(call_args["std_dev"], 5.0)
            self.assertEqual(call_args["power"], 0.8)
            self.assertEqual(call_args["nsim"], 1000)
            
            # Verify results
            self.assertEqual(result["n1"], 64)
            self.assertEqual(result["n2"], 64)
            self.assertEqual(result["total_n"], 128)
    
    def test_continuous_non_inferiority_power(self):
        """Test non-inferiority power calculation for continuous outcomes."""
        params = self.continuous_params.copy()
        params["hypothesis_type"] = "Non-Inferiority"
        params["non_inferiority_margin"] = 1.0
        params["assumed_difference"] = 0.0
        params["non_inferiority_direction"] = "lower"
        
        with patch('app.components.parallel_rct.analytical_continuous.power_continuous_non_inferiority') as mock_ni:
            expected_result = {"power": 0.85}
            mock_ni.return_value = expected_result
            
            result = parallel_rct.calculate_parallel_continuous(params)
            
            # Verify the mock was called
            mock_ni.assert_called_once()
            self.assertEqual(result["power"], 0.85)
    
    # ==================== BINARY OUTCOME TESTS ====================
    
    def test_binary_analytical_sample_size(self):
        """Test analytical sample size calculation for binary outcomes."""
        params = self.binary_params.copy()
        
        with patch('app.components.parallel_rct.analytical_binary.sample_size_binary') as mock_ss:
            expected_result = {
                "sample_size_1": 47,
                "sample_size_2": 47,
                "total_sample_size": 94
            }
            mock_ss.return_value = expected_result
            
            result = parallel_rct.calculate_parallel_binary(params)
            
            # Verify the mock was called with correct parameters
            mock_ss.assert_called_once_with(
                p1=0.3,
                p2=0.5,
                power=0.8,
                alpha=0.05,
                allocation_ratio=1.0,
                test_type="normal approximation",  # Mapped from UI
                correction=False
            )
            
            # Verify results
            self.assertEqual(result["n1"], 47)
            self.assertEqual(result["n2"], 47)
            self.assertEqual(result["total_n"], 94)
    
    def test_binary_simulation_power(self):
        """Test simulation power calculation for binary outcomes."""
        params = self.binary_params.copy()
        params["calculation_type"] = "Power"
        params["method"] = "simulation"
        params["n1"] = 50
        params["n2"] = 50
        params["nsim"] = 1000
        params["seed"] = 42
        params["test_type"] = "Fisher's Exact Test"
        
        with patch('app.components.parallel_rct.simulation_binary.power_binary_sim') as mock_power:
            expected_result = {"power": 0.823}
            mock_power.return_value = expected_result
            
            result = parallel_rct.calculate_parallel_binary(params)
            
            # Verify the mock was called with correct parameters
            mock_power.assert_called_once()
            call_args = mock_power.call_args[1]
            self.assertEqual(call_args["test_type"], "fishers_exact")  # Mapped
            self.assertEqual(call_args["nsim"], 1000)
            
            # Verify results
            self.assertEqual(result["power"], 0.823)
    
    def test_binary_mde_calculation(self):
        """Test minimum detectable effect for binary outcomes."""
        params = self.binary_params.copy()
        params["calculation_type"] = "Minimum Detectable Effect"
        params["n1"] = 100
        params["n2"] = 100
        params["power"] = 0.8
        
        with patch('app.components.parallel_rct.analytical_binary.min_detectable_effect_binary') as mock_mde:
            expected_result = {"p2": 0.45, "minimum_detectable_p2": 0.45}
            mock_mde.return_value = expected_result
            
            result = parallel_rct.calculate_parallel_binary(params)
            
            # Verify results
            self.assertEqual(result["mde"], 0.15)  # 0.45 - 0.3
            self.assertEqual(result["p2_mde"], 0.45)
    
    # ==================== SURVIVAL OUTCOME TESTS ====================
    
    def test_survival_analytical_power(self):
        """Test analytical power calculation for survival outcomes."""
        params = self.survival_params.copy()
        
        with patch('app.components.parallel_rct.analytical_survival.power_survival') as mock_power:
            expected_result = {
                "power": 0.85,
                "events": 50,
                "total_events": 50
            }
            mock_power.return_value = expected_result
            
            result = parallel_rct.calculate_parallel_survival(params)
            
            # Verify the mock was called with correct parameters
            mock_power.assert_called_once_with(
                n1=100,
                n2=100,
                median1=12.0,
                median2=16.0,  # Calculated from median1/hr = 12/0.75
                enrollment_period=12.0,
                follow_up_period=18.0,
                dropout_rate=0.1,
                alpha=0.05,
                sides=2
            )
            
            # Verify results
            self.assertEqual(result["power"], 0.85)
            self.assertEqual(result["events"], 50)
    
    def test_survival_simulation_sample_size(self):
        """Test simulation sample size for survival outcomes."""
        params = self.survival_params.copy()
        params["calculation_type"] = "Sample Size"
        params["method"] = "simulation"
        params["power"] = 0.8
        params["nsim"] = 500
        params["seed"] = 42
        
        with patch('app.components.parallel_rct.simulation_survival.sample_size_survival_sim') as mock_ss:
            expected_result = {
                "n1": 150,
                "n2": 150,
                "sample_size_1": 150,
                "sample_size_2": 150,
                "events": 120,
                "total_events": 120
            }
            mock_ss.return_value = expected_result
            
            result = parallel_rct.calculate_parallel_survival(params)
            
            # Verify the mock was called
            mock_ss.assert_called_once()
            call_args = mock_ss.call_args[1]
            self.assertEqual(call_args["median1"], 12.0)
            self.assertEqual(call_args["median2"], 16.0)
            self.assertEqual(call_args["power"], 0.8)
            
            # Verify results
            self.assertEqual(result["n1"], 150)
            self.assertEqual(result["n2"], 150)
            self.assertEqual(result["total_n"], 300)
    
    def test_survival_non_inferiority_power(self):
        """Test non-inferiority power for survival outcomes."""
        params = self.survival_params.copy()
        params["hypothesis_type"] = "Non-Inferiority"
        params["non_inferiority_margin_hr"] = 1.3
        params["assumed_true_hr"] = 1.0
        
        with patch('app.components.parallel_rct.analytical_survival.power_survival_non_inferiority') as mock_ni:
            expected_result = {
                "power": 0.90,
                "events": 80
            }
            mock_ni.return_value = expected_result
            
            result = parallel_rct.calculate_parallel_survival(params)
            
            # Verify the mock was called with correct parameters
            mock_ni.assert_called_once()
            call_args = mock_ni.call_args[1]
            self.assertEqual(call_args["non_inferiority_margin"], 1.3)
            self.assertEqual(call_args["assumed_hazard_ratio"], 1.0)
            
            # Verify results
            self.assertEqual(result["power"], 0.90)
    
    # ==================== EDGE CASES AND ERROR HANDLING ====================
    
    def test_continuous_repeated_measures(self):
        """Test continuous calculation with repeated measures."""
        params = self.continuous_params.copy()
        params["repeated_measures"] = True
        params["correlation"] = 0.5
        params["analysis_method"] = "ANCOVA"
        params["calculation_type"] = "Sample Size"
        params["power"] = 0.8
        
        with patch('app.components.parallel_rct.analytical_continuous.sample_size_repeated_measures') as mock_rm:
            expected_result = {
                "n1": 32,
                "n2": 32,
                "sample_size_1": 32,
                "sample_size_2": 32
            }
            mock_rm.return_value = expected_result
            
            result = parallel_rct.calculate_parallel_continuous(params)
            
            # Verify the mock was called
            mock_rm.assert_called_once()
            self.assertEqual(result["n1"], 32)
    
    def test_method_case_insensitivity(self):
        """Test that method parameter is case-insensitive."""
        params = self.continuous_params.copy()
        params["method"] = "Analytical"  # Capital A
        
        with patch('app.components.parallel_rct.analytical_continuous.power_continuous') as mock_power:
            mock_power.return_value = {"power": 0.8}
            
            result = parallel_rct.calculate_parallel_continuous(params)
            
            # Should still work after our fix
            mock_power.assert_called_once()
            self.assertEqual(result["power"], 0.8)


if __name__ == '__main__':
    unittest.main()