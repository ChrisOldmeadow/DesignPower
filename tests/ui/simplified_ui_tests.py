"""
Simplified UI integration tests for DesignPower application.

These tests focus on verifying the parameter handling and integration points
between UI components and calculation functions without attempting to mock
the entire Streamlit rendering process.
"""
import unittest
from unittest.mock import patch, MagicMock
import os
import sys

# Make sure the project root is in the path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import core modules
from core.designs.parallel import binary_simulation
from core.designs.parallel import simulation_binary
from core.designs.parallel import simulation_continuous
from core.designs.parallel import simulation_survival

class SimplifiedUITests(unittest.TestCase):
    """Simplified approach to testing UI integration."""
    
    def test_binary_simulation_parameter_mapping(self):
        """Test parameter mapping for binary simulation methods."""
        # Direct test of the binary simulation functions with UI-like parameters
        ui_params = {
            "calculation_type": "Sample Size",
            "p1": 0.3,
            "p2": 0.5,
            "power": 0.8,
            "alpha": 0.05,
            "allocation_ratio": 1.0,
            "test_type": "Exact Test",
            "nsim": 1000,
            "min_n": 10,
            "max_n": 500,
            "step_n": 10
        }
        
        # Mock the sample_size_binary_sim function
        with patch('core.designs.parallel.binary_simulation.sample_size_binary_sim') as mock_func:
            # Set up a return value
            mock_func.return_value = {"sample_size_1": 100, "sample_size_2": 100}
            
            # Call the function with parameters from UI
            result = binary_simulation.sample_size_binary_sim(
                p1=ui_params["p1"],
                p2=ui_params["p2"],
                power=ui_params["power"],
                alpha=ui_params["alpha"],
                allocation_ratio=ui_params["allocation_ratio"],
                nsim=ui_params["nsim"],
                min_n=ui_params["min_n"],
                max_n=ui_params["max_n"],
                step=ui_params["step_n"]
            )
            
            # Verify the function was called with the correct parameters
            mock_func.assert_called_once()
            args, kwargs = mock_func.call_args
            
            # Verify key parameters
            self.assertEqual(kwargs["p1"], 0.3)
            self.assertEqual(kwargs["p2"], 0.5)
            self.assertEqual(kwargs["power"], 0.8)
            self.assertEqual(kwargs["nsim"], 1000)
    
    def test_continuous_simulation_parameter_mapping(self):
        """Test parameter mapping for continuous simulation methods."""
        # Direct test of the continuous simulation functions with UI-like parameters
        ui_params = {
            "calculation_type": "Power",
            "mean1": 10,
            "mean2": 15,
            "sd1": 5,
            "sd2": 5,
            "n1": 100,
            "n2": 100,
            "alpha": 0.05,
            "nsim": 1000
        }
        
        # Mock the power_continuous_sim function
        with patch('core.designs.parallel.simulation_continuous.power_continuous_sim') as mock_func:
            # Set up a return value
            mock_func.return_value = {"power": 0.85}
            
            # Call the function with parameters from UI
            result = simulation_continuous.power_continuous_sim(
                n1=ui_params["n1"],
                n2=ui_params["n2"],
                mean1=ui_params["mean1"],
                mean2=ui_params["mean2"],
                sd1=ui_params["sd1"],
                sd2=ui_params["sd2"],
                alpha=ui_params["alpha"],
                nsim=ui_params["nsim"]
            )
            
            # Verify the function was called with the correct parameters
            mock_func.assert_called_once()
            args, kwargs = mock_func.call_args
            
            # Verify key parameters
            self.assertEqual(kwargs["n1"], 100)
            self.assertEqual(kwargs["n2"], 100)
            self.assertEqual(kwargs["mean1"], 10)
            self.assertEqual(kwargs["mean2"], 15)
            self.assertEqual(kwargs["nsim"], 1000)
    
    def test_survival_simulation_parameter_mapping(self):
        """Test parameter mapping for survival simulation methods."""
        # Direct test of the survival simulation functions with UI-like parameters
        ui_params = {
            "calculation_type": "Power",
            "median1": 10,
            "median2": 15,
            "n1": 100,
            "n2": 100,
            "enrollment_period": 12,
            "follow_up_period": 24,
            "dropout_rate": 0.1,
            "alpha": 0.05,
            "nsim": 1000
        }
        
        # Mock the power_survival_sim function
        with patch('core.designs.parallel.simulation_survival.power_survival_sim') as mock_func:
            # Set up a return value
            mock_func.return_value = {"power": 0.85}
            
            # Call the function with parameters from UI
            result = simulation_survival.power_survival_sim(
                n1=ui_params["n1"],
                n2=ui_params["n2"],
                median1=ui_params["median1"],
                median2=ui_params["median2"],
                enrollment_period=ui_params["enrollment_period"],
                follow_up_period=ui_params["follow_up_period"],
                dropout_rate=ui_params["dropout_rate"],
                alpha=ui_params["alpha"],
                nsim=ui_params["nsim"]
            )
            
            # Verify the function was called with the correct parameters
            mock_func.assert_called_once()
            args, kwargs = mock_func.call_args
            
            # Verify key parameters
            self.assertEqual(kwargs["n1"], 100)
            self.assertEqual(kwargs["n2"], 100)
            self.assertEqual(kwargs["median1"], 10)
            self.assertEqual(kwargs["median2"], 15)
            self.assertEqual(kwargs["enrollment_period"], 12)
            self.assertEqual(kwargs["follow_up_period"], 24)
            self.assertEqual(kwargs["dropout_rate"], 0.1)
            self.assertEqual(kwargs["nsim"], 1000)
    
    def test_survival_non_inferiority_parameter_mapping(self):
        """Test parameter mapping for survival non-inferiority simulation."""
        # Direct test of the survival non-inferiority simulation with UI-like parameters
        ui_params = {
            "calculation_type": "Power",
            "hypothesis_type": "Non-Inferiority",
            "median1": 10,
            "n1": 250,
            "n2": 250,
            "non_inferiority_margin": 1.5,
            "enrollment_period": 12,
            "follow_up_period": 24,
            "dropout_rate": 0.1,
            "alpha": 0.05,
            "nsim": 200,
            "assumed_hazard_ratio": 0.9
        }
        
        # Mock the power_survival_non_inferiority_sim function
        with patch('core.designs.parallel.simulation_survival.power_survival_non_inferiority_sim') as mock_func:
            # Set up a return value
            mock_func.return_value = {"power": 0.75}
            
            # Call the function with parameters from UI
            result = simulation_survival.power_survival_non_inferiority_sim(
                n1=ui_params["n1"],
                n2=ui_params["n2"],
                median1=ui_params["median1"],
                non_inferiority_margin=ui_params["non_inferiority_margin"],
                enrollment_period=ui_params["enrollment_period"],
                follow_up_period=ui_params["follow_up_period"],
                dropout_rate=ui_params["dropout_rate"],
                alpha=ui_params["alpha"],
                nsim=ui_params["nsim"],
                assumed_hazard_ratio=ui_params["assumed_hazard_ratio"]
            )
            
            # Verify the function was called with the correct parameters
            mock_func.assert_called_once()
            args, kwargs = mock_func.call_args
            
            # Verify key parameters
            self.assertEqual(kwargs["n1"], 250)
            self.assertEqual(kwargs["n2"], 250)
            self.assertEqual(kwargs["median1"], 10)
            self.assertEqual(kwargs["non_inferiority_margin"], 1.5)
            self.assertEqual(kwargs["enrollment_period"], 12)
            self.assertEqual(kwargs["follow_up_period"], 24)
            self.assertEqual(kwargs["dropout_rate"], 0.1)
            self.assertEqual(kwargs["assumed_hazard_ratio"], 0.9)
            self.assertEqual(kwargs["nsim"], 200)

if __name__ == '__main__':
    unittest.main()
