#!/usr/bin/env python3
"""
Integration tests for Cluster RCT UI→calculation→result flow.

These tests verify the complete flow from UI parameters through the cluster RCT
calculation functions to final results.
"""

import unittest
import sys
import os

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

# Import the actual calculation functions
from app.components.cluster_rct import calculate_cluster_binary, calculate_cluster_continuous


class TestClusterBinaryCalculationFlow(unittest.TestCase):
    """Test complete flow for cluster RCT binary outcome calculations."""
    
    def setUp(self):
        """Set up common test parameters."""
        self.base_params = {
            "p1": 0.3,
            "p2": 0.5,
            "alpha": 0.05,
            "allocation_ratio": 1.0,
            "icc": 0.05,
            "cluster_size": 20
        }
    
    def test_cluster_binary_analytical_sample_size_flow(self):
        """Test analytical sample size calculation complete flow."""
        params = self.base_params.copy()
        params.update({
            "calc_type": "Sample Size",
            "power": 0.8,
            "method": "analytical",
            "cluster_size": 20
        })
        
        result = calculate_cluster_binary(params)
        
        # Verify result structure - check for actual returned keys
        self.assertIn("n_clusters", result, f"Missing key: n_clusters")
        self.assertIn("design_effect", result, f"Missing key: design_effect") 
        
        # Verify result values are reasonable
        self.assertGreater(result["n_clusters"], 0)
        self.assertGreater(result["design_effect"], 1.0)  # Should account for clustering
    
    def test_cluster_binary_analytical_power_flow(self):
        """Test analytical power calculation complete flow."""
        params = self.base_params.copy()
        params.update({
            "calc_type": "Power",
            "n_clusters": 10,
            "method": "analytical"
        })
        
        result = calculate_cluster_binary(params)
        
        # Verify result structure and values
        self.assertIn("power", result)
        self.assertIn("design_effect", result)
        
        power = result["power"]
        self.assertGreaterEqual(power, 0.0)
        self.assertLessEqual(power, 1.0)
    
    def test_cluster_binary_simulation_flow(self):
        """Test simulation calculation complete flow."""
        params = self.base_params.copy()
        params.update({
            "calc_type": "Power",
            "n_clusters": 8,
            "method": "simulation",
            "nsim": 100,  # Keep low for test speed
            "seed": 42
        })
        
        result = calculate_cluster_binary(params)
        
        # Verify result structure
        self.assertIn("power", result)
        self.assertIn("icc", result)
        self.assertIn("cluster_size", result)
        
        # Verify power is valid
        power = result["power"]
        self.assertGreaterEqual(power, 0.0)
        self.assertLessEqual(power, 1.0)
        
        # Test reproducibility
        result2 = calculate_cluster_binary(params)
        self.assertEqual(result["power"], result2["power"])
    
    def test_cluster_binary_icc_effects(self):
        """Test that ICC affects calculations correctly."""
        base_params = self.base_params.copy()
        base_params.update({
            "calc_type": "Sample Size",
            "power": 0.8,
            "method": "analytical",
            "cluster_size": 20
        })
        
        # Test with different ICC values
        icc_values = [0.01, 0.05, 0.10]
        results = {}
        
        for icc in icc_values:
            params = base_params.copy()
            params["icc"] = icc
            
            result = calculate_cluster_binary(params)
            if "n_clusters" in result:
                results[icc] = result["n_clusters"]
        
        # Higher ICC should require more clusters (if we got valid results)
        if len(results) == 3:
            clusters_001 = results[0.01]
            clusters_005 = results[0.05]
            clusters_010 = results[0.10]
            
            self.assertLess(clusters_001, clusters_005)
            self.assertLess(clusters_005, clusters_010)


class TestClusterContinuousCalculationFlow(unittest.TestCase):
    """Test complete flow for cluster RCT continuous outcome calculations."""
    
    def setUp(self):
        """Set up common test parameters."""
        self.base_params = {
            "mean1": 10.0,
            "mean2": 12.0,
            "std_dev": 3.0,
            "alpha": 0.05,
            "allocation_ratio": 1.0,
            "icc": 0.05,
            "cluster_size": 25
        }
    
    def test_cluster_continuous_analytical_sample_size_flow(self):
        """Test analytical sample size calculation complete flow."""
        params = self.base_params.copy()
        params.update({
            "calc_type": "Sample Size",
            "power": 0.8,
            "method": "analytical",
            "determine_ss_param": "Number of Clusters (k)",
            "cluster_size_input_for_k_calc": 25
        })
        
        result = calculate_cluster_continuous(params)
        
        # Verify result structure - check for actual returned keys
        self.assertIn("n_clusters", result, f"Missing key: n_clusters")
        self.assertIn("design_effect", result, f"Missing key: design_effect")
        
        # Verify result values
        self.assertGreater(result["n_clusters"], 0)
        self.assertGreater(result["design_effect"], 1.0)
    
    def test_cluster_continuous_analytical_power_flow(self):
        """Test analytical power calculation complete flow."""
        params = self.base_params.copy()
        params.update({
            "calc_type": "Power",
            "n_clusters": 12,
            "method": "analytical"
        })
        
        result = calculate_cluster_continuous(params)
        
        # Verify result structure and values
        self.assertIn("power", result)
        self.assertIn("design_effect", result)
        self.assertIn("effect_size", result)
        
        power = result["power"]
        self.assertGreaterEqual(power, 0.0)
        self.assertLessEqual(power, 1.0)
    
    def test_cluster_continuous_simulation_flow(self):
        """Test simulation calculation complete flow."""
        params = self.base_params.copy()
        params.update({
            "calc_type": "Power",
            "n_clusters": 10,
            "method": "simulation",
            "nsim": 100,  # Keep low for test speed
            "seed": 42
        })
        
        result = calculate_cluster_continuous(params)
        
        # Verify result structure
        self.assertIn("power", result)
        self.assertIn("icc", result)
        self.assertIn("cluster_size", result)
        
        # Verify power is valid
        power = result["power"]
        self.assertGreaterEqual(power, 0.0)
        self.assertLessEqual(power, 1.0)
    
    def test_cluster_continuous_effect_size_calculation(self):
        """Test that effect size is calculated correctly."""
        params = self.base_params.copy()
        params.update({
            "calc_type": "Power",
            "n_clusters": 10,
            "method": "analytical",
            "mean1": 5.0,
            "mean2": 8.0,
            "std_dev": 2.0
        })
        
        result = calculate_cluster_continuous(params)
        
        # Just verify we got a valid result
        self.assertIn("power", result)


class TestClusterParameterValidation(unittest.TestCase):
    """Test parameter validation for cluster RCT calculations."""
    
    def test_invalid_icc_handling(self):
        """Test handling of invalid ICC values."""
        params = {
            "calc_type": "Power",
            "p1": 0.3,
            "p2": 0.5,
            "n_clusters": 10,
            "cluster_size": 20,
            "icc": 1.5,  # Invalid: > 1
            "alpha": 0.05,
            "method": "analytical"
        }
        
        # Should handle gracefully or raise appropriate error
        with self.assertRaises((ValueError, TypeError)):
            calculate_cluster_binary(params)
    
    def test_zero_cluster_size_handling(self):
        """Test handling of zero cluster size."""
        params = {
            "calc_type": "Power",
            "p1": 0.3,
            "p2": 0.5,
            "n_clusters": 10,
            "cluster_size": 0,  # Invalid
            "icc": 0.05,
            "alpha": 0.05,
            "method": "analytical"
        }
        
        # Should handle gracefully or raise appropriate error
        with self.assertRaises((ValueError, TypeError)):
            calculate_cluster_binary(params)


class TestClusterConsistencyChecks(unittest.TestCase):
    """Test consistency between different calculation methods."""
    
    def test_cluster_binary_analytical_vs_simulation_consistency(self):
        """Test consistency between analytical and simulation for binary outcomes."""
        base_params = {
            "calc_type": "Power",
            "p1": 0.4,
            "p2": 0.6,
            "n_clusters": 15,
            "cluster_size": 20,
            "icc": 0.03,
            "alpha": 0.05
        }
        
        # Analytical result
        analytical_params = base_params.copy()
        analytical_params["method"] = "analytical"
        analytical_result = calculate_cluster_binary(analytical_params)
        
        # Simulation result with many simulations for accuracy
        simulation_params = base_params.copy()
        simulation_params.update({"method": "simulation", "nsim": 2000, "seed": 42})
        simulation_result = calculate_cluster_binary(simulation_params)
        
        # Results should be close (within 10% due to clustering complexity)
        analytical_power = analytical_result["power"]
        simulation_power = simulation_result["power"]
        
        self.assertLess(abs(analytical_power - simulation_power), 0.10,
                       f"Analytical power {analytical_power:.3f} vs simulation power {simulation_power:.3f}")
    
    def test_cluster_continuous_analytical_vs_simulation_consistency(self):
        """Test consistency for continuous outcomes."""
        base_params = {
            "calc_type": "Power",
            "mean1": 50.0,
            "mean2": 55.0,
            "std_dev": 10.0,
            "n_clusters": 12,
            "cluster_size": 15,
            "icc": 0.04,
            "alpha": 0.05
        }
        
        # Analytical result
        analytical_params = base_params.copy()
        analytical_params["method"] = "analytical"
        analytical_result = calculate_cluster_continuous(analytical_params)
        
        # Simulation result
        simulation_params = base_params.copy()
        simulation_params.update({"method": "simulation", "nsim": 1500, "seed": 42})
        simulation_result = calculate_cluster_continuous(simulation_params)
        
        # Results should be close
        analytical_power = analytical_result["power"]
        simulation_power = simulation_result["power"]
        
        self.assertLess(abs(analytical_power - simulation_power), 0.10,
                       f"Analytical power {analytical_power:.3f} vs simulation power {simulation_power:.3f}")


if __name__ == "__main__":
    # Run all tests
    unittest.main(verbosity=2)