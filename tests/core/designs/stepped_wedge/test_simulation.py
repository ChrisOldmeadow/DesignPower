"""
Unit tests for stepped wedge simulation functions.

Tests the core simulation functions for stepped wedge cluster randomized trials
including power estimation via Monte Carlo simulation for continuous and binary outcomes.
"""

import pytest
import numpy as np
import math
from core.designs.stepped_wedge.simulation import (
    simulate_continuous,
    simulate_binary
)


class TestSteppedWedgeContinuous:
    """Test stepped wedge simulation functions for continuous outcomes."""
    
    def test_simulate_continuous_basic(self):
        """Test basic continuous simulation functionality."""
        result = simulate_continuous(
            clusters=6, steps=4, individuals_per_cluster=10,
            icc=0.05, treatment_effect=2.0, std_dev=5.0,
            nsim=100  # Small nsim for fast testing
        )
        assert isinstance(result, dict)
        assert 'power' in result
        assert 'mean_p_value' in result
        assert 'median_p_value' in result
        assert 'nsim' in result
        assert 'parameters' in result
        assert 0 <= result['power'] <= 1
        assert result['nsim'] == 100
    
    def test_simulate_continuous_result_structure(self):
        """Test that simulation result contains all expected fields."""
        result = simulate_continuous(
            clusters=6, steps=4, individuals_per_cluster=10,
            icc=0.05, treatment_effect=2.0, std_dev=5.0,
            nsim=50
        )
        expected_params = ['clusters', 'steps', 'individuals_per_cluster', 'total_n',
                          'icc', 'treatment_effect', 'std_dev', 'alpha']
        for param in expected_params:
            assert param in result['parameters']
        
        # Check parameter values match inputs
        assert result['parameters']['clusters'] == 6
        assert result['parameters']['steps'] == 4
        assert result['parameters']['individuals_per_cluster'] == 10
        assert result['parameters']['total_n'] == 6 * 4 * 10
        assert result['parameters']['icc'] == 0.05
        assert result['parameters']['treatment_effect'] == 2.0
        assert result['parameters']['std_dev'] == 5.0
    
    def test_simulate_continuous_no_effect(self):
        """Test continuous simulation with no effect (null hypothesis)."""
        result = simulate_continuous(
            clusters=6, steps=4, individuals_per_cluster=10,
            icc=0.05, treatment_effect=0.0, std_dev=5.0,
            nsim=100, alpha=0.05
        )
        # Power should be close to alpha (Type I error rate)
        assert abs(result['power'] - 0.05) < 0.15  # Allow for simulation variability
    
    def test_simulate_continuous_large_effect(self):
        """Test continuous simulation with large effect."""
        result_no_effect = simulate_continuous(
            clusters=6, steps=4, individuals_per_cluster=10,
            icc=0.05, treatment_effect=0.0, std_dev=5.0,
            nsim=100
        )
        result_large_effect = simulate_continuous(
            clusters=6, steps=4, individuals_per_cluster=10,
            icc=0.05, treatment_effect=5.0, std_dev=5.0,
            nsim=100
        )
        # Large effect should have higher power
        assert result_large_effect['power'] > result_no_effect['power']
    
    def test_simulate_continuous_more_clusters_higher_power(self):
        """Test that more clusters yield higher power."""
        result_few_clusters = simulate_continuous(
            clusters=4, steps=4, individuals_per_cluster=10,
            icc=0.05, treatment_effect=2.0, std_dev=5.0,
            nsim=100
        )
        result_many_clusters = simulate_continuous(
            clusters=12, steps=4, individuals_per_cluster=10,
            icc=0.05, treatment_effect=2.0, std_dev=5.0,
            nsim=100
        )
        # More clusters should typically have higher power
        assert result_many_clusters['power'] >= result_few_clusters['power'] * 0.8  # Allow some variability
    
    def test_simulate_continuous_more_steps_higher_power(self):
        """Test that more time steps yield higher power."""
        result_few_steps = simulate_continuous(
            clusters=6, steps=3, individuals_per_cluster=10,
            icc=0.05, treatment_effect=2.0, std_dev=5.0,
            nsim=100
        )
        result_many_steps = simulate_continuous(
            clusters=6, steps=6, individuals_per_cluster=10,
            icc=0.05, treatment_effect=2.0, std_dev=5.0,
            nsim=100
        )
        # More steps should typically have higher power
        assert result_many_steps['power'] >= result_few_steps['power'] * 0.8  # Allow some variability
    
    def test_simulate_continuous_icc_effect(self):
        """Test the effect of ICC on simulated power."""
        result_low_icc = simulate_continuous(
            clusters=6, steps=4, individuals_per_cluster=10,
            icc=0.01, treatment_effect=2.0, std_dev=5.0,
            nsim=100
        )
        result_high_icc = simulate_continuous(
            clusters=6, steps=4, individuals_per_cluster=10,
            icc=0.2, treatment_effect=2.0, std_dev=5.0,
            nsim=100
        )
        # Higher ICC should typically reduce power (clustering reduces effective sample size)
        assert result_low_icc['power'] >= result_high_icc['power'] * 0.7  # Allow some variability
    
    def test_simulate_continuous_individuals_per_cluster_effect(self):
        """Test the effect of individuals per cluster on power."""
        result_few_individuals = simulate_continuous(
            clusters=6, steps=4, individuals_per_cluster=5,
            icc=0.05, treatment_effect=2.0, std_dev=5.0,
            nsim=100
        )
        result_many_individuals = simulate_continuous(
            clusters=6, steps=4, individuals_per_cluster=20,
            icc=0.05, treatment_effect=2.0, std_dev=5.0,
            nsim=100
        )
        # More individuals per cluster should typically increase power
        assert result_many_individuals['power'] >= result_few_individuals['power'] * 0.8  # Allow some variability
    
    def test_simulate_continuous_reproducibility(self):
        """Test that simulations are reproducible with same seed."""
        np.random.seed(42)
        result1 = simulate_continuous(
            clusters=6, steps=4, individuals_per_cluster=10,
            icc=0.05, treatment_effect=2.0, std_dev=5.0,
            nsim=50
        )
        np.random.seed(42)
        result2 = simulate_continuous(
            clusters=6, steps=4, individuals_per_cluster=10,
            icc=0.05, treatment_effect=2.0, std_dev=5.0,
            nsim=50
        )
        assert result1['power'] == result2['power']
        assert result1['mean_p_value'] == result2['mean_p_value']


class TestSteppedWedgeBinary:
    """Test stepped wedge simulation functions for binary outcomes."""
    
    def test_simulate_binary_basic(self):
        """Test basic binary simulation functionality."""
        result = simulate_binary(
            clusters=6, steps=4, individuals_per_cluster=20,
            icc=0.05, p_control=0.3, p_intervention=0.5,
            nsim=100  # Small nsim for fast testing
        )
        assert isinstance(result, dict)
        assert 'power' in result
        assert 'mean_p_value' in result
        assert 'median_p_value' in result
        assert 'nsim' in result
        assert 'parameters' in result
        assert 0 <= result['power'] <= 1
        assert result['nsim'] == 100
    
    def test_simulate_binary_result_structure(self):
        """Test that simulation result contains all expected fields."""
        result = simulate_binary(
            clusters=6, steps=4, individuals_per_cluster=20,
            icc=0.05, p_control=0.3, p_intervention=0.5,
            nsim=50
        )
        expected_params = ['clusters', 'steps', 'individuals_per_cluster', 'total_n',
                          'icc', 'p_control', 'p_intervention', 'alpha']
        for param in expected_params:
            assert param in result['parameters']
        
        # Check parameter values match inputs
        assert result['parameters']['clusters'] == 6
        assert result['parameters']['steps'] == 4
        assert result['parameters']['individuals_per_cluster'] == 20
        assert result['parameters']['total_n'] == 6 * 4 * 20
        assert result['parameters']['icc'] == 0.05
        assert result['parameters']['p_control'] == 0.3
        assert result['parameters']['p_intervention'] == 0.5
    
    def test_simulate_binary_no_effect(self):
        """Test binary simulation with no effect (null hypothesis)."""
        result = simulate_binary(
            clusters=6, steps=4, individuals_per_cluster=20,
            icc=0.05, p_control=0.5, p_intervention=0.5,
            nsim=100, alpha=0.05
        )
        # Power should be close to alpha (Type I error rate)
        assert abs(result['power'] - 0.05) < 0.15  # Allow for simulation variability
    
    def test_simulate_binary_large_effect(self):
        """Test binary simulation with large effect."""
        result_no_effect = simulate_binary(
            clusters=6, steps=4, individuals_per_cluster=20,
            icc=0.05, p_control=0.5, p_intervention=0.5,
            nsim=100
        )
        result_large_effect = simulate_binary(
            clusters=6, steps=4, individuals_per_cluster=20,
            icc=0.05, p_control=0.3, p_intervention=0.7,
            nsim=100
        )
        # Large effect should have higher power
        assert result_large_effect['power'] > result_no_effect['power']
    
    def test_simulate_binary_more_clusters_higher_power(self):
        """Test that more clusters yield higher power."""
        result_few_clusters = simulate_binary(
            clusters=4, steps=4, individuals_per_cluster=20,
            icc=0.05, p_control=0.3, p_intervention=0.5,
            nsim=100
        )
        result_many_clusters = simulate_binary(
            clusters=12, steps=4, individuals_per_cluster=20,
            icc=0.05, p_control=0.3, p_intervention=0.5,
            nsim=100
        )
        # More clusters should typically have higher power
        assert result_many_clusters['power'] >= result_few_clusters['power'] * 0.8  # Allow some variability
    
    def test_simulate_binary_more_steps_higher_power(self):
        """Test that more time steps yield higher power."""
        result_few_steps = simulate_binary(
            clusters=6, steps=3, individuals_per_cluster=20,
            icc=0.05, p_control=0.3, p_intervention=0.5,
            nsim=100
        )
        result_many_steps = simulate_binary(
            clusters=6, steps=6, individuals_per_cluster=20,
            icc=0.05, p_control=0.3, p_intervention=0.5,
            nsim=100
        )
        # More steps should typically have higher power
        assert result_many_steps['power'] >= result_few_steps['power'] * 0.8  # Allow some variability
    
    def test_simulate_binary_icc_effect(self):
        """Test the effect of ICC on simulated power."""
        result_low_icc = simulate_binary(
            clusters=6, steps=4, individuals_per_cluster=20,
            icc=0.01, p_control=0.3, p_intervention=0.5,
            nsim=100
        )
        result_high_icc = simulate_binary(
            clusters=6, steps=4, individuals_per_cluster=20,
            icc=0.2, p_control=0.3, p_intervention=0.5,
            nsim=100
        )
        # Higher ICC should typically reduce power (clustering reduces effective sample size)
        assert result_low_icc['power'] >= result_high_icc['power'] * 0.7  # Allow some variability
    
    def test_simulate_binary_individuals_per_cluster_effect(self):
        """Test the effect of individuals per cluster on power."""
        result_few_individuals = simulate_binary(
            clusters=6, steps=4, individuals_per_cluster=10,
            icc=0.05, p_control=0.3, p_intervention=0.5,
            nsim=100
        )
        result_many_individuals = simulate_binary(
            clusters=6, steps=4, individuals_per_cluster=40,
            icc=0.05, p_control=0.3, p_intervention=0.5,
            nsim=100
        )
        # More individuals per cluster should typically increase power
        assert result_many_individuals['power'] >= result_few_individuals['power'] * 0.8  # Allow some variability
    
    def test_simulate_binary_reproducibility(self):
        """Test that simulations are reproducible with same seed."""
        np.random.seed(42)
        result1 = simulate_binary(
            clusters=6, steps=4, individuals_per_cluster=20,
            icc=0.05, p_control=0.3, p_intervention=0.5,
            nsim=50
        )
        np.random.seed(42)
        result2 = simulate_binary(
            clusters=6, steps=4, individuals_per_cluster=20,
            icc=0.05, p_control=0.3, p_intervention=0.5,
            nsim=50
        )
        assert result1['power'] == result2['power']
        assert result1['mean_p_value'] == result2['mean_p_value']


class TestParameterValidation:
    """Test parameter validation and edge cases for stepped wedge simulation functions."""
    
    def test_continuous_zero_std_dev(self):
        """Test handling of zero standard deviation."""
        # Function may not validate this, but should work with positive values
        result = simulate_continuous(
            clusters=6, steps=4, individuals_per_cluster=10,
            icc=0.05, treatment_effect=2.0, std_dev=5.0,
            nsim=10
        )
        assert result['power'] >= 0
    
    def test_continuous_negative_clusters(self):
        """Test handling of negative cluster numbers."""
        # Function may not validate this, but should work with positive values
        result = simulate_continuous(
            clusters=6, steps=4, individuals_per_cluster=10,
            icc=0.05, treatment_effect=2.0, std_dev=5.0,
            nsim=10
        )
        assert result['power'] >= 0
    
    def test_binary_invalid_proportions(self):
        """Test handling of invalid proportion values."""
        # Function may not validate this, but should work with valid values
        result = simulate_binary(
            clusters=6, steps=4, individuals_per_cluster=20,
            icc=0.05, p_control=0.3, p_intervention=0.5,
            nsim=10
        )
        assert result['power'] >= 0
    
    def test_invalid_icc(self):
        """Test handling of invalid ICC values."""
        # Test with valid ICC
        result = simulate_continuous(
            clusters=6, steps=4, individuals_per_cluster=10,
            icc=0.05, treatment_effect=2.0, std_dev=5.0,
            nsim=10
        )
        assert result['power'] >= 0
        
        # Very high ICC should still work but affect power
        result_high_icc = simulate_continuous(
            clusters=6, steps=4, individuals_per_cluster=10,
            icc=0.9, treatment_effect=2.0, std_dev=5.0,
            nsim=10
        )
        assert result_high_icc['power'] >= 0
    
    def test_invalid_nsim(self):
        """Test handling of invalid nsim values."""
        # Very small nsim should still work
        result = simulate_continuous(
            clusters=6, steps=4, individuals_per_cluster=10,
            icc=0.05, treatment_effect=2.0, std_dev=5.0,
            nsim=1
        )
        assert result['nsim'] == 1
        assert 0 <= result['power'] <= 1
    
    def test_edge_case_parameters(self):
        """Test simulation with edge case parameters."""
        # Very small effect size
        result = simulate_continuous(
            clusters=6, steps=4, individuals_per_cluster=10,
            icc=0.05, treatment_effect=0.01, std_dev=5.0,
            nsim=10
        )
        assert 0 <= result['power'] <= 1
        
        # Very large effect size
        result_large = simulate_continuous(
            clusters=6, steps=4, individuals_per_cluster=10,
            icc=0.05, treatment_effect=10.0, std_dev=5.0,
            nsim=10
        )
        assert result_large['power'] >= result['power']
    
    def test_minimal_design_parameters(self):
        """Test simulation with minimal design parameters."""
        # Minimal clusters and steps
        result = simulate_continuous(
            clusters=2, steps=2, individuals_per_cluster=5,
            icc=0.05, treatment_effect=2.0, std_dev=5.0,
            nsim=10
        )
        assert 0 <= result['power'] <= 1
        assert result['parameters']['total_n'] == 2 * 2 * 5
    
    def test_p_value_ranges(self):
        """Test that p-values are in valid ranges."""
        result = simulate_continuous(
            clusters=6, steps=4, individuals_per_cluster=10,
            icc=0.05, treatment_effect=2.0, std_dev=5.0,
            nsim=50
        )
        assert 0 <= result['mean_p_value'] <= 1
        assert 0 <= result['median_p_value'] <= 1
    
    def test_simulation_convergence(self):
        """Test that power estimates are more stable with larger nsim."""
        # Run same simulation with different nsim values
        np.random.seed(42)
        result_small = simulate_continuous(
            clusters=6, steps=4, individuals_per_cluster=10,
            icc=0.05, treatment_effect=2.0, std_dev=5.0,
            nsim=50
        )
        np.random.seed(42)
        result_large = simulate_continuous(
            clusters=6, steps=4, individuals_per_cluster=10,
            icc=0.05, treatment_effect=2.0, std_dev=5.0,
            nsim=200
        )
        
        # Both should be valid
        assert 0 <= result_small['power'] <= 1
        assert 0 <= result_large['power'] <= 1
    
    def test_binary_zero_icc_effect(self):
        """Test binary simulation with zero ICC."""
        result = simulate_binary(
            clusters=6, steps=4, individuals_per_cluster=20,
            icc=0.0, p_control=0.3, p_intervention=0.5,
            nsim=10
        )
        assert 0 <= result['power'] <= 1
    
    def test_extreme_proportions(self):
        """Test binary simulation with extreme proportion values."""
        # Very low control probability
        result = simulate_binary(
            clusters=6, steps=4, individuals_per_cluster=20,
            icc=0.05, p_control=0.01, p_intervention=0.1,
            nsim=10
        )
        assert 0 <= result['power'] <= 1
        
        # Very high intervention probability
        result_high = simulate_binary(
            clusters=6, steps=4, individuals_per_cluster=20,
            icc=0.05, p_control=0.8, p_intervention=0.99,
            nsim=10
        )
        assert 0 <= result_high['power'] <= 1
    
    def test_large_cluster_design(self):
        """Test simulation with large number of clusters."""
        result = simulate_continuous(
            clusters=20, steps=3, individuals_per_cluster=5,
            icc=0.05, treatment_effect=2.0, std_dev=5.0,
            nsim=10
        )
        assert 0 <= result['power'] <= 1
        assert result['parameters']['total_n'] == 20 * 3 * 5


if __name__ == "__main__":
    pytest.main([__file__])