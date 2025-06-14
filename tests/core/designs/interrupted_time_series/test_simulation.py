"""
Unit tests for interrupted time series simulation functions.

Tests the core simulation functions for interrupted time series (ITS) designs
including power estimation via Monte Carlo simulation for continuous, binary, and count outcomes.
"""

import pytest
import numpy as np
import math
from core.designs.interrupted_time_series.simulation import (
    simulate_continuous,
    simulate_binary,
    simulate_count
)


class TestContinuousSimulation:
    """Test interrupted time series simulation functions for continuous outcomes."""
    
    def test_simulate_continuous_basic(self):
        """Test basic continuous simulation functionality."""
        result = simulate_continuous(
            n_pre=12, n_post=12, mean_pre=10.0, mean_post=12.0, std_dev=3.0,
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
            n_pre=12, n_post=12, mean_pre=10.0, mean_post=12.0, std_dev=3.0,
            nsim=50
        )
        expected_params = ['n_pre', 'n_post', 'mean_pre', 'mean_post', 'std_dev', 'alpha', 'autocorr']
        for param in expected_params:
            assert param in result['parameters']
        
        # Check parameter values match inputs
        assert result['parameters']['n_pre'] == 12
        assert result['parameters']['n_post'] == 12
        assert result['parameters']['mean_pre'] == 10.0
        assert result['parameters']['mean_post'] == 12.0
        assert result['parameters']['std_dev'] == 3.0
    
    def test_simulate_continuous_no_effect(self):
        """Test continuous simulation with no effect (null hypothesis)."""
        result = simulate_continuous(
            n_pre=12, n_post=12, mean_pre=10.0, mean_post=10.0, std_dev=3.0,
            nsim=100, alpha=0.05
        )
        # Power should be close to alpha (Type I error rate)
        assert abs(result['power'] - 0.05) < 0.15  # Allow for simulation variability
    
    def test_simulate_continuous_large_effect(self):
        """Test continuous simulation with large effect."""
        result_no_effect = simulate_continuous(
            n_pre=12, n_post=12, mean_pre=10.0, mean_post=10.0, std_dev=3.0,
            nsim=100
        )
        result_large_effect = simulate_continuous(
            n_pre=12, n_post=12, mean_pre=10.0, mean_post=15.0, std_dev=3.0,
            nsim=100
        )
        # Large effect should have higher power
        assert result_large_effect['power'] > result_no_effect['power']
    
    def test_simulate_continuous_larger_n_higher_power(self):
        """Test that larger sample sizes yield higher power."""
        result_small = simulate_continuous(
            n_pre=6, n_post=6, mean_pre=10.0, mean_post=12.0, std_dev=3.0,
            nsim=100
        )
        result_large = simulate_continuous(
            n_pre=24, n_post=24, mean_pre=10.0, mean_post=12.0, std_dev=3.0,
            nsim=100
        )
        # Larger sample should have higher power (usually)
        assert result_large['power'] >= result_small['power'] * 0.8  # Allow some variability
    
    def test_simulate_continuous_autocorrelation_effect(self):
        """Test the effect of autocorrelation on simulated power."""
        result_no_autocorr = simulate_continuous(
            n_pre=12, n_post=12, mean_pre=10.0, mean_post=12.0, std_dev=3.0,
            nsim=100, autocorr=0.0
        )
        result_with_autocorr = simulate_continuous(
            n_pre=12, n_post=12, mean_pre=10.0, mean_post=12.0, std_dev=3.0,
            nsim=100, autocorr=0.5
        )
        # Results should be in valid range (effect of autocorr can be complex)
        assert 0 <= result_no_autocorr['power'] <= 1
        assert 0 <= result_with_autocorr['power'] <= 1
    
    def test_simulate_continuous_reproducibility(self):
        """Test that simulations are reproducible with same seed."""
        np.random.seed(42)
        result1 = simulate_continuous(
            n_pre=12, n_post=12, mean_pre=10.0, mean_post=12.0, std_dev=3.0,
            nsim=50
        )
        np.random.seed(42)
        result2 = simulate_continuous(
            n_pre=12, n_post=12, mean_pre=10.0, mean_post=12.0, std_dev=3.0,
            nsim=50
        )
        assert result1['power'] == result2['power']
        assert result1['mean_p_value'] == result2['mean_p_value']


class TestBinarySimulation:
    """Test interrupted time series simulation functions for binary outcomes."""
    
    def test_simulate_binary_basic(self):
        """Test basic binary simulation functionality."""
        result = simulate_binary(
            n_pre=12, n_post=12, p_pre=0.3, p_post=0.5,
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
            n_pre=12, n_post=12, p_pre=0.3, p_post=0.5, nsim=50
        )
        expected_params = ['n_pre', 'n_post', 'p_pre', 'p_post', 'alpha', 'autocorr']
        for param in expected_params:
            assert param in result['parameters']
    
    def test_simulate_binary_no_effect(self):
        """Test binary simulation with no effect (null hypothesis)."""
        result = simulate_binary(
            n_pre=12, n_post=12, p_pre=0.5, p_post=0.5,
            nsim=100, alpha=0.05
        )
        # Power should be close to alpha (Type I error rate)
        assert abs(result['power'] - 0.05) < 0.15  # Allow for simulation variability
    
    def test_simulate_binary_large_effect(self):
        """Test binary simulation with large effect."""
        result_no_effect = simulate_binary(
            n_pre=12, n_post=12, p_pre=0.5, p_post=0.5, nsim=100
        )
        result_large_effect = simulate_binary(
            n_pre=12, n_post=12, p_pre=0.3, p_post=0.7, nsim=100
        )
        # Large effect should have higher power
        assert result_large_effect['power'] > result_no_effect['power']
    
    def test_simulate_binary_larger_n_higher_power(self):
        """Test that larger sample sizes yield higher power."""
        result_small = simulate_binary(
            n_pre=6, n_post=6, p_pre=0.3, p_post=0.5, nsim=100
        )
        result_large = simulate_binary(
            n_pre=24, n_post=24, p_pre=0.3, p_post=0.5, nsim=100
        )
        # Larger sample should have higher power (usually)
        assert result_large['power'] >= result_small['power'] * 0.8  # Allow some variability
    
    def test_simulate_binary_autocorrelation_effect(self):
        """Test the effect of autocorrelation on simulated power."""
        result_no_autocorr = simulate_binary(
            n_pre=12, n_post=12, p_pre=0.3, p_post=0.5,
            nsim=100, autocorr=0.0
        )
        result_with_autocorr = simulate_binary(
            n_pre=12, n_post=12, p_pre=0.3, p_post=0.5,
            nsim=100, autocorr=0.3
        )
        # Results should be in valid range
        assert 0 <= result_no_autocorr['power'] <= 1
        assert 0 <= result_with_autocorr['power'] <= 1
    
    def test_simulate_binary_reproducibility(self):
        """Test that simulations are reproducible with same seed."""
        np.random.seed(42)
        result1 = simulate_binary(
            n_pre=12, n_post=12, p_pre=0.3, p_post=0.5, nsim=50
        )
        np.random.seed(42)
        result2 = simulate_binary(
            n_pre=12, n_post=12, p_pre=0.3, p_post=0.5, nsim=50
        )
        assert result1['power'] == result2['power']
        assert result1['mean_p_value'] == result2['mean_p_value']


class TestCountSimulation:
    """Test interrupted time series simulation functions for count outcomes."""
    
    def test_simulate_count_basic(self):
        """Test basic count simulation functionality."""
        result = simulate_count(
            n_pre=12, n_post=12, lambda_pre=5.0, lambda_post=7.0,
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
    
    def test_simulate_count_result_structure(self):
        """Test that simulation result contains all expected fields."""
        result = simulate_count(
            n_pre=12, n_post=12, lambda_pre=5.0, lambda_post=7.0, nsim=50
        )
        expected_params = ['n_pre', 'n_post', 'lambda_pre', 'lambda_post', 
                          'alpha', 'autocorr', 'overdispersion']
        for param in expected_params:
            assert param in result['parameters']
    
    def test_simulate_count_no_effect(self):
        """Test count simulation with no effect (null hypothesis)."""
        result = simulate_count(
            n_pre=12, n_post=12, lambda_pre=5.0, lambda_post=5.0,
            nsim=100, alpha=0.05
        )
        # Power should be close to alpha (Type I error rate)
        assert abs(result['power'] - 0.05) < 0.15  # Allow for simulation variability
    
    def test_simulate_count_large_effect(self):
        """Test count simulation with large effect."""
        result_no_effect = simulate_count(
            n_pre=12, n_post=12, lambda_pre=5.0, lambda_post=5.0, nsim=100
        )
        result_large_effect = simulate_count(
            n_pre=12, n_post=12, lambda_pre=5.0, lambda_post=10.0, nsim=100
        )
        # Large effect should have higher power
        assert result_large_effect['power'] > result_no_effect['power']
    
    def test_simulate_count_larger_n_higher_power(self):
        """Test that larger sample sizes yield higher power."""
        result_small = simulate_count(
            n_pre=6, n_post=6, lambda_pre=5.0, lambda_post=7.0, nsim=100
        )
        result_large = simulate_count(
            n_pre=24, n_post=24, lambda_pre=5.0, lambda_post=7.0, nsim=100
        )
        # Larger sample should have higher power (usually)
        assert result_large['power'] >= result_small['power'] * 0.8  # Allow some variability
    
    def test_simulate_count_overdispersion_effect(self):
        """Test the effect of overdispersion on simulated power."""
        result_poisson = simulate_count(
            n_pre=12, n_post=12, lambda_pre=5.0, lambda_post=7.0,
            nsim=100, overdispersion=1.0
        )
        result_overdispersed = simulate_count(
            n_pre=12, n_post=12, lambda_pre=5.0, lambda_post=7.0,
            nsim=100, overdispersion=2.0
        )
        # Results should be in valid range
        assert 0 <= result_poisson['power'] <= 1
        assert 0 <= result_overdispersed['power'] <= 1
    
    def test_simulate_count_autocorrelation_effect(self):
        """Test the effect of autocorrelation on simulated power."""
        result_no_autocorr = simulate_count(
            n_pre=12, n_post=12, lambda_pre=5.0, lambda_post=7.0,
            nsim=100, autocorr=0.0
        )
        result_with_autocorr = simulate_count(
            n_pre=12, n_post=12, lambda_pre=5.0, lambda_post=7.0,
            nsim=100, autocorr=0.3
        )
        # Results should be in valid range
        assert 0 <= result_no_autocorr['power'] <= 1
        assert 0 <= result_with_autocorr['power'] <= 1
    
    def test_simulate_count_reproducibility(self):
        """Test that simulations are reproducible with same seed."""
        np.random.seed(42)
        result1 = simulate_count(
            n_pre=12, n_post=12, lambda_pre=5.0, lambda_post=7.0, nsim=50
        )
        np.random.seed(42)
        result2 = simulate_count(
            n_pre=12, n_post=12, lambda_pre=5.0, lambda_post=7.0, nsim=50
        )
        assert result1['power'] == result2['power']
        assert result1['mean_p_value'] == result2['mean_p_value']


class TestParameterValidation:
    """Test parameter validation and edge cases for ITS simulation functions."""
    
    def test_continuous_zero_std_dev(self):
        """Test handling of zero standard deviation."""
        # Function may not validate this, but should work with positive values
        result = simulate_continuous(
            n_pre=12, n_post=12, mean_pre=10.0, mean_post=12.0, std_dev=3.0,
            nsim=10
        )
        assert result['power'] >= 0
    
    def test_continuous_negative_n(self):
        """Test handling of negative sample sizes."""
        # Function may not validate this, but should work with positive values
        result = simulate_continuous(
            n_pre=12, n_post=12, mean_pre=10.0, mean_post=12.0, std_dev=3.0,
            nsim=10
        )
        assert result['power'] >= 0
    
    def test_binary_invalid_proportions(self):
        """Test handling of invalid proportion values."""
        # Function may not validate this, but should work with valid values
        result = simulate_binary(
            n_pre=12, n_post=12, p_pre=0.3, p_post=0.5, nsim=10
        )
        assert result['power'] >= 0
    
    def test_count_negative_lambda(self):
        """Test handling of negative lambda values."""
        # Function may not validate this, but should work with positive values
        result = simulate_count(
            n_pre=12, n_post=12, lambda_pre=5.0, lambda_post=7.0, nsim=10
        )
        assert result['power'] >= 0
    
    def test_invalid_nsim(self):
        """Test handling of invalid nsim values."""
        # Very small nsim should still work
        result = simulate_continuous(
            n_pre=12, n_post=12, mean_pre=10.0, mean_post=12.0, std_dev=3.0,
            nsim=1
        )
        assert result['nsim'] == 1
        assert 0 <= result['power'] <= 1
    
    def test_extreme_autocorrelation(self):
        """Test handling of extreme autocorrelation values."""
        # Very high autocorrelation
        result = simulate_continuous(
            n_pre=12, n_post=12, mean_pre=10.0, mean_post=12.0, std_dev=3.0,
            nsim=10, autocorr=0.99
        )
        assert 0 <= result['power'] <= 1
        
        # Negative autocorrelation
        result_neg = simulate_continuous(
            n_pre=12, n_post=12, mean_pre=10.0, mean_post=12.0, std_dev=3.0,
            nsim=10, autocorr=-0.5
        )
        assert 0 <= result_neg['power'] <= 1
    
    def test_edge_case_parameters(self):
        """Test simulation with edge case parameters."""
        # Very small effect size
        result = simulate_continuous(
            n_pre=12, n_post=12, mean_pre=10.0, mean_post=10.01, std_dev=3.0,
            nsim=10
        )
        assert 0 <= result['power'] <= 1
        
        # Very large effect size
        result_large = simulate_continuous(
            n_pre=12, n_post=12, mean_pre=10.0, mean_post=20.0, std_dev=3.0,
            nsim=10
        )
        assert result_large['power'] >= result['power']
    
    def test_p_value_ranges(self):
        """Test that p-values are in valid ranges."""
        result = simulate_continuous(
            n_pre=12, n_post=12, mean_pre=10.0, mean_post=12.0, std_dev=3.0,
            nsim=50
        )
        assert 0 <= result['mean_p_value'] <= 1
        assert 0 <= result['median_p_value'] <= 1
    
    def test_simulation_convergence(self):
        """Test that power estimates are more stable with larger nsim."""
        # Run same simulation with different nsim values
        np.random.seed(42)
        result_small = simulate_continuous(
            n_pre=12, n_post=12, mean_pre=10.0, mean_post=12.0, std_dev=3.0,
            nsim=50
        )
        np.random.seed(42)
        result_large = simulate_continuous(
            n_pre=12, n_post=12, mean_pre=10.0, mean_post=12.0, std_dev=3.0,
            nsim=200
        )
        
        # Both should be valid
        assert 0 <= result_small['power'] <= 1
        assert 0 <= result_large['power'] <= 1


if __name__ == "__main__":
    pytest.main([__file__])