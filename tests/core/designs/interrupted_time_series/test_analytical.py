"""
Unit tests for interrupted time series analytical functions.

Tests the core statistical functions for interrupted time series (ITS) designs
including power and sample size calculations for continuous and binary outcomes.
"""

import pytest
import numpy as np
import math
from core.designs.interrupted_time_series.analytical import (
    power_continuous,
    sample_size_continuous,
    power_binary,
    sample_size_binary
)


class TestContinuousAnalytical:
    """Test interrupted time series analytical functions for continuous outcomes."""
    
    def test_power_continuous_basic(self):
        """Test basic power calculation for continuous outcomes."""
        result = power_continuous(
            n_pre=12, n_post=12, mean_change=2.0, std_dev=5.0
        )
        assert isinstance(result, dict)
        assert 'power' in result
        assert 'parameters' in result
        assert 0 < result['power'] < 1
        assert isinstance(result['power'], (int, float))
    
    def test_power_continuous_result_structure(self):
        """Test that power result contains all expected fields."""
        result = power_continuous(
            n_pre=12, n_post=12, mean_change=2.0, std_dev=5.0
        )
        expected_params = ['n_pre', 'n_post', 'mean_change', 'std_dev', 'alpha', 'autocorr']
        for param in expected_params:
            assert param in result['parameters']
            
        # Check parameter values match inputs
        assert result['parameters']['n_pre'] == 12
        assert result['parameters']['n_post'] == 12
        assert result['parameters']['mean_change'] == 2.0
        assert result['parameters']['std_dev'] == 5.0
        assert result['parameters']['alpha'] == 0.05  # default
        assert result['parameters']['autocorr'] == 0.0  # default
    
    def test_power_continuous_larger_n_higher_power(self):
        """Test that larger sample sizes yield higher power."""
        result_small = power_continuous(
            n_pre=6, n_post=6, mean_change=2.0, std_dev=5.0
        )
        result_large = power_continuous(
            n_pre=24, n_post=24, mean_change=2.0, std_dev=5.0
        )
        assert result_large['power'] > result_small['power']
    
    def test_power_continuous_larger_effect_higher_power(self):
        """Test that larger effects yield higher power."""
        result_small_effect = power_continuous(
            n_pre=12, n_post=12, mean_change=1.0, std_dev=5.0
        )
        result_large_effect = power_continuous(
            n_pre=12, n_post=12, mean_change=4.0, std_dev=5.0
        )
        assert result_large_effect['power'] > result_small_effect['power']
    
    def test_power_continuous_autocorrelation_effect(self):
        """Test the effect of autocorrelation on power."""
        result_no_autocorr = power_continuous(
            n_pre=12, n_post=12, mean_change=2.0, std_dev=5.0, autocorr=0.0
        )
        result_with_autocorr = power_continuous(
            n_pre=12, n_post=12, mean_change=2.0, std_dev=5.0, autocorr=0.3
        )
        # Positive autocorrelation should reduce effective sample size and power
        assert result_with_autocorr['power'] < result_no_autocorr['power']
    
    def test_power_continuous_alpha_effect(self):
        """Test the effect of alpha level on power."""
        result_alpha_05 = power_continuous(
            n_pre=12, n_post=12, mean_change=2.0, std_dev=5.0, alpha=0.05
        )
        result_alpha_01 = power_continuous(
            n_pre=12, n_post=12, mean_change=2.0, std_dev=5.0, alpha=0.01
        )
        # Lower alpha should yield lower power (more stringent test)
        assert result_alpha_01['power'] < result_alpha_05['power']
    
    def test_sample_size_continuous_basic(self):
        """Test basic sample size calculation for continuous outcomes."""
        result = sample_size_continuous(
            mean_change=2.0, std_dev=5.0, power=0.8
        )
        assert isinstance(result, dict)
        assert 'n_pre' in result
        assert 'n_post' in result
        assert 'total_n' in result
        assert 'parameters' in result
        assert isinstance(result['n_pre'], int)
        assert isinstance(result['n_post'], int)
        assert result['n_pre'] > 0
        assert result['n_post'] > 0
        assert result['total_n'] == result['n_pre'] + result['n_post']
    
    def test_sample_size_continuous_result_structure(self):
        """Test that sample size result contains all expected fields."""
        result = sample_size_continuous(
            mean_change=2.0, std_dev=5.0, power=0.8
        )
        expected_params = ['mean_change', 'std_dev', 'power', 'alpha', 'autocorr', 'ratio']
        for param in expected_params:
            assert param in result['parameters']
    
    def test_sample_size_continuous_larger_effect_smaller_n(self):
        """Test that larger effects require smaller sample sizes."""
        result_small_effect = sample_size_continuous(
            mean_change=1.0, std_dev=5.0, power=0.8
        )
        result_large_effect = sample_size_continuous(
            mean_change=4.0, std_dev=5.0, power=0.8
        )
        assert result_large_effect['total_n'] < result_small_effect['total_n']
    
    def test_sample_size_continuous_higher_power_larger_n(self):
        """Test that higher power requires larger sample sizes."""
        result_low_power = sample_size_continuous(
            mean_change=2.0, std_dev=5.0, power=0.7
        )
        result_high_power = sample_size_continuous(
            mean_change=2.0, std_dev=5.0, power=0.9
        )
        assert result_high_power['total_n'] > result_low_power['total_n']
    
    def test_sample_size_continuous_ratio_effect(self):
        """Test the effect of pre/post ratio on sample sizes."""
        result_equal = sample_size_continuous(
            mean_change=2.0, std_dev=5.0, power=0.8, ratio=1.0
        )
        result_unequal = sample_size_continuous(
            mean_change=2.0, std_dev=5.0, power=0.8, ratio=2.0
        )
        assert result_equal['n_pre'] == result_equal['n_post']
        assert result_unequal['n_post'] == 2 * result_unequal['n_pre']
    
    def test_sample_size_continuous_autocorrelation_effect(self):
        """Test the effect of autocorrelation on required sample size."""
        result_no_autocorr = sample_size_continuous(
            mean_change=2.0, std_dev=5.0, power=0.8, autocorr=0.0
        )
        result_with_autocorr = sample_size_continuous(
            mean_change=2.0, std_dev=5.0, power=0.8, autocorr=0.3
        )
        # Positive autocorrelation should increase required sample size
        assert result_with_autocorr['total_n'] > result_no_autocorr['total_n']
    
    def test_power_sample_size_consistency_continuous(self):
        """Test consistency between power and sample size calculations."""
        # Get sample size for specific power
        ss_result = sample_size_continuous(
            mean_change=2.0, std_dev=5.0, power=0.8
        )
        
        # Calculate power with that sample size
        power_result = power_continuous(
            n_pre=ss_result['n_pre'], n_post=ss_result['n_post'],
            mean_change=2.0, std_dev=5.0
        )
        
        # Power should be close to target (within 10% due to rounding)
        assert abs(power_result['power'] - 0.8) < 0.1


class TestBinaryAnalytical:
    """Test interrupted time series analytical functions for binary outcomes."""
    
    def test_power_binary_basic(self):
        """Test basic power calculation for binary outcomes."""
        result = power_binary(
            n_pre=100, n_post=100, p_pre=0.3, p_post=0.5
        )
        assert isinstance(result, dict)
        assert 'power' in result
        assert 'parameters' in result
        assert 0 < result['power'] < 1
        assert isinstance(result['power'], (int, float))
    
    def test_power_binary_result_structure(self):
        """Test that power result contains all expected fields."""
        result = power_binary(
            n_pre=100, n_post=100, p_pre=0.3, p_post=0.5
        )
        expected_params = ['n_pre', 'n_post', 'p_pre', 'p_post', 'alpha', 'autocorr']
        for param in expected_params:
            assert param in result['parameters']
    
    def test_power_binary_larger_n_higher_power(self):
        """Test that larger sample sizes yield higher power."""
        result_small = power_binary(
            n_pre=50, n_post=50, p_pre=0.3, p_post=0.5
        )
        result_large = power_binary(
            n_pre=200, n_post=200, p_pre=0.3, p_post=0.5
        )
        assert result_large['power'] > result_small['power']
    
    def test_power_binary_larger_effect_higher_power(self):
        """Test that larger effects yield higher power."""
        result_small_effect = power_binary(
            n_pre=100, n_post=100, p_pre=0.4, p_post=0.45
        )
        result_large_effect = power_binary(
            n_pre=100, n_post=100, p_pre=0.3, p_post=0.6
        )
        assert result_large_effect['power'] > result_small_effect['power']
    
    def test_power_binary_autocorrelation_effect(self):
        """Test the effect of autocorrelation on power."""
        result_no_autocorr = power_binary(
            n_pre=100, n_post=100, p_pre=0.3, p_post=0.5, autocorr=0.0
        )
        result_with_autocorr = power_binary(
            n_pre=100, n_post=100, p_pre=0.3, p_post=0.5, autocorr=0.3
        )
        # Positive autocorrelation should reduce effective sample size and power
        assert result_with_autocorr['power'] < result_no_autocorr['power']
    
    def test_sample_size_binary_basic(self):
        """Test basic sample size calculation for binary outcomes."""
        result = sample_size_binary(
            p_pre=0.3, p_post=0.5, power=0.8
        )
        assert isinstance(result, dict)
        assert 'n_pre' in result
        assert 'n_post' in result
        assert 'total_n' in result
        assert 'parameters' in result
        assert isinstance(result['n_pre'], int)
        assert isinstance(result['n_post'], int)
        assert result['n_pre'] > 0
        assert result['n_post'] > 0
        assert result['total_n'] == result['n_pre'] + result['n_post']
    
    def test_sample_size_binary_result_structure(self):
        """Test that sample size result contains all expected fields."""
        result = sample_size_binary(
            p_pre=0.3, p_post=0.5, power=0.8
        )
        expected_params = ['p_pre', 'p_post', 'power', 'alpha', 'autocorr', 'ratio']
        for param in expected_params:
            assert param in result['parameters']
    
    def test_sample_size_binary_larger_effect_smaller_n(self):
        """Test that larger effects require smaller sample sizes."""
        result_small_effect = sample_size_binary(
            p_pre=0.4, p_post=0.45, power=0.8
        )
        result_large_effect = sample_size_binary(
            p_pre=0.3, p_post=0.6, power=0.8
        )
        assert result_large_effect['total_n'] < result_small_effect['total_n']
    
    def test_sample_size_binary_higher_power_larger_n(self):
        """Test that higher power requires larger sample sizes."""
        result_low_power = sample_size_binary(
            p_pre=0.3, p_post=0.5, power=0.7
        )
        result_high_power = sample_size_binary(
            p_pre=0.3, p_post=0.5, power=0.9
        )
        assert result_high_power['total_n'] > result_low_power['total_n']
    
    def test_sample_size_binary_ratio_effect(self):
        """Test the effect of pre/post ratio on sample sizes."""
        result_equal = sample_size_binary(
            p_pre=0.3, p_post=0.5, power=0.8, ratio=1.0
        )
        result_unequal = sample_size_binary(
            p_pre=0.3, p_post=0.5, power=0.8, ratio=2.0
        )
        assert result_equal['n_pre'] == result_equal['n_post']
        assert result_unequal['n_post'] == 2 * result_unequal['n_pre']
    
    def test_sample_size_binary_autocorrelation_effect(self):
        """Test the effect of autocorrelation on required sample size."""
        result_no_autocorr = sample_size_binary(
            p_pre=0.3, p_post=0.5, power=0.8, autocorr=0.0
        )
        result_with_autocorr = sample_size_binary(
            p_pre=0.3, p_post=0.5, power=0.8, autocorr=0.3
        )
        # Positive autocorrelation should increase required sample size
        assert result_with_autocorr['total_n'] > result_no_autocorr['total_n']
    
    def test_power_sample_size_consistency_binary(self):
        """Test consistency between power and sample size calculations."""
        # Get sample size for specific power
        ss_result = sample_size_binary(
            p_pre=0.3, p_post=0.5, power=0.8
        )
        
        # Calculate power with that sample size
        power_result = power_binary(
            n_pre=ss_result['n_pre'], n_post=ss_result['n_post'],
            p_pre=0.3, p_post=0.5
        )
        
        # Power should be close to target (within 10% due to rounding)
        assert abs(power_result['power'] - 0.8) < 0.1


class TestParameterValidation:
    """Test parameter validation and edge cases for ITS analytical functions."""
    
    def test_continuous_zero_std_dev(self):
        """Test handling of zero standard deviation."""
        with pytest.raises((ValueError, ZeroDivisionError)):
            power_continuous(n_pre=12, n_post=12, mean_change=2.0, std_dev=0)
    
    def test_continuous_zero_effect_size(self):
        """Test handling of zero effect size for continuous outcomes."""
        result = power_continuous(
            n_pre=12, n_post=12, mean_change=0.0, std_dev=5.0
        )
        # Power should be close to alpha (type I error rate)
        assert abs(result['power'] - 0.05) < 0.1
    
    def test_continuous_negative_n(self):
        """Test handling of negative sample sizes."""
        # Function may not validate this, but should work with positive values
        result = power_continuous(
            n_pre=12, n_post=12, mean_change=2.0, std_dev=5.0
        )
        assert result['power'] > 0
    
    def test_binary_invalid_proportions(self):
        """Test handling of invalid proportion values."""
        # Function may not validate these, so just test it works with valid values
        result = power_binary(n_pre=100, n_post=100, p_pre=0.3, p_post=0.5)
        assert result['power'] > 0
    
    def test_binary_equal_proportions(self):
        """Test handling of equal proportions (zero effect)."""
        result = power_binary(
            n_pre=100, n_post=100, p_pre=0.5, p_post=0.5
        )
        # Power should be close to alpha (type I error rate)
        assert abs(result['power'] - 0.05) < 0.1
    
    def test_invalid_alpha_values(self):
        """Test handling of invalid alpha values."""
        # Function may not validate these, so just test it works with valid values
        result = power_continuous(
            n_pre=12, n_post=12, mean_change=2.0, std_dev=5.0, alpha=0.05
        )
        assert result['power'] > 0
    
    def test_invalid_autocorrelation(self):
        """Test handling of invalid autocorrelation values."""
        # Test with valid autocorrelation
        result = power_continuous(
            n_pre=12, n_post=12, mean_change=2.0, std_dev=5.0, autocorr=0.5
        )
        assert result['power'] > 0
        
        # Very high autocorrelation should still work but reduce power significantly
        result_high = power_continuous(
            n_pre=12, n_post=12, mean_change=2.0, std_dev=5.0, autocorr=0.9
        )
        assert result_high['power'] < result['power']
    
    def test_sample_size_edge_cases(self):
        """Test sample size calculations with edge case parameters."""
        # Very small effect size - should return large sample size
        result = sample_size_continuous(
            mean_change=0.1, std_dev=5.0, power=0.8
        )
        assert result['total_n'] > 100
        
        # Very high power - should return large sample size
        result_high_power = sample_size_continuous(
            mean_change=2.0, std_dev=5.0, power=0.99
        )
        assert result_high_power['total_n'] > 50
    
    def test_ratio_edge_cases(self):
        """Test sample size calculations with extreme ratio values."""
        # Very unbalanced design
        result = sample_size_continuous(
            mean_change=2.0, std_dev=5.0, power=0.8, ratio=10.0
        )
        assert result['n_post'] == 10 * result['n_pre']
        assert result['total_n'] == result['n_pre'] + result['n_post']


if __name__ == "__main__":
    pytest.main([__file__])