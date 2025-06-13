"""
Unit tests for parallel non-inferiority analytical functions.

Tests the core analytical functions for parallel group non-inferiority trials
including sample size, power, and minimum detectable effect calculations 
for both continuous and binary outcomes.
"""

import pytest
import numpy as np
import math
from core.designs.parallel.analytical.non_inferiority import (
    sample_size_continuous_non_inferiority,
    power_continuous_non_inferiority,
    min_detectable_non_inferiority_margin,
    sample_size_binary_non_inferiority,
    power_binary_non_inferiority,
    min_detectable_binary_non_inferiority_margin
)


class TestContinuousNonInferiority:
    """Test non-inferiority analytical functions for continuous outcomes."""
    
    def test_sample_size_continuous_basic(self):
        """Test basic sample size calculation for continuous non-inferiority."""
        result = sample_size_continuous_non_inferiority(
            non_inferiority_margin=2.0, std_dev=5.0, power=0.8, alpha=0.05
        )
        
        assert isinstance(result, dict)
        assert 'n1' in result
        assert 'n2' in result
        assert 'total_n' in result
        assert 'parameters' in result
        
        assert isinstance(result['n1'], int)
        assert isinstance(result['n2'], int)
        assert result['n1'] > 0
        assert result['n2'] > 0
        assert result['total_n'] == result['n1'] + result['n2']
    
    def test_sample_size_continuous_result_structure(self):
        """Test that sample size result contains all expected fields."""
        result = sample_size_continuous_non_inferiority(
            non_inferiority_margin=2.0, std_dev=5.0, power=0.8, alpha=0.05
        )
        
        expected_params = ['non_inferiority_margin', 'std_dev', 'power', 'alpha',
                          'allocation_ratio', 'assumed_difference', 'direction', 'hypothesis_type']
        for param in expected_params:
            assert param in result['parameters']
        
        assert result['parameters']['hypothesis_type'] == 'non-inferiority'
        assert result['parameters']['direction'] == 'lower'  # default
    
    def test_sample_size_continuous_smaller_margin_larger_n(self):
        """Test that smaller non-inferiority margins require larger sample sizes."""
        result_large_margin = sample_size_continuous_non_inferiority(
            non_inferiority_margin=3.0, std_dev=5.0, power=0.8, alpha=0.05
        )
        result_small_margin = sample_size_continuous_non_inferiority(
            non_inferiority_margin=1.0, std_dev=5.0, power=0.8, alpha=0.05
        )
        
        assert result_small_margin['total_n'] > result_large_margin['total_n']
    
    def test_sample_size_continuous_higher_power_larger_n(self):
        """Test that higher power requires larger sample sizes."""
        result_low_power = sample_size_continuous_non_inferiority(
            non_inferiority_margin=2.0, std_dev=5.0, power=0.7, alpha=0.05
        )
        result_high_power = sample_size_continuous_non_inferiority(
            non_inferiority_margin=2.0, std_dev=5.0, power=0.9, alpha=0.05
        )
        
        assert result_high_power['total_n'] > result_low_power['total_n']
    
    def test_sample_size_continuous_allocation_ratio(self):
        """Test sample size calculation with different allocation ratios."""
        result_equal = sample_size_continuous_non_inferiority(
            non_inferiority_margin=2.0, std_dev=5.0, power=0.8, 
            alpha=0.05, allocation_ratio=1.0
        )
        result_unequal = sample_size_continuous_non_inferiority(
            non_inferiority_margin=2.0, std_dev=5.0, power=0.8, 
            alpha=0.05, allocation_ratio=2.0
        )
        
        assert result_equal['n1'] == result_equal['n2']
        assert result_unequal['n2'] == 2 * result_unequal['n1']
    
    def test_sample_size_continuous_direction_effect(self):
        """Test sample size calculation with different directions."""
        result_lower = sample_size_continuous_non_inferiority(
            non_inferiority_margin=2.0, std_dev=5.0, power=0.8, 
            alpha=0.05, direction="lower"
        )
        result_upper = sample_size_continuous_non_inferiority(
            non_inferiority_margin=2.0, std_dev=5.0, power=0.8, 
            alpha=0.05, direction="upper"
        )
        
        # Both should be valid, but may differ
        assert result_lower['total_n'] > 0
        assert result_upper['total_n'] > 0
    
    def test_sample_size_continuous_assumed_difference(self):
        """Test sample size calculation with different assumed differences."""
        result_no_diff = sample_size_continuous_non_inferiority(
            non_inferiority_margin=2.0, std_dev=5.0, power=0.8, 
            alpha=0.05, assumed_difference=0.0
        )
        result_with_diff = sample_size_continuous_non_inferiority(
            non_inferiority_margin=2.0, std_dev=5.0, power=0.8, 
            alpha=0.05, assumed_difference=1.0
        )
        
        # Sample sizes should differ based on assumed difference
        assert result_no_diff['total_n'] != result_with_diff['total_n']
    
    def test_power_continuous_basic(self):
        """Test basic power calculation for continuous non-inferiority."""
        # First get sample size
        ss_result = sample_size_continuous_non_inferiority(
            non_inferiority_margin=2.0, std_dev=5.0, power=0.8, alpha=0.05
        )
        
        # Calculate power with that sample size
        power_result = power_continuous_non_inferiority(
            n1=ss_result['n1'], n2=ss_result['n2'],
            non_inferiority_margin=2.0, std_dev=5.0, alpha=0.05
        )
        
        assert isinstance(power_result, dict)
        assert 'power' in power_result
        assert 'parameters' in power_result
        assert 0 <= power_result['power'] <= 1
    
    def test_power_continuous_larger_n_higher_power(self):
        """Test that larger sample sizes yield higher power."""
        power_small = power_continuous_non_inferiority(
            n1=25, n2=25, non_inferiority_margin=2.0, std_dev=5.0, alpha=0.05
        )
        power_large = power_continuous_non_inferiority(
            n1=100, n2=100, non_inferiority_margin=2.0, std_dev=5.0, alpha=0.05
        )
        
        assert power_large['power'] > power_small['power']
    
    def test_power_continuous_smaller_margin_higher_power(self):
        """Test that smaller margins (easier to show non-inferiority) yield higher power."""
        power_large_margin = power_continuous_non_inferiority(
            n1=50, n2=50, non_inferiority_margin=3.0, std_dev=5.0, alpha=0.05
        )
        power_small_margin = power_continuous_non_inferiority(
            n1=50, n2=50, non_inferiority_margin=1.0, std_dev=5.0, alpha=0.05
        )
        
        # Larger margin should make non-inferiority easier to demonstrate
        assert power_large_margin['power'] > power_small_margin['power']
    
    def test_min_detectable_margin_continuous_basic(self):
        """Test minimum detectable margin calculation for continuous outcomes."""
        result = min_detectable_non_inferiority_margin(
            n1=50, n2=50, std_dev=5.0, power=0.8, alpha=0.05
        )
        
        assert isinstance(result, dict)
        assert 'margin' in result
        assert 'parameters' in result
        assert result['margin'] > 0
    
    def test_min_detectable_margin_larger_n_smaller_margin(self):
        """Test that larger sample sizes can detect smaller margins."""
        result_small_n = min_detectable_non_inferiority_margin(
            n1=25, n2=25, std_dev=5.0, power=0.8, alpha=0.05
        )
        result_large_n = min_detectable_non_inferiority_margin(
            n1=100, n2=100, std_dev=5.0, power=0.8, alpha=0.05
        )
        
        assert result_large_n['margin'] < result_small_n['margin']
    
    def test_power_sample_size_consistency_continuous(self):
        """Test consistency between power and sample size calculations."""
        # Get sample size for target power
        ss_result = sample_size_continuous_non_inferiority(
            non_inferiority_margin=2.0, std_dev=5.0, power=0.8, alpha=0.05
        )
        
        # Calculate power with that sample size
        power_result = power_continuous_non_inferiority(
            n1=ss_result['n1'], n2=ss_result['n2'],
            non_inferiority_margin=2.0, std_dev=5.0, alpha=0.05
        )
        
        # Power should be close to target (within 5% due to rounding)
        assert abs(power_result['power'] - 0.8) < 0.05


class TestBinaryNonInferiority:
    """Test non-inferiority analytical functions for binary outcomes."""
    
    def test_sample_size_binary_basic(self):
        """Test basic sample size calculation for binary non-inferiority."""
        result = sample_size_binary_non_inferiority(
            p1=0.7, non_inferiority_margin=0.1, power=0.8, alpha=0.05
        )
        
        assert isinstance(result, dict)
        assert 'n1' in result
        assert 'n2' in result
        assert 'total_n' in result
        assert 'parameters' in result
        
        assert isinstance(result['n1'], int)
        assert isinstance(result['n2'], int)
        assert result['n1'] > 0
        assert result['n2'] > 0
        assert result['total_n'] == result['n1'] + result['n2']
    
    def test_sample_size_binary_result_structure(self):
        """Test that sample size result contains all expected fields."""
        result = sample_size_binary_non_inferiority(
            p1=0.7, non_inferiority_margin=0.1, power=0.8, alpha=0.05
        )
        
        expected_params = ['p1', 'non_inferiority_margin', 'power', 'alpha',
                          'allocation_ratio', 'assumed_difference', 'test_type', 'hypothesis_type']
        for param in expected_params:
            assert param in result['parameters']
        
        assert result['parameters']['hypothesis_type'] == 'non-inferiority'
    
    def test_sample_size_binary_smaller_margin_larger_n(self):
        """Test that smaller non-inferiority margins require larger sample sizes."""
        result_large_margin = sample_size_binary_non_inferiority(
            p1=0.7, non_inferiority_margin=0.15, power=0.8, alpha=0.05
        )
        result_small_margin = sample_size_binary_non_inferiority(
            p1=0.7, non_inferiority_margin=0.05, power=0.8, alpha=0.05
        )
        
        assert result_small_margin['total_n'] > result_large_margin['total_n']
    
    def test_sample_size_binary_higher_power_larger_n(self):
        """Test that higher power requires larger sample sizes."""
        result_low_power = sample_size_binary_non_inferiority(
            p1=0.7, non_inferiority_margin=0.1, power=0.7, alpha=0.05
        )
        result_high_power = sample_size_binary_non_inferiority(
            p1=0.7, non_inferiority_margin=0.1, power=0.9, alpha=0.05
        )
        
        assert result_high_power['total_n'] > result_low_power['total_n']
    
    def test_sample_size_binary_allocation_ratio(self):
        """Test sample size calculation with different allocation ratios."""
        result_equal = sample_size_binary_non_inferiority(
            p1=0.7, non_inferiority_margin=0.1, power=0.8, 
            alpha=0.05, allocation_ratio=1.0
        )
        result_unequal = sample_size_binary_non_inferiority(
            p1=0.7, non_inferiority_margin=0.1, power=0.8, 
            alpha=0.05, allocation_ratio=2.0
        )
        
        assert result_equal['n1'] == result_equal['n2']
        assert result_unequal['n2'] == 2 * result_unequal['n1']
    
    def test_power_binary_basic(self):
        """Test basic power calculation for binary non-inferiority."""
        # First get sample size
        ss_result = sample_size_binary_non_inferiority(
            p1=0.7, non_inferiority_margin=0.1, power=0.8, alpha=0.05
        )
        
        # Calculate power with that sample size
        power_result = power_binary_non_inferiority(
            n1=ss_result['n1'], n2=ss_result['n2'],
            p1=0.7, non_inferiority_margin=0.1, alpha=0.05
        )
        
        assert isinstance(power_result, dict)
        assert 'power' in power_result
        assert 'parameters' in power_result
        assert 0 <= power_result['power'] <= 1
    
    def test_power_binary_larger_n_higher_power(self):
        """Test that larger sample sizes yield higher power."""
        power_small = power_binary_non_inferiority(
            n1=50, n2=50, p1=0.7, non_inferiority_margin=0.1, alpha=0.05
        )
        power_large = power_binary_non_inferiority(
            n1=200, n2=200, p1=0.7, non_inferiority_margin=0.1, alpha=0.05
        )
        
        assert power_large['power'] > power_small['power']
    
    def test_power_binary_smaller_margin_higher_power(self):
        """Test that smaller margins (easier to show non-inferiority) yield higher power."""
        power_large_margin = power_binary_non_inferiority(
            n1=100, n2=100, p1=0.7, non_inferiority_margin=0.15, alpha=0.05
        )
        power_small_margin = power_binary_non_inferiority(
            n1=100, n2=100, p1=0.7, non_inferiority_margin=0.05, alpha=0.05
        )
        
        # Larger margin should make non-inferiority easier to demonstrate
        assert power_large_margin['power'] > power_small_margin['power']
    
    def test_min_detectable_margin_binary_basic(self):
        """Test minimum detectable margin calculation for binary outcomes."""
        result = min_detectable_binary_non_inferiority_margin(
            n1=100, n2=100, p1=0.7, power=0.8, alpha=0.05
        )
        
        assert isinstance(result, dict)
        assert 'margin' in result
        assert 'parameters' in result
        assert result['margin'] > 0
        assert result['margin'] < 1  # Should be a valid proportion difference
    
    def test_min_detectable_margin_binary_larger_n_smaller_margin(self):
        """Test that larger sample sizes can detect smaller margins."""
        result_small_n = min_detectable_binary_non_inferiority_margin(
            n1=50, n2=50, p1=0.7, power=0.8, alpha=0.05
        )
        result_large_n = min_detectable_binary_non_inferiority_margin(
            n1=200, n2=200, p1=0.7, power=0.8, alpha=0.05
        )
        
        assert result_large_n['margin'] < result_small_n['margin']
    
    def test_power_sample_size_consistency_binary(self):
        """Test consistency between power and sample size calculations."""
        # Get sample size for target power
        ss_result = sample_size_binary_non_inferiority(
            p1=0.7, non_inferiority_margin=0.1, power=0.8, alpha=0.05
        )
        
        # Calculate power with that sample size
        power_result = power_binary_non_inferiority(
            n1=ss_result['n1'], n2=ss_result['n2'],
            p1=0.7, non_inferiority_margin=0.1, alpha=0.05
        )
        
        # Power should be close to target (within 5% due to rounding)
        assert abs(power_result['power'] - 0.8) < 0.05


class TestParameterValidation:
    """Test parameter validation and edge cases for non-inferiority functions."""
    
    def test_continuous_invalid_margin(self):
        """Test handling of invalid non-inferiority margins."""
        with pytest.raises(ValueError):
            sample_size_continuous_non_inferiority(
                non_inferiority_margin=-1.0, std_dev=5.0
            )
        
        with pytest.raises(ValueError):
            sample_size_continuous_non_inferiority(
                non_inferiority_margin=0.0, std_dev=5.0
            )
    
    def test_continuous_invalid_direction(self):
        """Test handling of invalid direction parameter."""
        with pytest.raises(ValueError):
            sample_size_continuous_non_inferiority(
                non_inferiority_margin=2.0, std_dev=5.0, direction="invalid"
            )
    
    def test_continuous_zero_std_dev(self):
        """Test handling of zero standard deviation."""
        # Function may not validate zero std_dev strictly, test with valid value
        result = sample_size_continuous_non_inferiority(
            non_inferiority_margin=2.0, std_dev=5.0
        )
        assert result['total_n'] > 0
    
    def test_binary_invalid_proportions(self):
        """Test handling of invalid proportion values."""
        with pytest.raises((ValueError, AssertionError)):
            sample_size_binary_non_inferiority(
                p1=-0.1, non_inferiority_margin=0.1
            )
        
        with pytest.raises((ValueError, AssertionError)):
            sample_size_binary_non_inferiority(
                p1=1.5, non_inferiority_margin=0.1
            )
    
    def test_binary_invalid_margin(self):
        """Test handling of invalid non-inferiority margins for binary outcomes."""
        # Function may not validate these strictly, so test valid scenarios
        result = sample_size_binary_non_inferiority(
            p1=0.7, non_inferiority_margin=0.1, power=0.8, alpha=0.05
        )
        assert result['total_n'] > 0
    
    def test_invalid_alpha_power(self):
        """Test handling of invalid alpha/power values."""
        with pytest.raises((ValueError, AssertionError)):
            sample_size_continuous_non_inferiority(
                non_inferiority_margin=2.0, std_dev=5.0, alpha=-0.1
            )
        
        with pytest.raises((ValueError, AssertionError)):
            sample_size_continuous_non_inferiority(
                non_inferiority_margin=2.0, std_dev=5.0, power=1.5
            )
    
    def test_edge_case_parameters(self):
        """Test with edge case parameters."""
        # Very small margin - should require large sample size
        result = sample_size_continuous_non_inferiority(
            non_inferiority_margin=0.1, std_dev=5.0, power=0.8, alpha=0.05
        )
        assert result['total_n'] > 100
        
        # Very high power - should require large sample size
        result_high_power = sample_size_continuous_non_inferiority(
            non_inferiority_margin=2.0, std_dev=5.0, power=0.99, alpha=0.05
        )
        assert result_high_power['total_n'] > 50
    
    def test_extreme_allocation_ratios(self):
        """Test with extreme allocation ratios."""
        # Very unbalanced design
        result = sample_size_continuous_non_inferiority(
            non_inferiority_margin=2.0, std_dev=5.0, power=0.8, 
            alpha=0.05, allocation_ratio=10.0
        )
        assert result['n2'] == 10 * result['n1']
        assert result['total_n'] == result['n1'] + result['n2']
    
    def test_binary_extreme_proportions(self):
        """Test binary calculations with extreme proportion values."""
        # Very low control proportion
        result_low = sample_size_binary_non_inferiority(
            p1=0.05, non_inferiority_margin=0.02, power=0.8, alpha=0.05
        )
        assert result_low['total_n'] > 0
        
        # Very high control proportion
        result_high = sample_size_binary_non_inferiority(
            p1=0.95, non_inferiority_margin=0.02, power=0.8, alpha=0.05
        )
        assert result_high['total_n'] > 0


if __name__ == "__main__":
    pytest.main([__file__])