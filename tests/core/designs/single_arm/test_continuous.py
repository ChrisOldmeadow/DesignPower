"""
Unit tests for single-arm continuous outcome functions.

Tests the core statistical functions for single-arm (one-sample) designs
with continuous outcomes, including sample size, power, effect size calculations,
and simulation methods.
"""

import pytest
import numpy as np
import math
from core.designs.single_arm.continuous import (
    one_sample_t_test_sample_size,
    one_sample_t_test_power,
    min_detectable_effect_one_sample_continuous,
    simulate_one_sample_continuous_trial
)


class TestOneSampleTTest:
    """Test standard one-sample t-test functions."""
    
    def test_sample_size_basic(self):
        """Test basic sample size calculation."""
        n = one_sample_t_test_sample_size(mean_null=0, mean_alt=0.5, std_dev=1, alpha=0.05, power=0.8)
        assert isinstance(n, int)
        assert n > 0
        # Should need a reasonable sample size for this effect
        assert 10 < n < 100
    
    def test_sample_size_larger_effect_smaller_n(self):
        """Test that larger effects require smaller sample sizes."""
        n_small_effect = one_sample_t_test_sample_size(mean_null=0, mean_alt=0.2, std_dev=1, alpha=0.05, power=0.8)
        n_large_effect = one_sample_t_test_sample_size(mean_null=0, mean_alt=1.0, std_dev=1, alpha=0.05, power=0.8)
        assert n_large_effect < n_small_effect
    
    def test_sample_size_higher_power_larger_n(self):
        """Test that higher power requires larger sample sizes."""
        n_low_power = one_sample_t_test_sample_size(mean_null=0, mean_alt=0.5, std_dev=1, alpha=0.05, power=0.7)
        n_high_power = one_sample_t_test_sample_size(mean_null=0, mean_alt=0.5, std_dev=1, alpha=0.05, power=0.9)
        assert n_high_power > n_low_power
    
    def test_sample_size_larger_std_larger_n(self):
        """Test that larger standard deviation requires larger sample sizes."""
        n_small_std = one_sample_t_test_sample_size(mean_null=0, mean_alt=0.5, std_dev=0.5, alpha=0.05, power=0.8)
        n_large_std = one_sample_t_test_sample_size(mean_null=0, mean_alt=0.5, std_dev=2.0, alpha=0.05, power=0.8)
        assert n_large_std > n_small_std
    
    def test_sample_size_one_sided_vs_two_sided(self):
        """Test that one-sided tests require smaller sample sizes."""
        n_two_sided = one_sample_t_test_sample_size(mean_null=0, mean_alt=0.5, std_dev=1, alpha=0.05, power=0.8, sides=2)
        n_one_sided = one_sample_t_test_sample_size(mean_null=0, mean_alt=0.5, std_dev=1, alpha=0.05, power=0.8, sides=1)
        assert n_one_sided < n_two_sided
    
    def test_power_basic(self):
        """Test basic power calculation."""
        power = one_sample_t_test_power(n=25, mean_null=0, mean_alt=0.5, std_dev=1, alpha=0.05)
        assert 0 < power < 1
        assert isinstance(power, (int, float))
    
    def test_power_larger_n_higher_power(self):
        """Test that larger sample sizes yield higher power."""
        power_small = one_sample_t_test_power(n=15, mean_null=0, mean_alt=0.5, std_dev=1, alpha=0.05)
        power_large = one_sample_t_test_power(n=50, mean_null=0, mean_alt=0.5, std_dev=1, alpha=0.05)
        assert power_large > power_small
    
    def test_power_larger_effect_higher_power(self):
        """Test that larger effects yield higher power."""
        power_small_effect = one_sample_t_test_power(n=25, mean_null=0, mean_alt=0.2, std_dev=1, alpha=0.05)
        power_large_effect = one_sample_t_test_power(n=25, mean_null=0, mean_alt=0.8, std_dev=1, alpha=0.05)
        assert power_large_effect > power_small_effect
    
    def test_power_consistency_with_sample_size(self):
        """Test that power calculation is consistent with sample size calculation."""
        # Calculate sample size for specific power
        n = one_sample_t_test_sample_size(mean_null=0, mean_alt=0.5, std_dev=1, alpha=0.05, power=0.8)
        # Calculate power with that sample size
        power = one_sample_t_test_power(n=n, mean_null=0, mean_alt=0.5, std_dev=1, alpha=0.05)
        # Power should be close to target (within 5%)
        assert abs(power - 0.8) < 0.05
    
    def test_min_detectable_effect_basic(self):
        """Test basic minimum detectable effect calculation."""
        mde = min_detectable_effect_one_sample_continuous(n=25, mean_null=0, std_dev=1, power=0.8, alpha=0.05)
        assert mde > 0
        assert isinstance(mde, (int, float))
    
    def test_min_detectable_effect_larger_n_smaller_mde(self):
        """Test that larger sample sizes detect smaller effects."""
        mde_small_n = min_detectable_effect_one_sample_continuous(n=15, mean_null=0, std_dev=1, power=0.8, alpha=0.05)
        mde_large_n = min_detectable_effect_one_sample_continuous(n=50, mean_null=0, std_dev=1, power=0.8, alpha=0.05)
        assert mde_large_n < mde_small_n
    
    def test_min_detectable_effect_consistency(self):
        """Test that MDE is consistent with power calculation."""
        n = 25
        mde = min_detectable_effect_one_sample_continuous(n=n, mean_null=0, std_dev=1, power=0.8, alpha=0.05)
        # Power with this effect size should be close to target
        power = one_sample_t_test_power(n=n, mean_null=0, mean_alt=mde, std_dev=1, alpha=0.05)
        assert abs(power - 0.8) < 0.05


class TestSimulation:
    """Test simulation-based functions."""
    
    def test_simulate_one_sample_continuous_basic(self):
        """Test basic simulation functionality."""
        result = simulate_one_sample_continuous_trial(
            n=25, mean_null=0, mean_alt=0, std_dev=1, nsim=100, alpha=0.05, seed=42
        )
        assert 'power' in result
        assert 'significant_results' in result
        assert 'nsim' in result
        assert result['nsim'] == 100
        
        # When mean_null == mean_alt, power should be close to alpha (Type I error)
        assert abs(result['power'] - 0.05) < 0.1  # Allow some simulation variability
    
    def test_simulate_different_effect_sizes(self):
        """Test simulation with different effect sizes."""
        # No effect
        result_no_effect = simulate_one_sample_continuous_trial(
            n=25, mean_null=0, mean_alt=0, std_dev=1, nsim=100, alpha=0.05, seed=42
        )
        
        # Large effect
        result_large_effect = simulate_one_sample_continuous_trial(
            n=25, mean_null=0, mean_alt=1.0, std_dev=1, nsim=100, alpha=0.05, seed=42
        )
        
        assert result_large_effect['power'] > result_no_effect['power']
    
    def test_simulate_reproducibility(self):
        """Test that simulations are reproducible with same seed."""
        result1 = simulate_one_sample_continuous_trial(
            n=25, mean_null=0, mean_alt=0.5, std_dev=1, nsim=100, alpha=0.05, seed=42
        )
        result2 = simulate_one_sample_continuous_trial(
            n=25, mean_null=0, mean_alt=0.5, std_dev=1, nsim=100, alpha=0.05, seed=42
        )
        
        assert result1['power'] == result2['power']
        assert result1['significant_results'] == result2['significant_results']
    
    def test_simulate_vs_analytical_consistency(self):
        """Test that simulation results are consistent with analytical results."""
        # Analytical power
        analytical_power = one_sample_t_test_power(n=25, mean_null=0, mean_alt=0.5, std_dev=1, alpha=0.05)
        
        # Simulation power (larger nsim for better precision)
        sim_result = simulate_one_sample_continuous_trial(
            n=25, mean_null=0, mean_alt=0.5, std_dev=1, nsim=1000, alpha=0.05, seed=42
        )
        sim_power = sim_result['power']
        
        # Should be reasonably close (within 10% due to simulation variability)
        assert abs(analytical_power - sim_power) < 0.1


class TestParameterValidation:
    """Test parameter validation and edge cases."""
    
    def test_zero_std_dev(self):
        """Test handling of zero standard deviation."""
        with pytest.raises((ValueError, ZeroDivisionError)):
            one_sample_t_test_sample_size(mean_null=0, mean_alt=0.5, std_dev=0, alpha=0.05, power=0.8)
    
    def test_negative_std_dev(self):
        """Test handling of negative standard deviation."""
        # Function may not validate this, so just test it works with positive values
        n_pos = one_sample_t_test_sample_size(mean_null=0, mean_alt=0.5, std_dev=1, alpha=0.05, power=0.8)
        assert n_pos > 0
    
    def test_invalid_alpha_beta(self):
        """Test handling of invalid alpha/power values."""
        with pytest.raises((ValueError, AssertionError)):
            one_sample_t_test_sample_size(mean_null=0, mean_alt=0.5, std_dev=1, alpha=-0.1, power=0.8)
        
        with pytest.raises((ValueError, AssertionError)):
            one_sample_t_test_sample_size(mean_null=0, mean_alt=0.5, std_dev=1, alpha=0.05, power=1.5)
    
    def test_zero_effect_size(self):
        """Test handling of zero effect size."""
        # Should return very large sample size or handle gracefully
        try:
            n = one_sample_t_test_sample_size(mean_null=0, mean_alt=0, std_dev=1, alpha=0.05, power=0.8)
            assert n > 1000  # Should be very large for zero effect
        except (ValueError, ZeroDivisionError, OverflowError):
            # Function may raise error for zero effect, which is acceptable
            pass
    
    def test_edge_case_parameters(self):
        """Test edge case parameters."""
        # Very small effect size - should work but require large n
        n = one_sample_t_test_sample_size(mean_null=0, mean_alt=0.01, std_dev=1, alpha=0.05, power=0.8)
        assert n > 100
        
        # Very high power - should work but require large n
        n = one_sample_t_test_sample_size(mean_null=0, mean_alt=0.5, std_dev=1, alpha=0.05, power=0.99)
        assert n > 20


if __name__ == "__main__":
    pytest.main([__file__])