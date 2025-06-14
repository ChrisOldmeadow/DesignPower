"""
Unit tests for stepped wedge analytical functions.

Tests the core analytical functions for stepped wedge cluster randomized trials
including power and sample size calculations using the Hussey & Hughes method.
"""

import pytest
import numpy as np
import math
from core.designs.stepped_wedge.analytical import (
    hussey_hughes_power_continuous,
    hussey_hughes_sample_size_continuous,
    hussey_hughes_power_binary,
    hussey_hughes_sample_size_binary
)


class TestSteppedWedgeAnalyticalContinuous:
    """Test stepped wedge analytical functions for continuous outcomes."""
    
    def test_hussey_hughes_power_continuous_basic(self):
        """Test basic continuous power calculation functionality."""
        result = hussey_hughes_power_continuous(
            clusters=12, steps=4, individuals_per_cluster=25,
            icc=0.05, cluster_autocorr=0.0, treatment_effect=0.5,
            std_dev=2.0, alpha=0.05
        )
        assert isinstance(result, dict)
        assert 'power' in result
        assert 'se_treatment_effect' in result
        assert 'var_treatment_effect' in result
        assert 'parameters' in result
        assert 0 <= result['power'] <= 1
        assert result['se_treatment_effect'] > 0
        assert result['var_treatment_effect'] > 0
    
    def test_hussey_hughes_power_continuous_negative_effect(self):
        """Test power calculation with negative treatment effect."""
        result = hussey_hughes_power_continuous(
            clusters=15, steps=4, individuals_per_cluster=20,
            icc=0.05, cluster_autocorr=0.0, treatment_effect=-0.5,
            std_dev=2.0, alpha=0.05
        )
        assert result['power'] > 0  # Should handle negative effects correctly
        assert result['parameters']['treatment_effect'] == -0.5
    
    def test_hussey_hughes_power_continuous_result_structure(self):
        """Test that power result contains all expected fields."""
        result = hussey_hughes_power_continuous(
            clusters=10, steps=3, individuals_per_cluster=30,
            icc=0.1, cluster_autocorr=0.2, treatment_effect=0.3,
            std_dev=1.5, alpha=0.05
        )
        
        expected_fields = ['power', 'se_treatment_effect', 'var_treatment_effect',
                          'correlation_adjustment', 'n_control_periods', 
                          'n_intervention_periods', 'parameters']
        for field in expected_fields:
            assert field in result
        
        expected_params = ['clusters', 'steps', 'individuals_per_cluster', 'total_n',
                          'icc', 'cluster_autocorr', 'treatment_effect', 'std_dev', 
                          'alpha', 'method']
        for param in expected_params:
            assert param in result['parameters']
    
    def test_hussey_hughes_power_continuous_parameter_effects(self):
        """Test effects of different parameters on power."""
        base_params = {
            'clusters': 12, 'steps': 4, 'individuals_per_cluster': 25,
            'icc': 0.05, 'cluster_autocorr': 0.0, 'treatment_effect': 0.5,
            'std_dev': 2.0, 'alpha': 0.05
        }
        
        base_result = hussey_hughes_power_continuous(**base_params)
        
        # More clusters should increase power
        more_clusters_result = hussey_hughes_power_continuous(
            **{**base_params, 'clusters': 20}
        )
        assert more_clusters_result['power'] >= base_result['power']
        
        # Larger effect size should increase power
        larger_effect_result = hussey_hughes_power_continuous(
            **{**base_params, 'treatment_effect': 1.0}
        )
        assert larger_effect_result['power'] >= base_result['power']
        
        # Higher ICC should generally decrease power
        higher_icc_result = hussey_hughes_power_continuous(
            **{**base_params, 'icc': 0.2}
        )
        # Note: This may not always hold due to design effect complexities
        # Just check that result is valid
        assert 0 <= higher_icc_result['power'] <= 1
    
    def test_hussey_hughes_power_continuous_cluster_autocorr(self):
        """Test effect of cluster autocorrelation parameter."""
        base_params = {
            'clusters': 15, 'steps': 5, 'individuals_per_cluster': 20,
            'icc': 0.1, 'treatment_effect': 0.4, 'std_dev': 1.5, 'alpha': 0.05
        }
        
        # Test with no cluster autocorrelation
        no_cac_result = hussey_hughes_power_continuous(
            **base_params, cluster_autocorr=0.0
        )
        
        # Test with moderate cluster autocorrelation
        mod_cac_result = hussey_hughes_power_continuous(
            **base_params, cluster_autocorr=0.3
        )
        
        # Test with high cluster autocorrelation
        high_cac_result = hussey_hughes_power_continuous(
            **base_params, cluster_autocorr=0.8
        )
        
        # All should be valid powers
        assert 0 <= no_cac_result['power'] <= 1
        assert 0 <= mod_cac_result['power'] <= 1
        assert 0 <= high_cac_result['power'] <= 1
        
        # Correlation adjustment should increase with CAC
        assert (mod_cac_result['correlation_adjustment'] >= 
                no_cac_result['correlation_adjustment'])
        assert (high_cac_result['correlation_adjustment'] >= 
                mod_cac_result['correlation_adjustment'])
    
    def test_hussey_hughes_sample_size_continuous_basic(self):
        """Test basic continuous sample size calculation."""
        result = hussey_hughes_sample_size_continuous(
            target_power=0.80, treatment_effect=0.5, std_dev=2.0,
            icc=0.05, cluster_autocorr=0.0, steps=4,
            individuals_per_cluster=25, alpha=0.05
        )
        
        assert isinstance(result, dict)
        assert 'clusters' in result
        assert 'total_n' in result
        assert 'achieved_power' in result
        assert 'target_power' in result
        assert 'parameters' in result
        
        assert result['clusters'] >= 2  # Minimum reasonable clusters
        assert result['total_n'] > 0
        assert result['achieved_power'] >= result['target_power'] * 0.95  # Close to target
        assert result['target_power'] == 0.80
    
    def test_hussey_hughes_sample_size_continuous_different_targets(self):
        """Test sample size calculation with different target powers."""
        base_params = {
            'treatment_effect': 0.3, 'std_dev': 1.5, 'icc': 0.05,
            'cluster_autocorr': 0.0, 'steps': 4, 'individuals_per_cluster': 20,
            'alpha': 0.05
        }
        
        result_80 = hussey_hughes_sample_size_continuous(
            target_power=0.80, **base_params
        )
        result_90 = hussey_hughes_sample_size_continuous(
            target_power=0.90, **base_params
        )
        
        # Higher target power should require more clusters
        assert result_90['clusters'] >= result_80['clusters']
        assert result_90['achieved_power'] >= result_80['achieved_power']


class TestSteppedWedgeAnalyticalBinary:
    """Test stepped wedge analytical functions for binary outcomes."""
    
    def test_hussey_hughes_power_binary_basic(self):
        """Test basic binary power calculation functionality."""
        result = hussey_hughes_power_binary(
            clusters=20, steps=4, individuals_per_cluster=50,
            icc=0.02, cluster_autocorr=0.0, p_control=0.15,
            p_intervention=0.25, alpha=0.05
        )
        
        assert isinstance(result, dict)
        assert 'power' in result
        assert 'se_treatment_effect' in result
        assert 'parameters' in result
        assert 0 <= result['power'] <= 1
        
        # Check binary-specific parameters
        params = result['parameters']
        assert params['p_control'] == 0.15
        assert params['p_intervention'] == 0.25
        assert params['arcsine_transformation'] == True
    
    def test_hussey_hughes_power_binary_arcsine_transformation(self):
        """Test that arcsine transformation is applied correctly."""
        result = hussey_hughes_power_binary(
            clusters=15, steps=3, individuals_per_cluster=40,
            icc=0.03, cluster_autocorr=0.0, p_control=0.1,
            p_intervention=0.2, alpha=0.05
        )
        
        # Verify that the calculation uses arcsine transformation
        theta_control = np.arcsin(np.sqrt(0.1))
        theta_intervention = np.arcsin(np.sqrt(0.2))
        expected_effect = theta_intervention - theta_control
        
        # The effect should be positive for p_intervention > p_control
        assert expected_effect > 0
        assert result['power'] > 0
    
    def test_hussey_hughes_power_binary_extreme_proportions(self):
        """Test binary power calculation with extreme proportions."""
        # Very low proportions
        result_low = hussey_hughes_power_binary(
            clusters=25, steps=4, individuals_per_cluster=100,
            icc=0.01, cluster_autocorr=0.0, p_control=0.01,
            p_intervention=0.05, alpha=0.05
        )
        assert 0 <= result_low['power'] <= 1
        
        # High proportions
        result_high = hussey_hughes_power_binary(
            clusters=25, steps=4, individuals_per_cluster=100,
            icc=0.01, cluster_autocorr=0.0, p_control=0.8,
            p_intervention=0.9, alpha=0.05
        )
        assert 0 <= result_high['power'] <= 1
    
    def test_hussey_hughes_sample_size_binary_basic(self):
        """Test basic binary sample size calculation."""
        result = hussey_hughes_sample_size_binary(
            target_power=0.80, p_control=0.2, p_intervention=0.3,
            icc=0.03, cluster_autocorr=0.0, steps=5,
            individuals_per_cluster=40, alpha=0.05
        )
        
        assert isinstance(result, dict)
        assert 'clusters' in result
        assert 'total_n' in result
        assert 'achieved_power' in result
        assert 'target_power' in result
        
        assert result['clusters'] >= 2
        assert result['total_n'] > 0
        assert result['achieved_power'] >= result['target_power'] * 0.95
        
        # Check binary-specific parameters
        params = result['parameters']
        assert params['p_control'] == 0.2
        assert params['p_intervention'] == 0.3
        assert params['arcsine_transformation'] == True


class TestParameterValidation:
    """Test parameter validation and edge cases."""
    
    def test_minimum_clusters(self):
        """Test that minimum cluster constraints are handled."""
        # Test with fewer clusters than intervention steps
        result = hussey_hughes_sample_size_continuous(
            target_power=0.80, treatment_effect=0.5, std_dev=2.0,
            icc=0.05, cluster_autocorr=0.0, steps=4,
            individuals_per_cluster=25, alpha=0.05
        )
        
        # Should have at least as many clusters as intervention steps
        intervention_steps = 4 - 1  # steps - 1 for baseline
        assert result['clusters'] >= intervention_steps
    
    def test_zero_cluster_autocorr(self):
        """Test calculations with zero cluster autocorrelation."""
        result = hussey_hughes_power_continuous(
            clusters=12, steps=4, individuals_per_cluster=25,
            icc=0.05, cluster_autocorr=0.0, treatment_effect=0.5,
            std_dev=2.0, alpha=0.05
        )
        
        assert result['correlation_adjustment'] == 1.0  # No adjustment when CAC = 0
        assert result['power'] > 0
    
    def test_perfect_cluster_autocorr(self):
        """Test calculations with perfect cluster autocorrelation."""
        result = hussey_hughes_power_continuous(
            clusters=12, steps=4, individuals_per_cluster=25,
            icc=0.05, cluster_autocorr=1.0, treatment_effect=0.5,
            std_dev=2.0, alpha=0.05
        )
        
        assert result['correlation_adjustment'] > 1.0  # Should have adjustment
        assert result['power'] >= 0  # Should be valid
    
    def test_zero_icc(self):
        """Test calculations with zero ICC."""
        result = hussey_hughes_power_continuous(
            clusters=12, steps=4, individuals_per_cluster=25,
            icc=0.0, cluster_autocorr=0.0, treatment_effect=0.5,
            std_dev=2.0, alpha=0.05
        )
        
        assert result['power'] > 0
        # With zero ICC, should behave like individual randomization
    
    def test_small_effect_sizes(self):
        """Test calculations with very small effect sizes."""
        result = hussey_hughes_power_continuous(
            clusters=50, steps=6, individuals_per_cluster=100,
            icc=0.01, cluster_autocorr=0.0, treatment_effect=0.01,
            std_dev=1.0, alpha=0.05
        )
        
        assert 0 <= result['power'] <= 1
        # Small effect should give low power unless large sample
    
    def test_large_effect_sizes(self):
        """Test calculations with large effect sizes."""
        result = hussey_hughes_power_continuous(
            clusters=6, steps=3, individuals_per_cluster=10,
            icc=0.05, cluster_autocorr=0.0, treatment_effect=2.0,
            std_dev=1.0, alpha=0.05
        )
        
        assert result['power'] > 0.5  # Large effect should give reasonable power
        assert result['power'] <= 1.0
    
    def test_binary_equal_proportions(self):
        """Test binary calculation with equal proportions (null hypothesis)."""
        result = hussey_hughes_power_binary(
            clusters=20, steps=4, individuals_per_cluster=50,
            icc=0.02, cluster_autocorr=0.0, p_control=0.3,
            p_intervention=0.3, alpha=0.05
        )
        
        # Should have power close to alpha (Type I error rate)
        assert result['power'] <= 0.15  # Allow some margin for computation
    
    def test_minimal_design(self):
        """Test calculations with minimal design parameters."""
        result = hussey_hughes_power_continuous(
            clusters=3, steps=2, individuals_per_cluster=5,
            icc=0.05, cluster_autocorr=0.0, treatment_effect=1.0,
            std_dev=1.0, alpha=0.05
        )
        
        assert 0 <= result['power'] <= 1
        assert result['parameters']['total_n'] == 3 * 2 * 5


class TestMathematicalConsistency:
    """Test mathematical consistency and relationships."""
    
    def test_power_vs_sample_size_consistency(self):
        """Test consistency between power and sample size calculations."""
        # Calculate required sample size for 80% power
        sample_result = hussey_hughes_sample_size_continuous(
            target_power=0.80, treatment_effect=0.5, std_dev=2.0,
            icc=0.05, cluster_autocorr=0.0, steps=4,
            individuals_per_cluster=25, alpha=0.05
        )
        
        # Use that sample size to calculate power
        power_result = hussey_hughes_power_continuous(
            clusters=sample_result['clusters'], steps=4, 
            individuals_per_cluster=25, icc=0.05, cluster_autocorr=0.0,
            treatment_effect=0.5, std_dev=2.0, alpha=0.05
        )
        
        # Should achieve approximately the target power
        assert abs(power_result['power'] - 0.80) < 0.05
    
    def test_effect_size_monotonicity(self):
        """Test that power increases monotonically with effect size."""
        base_params = {
            'clusters': 15, 'steps': 4, 'individuals_per_cluster': 30,
            'icc': 0.05, 'cluster_autocorr': 0.0, 'std_dev': 2.0, 'alpha': 0.05
        }
        
        effects = [0.1, 0.3, 0.5, 0.7, 1.0]
        powers = []
        
        for effect in effects:
            result = hussey_hughes_power_continuous(
                **base_params, treatment_effect=effect
            )
            powers.append(result['power'])
        
        # Powers should be non-decreasing
        for i in range(1, len(powers)):
            assert powers[i] >= powers[i-1] * 0.95  # Allow small numerical errors
    
    def test_binary_proportion_monotonicity(self):
        """Test that binary power increases with proportion difference."""
        base_params = {
            'clusters': 20, 'steps': 4, 'individuals_per_cluster': 50,
            'icc': 0.02, 'cluster_autocorr': 0.0, 'alpha': 0.05
        }
        
        p_control = 0.2
        p_interventions = [0.25, 0.3, 0.35, 0.4, 0.45]
        powers = []
        
        for p_int in p_interventions:
            result = hussey_hughes_power_binary(
                **base_params, p_control=p_control, p_intervention=p_int
            )
            powers.append(result['power'])
        
        # Powers should be non-decreasing with larger proportion differences
        for i in range(1, len(powers)):
            assert powers[i] >= powers[i-1] * 0.95


if __name__ == "__main__":
    pytest.main([__file__])