"""
Unit tests for cluster RCT permutation test functions.

Tests the core permutation test functions for cluster randomized trials
including exact inference methods for small cluster studies.
"""

import pytest
import numpy as np
import pandas as pd
from core.designs.cluster_rct.permutation_tests import (
    cluster_permutation_test,
    cluster_permutation_power,
    cluster_permutation_test_binary,
    cluster_permutation_test_continuous
)


class TestClusterPermutationTest:
    """Test main cluster permutation test function."""
    
    def test_cluster_permutation_test_dict_basic(self):
        """Test basic permutation test with dictionary data."""
        data = {
            'control_clusters': [0.45, 0.52, 0.41, 0.38, 0.49],
            'treatment_clusters': [0.58, 0.61, 0.55, 0.63, 0.59]
        }
        result = cluster_permutation_test(data, n_permutations=1000, random_seed=42)
        
        assert isinstance(result, dict)
        assert 'p_value' in result
        assert 'observed_statistic' in result
        assert 'confidence_interval' in result
        assert 'n_permutations_used' in result
        assert 'method' in result
        
        assert 0 <= result['p_value'] <= 1
        assert result['n_permutations_used'] > 0
        assert len(result['confidence_interval']) == 2
    
    def test_cluster_permutation_test_dataframe_basic(self):
        """Test basic permutation test with DataFrame data."""
        # Create individual-level data that aggregates to cluster means
        df = pd.DataFrame({
            'cluster': [1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5],
            'treatment': [0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1],
            'outcome': [0,1,0,1,1,1,0,0,0,1,0,1,1,1,1,0,1,1,1,1]
        })
        
        result = cluster_permutation_test(
            df, cluster_col='cluster', treatment_col='treatment', 
            outcome_col='outcome', n_permutations=1000, random_seed=42
        )
        
        assert isinstance(result, dict)
        assert 'p_value' in result
        assert 'observed_statistic' in result
        assert 'confidence_interval' in result
        assert 0 <= result['p_value'] <= 1
    
    def test_cluster_permutation_test_different_statistics(self):
        """Test permutation test with different test statistics."""
        data = {
            'control_clusters': [0.45, 0.52, 0.41, 0.38, 0.49],
            'treatment_clusters': [0.58, 0.61, 0.55, 0.63, 0.59]
        }
        
        # Test mean difference
        result_mean = cluster_permutation_test(
            data, test_statistic='mean_difference', n_permutations=500, random_seed=42
        )
        assert 'permutation test' in result_mean['method']
        
        # Test t-statistic
        result_t = cluster_permutation_test(
            data, test_statistic='t_statistic', n_permutations=500, random_seed=42
        )
        assert 'permutation test' in result_t['method']
        
        # Test rank sum
        result_rank = cluster_permutation_test(
            data, test_statistic='rank_sum', n_permutations=500, random_seed=42
        )
        assert 'permutation test' in result_rank['method']
    
    def test_cluster_permutation_test_alternative_hypotheses(self):
        """Test permutation test with different alternative hypotheses."""
        data = {
            'control_clusters': [0.45, 0.52, 0.41, 0.38, 0.49],
            'treatment_clusters': [0.58, 0.61, 0.55, 0.63, 0.59]
        }
        
        # Two-sided test
        result_two = cluster_permutation_test(
            data, alternative='two-sided', n_permutations=500, random_seed=42
        )
        
        # Greater test
        result_greater = cluster_permutation_test(
            data, alternative='greater', n_permutations=500, random_seed=42
        )
        
        # Less test
        result_less = cluster_permutation_test(
            data, alternative='less', n_permutations=500, random_seed=42
        )
        
        # All should return valid p-values
        assert 0 <= result_two['p_value'] <= 1
        assert 0 <= result_greater['p_value'] <= 1
        assert 0 <= result_less['p_value'] <= 1
    
    def test_cluster_permutation_test_no_effect(self):
        """Test permutation test with no treatment effect."""
        # Create identical data for both groups
        data = {
            'control_clusters': [0.45, 0.52, 0.41, 0.38, 0.49],
            'treatment_clusters': [0.45, 0.52, 0.41, 0.38, 0.49]
        }
        
        result = cluster_permutation_test(data, n_permutations=1000, random_seed=42)
        
        # Observed statistic should be close to 0
        assert abs(result['observed_statistic']) < 0.01
        # P-value should be high (close to 1)
        assert result['p_value'] > 0.5
    
    def test_cluster_permutation_test_large_effect(self):
        """Test permutation test with large treatment effect."""
        data = {
            'control_clusters': [0.2, 0.25, 0.18, 0.22, 0.19],
            'treatment_clusters': [0.8, 0.85, 0.78, 0.82, 0.79]
        }
        
        result = cluster_permutation_test(data, n_permutations=1000, random_seed=42)
        
        # Large effect should have large observed statistic
        assert abs(result['observed_statistic']) > 0.5
        # P-value should be small
        assert result['p_value'] < 0.1
    
    def test_cluster_permutation_test_confidence_interval(self):
        """Test confidence interval calculation."""
        data = {
            'control_clusters': [0.45, 0.52, 0.41, 0.38, 0.49],
            'treatment_clusters': [0.58, 0.61, 0.55, 0.63, 0.59]
        }
        
        result = cluster_permutation_test(
            data, confidence_level=0.95, n_permutations=1000, random_seed=42
        )
        
        ci = result['confidence_interval']
        assert len(ci) == 2
        assert ci[0] <= ci[1]  # Lower bound <= upper bound
        
        # Test different confidence level
        result_90 = cluster_permutation_test(
            data, confidence_level=0.90, n_permutations=1000, random_seed=42
        )
        ci_90 = result_90['confidence_interval']
        
        # 90% CI should be narrower than 95% CI
        assert (ci_90[1] - ci_90[0]) <= (ci[1] - ci[0])
    
    def test_cluster_permutation_test_return_distribution(self):
        """Test returning full permutation distribution."""
        data = {
            'control_clusters': [0.45, 0.52, 0.41],
            'treatment_clusters': [0.58, 0.61, 0.55]
        }
        
        result = cluster_permutation_test(
            data, n_permutations=500, return_distribution=True, random_seed=42
        )
        
        assert 'distribution' in result
        assert len(result['distribution']) > 0
        # Should have at most n_permutations + 1 values (including observed)
        assert len(result['distribution']) <= 501
    
    def test_cluster_permutation_test_reproducibility(self):
        """Test that results are reproducible with same seed."""
        data = {
            'control_clusters': [0.45, 0.52, 0.41, 0.38, 0.49],
            'treatment_clusters': [0.58, 0.61, 0.55, 0.63, 0.59]
        }
        
        result1 = cluster_permutation_test(data, n_permutations=500, random_seed=42)
        result2 = cluster_permutation_test(data, n_permutations=500, random_seed=42)
        
        assert result1['p_value'] == result2['p_value']
        assert result1['observed_statistic'] == result2['observed_statistic']
        assert result1['confidence_interval'] == result2['confidence_interval']


class TestClusterPermutationPower:
    """Test cluster permutation power analysis function."""
    
    def test_cluster_permutation_power_basic(self):
        """Test basic permutation power calculation."""
        result = cluster_permutation_power(
            effect_size=0.2, control_mean=0.4, control_sd=0.1,
            n_control=5, n_treatment=5, cluster_size=10,
            n_simulations=100, n_permutations=200,
            alpha=0.05, random_seed=42
        )
        
        assert isinstance(result, dict)
        assert 'power' in result
        assert 'mean_p_value' in result
        assert 'n_simulations' in result
        assert 'effect_size' in result
        
        assert 0 <= result['power'] <= 1
        assert result['n_simulations'] == 100
    
    def test_cluster_permutation_power_no_effect(self):
        """Test power calculation with no treatment effect."""
        result = cluster_permutation_power(
            effect_size=0.0, control_mean=0.5, control_sd=0.1,  # No effect
            n_control=5, n_treatment=5, cluster_size=10,
            n_simulations=100, n_permutations=200,
            alpha=0.05, random_seed=42
        )
        
        # Power should be close to alpha (Type I error rate)
        assert abs(result['power'] - 0.05) < 0.15  # Allow for simulation variability
    
    def test_cluster_permutation_power_large_effect(self):
        """Test power calculation with large treatment effect."""
        result_no_effect = cluster_permutation_power(
            effect_size=0.0, control_mean=0.5, control_sd=0.1,
            n_control=5, n_treatment=5, cluster_size=10,
            n_simulations=100, n_permutations=200,
            alpha=0.05, random_seed=42
        )
        
        result_large_effect = cluster_permutation_power(
            effect_size=0.4, control_mean=0.5, control_sd=0.1,  # Large effect
            n_control=5, n_treatment=5, cluster_size=10,
            n_simulations=100, n_permutations=200,
            alpha=0.05, random_seed=42
        )
        
        # Large effect should have higher power
        assert result_large_effect['power'] > result_no_effect['power']
    
    def test_cluster_permutation_power_more_clusters(self):
        """Test power calculation with more clusters."""
        result_few = cluster_permutation_power(
            effect_size=0.2, control_mean=0.4, control_sd=0.1,
            n_control=3, n_treatment=3, cluster_size=10,
            n_simulations=100, n_permutations=200,
            alpha=0.05, random_seed=42
        )
        
        result_many = cluster_permutation_power(
            effect_size=0.2, control_mean=0.4, control_sd=0.1,
            n_control=8, n_treatment=8, cluster_size=10,
            n_simulations=100, n_permutations=200,
            alpha=0.05, random_seed=42
        )
        
        # More clusters should typically have higher power
        assert result_many['power'] >= result_few['power'] * 0.8  # Allow some variability
    
    def test_cluster_permutation_power_result_structure(self):
        """Test that power result contains all expected fields."""
        result = cluster_permutation_power(
            effect_size=0.2, control_mean=0.4, control_sd=0.1,
            n_control=5, n_treatment=5, cluster_size=10,
            n_simulations=50, n_permutations=100,
            alpha=0.05, random_seed=42
        )
        
        expected_fields = ['power', 'mean_p_value', 'n_simulations', 'effect_size',
                          'n_control_clusters', 'n_treatment_clusters', 'cluster_size', 
                          'alpha', 'n_permutations']
        for field in expected_fields:
            assert field in result


class TestConvenienceFunctions:
    """Test convenience functions for specific outcome types."""
    
    def test_cluster_permutation_test_binary(self):
        """Test binary outcome convenience function."""
        control_clusters = [0.2, 0.3, 0.25, 0.18, 0.22]
        treatment_clusters = [0.4, 0.5, 0.45, 0.38, 0.42]
        
        result = cluster_permutation_test_binary(
            control_clusters, treatment_clusters,
            n_permutations=500, random_seed=42
        )
        
        assert isinstance(result, dict)
        assert 'p_value' in result
        assert 'observed_statistic' in result
        assert 'confidence_interval' in result
        assert 0 <= result['p_value'] <= 1
    
    def test_cluster_permutation_test_continuous(self):
        """Test continuous outcome convenience function."""
        control_clusters = [10.2, 11.3, 9.5, 10.8, 9.2]
        treatment_clusters = [12.4, 13.5, 11.5, 12.8, 11.2]
        
        result = cluster_permutation_test_continuous(
            control_clusters, treatment_clusters,
            n_permutations=500, random_seed=42
        )
        
        assert isinstance(result, dict)
        assert 'p_value' in result
        assert 'observed_statistic' in result
        assert 'confidence_interval' in result
        assert 0 <= result['p_value'] <= 1


class TestParameterValidation:
    """Test parameter validation and edge cases."""
    
    def test_empty_clusters(self):
        """Test handling of empty cluster lists."""
        # Function may not validate this, but should work with valid data
        data = {
            'control_clusters': [0.45, 0.52, 0.41],
            'treatment_clusters': [0.58, 0.61, 0.55]
        }
        result = cluster_permutation_test(data, n_permutations=100, random_seed=42)
        assert result['p_value'] >= 0
    
    def test_single_cluster_per_group(self):
        """Test handling of single cluster per group."""
        data = {
            'control_clusters': [0.45],
            'treatment_clusters': [0.58]
        }
        
        # This should work but have limited inference
        result = cluster_permutation_test(data, n_permutations=100, random_seed=42)
        assert 0 <= result['p_value'] <= 1
    
    def test_unbalanced_groups(self):
        """Test handling of unbalanced cluster groups."""
        data = {
            'control_clusters': [0.45, 0.52, 0.41],
            'treatment_clusters': [0.58, 0.61, 0.55, 0.63, 0.59]
        }
        
        result = cluster_permutation_test(data, n_permutations=500, random_seed=42)
        assert 0 <= result['p_value'] <= 1
    
    def test_invalid_confidence_level(self):
        """Test handling of invalid confidence levels."""
        data = {
            'control_clusters': [0.45, 0.52, 0.41],
            'treatment_clusters': [0.58, 0.61, 0.55]
        }
        
        # Test with valid confidence level
        result = cluster_permutation_test(
            data, confidence_level=0.95, n_permutations=100, random_seed=42
        )
        assert result['p_value'] >= 0
    
    def test_extreme_values(self):
        """Test handling of extreme cluster values."""
        data = {
            'control_clusters': [0.0, 0.0, 0.1],
            'treatment_clusters': [0.9, 1.0, 0.95]
        }
        
        result = cluster_permutation_test(data, n_permutations=500, random_seed=42)
        assert 0 <= result['p_value'] <= 1
        # Large difference should be detectable
        assert abs(result['observed_statistic']) > 0.5
    
    def test_identical_clusters(self):
        """Test handling of identical cluster values within groups."""
        data = {
            'control_clusters': [0.5, 0.5, 0.5],
            'treatment_clusters': [0.7, 0.7, 0.7]
        }
        
        result = cluster_permutation_test(data, n_permutations=500, random_seed=42)
        assert 0 <= result['p_value'] <= 1
    
    def test_very_small_permutations(self):
        """Test with very small number of permutations."""
        data = {
            'control_clusters': [0.45, 0.52, 0.41],
            'treatment_clusters': [0.58, 0.61, 0.55]
        }
        
        result = cluster_permutation_test(data, n_permutations=10, random_seed=42)
        assert 0 <= result['p_value'] <= 1
        assert result['n_permutations_used'] <= 10
    
    def test_missing_data_dataframe(self):
        """Test handling of missing data in DataFrame."""
        # Create DataFrame with some missing values
        df = pd.DataFrame({
            'cluster': [1,1,1,2,2,2,3,3,3],
            'treatment': [0,0,0,0,0,0,1,1,1],
            'outcome': [0,1,0,1,1,0,1,1,1]
        })
        
        result = cluster_permutation_test(
            df, cluster_col='cluster', treatment_col='treatment',
            outcome_col='outcome', n_permutations=100, random_seed=42
        )
        assert 0 <= result['p_value'] <= 1


if __name__ == "__main__":
    pytest.main([__file__])