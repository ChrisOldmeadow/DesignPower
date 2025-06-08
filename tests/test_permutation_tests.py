"""
Test suite for cluster RCT permutation tests.

Tests the permutation test functionality for both continuous and binary outcomes
with various cluster sizes and scenarios.
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
    """Test basic permutation test functionality."""
    
    def test_simple_continuous_data(self):
        """Test permutation test with simple continuous cluster data."""
        # Small effect scenario
        control_clusters = [2.1, 2.3, 1.9, 2.0, 2.2]
        treatment_clusters = [2.4, 2.6, 2.2, 2.3, 2.5]
        
        data = {
            'control_clusters': control_clusters,
            'treatment_clusters': treatment_clusters
        }
        
        result = cluster_permutation_test(
            data=data,
            test_statistic='mean_difference',
            n_permutations=1000,
            random_seed=42
        )
        
        # Basic checks
        assert 'p_value' in result
        assert 'observed_statistic' in result
        assert 'confidence_interval' in result
        assert 0 <= result['p_value'] <= 1
        assert result['n_permutations_used'] == 1000
        assert result['total_clusters'] == 10
        
        # Should detect positive effect (treatment > control)
        assert result['observed_statistic'] > 0
    
    def test_exact_permutation_small_clusters(self):
        """Test exact permutation with very small number of clusters."""
        # Very small cluster trial
        control_clusters = [0.4, 0.5, 0.3]
        treatment_clusters = [0.6, 0.7, 0.8]
        
        result = cluster_permutation_test_binary(
            control_clusters=control_clusters,
            treatment_clusters=treatment_clusters,
            n_permutations='exact'
        )
        
        # With 6 clusters total, choose 3 for control = C(6,3) = 20 permutations
        assert result['n_permutations_used'] == 20
        assert 'Exact permutation test' in result['method']
        assert result['p_value'] <= 0.1  # Should be significant with large effect
    
    def test_dataframe_input(self):
        """Test permutation test with DataFrame input."""
        # Create individual-level data
        data = []
        
        # Control clusters (0, 1, 2)
        for cluster_id in range(3):
            for _ in range(20):  # 20 individuals per cluster
                outcome = np.random.normal(10, 2)  # Control mean = 10
                data.append({
                    'cluster': cluster_id,
                    'treatment': 0,
                    'outcome': outcome
                })
        
        # Treatment clusters (3, 4, 5)  
        for cluster_id in range(3, 6):
            for _ in range(20):  # 20 individuals per cluster
                outcome = np.random.normal(12, 2)  # Treatment mean = 12
                data.append({
                    'cluster': cluster_id,
                    'treatment': 1,
                    'outcome': outcome
                })
        
        df = pd.DataFrame(data)
        
        result = cluster_permutation_test(
            data=df,
            cluster_col='cluster',
            treatment_col='treatment',
            outcome_col='outcome',
            random_seed=42
        )
        
        assert result['total_clusters'] == 6
        assert result['n_control_clusters'] == 3
        assert result['n_treatment_clusters'] == 3
        assert result['p_value'] < 0.1  # Should detect the 2-point difference
    
    def test_different_test_statistics(self):
        """Test different test statistics give reasonable results."""
        np.random.seed(42)
        
        # Generate data with clear effect
        control_clusters = np.random.normal(5, 1, 8)
        treatment_clusters = np.random.normal(7, 1, 8)  # 2-point effect
        
        data = {
            'control_clusters': control_clusters,
            'treatment_clusters': treatment_clusters
        }
        
        # Test mean difference
        result_mean = cluster_permutation_test(
            data=data,
            test_statistic='mean_difference',
            n_permutations=1000,
            random_seed=42
        )
        
        # Test t-statistic
        result_t = cluster_permutation_test(
            data=data,
            test_statistic='t_statistic',
            n_permutations=1000,
            random_seed=42
        )
        
        # Test rank sum
        result_rank = cluster_permutation_test(
            data=data,
            test_statistic='rank_sum',
            n_permutations=1000,
            random_seed=42
        )
        
        # All should detect significant effect
        assert result_mean['p_value'] < 0.05
        assert result_t['p_value'] < 0.05
        assert result_rank['p_value'] < 0.05
        
        # Mean difference should be positive
        assert result_mean['observed_statistic'] > 0
        assert result_t['observed_statistic'] > 0
    
    def test_confidence_intervals(self):
        """Test permutation confidence intervals."""
        control_clusters = [2.0, 2.1, 1.9, 2.0, 2.2]
        treatment_clusters = [2.5, 2.6, 2.4, 2.5, 2.7]  # 0.5 point effect
        
        result = cluster_permutation_test_continuous(
            control_clusters=control_clusters,
            treatment_clusters=treatment_clusters,
            confidence_level=0.95,
            random_seed=42
        )
        
        ci_lower, ci_upper = result['confidence_interval']
        observed_effect = result['observed_statistic']
        
        # CI should be reasonable around the observed effect
        assert ci_lower < observed_effect < ci_upper
        assert ci_upper - ci_lower > 0  # CI should have positive width
        
        # For this clear effect, CI should not include 0
        assert ci_lower > 0 or ci_upper < 0  # One-sided confidence
    
    def test_alternative_hypotheses(self):
        """Test one-sided and two-sided alternatives."""
        control_clusters = [2.0, 2.1, 1.9]
        treatment_clusters = [2.8, 2.9, 2.7]  # Clear positive effect
        
        data = {
            'control_clusters': control_clusters,
            'treatment_clusters': treatment_clusters
        }
        
        # Two-sided test
        result_two = cluster_permutation_test(
            data=data,
            alternative='two-sided',
            n_permutations='exact'
        )
        
        # One-sided test (greater)
        result_greater = cluster_permutation_test(
            data=data,
            alternative='greater',
            n_permutations='exact'
        )
        
        # One-sided test (less)
        result_less = cluster_permutation_test(
            data=data,
            alternative='less',
            n_permutations='exact'
        )
        
        # Greater should be most significant, less should be least significant
        assert result_greater['p_value'] < result_two['p_value']
        assert result_less['p_value'] > result_two['p_value']
        assert result_less['p_value'] > 0.5  # Should not be significant


class TestClusterPermutationPower:
    """Test permutation test power estimation."""
    
    def test_power_calculation(self):
        """Test power calculation via simulation."""
        # Small effect with adequate power
        result = cluster_permutation_power(
            effect_size=0.5,  # Half unit effect
            control_mean=2.0,
            control_sd=0.5,
            n_control=8,
            n_treatment=8,
            cluster_size=25,
            n_simulations=100,  # Small number for testing speed
            n_permutations=500,
            random_seed=42
        )
        
        assert 'power' in result
        assert 0 <= result['power'] <= 1
        assert result['effect_size'] == 0.5
        assert result['total_clusters'] == 16
        assert result['n_simulations'] == 100
        
        # Power should be reasonable for this scenario
        assert result['power'] > 0.3  # At least some power
    
    def test_zero_effect_low_power(self):
        """Test that zero effect gives low power."""
        result = cluster_permutation_power(
            effect_size=0.0,  # No effect
            control_mean=2.0,
            control_sd=0.5,
            n_control=5,
            n_treatment=5,
            cluster_size=20,
            n_simulations=50,
            n_permutations=200,
            alpha=0.05,
            random_seed=42
        )
        
        # With no effect, power should be close to alpha level
        assert result['power'] <= 0.15  # Should be low
        assert result['mean_p_value'] > 0.3  # P-values should be large on average


class TestIntegration:
    """Test integration with simulation modules."""
    
    def test_continuous_simulation_integration(self):
        """Test permutation test integration with continuous simulation."""
        from core.designs.cluster_rct.simulation_continuous import _analyze_continuous_trial
        
        # Create sample data
        data = []
        
        # Control clusters
        for cluster_id in range(5):
            for _ in range(10):
                data.append({
                    'y': np.random.normal(10, 2),
                    'treatment': 0,
                    'cluster': cluster_id
                })
        
        # Treatment clusters
        for cluster_id in range(5, 10):
            for _ in range(10):
                data.append({
                    'y': np.random.normal(12, 2),  # 2-point effect
                    'treatment': 1,
                    'cluster': cluster_id
                })
        
        df = pd.DataFrame(data)
        df['cluster'] = df['cluster'].astype('category')
        
        # Test permutation analysis
        t_stat, p_value, details = _analyze_continuous_trial(
            df, 
            icc=0.05,
            analysis_model="permutation",
            return_details=True
        )
        
        assert details['model'] == 'permutation'
        assert 'permutation_method' in details
        assert 'confidence_interval' in details
        assert 0 <= p_value <= 1
        assert p_value < 0.1  # Should detect the effect
    
    def test_error_handling(self):
        """Test error handling with invalid inputs."""
        # Test with insufficient clusters
        with pytest.raises(ValueError):
            cluster_permutation_test({
                'control_clusters': [1, 2],
                'treatment_clusters': []  # Empty treatment group
            })
        
        # Test with missing data keys
        with pytest.raises(ValueError):
            cluster_permutation_test({
                'wrong_key': [1, 2, 3]
            })
        
        # Test with invalid test statistic
        with pytest.raises(ValueError):
            cluster_permutation_test(
                data={'control_clusters': [1, 2], 'treatment_clusters': [3, 4]},
                test_statistic='invalid_statistic'
            )


if __name__ == "__main__":
    # Run basic functionality test
    test_obj = TestClusterPermutationTest()
    test_obj.test_simple_continuous_data()
    print("✅ Basic permutation test functionality works")
    
    test_obj.test_exact_permutation_small_clusters()
    print("✅ Exact permutation for small clusters works")
    
    power_test = TestClusterPermutationPower()
    power_test.test_power_calculation()
    print("✅ Power calculation functionality works")
    
    print("\n🎉 All permutation test functionality validated!")