"""Integration test for cluster RCT non-inferiority parameter handling bug fix.

This test ensures that the 'Missing required parameter: mean2' error for 
continuous cluster power non-inferiority calculations is fixed.
"""

import math
import pytest
from app.components.cluster_rct.calculations import calculate_cluster_continuous


def test_cluster_continuous_non_inferiority_power():
    """Test that non-inferiority power calculation works without mean2 parameter."""
    params = {
        'calc_type': 'Power',
        'hypothesis_type': 'Non-Inferiority',
        'n_clusters': 10,
        'cluster_size': 20,
        'icc': 0.05,
        'mean1': 5.0,
        'non_inferiority_margin': 0.5,
        'assumed_difference': 0.0,
        'non_inferiority_direction': 'lower',
        'std_dev': 1.0,
        'alpha': 0.05,
        'method': 'analytical'
    }
    
    result = calculate_cluster_continuous(params)
    
    # Should not have error
    assert 'error' not in result
    
    # Should have power result
    assert 'power' in result
    assert isinstance(result['power'], (int, float))
    assert 0 <= result['power'] <= 1
    
    # Should have calculated mean2 correctly
    assert 'mean2' in result
    expected_mean2 = params['mean1'] + params['assumed_difference']
    assert result['mean2'] == expected_mean2


def test_cluster_continuous_non_inferiority_sample_size():
    """Test that non-inferiority sample size calculation works without mean2 parameter."""
    params = {
        'calc_type': 'Sample Size',
        'hypothesis_type': 'Non-Inferiority',
        'determine_ss_param': 'Number of Clusters (k)',
        'cluster_size_input_for_k_calc': 20,
        'mean1': 5.0,
        'non_inferiority_margin': 0.5,
        'assumed_difference': 0.2,
        'non_inferiority_direction': 'lower',
        'std_dev': 1.0,
        'power': 0.8,
        'icc': 0.05,
        'alpha': 0.05,
        'method': 'analytical'
    }
    
    result = calculate_cluster_continuous(params)
    
    # Should not have error
    assert 'error' not in result
    
    # Should have sample size result
    assert 'n_clusters' in result
    assert isinstance(result['n_clusters'], (int, float))
    assert result['n_clusters'] > 0
    
    # Should have calculated mean2 correctly for sample size calculation
    # For sample size: effective_difference = assumed_difference + margin = 0.2 + 0.5 = 0.7
    assert 'mean2' in result
    expected_mean2 = params['mean1'] + params['assumed_difference'] + params['non_inferiority_margin']
    assert result['mean2'] == expected_mean2


def test_cluster_continuous_non_inferiority_mde():
    """Test that non-inferiority MDE calculation works without mean2 parameter."""
    params = {
        'calc_type': 'Minimum Detectable Effect',
        'hypothesis_type': 'Non-Inferiority',
        'n_clusters': 10,
        'cluster_size': 20,
        'icc': 0.05,
        'mean1': 5.0,
        'non_inferiority_margin': 0.5,
        'assumed_difference': 0.0,
        'non_inferiority_direction': 'lower',
        'std_dev': 1.0,
        'power': 0.8,
        'alpha': 0.05,
        'method': 'analytical'
    }
    
    result = calculate_cluster_continuous(params)
    
    # Should not have error
    assert 'error' not in result
    
    # Should have MDE result
    assert 'mde' in result
    assert isinstance(result['mde'], (int, float))
    assert result['mde'] > 0


def test_cluster_continuous_non_inferiority_sample_size_zero_assumed_difference():
    """Test that non-inferiority sample size works even with zero assumed difference."""
    params = {
        'calc_type': 'Sample Size',
        'hypothesis_type': 'Non-Inferiority',
        'determine_ss_param': 'Number of Clusters (k)',
        'cluster_size_input_for_k_calc': 20,
        'mean1': 5.0,
        'non_inferiority_margin': 0.5,
        'assumed_difference': 0.0,  # This was causing infinity before
        'non_inferiority_direction': 'lower',
        'std_dev': 1.0,
        'power': 0.8,
        'icc': 0.05,
        'alpha': 0.05,
        'method': 'analytical'
    }
    
    result = calculate_cluster_continuous(params)
    
    # Should not have error or infinity
    assert 'error' not in result
    assert 'n_clusters' in result
    assert isinstance(result['n_clusters'], (int, float))
    assert result['n_clusters'] > 0
    assert not math.isinf(result['n_clusters'])
    
    # Should have calculated mean2 using effective difference (margin only)
    # effective_difference = assumed_difference + margin = 0.0 + 0.5 = 0.5
    expected_mean2 = params['mean1'] + params['non_inferiority_margin']
    assert result['mean2'] == expected_mean2


def test_cluster_continuous_superiority_still_requires_mean2():
    """Test that superiority calculations still require mean2 parameter."""
    params = {
        'calc_type': 'Power',
        'hypothesis_type': 'Superiority',
        'n_clusters': 10,
        'cluster_size': 20,
        'icc': 0.05,
        'mean1': 5.0,
        # Deliberately omit mean2
        'std_dev': 1.0,
        'alpha': 0.05,
        'method': 'analytical'
    }
    
    result = calculate_cluster_continuous(params)
    
    # Should have error for missing mean2
    assert 'error' in result
    assert 'Missing required parameter: mean2' in result['error']


if __name__ == "__main__":
    test_cluster_continuous_non_inferiority_power()
    test_cluster_continuous_non_inferiority_sample_size()
    test_cluster_continuous_non_inferiority_mde()
    test_cluster_continuous_non_inferiority_sample_size_zero_assumed_difference()
    test_cluster_continuous_superiority_still_requires_mean2()
    print("All tests passed!")