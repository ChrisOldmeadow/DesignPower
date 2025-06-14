"""Integration test for cluster RCT infinity results handling.

This test ensures that infinity results are handled gracefully with
appropriate warnings and user-friendly messages.
"""

import math
import pytest
from app.components.cluster_rct.calculations import calculate_cluster_continuous
from app.components.unified_results_display import MetricConfig


def test_cluster_size_impossible_constraints():
    """Test cluster size calculation with impossible constraints returns infinity with warnings."""
    params = {
        'calc_type': 'Sample Size',
        'hypothesis_type': 'Non-Inferiority',
        'determine_ss_param': 'Average Cluster Size (m)',
        'n_clusters_input_for_m_calc': 3,  # Too few clusters
        'mean1': 5.0,
        'non_inferiority_margin': 0.5,
        'assumed_difference': 0.0,
        'non_inferiority_direction': 'lower',
        'std_dev': 1.0,
        'power': 0.8,
        'icc': 0.05,
        'alpha': 0.05,
        'method': 'analytical'
    }
    
    result = calculate_cluster_continuous(params)
    
    # Should have infinity result
    assert 'cluster_size' in result
    assert math.isinf(result['cluster_size'])
    
    # Should have enhanced warning information
    assert 'warning' in result
    assert 'user_message' in result
    assert result.get('warning_level') == 'high'
    assert result.get('calculation_status') == 'warning'
    
    # User message should be helpful
    assert 'infinite' in result['user_message'].lower()
    assert 'cluster size' in result['user_message'].lower()


def test_number_of_clusters_zero_effect_size():
    """Test number of clusters calculation with zero effect size returns infinity."""
    params = {
        'calc_type': 'Sample Size',
        'hypothesis_type': 'Superiority',
        'determine_ss_param': 'Number of Clusters (k)',
        'cluster_size_input_for_k_calc': 20,
        'mean1': 5.0,
        'mean2': 5.0,  # Zero effect size
        'std_dev': 1.0,
        'power': 0.8,
        'icc': 0.05,
        'alpha': 0.05,
        'method': 'analytical'
    }
    
    result = calculate_cluster_continuous(params)
    
    # Should have infinity result
    assert 'n_clusters' in result
    assert math.isinf(result['n_clusters'])
    
    # Should have enhanced warning information
    assert 'warning' in result
    assert 'user_message' in result
    assert result.get('warning_level') == 'high'
    assert result.get('calculation_status') == 'warning'
    
    # Warning should mention identical means
    assert 'identical' in result['warning'].lower()


def test_infinity_value_formatting():
    """Test that infinity values are formatted nicely in the UI."""
    metric = MetricConfig(key='test', label='Test Metric')
    
    # Test infinity formatting
    inf_value = float('inf')
    formatted = metric.format_value(inf_value)
    assert formatted == '∞ (Not achievable)'
    
    # Test normal value still works
    normal_value = 25.5
    formatted_normal = metric.format_value(normal_value)
    assert formatted_normal == '25.500'
    
    # Test None still works
    none_value = None
    formatted_none = metric.format_value(none_value)
    assert formatted_none == 'N/A'


def test_finite_results_no_enhanced_warnings():
    """Test that finite results don't trigger enhanced warning system."""
    params = {
        'calc_type': 'Sample Size',
        'hypothesis_type': 'Non-Inferiority',
        'determine_ss_param': 'Number of Clusters (k)',
        'cluster_size_input_for_k_calc': 20,
        'mean1': 5.0,
        'non_inferiority_margin': 0.5,
        'assumed_difference': 0.0,
        'non_inferiority_direction': 'lower',
        'std_dev': 1.0,
        'power': 0.8,
        'icc': 0.05,
        'alpha': 0.05,
        'method': 'analytical'
    }
    
    result = calculate_cluster_continuous(params)
    
    # Should have finite result
    assert 'n_clusters' in result
    assert not math.isinf(result['n_clusters'])
    assert result['n_clusters'] > 0
    
    # Should NOT have enhanced warning information
    assert result.get('warning_level') != 'high'
    assert result.get('calculation_status') != 'warning'
    assert 'user_message' not in result


def test_high_icc_cluster_size_constraint():
    """Test cluster size estimation with high ICC that makes it impossible."""
    params = {
        'calc_type': 'Sample Size',
        'hypothesis_type': 'Superiority',
        'determine_ss_param': 'Average Cluster Size (m)',
        'n_clusters_input_for_m_calc': 5,  # Small number of clusters
        'mean1': 5.0,
        'mean2': 5.2,
        'std_dev': 1.0,
        'power': 0.8,
        'icc': 0.5,  # High ICC
        'alpha': 0.05,
        'method': 'analytical'
    }
    
    result = calculate_cluster_continuous(params)
    
    # Should have infinity result with warnings
    assert 'cluster_size' in result
    assert math.isinf(result['cluster_size'])
    assert 'warning' in result
    
    # Warning should provide specific guidance
    assert 'Cannot achieve target power' in result['warning']
    assert 'maximum feasible ICC' in result['warning']


if __name__ == "__main__":
    test_cluster_size_impossible_constraints()
    test_number_of_clusters_zero_effect_size()
    test_infinity_value_formatting()
    test_finite_results_no_enhanced_warnings()
    test_high_icc_cluster_size_constraint()
    print("All infinity handling tests passed!")