#!/usr/bin/env python3
"""
Repeated measures design validation tests.

This module provides validation of repeated measures design implementations
against theoretical benchmarks based on Vickers (2001) and Van Breukelen (2006).
"""

import pytest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

# Import directly to avoid circular imports
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from tests.validation.repeated_measures_benchmarks import (
    REPEATED_MEASURES_BENCHMARKS, validate_repeated_measures
)


class TestRepeatedMeasuresValidation:
    """Test class for repeated measures design validation."""

    def test_all_benchmarks(self):
        """Test all repeated measures benchmarks."""
        results = validate_repeated_measures()
        
        passed = sum(1 for r in results if r['pass'])
        total = len(results)
        success_rate = passed / total * 100
        
        # Log results for debugging
        print(f"\nRepeated Measures Validation Results:")
        print(f"Passed: {passed}/{total} ({success_rate:.1f}%)")
        
        for result in results:
            status = "✓" if result['pass'] else "✗"
            print(f"{status} {result['benchmark']}")
            if not result['pass'] and 'error' not in result:
                print(f"   Expected n1: {result['n1'].split('(')[1].split(')')[0]}")
                print(f"   Actual n1: {result['n1'].split()[0]}")
        
        # Assert that all benchmarks pass
        assert success_rate >= 95.0, f"Validation success rate {success_rate:.1f}% below 95% threshold"

    def test_vickers_high_correlation_ancova(self):
        """Test specific Vickers (2001) example: high correlation ANCOVA."""
        benchmark = REPEATED_MEASURES_BENCHMARKS[0]  # High correlation ANCOVA
        
        from core.designs.parallel.analytical.repeated_measures import sample_size_repeated_measures
        
        result = sample_size_repeated_measures(
            delta=benchmark.delta,
            std_dev=benchmark.std_dev,
            correlation=benchmark.correlation,
            power=benchmark.power,
            alpha=benchmark.alpha,
            method=benchmark.method
        )
        
        # Verify exact match for this well-known benchmark
        assert result['n1'] == benchmark.expected_n1, f"Expected n1={benchmark.expected_n1}, got {result['n1']}"
        assert result['total_n'] == benchmark.expected_total_n, f"Expected total_n={benchmark.expected_total_n}, got {result['total_n']}"

    def test_vickers_high_correlation_change_score(self):
        """Test specific Vickers (2001) example: high correlation change score."""
        benchmark = REPEATED_MEASURES_BENCHMARKS[1]  # High correlation change score
        
        from core.designs.parallel.analytical.repeated_measures import sample_size_repeated_measures
        
        result = sample_size_repeated_measures(
            delta=benchmark.delta,
            std_dev=benchmark.std_dev,
            correlation=benchmark.correlation,
            power=benchmark.power,
            alpha=benchmark.alpha,
            method=benchmark.method
        )
        
        # Verify exact match for this well-known benchmark
        assert result['n1'] == benchmark.expected_n1, f"Expected n1={benchmark.expected_n1}, got {result['n1']}"
        assert result['total_n'] == benchmark.expected_total_n, f"Expected total_n={benchmark.expected_total_n}, got {result['total_n']}"
        
        # Verify that change score is less efficient than ANCOVA with high correlation
        ancova_benchmark = REPEATED_MEASURES_BENCHMARKS[0]
        assert result['n1'] > ancova_benchmark.expected_n1, "Change score should require larger sample size than ANCOVA with high correlation"

    def test_correlation_effect_on_efficiency(self):
        """Test that higher correlation increases efficiency for both methods."""
        from core.designs.parallel.analytical.repeated_measures import sample_size_repeated_measures
        
        # Test with same parameters but different correlations
        base_params = {
            'delta': 0.5,
            'std_dev': 1.0,
            'power': 0.8,
            'alpha': 0.05
        }
        
        # Low correlation
        result_low = sample_size_repeated_measures(correlation=0.2, method='ancova', **base_params)
        
        # High correlation  
        result_high = sample_size_repeated_measures(correlation=0.8, method='ancova', **base_params)
        
        # Higher correlation should require smaller sample size
        assert result_high['n1'] < result_low['n1'], "Higher correlation should reduce required sample size"

    def test_ancova_vs_change_score_efficiency(self):
        """Test that ANCOVA is more efficient than change score analysis."""
        from core.designs.parallel.analytical.repeated_measures import sample_size_repeated_measures
        
        params = {
            'delta': 0.5,
            'std_dev': 1.0,
            'correlation': 0.6,  # Moderate correlation
            'power': 0.8,
            'alpha': 0.05
        }
        
        ancova_result = sample_size_repeated_measures(method='ancova', **params)
        change_result = sample_size_repeated_measures(method='change_score', **params)
        
        # ANCOVA should require smaller sample size than change score
        assert ancova_result['n1'] < change_result['n1'], "ANCOVA should be more efficient than change score analysis"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])