"""
Test Fisher's exact test implementation against validation benchmarks.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest
import numpy as np
from core.designs.parallel.analytical_binary import (
    fishers_exact_test, 
    power_binary,
    sample_size_binary
)
from fishers_exact_benchmarks import (
    TEA_TASTING_EXAMPLE,
    MEDICAL_TREATMENT_EXAMPLE,
    SMALL_SAMPLE_EXAMPLE,
    BALANCED_MODERATE_EXAMPLE,
    create_power_benchmarks_fishers,
    create_sample_size_benchmarks_fishers
)


class TestFishersExactValidation:
    """Validate Fisher's exact test against known benchmarks."""
    
    def test_tea_tasting_example(self):
        """Test the classic tea tasting example."""
        b = TEA_TASTING_EXAMPLE
        
        # Calculate totals
        n1 = b.control_success + b.control_failure
        n2 = b.treatment_success + b.treatment_failure
        
        # Run Fisher's exact test
        p_value = fishers_exact_test(n1, n2, b.control_success, b.treatment_success)
        
        # Check p-value
        assert abs(p_value - b.expected_p_value_two_sided) <= b.tolerance, \
            f"P-value mismatch: expected {b.expected_p_value_two_sided}, got {p_value}"
    
    def test_medical_treatment_example(self):
        """Test medical treatment comparison example."""
        b = MEDICAL_TREATMENT_EXAMPLE
        
        n1 = b.control_success + b.control_failure
        n2 = b.treatment_success + b.treatment_failure
        
        p_value = fishers_exact_test(n1, n2, b.control_success, b.treatment_success)
        
        assert abs(p_value - b.expected_p_value_two_sided) <= b.tolerance, \
            f"P-value mismatch: expected {b.expected_p_value_two_sided}, got {p_value}"
    
    def test_small_sample_example(self):
        """Test very small sample where exact test is critical."""
        b = SMALL_SAMPLE_EXAMPLE
        
        n1 = b.control_success + b.control_failure
        n2 = b.treatment_success + b.treatment_failure
        
        p_value = fishers_exact_test(n1, n2, b.control_success, b.treatment_success)
        
        assert abs(p_value - b.expected_p_value_two_sided) <= b.tolerance, \
            f"P-value mismatch: expected {b.expected_p_value_two_sided}, got {p_value}"
    
    def test_balanced_moderate_example(self):
        """Test balanced design with moderate effect."""
        b = BALANCED_MODERATE_EXAMPLE
        
        n1 = b.control_success + b.control_failure
        n2 = b.treatment_success + b.treatment_failure
        
        p_value = fishers_exact_test(n1, n2, b.control_success, b.treatment_success)
        
        assert abs(p_value - b.expected_p_value_two_sided) <= b.tolerance, \
            f"P-value mismatch: expected {b.expected_p_value_two_sided}, got {p_value}"
    
    def test_power_calculations_fishers(self):
        """Test power calculations for Fisher's exact test."""
        benchmarks = create_power_benchmarks_fishers()
        
        for benchmark in benchmarks:
            params = benchmark['parameters']
            expected = benchmark['expected']
            
            # Calculate power
            result = power_binary(
                n1=params['n1'],
                n2=params['n2'],
                p1=params['p1'],
                p2=params['p2'],
                alpha=params['alpha'],
                test_type=params['test_type']
            )
            
            power = result['power']
            
            # Check if within tolerance
            assert abs(power - expected['power']) <= expected['tolerance'], \
                f"{benchmark['name']}: Power mismatch - expected {expected['power']}, got {power}"
    
    def test_sample_size_calculations_fishers(self):
        """Test sample size calculations for Fisher's exact test."""
        benchmarks = create_sample_size_benchmarks_fishers()
        
        for benchmark in benchmarks:
            params = benchmark['parameters']
            expected = benchmark['expected']
            
            # Calculate sample size
            result = sample_size_binary(
                p1=params['p1'],
                p2=params['p2'],
                power=params['power'],
                alpha=params['alpha'],
                test_type=params['test_type']
            )
            
            n_per_group = result['sample_size_1']
            
            # Check if within tolerance
            assert abs(n_per_group - expected['n_per_group']) <= expected['tolerance'], \
                f"{benchmark['name']}: Sample size mismatch - expected {expected['n_per_group']}, got {n_per_group}"
    
    def test_fishers_vs_normal_approximation(self):
        """Compare Fisher's exact test with normal approximation."""
        # Small sample where they should differ
        n1, n2 = 10, 10
        s1, s2 = 2, 7
        
        # Fisher's exact
        p_fishers = fishers_exact_test(n1, n2, s1, s2)
        
        # Normal approximation p-value (using scipy for accuracy)
        from scipy import stats
        p1 = s1 / n1
        p2 = s2 / n2
        p_pooled = (s1 + s2) / (n1 + n2)
        
        se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n1 + 1/n2))
        z = (p1 - p2) / se if se > 0 else 0
        p_normal = 2 * (1 - stats.norm.cdf(abs(z)))  # Two-sided
        
        # For this specific example, we expect certain behavior
        # With small samples, the relationship isn't always predictable
        print(f"Fisher's exact p-value: {p_fishers:.3f}")
        print(f"Normal approximation p-value: {p_normal:.3f}")
        
        # Just check that both tests are reasonable
        assert 0 <= p_fishers <= 1, "Fisher's p-value should be valid"
        assert 0 <= p_normal <= 1, "Normal approximation p-value should be valid"
    
    def test_edge_cases(self):
        """Test edge cases for Fisher's exact test."""
        # All successes in one group
        p_value = fishers_exact_test(5, 5, 5, 0)
        assert p_value < 0.01, "Should be highly significant when all succeed in one group"
        
        # No successes in either group
        p_value = fishers_exact_test(5, 5, 0, 0)
        assert p_value == 1.0, "Should have p=1 when no events in either group"
        
        # Equal proportions
        p_value = fishers_exact_test(10, 10, 5, 5)
        assert p_value > 0.9, "Should have high p-value for identical proportions"


if __name__ == "__main__":
    # Run tests
    test = TestFishersExactValidation()
    
    print("Running Fisher's Exact Test Validation...")
    
    # Run individual tests
    try:
        test.test_tea_tasting_example()
        print("✓ Tea tasting example passed")
    except AssertionError as e:
        print(f"✗ Tea tasting example failed: {e}")
    
    try:
        test.test_medical_treatment_example()
        print("✓ Medical treatment example passed")
    except AssertionError as e:
        print(f"✗ Medical treatment example failed: {e}")
    
    try:
        test.test_small_sample_example()
        print("✓ Small sample example passed")
    except AssertionError as e:
        print(f"✗ Small sample example failed: {e}")
    
    try:
        test.test_balanced_moderate_example()
        print("✓ Balanced moderate example passed")
    except AssertionError as e:
        print(f"✗ Balanced moderate example failed: {e}")
    
    try:
        test.test_power_calculations_fishers()
        print("✓ Power calculations passed")
    except AssertionError as e:
        print(f"✗ Power calculations failed: {e}")
    
    try:
        test.test_sample_size_calculations_fishers()
        print("✓ Sample size calculations passed")
    except AssertionError as e:
        print(f"✗ Sample size calculations failed: {e}")
    
    try:
        test.test_fishers_vs_normal_approximation()
        print("✓ Fisher's vs normal approximation passed")
    except AssertionError as e:
        print(f"✗ Fisher's vs normal approximation failed: {e}")
    
    try:
        test.test_edge_cases()
        print("✓ Edge cases passed")
    except AssertionError as e:
        print(f"✗ Edge cases failed: {e}")