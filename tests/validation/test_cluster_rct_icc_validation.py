"""
Test cluster RCT implementation against ICC benchmarks.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest
from core.designs.cluster_rct.analytical_continuous import (
    sample_size_continuous as cluster_sample_size_continuous,
    power_continuous as cluster_power_continuous
)
from core.designs.cluster_rct.analytical_binary import (
    sample_size_binary as cluster_sample_size_binary,
    power_binary as cluster_power_binary
)
from cluster_rct_icc_benchmarks import (
    CONTINUOUS_ICC_BENCHMARKS,
    BINARY_ICC_BENCHMARKS,
    CLUSTER_SIZE_BENCHMARKS
)


class TestClusterRCTICCValidation:
    """Validate cluster RCT calculations with various ICC values."""
    
    def test_continuous_low_icc(self):
        """Test continuous outcome with low ICC (0.01)."""
        benchmark = CONTINUOUS_ICC_BENCHMARKS[0]  # Low ICC (0.01)
        
        result = cluster_sample_size_continuous(
            mean1=0,
            mean2=benchmark.mean_diff,
            std_dev=benchmark.std_dev,
            cluster_size=benchmark.cluster_size,
            icc=benchmark.icc,
            power=benchmark.power,
            alpha=benchmark.alpha
        )
        
        clusters_per_arm = result['n_clusters']  # This is per arm
        design_effect = result['design_effect']
        
        # Check clusters within tolerance
        clusters_error = abs(clusters_per_arm - benchmark.expected_clusters_per_arm) / benchmark.expected_clusters_per_arm
        assert clusters_error <= benchmark.tolerance, \
            f"Clusters mismatch: expected {benchmark.expected_clusters_per_arm}, got {clusters_per_arm}"
        
        # Check design effect (should be exact)
        assert abs(design_effect - benchmark.expected_design_effect) < 0.01, \
            f"Design effect mismatch: expected {benchmark.expected_design_effect}, got {design_effect}"
    
    def test_continuous_moderate_icc(self):
        """Test continuous outcome with moderate ICC (0.05)."""
        benchmark = CONTINUOUS_ICC_BENCHMARKS[1]  # Moderate ICC (0.05)
        
        result = cluster_sample_size_continuous(
            mean1=0,
            mean2=benchmark.mean_diff,
            std_dev=benchmark.std_dev,
            cluster_size=benchmark.cluster_size,
            icc=benchmark.icc,
            power=benchmark.power,
            alpha=benchmark.alpha
        )
        
        clusters_per_arm = result['n_clusters']  # This is per arm
        
        clusters_error = abs(clusters_per_arm - benchmark.expected_clusters_per_arm) / benchmark.expected_clusters_per_arm
        assert clusters_error <= benchmark.tolerance, \
            f"Clusters mismatch: expected {benchmark.expected_clusters_per_arm}, got {clusters_per_arm}"
    
    def test_continuous_high_icc(self):
        """Test continuous outcome with high ICC (0.10)."""
        benchmark = CONTINUOUS_ICC_BENCHMARKS[2]  # High ICC (0.10)
        
        result = cluster_sample_size_continuous(
            mean1=0,
            mean2=benchmark.mean_diff,
            std_dev=benchmark.std_dev,
            cluster_size=benchmark.cluster_size,
            icc=benchmark.icc,
            power=benchmark.power,
            alpha=benchmark.alpha
        )
        
        clusters_per_arm = result['n_clusters']  # This is per arm
        
        clusters_error = abs(clusters_per_arm - benchmark.expected_clusters_per_arm) / benchmark.expected_clusters_per_arm
        assert clusters_error <= benchmark.tolerance, \
            f"Clusters mismatch: expected {benchmark.expected_clusters_per_arm}, got {clusters_per_arm}"
    
    def test_continuous_very_high_icc(self):
        """Test continuous outcome with very high ICC (0.20)."""
        benchmark = CONTINUOUS_ICC_BENCHMARKS[3]  # Very high ICC (0.20)
        
        result = cluster_sample_size_continuous(
            mean1=0,
            mean2=benchmark.mean_diff,
            std_dev=benchmark.std_dev,
            cluster_size=benchmark.cluster_size,
            icc=benchmark.icc,
            power=benchmark.power,
            alpha=benchmark.alpha
        )
        
        clusters_per_arm = result['n_clusters']  # This is per arm
        
        clusters_error = abs(clusters_per_arm - benchmark.expected_clusters_per_arm) / benchmark.expected_clusters_per_arm
        assert clusters_error <= benchmark.tolerance, \
            f"Clusters mismatch: expected {benchmark.expected_clusters_per_arm}, got {clusters_per_arm}"
    
    def test_binary_low_icc(self):
        """Test binary outcome with low ICC (0.02)."""
        benchmark = BINARY_ICC_BENCHMARKS[0]  # Low ICC (0.02)
        
        result = cluster_sample_size_binary(
            p1=benchmark.p1,
            p2=benchmark.p2,
            cluster_size=benchmark.cluster_size,
            icc=benchmark.icc,
            power=benchmark.power,
            alpha=benchmark.alpha
        )
        
        clusters_per_arm = result['n_clusters']  # This is per arm
        design_effect = result['design_effect']
        
        clusters_error = abs(clusters_per_arm - benchmark.expected_clusters_per_arm) / benchmark.expected_clusters_per_arm
        assert clusters_error <= benchmark.tolerance, \
            f"Clusters mismatch: expected {benchmark.expected_clusters_per_arm}, got {clusters_per_arm}"
        
        assert abs(design_effect - benchmark.expected_design_effect) < 0.01, \
            f"Design effect mismatch: expected {benchmark.expected_design_effect}, got {design_effect}"
    
    def test_binary_moderate_icc(self):
        """Test binary outcome with moderate ICC (0.05)."""
        benchmark = BINARY_ICC_BENCHMARKS[1]  # Moderate ICC (0.05)
        
        result = cluster_sample_size_binary(
            p1=benchmark.p1,
            p2=benchmark.p2,
            cluster_size=benchmark.cluster_size,
            icc=benchmark.icc,
            power=benchmark.power,
            alpha=benchmark.alpha
        )
        
        clusters_per_arm = result['n_clusters']  # This is per arm
        
        clusters_error = abs(clusters_per_arm - benchmark.expected_clusters_per_arm) / benchmark.expected_clusters_per_arm
        assert clusters_error <= benchmark.tolerance, \
            f"Clusters mismatch: expected {benchmark.expected_clusters_per_arm}, got {clusters_per_arm}"
    
    def test_binary_high_icc(self):
        """Test binary outcome with high ICC (0.10)."""
        benchmark = BINARY_ICC_BENCHMARKS[2]  # High ICC (0.10)
        
        result = cluster_sample_size_binary(
            p1=benchmark.p1,
            p2=benchmark.p2,
            cluster_size=benchmark.cluster_size,
            icc=benchmark.icc,
            power=benchmark.power,
            alpha=benchmark.alpha
        )
        
        clusters_per_arm = result['n_clusters']  # This is per arm
        
        clusters_error = abs(clusters_per_arm - benchmark.expected_clusters_per_arm) / benchmark.expected_clusters_per_arm
        assert clusters_error <= benchmark.tolerance, \
            f"Clusters mismatch: expected {benchmark.expected_clusters_per_arm}, got {clusters_per_arm}"
    
    def test_design_effect_calculation(self):
        """Test that design effect is calculated correctly for all ICC values."""
        test_cases = [
            (0.01, 50, 1.49),   # 1 + (50-1)*0.01 = 1.49
            (0.05, 30, 2.45),   # 1 + (30-1)*0.05 = 2.45
            (0.10, 20, 2.90),   # 1 + (20-1)*0.10 = 2.90
            (0.20, 15, 3.80),   # 1 + (15-1)*0.20 = 3.80
        ]
        
        for icc, cluster_size, expected_deff in test_cases:
            # Test with continuous outcome
            result = cluster_sample_size_continuous(
                mean1=0, mean2=0.5, std_dev=1.0,
                cluster_size=cluster_size,
                icc=icc,
                power=0.8, alpha=0.05
            )
            
            calculated_deff = result['design_effect']
            assert abs(calculated_deff - expected_deff) < 0.01, \
                f"Design effect mismatch for ICC={icc}, m={cluster_size}: expected {expected_deff}, got {calculated_deff}"
    
    def test_power_with_varying_icc(self):
        """Test that power decreases as ICC increases (fixed sample size)."""
        n_clusters = 20
        cluster_size = 50
        
        powers = []
        
        for icc in [0.01, 0.05, 0.10, 0.15, 0.20]:
            result = cluster_power_continuous(
                n_clusters=n_clusters,
                cluster_size=cluster_size,
                mean1=0,
                mean2=0.5,
                std_dev=1.0,
                icc=icc,
                alpha=0.05
            )
            powers.append(result['power'])
        
        # Power should decrease as ICC increases
        for i in range(len(powers) - 1):
            assert powers[i] > powers[i+1], \
                f"Power should decrease with increasing ICC: {powers}"
    
    def test_cluster_size_effect(self):
        """Test effect of cluster size on required number of clusters."""
        benchmark_small = CLUSTER_SIZE_BENCHMARKS[0]  # Small clusters (m=10)
        benchmark_large = CLUSTER_SIZE_BENCHMARKS[1]  # Large clusters (m=200)
        
        # Small clusters
        result_small = cluster_sample_size_continuous(
            mean1=0,
            mean2=benchmark_small.mean_diff,
            std_dev=benchmark_small.std_dev,
            cluster_size=benchmark_small.cluster_size,
            icc=benchmark_small.icc,
            power=benchmark_small.power,
            alpha=benchmark_small.alpha
        )
        
        # Large clusters
        result_large = cluster_sample_size_continuous(
            mean1=0,
            mean2=benchmark_large.mean_diff,
            std_dev=benchmark_large.std_dev,
            cluster_size=benchmark_large.cluster_size,
            icc=benchmark_large.icc,
            power=benchmark_large.power,
            alpha=benchmark_large.alpha
        )
        
        # With same ICC, smaller clusters need more clusters but smaller design effect
        assert result_small['design_effect'] < result_large['design_effect'], \
            "Smaller clusters should have smaller design effect"


if __name__ == "__main__":
    test = TestClusterRCTICCValidation()
    
    print("Running Cluster RCT ICC Validation Tests...")
    print("=" * 50)
    
    tests = [
        ("Low ICC (0.01) continuous", test.test_continuous_low_icc),
        ("Moderate ICC (0.05) continuous", test.test_continuous_moderate_icc),
        ("High ICC (0.10) continuous", test.test_continuous_high_icc),
        ("Very high ICC (0.20) continuous", test.test_continuous_very_high_icc),
        ("Low ICC (0.02) binary", test.test_binary_low_icc),
        ("Moderate ICC (0.05) binary", test.test_binary_moderate_icc),
        ("High ICC (0.10) binary", test.test_binary_high_icc),
        ("Design effect calculation", test.test_design_effect_calculation),
        ("Power vs ICC relationship", test.test_power_with_varying_icc),
        ("Cluster size effects", test.test_cluster_size_effect),
    ]
    
    passed = 0
    for test_name, test_func in tests:
        try:
            test_func()
            print(f"✓ {test_name}")
            passed += 1
        except AssertionError as e:
            print(f"✗ {test_name}: {e}")
        except Exception as e:
            print(f"✗ {test_name}: Error - {e}")
    
    print(f"\nOverall: {passed}/{len(tests)} tests passed ({100*passed/len(tests):.1f}%)")