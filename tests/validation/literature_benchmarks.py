"""
Literature-based validation benchmarks for DesignPower.

This module contains validation tests against published examples from
authoritative sources in clinical trial methodology.
"""

import pytest
import math
from typing import Dict, Any, List, Tuple
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from core.designs.parallel import (
    sample_size_binary, power_binary, 
    sample_size_continuous, power_continuous,
    sample_size_survival, power_survival
)
from core.designs.single_arm import (
    calculate_single_arm_binary, calculate_single_arm_continuous
)
from core.designs.cluster_rct import (
    calculate_cluster_binary, calculate_cluster_continuous
)


class LiteratureBenchmark:
    """Container for literature benchmark data."""
    
    def __init__(self, 
                 source: str,
                 page: str,
                 example_name: str,
                 design_type: str,
                 outcome_type: str,
                 parameters: Dict[str, Any],
                 expected_result: Dict[str, Any],
                 tolerance: float = 0.05):
        self.source = source
        self.page = page  
        self.example_name = example_name
        self.design_type = design_type
        self.outcome_type = outcome_type
        self.parameters = parameters
        self.expected_result = expected_result
        self.tolerance = tolerance

    def __repr__(self):
        return f"Benchmark({self.source}, {self.example_name})"


# =============================================================================
# PARALLEL RCT BENCHMARKS
# =============================================================================

FLEISS_1973_BENCHMARKS = [
    LiteratureBenchmark(
        source="Fleiss et al. (1973) - Statistical Methods for Rates and Proportions",
        page="Chapter 3, Example 3.1",
        example_name="Binary outcome sample size",
        design_type="parallel",
        outcome_type="binary",
        parameters={
            "p1": 0.65,
            "p2": 0.85,
            "alpha": 0.05,
            "power": 0.80,
            "allocation_ratio": 1.0
        },
        expected_result={
            "total_sample_size": 62,  # 31 per group
            "sample_size_1": 31,
            "sample_size_2": 31
        },
        tolerance=0.1  # Allow ±10% for rounding differences
    ),
    
    LiteratureBenchmark(
        source="Fleiss et al. (1973)",
        page="Chapter 3, Example 3.2", 
        example_name="Unequal allocation binary",
        design_type="parallel",
        outcome_type="binary",
        parameters={
            "p1": 0.20,
            "p2": 0.35,
            "alpha": 0.05,
            "power": 0.90,
            "allocation_ratio": 2.0  # 2:1 allocation
        },
        expected_result={
            "total_sample_size": 246,  # Approximately 82 + 164
            "sample_size_1": 82,
            "sample_size_2": 164
        },
        tolerance=0.15
    )
]

COHEN_1988_BENCHMARKS = [
    LiteratureBenchmark(
        source="Cohen (1988) - Statistical Power Analysis for Behavioral Sciences",
        page="Chapter 2, Table 2.3.1",
        example_name="Medium effect size continuous",
        design_type="parallel", 
        outcome_type="continuous",
        parameters={
            "mean1": 0.0,
            "mean2": 0.5,  # Effect size d = 0.5 (medium)
            "std_dev": 1.0,
            "alpha": 0.05,
            "power": 0.80
        },
        expected_result={
            "total_sample_size": 64,  # 32 per group for d=0.5
            "sample_size_1": 32,
            "sample_size_2": 32
        },
        tolerance=0.1
    ),
    
    LiteratureBenchmark(
        source="Cohen (1988)",
        page="Chapter 2, Table 2.3.1",
        example_name="Large effect size continuous",
        design_type="parallel",
        outcome_type="continuous", 
        parameters={
            "mean1": 0.0,
            "mean2": 0.8,  # Effect size d = 0.8 (large)
            "std_dev": 1.0,
            "alpha": 0.05,
            "power": 0.80
        },
        expected_result={
            "total_sample_size": 26,  # 13 per group for d=0.8
            "sample_size_1": 13,
            "sample_size_2": 13
        },
        tolerance=0.15
    )
]

SCHOENFELD_1981_BENCHMARKS = [
    LiteratureBenchmark(
        source="Schoenfeld (1981) - The asymptotic properties of nonparametric tests",
        page="Section 4, Example 1",
        example_name="Survival log-rank test",
        design_type="parallel",
        outcome_type="survival",
        parameters={
            "median1": 12.0,  # Control group
            "median2": 18.0,  # Treatment group  
            "enrollment_period": 12.0,
            "follow_up_period": 12.0,
            "alpha": 0.05,
            "power": 0.80,
            "dropout_rate": 0.0
        },
        expected_result={
            "total_sample_size": 182,  # Approximate
            "events_required": 91
        },
        tolerance=0.2  # Survival calculations can vary more
    )
]


# =============================================================================
# CLUSTER RCT BENCHMARKS  
# =============================================================================

DONNER_KLAR_2000_BENCHMARKS = [
    LiteratureBenchmark(
        source="Donner & Klar (2000) - Design and Analysis of Cluster Randomization Trials",
        page="Chapter 4, Example 4.1",
        example_name="Binary outcome cluster trial",
        design_type="cluster",
        outcome_type="binary",
        parameters={
            "p1": 0.10,
            "p2": 0.15,
            "cluster_size": 100,
            "icc": 0.02,
            "alpha": 0.05,
            "power": 0.80
        },
        expected_result={
            "clusters_per_arm": 17,
            "total_clusters": 34,
            "total_sample_size": 3400,
            "design_effect": 2.98  # 1 + (m-1)*ICC = 1 + 99*0.02
        },
        tolerance=0.1
    ),
    
    LiteratureBenchmark(
        source="Donner & Klar (2000)",
        page="Chapter 5, Example 5.2",
        example_name="Continuous outcome cluster trial",
        design_type="cluster",
        outcome_type="continuous",
        parameters={
            "mean1": 140.0,  # Systolic BP control
            "mean2": 130.0,  # Systolic BP intervention
            "std_dev": 20.0,
            "cluster_size": 50,
            "icc": 0.05,
            "alpha": 0.05,
            "power": 0.80
        },
        expected_result={
            "clusters_per_arm": 12,
            "total_clusters": 24,
            "total_sample_size": 1200,
            "design_effect": 3.45  # 1 + (m-1)*ICC = 1 + 49*0.05
        },
        tolerance=0.15
    )
]


# =============================================================================
# SINGLE ARM BENCHMARKS
# =============================================================================

AHERN_2001_BENCHMARKS = [
    LiteratureBenchmark(
        source="A'Hern (2001) - Sample size tables for exact single-stage phase II designs",
        page="Table 1, p0=0.05, p1=0.20",
        example_name="Single stage design low response rate",
        design_type="single_arm",
        outcome_type="binary",
        parameters={
            "p0": 0.05,  # Null hypothesis (uninteresting response rate)
            "p1": 0.20,  # Alternative hypothesis (target response rate)
            "alpha": 0.05,
            "power": 0.80
        },
        expected_result={
            "n": 29,
            "r": 4  # Reject if ≥ 4 responses observed
        },
        tolerance=0.1
    ),
    
    LiteratureBenchmark(
        source="A'Hern (2001)",
        page="Table 1, p0=0.20, p1=0.40",
        example_name="Single stage design moderate response rate",
        design_type="single_arm",
        outcome_type="binary",
        parameters={
            "p0": 0.20,
            "p1": 0.40,
            "alpha": 0.05,
            "power": 0.80
        },
        expected_result={
            "n": 43,
            "r": 13
        },
        tolerance=0.1
    )
]

SIMON_1989_BENCHMARKS = [
    LiteratureBenchmark(
        source="Simon (1989) - Optimal two-stage designs for phase II clinical trials",
        page="Table 1, p0=0.05, p1=0.25",
        example_name="Two-stage minimax design",
        design_type="single_arm", 
        outcome_type="binary",
        parameters={
            "p0": 0.05,
            "p1": 0.25,
            "alpha": 0.05,
            "power": 0.80,
            "design_type": "minimax"
        },
        expected_result={
            "n1": 12,  # Stage 1 sample size
            "r1": 0,   # Stage 1 threshold
            "n": 35,   # Total sample size  
            "r": 5     # Total threshold
        },
        tolerance=0.1
    )
]


# =============================================================================
# VALIDATION TEST FUNCTIONS
# =============================================================================

def validate_benchmark(benchmark: LiteratureBenchmark, 
                      calculation_function,
                      verbose: bool = False) -> Tuple[bool, Dict[str, Any]]:
    """
    Validate a single benchmark against DesignPower calculations.
    
    Returns:
        Tuple of (passed, results_comparison)
    """
    try:
        # Run DesignPower calculation
        if benchmark.design_type == "parallel" and benchmark.outcome_type == "binary":
            if "power" in benchmark.expected_result and "total_sample_size" not in benchmark.parameters:
                # Power calculation
                result = power_binary(**benchmark.parameters)
            else:
                # Sample size calculation  
                result = sample_size_binary(**benchmark.parameters)
                
        elif benchmark.design_type == "parallel" and benchmark.outcome_type == "continuous":
            if "power" in benchmark.expected_result and "total_sample_size" not in benchmark.parameters:
                result = power_continuous(**benchmark.parameters)
            else:
                result = sample_size_continuous(**benchmark.parameters)
                
        # Add more design types as needed...
        else:
            return False, {"error": f"Design type {benchmark.design_type}/{benchmark.outcome_type} not implemented"}
        
        # Compare results
        comparison = {}
        all_passed = True
        
        for key, expected_value in benchmark.expected_result.items():
            if key in result:
                actual_value = result[key]
                relative_error = abs(actual_value - expected_value) / expected_value
                passed = relative_error <= benchmark.tolerance
                
                comparison[key] = {
                    "expected": expected_value,
                    "actual": actual_value,
                    "relative_error": relative_error,
                    "tolerance": benchmark.tolerance,
                    "passed": passed
                }
                
                if not passed:
                    all_passed = False
            else:
                comparison[key] = {
                    "expected": expected_value,
                    "actual": "NOT_FOUND",
                    "passed": False
                }
                all_passed = False
        
        if verbose:
            print(f"\n{benchmark.source}")
            print(f"Example: {benchmark.example_name}")
            print(f"Overall: {'PASS' if all_passed else 'FAIL'}")
            for key, comp in comparison.items():
                status = "✓" if comp["passed"] else "✗"
                print(f"  {status} {key}: {comp['expected']} vs {comp['actual']}")
        
        return all_passed, comparison
        
    except Exception as e:
        return False, {"error": str(e)}


def run_all_benchmarks(verbose: bool = False) -> Dict[str, Any]:
    """Run all literature benchmarks and return summary results."""
    
    all_benchmarks = [
        *FLEISS_1973_BENCHMARKS,
        *COHEN_1988_BENCHMARKS,
        *SCHOENFELD_1981_BENCHMARKS,
        *DONNER_KLAR_2000_BENCHMARKS,
        *AHERN_2001_BENCHMARKS,
        *SIMON_1989_BENCHMARKS
    ]
    
    results = {
        "total_benchmarks": len(all_benchmarks),
        "passed": 0,
        "failed": 0,
        "details": []
    }
    
    print(f"Running {len(all_benchmarks)} literature benchmarks...")
    print("=" * 60)
    
    for benchmark in all_benchmarks:
        passed, comparison = validate_benchmark(benchmark, None, verbose)
        
        if passed:
            results["passed"] += 1
        else:
            results["failed"] += 1
            
        results["details"].append({
            "benchmark": benchmark,
            "passed": passed,
            "comparison": comparison
        })
    
    success_rate = results["passed"] / results["total_benchmarks"] * 100
    
    print(f"\n" + "=" * 60)
    print(f"VALIDATION SUMMARY")
    print(f"=" * 60)
    print(f"Total Benchmarks: {results['total_benchmarks']}")
    print(f"Passed: {results['passed']}")
    print(f"Failed: {results['failed']}")
    print(f"Success Rate: {success_rate:.1f}%")
    
    if results["failed"] > 0:
        print(f"\nFAILED BENCHMARKS:")
        for detail in results["details"]:
            if not detail["passed"]:
                benchmark = detail["benchmark"]
                print(f"  ✗ {benchmark.source} - {benchmark.example_name}")
    
    return results


if __name__ == "__main__":
    # Run validation when script is executed directly
    results = run_all_benchmarks(verbose=True)
    
    # Exit with error code if any benchmarks failed
    exit_code = 1 if results["failed"] > 0 else 0
    exit(exit_code)