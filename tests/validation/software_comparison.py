"""
Cross-validation against established software packages.

This module provides validation tests comparing DesignPower results
against well-established software packages like R's pwr, SAS PROC POWER, etc.
"""

import pytest
import numpy as np
from typing import Dict, Any, List, Tuple
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from core.designs.parallel import sample_size_binary, power_binary


class SoftwareComparison:
    """Container for software comparison data."""
    
    def __init__(self,
                 software: str,
                 function_call: str,
                 parameters: Dict[str, Any],
                 expected_result: Dict[str, Any],
                 tolerance: float = 0.02):
        self.software = software
        self.function_call = function_call
        self.parameters = parameters
        self.expected_result = expected_result
        self.tolerance = tolerance


# =============================================================================
# R PWR PACKAGE COMPARISONS
# =============================================================================

R_PWR_COMPARISONS = [
    SoftwareComparison(
        software="R pwr package",
        function_call="pwr.2p.test(h=0.4636, power=0.8, sig.level=0.05)",
        parameters={
            "p1": 0.3,
            "p2": 0.5,
            "power": 0.8,
            "alpha": 0.05,
            "allocation_ratio": 1.0
        },
        expected_result={
            "sample_size_1": 91,
            "sample_size_2": 91,
            "total_sample_size": 182
        },
        tolerance=0.05
    ),
    
    SoftwareComparison(
        software="R pwr package",
        function_call="pwr.t.test(d=0.5, power=0.8, sig.level=0.05, type='two.sample')",
        parameters={
            "mean1": 0.0,
            "mean2": 0.5,
            "std_dev": 1.0,
            "power": 0.8,
            "alpha": 0.05
        },
        expected_result={
            "sample_size_1": 32,
            "sample_size_2": 32,
            "total_sample_size": 64
        },
        tolerance=0.05
    )
]


# =============================================================================
# SAS PROC POWER COMPARISONS  
# =============================================================================

SAS_PROC_POWER_COMPARISONS = [
    SoftwareComparison(
        software="SAS PROC POWER",
        function_call="proc power; twosamplefreq test=pchi groupproportions=(0.3 0.5) power=0.8 alpha=0.05;",
        parameters={
            "p1": 0.3,
            "p2": 0.5,
            "power": 0.8,
            "alpha": 0.05
        },
        expected_result={
            "total_sample_size": 182
        },
        tolerance=0.05
    )
]


# =============================================================================
# GPOWER COMPARISONS
# =============================================================================

GPOWER_COMPARISONS = [
    SoftwareComparison(
        software="G*Power 3.1",
        function_call="t-tests - Difference between two independent means",
        parameters={
            "mean1": 0.0,
            "mean2": 0.8,  # Effect size d = 0.8
            "std_dev": 1.0,
            "power": 0.8,
            "alpha": 0.05
        },
        expected_result={
            "sample_size_1": 13,
            "sample_size_2": 13,
            "total_sample_size": 26
        },
        tolerance=0.1
    )
]


def validate_software_comparison(comparison: SoftwareComparison,
                               verbose: bool = False) -> Tuple[bool, Dict[str, Any]]:
    """Validate DesignPower against other software results."""
    
    try:
        # Determine which DesignPower function to use based on parameters
        if "p1" in comparison.parameters and "p2" in comparison.parameters:
            if "power" in comparison.expected_result and "total_sample_size" not in comparison.parameters:
                result = power_binary(**comparison.parameters)
            else:
                result = sample_size_binary(**comparison.parameters)
        else:
            # Add more function mappings as needed
            return False, {"error": "Parameter mapping not implemented"}
        
        # Compare results
        results_comparison = {}
        all_passed = True
        
        for key, expected_value in comparison.expected_result.items():
            if key in result:
                actual_value = result[key]
                relative_error = abs(actual_value - expected_value) / expected_value
                passed = relative_error <= comparison.tolerance
                
                results_comparison[key] = {
                    "expected": expected_value,
                    "actual": actual_value,
                    "relative_error": relative_error,
                    "tolerance": comparison.tolerance,
                    "passed": passed
                }
                
                if not passed:
                    all_passed = False
            else:
                results_comparison[key] = {
                    "expected": expected_value,
                    "actual": "NOT_FOUND",
                    "passed": False
                }
                all_passed = False
        
        if verbose:
            print(f"\n{comparison.software}")
            print(f"Function: {comparison.function_call}")
            print(f"Overall: {'PASS' if all_passed else 'FAIL'}")
            for key, comp in results_comparison.items():
                status = "✓" if comp["passed"] else "✗"
                print(f"  {status} {key}: {comp['expected']} vs {comp['actual']}")
        
        return all_passed, results_comparison
        
    except Exception as e:
        return False, {"error": str(e)}


def run_all_software_comparisons(verbose: bool = False) -> Dict[str, Any]:
    """Run all software comparison tests."""
    
    all_comparisons = [
        *R_PWR_COMPARISONS,
        *SAS_PROC_POWER_COMPARISONS,
        *GPOWER_COMPARISONS
    ]
    
    results = {
        "total_comparisons": len(all_comparisons),
        "passed": 0,
        "failed": 0,
        "details": []
    }
    
    print(f"Running {len(all_comparisons)} software comparisons...")
    print("=" * 60)
    
    for comparison in all_comparisons:
        passed, comparison_result = validate_software_comparison(comparison, verbose)
        
        if passed:
            results["passed"] += 1
        else:
            results["failed"] += 1
            
        results["details"].append({
            "comparison": comparison,
            "passed": passed,
            "result": comparison_result
        })
    
    success_rate = results["passed"] / results["total_comparisons"] * 100
    
    print(f"\n" + "=" * 60)
    print(f"SOFTWARE COMPARISON SUMMARY")
    print(f"=" * 60)
    print(f"Total Comparisons: {results['total_comparisons']}")
    print(f"Passed: {results['passed']}")
    print(f"Failed: {results['failed']}")
    print(f"Success Rate: {success_rate:.1f}%")
    
    return results


if __name__ == "__main__":
    results = run_all_software_comparisons(verbose=True)
    exit_code = 1 if results["failed"] > 0 else 0
    exit(exit_code)