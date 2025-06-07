#!/usr/bin/env python3
"""
Main validation runner for DesignPower.

This script runs comprehensive validation tests and generates reports.
"""

import argparse
import sys
import os
from datetime import datetime
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

def test_basic_functionality():
    """Test that basic DesignPower functions work before validation."""
    try:
        from core.designs.parallel import sample_size_binary
        
        # Simple test
        result = sample_size_binary(
            p1=0.3,
            p2=0.5,
            power=0.8,
            alpha=0.05
        )
        
        # Check result has expected keys
        required_keys = ['sample_size_1', 'sample_size_2', 'total_sample_size']
        for key in required_keys:
            if key not in result:
                return False, f"Missing key: {key}"
        
        # Sanity check values
        if result['total_sample_size'] < 10 or result['total_sample_size'] > 1000:
            return False, f"Unreasonable sample size: {result['total_sample_size']}"
            
        return True, "Basic functionality test passed"
        
    except Exception as e:
        return False, f"Basic functionality test failed: {e}"


def run_quick_validation():
    """Run a subset of validation tests for quick feedback."""
    print("Running Quick Validation Tests...")
    print("=" * 50)
    
    # Test basic functionality first
    basic_passed, basic_msg = test_basic_functionality()
    print(f"Basic Functionality: {'PASS' if basic_passed else 'FAIL'}")
    if not basic_passed:
        print(f"  Error: {basic_msg}")
        return False
    
    # Test a few key benchmarks
    quick_benchmarks = [
        {
            "name": "Fleiss Binary Sample Size",
            "function": "sample_size_binary",
            "params": {"p1": 0.65, "p2": 0.85, "power": 0.8, "alpha": 0.05},
            "expected": {"total_sample_size": 62},
            "tolerance": 0.1
        },
        {
            "name": "Cohen Continuous Medium Effect",
            "function": "sample_size_continuous", 
            "params": {"mean1": 0.0, "mean2": 0.5, "std_dev": 1.0, "power": 0.8, "alpha": 0.05},
            "expected": {"total_sample_size": 64},
            "tolerance": 0.1
        }
    ]
    
    passed = 0
    total = len(quick_benchmarks)
    
    for benchmark in quick_benchmarks:
        try:
            if benchmark["function"] == "sample_size_binary":
                from core.designs.parallel import sample_size_binary
                result = sample_size_binary(**benchmark["params"])
            elif benchmark["function"] == "sample_size_continuous":
                from core.designs.parallel import sample_size_continuous
                result = sample_size_continuous(**benchmark["params"])
            else:
                print(f"  {benchmark['name']}: SKIP (function not implemented)")
                continue
            
            # Check results
            benchmark_passed = True
            for key, expected in benchmark["expected"].items():
                if key in result:
                    actual = result[key]
                    relative_error = abs(actual - expected) / expected
                    if relative_error > benchmark["tolerance"]:
                        benchmark_passed = False
                        print(f"  {benchmark['name']}: FAIL ({key}: {actual} vs {expected}, error: {relative_error:.2%})")
                        break
                else:
                    benchmark_passed = False
                    print(f"  {benchmark['name']}: FAIL (missing key: {key})")
                    break
            
            if benchmark_passed:
                passed += 1
                print(f"  {benchmark['name']}: PASS")
                
        except Exception as e:
            print(f"  {benchmark['name']}: ERROR ({e})")
    
    success_rate = passed / total * 100
    print(f"\nQuick Validation: {passed}/{total} passed ({success_rate:.1f}%)")
    
    return success_rate >= 80  # 80% threshold for quick validation


def run_comprehensive_validation():
    """Run full validation suite."""
    print("Running Comprehensive Validation...")
    print("=" * 50)
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "tests": {}
    }
    
    # Literature benchmarks
    try:
        from literature_benchmarks import run_all_benchmarks
        lit_results = run_all_benchmarks(verbose=False)
        results["tests"]["literature"] = lit_results
        print(f"Literature Benchmarks: {lit_results['passed']}/{lit_results['total_benchmarks']} passed")
    except Exception as e:
        print(f"Literature Benchmarks: ERROR ({e})")
        results["tests"]["literature"] = {"error": str(e)}
    
    # Software comparisons  
    try:
        from software_comparison import run_all_software_comparisons
        soft_results = run_all_software_comparisons(verbose=False)
        results["tests"]["software"] = soft_results
        print(f"Software Comparisons: {soft_results['passed']}/{soft_results['total_comparisons']} passed")
    except Exception as e:
        print(f"Software Comparisons: ERROR ({e})")
        results["tests"]["software"] = {"error": str(e)}
    
    # Calculate overall success rate
    total_tests = 0
    total_passed = 0
    
    for test_type, test_results in results["tests"].items():
        if "error" not in test_results:
            if "total_benchmarks" in test_results:
                total_tests += test_results["total_benchmarks"]
                total_passed += test_results["passed"]
            elif "total_comparisons" in test_results:
                total_tests += test_results["total_comparisons"]
                total_passed += test_results["passed"]
    
    if total_tests > 0:
        overall_success = total_passed / total_tests * 100
        print(f"\nOverall Validation: {total_passed}/{total_tests} passed ({overall_success:.1f}%)")
        
        if overall_success >= 95:
            print("✅ EXCELLENT: Validation exceeds target threshold")
        elif overall_success >= 90:
            print("✅ GOOD: Validation meets minimum threshold") 
        elif overall_success >= 85:
            print("⚠️  WARNING: Validation below target but acceptable")
        else:
            print("❌ CRITICAL: Validation below acceptable threshold")
            
        return overall_success >= 85
    else:
        print("❌ ERROR: No validation tests completed successfully")
        return False


def main():
    """Main validation runner."""
    parser = argparse.ArgumentParser(description="DesignPower Validation Runner")
    parser.add_argument("--quick", action="store_true", help="Run quick validation only")
    parser.add_argument("--comprehensive", action="store_true", help="Run comprehensive validation")
    parser.add_argument("--report", help="Generate HTML report (filename)")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.quick:
        success = run_quick_validation()
    elif args.comprehensive:
        success = run_comprehensive_validation()
    else:
        # Default: run quick validation
        print("Running default quick validation (use --comprehensive for full suite)")
        success = run_quick_validation()
    
    if args.report:
        print(f"\nGenerating validation report: {args.report}")
        # TODO: Implement HTML report generation
        print("HTML report generation not yet implemented")
    
    # Exit with appropriate code
    exit_code = 0 if success else 1
    print(f"\nValidation {'PASSED' if success else 'FAILED'}")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()