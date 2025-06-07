#!/usr/bin/env python3
"""
Validation runner using authoritative benchmarks.

This script validates DesignPower calculations against carefully researched
benchmarks from authoritative sources in statistical methodology.
"""

import sys
import os
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import asdict
import json

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from authoritative_benchmarks import (
    AuthoritativeBenchmark, ALL_BENCHMARKS, COHEN_1988_BENCHMARKS,
    get_benchmarks_by_design, get_cross_validated_benchmarks
)


class BenchmarkValidator:
    """Validates DesignPower against authoritative benchmarks."""
    
    def __init__(self):
        self.results = []
        
    def validate_benchmark(self, benchmark: AuthoritativeBenchmark, verbose: bool = False) -> Dict[str, Any]:
        """
        Validate a single benchmark against DesignPower.
        
        Returns:
            Dict with validation results including pass/fail status and details
        """
        result = {
            "benchmark": benchmark,
            "passed": False,
            "error": None,
            "comparisons": {},
            "designpower_result": None
        }
        
        try:
            # Map benchmark to DesignPower function
            designpower_result = self._call_designpower_function(benchmark)
            result["designpower_result"] = designpower_result
            
            if "error" in designpower_result:
                result["error"] = designpower_result["error"]
                return result
            
            # Compare each expected result
            all_passed = True
            for key, expected_value in benchmark.expected_results.items():
                comparison = self._compare_values(
                    key, expected_value, designpower_result, benchmark.tolerance
                )
                result["comparisons"][key] = comparison
                
                if not comparison["passed"]:
                    all_passed = False
            
            result["passed"] = all_passed
            
            if verbose:
                self._print_benchmark_result(benchmark, result)
                
        except Exception as e:
            result["error"] = str(e)
            if verbose:
                print(f"ERROR validating {benchmark.test_name}: {e}")
        
        return result
    
    def _call_designpower_function(self, benchmark: AuthoritativeBenchmark) -> Dict[str, Any]:
        """Map benchmark parameters to appropriate DesignPower function."""
        
        try:
            if benchmark.design_type == "parallel" and benchmark.outcome_type == "continuous":
                return self._validate_parallel_continuous(benchmark)
            elif benchmark.design_type == "parallel" and benchmark.outcome_type == "binary":
                return self._validate_parallel_binary(benchmark)
            elif benchmark.design_type == "single_arm" and benchmark.outcome_type == "binary":
                return self._validate_single_arm_binary(benchmark)
            elif benchmark.design_type == "cluster" and benchmark.outcome_type == "binary":
                return self._validate_cluster_binary(benchmark)
            else:
                return {"error": f"Design type {benchmark.design_type}/{benchmark.outcome_type} not implemented"}
                
        except ImportError as e:
            return {"error": f"Import error: {e}"}
        except Exception as e:
            return {"error": f"Calculation error: {e}"}
    
    def _validate_parallel_continuous(self, benchmark: AuthoritativeBenchmark) -> Dict[str, Any]:
        """Validate parallel continuous outcome benchmark."""
        
        # Try different potential function signatures
        params = benchmark.parameters
        
        # Try analytical functions first
        try:
            from core.designs.parallel.analytical import sample_size_continuous
            
            # Convert Cohen's d to mean difference (assuming std_dev = 1)
            if "effect_size_d" in params:
                effect_size = params["effect_size_d"]
                # For Cohen's d, we can use delta = d * pooled_std_dev
                # Assuming std_dev = 1 for standardized effect
                result = sample_size_continuous(
                    delta=effect_size,
                    std_dev=1.0,
                    power=params.get("power", 0.8),
                    alpha=params.get("alpha", 0.05),
                    allocation_ratio=params.get("allocation_ratio", 1.0)
                )
                return result
                
        except ImportError:
            pass
        except Exception as e:
            pass
        
        # Try main module functions
        try:
            from core.designs.parallel import sample_size_continuous
            
            if "effect_size_d" in params:
                effect_size = params["effect_size_d"]
                result = sample_size_continuous(
                    mean1=0.0,
                    mean2=effect_size,  # Effect size as mean difference
                    sd1=1.0,
                    power=params.get("power", 0.8),
                    alpha=params.get("alpha", 0.05),
                    allocation_ratio=params.get("allocation_ratio", 1.0)
                )
                return result
                
        except ImportError:
            pass
        except Exception as e:
            pass
            
        return {"error": "No suitable continuous function found"}
    
    def _validate_parallel_binary(self, benchmark: AuthoritativeBenchmark) -> Dict[str, Any]:
        """Validate parallel binary outcome benchmark."""
        
        params = benchmark.parameters
        
        try:
            from core.designs.parallel import sample_size_binary
            
            result = sample_size_binary(
                p1=params["p1"],
                p2=params["p2"],
                power=params.get("power", 0.8),
                alpha=params.get("alpha", 0.05),
                allocation_ratio=params.get("allocation_ratio", 1.0)
            )
            return result
            
        except ImportError:
            return {"error": "Could not import sample_size_binary"}
        except Exception as e:
            return {"error": f"Binary calculation failed: {e}"}
    
    def _validate_single_arm_binary(self, benchmark: AuthoritativeBenchmark) -> Dict[str, Any]:
        """Validate single arm binary benchmark."""
        
        params = benchmark.parameters
        
        try:
            from app.components.single_arm import calculate_single_arm_binary
            
            # Map parameters to DesignPower format
            dp_params = {
                "calculation_type": "Sample Size",
                "p0": params["p0"],
                "p": params["p1"],  # Note: p1 in benchmark = p in DesignPower
                "alpha": params.get("alpha", 0.05),
                "power": params.get("power", 0.8),
                "design_method": "A'Hern"
            }
            
            result = calculate_single_arm_binary(dp_params)
            return result
            
        except ImportError:
            return {"error": "Could not import single arm functions"}
        except Exception as e:
            return {"error": f"Single arm calculation failed: {e}"}
    
    def _validate_cluster_binary(self, benchmark: AuthoritativeBenchmark) -> Dict[str, Any]:
        """Validate cluster binary benchmark."""
        
        params = benchmark.parameters
        
        try:
            from app.components.cluster_rct import calculate_cluster_binary
            
            dp_params = {
                "calc_type": "Sample Size",
                "p1": params["p1"],
                "p2": params["p2"],
                "cluster_size": params["cluster_size"],
                "icc": params["icc"],
                "power": params.get("power", 0.8),
                "alpha": params.get("alpha", 0.05)
            }
            
            result = calculate_cluster_binary(dp_params)
            return result
            
        except ImportError:
            return {"error": "Could not import cluster functions"}
        except Exception as e:
            return {"error": f"Cluster calculation failed: {e}"}
    
    def _compare_values(self, key: str, expected: Any, actual_result: Dict[str, Any], 
                       tolerance: float) -> Dict[str, Any]:
        """Compare expected vs actual values with tolerance."""
        
        # Try different possible key mappings
        possible_keys = [
            key,
            key.replace("_", ""),
            key.replace("sample_size_per_group", "sample_size_1"),
            key.replace("sample_size_per_group", "n1"),
            key.replace("total_sample_size", "total_n"),
            key.replace("total_sample_size", "total_sample_size"),
            key.replace("sample_size", "n")
        ]
        
        actual_value = None
        actual_key_used = None
        
        for possible_key in possible_keys:
            if possible_key in actual_result:
                actual_value = actual_result[possible_key]
                actual_key_used = possible_key
                break
        
        if actual_value is None:
            return {
                "expected": expected,
                "actual": "NOT_FOUND",
                "actual_key": None,
                "relative_error": None,
                "tolerance": tolerance,
                "passed": False
            }
        
        # Calculate relative error
        if expected == 0:
            relative_error = abs(actual_value) if actual_value != 0 else 0.0
        else:
            relative_error = abs(actual_value - expected) / abs(expected)
        
        passed = relative_error <= tolerance
        
        return {
            "expected": expected,
            "actual": actual_value,
            "actual_key": actual_key_used,
            "relative_error": relative_error,
            "tolerance": tolerance,
            "passed": passed
        }
    
    def _print_benchmark_result(self, benchmark: AuthoritativeBenchmark, result: Dict[str, Any]):
        """Print detailed results for a benchmark."""
        
        print(f"\n{benchmark.authors} ({benchmark.year}): {benchmark.test_name}")
        print(f"Source: {benchmark.page_reference}")
        
        if result["error"]:
            print(f"❌ ERROR: {result['error']}")
            return
        
        overall_status = "✅ PASS" if result["passed"] else "❌ FAIL"
        print(f"{overall_status}")
        
        for key, comparison in result["comparisons"].items():
            if comparison["passed"]:
                print(f"  ✓ {key}: {comparison['actual']} (expected: {comparison['expected']})")
            else:
                error_pct = comparison["relative_error"] * 100 if comparison["relative_error"] else "N/A"
                print(f"  ✗ {key}: {comparison['actual']} vs {comparison['expected']} "
                      f"(error: {error_pct:.1f}%, tolerance: {comparison['tolerance']*100:.1f}%)")
        
        if benchmark.notes:
            print(f"  Note: {benchmark.notes}")
    
    def run_all_benchmarks(self, verbose: bool = False, 
                          filter_design: Optional[str] = None,
                          filter_cross_validated: bool = False) -> Dict[str, Any]:
        """Run validation against all benchmarks."""
        
        # Select benchmarks to run
        benchmarks = ALL_BENCHMARKS
        
        if filter_cross_validated:
            benchmarks = get_cross_validated_benchmarks()
        
        if filter_design:
            benchmarks = [b for b in benchmarks if b.design_type == filter_design]
        
        print(f"Running validation against {len(benchmarks)} authoritative benchmarks...")
        print("=" * 70)
        
        results = []
        passed_count = 0
        
        for benchmark in benchmarks:
            result = self.validate_benchmark(benchmark, verbose)
            results.append(result)
            
            if result["passed"]:
                passed_count += 1
            
            # Brief status for non-verbose mode
            if not verbose:
                status = "✅" if result["passed"] else "❌"
                error_info = f" ({result['error']})" if result["error"] else ""
                print(f"{status} {benchmark.authors} ({benchmark.year}): {benchmark.test_name}{error_info}")
        
        # Summary
        total_count = len(results)
        success_rate = (passed_count / total_count * 100) if total_count > 0 else 0
        
        print("\n" + "=" * 70)
        print("VALIDATION SUMMARY")
        print("=" * 70)
        print(f"Total Benchmarks: {total_count}")
        print(f"Passed: {passed_count}")
        print(f"Failed: {total_count - passed_count}")
        print(f"Success Rate: {success_rate:.1f}%")
        
        # Success thresholds
        if success_rate >= 95:
            print("🎉 EXCELLENT: Validation exceeds target threshold (≥95%)")
        elif success_rate >= 90:
            print("✅ GOOD: Validation meets minimum threshold (≥90%)")
        elif success_rate >= 80:
            print("⚠️  WARNING: Validation below target but acceptable (≥80%)")
        else:
            print("❌ CRITICAL: Validation below acceptable threshold (<80%)")
        
        return {
            "total": total_count,
            "passed": passed_count,
            "failed": total_count - passed_count,
            "success_rate": success_rate,
            "results": results
        }


def main():
    """Main validation runner."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate DesignPower against authoritative benchmarks")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--design", choices=["parallel", "cluster", "single_arm"], 
                       help="Filter by design type")
    parser.add_argument("--cross-validated", action="store_true", 
                       help="Only run cross-validated benchmarks")
    parser.add_argument("--list", action="store_true", help="List all available benchmarks")
    
    args = parser.parse_args()
    
    if args.list:
        from authoritative_benchmarks import print_benchmark_summary
        print_benchmark_summary()
        return
    
    validator = BenchmarkValidator()
    results = validator.run_all_benchmarks(
        verbose=args.verbose,
        filter_design=args.design,
        filter_cross_validated=args.cross_validated
    )
    
    # Exit with error code if validation failed
    success_threshold = 80  # 80% minimum
    exit_code = 0 if results["success_rate"] >= success_threshold else 1
    sys.exit(exit_code)


if __name__ == "__main__":
    main()