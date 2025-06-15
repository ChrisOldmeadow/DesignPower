"""
Parameter Range Validation Against R Packages.

This module tests DesignPower calculations against R packages across
comprehensive parameter ranges to identify systematic differences and
edge case behaviors.
"""

import numpy as np
import pandas as pd
import itertools
from typing import Dict, List, Tuple, Any
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from .r_package_validation import (
    RValidationTest, run_r_calculation, generate_r_code, compare_results
)
from core.designs.parallel import sample_size_binary, sample_size_continuous
from core.designs.single_arm import one_sample_proportion_test_sample_size, one_sample_t_test_sample_size
from core.designs.cluster_rct import sample_size_binary as cluster_sample_size_binary


def generate_parameter_ranges() -> Dict[str, Dict[str, List]]:
    """Generate comprehensive parameter ranges for testing."""
    
    ranges = {
        "parallel_binary": {
            "p1": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            "p2": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            "alpha": [0.01, 0.05, 0.10],
            "power": [0.7, 0.8, 0.9, 0.95]
        },
        
        "parallel_continuous": {
            "mean1": [10, 50, 100],
            "mean2": [15, 60, 110], 
            "sd1": [5, 10, 15, 20, 25],
            "alpha": [0.01, 0.05, 0.10],
            "power": [0.7, 0.8, 0.9, 0.95]
        },
        
        "single_arm_binary": {
            "p0": [0.1, 0.2, 0.3, 0.4, 0.5],
            "alt": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            "alpha": [0.01, 0.05, 0.10],
            "power": [0.7, 0.8, 0.9]
        },
        
        "single_arm_continuous": {
            "mean_diff": [2, 5, 10, 15],
            "sd": [5, 10, 15, 20],
            "alpha": [0.01, 0.05, 0.10],
            "power": [0.7, 0.8, 0.9]
        },
        
        "cluster_binary": {
            "p1": [0.2, 0.3, 0.4, 0.5],
            "p2": [0.3, 0.4, 0.5, 0.6, 0.7],
            "cluster_size": [10, 25, 50, 100],
            "icc": [0.01, 0.02, 0.05, 0.10],
            "alpha": [0.05],
            "power": [0.8, 0.9]
        }
    }
    
    return ranges


def create_parameter_combinations(ranges: Dict[str, List], max_combinations: int = 100) -> List[Dict]:
    """Create parameter combinations, sampling if necessary to limit computational load."""
    
    # Get all possible combinations
    keys = list(ranges.keys())
    values = list(ranges.values())
    all_combinations = list(itertools.product(*values))
    
    # Sample if too many combinations
    if len(all_combinations) > max_combinations:
        indices = np.random.choice(len(all_combinations), max_combinations, replace=False)
        sampled_combinations = [all_combinations[i] for i in indices]
    else:
        sampled_combinations = all_combinations
    
    # Convert to list of dictionaries
    parameter_sets = []
    for combo in sampled_combinations:
        param_dict = dict(zip(keys, combo))
        
        # Add validation filters
        if "p1" in param_dict and "p2" in param_dict:
            # Ensure meaningful difference for binary outcomes
            if abs(param_dict["p2"] - param_dict["p1"]) < 0.05:
                continue
        
        if "p0" in param_dict and "alt" in param_dict:
            # Ensure alternative is greater than null for single-arm
            if param_dict["alt"] <= param_dict["p0"]:
                continue
        
        if "mean1" in param_dict and "mean2" in param_dict:
            # Ensure meaningful difference for continuous outcomes
            if abs(param_dict["mean2"] - param_dict["mean1"]) < 2:
                continue
        
        parameter_sets.append(param_dict)
    
    return parameter_sets


def run_range_validation_parallel_binary(max_tests: int = 50) -> pd.DataFrame:
    """Test parallel binary calculations across parameter ranges."""
    
    ranges = generate_parameter_ranges()["parallel_binary"]
    parameter_sets = create_parameter_combinations(ranges, max_tests)
    
    results = []
    
    print(f"Testing {len(parameter_sets)} parameter combinations for Parallel Binary...")
    
    for i, params in enumerate(parameter_sets):
        try:
            # DesignPower calculation
            py_result = sample_size_binary(**params)
            
            # R calculation using pwr package
            test = RValidationTest(
                test_name=f"Range_Test_{i}",
                design_type="parallel",
                outcome_type="binary",
                r_package="pwr",
                r_function="pwr.2p.test",
                parameters=params,
                python_function=sample_size_binary
            )
            
            r_code = generate_r_code(test)
            r_result = run_r_calculation(r_code)
            
            if "error" not in r_result:
                comparison = compare_results(py_result, r_result, tolerance=0.15)
                
                result_row = {
                    'test_id': i,
                    'design_type': 'parallel_binary',
                    'p1': params['p1'],
                    'p2': params['p2'], 
                    'alpha': params['alpha'],
                    'power': params['power'],
                    'python_n': py_result.get('sample_size_1', 'N/A'),
                    'r_n': r_result.get('n', 'N/A'),
                    'relative_error': comparison['details'].get('sample_size_1', {}).get('relative_error', 'N/A'),
                    'passed': comparison['passed'],
                    'effect_size': abs(params['p2'] - params['p1'])
                }
                
                results.append(result_row)
                
                if i % 10 == 0:
                    print(f"  Completed {i+1}/{len(parameter_sets)} tests")
            
        except Exception as e:
            print(f"  Error in test {i}: {str(e)}")
    
    return pd.DataFrame(results)


def run_range_validation_parallel_continuous(max_tests: int = 50) -> pd.DataFrame:
    """Test parallel continuous calculations across parameter ranges."""
    
    ranges = generate_parameter_ranges()["parallel_continuous"]
    parameter_sets = create_parameter_combinations(ranges, max_tests)
    
    results = []
    
    print(f"Testing {len(parameter_sets)} parameter combinations for Parallel Continuous...")
    
    for i, params in enumerate(parameter_sets):
        try:
            # DesignPower calculation
            py_result = sample_size_continuous(**params)
            
            # R calculation using pwr package
            test = RValidationTest(
                test_name=f"Range_Test_{i}",
                design_type="parallel",
                outcome_type="continuous",
                r_package="pwr",
                r_function="pwr.t.test",
                parameters=params,
                python_function=sample_size_continuous
            )
            
            r_code = generate_r_code(test)
            r_result = run_r_calculation(r_code)
            
            if "error" not in r_result:
                comparison = compare_results(py_result, r_result, tolerance=0.15)
                
                cohens_d = abs(params['mean2'] - params['mean1']) / params['sd1']
                
                result_row = {
                    'test_id': i,
                    'design_type': 'parallel_continuous',
                    'mean_diff': abs(params['mean2'] - params['mean1']),
                    'sd': params['sd1'],
                    'alpha': params['alpha'],
                    'power': params['power'],
                    'python_n': py_result.get('sample_size_1', 'N/A'),
                    'r_n': r_result.get('n', 'N/A'),
                    'relative_error': comparison['details'].get('sample_size_1', {}).get('relative_error', 'N/A'),
                    'passed': comparison['passed'],
                    'cohens_d': cohens_d
                }
                
                results.append(result_row)
                
                if i % 10 == 0:
                    print(f"  Completed {i+1}/{len(parameter_sets)} tests")
            
        except Exception as e:
            print(f"  Error in test {i}: {str(e)}")
    
    return pd.DataFrame(results)


def run_range_validation_single_arm_binary(max_tests: int = 30) -> pd.DataFrame:
    """Test single-arm binary calculations across parameter ranges."""
    
    ranges = generate_parameter_ranges()["single_arm_binary"]
    parameter_sets = create_parameter_combinations(ranges, max_tests)
    
    results = []
    
    print(f"Testing {len(parameter_sets)} parameter combinations for Single-arm Binary...")
    
    for i, params in enumerate(parameter_sets):
        try:
            # DesignPower calculation
            py_result = one_sample_proportion_test_sample_size(**params)
            
            # R calculation using pwr package
            test = RValidationTest(
                test_name=f"Range_Test_{i}",
                design_type="single_arm",
                outcome_type="binary",
                r_package="pwr",
                r_function="pwr.p.test",
                parameters=params,
                python_function=one_sample_proportion_test_sample_size
            )
            
            r_code = generate_r_code(test)
            r_result = run_r_calculation(r_code)
            
            if "error" not in r_result:
                comparison = compare_results(py_result, r_result, tolerance=0.20)
                
                result_row = {
                    'test_id': i,
                    'design_type': 'single_arm_binary',
                    'p0': params['p0'],
                    'alt': params['alt'],
                    'alpha': params['alpha'],
                    'power': params['power'],
                    'python_n': py_result.get('n', 'N/A'),
                    'r_n': r_result.get('n', 'N/A'),
                    'relative_error': comparison['details'].get('n', {}).get('relative_error', 'N/A'),
                    'passed': comparison['passed'],
                    'effect_size': abs(params['alt'] - params['p0'])
                }
                
                results.append(result_row)
                
                if i % 10 == 0:
                    print(f"  Completed {i+1}/{len(parameter_sets)} tests")
            
        except Exception as e:
            print(f"  Error in test {i}: {str(e)}")
    
    return pd.DataFrame(results)


def run_range_validation_cluster_binary(max_tests: int = 25) -> pd.DataFrame:
    """Test cluster binary calculations across parameter ranges."""
    
    ranges = generate_parameter_ranges()["cluster_binary"]
    parameter_sets = create_parameter_combinations(ranges, max_tests)
    
    results = []
    
    print(f"Testing {len(parameter_sets)} parameter combinations for Cluster Binary...")
    
    for i, params in enumerate(parameter_sets):
        try:
            # DesignPower calculation
            py_result = cluster_sample_size_binary(**params)
            
            # R calculation using clusterPower package
            test = RValidationTest(
                test_name=f"Range_Test_{i}",
                design_type="cluster",
                outcome_type="binary",
                r_package="clusterPower",
                r_function="crtpwr.2prop",
                parameters=params,
                python_function=cluster_sample_size_binary
            )
            
            r_code = generate_r_code(test)
            r_result = run_r_calculation(r_code)
            
            if "error" not in r_result:
                comparison = compare_results(py_result, r_result, tolerance=0.20)
                
                design_effect = 1 + (params['cluster_size'] - 1) * params['icc']
                
                result_row = {
                    'test_id': i,
                    'design_type': 'cluster_binary',
                    'p1': params['p1'],
                    'p2': params['p2'],
                    'cluster_size': params['cluster_size'],
                    'icc': params['icc'],
                    'alpha': params['alpha'],
                    'power': params['power'],
                    'python_clusters': py_result.get('n_clusters', 'N/A'),
                    'r_clusters': r_result.get('n_clusters', 'N/A'),
                    'relative_error': comparison['details'].get('n_clusters', {}).get('relative_error', 'N/A'),
                    'passed': comparison['passed'],
                    'design_effect': design_effect
                }
                
                results.append(result_row)
                
                if i % 5 == 0:
                    print(f"  Completed {i+1}/{len(parameter_sets)} tests")
            
        except Exception as e:
            print(f"  Error in test {i}: {str(e)}")
    
    return pd.DataFrame(results)


def analyze_validation_results(results_df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze validation results to identify patterns and issues."""
    
    analysis = {
        "total_tests": len(results_df),
        "pass_rate": (results_df['passed'].sum() / len(results_df)) * 100,
        "mean_relative_error": results_df['relative_error'].mean(),
        "median_relative_error": results_df['relative_error'].median(),
        "max_relative_error": results_df['relative_error'].max(),
        "failed_tests": results_df[~results_df['passed']].copy()
    }
    
    # Identify patterns in failures
    if len(analysis["failed_tests"]) > 0:
        failed = analysis["failed_tests"]
        
        # Check if failures correlate with specific parameter ranges
        patterns = {}
        
        if 'alpha' in failed.columns:
            patterns['alpha_distribution'] = failed['alpha'].value_counts().to_dict()
        
        if 'power' in failed.columns:
            patterns['power_distribution'] = failed['power'].value_counts().to_dict()
        
        if 'effect_size' in failed.columns:
            patterns['effect_size_stats'] = {
                'mean': failed['effect_size'].mean(),
                'median': failed['effect_size'].median(),
                'min': failed['effect_size'].min(),
                'max': failed['effect_size'].max()
            }
        
        if 'relative_error' in failed.columns:
            patterns['error_stats'] = {
                'mean': failed['relative_error'].mean(),
                'median': failed['relative_error'].median(),
                'std': failed['relative_error'].std()
            }
        
        analysis['failure_patterns'] = patterns
    
    return analysis


def run_comprehensive_range_validation() -> Dict[str, Any]:
    """Run comprehensive parameter range validation across all design types."""
    
    print("=" * 80)
    print("COMPREHENSIVE PARAMETER RANGE VALIDATION AGAINST R PACKAGES")
    print("=" * 80)
    
    results = {}
    
    # Test each design type
    design_types = [
        ("parallel_binary", run_range_validation_parallel_binary),
        ("parallel_continuous", run_range_validation_parallel_continuous),
        ("single_arm_binary", run_range_validation_single_arm_binary),
        ("cluster_binary", run_range_validation_cluster_binary)
    ]
    
    for design_type, test_function in design_types:
        print(f"\n{design_type.upper().replace('_', ' ')}")
        print("-" * 50)
        
        try:
            df = test_function()
            analysis = analyze_validation_results(df)
            
            results[design_type] = {
                "data": df,
                "analysis": analysis
            }
            
            print(f"Results: {analysis['total_tests']} tests, {analysis['pass_rate']:.1f}% pass rate")
            print(f"Mean relative error: {analysis['mean_relative_error']:.3f}")
            
            if analysis['pass_rate'] < 90:
                print(f"⚠️  Low pass rate detected for {design_type}")
            
        except Exception as e:
            print(f"❌ Error testing {design_type}: {str(e)}")
            results[design_type] = {"error": str(e)}
    
    # Overall summary
    print(f"\n" + "=" * 80)
    print("OVERALL SUMMARY")
    print("=" * 80)
    
    total_tests = sum(r["analysis"]["total_tests"] for r in results.values() if "analysis" in r)
    total_passed = sum(r["analysis"]["total_tests"] * r["analysis"]["pass_rate"] / 100 
                      for r in results.values() if "analysis" in r)
    overall_pass_rate = (total_passed / total_tests) * 100 if total_tests > 0 else 0
    
    print(f"Total Tests: {total_tests}")
    print(f"Overall Pass Rate: {overall_pass_rate:.1f}%")
    
    # Identify areas needing attention
    low_performers = []
    for design_type, result in results.items():
        if "analysis" in result and result["analysis"]["pass_rate"] < 85:
            low_performers.append((design_type, result["analysis"]["pass_rate"]))
    
    if low_performers:
        print(f"\nAreas needing attention:")
        for design_type, pass_rate in low_performers:
            print(f"  • {design_type}: {pass_rate:.1f}% pass rate")
    
    return results


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Run comprehensive validation
    results = run_comprehensive_range_validation()
    
    # Save results for further analysis
    import pickle
    with open("r_range_validation_results.pkl", "wb") as f:
        pickle.dump(results, f)
    
    print(f"\nResults saved to r_range_validation_results.pkl")