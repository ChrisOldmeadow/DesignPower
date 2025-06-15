"""
R Package Cross-Validation for DesignPower.

This module provides comprehensive validation of DesignPower analytical calculations
against established R packages across multiple parameter ranges. This helps identify
methodological differences and ensures our implementations align with industry standards.
"""

import pytest
import numpy as np
import subprocess
import json
import tempfile
import os
from typing import Dict, List, Tuple, Any
import sys

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from core.designs.parallel import (
    sample_size_binary, power_binary, 
    sample_size_continuous, power_continuous
)
from core.designs.single_arm import (
    one_sample_proportion_test_sample_size, one_sample_proportion_test_power,
    one_sample_t_test_sample_size, one_sample_t_test_power
)
from core.designs.single_arm.binary import (
    ahern_sample_size, ahern_power, simons_two_stage_design
)
from core.designs.cluster_rct import (
    sample_size_binary as cluster_sample_size_binary, 
    sample_size_continuous as cluster_sample_size_continuous,
    power_binary as cluster_power_binary,
    power_continuous as cluster_power_continuous
)


class RValidationTest:
    """Container for R validation test data."""
    
    def __init__(self, 
                 test_name: str,
                 design_type: str,
                 outcome_type: str,
                 r_package: str,
                 r_function: str,
                 parameters: Dict[str, Any],
                 python_function,
                 tolerance: float = 0.05):
        self.test_name = test_name
        self.design_type = design_type
        self.outcome_type = outcome_type
        self.r_package = r_package
        self.r_function = r_function
        self.parameters = parameters
        self.python_function = python_function
        self.tolerance = tolerance


def run_r_calculation(r_code: str) -> Dict[str, Any]:
    """
    Execute R code and return results as JSON.
    
    Parameters
    ----------
    r_code : str
        R code to execute
        
    Returns
    -------
    dict
        Results from R calculation
    """
    try:
        # Create temporary R script
        with tempfile.NamedTemporaryFile(mode='w', suffix='.R', delete=False) as f:
            f.write(r_code)
            r_script_path = f.name
        
        # Run R script
        result = subprocess.run(
            ['Rscript', r_script_path],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        # Clean up
        os.unlink(r_script_path)
        
        if result.returncode != 0:
            raise RuntimeError(f"R script failed: {result.stderr}")
        
        # Parse JSON output
        return json.loads(result.stdout.strip())
        
    except Exception as e:
        return {"error": str(e)}


def validate_r_test(test: RValidationTest, verbose: bool = False) -> Tuple[bool, Dict[str, Any]]:
    """
    Validate a single test against R package.
    
    Returns
    -------
    tuple
        (passed, results_comparison)
    """
    try:
        # Run DesignPower calculation
        python_result = test.python_function(**test.parameters)
        
        # Generate R code based on test type
        r_code = generate_r_code(test)
        
        # Run R calculation
        r_result = run_r_calculation(r_code)
        
        if "error" in r_result:
            return False, {"error": f"R calculation failed: {r_result['error']}"}
        
        # Compare results
        comparison = compare_results(python_result, r_result, test.tolerance)
        
        if verbose:
            print(f"\n{test.test_name}")
            print(f"R Package: {test.r_package}")
            print(f"Overall: {'PASS' if comparison['passed'] else 'FAIL'}")
            for key, comp in comparison['details'].items():
                if isinstance(comp, dict) and 'passed' in comp:
                    status = "✓" if comp['passed'] else "✗"
                    print(f"  {status} {key}: Python={comp.get('python', 'N/A')}, R={comp.get('r', 'N/A')}")
        
        return comparison['passed'], comparison
        
    except Exception as e:
        return False, {"error": str(e)}


def generate_r_code(test: RValidationTest) -> str:
    """Generate R code for the given test."""
    
    if test.design_type == "parallel" and test.outcome_type == "binary":
        return generate_r_parallel_binary(test)
    elif test.design_type == "parallel" and test.outcome_type == "continuous":
        return generate_r_parallel_continuous(test)
    elif test.design_type == "single_arm" and test.outcome_type == "binary":
        return generate_r_single_arm_binary(test)
    elif test.design_type == "single_arm" and test.outcome_type == "continuous":
        return generate_r_single_arm_continuous(test)
    elif test.design_type == "cluster":
        return generate_r_cluster(test)
    else:
        raise ValueError(f"Unsupported test type: {test.design_type}/{test.outcome_type}")


def generate_r_parallel_binary(test: RValidationTest) -> str:
    """Generate R code for parallel binary tests."""
    params = test.parameters
    
    if test.r_package == "pwr":
        # Use pwr package for effect size-based calculations
        p1, p2 = params['p1'], params['p2']
        h = f"2 * (asin(sqrt({p2})) - asin(sqrt({p1})))"  # Cohen's h
        
        if 'power' in params and 'n' not in params:
            # Sample size calculation
            r_code = f"""
library(pwr)
library(jsonlite)

h <- {h}
result <- pwr.2p.test(h = h, power = {params['power']}, sig.level = {params['alpha']})

output <- list(
    n = ceiling(result$n),
    power = result$power,
    effect_size = h
)

cat(toJSON(output))
"""
        else:
            # Power calculation
            n1 = params.get('n1', params.get('n', 50))
            r_code = f"""
library(pwr)
library(jsonlite)

h <- {h}
result <- pwr.2p.test(h = h, n = {n1}, sig.level = {params['alpha']})

output <- list(
    power = result$power,
    n = {n1},
    effect_size = h
)

cat(toJSON(output))
"""
    
    elif test.r_package == "Hmisc":
        # Use Hmisc for more direct proportion calculations
        p1, p2 = params['p1'], params['p2']
        r_code = f"""
library(Hmisc)
library(jsonlite)

result <- bsamsize(p1 = {p1}, p2 = {p2}, fraction = 0.5, alpha = {params['alpha']}, power = {params['power']})

output <- list(
    n = ceiling(result$n),
    n1 = ceiling(result$n),
    n2 = ceiling(result$n)
)

cat(toJSON(output))
"""
    
    return r_code


def generate_r_parallel_continuous(test: RValidationTest) -> str:
    """Generate R code for parallel continuous tests."""
    params = test.parameters
    
    if test.r_package == "pwr":
        # Calculate Cohen's d
        mean1, mean2 = params['mean1'], params['mean2']
        if 'sd1' in params and 'sd2' in params:
            pooled_sd = f"sqrt(({params['sd1']}^2 + {params['sd2']}^2) / 2)"
        else:
            pooled_sd = str(params.get('sd1', params.get('sd', 15)))
        
        d = f"abs({mean2} - {mean1}) / {pooled_sd}"
        
        if 'power' in params and 'n1' not in params:
            # Sample size calculation
            r_code = f"""
library(pwr)
library(jsonlite)

d <- {d}
result <- pwr.t.test(d = d, power = {params['power']}, sig.level = {params['alpha']}, type = "two.sample")

output <- list(
    n = ceiling(result$n),
    n1 = ceiling(result$n),
    n2 = ceiling(result$n),
    power = result$power,
    effect_size = d
)

cat(toJSON(output))
"""
        else:
            # Power calculation
            n1 = params.get('n1', params.get('n', 50))
            r_code = f"""
library(pwr)
library(jsonlite)

d <- {d}
result <- pwr.t.test(d = d, n = {n1}, sig.level = {params['alpha']}, type = "two.sample")

output <- list(
    power = result$power,
    n = {n1},
    effect_size = d
)

cat(toJSON(output))
"""
    
    return r_code


def generate_r_single_arm_binary(test: RValidationTest) -> str:
    """Generate R code for single-arm binary tests."""
    params = test.parameters
    
    if test.r_package == "pwr":
        # Use one-sample proportion test
        p0, p1 = params['p0'], params.get('p1', params.get('alt', 0.5))
        h = f"2 * (asin(sqrt({p1})) - asin(sqrt({p0})))"
        
        r_code = f"""
library(pwr)
library(jsonlite)

h <- {h}
result <- pwr.p.test(h = h, power = {params['power']}, sig.level = {params['alpha']}, alternative = "greater")

output <- list(
    n = ceiling(result$n),
    power = result$power,
    effect_size = h
)

cat(toJSON(output))
"""
    
    elif test.r_package == "gsDesign":
        # Use gsDesign for exact single-arm calculations
        p0, p1 = params['p0'], params.get('p1', params.get('alt', 0.5))
        
        r_code = f"""
library(gsDesign)
library(jsonlite)

result <- nBinomial(p1 = {p1}, p2 = {p0}, alpha = {params['alpha']}, beta = {1 - params['power']}, sided = 1)

output <- list(
    n = result$n,
    power = {params['power']},
    alpha = {params['alpha']}
)

cat(toJSON(output))
"""
    
    return r_code


def generate_r_single_arm_continuous(test: RValidationTest) -> str:
    """Generate R code for single-arm continuous tests."""
    params = test.parameters
    
    if test.r_package == "pwr":
        # Calculate effect size
        mean_diff = params.get('mean_diff', params.get('delta', 5))
        sd = params.get('sd', params.get('sigma', 15))
        d = f"{mean_diff} / {sd}"
        
        r_code = f"""
library(pwr)
library(jsonlite)

d <- {d}
result <- pwr.t.test(d = d, power = {params['power']}, sig.level = {params['alpha']}, type = "one.sample")

output <- list(
    n = ceiling(result$n),
    power = result$power,
    effect_size = d
)

cat(toJSON(output))
"""
    
    return r_code


def generate_r_cluster(test: RValidationTest) -> str:
    """Generate R code for cluster randomized trials."""
    params = test.parameters
    
    if test.r_package == "clusterPower" and test.outcome_type == "binary":
        r_code = f"""
library(clusterPower)
library(jsonlite)

result <- crtpwr.2prop(
    alpha = {params['alpha']},
    power = {params['power']},
    m = {params['cluster_size']},
    p1 = {params['p1']},
    p2 = {params['p2']},
    icc = {params['icc']}
)

output <- list(
    n_clusters = result$n,
    total_clusters = result$n * 2,
    cluster_size = {params['cluster_size']},
    icc = {params['icc']}
)

cat(toJSON(output))
"""
    
    elif test.r_package == "clusterPower" and test.outcome_type == "continuous":
        r_code = f"""
library(clusterPower)
library(jsonlite)

result <- crtpwr.2mean(
    alpha = {params['alpha']},
    power = {params['power']},
    m = {params['cluster_size']},
    d = abs({params['mean2']} - {params['mean1']}) / {params['std_dev']},
    icc = {params['icc']}
)

output <- list(
    n_clusters = result$n,
    total_clusters = result$n * 2,
    cluster_size = {params['cluster_size']},
    icc = {params['icc']}
)

cat(toJSON(output))
"""
    
    return r_code


def compare_results(python_result: Dict, r_result: Dict, tolerance: float) -> Dict[str, Any]:
    """Compare Python and R results."""
    
    comparison = {
        "passed": True,
        "details": {}
    }
    
    # Key mappings between Python and R outputs
    key_mappings = {
        'sample_size_1': ['n', 'n1'],
        'sample_size_2': ['n', 'n2'], 
        'total_sample_size': ['n', 'total_n'],
        'n_clusters': ['n_clusters', 'n'],
        'power': ['power'],
        'n': ['n'],
        'n1': ['n1', 'n'],
        'n2': ['n2', 'n']
    }
    
    for py_key, r_keys in key_mappings.items():
        if py_key in python_result:
            py_value = python_result[py_key]
            
            # Find matching R key
            r_value = None
            for r_key in r_keys:
                if r_key in r_result:
                    r_value = r_result[r_key]
                    break
            
            if r_value is not None:
                if isinstance(py_value, (int, float)) and isinstance(r_value, (int, float)):
                    relative_error = abs(py_value - r_value) / max(abs(r_value), 1e-10)
                    passed = relative_error <= tolerance
                    
                    comparison['details'][py_key] = {
                        'python': py_value,
                        'r': r_value,
                        'relative_error': relative_error,
                        'tolerance': tolerance,
                        'passed': passed
                    }
                    
                    if not passed:
                        comparison['passed'] = False
    
    return comparison


# =============================================================================
# R VALIDATION TEST DEFINITIONS
# =============================================================================

def get_r_validation_tests() -> List[RValidationTest]:
    """Get all R validation tests."""
    
    tests = []
    
    # Parallel Binary Tests
    tests.extend([
        RValidationTest(
            test_name="Parallel Binary - pwr package",
            design_type="parallel",
            outcome_type="binary",
            r_package="pwr",
            r_function="pwr.2p.test",
            parameters={"p1": 0.3, "p2": 0.5, "alpha": 0.05, "power": 0.8},
            python_function=sample_size_binary,
            tolerance=0.10
        ),
        RValidationTest(
            test_name="Parallel Binary Power - pwr package",
            design_type="parallel",
            outcome_type="binary",
            r_package="pwr",
            r_function="pwr.2p.test",
            parameters={"p1": 0.3, "p2": 0.5, "alpha": 0.05, "n1": 90, "n2": 90},
            python_function=power_binary,
            tolerance=0.05
        ),
    ])
    
    # Parallel Continuous Tests
    tests.extend([
        RValidationTest(
            test_name="Parallel Continuous - pwr package",
            design_type="parallel",
            outcome_type="continuous",
            r_package="pwr",
            r_function="pwr.t.test",
            parameters={"mean1": 100, "mean2": 110, "sd1": 15, "alpha": 0.05, "power": 0.8},
            python_function=sample_size_continuous,
            tolerance=0.10
        ),
        RValidationTest(
            test_name="Parallel Continuous Power - pwr package",
            design_type="parallel",
            outcome_type="continuous",
            r_package="pwr",
            r_function="pwr.t.test",
            parameters={"mean1": 100, "mean2": 110, "sd1": 15, "alpha": 0.05, "n1": 35, "n2": 35},
            python_function=power_continuous,
            tolerance=0.05
        ),
    ])
    
    # Single-arm Binary Tests
    tests.extend([
        RValidationTest(
            test_name="Single-arm Binary - pwr package",
            design_type="single_arm",
            outcome_type="binary",
            r_package="pwr",
            r_function="pwr.p.test",
            parameters={"p0": 0.3, "alt": 0.5, "alpha": 0.05, "power": 0.8},
            python_function=one_sample_proportion_test_sample_size,
            tolerance=0.10
        ),
    ])
    
    # Single-arm Continuous Tests
    tests.extend([
        RValidationTest(
            test_name="Single-arm Continuous - pwr package",
            design_type="single_arm",
            outcome_type="continuous",
            r_package="pwr",
            r_function="pwr.t.test",
            parameters={"mean_diff": 5, "sd": 15, "alpha": 0.05, "power": 0.8},
            python_function=one_sample_t_test_sample_size,
            tolerance=0.10
        ),
    ])
    
    # Cluster RCT Tests
    tests.extend([
        RValidationTest(
            test_name="Cluster Binary - clusterPower package",
            design_type="cluster",
            outcome_type="binary",
            r_package="clusterPower",
            r_function="crtpwr.2prop",
            parameters={"p1": 0.3, "p2": 0.5, "cluster_size": 50, "icc": 0.02, "alpha": 0.05, "power": 0.8},
            python_function=cluster_sample_size_binary,
            tolerance=0.15
        ),
        RValidationTest(
            test_name="Cluster Continuous - clusterPower package",
            design_type="cluster",
            outcome_type="continuous",
            r_package="clusterPower",
            r_function="crtpwr.2mean",
            parameters={"mean1": 10, "mean2": 15, "std_dev": 8, "cluster_size": 30, "icc": 0.05, "alpha": 0.05, "power": 0.8},
            python_function=cluster_sample_size_continuous,
            tolerance=0.15
        ),
    ])
    
    return tests


def run_r_validation_suite(verbose: bool = False) -> Dict[str, Any]:
    """Run the complete R validation suite."""
    
    tests = get_r_validation_tests()
    
    results = {
        "total_tests": len(tests),
        "passed": 0,
        "failed": 0,
        "errors": 0,
        "details": []
    }
    
    print(f"Running {len(tests)} R package validation tests...")
    print("=" * 80)
    
    for test in tests:
        try:
            passed, comparison = validate_r_test(test, verbose)
            
            if "error" in comparison:
                results["errors"] += 1
                status = "ERROR"
            elif passed:
                results["passed"] += 1
                status = "PASS"
            else:
                results["failed"] += 1
                status = "FAIL"
            
            results["details"].append({
                "test": test,
                "status": status,
                "comparison": comparison
            })
            
            if not verbose:
                print(f"  {status:<6} {test.test_name} ({test.r_package})")
                
        except Exception as e:
            results["errors"] += 1
            results["details"].append({
                "test": test,
                "status": "ERROR",
                "comparison": {"error": str(e)}
            })
            print(f"  ERROR  {test.test_name} - {str(e)}")
    
    success_rate = results["passed"] / results["total_tests"] * 100
    
    print(f"\n" + "=" * 80)
    print(f"R VALIDATION SUMMARY")
    print(f"=" * 80)
    print(f"Total Tests: {results['total_tests']}")
    print(f"Passed: {results['passed']}")
    print(f"Failed: {results['failed']}")
    print(f"Errors: {results['errors']}")
    print(f"Success Rate: {success_rate:.1f}%")
    
    if results["failed"] > 0 or results["errors"] > 0:
        print(f"\nFAILED/ERROR TESTS:")
        for detail in results["details"]:
            if detail["status"] in ["FAIL", "ERROR"]:
                test = detail["test"]
                print(f"  {detail['status']:<6} {test.test_name} ({test.r_package})")
    
    return results


if __name__ == "__main__":
    # Run validation when script is executed directly
    results = run_r_validation_suite(verbose=True)
    
    # Exit with error code if any tests failed
    exit_code = 1 if (results["failed"] > 0 or results["errors"] > 0) else 0
    exit(exit_code)