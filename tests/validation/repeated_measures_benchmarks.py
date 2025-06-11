"""
Repeated Measures Design validation benchmarks.

This module contains benchmarks for validating repeated measures design implementation
against published examples and theoretical calculations.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
import math


@dataclass
class RepeatedMeasuresBenchmark:
    """Container for repeated measures design validation benchmark."""
    name: str
    source: str
    description: str
    
    # Design parameters
    delta: float  # Effect size (difference in means)
    std_dev: float  # Standard deviation of outcome
    correlation: float  # Baseline-followup correlation
    alpha: float  # Significance level
    power: float  # Desired power
    method: str  # "change_score" or "ancova"
    
    # Expected results
    expected_n1: int  # Expected sample size group 1
    expected_n2: int  # Expected sample size group 2  
    expected_total_n: int  # Expected total sample size
    
    # Validation settings
    tolerance: float = 0.05  # 5% tolerance for sample sizes
    notes: Optional[str] = None


# Benchmarks from Vickers (2001) and other sources
REPEATED_MEASURES_BENCHMARKS = [
    RepeatedMeasuresBenchmark(
        name="Vickers Example 1: Medium effect, high correlation",
        source="Vickers, A.J. (2001). The use of percentage change from baseline as an outcome in a controlled trial is statistically inefficient: a simulation study. BMC Med Res Methodol, 1:6.",
        description="Pre-post design with high baseline correlation - ANCOVA vs change score",
        
        delta=0.5,
        std_dev=1.0,
        correlation=0.8,
        alpha=0.05,
        power=0.8,
        method="ancova",
        
        # ANCOVA should be much more efficient with high correlation
        expected_n1=23,  # Calculated: 2 * (1.96+0.84)^2 * (1-0.8^2) / 0.5^2 ≈ 23
        expected_n2=23,
        expected_total_n=46,
        
        notes="High correlation makes ANCOVA very efficient"
    ),
    
    RepeatedMeasuresBenchmark(
        name="Vickers Example 1: Same parameters, change score method",
        source="Vickers, A.J. (2001). BMC Med Res Methodol, 1:6.",
        description="Same as above but using change score analysis",
        
        delta=0.5,
        std_dev=1.0,
        correlation=0.8,
        alpha=0.05,
        power=0.8,
        method="change_score",
        
        # Change score: 2 * (1.96+0.84)^2 * sqrt(2*(1-0.8))^2 / 0.5^2 ≈ 26
        expected_n1=26,
        expected_n2=26,
        expected_total_n=52,
        
        notes="Change score less efficient than ANCOVA when r=0.8"
    ),
    
    RepeatedMeasuresBenchmark(
        name="Low correlation scenario: ANCOVA advantage minimal",
        source="Theoretical calculation based on Vickers (2001) principles",
        description="Low baseline correlation - methods should be similar",
        
        delta=0.3,
        std_dev=1.0,
        correlation=0.2,
        alpha=0.05,
        power=0.8,
        method="ancova",
        
        # ANCOVA: (1.96+0.84)^2 * (1-0.2^2) * 2 / 0.3^2 ≈ 172
        expected_n1=172,
        expected_n2=172,
        expected_total_n=344,
        
        notes="Low correlation: minimal ANCOVA advantage"
    ),
    
    RepeatedMeasuresBenchmark(
        name="Low correlation scenario: Change score method",
        source="Theoretical calculation based on Vickers (2001) principles",
        description="Same parameters with change score analysis",
        
        delta=0.3,
        std_dev=1.0,
        correlation=0.2,
        alpha=0.05,
        power=0.8,
        method="change_score",
        
        # Change score: (1.96+0.84)^2 * 2*(1-0.2) * 2 / 0.3^2 ≈ 280
        expected_n1=280,
        expected_n2=280,
        expected_total_n=560,
        
        notes="Change score slightly less efficient even with low correlation"
    ),
    
    RepeatedMeasuresBenchmark(
        name="Medium correlation crossover point",
        source="Van Breukelen, G.J. (2006). ANCOVA versus change from baseline had more power in randomized trials and more bias in nonrandomized trials. J Clin Epidemiol, 59(9):920-5.",
        description="Moderate correlation where ANCOVA shows clear advantage",
        
        delta=0.4,
        std_dev=1.2,
        correlation=0.5,
        alpha=0.05,
        power=0.9,  # Higher power requirement
        method="ancova",
        
        # ANCOVA: (1.96+1.28)^2 * (1-0.5^2) * 2 / 0.4^2 ≈ 142
        expected_n1=142,
        expected_n2=142,
        expected_total_n=284,
        
        notes="Moderate correlation, high power requirement"
    ),
    
    RepeatedMeasuresBenchmark(
        name="Large effect size, crossover design",
        source="Theoretical calculation following Jones & Kenward (2003) principles",
        description="Large effect with moderate correlation",
        
        delta=0.8,  # Large effect size
        std_dev=1.0,
        correlation=0.6,
        alpha=0.05,
        power=0.8,
        method="ancova",
        
        # ANCOVA: (1.96+0.84)^2 * (1-0.6^2) * 2 / 0.8^2 ≈ 16
        expected_n1=16,
        expected_n2=16,
        expected_total_n=32,
        
        notes="Large effect size reduces required sample size substantially"
    )
]


def validate_repeated_measures():
    """Validate repeated measures implementation against benchmarks."""
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    
    from core.designs.parallel.analytical.repeated_measures import sample_size_repeated_measures
    
    results = []
    
    for benchmark in REPEATED_MEASURES_BENCHMARKS:
        try:
            result = sample_size_repeated_measures(
                delta=benchmark.delta,
                std_dev=benchmark.std_dev,
                correlation=benchmark.correlation,
                power=benchmark.power,
                alpha=benchmark.alpha,
                method=benchmark.method
            )
            
            # Check results against expected values
            n1_error = abs(result['n1'] - benchmark.expected_n1) / benchmark.expected_n1
            n2_error = abs(result['n2'] - benchmark.expected_n2) / benchmark.expected_n2
            total_error = abs(result['total_n'] - benchmark.expected_total_n) / benchmark.expected_total_n
            
            n1_pass = n1_error <= benchmark.tolerance
            n2_pass = n2_error <= benchmark.tolerance
            total_pass = total_error <= benchmark.tolerance
            
            overall_pass = all([n1_pass, n2_pass, total_pass])
            
            results.append({
                'benchmark': benchmark.name,
                'method': benchmark.method,
                'correlation': benchmark.correlation,
                'n1': f"{result['n1']} (expected {benchmark.expected_n1})",
                'n2': f"{result['n2']} (expected {benchmark.expected_n2})",
                'total_n': f"{result['total_n']} (expected {benchmark.expected_total_n})",
                'n1_error': f"{n1_error:.1%}",
                'total_error': f"{total_error:.1%}",
                'pass': overall_pass
            })
            
        except Exception as e:
            results.append({
                'benchmark': benchmark.name,
                'method': benchmark.method,
                'error': str(e),
                'pass': False
            })
    
    return results


if __name__ == "__main__":
    print("Repeated Measures Design Validation")
    print("=" * 60)
    
    results = validate_repeated_measures()
    
    passed = sum(1 for r in results if r['pass'])
    total = len(results)
    
    print(f"\nOverall: {passed}/{total} designs validated successfully ({100*passed/total:.1f}%)")
    print("\nDetailed Results:")
    
    for result in results:
        status = "✓" if result['pass'] else "✗"
        print(f"\n{status} {result['benchmark']}")
        print(f"   Method: {result['method']}")
        
        if 'error' in result:
            print(f"   ERROR: {result['error']}")
        else:
            print(f"   n1: {result['n1']} (error: {result['n1_error']})")
            print(f"   n2: {result['n2']}")
            print(f"   Total: {result['total_n']} (error: {result['total_error']})")
            
            if 'correlation' in result:
                print(f"   Correlation: {result['correlation']}")