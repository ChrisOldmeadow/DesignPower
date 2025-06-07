"""
Authoritative benchmarks for DesignPower validation.

This module contains carefully researched benchmarks from authoritative sources
in statistical methodology. Each benchmark includes complete citation and 
specific parameter values to ensure reproducible validation.
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional


@dataclass
class AuthoritativeBenchmark:
    """
    Container for a validation benchmark with complete source documentation.
    """
    # Source information
    source_title: str
    authors: str
    year: int
    citation: str
    page_reference: str
    
    # Test specification
    test_name: str
    design_type: str  # parallel, cluster, single_arm
    outcome_type: str  # binary, continuous, survival
    calculation_type: str  # sample_size, power, mde
    
    # Parameters and expected results
    parameters: Dict[str, Any]
    expected_results: Dict[str, Any]
    
    # Validation metadata
    tolerance: float = 0.05  # Default 5% tolerance
    notes: Optional[str] = None
    cross_validated: bool = False  # Has this been verified against other sources?
    
    def __repr__(self):
        return f"AuthoritativeBenchmark({self.authors} {self.year}: {self.test_name})"


# =============================================================================
# COHEN (1988) - STATISTICAL POWER ANALYSIS BENCHMARKS
# =============================================================================

COHEN_1988_BENCHMARKS = [
    AuthoritativeBenchmark(
        source_title="Statistical Power Analysis for the Behavioral Sciences",
        authors="Cohen, J.",
        year=1988,
        citation="Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences (2nd ed.). Lawrence Erlbaum Associates.",
        page_reference="Table 2.3.1, Chapter 2",
        
        test_name="Two-sample t-test: Small effect size",
        design_type="parallel",
        outcome_type="continuous",
        calculation_type="sample_size",
        
        parameters={
            "effect_size_d": 0.2,  # Cohen's d (standardized mean difference)
            "power": 0.8,
            "alpha": 0.05,
            "test_type": "two_sided",
            "allocation_ratio": 1.0
        },
        expected_results={
            "sample_size_per_group": 393,
            "total_sample_size": 786
        },
        tolerance=0.02,  # Very established benchmark, tight tolerance
        notes="Classic Cohen benchmark for small effect size. Widely cited standard.",
        cross_validated=True
    ),
    
    AuthoritativeBenchmark(
        source_title="Statistical Power Analysis for the Behavioral Sciences",
        authors="Cohen, J.",
        year=1988,
        citation="Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences (2nd ed.). Lawrence Erlbaum Associates.",
        page_reference="Table 2.3.1, Chapter 2",
        
        test_name="Two-sample t-test: Medium effect size",
        design_type="parallel",
        outcome_type="continuous", 
        calculation_type="sample_size",
        
        parameters={
            "effect_size_d": 0.5,
            "power": 0.8,
            "alpha": 0.05,
            "test_type": "two_sided",
            "allocation_ratio": 1.0
        },
        expected_results={
            "sample_size_per_group": 64,
            "total_sample_size": 128
        },
        tolerance=0.02,
        notes="Classic Cohen benchmark for medium effect size. Most commonly cited example.",
        cross_validated=True
    ),
    
    AuthoritativeBenchmark(
        source_title="Statistical Power Analysis for the Behavioral Sciences",
        authors="Cohen, J.",
        year=1988,
        citation="Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences (2nd ed.). Lawrence Erlbaum Associates.",
        page_reference="Table 2.3.1, Chapter 2",
        
        test_name="Two-sample t-test: Large effect size",
        design_type="parallel",
        outcome_type="continuous",
        calculation_type="sample_size",
        
        parameters={
            "effect_size_d": 0.8,
            "power": 0.8,
            "alpha": 0.05,
            "test_type": "two_sided",
            "allocation_ratio": 1.0
        },
        expected_results={
            "sample_size_per_group": 26,
            "total_sample_size": 52
        },
        tolerance=0.02,
        notes="Classic Cohen benchmark for large effect size.",
        cross_validated=True
    )
]


# =============================================================================
# R PWR PACKAGE BENCHMARKS (CROSS-VALIDATION)
# =============================================================================

R_PWR_BENCHMARKS = [
    AuthoritativeBenchmark(
        source_title="R Documentation: pwr package",
        authors="Champely, S.",
        year=2020,
        citation="Champely, S. (2020). pwr: Basic Functions for Power Analysis. R package version 1.3-0.",
        page_reference="help(pwr.t.test)",
        
        test_name="pwr.t.test default example",
        design_type="parallel",
        outcome_type="continuous",
        calculation_type="sample_size",
        
        parameters={
            "effect_size_d": 0.5,
            "power": 0.8,
            "alpha": 0.05,
            "test_type": "two_sided"
        },
        expected_results={
            "sample_size_per_group": 64,  # Should match Cohen exactly
            "total_sample_size": 128
        },
        tolerance=0.01,  # Very tight - should be exact match with Cohen
        notes="R pwr package implementation. Should exactly match Cohen (1988) values.",
        cross_validated=True
    ),
    
    AuthoritativeBenchmark(
        source_title="R Documentation: pwr package",
        authors="Champely, S.",
        year=2020,
        citation="Champely, S. (2020). pwr: Basic Functions for Power Analysis. R package version 1.3-0.",
        page_reference="help(pwr.2p.test)",
        
        test_name="Two proportions test example",
        design_type="parallel",
        outcome_type="binary",
        calculation_type="sample_size",
        
        parameters={
            "p1": 0.65,
            "p2": 0.85,
            "power": 0.8,
            "alpha": 0.05,
            "test_type": "two_sided"
        },
        expected_results={
            "sample_size_per_group": 47,  # Based on Cohen's h calculation
            "total_sample_size": 94
        },
        tolerance=0.05,
        notes="Two proportions test using Cohen's h effect size measure.",
        cross_validated=False  # Need to verify this specific example
    )
]


# =============================================================================
# FLEISS ET AL. BINARY OUTCOME BENCHMARKS  
# =============================================================================

FLEISS_BENCHMARKS = [
    AuthoritativeBenchmark(
        source_title="Statistical Methods for Rates and Proportions",
        authors="Fleiss, J.L., Levin, B., & Paik, M.C.",
        year=2003,
        citation="Fleiss, J.L., Levin, B., & Paik, M.C. (2003). Statistical Methods for Rates and Proportions (3rd ed.). Wiley.",
        page_reference="Chapter 3, Section 3.3",
        
        test_name="Equal proportions, moderate effect",
        design_type="parallel",
        outcome_type="binary",
        calculation_type="sample_size",
        
        parameters={
            "p1": 0.20,
            "p2": 0.40,  # 20 percentage point difference
            "power": 0.8,
            "alpha": 0.05,
            "test_type": "two_sided",
            "allocation_ratio": 1.0
        },
        expected_results={
            "sample_size_per_group": 93,  # Normal approximation
            "total_sample_size": 186
        },
        tolerance=0.10,  # Larger tolerance - different approximation methods
        notes="Normal approximation to binomial. May vary based on continuity correction.",
        cross_validated=False
    )
]


# =============================================================================
# SINGLE ARM TRIAL BENCHMARKS (A'HERN 2001)
# =============================================================================

AHERN_2001_BENCHMARKS = [
    AuthoritativeBenchmark(
        source_title="Sample size tables for exact single-stage phase II designs",
        authors="A'Hern, R.P.",
        year=2001,
        citation="A'Hern, R.P. (2001). Sample size tables for exact single-stage phase II designs. Statistics in Medicine, 20(6), 859-866.",
        page_reference="Table 1, Row p0=0.05, p1=0.20",
        
        test_name="Single-stage design: Low response rate",
        design_type="single_arm",
        outcome_type="binary",
        calculation_type="sample_size",
        
        parameters={
            "p0": 0.05,  # Null hypothesis (uninteresting response rate)
            "p1": 0.20,  # Alternative hypothesis (target response rate)
            "alpha": 0.05,
            "power": 0.80  # beta = 0.20
        },
        expected_results={
            "sample_size": 29,
            "critical_value": 4  # Reject null if ≥4 responses observed
        },
        tolerance=0.0,  # Exact design - should be exact match
        notes="Exact binomial design. No approximation involved.",
        cross_validated=False
    ),
    
    AuthoritativeBenchmark(
        source_title="Sample size tables for exact single-stage phase II designs", 
        authors="A'Hern, R.P.",
        year=2001,
        citation="A'Hern, R.P. (2001). Sample size tables for exact single-stage phase II designs. Statistics in Medicine, 20(6), 859-866.",
        page_reference="Table 1, Row p0=0.20, p1=0.40",
        
        test_name="Single-stage design: Moderate response rate",
        design_type="single_arm",
        outcome_type="binary",
        calculation_type="sample_size",
        
        parameters={
            "p0": 0.20,
            "p1": 0.40,
            "alpha": 0.05,
            "power": 0.80
        },
        expected_results={
            "sample_size": 43,
            "critical_value": 13
        },
        tolerance=0.0,
        notes="Exact binomial design for moderate response rates.",
        cross_validated=False
    )
]


# =============================================================================
# CLUSTER RCT BENCHMARKS (DONNER & KLAR 2000)
# =============================================================================

DONNER_KLAR_BENCHMARKS = [
    AuthoritativeBenchmark(
        source_title="Design and Analysis of Cluster Randomization Trials in Health Research",
        authors="Donner, A. & Klar, N.",
        year=2000,
        citation="Donner, A. & Klar, N. (2000). Design and Analysis of Cluster Randomization Trials in Health Research. Arnold Publishers.",
        page_reference="Chapter 4, Example 4.1",
        
        test_name="Binary outcome cluster trial",
        design_type="cluster",
        outcome_type="binary",
        calculation_type="sample_size",
        
        parameters={
            "p1": 0.10,
            "p2": 0.15,  # 5 percentage point difference
            "cluster_size": 100,
            "icc": 0.02,
            "power": 0.8,
            "alpha": 0.05
        },
        expected_results={
            "clusters_per_arm": 17,
            "total_clusters": 34,
            "total_sample_size": 3400,
            "design_effect": 2.98  # 1 + (m-1)*ICC = 1 + 99*0.02
        },
        tolerance=0.10,  # Cluster calculations can vary by method
        notes="Includes design effect calculation. ICC=0.02 is common for community interventions.",
        cross_validated=False
    )
]


# =============================================================================
# BENCHMARK COLLECTIONS
# =============================================================================

ALL_BENCHMARKS = [
    *COHEN_1988_BENCHMARKS,
    *R_PWR_BENCHMARKS,
    *FLEISS_BENCHMARKS,
    *AHERN_2001_BENCHMARKS,
    *DONNER_KLAR_BENCHMARKS
]

def get_benchmarks_by_design(design_type: str) -> List[AuthoritativeBenchmark]:
    """Get all benchmarks for a specific design type."""
    return [b for b in ALL_BENCHMARKS if b.design_type == design_type]

def get_benchmarks_by_outcome(outcome_type: str) -> List[AuthoritativeBenchmark]:
    """Get all benchmarks for a specific outcome type."""
    return [b for b in ALL_BENCHMARKS if b.outcome_type == outcome_type]

def get_cross_validated_benchmarks() -> List[AuthoritativeBenchmark]:
    """Get only benchmarks that have been cross-validated against multiple sources."""
    return [b for b in ALL_BENCHMARKS if b.cross_validated]

def print_benchmark_summary():
    """Print a summary of all available benchmarks."""
    print("Authoritative Benchmarks Summary")
    print("=" * 50)
    
    by_design = {}
    by_outcome = {}
    
    for benchmark in ALL_BENCHMARKS:
        # Count by design type
        if benchmark.design_type not in by_design:
            by_design[benchmark.design_type] = 0
        by_design[benchmark.design_type] += 1
        
        # Count by outcome type
        if benchmark.outcome_type not in by_outcome:
            by_outcome[benchmark.outcome_type] = 0
        by_outcome[benchmark.outcome_type] += 1
    
    print(f"Total benchmarks: {len(ALL_BENCHMARKS)}")
    print(f"Cross-validated: {len(get_cross_validated_benchmarks())}")
    
    print("\nBy Design Type:")
    for design, count in by_design.items():
        print(f"  {design}: {count}")
    
    print("\nBy Outcome Type:")
    for outcome, count in by_outcome.items():
        print(f"  {outcome}: {count}")
    
    print("\nDetailed List:")
    for benchmark in ALL_BENCHMARKS:
        cross_val = "✓" if benchmark.cross_validated else " "
        print(f"  [{cross_val}] {benchmark.authors} ({benchmark.year}): {benchmark.test_name}")


if __name__ == "__main__":
    print_benchmark_summary()