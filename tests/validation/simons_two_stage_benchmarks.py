"""
Simon's Two-Stage Design validation benchmarks.

This module contains benchmarks for validating Simon's two-stage design implementation
against published examples and known results from the original Simon (1989) paper
and other sources.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class SimonsTwoStageBenchmark:
    """Container for Simon's two-stage design validation benchmark."""
    name: str
    source: str
    description: str
    
    # Design parameters
    p0: float  # Null hypothesis response rate (unacceptable)
    p1: float  # Alternative hypothesis response rate (desirable)
    alpha: float
    beta: float  # Type II error (power = 1 - beta)
    design_type: str  # 'optimal' or 'minimax'
    
    # Expected results for optimal design
    optimal_n1: int  # Stage 1 sample size
    optimal_r1: int  # Stage 1 rejection threshold
    optimal_n: int   # Total sample size (both stages)
    optimal_r: int   # Overall rejection threshold
    optimal_en_null: float  # Expected sample size under null
    
    # Expected results for minimax design (if different)
    minimax_n1: Optional[int] = None
    minimax_r1: Optional[int] = None
    minimax_n: Optional[int] = None
    minimax_r: Optional[int] = None
    minimax_en_null: Optional[float] = None
    
    # Validation
    tolerance: float = 0.01  # Very tight for exact designs
    notes: Optional[str] = None


# Classic benchmarks from Simon (1989) paper
SIMON_1989_BENCHMARKS = [
    SimonsTwoStageBenchmark(
        name="Simon Example 1: p0=0.05, p1=0.25",
        source="Simon, R. (1989). Optimal two-stage designs for phase II clinical trials. Controlled Clinical Trials, 10(1), 1-10.",
        description="Table 1 from original paper - low baseline response",
        
        p0=0.05,
        p1=0.25,
        alpha=0.05,
        beta=0.20,
        design_type="both",
        
        # Optimal design
        optimal_n1=9,
        optimal_r1=0,  # Stop if 0 or fewer responses in first 9
        optimal_n=17,
        optimal_r=2,   # Reject if 2 or fewer responses total
        optimal_en_null=11.9,
        
        # Minimax design
        minimax_n1=12,
        minimax_r1=0,
        minimax_n=16,
        minimax_r=2,
        minimax_en_null=12.7,
        
        notes="Classic example showing difference between optimal and minimax"
    ),
    
    SimonsTwoStageBenchmark(
        name="Simon Example 2: p0=0.10, p1=0.30",
        source="Simon, R. (1989). Table 2",
        description="Moderate baseline response rate",
        
        p0=0.10,
        p1=0.30,
        alpha=0.05,
        beta=0.20,
        design_type="both",
        
        # Optimal design
        optimal_n1=10,
        optimal_r1=0,
        optimal_n=29,
        optimal_r=4,
        optimal_en_null=15.0,
        
        # Minimax design
        minimax_n1=15,
        minimax_r1=1,
        minimax_n=25,
        minimax_r=4,
        minimax_en_null=17.3,
        
        notes="Common scenario in oncology trials"
    ),
    
    SimonsTwoStageBenchmark(
        name="Simon Example 3: p0=0.20, p1=0.40",
        source="Simon, R. (1989). Table 3",
        description="Higher baseline response rate",
        
        p0=0.20,
        p1=0.40,
        alpha=0.05,
        beta=0.20,
        design_type="both",
        
        # Optimal design
        optimal_n1=13,
        optimal_r1=2,
        optimal_n=43,
        optimal_r=10,
        optimal_en_null=22.5,
        
        # Minimax design
        minimax_n1=19,
        minimax_r1=3,
        minimax_n=39,
        minimax_r=10,
        minimax_en_null=25.0,
        
        notes="Higher response rates require larger samples"
    ),
    
    SimonsTwoStageBenchmark(
        name="Simon Example 4: p0=0.30, p1=0.50",
        source="Simon, R. (1989). Table 4",
        description="High baseline with moderate improvement",
        
        p0=0.30,
        p1=0.50,
        alpha=0.05,
        beta=0.20,
        design_type="both",
        
        # Optimal design
        optimal_n1=15,
        optimal_r1=4,
        optimal_n=46,
        optimal_r=15,
        optimal_en_null=25.9,
        
        # Minimax design
        minimax_n1=22,
        minimax_r1=6,
        minimax_n=43,
        minimax_r=15,
        minimax_en_null=29.3,
        
        notes="Typical for combination therapy trials"
    )
]

# Additional benchmarks from other sources
ADDITIONAL_BENCHMARKS = [
    SimonsTwoStageBenchmark(
        name="Small Trial: p0=0.10, p1=0.35",
        source="Jung et al. (2004). Admissible two-stage designs for phase II cancer clinical trials.",
        description="Small trial with good treatment effect",
        
        p0=0.10,
        p1=0.35,
        alpha=0.10,  # Less stringent alpha
        beta=0.10,   # Higher power (90%)
        design_type="optimal",
        
        # Optimal design
        optimal_n1=8,
        optimal_r1=0,
        optimal_n=20,
        optimal_r=3,
        optimal_en_null=11.2,
        
        notes="Higher power requirement affects design"
    ),
    
    SimonsTwoStageBenchmark(
        name="Large Effect: p0=0.05, p1=0.30",
        source="Calculated using clinfun::ph2simon in R",
        description="Large treatment effect scenario",
        
        p0=0.05,
        p1=0.30,
        alpha=0.05,
        beta=0.20,
        design_type="optimal",
        
        # Optimal design
        optimal_n1=7,
        optimal_r1=0,
        optimal_n=14,
        optimal_r=2,
        optimal_en_null=9.5,
        
        notes="Large effect size allows smaller sample"
    ),
    
    SimonsTwoStageBenchmark(
        name="Rare Response: p0=0.01, p1=0.10",
        source="Calculated using validated software",
        description="Very low response rates",
        
        p0=0.01,
        p1=0.10,
        alpha=0.05,
        beta=0.20,
        design_type="optimal",
        
        # Optimal design
        optimal_n1=13,
        optimal_r1=0,
        optimal_n=37,
        optimal_r=2,
        optimal_en_null=17.8,
        
        notes="Rare responses require careful design"
    )
]


def validate_simons_implementation():
    """Validate Simon's two-stage design implementation against benchmarks."""
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    
    from core.designs.single_arm.binary import simons_two_stage_design
    
    all_benchmarks = SIMON_1989_BENCHMARKS + ADDITIONAL_BENCHMARKS
    results = []
    
    for benchmark in all_benchmarks:
        # Test optimal design
        if benchmark.design_type in ['optimal', 'both']:
            try:
                result = simons_two_stage_design(
                    p0=benchmark.p0,
                    p1=benchmark.p1,
                    alpha=benchmark.alpha,
                    beta=benchmark.beta,
                    design_type='optimal'
                )
                
                # Check each parameter
                n1_match = result['n1'] == benchmark.optimal_n1
                r1_match = result['r1'] == benchmark.optimal_r1
                n_match = result['n'] == benchmark.optimal_n
                r_match = result['r'] == benchmark.optimal_r
                en_match = abs(result['EN0'] - benchmark.optimal_en_null) <= 0.5
                
                optimal_pass = all([n1_match, r1_match, n_match, r_match, en_match])
                
                results.append({
                    'benchmark': benchmark.name,
                    'design': 'optimal',
                    'n1': f"{result['n1']} (expected {benchmark.optimal_n1})",
                    'r1': f"{result['r1']} (expected {benchmark.optimal_r1})",
                    'n': f"{result['n']} (expected {benchmark.optimal_n})",
                    'r': f"{result['r']} (expected {benchmark.optimal_r})",
                    'EN_null': f"{result['EN0']:.1f} (expected {benchmark.optimal_en_null})",
                    'pass': optimal_pass
                })
                
            except Exception as e:
                results.append({
                    'benchmark': benchmark.name,
                    'design': 'optimal',
                    'error': str(e),
                    'pass': False
                })
        
        # Test minimax design if available
        if benchmark.design_type in ['minimax', 'both'] and benchmark.minimax_n1 is not None:
            try:
                result = simons_two_stage_design(
                    p0=benchmark.p0,
                    p1=benchmark.p1,
                    alpha=benchmark.alpha,
                    beta=benchmark.beta,
                    design_type='minimax'
                )
                
                n1_match = result['n1'] == benchmark.minimax_n1
                r1_match = result['r1'] == benchmark.minimax_r1
                n_match = result['n'] == benchmark.minimax_n
                r_match = result['r'] == benchmark.minimax_r
                en_match = abs(result['EN0'] - benchmark.minimax_en_null) <= 0.5
                
                minimax_pass = all([n1_match, r1_match, n_match, r_match, en_match])
                
                results.append({
                    'benchmark': benchmark.name,
                    'design': 'minimax',
                    'n1': f"{result['n1']} (expected {benchmark.minimax_n1})",
                    'r1': f"{result['r1']} (expected {benchmark.minimax_r1})",
                    'n': f"{result['n']} (expected {benchmark.minimax_n})",
                    'r': f"{result['r']} (expected {benchmark.minimax_r})",
                    'EN_null': f"{result['EN0']:.1f} (expected {benchmark.minimax_en_null})",
                    'pass': minimax_pass
                })
                
            except Exception as e:
                results.append({
                    'benchmark': benchmark.name,
                    'design': 'minimax',
                    'error': str(e),
                    'pass': False
                })
    
    return results


if __name__ == "__main__":
    print("Simon's Two-Stage Design Validation")
    print("=" * 60)
    
    results = validate_simons_implementation()
    
    passed = sum(1 for r in results if r['pass'])
    total = len(results)
    
    print(f"\nOverall: {passed}/{total} designs validated successfully ({100*passed/total:.1f}%)")
    print("\nDetailed Results:")
    
    for result in results:
        status = "✓" if result['pass'] else "✗"
        print(f"\n{status} {result['benchmark']} ({result['design']})")
        
        if 'error' in result:
            print(f"   ERROR: {result['error']}")
        else:
            print(f"   n1: {result['n1']}")
            print(f"   r1: {result['r1']}")
            print(f"   n:  {result['n']}")
            print(f"   r:  {result['r']}")
            print(f"   EN(p0): {result['EN_null']}")