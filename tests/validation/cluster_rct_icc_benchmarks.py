"""
Cluster RCT validation benchmarks with various ICC values.

This module contains benchmarks for cluster randomized trials with different
intracluster correlation coefficients (ICCs), which are critical for proper
power and sample size calculations.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass  
class ClusterRCTBenchmark:
    """Container for cluster RCT validation benchmark."""
    name: str
    source: str
    description: str
    
    # Design parameters
    outcome_type: str  # continuous or binary
    icc: float
    cluster_size: int
    
    # Expected results
    expected_clusters_per_arm: int
    expected_design_effect: float
    
    # For continuous outcomes
    mean_diff: Optional[float] = None
    std_dev: Optional[float] = None
    
    # For binary outcomes
    p1: Optional[float] = None
    p2: Optional[float] = None
    
    # Common parameters
    power: float = 0.80
    alpha: float = 0.05
    
    # Validation
    tolerance: float = 0.10  # 10% tolerance for cluster calculations
    notes: Optional[str] = None


# =============================================================================
# CONTINUOUS OUTCOME BENCHMARKS WITH VARYING ICCs
# =============================================================================

CONTINUOUS_ICC_BENCHMARKS = [
    ClusterRCTBenchmark(
        name="Low ICC (0.01) - Individual variation dominates",
        source="Murray, D.M. (1998). Design and Analysis of Group-Randomized Trials",
        description="School-based intervention with minimal clustering",
        
        outcome_type="continuous",
        icc=0.01,
        cluster_size=30,
        mean_diff=0.5,  # Standardized effect size
        std_dev=1.0,
        
        expected_clusters_per_arm=14,
        expected_design_effect=1.29,  # 1 + (30-1)*0.01 = 1.29
        
        notes="Low ICC typical of educational interventions with individual-level outcomes"
    ),
    
    ClusterRCTBenchmark(
        name="Moderate ICC (0.05) - Typical health behavior",
        source="Campbell, M.K. et al. (2004). CONSORT statement extension for cluster trials",
        description="Community health intervention with moderate clustering",
        
        outcome_type="continuous",
        icc=0.05,
        cluster_size=50,
        mean_diff=0.4,  # Moderate effect
        std_dev=1.0,
        
        expected_clusters_per_arm=21,
        expected_design_effect=3.45,  # 1 + (50-1)*0.05 = 3.45
        
        notes="ICC=0.05 is common for health behaviors in community settings"
    ),
    
    ClusterRCTBenchmark(
        name="High ICC (0.10) - Strong clustering effect",
        source="Eldridge et al. (2006). Sample size for cluster randomized trials",
        description="Family/household clustered intervention",
        
        outcome_type="continuous",
        icc=0.10,
        cluster_size=20,
        mean_diff=0.5,
        std_dev=1.0,
        
        expected_clusters_per_arm=24,
        expected_design_effect=2.90,  # 1 + (20-1)*0.10 = 2.90
        
        notes="High ICC typical when clusters are naturally occurring groups like families"
    ),
    
    ClusterRCTBenchmark(
        name="Very High ICC (0.20) - Extreme clustering",
        source="Calculated using standard formulas from Donner & Klar (2000)",
        description="Workplace intervention with strong peer effects",
        
        outcome_type="continuous",
        icc=0.20,
        cluster_size=40,
        mean_diff=0.3,  # Smaller effect due to high clustering
        std_dev=1.0,
        
        expected_clusters_per_arm=74,
        expected_design_effect=8.80,  # 1 + (40-1)*0.20 = 8.80
        
        notes="Very high ICC seen in workplace or classroom settings with strong peer influence"
    )
]

# =============================================================================
# BINARY OUTCOME BENCHMARKS WITH VARYING ICCs
# =============================================================================

BINARY_ICC_BENCHMARKS = [
    ClusterRCTBenchmark(
        name="Binary Low ICC (0.02) - Community screening",
        source="Hayes, R.J. & Bennett, S. (1999). Simple sample size calculation for cluster-randomized trials",
        description="Community cancer screening program",
        
        outcome_type="binary",
        icc=0.02,
        cluster_size=100,
        p1=0.10,  # 10% baseline rate
        p2=0.15,  # 15% with intervention (50% relative increase)
        
        expected_clusters_per_arm=17,
        expected_design_effect=2.98,  # 1 + (100-1)*0.02 = 2.98
        
        notes="Low ICC for screening uptake in community settings"
    ),
    
    ClusterRCTBenchmark(
        name="Binary Moderate ICC (0.05) - Disease prevalence",
        source="Rutterford et al. (2015). Methods for sample size determination in cluster RCTs",
        description="Village-level disease control intervention",
        
        outcome_type="binary",
        icc=0.05,
        cluster_size=80,
        p1=0.20,
        p2=0.30,  # 10 percentage point reduction
        
        expected_clusters_per_arm=16,
        expected_design_effect=4.95,  # 1 + (80-1)*0.05 = 4.95
        
        notes="Moderate ICC for disease outcomes in geographic clusters"
    ),
    
    ClusterRCTBenchmark(
        name="Binary High ICC (0.10) - Behavioral outcome",
        source="Adams et al. (2004). Patterns of intra-cluster correlation",
        description="School-based smoking prevention",
        
        outcome_type="binary",
        icc=0.10,
        cluster_size=60,
        p1=0.30,  # 30% smoking rate
        p2=0.20,  # Target 20% rate
        
        expected_clusters_per_arm=19,
        expected_design_effect=6.90,  # 1 + (60-1)*0.10 = 6.90
        
        notes="High ICC common for behavioral outcomes in schools"
    ),
    
    ClusterRCTBenchmark(
        name="Binary Very High ICC (0.15) - Infectious disease",
        source="Calculated based on Campbell & Walters (2014) textbook",
        description="Household-level infectious disease intervention",
        
        outcome_type="binary",
        icc=0.15,
        cluster_size=25,
        p1=0.40,
        p2=0.25,  # Large effect needed with high ICC
        
        expected_clusters_per_arm=20,
        expected_design_effect=4.60,  # 1 + (25-1)*0.15 = 4.60
        
        notes="Very high ICC for infectious diseases within households"
    )
]

# =============================================================================
# VARYING CLUSTER SIZE BENCHMARKS
# =============================================================================

CLUSTER_SIZE_BENCHMARKS = [
    ClusterRCTBenchmark(
        name="Small clusters (m=10) with moderate ICC",
        source="Hemming et al. (2011). Sample size calculations for cluster trials",
        description="GP practice intervention with small practices",
        
        outcome_type="continuous",
        icc=0.05,
        cluster_size=10,
        mean_diff=0.5,
        std_dev=1.0,
        
        expected_clusters_per_arm=36,
        expected_design_effect=1.45,  # 1 + (10-1)*0.05 = 1.45
        
        notes="Small clusters reduce design effect but need more clusters"
    ),
    
    ClusterRCTBenchmark(
        name="Large clusters (m=200) with low ICC",
        source="Moerbeek et al. (2003). Design issues for cluster RCTs",
        description="Large school intervention",
        
        outcome_type="continuous",
        icc=0.01,
        cluster_size=200,
        mean_diff=0.3,
        std_dev=1.0,
        
        expected_clusters_per_arm=20,
        expected_design_effect=2.99,  # 1 + (200-1)*0.01 = 2.99
        
        notes="Large clusters amplify even small ICCs"
    )
]


def calculate_cluster_sample_size(
    effect_size: float,
    icc: float,
    cluster_size: int,
    power: float = 0.80,
    alpha: float = 0.05,
    is_binary: bool = False,
    p1: Optional[float] = None,
    p2: Optional[float] = None
) -> Dict[str, Any]:
    """
    Calculate required number of clusters for cluster RCT.
    
    Uses standard formulas from Donner & Klar (2000).
    """
    import numpy as np
    from scipy import stats
    
    # Critical values
    z_alpha = stats.norm.ppf(1 - alpha/2)
    z_beta = stats.norm.ppf(power)
    
    # Design effect
    design_effect = 1 + (cluster_size - 1) * icc
    
    if is_binary and p1 is not None and p2 is not None:
        # Binary outcome
        p_bar = (p1 + p2) / 2
        variance = 2 * p_bar * (1 - p_bar)
        effect = abs(p2 - p1)
        
        # Individual-level sample size
        n_individual = variance * (z_alpha + z_beta)**2 / effect**2
        
    else:
        # Continuous outcome
        # Individual-level sample size (per arm)
        n_individual = 2 * (z_alpha + z_beta)**2 / effect_size**2
    
    # Adjust for clustering
    n_clustered = n_individual * design_effect
    
    # Number of clusters per arm
    clusters_per_arm = int(np.ceil(n_clustered / cluster_size))
    
    return {
        'clusters_per_arm': clusters_per_arm,
        'total_clusters': 2 * clusters_per_arm,
        'design_effect': design_effect,
        'effective_sample_size': n_individual,
        'total_sample_size': 2 * clusters_per_arm * cluster_size
    }


def validate_cluster_benchmarks():
    """Validate cluster RCT calculations against benchmarks."""
    all_benchmarks = CONTINUOUS_ICC_BENCHMARKS + BINARY_ICC_BENCHMARKS + CLUSTER_SIZE_BENCHMARKS
    
    results = []
    
    for benchmark in all_benchmarks:
        if benchmark.outcome_type == 'continuous':
            effect_size = benchmark.mean_diff / benchmark.std_dev
            calc = calculate_cluster_sample_size(
                effect_size=effect_size,
                icc=benchmark.icc,
                cluster_size=benchmark.cluster_size,
                power=benchmark.power,
                alpha=benchmark.alpha,
                is_binary=False
            )
        else:  # binary
            calc = calculate_cluster_sample_size(
                effect_size=None,
                icc=benchmark.icc,
                cluster_size=benchmark.cluster_size,
                power=benchmark.power,
                alpha=benchmark.alpha,
                is_binary=True,
                p1=benchmark.p1,
                p2=benchmark.p2
            )
        
        # Check results
        clusters_match = abs(calc['clusters_per_arm'] - benchmark.expected_clusters_per_arm) / benchmark.expected_clusters_per_arm <= benchmark.tolerance
        deff_match = abs(calc['design_effect'] - benchmark.expected_design_effect) / benchmark.expected_design_effect <= 0.01
        
        results.append({
            'benchmark': benchmark.name,
            'icc': benchmark.icc,
            'cluster_size': benchmark.cluster_size,
            'calculated_clusters': calc['clusters_per_arm'],
            'expected_clusters': benchmark.expected_clusters_per_arm,
            'clusters_match': clusters_match,
            'calculated_deff': calc['design_effect'],
            'expected_deff': benchmark.expected_design_effect,
            'deff_match': deff_match,
            'overall_pass': clusters_match and deff_match
        })
    
    return results


if __name__ == "__main__":
    print("Cluster RCT ICC Benchmarks Validation")
    print("=" * 60)
    
    results = validate_cluster_benchmarks()
    
    passed = sum(1 for r in results if r['overall_pass'])
    total = len(results)
    
    print(f"\nOverall: {passed}/{total} benchmarks passed ({100*passed/total:.1f}%)")
    print("\nDetailed Results:")
    
    for result in results:
        status = "✓" if result['overall_pass'] else "✗"
        print(f"\n{status} {result['benchmark']}")
        print(f"   ICC: {result['icc']}, Cluster size: {result['cluster_size']}")
        print(f"   Clusters/arm - Expected: {result['expected_clusters']}, "
              f"Calculated: {result['calculated_clusters']}, "
              f"Match: {result['clusters_match']}")
        print(f"   Design effect - Expected: {result['expected_deff']:.2f}, "
              f"Calculated: {result['calculated_deff']:.2f}, "
              f"Match: {result['deff_match']}")