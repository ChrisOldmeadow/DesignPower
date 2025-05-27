"""Test script for cluster RCT with binary outcomes.

This script demonstrates the enhanced functionality for cluster randomized
controlled trials with binary outcomes, including:
1. Support for unequal cluster sizes
2. Different effect measures (risk difference, risk ratio, odds ratio)
3. Validation for small numbers of clusters
"""

import json
import numpy as np
from core.designs.cluster_rct.analytical_binary import (
    power_binary, 
    sample_size_binary, 
    min_detectable_effect_binary
)
from core.designs.cluster_rct.simulation_binary import (
    power_binary_sim,
    sample_size_binary_sim,
    min_detectable_effect_binary_sim
)

def print_results(title, results):
    """Format and print results dictionary."""
    print(f"\n{title}")
    print("=" * len(title))
    print(json.dumps(results, indent=2))
    if "warnings" in results:
        print("\nWARNINGS:")
        for warning in results["warnings"]:
            print(f"- {warning}")

def main():
    # Define common parameters
    p1 = 0.3  # Control group proportion
    p2 = 0.45  # Intervention group proportion
    icc = 0.05  # Intracluster correlation coefficient
    cluster_size = 30  # Average cluster size
    
    print("\n\n===== ANALYTICAL METHODS =====\n")
    
    # Test 1: Power calculation with equal cluster sizes
    print("\nTEST 1: Power calculation with equal cluster sizes")
    power_equal = power_binary(
        n_clusters=15, 
        cluster_size=cluster_size, 
        icc=icc, 
        p1=p1, 
        p2=p2
    )
    print_results("Power (Equal Cluster Sizes)", power_equal)
    
    # Test 2: Power calculation with unequal cluster sizes (CV = 0.5)
    print("\nTEST 2: Power calculation with unequal cluster sizes (CV = 0.5)")
    power_unequal = power_binary(
        n_clusters=15, 
        cluster_size=cluster_size, 
        icc=icc, 
        p1=p1, 
        p2=p2, 
        cv_cluster_size=0.5
    )
    print_results("Power (Unequal Cluster Sizes, CV = 0.5)", power_unequal)
    
    # Test 3: Sample size calculation with risk difference
    print("\nTEST 3: Sample size calculation with risk difference")
    sample_size_rd = sample_size_binary(
        p1=p1,
        effect_measure="risk_difference",
        effect_value=0.15,
        icc=icc,
        cluster_size=cluster_size
    )
    print_results("Sample Size (Risk Difference = 0.15)", sample_size_rd)
    
    # Test 4: Sample size calculation with risk ratio
    print("\nTEST 4: Sample size calculation with risk ratio")
    sample_size_rr = sample_size_binary(
        p1=p1,
        effect_measure="risk_ratio",
        effect_value=1.5,
        icc=icc,
        cluster_size=cluster_size
    )
    print_results("Sample Size (Risk Ratio = 1.5)", sample_size_rr)
    
    # Test 5: Minimum detectable effect with small number of clusters
    print("\nTEST 5: Minimum detectable effect with small number of clusters")
    mde_small = min_detectable_effect_binary(
        n_clusters=5,  # Small number of clusters will trigger warnings
        cluster_size=cluster_size,
        icc=icc,
        p1=p1
    )
    print_results("MDE (Small Number of Clusters)", mde_small)
    
    # Test 6: MDE with different effect measure (odds ratio)
    print("\nTEST 6: MDE with different effect measure (odds ratio)")
    mde_or = min_detectable_effect_binary(
        n_clusters=15,
        cluster_size=cluster_size,
        icc=icc,
        p1=p1,
        effect_measure="odds_ratio"
    )
    print_results("MDE (Odds Ratio)", mde_or)
    
    print("\n\n===== SIMULATION METHODS =====\n")
    
    # Test 7: Power calculation with simulation (equal cluster sizes)
    print("\nTEST 7: Power calculation with simulation (equal cluster sizes)")
    power_sim = power_binary_sim(
        n_clusters=15,
        cluster_size=cluster_size,
        icc=icc,
        p1=p1,
        p2=p2,
        nsim=500  # Reduced for faster execution
    )
    print_results("Power Simulation (Equal Cluster Sizes)", power_sim)
    
    # Test 8: Power calculation with simulation (unequal cluster sizes)
    print("\nTEST 8: Power calculation with simulation (unequal cluster sizes)")
    # Generate some unequal cluster sizes with CV ≈ 0.5
    np.random.seed(42)
    cluster_sizes = np.random.normal(cluster_size, cluster_size * 0.5, size=30)
    cluster_sizes = np.clip(cluster_sizes, 10, 100).astype(int)  # Ensure reasonable values
    
    power_sim_unequal = power_binary_sim(
        n_clusters=15,
        cluster_size=cluster_size,
        icc=icc,
        p1=p1,
        p2=p2,
        nsim=500,
        cluster_sizes=cluster_sizes
    )
    print_results("Power Simulation (Unequal Cluster Sizes)", power_sim_unequal)
    
    # Compare analytical vs simulation results
    print("\n\nCOMPARISON: Analytical vs Simulation")
    print("------------------------------------")
    print(f"Equal clusters power (analytical):   {power_equal['power']:.4f}")
    print(f"Equal clusters power (simulation):   {power_sim['power']:.4f}")
    print(f"Unequal clusters power (analytical): {power_unequal['power']:.4f}")
    print(f"Unequal clusters power (simulation): {power_sim_unequal['power']:.4f}")
    
    # Effect of cluster size variation on design effect
    print("\n\nEFFECT OF CLUSTER SIZE VARIATION ON DESIGN EFFECT")
    print("------------------------------------------------")
    cvs = [0.0, 0.1, 0.2, 0.5, 1.0]
    for cv in cvs:
        result = power_binary(
            n_clusters=15, 
            cluster_size=cluster_size, 
            icc=icc, 
            p1=p1, 
            p2=p2, 
            cv_cluster_size=cv
        )
        print(f"CV = {cv:.1f}: Design Effect = {result['design_effect']:.4f}, Power = {result['power']:.4f}")

if __name__ == "__main__":
    main()
