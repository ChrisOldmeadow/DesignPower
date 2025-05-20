#!/usr/bin/env python3
"""
Test script for verifying the refactored DesignPower modules.
This script tests various outcome types (binary, continuous, survival)
with both analytical and simulation methods.
"""

import sys
import os
from pprint import pprint

# Add the project root to Python path
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import from refactored modules
from core.designs.parallel import (
    # Binary outcome functions
    sample_size_binary, power_binary, min_detectable_effect_binary,
    sample_size_binary_sim, power_binary_sim, min_detectable_effect_binary_sim,
    
    # Continuous outcome functions
    sample_size_continuous, power_continuous, min_detectable_effect_continuous,
    sample_size_continuous_sim, power_continuous_sim, min_detectable_effect_continuous_sim,
    
    # Survival outcome functions
    sample_size_survival, power_survival, min_detectable_effect_survival,
    sample_size_survival_sim, power_survival_sim, min_detectable_effect_survival_sim
)

def test_binary_outcomes():
    """Test binary outcome functions."""
    print("\n*** TESTING BINARY OUTCOMES ***\n")
    
    # Define test parameters
    p1 = 0.3
    p2 = 0.45
    power_val = 0.8
    alpha = 0.05
    n1 = 150
    n2 = 150
    
    # Test analytical methods
    print("* Analytical Methods:")
    
    print("\nSample Size Calculation:")
    sample_size_result = sample_size_binary(p1, p2, power=power_val, alpha=alpha)
    pprint(sample_size_result)
    
    # Get the calculated sample sizes for power calculation
    n1_calc = sample_size_result["sample_size_1"]
    n2_calc = sample_size_result["sample_size_2"]
    
    print("\nPower Calculation:")
    power_result = power_binary(n1_calc, n2_calc, p1, p2, alpha=alpha)
    pprint(power_result)
    
    print("\nMinimum Detectable Effect:")
    mde_result = min_detectable_effect_binary(n1, n2, p1, power=power_val, alpha=alpha)
    pprint(mde_result)
    
    # Test simulation methods
    print("\n* Simulation Methods:")
    
    print("\nSample Size Calculation (Simulation):")
    sample_size_sim_result = sample_size_binary_sim(p1, p2, power=power_val, alpha=alpha, nsim=500)
    pprint(sample_size_sim_result)
    
    # Get the calculated sample sizes for power calculation
    n1_calc_sim = sample_size_sim_result["sample_size_1"]
    n2_calc_sim = sample_size_sim_result["sample_size_2"]
    
    print("\nPower Calculation (Simulation):")
    power_sim_result = power_binary_sim(n1_calc_sim, n2_calc_sim, p1, p2, alpha=alpha, nsim=500)
    pprint(power_sim_result)
    
    print("\nMinimum Detectable Effect (Simulation):")
    mde_sim_result = min_detectable_effect_binary_sim(n1, n2, p1, power=power_val, alpha=alpha, nsim=500)
    pprint(mde_sim_result)

def test_continuous_outcomes():
    """Test continuous outcome functions."""
    print("\n*** TESTING CONTINUOUS OUTCOMES ***\n")
    
    # Define test parameters
    mean1 = 10.0
    mean2 = 12.0
    std_dev = 5.0
    power_val = 0.8
    alpha = 0.05
    n1 = 100
    n2 = 100
    
    # Test analytical methods
    print("* Analytical Methods:")
    
    print("\nSample Size Calculation:")
    sample_size_result = sample_size_continuous(mean1, mean2, std_dev, power=power_val, alpha=alpha)
    pprint(sample_size_result)
    
    # Get the calculated sample sizes for power calculation
    n1_calc = sample_size_result["sample_size_1"]
    n2_calc = sample_size_result["sample_size_2"]
    
    print("\nPower Calculation:")
    power_result = power_continuous(n1_calc, n2_calc, mean1, mean2, std_dev, alpha=alpha)
    pprint(power_result)
    
    print("\nMinimum Detectable Effect:")
    mde_result = min_detectable_effect_continuous(n1, n2, std_dev, power=power_val, alpha=alpha)
    pprint(mde_result)
    
    # Test simulation methods
    print("\n* Simulation Methods:")
    
    # Calculate effect size (delta) for sample size calculation
    delta = abs(mean2 - mean1)
    
    print("\nSample Size Calculation (Simulation):")
    sample_size_sim_result = sample_size_continuous_sim(delta=delta, std_dev=std_dev, 
                                                      power=power_val, alpha=alpha, nsim=500)
    pprint(sample_size_sim_result)
    
    # Get the calculated sample sizes for power calculation
    n1_calc_sim = sample_size_sim_result["sample_size_1"]
    n2_calc_sim = sample_size_sim_result["sample_size_2"]
    
    print("\nPower Calculation (Simulation):")
    power_sim_result = power_continuous_sim(n1=n1_calc_sim, n2=n2_calc_sim, 
                                          mean1=mean1, mean2=mean2, 
                                          sd1=std_dev, alpha=alpha, nsim=500)
    pprint(power_sim_result)
    
    print("\nMinimum Detectable Effect (Simulation):")
    mde_sim_result = min_detectable_effect_continuous_sim(n1=n1, n2=n2, std_dev=std_dev, 
                                                        power=power_val, alpha=alpha, nsim=500)
    pprint(mde_sim_result)

def test_survival_outcomes():
    """Test survival outcome functions."""
    print("\n*** TESTING SURVIVAL OUTCOMES ***\n")
    
    # Define test parameters
    median1 = 12.0
    median2 = 18.0
    enrollment_period = 12.0
    follow_up_period = 12.0
    dropout_rate = 0.1
    power_val = 0.8
    alpha = 0.05
    n1 = 120
    n2 = 120
    
    # Test analytical methods
    print("* Analytical Methods:")
    
    print("\nSample Size Calculation:")
    sample_size_result = sample_size_survival(median1, median2, 
                                             enrollment_period=enrollment_period,
                                             follow_up_period=follow_up_period,
                                             dropout_rate=dropout_rate,
                                             power=power_val, alpha=alpha)
    pprint(sample_size_result)
    
    # Get the calculated sample sizes for power calculation
    n1_calc = sample_size_result["sample_size_1"]
    n2_calc = sample_size_result["sample_size_2"]
    
    print("\nPower Calculation:")
    power_result = power_survival(n1_calc, n2_calc, median1, median2,
                                 enrollment_period=enrollment_period,
                                 follow_up_period=follow_up_period,
                                 dropout_rate=dropout_rate, alpha=alpha)
    pprint(power_result)
    
    print("\nMinimum Detectable Effect:")
    mde_result = min_detectable_effect_survival(n1, n2, median1,
                                               enrollment_period=enrollment_period,
                                               follow_up_period=follow_up_period,
                                               dropout_rate=dropout_rate,
                                               power=power_val, alpha=alpha)
    pprint(mde_result)
    
    # Test simulation methods
    print("\n* Simulation Methods:")
    
    print("\nSample Size Calculation (Simulation):")
    sample_size_sim_result = sample_size_survival_sim(median1, median2,
                                                    enrollment_period=enrollment_period,
                                                    follow_up_period=follow_up_period,
                                                    dropout_rate=dropout_rate,
                                                    power=power_val, alpha=alpha, nsim=500)
    pprint(sample_size_sim_result)
    
    # Get the calculated sample sizes for power calculation
    n1_calc_sim = sample_size_sim_result["sample_size_1"]
    n2_calc_sim = sample_size_sim_result["sample_size_2"]
    
    print("\nPower Calculation (Simulation):")
    power_sim_result = power_survival_sim(n1_calc_sim, n2_calc_sim, median1, median2,
                                        enrollment_period=enrollment_period,
                                        follow_up_period=follow_up_period,
                                        dropout_rate=dropout_rate, alpha=alpha, nsim=500)
    pprint(power_sim_result)
    
    print("\nMinimum Detectable Effect (Simulation):")
    mde_sim_result = min_detectable_effect_survival_sim(n1, n2, median1,
                                                      enrollment_period=enrollment_period,
                                                      follow_up_period=follow_up_period,
                                                      dropout_rate=dropout_rate,
                                                      power=power_val, alpha=alpha, nsim=500)
    pprint(mde_sim_result)

def main():
    """Run all tests."""
    print("TESTING REFACTORED DESIGNPOWER MODULES")
    print("======================================")
    
    test_binary_outcomes()
    test_continuous_outcomes()
    test_survival_outcomes()
    
    print("\nAll tests completed!")

if __name__ == "__main__":
    main()
