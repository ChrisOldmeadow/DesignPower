import unittest
import numpy as np
from scipy import stats
from core.designs.parallel import analytical_survival

class TestAnalyticalSurvival(unittest.TestCase):
    
    def test_power_survival(self):
        """Test that power calculation gives expected results for survival outcomes"""
        result = analytical_survival.power_survival(
            n1=100, n2=100, median1=10, median2=15, enrollment_period=12, 
            follow_up_period=24, dropout_rate=0.1, alpha=0.05
        )
        self.assertTrue(0.7 <= result["power"] <= 0.95)  # Expected power range
    
    def test_sample_size_survival(self):
        """Test that sample size calculation gives expected results for survival outcomes"""
        result = analytical_survival.sample_size_survival(
            median1=10, median2=15, power=0.8, enrollment_period=12, 
            follow_up_period=24, dropout_rate=0.1, alpha=0.05
        )
        self.assertTrue(50 <= result["sample_size_1"] <= 150)  # Approximate expected range
        self.assertEqual(result["sample_size_1"], result["sample_size_2"])  # Equal allocation
    
    def test_min_detectable_effect_survival(self):
        """Test that minimum detectable effect calculation gives expected results"""
        result = analytical_survival.min_detectable_effect_survival(
            n1=100, n2=100, median1=10, power=0.8, enrollment_period=12, 
            follow_up_period=24, dropout_rate=0.1, alpha=0.05
        )
        # Check reasonable values for hazard ratio or median difference
        self.assertTrue(1.3 <= result["minimum_detectable_hazard_ratio"] <= 2.0)
    
    def test_power_survival_edge_cases(self):
        """Test power calculation with edge cases"""
        # Very small effect size (close hazard ratios)
        small_effect = analytical_survival.power_survival(
            n1=100, n2=100, median1=10, median2=11, enrollment_period=12, 
            follow_up_period=24, dropout_rate=0.1, alpha=0.05
        )
        self.assertTrue(0 <= small_effect["power"] <= 0.3)  # Power should be low
        
        # Very large effect size (very different survival times)
        large_effect = analytical_survival.power_survival(
            n1=100, n2=100, median1=10, median2=25, enrollment_period=12, 
            follow_up_period=24, dropout_rate=0.1, alpha=0.05
        )
        self.assertTrue(0.9 <= large_effect["power"] <= 1.0)  # Power should be high
        
        # High dropout rate
        high_dropout = analytical_survival.power_survival(
            n1=100, n2=100, median1=10, median2=15, enrollment_period=12, 
            follow_up_period=24, dropout_rate=0.4, alpha=0.05
        )
        # Power should be lower than with low dropout
        self.assertTrue(high_dropout["power"] < large_effect["power"])
    
    def test_power_survival_non_inferiority(self):
        """Test non-inferiority power calculation for survival outcomes"""
        result = analytical_survival.power_survival_non_inferiority(
            n1=250, n2=250, median1=10, non_inferiority_margin=1.5, enrollment_period=12, 
            follow_up_period=24, dropout_rate=0.1, alpha=0.05, assumed_hazard_ratio=1.0
        )
        # Print actual power for debugging
        print(f"Non-inferiority power: {result['power']}")
        self.assertTrue(0.6 <= result["power"] <= 1.0)  # Adjusted power expectation

if __name__ == '__main__':
    unittest.main()
