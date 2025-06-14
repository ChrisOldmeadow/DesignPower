import unittest
import numpy as np
from scipy import stats
from core.designs.parallel import analytical_binary

class TestAnalyticalBinary(unittest.TestCase):
    
    def test_power_binary(self):
        """Test that power calculation gives expected results for binary outcomes"""
        result = analytical_binary.power_binary(
            n1=100, n2=100, p1=0.3, p2=0.5, alpha=0.05
        )
        self.assertTrue(0.80 <= result["power"] <= 0.85)  # Updated expected range
    
    def test_sample_size_binary(self):
        """Test that sample size calculation gives expected results for binary outcomes"""
        result = analytical_binary.sample_size_binary(
            p1=0.3, p2=0.5, power=0.8, alpha=0.05
        )
        # Check the sample sizes in the returned dictionary
        self.assertTrue(85 <= result["sample_size_1"] <= 95)  # Approximate expected range
        self.assertEqual(result["sample_size_1"], result["sample_size_2"])  # Equal allocation
    
    def test_min_detectable_effect_binary(self):
        """Test that minimum detectable effect calculation gives expected results"""
        result = analytical_binary.min_detectable_effect_binary(
            n1=100, n2=100, p1=0.3, power=0.8, alpha=0.05
        )
        self.assertTrue(0.15 <= result["minimum_detectable_difference"] <= 0.2)

    def test_power_binary_edge_cases(self):
        """Test power calculation with edge cases"""
        # Very small proportions
        small_prop = analytical_binary.power_binary(
            n1=200, n2=200, p1=0.01, p2=0.05, alpha=0.05
        )
        self.assertTrue(0 <= small_prop["power"] <= 1)
        
        # Very large proportions
        large_prop = analytical_binary.power_binary(
            n1=200, n2=200, p1=0.95, p2=0.98, alpha=0.05
        )
        self.assertTrue(0 <= large_prop["power"] <= 1)
        
        # Very small effect size
        small_effect = analytical_binary.power_binary(
            n1=200, n2=200, p1=0.3, p2=0.31, alpha=0.05
        )
        self.assertTrue(0 <= small_effect["power"] <= 0.2)  # Power should be low
        
        # Equal proportions (null hypothesis)
        equal_prop = analytical_binary.power_binary(
            n1=200, n2=200, p1=0.3, p2=0.3, alpha=0.05
        )
        self.assertAlmostEqual(equal_prop["power"], 0.05, places=2)  # Type I error rate

    def test_power_binary_non_inferiority(self):
        """Test non-inferiority power calculation for binary outcomes"""
        result = analytical_binary.power_binary_non_inferiority(
            n1=200, n2=200, p1=0.7, non_inferiority_margin=0.1, alpha=0.05
        )
        self.assertTrue(0.9 <= result["power"] <= 1.0)  # Power should be high

    def test_sample_size_binary_non_inferiority(self):
        """Test non-inferiority sample size calculation for binary outcomes"""
        result = analytical_binary.sample_size_binary_non_inferiority(
            p1=0.7, 
            non_inferiority_margin=0.1, 
            power=0.8, 
            alpha=0.05, 
            assumed_difference=0.0, # Assuming p1 and p2 are truly equivalent
            direction="lower"
        )
        # Expected sample size for p1=0.7, margin=0.1, power=0.8, alpha=0.05 (1-sided)
        # For p2 >= p1 - margin, i.e. p2 >= 0.6
        # Actual calculated value after removing arbitrary factors: 260
        self.assertTrue(258 <= result["sample_size_1"] <= 262, f"Unexpected sample_size_1: {result['sample_size_1']}")
        self.assertEqual(result["sample_size_1"], result["sample_size_2"]) # Assuming allocation_ratio=1.0 (default)

if __name__ == '__main__':
    unittest.main()
