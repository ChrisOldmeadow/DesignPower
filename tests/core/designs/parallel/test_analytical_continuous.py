import unittest
import numpy as np
from scipy import stats
from core.designs.parallel import analytical_continuous

class TestAnalyticalContinuous(unittest.TestCase):
    
    def test_power_continuous(self):
        """Test that power calculation gives expected results for continuous outcomes"""
        result = analytical_continuous.power_continuous(
            n1=50, n2=50, mean1=10, mean2=12, sd1=3, sd2=3, alpha=0.05
        )
        self.assertTrue(0.9 <= result["power"] <= 0.95)  # Updated expected power range
    
    def test_sample_size_continuous(self):
        """Test that sample size calculation gives expected results for continuous outcomes"""
        result = analytical_continuous.sample_size_continuous(
            mean1=10, mean2=12, sd1=3, sd2=3, power=0.8, alpha=0.05
        )
        self.assertTrue(32 <= result["sample_size_1"] <= 40)  # Updated expected range
        self.assertEqual(result["sample_size_1"], result["sample_size_2"])  # Equal allocation
    
    def test_min_detectable_effect_continuous(self):
        """Test that minimum detectable effect calculation gives expected results"""
        result = analytical_continuous.min_detectable_effect_continuous(
            n1=50, n2=50, sd1=3, sd2=3, power=0.8, alpha=0.05
        )
        # Use the correct key from the function's return dictionary
        self.assertTrue(1.5 <= result["minimum_detectable_effect"] <= 2.5)

    def test_power_continuous_edge_cases(self):
        """Test power calculation with edge cases"""
        # Very small effect size
        small_effect = analytical_continuous.power_continuous(
            n1=50, n2=50, mean1=10, mean2=10.1, sd1=3, sd2=3, alpha=0.05
        )
        self.assertTrue(0 <= small_effect["power"] <= 0.1)  # Power should be low
        
        # Very large effect size
        large_effect = analytical_continuous.power_continuous(
            n1=50, n2=50, mean1=10, mean2=15, sd1=3, sd2=3, alpha=0.05
        )
        self.assertTrue(0.9 <= large_effect["power"] <= 1.0)  # Power should be high
        
        # Equal means (null hypothesis)
        equal_means = analytical_continuous.power_continuous(
            n1=50, n2=50, mean1=10, mean2=10, sd1=3, sd2=3, alpha=0.05
        )
        self.assertAlmostEqual(equal_means["power"], 0.05, places=2)  # Type I error rate
        
        # Very small standard deviation
        small_sd = analytical_continuous.power_continuous(
            n1=50, n2=50, mean1=10, mean2=11, sd1=0.5, sd2=0.5, alpha=0.05
        )
        self.assertTrue(0.9 <= small_sd["power"] <= 1.0)  # Power should be high

    def test_power_continuous_non_inferiority(self):
        """Test non-inferiority power calculation for continuous outcomes"""
        result = analytical_continuous.power_continuous_non_inferiority(
            n1=50, n2=50, mean1=10, non_inferiority_margin=2, sd1=3, sd2=3,
            alpha=0.05, assumed_difference=0.5
        )
        self.assertTrue(0.8 <= result["power"] <= 1.0)  # Power should be high

    def test_sample_size_continuous_non_inferiority(self):
        """Test sample size calculation for non-inferiority continuous outcomes"""
        result = analytical_continuous.sample_size_continuous_non_inferiority(
            mean1=10,
            non_inferiority_margin=2,
            sd1=3, # sd2 will default to sd1
            power=0.8,
            alpha=0.05, # one-sided
            assumed_difference=0.5, # mean2 = 10.5
            direction="lower"
        )
        # Expected n1 = ceil(((1.64485 + 0.84162)^2 * (3^2 + 3^2)) / ( (10.5 - (10-2)) ^2 ))
        # Expected n1 = ceil((2.48647^2 * 18) / (2.5^2)) = ceil(6.1825 * 18 / 6.25) = ceil(111.285 / 6.25) = ceil(17.8056) = 18
        self.assertEqual(result["sample_size_1"], 18)
        self.assertEqual(result["sample_size_2"], 18) # Default allocation_ratio=1.0

if __name__ == '__main__':
    unittest.main()
