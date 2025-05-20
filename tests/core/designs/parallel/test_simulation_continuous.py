import unittest
import numpy as np
from scipy import stats
from core.designs.parallel import simulation_continuous

class TestSimulationContinuous(unittest.TestCase):
    
    def test_power_continuous_sim(self):
        """Test that simulation-based power calculation gives reasonable results"""
        result = simulation_continuous.power_continuous_sim(
            n1=50, n2=50, mean1=10, mean2=12, sd1=3, sd2=3, 
            alpha=0.05, nsim=1000, seed=42
        )
        # With 1000 simulations, we expect some variation but still in a reasonable range
        self.assertTrue(0.75 <= result["power"] <= 0.95)
    
    def test_sample_size_continuous_sim(self):
        """Test that simulation-based sample size calculation gives reasonable results"""
        result = simulation_continuous.sample_size_continuous_sim(
            delta=2, std_dev=3, power=0.8, alpha=0.05, 
            nsim=500, min_n=20, max_n=100, step=5
        )
        # Sample size should be similar to analytical result but with some variation
        self.assertTrue(20 <= result["sample_size_1"] <= 50)
    
    def test_min_detectable_effect_continuous_sim(self):
        """Test that simulation-based MDE calculation gives reasonable results"""
        result = simulation_continuous.min_detectable_effect_continuous_sim(
            n1=50, n2=50, std_dev=3, power=0.8, 
            alpha=0.05, nsim=500
        )
        # MDE should be in a reasonable range
        self.assertTrue(1.0 <= result["minimum_detectable_effect"] <= 2.5)
    
    def test_simulate_continuous_trial(self):
        """Test that continuous trial simulation function works correctly"""
        # Test with mean1=mean2 (null hypothesis)
        mean1 = 10
        mean2 = 10
        sd1 = 3
        sd2 = 3
        n1 = 50
        n2 = 50
        
        # Use seed parameter in the function call
        result = simulation_continuous.simulate_continuous_trial(
            n1=n1, n2=n2, mean1=mean1, mean2=mean2, sd1=sd1, sd2=sd2, seed=42
        )
        
        # For null hypothesis, rejection rate should be approximately alpha (Type I error)
        # This is a single test so we just check for valid results
        self.assertTrue(0 <= result["empirical_power"] <= 1)
        self.assertTrue(0 <= result["mean_p_value"] <= 1)
        
        # Run multiple simulations in a single call
        multi_result = simulation_continuous.simulate_continuous_trial(
            n1=n1, n2=n2, mean1=mean1, mean2=mean2, sd1=sd1, sd2=sd2,
            nsim=1000, seed=42
        )
        
        # Type I error rate should be approximately alpha=0.05
        self.assertTrue(abs(multi_result["empirical_power"] - 0.05) < 0.05)  # Allow 5% tolerance
    
    def test_consistency_with_analytical(self):
        """Test that simulation results are consistent with analytical results"""
        from core.designs.parallel import analytical_continuous
        
        # Parameters
        n1 = 50
        n2 = 50
        mean1 = 10
        mean2 = 12
        sd1 = 3
        sd2 = 3
        alpha = 0.05
        nsim = 2000
        seed = 42
        
        # Get analytical power
        analytical_result = analytical_continuous.power_continuous(
            n1=n1, n2=n2, mean1=mean1, mean2=mean2, sd1=sd1, sd2=sd2, alpha=alpha
        )
        
        # Get simulation power
        sim_result = simulation_continuous.power_continuous_sim(
            n1=n1, n2=n2, mean1=mean1, mean2=mean2, sd1=sd1, sd2=sd2, 
            alpha=alpha, nsim=nsim, seed=seed
        )
        
        # Results should be within a reasonable margin
        self.assertLess(abs(analytical_result["power"] - sim_result["power"]), 0.1)

if __name__ == '__main__':
    unittest.main()
