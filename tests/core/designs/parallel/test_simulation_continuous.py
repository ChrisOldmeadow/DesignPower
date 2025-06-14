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

    def test_power_continuous_non_inferiority_sim(self):
        """Test simulation-based power for non-inferiority continuous outcomes."""
        result = simulation_continuous.power_continuous_non_inferiority_sim(
            n1=50, 
            n2=50, 
            mean1_control=10, 
            non_inferiority_margin=2, 
            sd1=3, 
            sd2=3, 
            alpha=0.05, 
            assumed_difference=0.5, # true_mean2 = 10.5
            direction="lower", 
            nsim=2000, # Increased for stability
            seed=42
        )
        # Analytical power for these parameters is ~0.994
        # Expect simulation power to be close, e.g., within [0.97, 1.0]
        # Allow a slightly wider margin due to simulation variance
        self.assertTrue(0.96 <= result["power"] <= 1.0, 
                        msg=f"Simulated power {result['power']} out of expected range [0.96, 1.0]")

    def test_sample_size_continuous_non_inferiority_sim(self):
        """Test simulation-based sample size for non-inferiority continuous outcomes."""
        # Analytical sample size for similar params (margin=2, sd=3, power=0.8, alpha=0.05, diff=0.5) is ~18.
        # The simulation function iterates to find sample size.
        result = simulation_continuous.sample_size_continuous_non_inferiority_sim(
            non_inferiority_margin=2,
            std_dev=3,
            power=0.8,
            alpha=0.05,
            assumed_difference=0.5,
            direction="lower",
            nsim=500, # Number of sims per sample size step in the search
            min_n=10,
            max_n=40, # Search range for n1
            step=2      # Step for n1 search
            # allocation_ratio defaults to 1.0
            # repeated_measures defaults to False
        )

        # Expect sample_size_1 to be around 18. Allow a reasonable range due to simulation.
        # The function returns the first n that meets or exceeds power.
        self.assertTrue(14 <= result["sample_size_1"] <= 26, 
                        msg=f"Simulated sample_size_1 {result['sample_size_1']} out of expected range [14, 26]")
        self.assertEqual(result["sample_size_1"], result["sample_size_2"], 
                         msg="Sample sizes for group 1 and 2 should be equal with default allocation_ratio.")

    def test_simulate_continuous_non_inferiority(self):
        """Test the simulate_continuous_non_inferiority helper function."""
        params = {
            "n1": 50, 
            "n2": 50, 
            "non_inferiority_margin": 2, 
            "std_dev": 3, 
            "alpha": 0.05, 
            "assumed_difference": 0.5, # true_mean2 = control + 0.5
            "direction": "lower", 
            "nsim": 100, # Keep nsim low for test speed
            "seed": 123
        }

        # Call the function that returns empirical power results  
        sim_results = simulation_continuous.simulate_continuous_non_inferiority(**params)

        self.assertIn("empirical_power", sim_results)
        self.assertIn("n1", sim_results)
        self.assertIn("n2", sim_results)
        self.assertIn("non_inferiority_margin", sim_results)
        self.assertEqual(sim_results["simulations"], params["nsim"])
        
        # Verify power is between 0 and 1
        power = sim_results["empirical_power"] 
        self.assertTrue(0 <= power <= 1, f"Power should be between 0 and 1, got {power}")

        # Test that the function returns consistent results with the same seed
        sim_results2 = simulation_continuous.simulate_continuous_non_inferiority(**params)
        self.assertEqual(sim_results["empirical_power"], sim_results2["empirical_power"],
                        msg="Function should return consistent results with same seed")
        
        # Test with different seed gives potentially different results  
        params_diff_seed = params.copy()
        params_diff_seed["seed"] = 456
        sim_results3 = simulation_continuous.simulate_continuous_non_inferiority(**params_diff_seed)
        # Results may be different but should still be valid power values
        self.assertTrue(0 <= sim_results3["empirical_power"] <= 1)

    def test_power_continuous_non_inferiority_sim_repeated_measures_change_score(self):
        """Test power for non-inferiority with repeated measures (change score)."""
        # Higher correlation should lead to higher power due to reduced effective SD
        params_high_corr = {
            "n1": 50, "n2": 50, "mean1_control": 10, "non_inferiority_margin": 1,
            "sd1": 3, "alpha": 0.05, "assumed_difference": 0.2, "direction": "lower",
            "nsim": 1000, "seed": 420,
            "repeated_measures": True, "correlation": 0.8, "analysis_method": "change_score"
        }
        result_high_corr = simulation_continuous.power_continuous_non_inferiority_sim(**params_high_corr)
        self.assertTrue(0.1 < result_high_corr["power"] <= 1.0)
        self.assertTrue(result_high_corr["repeated_measures"])
        self.assertEqual(result_high_corr["correlation"], 0.8)
        self.assertEqual(result_high_corr["analysis_method"], "change_score")
        self.assertIn("sim_sd1_effective", result_high_corr)
        self.assertIn("sim_sd2_effective", result_high_corr)
        # Effective SD for change score: sd * sqrt(2*(1-corr)) = 3 * sqrt(2*(1-0.8)) = 3 * sqrt(0.4) approx 1.897
        self.assertAlmostEqual(result_high_corr["sim_sd1_effective"], 3 * np.sqrt(2 * (1 - 0.8)), places=3)

        params_low_corr = {**params_high_corr, "correlation": 0.2, "seed": 421}
        result_low_corr = simulation_continuous.power_continuous_non_inferiority_sim(**params_low_corr)
        self.assertTrue(0.0 < result_low_corr["power"] <= 1.0)
        self.assertLess(result_low_corr["power"], result_high_corr["power"] + 0.1) # Allow some noise

    def test_power_continuous_non_inferiority_sim_repeated_measures_ancova(self):
        """Test power for non-inferiority with repeated measures (ANCOVA)."""
        params_high_corr = {
            "n1": 50, "n2": 50, "mean1_control": 10, "non_inferiority_margin": 1,
            "sd1": 3, "alpha": 0.05, "assumed_difference": 0.2, "direction": "lower",
            "nsim": 1000, "seed": 422,
            "repeated_measures": True, "correlation": 0.8, "analysis_method": "ancova"
        }
        result_high_corr = simulation_continuous.power_continuous_non_inferiority_sim(**params_high_corr)
        self.assertTrue(0.1 < result_high_corr["power"] <= 1.0)
        self.assertTrue(result_high_corr["repeated_measures"])
        self.assertEqual(result_high_corr["analysis_method"], "ancova")
        # Effective SD for ANCOVA: sd * sqrt(1-corr^2) = 3 * sqrt(1-0.8^2) = 3 * sqrt(1-0.64) = 3 * sqrt(0.36) = 3 * 0.6 = 1.8
        self.assertAlmostEqual(result_high_corr["sim_sd1_effective"], 3 * np.sqrt(1 - 0.8**2), places=3)

        params_low_corr = {**params_high_corr, "correlation": 0.2, "seed": 423}
        result_low_corr = simulation_continuous.power_continuous_non_inferiority_sim(**params_low_corr)
        self.assertTrue(0.0 < result_low_corr["power"] <= 1.0)
        self.assertLess(result_low_corr["power"], result_high_corr["power"] + 0.1)

    def test_sample_size_continuous_non_inferiority_sim_repeated_measures_change_score(self):
        """Test sample size for non-inferiority with repeated measures (change score)."""
        # Higher correlation should lead to smaller sample size
        params_high_corr = {
            "non_inferiority_margin": 1, "std_dev": 3, "power": 0.8, "alpha": 0.05,
            "assumed_difference": 0.1, "direction": "lower", "nsim": 200, # Lower nsim for speed
            "min_n": 10, "max_n": 150, "step": 5,
            "repeated_measures": True, "correlation": 0.7, "method": "change_score"
        }
        result_high_corr = simulation_continuous.sample_size_continuous_non_inferiority_sim(**params_high_corr)
        self.assertTrue(10 <= result_high_corr["sample_size_1"] <= 150)
        self.assertTrue(result_high_corr["repeated_measures"])
        self.assertEqual(result_high_corr["correlation"], 0.7)
        self.assertEqual(result_high_corr["analysis_method"], "change_score")

        params_low_corr = {**params_high_corr, "correlation": 0.1}
        result_low_corr = simulation_continuous.sample_size_continuous_non_inferiority_sim(**params_low_corr)
        self.assertTrue(10 <= result_low_corr["sample_size_1"] <= 150)
        self.assertGreater(result_low_corr["sample_size_1"], result_high_corr["sample_size_1"] - params_high_corr["step"] * 2) # Allow some noise/step effects

    def test_sample_size_continuous_non_inferiority_sim_repeated_measures_ancova(self):
        """Test sample size for non-inferiority with repeated measures (ANCOVA)."""
        params_high_corr = {
            "non_inferiority_margin": 1, "std_dev": 3, "power": 0.8, "alpha": 0.05,
            "assumed_difference": 0.1, "direction": "lower", "nsim": 200, # Lower nsim for speed
            "min_n": 10, "max_n": 150, "step": 5,
            "repeated_measures": True, "correlation": 0.7, "method": "ancova"
        }
        result_high_corr = simulation_continuous.sample_size_continuous_non_inferiority_sim(**params_high_corr)
        self.assertTrue(10 <= result_high_corr["sample_size_1"] <= 150)
        self.assertTrue(result_high_corr["repeated_measures"])
        self.assertEqual(result_high_corr["analysis_method"], "ancova")

        params_low_corr = {**params_high_corr, "correlation": 0.1}
        result_low_corr = simulation_continuous.sample_size_continuous_non_inferiority_sim(**params_low_corr)
        self.assertTrue(10 <= result_low_corr["sample_size_1"] <= 150)
        self.assertGreater(result_low_corr["sample_size_1"], result_high_corr["sample_size_1"] - params_high_corr["step"] * 2)

if __name__ == '__main__':
    unittest.main()
