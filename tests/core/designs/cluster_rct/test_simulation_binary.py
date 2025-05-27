import pytest
import numpy as np
from core.designs.cluster_rct import simulation_binary

# Set a seed for reproducibility in simulation tests
SEED = 12345
NSIM_TEST = 200 # Reduced number of simulations for faster tests

def test_power_binary_cluster_rct_sim():
    """Test power calculation for binary outcome in cluster RCT using simulation."""
    np.random.seed(SEED)
    # Test with standard parameters
    result = simulation_binary.power_binary_sim(
        n_clusters=10,
        cluster_size=30,
        icc=0.05,
        p1=0.3, # Proportion in control
        p2=0.5, # Proportion in intervention
        alpha=0.05,
        nsim=NSIM_TEST
    )

    # Basic checks
    assert 0 <= result["power"] <= 1
    assert result["n_clusters"] == 10
    assert result["cluster_size"] == 30
    assert result["icc"] == 0.05
    assert result["p1"] == 0.3
    assert result["p2"] == 0.5
    assert result["nsim"] == NSIM_TEST

    # Check that increasing number of clusters generally increases power
    np.random.seed(SEED)
    result_higher_n_clusters = simulation_binary.power_binary_sim(
        n_clusters=15, # Increased
        cluster_size=30,
        icc=0.05,
        p1=0.3,
        p2=0.5,
        alpha=0.05,
        nsim=NSIM_TEST
    )
    # Allow for some simulation noise, but expect higher power
    assert result_higher_n_clusters["power"] > result["power"] - 0.1 

    # Check that increasing effect size (difference between p1 and p2) generally increases power
    np.random.seed(SEED)
    result_higher_effect = simulation_binary.power_binary_sim(
        n_clusters=10,
        cluster_size=30,
        icc=0.05,
        p1=0.3,
        p2=0.6, # Increased difference
        alpha=0.05,
        nsim=NSIM_TEST
    )
    assert result_higher_effect["power"] > result["power"] - 0.1

    # Check that increasing ICC generally decreases power
    np.random.seed(SEED)
    result_higher_icc = simulation_binary.power_binary_sim(
        n_clusters=10,
        cluster_size=30,
        icc=0.1, # Increased
        p1=0.3,
        p2=0.5,
        alpha=0.05,
        nsim=NSIM_TEST
    )
    assert result_higher_icc["power"] < result["power"] + 0.1

    # Test with zero ICC (should generally have higher power than with ICC > 0)
    np.random.seed(SEED)
    result_zero_icc = simulation_binary.power_binary_sim(
        n_clusters=10,
        cluster_size=30,
        icc=0.0, # Zero ICC
        p1=0.3,
        p2=0.5,
        alpha=0.05,
        nsim=NSIM_TEST
    )
    assert result_zero_icc["power"] > result["power"] - 0.1

def test_sample_size_binary_cluster_rct_sim():
    """Test sample size calculation for binary outcome in cluster RCT using simulation."""
    np.random.seed(SEED)
    input_target_power = 0.8
    # Test with standard parameters
    result = simulation_binary.sample_size_binary_sim(
        p1=0.3,
        p2=0.5,
        icc=0.05,
        cluster_size=30,
        power=input_target_power,
        alpha=0.05,
        nsim=NSIM_TEST,
        min_n=2, # Ensure at least 2 clusters per arm
        max_n=50 # Cap for test speed
    )

    # Basic checks
    assert result["n_clusters"] > 0
    assert result["cluster_size"] == 30
    assert result["icc"] == 0.05
    assert result["p1"] == 0.3
    assert result["p2"] == 0.5
    assert result["alpha"] == 0.05
    assert result["nsim"] == NSIM_TEST
    # Achieved power should be reasonably close to target power
    assert input_target_power - 0.15 <= result["power"] <= input_target_power + 0.15

    # Check that increasing target power generally increases n_clusters
    np.random.seed(SEED)
    result_higher_power = simulation_binary.sample_size_binary_sim(
        p1=0.3,
        p2=0.5,
        icc=0.05,
        cluster_size=30,
        power=0.9, # Increased
        alpha=0.05,
        nsim=NSIM_TEST,
        min_n=2,
        max_n=50
    )
    assert result_higher_power["n_clusters"] >= result["n_clusters"] # Can be equal due to step size or max_n_clusters

    # Check that increasing effect size (difference between p1 and p2) generally decreases n_clusters
    np.random.seed(SEED)
    result_higher_effect = simulation_binary.sample_size_binary_sim(
        p1=0.3,
        p2=0.6, # Increased difference
        icc=0.05,
        cluster_size=30,
        power=input_target_power,
        alpha=0.05,
        nsim=NSIM_TEST,
        min_n=2,
        max_n=50
    )
    assert result_higher_effect["n_clusters"] <= result["n_clusters"]
    assert input_target_power - 0.15 <= result_higher_effect["power"] <= input_target_power + 0.15

    # Check that increasing ICC generally increases n_clusters
    np.random.seed(SEED)
    result_higher_icc = simulation_binary.sample_size_binary_sim(
        p1=0.3,
        p2=0.5,
        icc=0.1, # Increased
        cluster_size=30,
        power=input_target_power,
        alpha=0.05,
        nsim=NSIM_TEST,
        min_n=2,
        max_n=50
    )
    assert result_higher_icc["n_clusters"] >= result["n_clusters"]
    assert input_target_power - 0.15 <= result_higher_icc["power"] <= input_target_power + 0.15

    # Test with zero ICC (should generally require fewer clusters)
    np.random.seed(SEED)
    result_zero_icc = simulation_binary.sample_size_binary_sim(
        p1=0.3,
        p2=0.5,
        icc=0.0, # Zero ICC
        cluster_size=30,
        power=input_target_power,
        alpha=0.05,
        nsim=NSIM_TEST,
        min_n=2,
        max_n=50
    )
    assert result_zero_icc["n_clusters"] <= result["n_clusters"]
    assert input_target_power - 0.15 <= result_zero_icc["power"] <= input_target_power + 0.15

    # Test with higher target power (should generally increase n_clusters)
    np.random.seed(SEED)
    higher_target_power = 0.9
    result_higher_power = simulation_binary.sample_size_binary_sim(
        p1=0.3,
        p2=0.5,
        icc=0.05,
        cluster_size=30,
        power=higher_target_power, # Higher target power
        alpha=0.05,
        nsim=NSIM_TEST,
        min_n=2, max_n=50
    )
    assert result_higher_power["n_clusters"] > result["n_clusters"] - 5 # Allow some noise
    assert higher_target_power - 0.15 <= result_higher_power["power"] <= higher_target_power + 0.15


def test_mde_binary_cluster_rct_sim():
    """Test MDE calculation for binary outcome in cluster RCT using simulation."""
    np.random.seed(SEED)
    # Test with standard parameters
    p1_mde_test = 0.3
    result = simulation_binary.min_detectable_effect_binary_sim(
        p1=p1_mde_test,
        n_clusters=15, # Using a slightly higher n_clusters for MDE stability
        cluster_size=30,
        icc=0.05,
        power=0.8,
        alpha=0.05,
        nsim=NSIM_TEST,
        min_effect=0.001, # Smallest difference p2-p1
        max_effect=0.95 - p1_mde_test   # Largest difference p2-p1, ensuring p2 <= 0.95
    )

    # Basic checks
    assert result["mde"] > 0
    assert result["p1"] == p1_mde_test
    assert result["p2"] > result["p1"]
    assert result["p2"] <= 0.95 # Check against the implied max_p2
    assert result["n_clusters"] == 15
    assert result["cluster_size"] == 30
    assert result["icc"] == 0.05
    assert result["power"] == 0.8
    assert result["alpha"] == 0.05
    assert result["nsim"] == NSIM_TEST
    # Achieved power should be reasonably close to target power
    assert result["power"] - 0.15 <= result["achieved_power"] <= result["power"] + 0.15

    # Check that increasing n_clusters generally decreases MDE
    np.random.seed(SEED)
    result_higher_n_clusters = simulation_binary.min_detectable_effect_binary_sim(
        p1=p1_mde_test,
        n_clusters=25, # Increased
        cluster_size=30,
        icc=0.05,
        power=0.8,
        alpha=0.05,
        nsim=NSIM_TEST,
        min_effect=0.001,
        max_effect=0.95 - p1_mde_test
    )
    assert result_higher_n_clusters["mde"] < result["mde"] + 0.05 # Allow some noise
    # Ensure p2 for higher_n_clusters is still valid
    assert result_higher_n_clusters["p2"] > result_higher_n_clusters["p1"]
    assert result_higher_n_clusters["power"] == 0.8 # Target power should be consistent

    # Check that increasing target power generally increases MDE
    np.random.seed(SEED)
    result_higher_power = simulation_binary.min_detectable_effect_binary_sim(
        p1=p1_mde_test,
        n_clusters=15,
        cluster_size=30,
        icc=0.05,
        power=0.9, # Increased
        alpha=0.05,
        nsim=NSIM_TEST,
        min_effect=0.001,
        max_effect=0.95 - p1_mde_test
    )
    assert result_higher_power["mde"] > result["mde"] - 0.05 # Allow some noise
    # Ensure p2 for higher_power is still valid
    assert result_higher_power["p2"] > result_higher_power["p1"]
    assert result_higher_power["power"] == 0.9 # Target power should reflect the input

    # Check that increasing ICC generally increases MDE
    np.random.seed(SEED)
    result_higher_icc = simulation_binary.min_detectable_effect_binary_sim(
        p1=p1_mde_test,
        n_clusters=15,
        cluster_size=30,
        icc=0.1, # Increased
        power=0.8,
        alpha=0.05,
        nsim=NSIM_TEST,
        min_effect=0.001,
        max_effect=0.95 - p1_mde_test
    )
    assert result_higher_icc["mde"] > result["mde"] - 0.05 # Allow some noise
    # Ensure p2 for higher_icc is still valid
    assert result_higher_icc["p2"] > result_higher_icc["p1"]
    assert result_higher_icc["power"] == 0.8 # Target power should be consistent

    # Test with zero ICC (should generally have smaller MDE)
    np.random.seed(SEED)
    result_zero_icc = simulation_binary.min_detectable_effect_binary_sim(
        p1=p1_mde_test,
        n_clusters=15,
        cluster_size=30,
        icc=0.0, # Zero ICC
        power=0.8,
        alpha=0.05,
        nsim=NSIM_TEST,
        min_effect=0.001,
        max_effect=0.95 - p1_mde_test
    )
    assert result_zero_icc["mde"] < result["mde"] + 0.05 # Allow some noise
    # Ensure p2 for zero_icc is still valid
    assert result_zero_icc["p2"] > result_zero_icc["p1"]
    assert result_zero_icc["power"] == 0.8 # Target power should be consistent


def test_simulate_cluster_binary_trial():
    """Test the helper function simulate_cluster_binary_trial."""
    np.random.seed(SEED)
    n_clusters_arm = 5
    cluster_size = 20
    p1 = 0.2
    p2 = 0.4
    icc = 0.1

    trial_data = simulation_binary.simulate_binary_trial(
        n_clusters=n_clusters_arm,
        cluster_size=cluster_size,
        p1=p1,
        p2=p2,
        icc=icc
    )

    # Check output structure and types
    assert isinstance(trial_data, tuple), "simulate_binary_trial should return a tuple"
    assert len(trial_data) == 2, "simulate_binary_trial should return a tuple of length 2 (z_stat, p_value)"
    
    z_stat, p_value = trial_data
    
    assert isinstance(z_stat, (float, np.floating)), "z_stat should be a float"
    assert isinstance(p_value, (float, np.floating)), "p_value should be a float"
    assert 0 <= p_value <= 1, "p_value should be between 0 and 1"

    # Test with zero ICC
    np.random.seed(SEED)
    trial_data_zero_icc = simulation_binary.simulate_binary_trial(
        n_clusters=n_clusters_arm,
        cluster_size=cluster_size,
        p1=p1,
        p2=p2,
        icc=0.0 # Zero ICC
    )
    assert isinstance(trial_data_zero_icc, tuple)
    assert len(trial_data_zero_icc) == 2
    z_stat_zero_icc, p_value_zero_icc = trial_data_zero_icc
    assert isinstance(z_stat_zero_icc, (float, np.floating))
    assert isinstance(p_value_zero_icc, (float, np.floating))
    assert 0 <= p_value_zero_icc <= 1

    # Test edge case: p1=0, p2=0
    np.random.seed(SEED)
    trial_data_p_zero = simulation_binary.simulate_binary_trial(
        n_clusters=n_clusters_arm,
        cluster_size=cluster_size,
        p1=0.0,
        p2=0.0,
        icc=icc
    )
    assert isinstance(trial_data_p_zero, tuple)
    assert len(trial_data_p_zero) == 2
    z_stat_p_zero, p_value_p_zero = trial_data_p_zero
    # z_stat might be nan or inf if all counts are zero, p_value might be 1.0
    # For now, just check type and p-value range if not nan
    assert isinstance(z_stat_p_zero, (float, np.floating))
    assert isinstance(p_value_p_zero, (float, np.floating))
    if not np.isnan(p_value_p_zero):
        assert 0 <= p_value_p_zero <= 1

    # Test edge case: p1=1, p2=1
    np.random.seed(SEED)
    trial_data_p_one = simulation_binary.simulate_binary_trial(
        n_clusters=n_clusters_arm,
        cluster_size=cluster_size,
        p1=1.0,
        p2=1.0,
        icc=icc
    )
    assert isinstance(trial_data_p_one, tuple)
    assert len(trial_data_p_one) == 2
    z_stat_p_one, p_value_p_one = trial_data_p_one
    assert isinstance(z_stat_p_one, (float, np.floating))
    assert isinstance(p_value_p_one, (float, np.floating))
    if not np.isnan(p_value_p_one):
        assert 0 <= p_value_p_one <= 1
