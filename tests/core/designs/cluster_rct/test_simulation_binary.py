import pytest
import numpy as np
from core.designs.cluster_rct import simulation_binary
import pandas as pd
from collections import Counter

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
    assert result["n_clusters_per_arm"] == 10
    assert result["cluster_size_avg_input"] == 30
    assert result["icc_input"] == 0.05
    assert result["p1"] == 0.3
    assert result["p2"] == 0.5
    assert result["nsim_run"] == NSIM_TEST

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
    assert isinstance(trial_data, pd.DataFrame), "simulate_binary_trial should return a pandas DataFrame"

    # Check DataFrame properties
    expected_cols = {'outcome', 'treatment', 'cluster_id'}
    assert set(trial_data.columns) == expected_cols, f"DataFrame columns should be {expected_cols}"

    expected_rows = n_clusters_arm * 2 * cluster_size
    assert len(trial_data) == expected_rows, f"DataFrame should have {expected_rows} rows (n_clusters_per_arm * 2 * cluster_size)"

    assert trial_data['outcome'].isin([0, 1]).all(), "Outcome column should contain only 0s and 1s"
    assert trial_data['treatment'].isin([0, 1]).all(), "Treatment column should contain only 0s and 1s"

    expected_unique_clusters = n_clusters_arm * 2
    assert trial_data['cluster_id'].nunique() == expected_unique_clusters, f"Should have {expected_unique_clusters} unique cluster IDs"

    # Check number of clusters per arm
    clusters_per_arm = trial_data.groupby('treatment')['cluster_id'].nunique()
    assert clusters_per_arm.get(0, 0) == n_clusters_arm, f"Control arm (treatment=0) should have {n_clusters_arm} clusters. Got {clusters_per_arm.get(0,0)}"
    assert clusters_per_arm.get(1, 0) == n_clusters_arm, f"Intervention arm (treatment=1) should have {n_clusters_arm} clusters. Got {clusters_per_arm.get(1,0)}"

    # Check overall proportions are somewhat reasonable (not a strict test due to randomness)
    # This is a very loose check, especially with small N; ensure data generation doesn't produce all 0s or all 1s unexpectedly
    if len(trial_data[trial_data['treatment'] == 0]['outcome']) > 0:
        p0_observed = trial_data[trial_data['treatment'] == 0]['outcome'].mean()
        assert 0 <= p0_observed <= 1, "Observed p0 should be between 0 and 1"
    if len(trial_data[trial_data['treatment'] == 1]['outcome']) > 0:
        p1_observed = trial_data[trial_data['treatment'] == 1]['outcome'].mean()
        assert 0 <= p1_observed <= 1, "Observed p1 should be between 0 and 1"

    # Test with zero ICC
    np.random.seed(SEED)
    trial_data_zero_icc = simulation_binary.simulate_binary_trial(
        n_clusters=n_clusters_arm,
        cluster_size=cluster_size,
        p1=p1,
        p2=p2,
        icc=0.0 # Zero ICC
    )
    assert isinstance(trial_data_zero_icc, pd.DataFrame), "trial_data_zero_icc should be a pandas DataFrame"

    # Check DataFrame properties for zero ICC case
    assert set(trial_data_zero_icc.columns) == expected_cols, f"DataFrame columns for zero ICC should be {expected_cols}"
    assert len(trial_data_zero_icc) == expected_rows, f"DataFrame for zero ICC should have {expected_rows} rows"
    assert trial_data_zero_icc['outcome'].isin([0, 1]).all(), "Outcome column for zero ICC should contain only 0s and 1s"
    assert trial_data_zero_icc['treatment'].isin([0, 1]).all(), "Treatment column for zero ICC should contain only 0s and 1s"
    assert trial_data_zero_icc['cluster_id'].nunique() == expected_unique_clusters, f"Should have {expected_unique_clusters} unique cluster IDs for zero ICC"
    clusters_per_arm_zero_icc = trial_data_zero_icc.groupby('treatment')['cluster_id'].nunique()
    assert clusters_per_arm_zero_icc.get(0, 0) == n_clusters_arm, f"Control arm (treatment=0) for zero ICC should have {n_clusters_arm} clusters. Got {clusters_per_arm_zero_icc.get(0,0)}"
    assert clusters_per_arm_zero_icc.get(1, 0) == n_clusters_arm, f"Intervention arm (treatment=1) for zero ICC should have {n_clusters_arm} clusters. Got {clusters_per_arm_zero_icc.get(1,0)}"

    # Test edge case: p1=0, p2=0
    np.random.seed(SEED)
    trial_data_p_zero = simulation_binary.simulate_binary_trial(
        n_clusters=n_clusters_arm,
        cluster_size=cluster_size,
        p1=0.0,
        p2=0.0,
        icc=icc
    )
    assert isinstance(trial_data_p_zero, pd.DataFrame), "trial_data_p_zero should be a pandas DataFrame"

    # Check DataFrame properties for p1=0, p2=0 case
    # Expected columns, rows, unique_clusters are defined earlier in the test function
    assert set(trial_data_p_zero.columns) == expected_cols, f"DataFrame columns for p1=0, p2=0 should be {expected_cols}"
    assert len(trial_data_p_zero) == expected_rows, f"DataFrame for p1=0, p2=0 should have {expected_rows} rows"
    assert (trial_data_p_zero['outcome'] == 0).all(), "Outcome column for p1=0, p2=0 should contain only 0s"
    assert trial_data_p_zero['treatment'].isin([0, 1]).all(), "Treatment column for p1=0, p2=0 should contain only 0s and 1s"
    assert trial_data_p_zero['cluster_id'].nunique() == expected_unique_clusters, f"Should have {expected_unique_clusters} unique cluster IDs for p1=0, p2=0"
    clusters_per_arm_p_zero = trial_data_p_zero.groupby('treatment')['cluster_id'].nunique()
    assert clusters_per_arm_p_zero.get(0, 0) == n_clusters_arm, f"Control arm (treatment=0) for p1=0, p2=0 should have {n_clusters_arm} clusters. Got {clusters_per_arm_p_zero.get(0,0)}"
    assert clusters_per_arm_p_zero.get(1, 0) == n_clusters_arm, f"Intervention arm (treatment=1) for p1=0, p2=0 should have {n_clusters_arm} clusters. Got {clusters_per_arm_p_zero.get(1,0)}"

    # Test edge case: p1=1, p2=1
    np.random.seed(SEED)
    trial_data_p_one = simulation_binary.simulate_binary_trial(
        n_clusters=n_clusters_arm,
        cluster_size=cluster_size,
        p1=1.0,
        p2=1.0,
        icc=icc
    )
    assert isinstance(trial_data_p_one, pd.DataFrame), "trial_data_p_one should be a pandas DataFrame"

    # Check DataFrame properties for p1=1, p2=1 case
    # Expected columns, rows, unique_clusters are defined earlier in the test function
    assert set(trial_data_p_one.columns) == expected_cols, f"DataFrame columns for p1=1, p2=1 should be {expected_cols}"
    assert len(trial_data_p_one) == expected_rows, f"DataFrame for p1=1, p2=1 should have {expected_rows} rows"
    assert (trial_data_p_one['outcome'] == 1).all(), "Outcome column for p1=1, p2=1 should contain only 1s"
    assert trial_data_p_one['treatment'].isin([0, 1]).all(), "Treatment column for p1=1, p2=1 should contain only 0s and 1s"
    assert trial_data_p_one['cluster_id'].nunique() == expected_unique_clusters, f"Should have {expected_unique_clusters} unique cluster IDs for p1=1, p2=1"
    clusters_per_arm_p_one = trial_data_p_one.groupby('treatment')['cluster_id'].nunique()
    assert clusters_per_arm_p_one.get(0, 0) == n_clusters_arm, f"Control arm (treatment=0) for p1=1, p2=1 should have {n_clusters_arm} clusters. Got {clusters_per_arm_p_one.get(0,0)}"
    assert clusters_per_arm_p_one.get(1, 0) == n_clusters_arm, f"Intervention arm (treatment=1) for p1=1, p2=1 should have {n_clusters_arm} clusters. Got {clusters_per_arm_p_one.get(1,0)}"


def test_analyze_binary_agg_ttest():
    """Test the _analyze_binary_agg_ttest function with various scenarios."""
    # Scenario 1: Success - Clear Difference
    data1 = {
        'outcome':   [1,0,1,1,0,0, 1,1,1,0,1,1],
        'treatment': [0,0,0,0,0,0, 1,1,1,1,1,1],
        'cluster_id':[0,0,0,1,1,1, 2,2,2,3,3,3]
    }
    df1 = pd.DataFrame(data1)
    res1 = simulation_binary._analyze_binary_agg_ttest(df1)
    assert res1['fit_status'] == 'success'
    assert 0 <= res1['p_value'] <= 1

    # Scenario 2: Success - No Variance, Means Equal
    data2 = { # All clusters have proportion 0.5
        'outcome':   [1,0,1,0, 1,0,1,0],
        'treatment': [0,0,0,0, 1,1,1,1],
        'cluster_id':[0,0,1,1, 2,2,3,3] 
    } 
    df2 = pd.DataFrame(data2)
    res2 = simulation_binary._analyze_binary_agg_ttest(df2)
    assert res2['fit_status'] == 'success_novar_means_equal'
    assert res2['p_value'] == 1.0

    # Scenario 3: Malformed DataFrame - empty
    df_empty = pd.DataFrame()
    res_empty = simulation_binary._analyze_binary_agg_ttest(df_empty)
    assert res_empty['fit_status'] == 'data_error_malformed_input_df'
    assert res_empty['p_value'] == 1.0

    # Scenario 4: Malformed DataFrame - missing columns
    df_missing_cols = pd.DataFrame({'outcome': [0,1]})
    res_missing_cols = simulation_binary._analyze_binary_agg_ttest(df_missing_cols)
    assert res_missing_cols['fit_status'] == 'data_error_malformed_input_df'
    assert res_missing_cols['p_value'] == 1.0

    # Scenario 5: Empty Arm
    data_empty_arm = {
        'outcome':   [1,0,1,1,0,0],
        'treatment': [0,0,0,0,0,0], # Only control arm data
        'cluster_id':[0,0,0,1,1,1]
    }
    df_empty_arm = pd.DataFrame(data_empty_arm)
    res_empty_arm = simulation_binary._analyze_binary_agg_ttest(df_empty_arm)
    assert res_empty_arm['fit_status'] == 'data_error_empty_arm'
    assert res_empty_arm['p_value'] == 1.0

    # Scenario 6: Too Few Clusters for t-test (e.g., 1 cluster in one arm)
    data_too_few = {
        'outcome':   [1,0,1, 0,1,0, 1,1], 
        'treatment': [0,0,0, 0,0,0, 1,1],
        'cluster_id':[0,0,0, 1,1,1, 2,2] # Control: C0, C1. Intervention: C2 (only 1 cluster)
    }
    df_too_few = pd.DataFrame(data_too_few)
    res_too_few = simulation_binary._analyze_binary_agg_ttest(df_too_few)
    assert res_too_few['fit_status'] == 'data_error_too_few_clusters_for_ttest'
    assert res_too_few['p_value'] == 1.0

    # Scenario 7: No Variance, Means Different
    data_novar_diff = { # Control prop 0.25, Intervention prop 0.75
        'outcome':   [1,0,0,0, 1,0,0,0,  1,1,1,0, 1,1,1,0], 
        'treatment': [0,0,0,0, 0,0,0,0,  1,1,1,1, 1,1,1,1], 
        'cluster_id':[0,0,0,0, 1,1,1,1,  2,2,2,2, 3,3,3,3] 
    } 
    df_novar_diff = pd.DataFrame(data_novar_diff)
    res_novar_diff = simulation_binary._analyze_binary_agg_ttest(df_novar_diff)
    assert res_novar_diff['fit_status'] == 'success_novar_means_diff'
    assert res_novar_diff['p_value'] == 0.0

    # Scenario 8: One group var > 0, other var = 0, means identical
    data_nan_potential = {
        'outcome':   [1,0,0,0, 1,1,1,0, 1,1,0,0, 1,1,0,0],
        'treatment': [0,0,0,0, 0,0,0,0, 1,1,1,1, 1,1,1,1],
        'cluster_id':[0,0,0,0, 1,1,1,1, 2,2,2,2, 3,3,3,3]
    }
    df_nan_potential = pd.DataFrame(data_nan_potential)
    res_nan_potential = simulation_binary._analyze_binary_agg_ttest(df_nan_potential)
    assert res_nan_potential['fit_status'] == 'success' 
    assert res_nan_potential['p_value'] > 0.05 

    # Scenario 9: One group var > 0, other var = 0, means different
    data_var_novar_means_diff = {
        'outcome':   [1,0,0,0, 1,1,1,0, 1,1,1,1, 1,1,1,1],
        'treatment': [0,0,0,0, 0,0,0,0, 1,1,1,1, 1,1,1,1],
        'cluster_id':[0,0,0,0, 1,1,1,1, 2,2,2,2, 3,3,3,3]
    }
    df_var_novar_means_diff = pd.DataFrame(data_var_novar_means_diff)
    res_var_novar_means_diff = simulation_binary._analyze_binary_agg_ttest(df_var_novar_means_diff)
    assert res_var_novar_means_diff['fit_status'] == 'success'
    assert np.isclose(res_var_novar_means_diff['p_value'], 0.2951672353008664) 


def test_power_binary_sim_agg_ttest():
    """Test power_binary_sim with analysis_method='aggregate_ttest'."""
    np.random.seed(SEED)
    result = simulation_binary.power_binary_sim(
        n_clusters=10,
        cluster_size=30,
        icc=0.05,
        p1=0.3,
        p2=0.5,
        alpha=0.05,
        nsim=NSIM_TEST, 
        analysis_method="aggregate_ttest"
    )

    assert 0 <= result["power"] <= 1
    assert result["n_clusters_per_arm"] == 10
    assert result["cluster_size_avg_input"] == 30
    assert result["icc_input"] == 0.05
    assert result["p1"] == 0.3
    assert result["p2"] == 0.5
    assert result["nsim_run"] == NSIM_TEST
    assert result["analysis_method"] == "aggregate_ttest"
    
    assert "fit_statuses" in result
    fit_statuses = result["fit_statuses"]
    assert isinstance(fit_statuses, dict) 
    
    acceptable_count = fit_statuses.get('success', 0) + \
                       fit_statuses.get('success_novar_means_equal', 0) + \
                       fit_statuses.get('success_novar_means_diff', 0)
    assert acceptable_count > 0
    assert acceptable_count <= NSIM_TEST

    total_fit_sims = sum(fit_statuses.values())
    assert total_fit_sims == NSIM_TEST

    np.random.seed(SEED)
    result_icc0 = simulation_binary.power_binary_sim(
        n_clusters=10, cluster_size=30, icc=0.0, p1=0.1, p2=0.2, 
        nsim=NSIM_TEST, analysis_method="aggregate_ttest"
    )
    assert 0 <= result_icc0["power"] <= 1
    assert result_icc0["analysis_method"] == "aggregate_ttest"
    assert sum(result_icc0["fit_statuses"].values()) == NSIM_TEST

    np.random.seed(SEED)
    result_few_clusters = simulation_binary.power_binary_sim(
        n_clusters=1, 
        cluster_size=30, icc=0.05, p1=0.3, p2=0.5, 
        nsim=NSIM_TEST, analysis_method="aggregate_ttest"
    )
    assert 0 <= result_few_clusters["power"] <= 1 
    assert result_few_clusters["fit_statuses"].get('data_error_too_few_clusters_for_ttest', 0) == NSIM_TEST
    assert result_few_clusters["power"] == 0.0

