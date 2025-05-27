"""Unit tests for continuous outcome simulation functions in cluster RCTs."""

import numpy as np
import pytest
import warnings
from core.designs.cluster_rct import simulation_continuous

# Seed for reproducibility in tests
SEED = 42
# Number of simulations for tests (keep low for speed)
NSIM_TEST = 50 # Reduced from 200 for continuous tests as they can be slower


def test_power_continuous_cluster_rct_sim():
    """Test power calculation for continuous outcome in cluster RCT using simulation."""
    np.random.seed(SEED)
    params = {
        "n_clusters": 10,
        "cluster_size": 30,
        "icc": 0.05,
        "mean1": 0.0,
        "mean2": 0.5, # Standardized effect size of 0.5 if std_dev=1
        "std_dev": 1.0,
        "nsim": NSIM_TEST, # Keep low for speed
        "alpha": 0.05,
        "analysis_model": "ttest",
        "seed": SEED
    }

    # Test with standard parameters using ttest
    result = simulation_continuous.power_continuous_sim(**params)

    assert isinstance(result, dict), "Result should be a dictionary"
    expected_keys = ["power", "n_clusters", "cluster_size", "icc", "mean1", "mean2", 
                     "std_dev", "nsim", "alpha", "analysis_model", "failed_sims", "converged_sims"]
    for key in expected_keys:
        assert key in result, f"Key '{key}' missing from results"
    
    assert 0 <= result["power"] <= 1, "Power should be between 0 and 1"
    assert result["n_clusters"] == params["n_clusters"]
    assert result["cluster_size"] == params["cluster_size"]
    assert result["icc"] == params["icc"]
    assert result["mean1"] == params["mean1"]
    assert result["mean2"] == params["mean2"]
    assert result["std_dev"] == params["std_dev"]
    assert result["nsim"] == params["nsim"]
    assert result["alpha"] == params["alpha"]
    assert result["analysis_model"] == params["analysis_model"]
    assert isinstance(result["failed_sims"], int)
    assert isinstance(result["converged_sims"], int)
    assert result["failed_sims"] + result["converged_sims"] == params["nsim"]

    power_base = result["power"]

    # Test with zero ICC (should generally have higher power)
    params_zero_icc = params.copy()
    params_zero_icc["icc"] = 0.0
    np.random.seed(SEED)
    result_zero_icc = simulation_continuous.power_continuous_sim(**params_zero_icc)
    assert result_zero_icc["power"] >= power_base - 0.1 # Allow for simulation noise

    # Test with more clusters (should generally have higher power)
    params_more_clusters = params.copy()
    params_more_clusters["n_clusters"] = 20 # Double clusters
    np.random.seed(SEED)
    result_more_clusters = simulation_continuous.power_continuous_sim(**params_more_clusters)
    # With NSIM_TEST being small, this might not always hold strictly, add tolerance
    assert result_more_clusters["power"] >= power_base - 0.1 

    # Test with larger effect size (should generally have higher power)
    params_larger_effect = params.copy()
    params_larger_effect["mean2"] = 0.8 # Larger effect
    np.random.seed(SEED)
    result_larger_effect = simulation_continuous.power_continuous_sim(**params_larger_effect)
    assert result_larger_effect["power"] >= power_base - 0.1

    # Test with 'mixedlm' model
    params_mixedlm = params.copy()
    params_mixedlm["analysis_model"] = "mixedlm"
    params_mixedlm["use_satterthwaite"] = True # Test a model-specific param
    np.random.seed(SEED)
    result_mixedlm = simulation_continuous.power_continuous_sim(**params_mixedlm)
    assert isinstance(result_mixedlm, dict)
    assert 0 <= result_mixedlm["power"] <= 1
    assert result_mixedlm["analysis_model"] == "mixedlm"
    assert result_mixedlm["use_satterthwaite"] is True
    assert result_mixedlm["failed_sims"] + result_mixedlm["converged_sims"] == params_mixedlm["nsim"]

    # Test with 'gee' model
    params_gee = params.copy()
    params_gee["analysis_model"] = "gee"
    params_gee["use_bias_correction"] = True # Test a model-specific param
    np.random.seed(SEED)
    result_gee = simulation_continuous.power_continuous_sim(**params_gee)
    assert isinstance(result_gee, dict)
    assert 0 <= result_gee["power"] <= 1
    assert result_gee["analysis_model"] == "gee"
    assert result_gee["use_bias_correction"] is True
    assert result_gee["failed_sims"] + result_gee["converged_sims"] == params_gee["nsim"]

    # Test with 'bayes' model (conditionally)
    if not simulation_continuous._STAN_AVAILABLE:
        warnings.warn("CmdStanPy or Stan installation not found, skipping Bayesian power tests.")
    else:
        params_bayes = params.copy()
        params_bayes["analysis_model"] = "bayes"
        params_bayes["bayes_draws"] = 50 # Keep low for speed
        params_bayes["bayes_warmup"] = 50
        np.random.seed(SEED)
        result_bayes = simulation_continuous.power_continuous_sim(**params_bayes)
        assert isinstance(result_bayes, dict)
        assert 0 <= result_bayes["power"] <= 1 # Power interpretation might differ for Bayes
        assert result_bayes["analysis_model"] == "bayes"
        assert result_bayes["bayes_draws"] == 50
        assert result_bayes["failed_sims"] + result_bayes["converged_sims"] == params_bayes["nsim"]

    # TODO: Test specific parameters for those models (e.g., lmm_method, use_satterthwaite for mixedlm)


def test_sample_size_continuous_cluster_rct_sim():
    """Test sample_size_continuous_sim for cluster RCTs."""
    params = {
        "mean1": 0,
        "mean2": 0.5, # Moderate effect size
        "std_dev": 1.0,
        "icc": 0.05,
        "cluster_size": 20,
        "power": 0.8, # Target power
        "alpha": 0.05,
        "nsim": NSIM_TEST, # Low for speed
        "min_n": 2,
        "max_n": 10, # Keep range small for tests
        "analysis_model": "ttest",
        "seed": SEED
    }

    np.random.seed(SEED)
    result = simulation_continuous.sample_size_continuous_sim(**params)

    assert isinstance(result, dict), "Result should be a dictionary"
    expected_keys = [
        "n_clusters", "cluster_size", "total_n", "icc", "mean1", "mean2", 
        "difference", "std_dev", "design_effect", "alpha", "target_power", 
        "achieved_power", "nsim", "analysis_model"
    ]
    for key in expected_keys:
        assert key in result, f"Key '{key}' missing from results for ttest"

    assert isinstance(result["n_clusters"], (int, np.integer)), "n_clusters should be an integer"
    assert params["min_n"] <= result["n_clusters"] <= params["max_n"], "n_clusters out of bounds"
    # Achieved power can deviate, especially with low nsim and discrete n_clusters
    assert 0 <= result["achieved_power"] <= 1, "Achieved power out of range"
    # Looser check for achieved_power vs target_power due to simulation noise and discrete cluster search
    assert abs(result["achieved_power"] - params["power"]) < 0.3, \
        f"Achieved power {result['achieved_power']:.2f} far from target {params['power']:.2f}"
    assert result["analysis_model"] == "ttest"

    n_clusters_base = result["n_clusters"]

    # Test: Higher target power -> more clusters
    params_higher_power = params.copy()
    params_higher_power["power"] = 0.9
    params_higher_power["max_n"] = 15 # Allow more clusters
    np.random.seed(SEED)
    result_higher_power = simulation_continuous.sample_size_continuous_sim(**params_higher_power)
    assert result_higher_power["n_clusters"] >= n_clusters_base or result_higher_power["n_clusters"] == params_higher_power["max_n"]

    # Test: Higher ICC -> more clusters
    params_higher_icc = params.copy()
    params_higher_icc["icc"] = 0.1
    params_higher_icc["max_n"] = 15 # Allow more clusters
    np.random.seed(SEED)
    result_higher_icc = simulation_continuous.sample_size_continuous_sim(**params_higher_icc)
    assert result_higher_icc["n_clusters"] >= n_clusters_base or result_higher_icc["n_clusters"] == params_higher_icc["max_n"]

    # Test: Larger std_dev -> more clusters
    params_larger_std = params.copy()
    params_larger_std["std_dev"] = 1.5
    params_larger_std["max_n"] = 15 # Allow more clusters
    np.random.seed(SEED)
    result_larger_std = simulation_continuous.sample_size_continuous_sim(**params_larger_std)
    assert result_larger_std["n_clusters"] >= n_clusters_base or result_larger_std["n_clusters"] == params_larger_std["max_n"]

    # Test: Larger effect size (smaller mean2) -> fewer clusters
    params_larger_effect = params.copy()
    params_larger_effect["mean2"] = 0.75 # Larger difference
    np.random.seed(SEED)
    result_larger_effect = simulation_continuous.sample_size_continuous_sim(**params_larger_effect)
    assert result_larger_effect["n_clusters"] <= n_clusters_base or result_larger_effect["n_clusters"] == params_larger_effect["min_n"]

    # Test with 'mixedlm' model
    params_mixedlm = params.copy()
    params_mixedlm["analysis_model"] = "mixedlm"
    np.random.seed(SEED)
    result_mixedlm = simulation_continuous.sample_size_continuous_sim(**params_mixedlm)
    assert isinstance(result_mixedlm, dict)
    assert result_mixedlm["analysis_model"] == "mixedlm"
    assert params["min_n"] <= result_mixedlm["n_clusters"] <= params["max_n"]

    # Test with 'gee' model
    params_gee = params.copy()
    params_gee["analysis_model"] = "gee"
    np.random.seed(SEED)
    result_gee = simulation_continuous.sample_size_continuous_sim(**params_gee)
    assert isinstance(result_gee, dict)
    assert result_gee["analysis_model"] == "gee"
    assert params["min_n"] <= result_gee["n_clusters"] <= params["max_n"]

    # Test with 'bayes' model (conditionally)
    if not simulation_continuous._STAN_AVAILABLE:
        warnings.warn("CmdStanPy or Stan installation not found, skipping Bayesian sample size tests.")
    else:
        params_bayes = params.copy()
        params_bayes["analysis_model"] = "bayes"
        params_bayes["bayes_draws"] = 30 # Very low for speed in iterative search
        params_bayes["bayes_warmup"] = 30
        np.random.seed(SEED)
        result_bayes = simulation_continuous.sample_size_continuous_sim(**params_bayes)
        assert isinstance(result_bayes, dict)
        assert result_bayes["analysis_model"] == "bayes"
        assert params["min_n"] <= result_bayes["n_clusters"] <= params["max_n"]


def test_mde_continuous_cluster_rct_sim():
    """Test min_detectable_effect_continuous_sim for cluster RCTs."""
    params = {
        "n_clusters": 10, # Fixed number of clusters
        "cluster_size": 20,
        "icc": 0.05,
        "std_dev": 1.0,
        "power": 0.8, # Target power
        "alpha": 0.05,
        "nsim": NSIM_TEST, # Low for speed
        "precision": 0.05, # Wider precision for faster tests
        "max_iterations": 5, # Fewer iterations for tests
        "analysis_model": "ttest",
        "seed": SEED
    }

    np.random.seed(SEED)
    result = simulation_continuous.min_detectable_effect_continuous_sim(**params)

    assert isinstance(result, dict), "Result should be a dictionary"
    expected_keys = [
        "mde", "standardized_mde", "n_clusters", "cluster_size", "total_n", "icc", 
        "std_dev", "design_effect", "effective_n", "alpha", "target_power", 
        "achieved_power", "nsim", "iterations", "analysis_model"
    ]
    for key in expected_keys:
        assert key in result, f"Key '{key}' missing from results for ttest"

    assert isinstance(result["mde"], float) and result["mde"] > 0, "MDE should be a positive float"
    assert isinstance(result["standardized_mde"], float) and result["standardized_mde"] > 0
    # Achieved power can deviate, especially with low nsim and iterative search
    assert 0 <= result["achieved_power"] <= 1, "Achieved power out of range"
    assert abs(result["achieved_power"] - params["power"]) < 0.3, \
        f"Achieved power {result['achieved_power']:.2f} far from target {params['power']:.2f}"
    assert result["analysis_model"] == "ttest"

    mde_base = result["mde"]

    # Test: More clusters (n_clusters) -> smaller MDE
    params_more_clusters = params.copy()
    params_more_clusters["n_clusters"] = 15
    np.random.seed(SEED)
    result_more_clusters = simulation_continuous.min_detectable_effect_continuous_sim(**params_more_clusters)
    assert result_more_clusters["mde"] <= mde_base + params["precision"] # Allow for precision

    # Test: Higher ICC -> larger MDE
    params_higher_icc = params.copy()
    params_higher_icc["icc"] = 0.1
    np.random.seed(SEED)
    result_higher_icc = simulation_continuous.min_detectable_effect_continuous_sim(**params_higher_icc)
    assert result_higher_icc["mde"] >= mde_base - params["precision"]

    # Test: Higher target power -> larger MDE
    params_higher_power = params.copy()
    params_higher_power["power"] = 0.9
    np.random.seed(SEED)
    result_higher_power = simulation_continuous.min_detectable_effect_continuous_sim(**params_higher_power)
    assert result_higher_power["mde"] >= mde_base - params["precision"]

    # Test: Larger std_dev -> larger MDE
    params_larger_std = params.copy()
    params_larger_std["std_dev"] = 1.5
    np.random.seed(SEED)
    result_larger_std = simulation_continuous.min_detectable_effect_continuous_sim(**params_larger_std)
    # MDE should scale roughly with std_dev
    assert result_larger_std["mde"] >= (mde_base * 1.5) - (params["precision"] * 1.5) 

    # Test with 'mixedlm' model
    params_mixedlm = params.copy()
    params_mixedlm["analysis_model"] = "mixedlm"
    np.random.seed(SEED)
    result_mixedlm = simulation_continuous.min_detectable_effect_continuous_sim(**params_mixedlm)
    assert isinstance(result_mixedlm, dict)
    assert result_mixedlm["analysis_model"] == "mixedlm"
    assert result_mixedlm["mde"] > 0

    # Test with 'gee' model
    params_gee = params.copy()
    params_gee["analysis_model"] = "gee"
    np.random.seed(SEED)
    result_gee = simulation_continuous.min_detectable_effect_continuous_sim(**params_gee)
    assert isinstance(result_gee, dict)
    assert result_gee["analysis_model"] == "gee"
    assert result_gee["mde"] > 0

    # Test with 'bayes' model (conditionally)
    if not simulation_continuous._STAN_AVAILABLE:
        warnings.warn("CmdStanPy or Stan installation not found, skipping Bayesian MDE tests.")
    else:
        params_bayes = params.copy()
        params_bayes["analysis_model"] = "bayes"
        params_bayes["bayes_draws"] = 30 # Very low for speed in iterative search
        params_bayes["bayes_warmup"] = 30
        np.random.seed(SEED)
        result_bayes = simulation_continuous.min_detectable_effect_continuous_sim(**params_bayes)
        assert isinstance(result_bayes, dict)
        assert result_bayes["analysis_model"] == "bayes"
        assert result_bayes["mde"] > 0

# All tests for this module are now defined.
# We can consider adding more specific checks for edge cases or parameter interactions if needed.


def test_simulate_continuous_cluster_trial():
    """Test the helper function simulate_continuous_trial."""
    np.random.seed(SEED)
    n_clusters_arm = 5
    cluster_size = 20
    mean1 = 10.0
    mean2 = 12.0
    std_dev = 5.0
    icc = 0.1

    # Test with default 'ttest' model
    t_stat, p_value = simulation_continuous.simulate_continuous_trial(
        n_clusters=n_clusters_arm,
        cluster_size=cluster_size,
        icc=icc,
        mean1=mean1,
        mean2=mean2,
        std_dev=std_dev
    )

    assert isinstance(t_stat, (float, np.floating)), "t_stat should be a float"
    assert isinstance(p_value, (float, np.floating)), "p_value should be a float"
    assert 0 <= p_value <= 1, "p_value should be between 0 and 1"

    # Test with zero ICC
    np.random.seed(SEED)
    t_stat_zero_icc, p_value_zero_icc = simulation_continuous.simulate_continuous_trial(
        n_clusters=n_clusters_arm,
        cluster_size=cluster_size,
        icc=0.0, # Zero ICC
        mean1=mean1,
        mean2=mean2,
        std_dev=std_dev
    )
    assert isinstance(t_stat_zero_icc, (float, np.floating))
    assert isinstance(p_value_zero_icc, (float, np.floating))
    assert 0 <= p_value_zero_icc <= 1

    # Test with 'mixedlm' model
    np.random.seed(SEED)
    t_stat_mixedlm, p_value_mixedlm = simulation_continuous.simulate_continuous_trial(
        n_clusters=n_clusters_arm,
        cluster_size=cluster_size,
        icc=icc,
        mean1=mean1,
        mean2=mean2,
        std_dev=std_dev,
        analysis_model='mixedlm'
    )
    assert isinstance(t_stat_mixedlm, (float, np.floating))
    assert isinstance(p_value_mixedlm, (float, np.floating))
    assert 0 <= p_value_mixedlm <= 1

    # Test with 'mixedlm' model and return_details=True
    np.random.seed(SEED)
    t_stat_mixedlm_details, p_value_mixedlm_details, details = simulation_continuous.simulate_continuous_trial(
        n_clusters=n_clusters_arm,
        cluster_size=cluster_size,
        icc=icc,
        mean1=mean1,
        mean2=mean2,
        std_dev=std_dev,
        analysis_model='mixedlm',
        return_details=True
    )
    assert isinstance(t_stat_mixedlm_details, (float, np.floating))
    assert isinstance(p_value_mixedlm_details, (float, np.floating))
    assert 0 <= p_value_mixedlm_details <= 1
    assert isinstance(details, dict)
    assert 'converged' in details
    assert isinstance(details['converged'], bool)

    # Test with 'gee' model
    np.random.seed(SEED)
    t_stat_gee, p_value_gee = simulation_continuous.simulate_continuous_trial(
        n_clusters=n_clusters_arm,
        cluster_size=cluster_size,
        icc=icc,
        mean1=mean1,
        mean2=mean2,
        std_dev=std_dev,
        analysis_model='gee'
    )
    assert isinstance(t_stat_gee, (float, np.floating))
    assert isinstance(p_value_gee, (float, np.floating))
    assert 0 <= p_value_gee <= 1

    # Test with 'gee' model and return_details=True
    np.random.seed(SEED)
    t_stat_gee_details, p_value_gee_details, details_gee = simulation_continuous.simulate_continuous_trial(
        n_clusters=n_clusters_arm,
        cluster_size=cluster_size,
        icc=icc,
        mean1=mean1,
        mean2=mean2,
        std_dev=std_dev,
        analysis_model='gee',
        return_details=True
    )
    assert isinstance(t_stat_gee_details, (float, np.floating))
    assert isinstance(p_value_gee_details, (float, np.floating))
    assert 0 <= p_value_gee_details <= 1
    assert isinstance(details_gee, dict)
    assert 'converged' in details_gee
    assert isinstance(details_gee['converged'], bool) # GEE convergence is usually True unless an error occurs

    # Test with 'gee' model, return_details=True, and use_bias_correction=True
    np.random.seed(SEED)
    t_stat_gee_bias, p_value_gee_bias, details_gee_bias = simulation_continuous.simulate_continuous_trial(
        n_clusters=n_clusters_arm,
        cluster_size=cluster_size,
        icc=icc,
        mean1=mean1,
        mean2=mean2,
        std_dev=std_dev,
        analysis_model='gee',
        return_details=True,
        use_bias_correction=True
    )
    assert isinstance(t_stat_gee_bias, (float, np.floating))
    assert isinstance(p_value_gee_bias, (float, np.floating))
    assert 0 <= p_value_gee_bias <= 1
    assert isinstance(details_gee_bias, dict)
    assert 'converged' in details_gee_bias
    assert isinstance(details_gee_bias['converged'], bool)

    # Test with 'bayes' model
    if not simulation_continuous._STAN_AVAILABLE:
        pytest.skip("CmdStanPy or Stan installation not found, skipping Bayesian tests.")
    else:
        np.random.seed(SEED)
        t_stat_bayes, p_value_bayes = simulation_continuous.simulate_continuous_trial(
            n_clusters=n_clusters_arm,
            cluster_size=cluster_size, # Smaller cluster size for faster Bayes test
            icc=icc,
            mean1=mean1,
            mean2=mean2,
            std_dev=std_dev,
            analysis_model='bayes',
            bayes_draws=50, # Keep low for testing speed
            bayes_warmup=50 # Keep low for testing speed
        )
        assert isinstance(t_stat_bayes, (float, np.floating)) # Bayesian model might return effect estimate
        assert isinstance(p_value_bayes, (float, np.floating)) # Bayesian model might return posterior probability
        # For Bayesian, p_value is often P(effect > 0) or similar, so it's still in [0,1]
        assert 0 <= p_value_bayes <= 1

        # Test with 'bayes' model and return_details=True
        np.random.seed(SEED)
        t_stat_bayes_details, p_value_bayes_details, details_bayes = simulation_continuous.simulate_continuous_trial(
            n_clusters=n_clusters_arm,
            cluster_size=cluster_size,
            icc=icc,
            mean1=mean1,
            mean2=mean2,
            std_dev=std_dev,
            analysis_model='bayes',
            return_details=True,
            bayes_draws=55, # Slightly different to ensure re-run
            bayes_warmup=55
        )
        assert isinstance(t_stat_bayes_details, (float, np.floating))
        assert isinstance(p_value_bayes_details, (float, np.floating))
        assert 0 <= p_value_bayes_details <= 1
        assert isinstance(details_bayes, dict)
        assert 'model' in details_bayes
        assert details_bayes['model'] == 'bayes'
        assert 'converged' in details_bayes
        assert isinstance(details_bayes['converged'], bool)
        if not details_bayes['converged']:
            assert 'fallback' in details_bayes
            assert details_bayes['fallback'] is True
        # 'model_summary' is not currently returned by simulate_continuous_trial

    # TODO: Test 'mixedlm' with different lmm_method options and use_satterthwaite
    pass
