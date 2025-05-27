import pytest
import numpy as np
import math
from scipy import stats
from core.designs.cluster_rct import analytical_continuous

def test_power_continuous_cluster_rct():
    """Test power calculation for continuous outcome in cluster RCT."""
    # Test with standard parameters
    result = analytical_continuous.power_continuous(
        n_clusters=20,
        cluster_size=50,
        icc=0.05,
        mean1=10,
        mean2=12,
        std_dev=5,
        alpha=0.05
    )

    # Basic checks
    assert 0 <= result["power"] <= 1
    assert result["n_clusters"] == 20
    assert result["cluster_size"] == 50
    assert result["icc"] == 0.05
    assert result["mean1"] == 10
    assert result["mean2"] == 12
    assert result["std_dev"] == 5

    # Check that increasing number of clusters increases power
    result_higher_n_clusters = analytical_continuous.power_continuous(
        n_clusters=30, # Increased
        cluster_size=50,
        icc=0.05,
        mean1=10,
        mean2=12,
        std_dev=5,
        alpha=0.05
    )
    assert result_higher_n_clusters["power"] > result["power"]

    # Check that increasing effect size (difference between means) increases power
    result_higher_effect = analytical_continuous.power_continuous(
        n_clusters=20,
        cluster_size=50,
        icc=0.05,
        mean1=10,
        mean2=13, # Increased difference
        std_dev=5,
        alpha=0.05
    )
    assert result_higher_effect["power"] > result["power"]

    # Check that increasing ICC decreases power
    result_higher_icc = analytical_continuous.power_continuous(
        n_clusters=20,
        cluster_size=50,
        icc=0.1, # Increased
        mean1=10,
        mean2=12,
        std_dev=5,
        alpha=0.05
    )
    assert result_higher_icc["power"] < result["power"]

    # Test with zero ICC (should behave like a simple t-test for individuals)
    result_zero_icc = analytical_continuous.power_continuous(
        n_clusters=20,
        cluster_size=50,
        icc=0.0, # Zero ICC
        mean1=10,
        mean2=12,
        std_dev=5,
        alpha=0.05
    )
    # Power with ICC=0 should be higher than with ICC > 0, all else equal
    assert result_zero_icc["power"] > result["power"]
    # Compare with a standard power calculation for individuals (approximate)
    # For a standard t-test, n_per_group = n_clusters * cluster_size
    # This is an approximation as the cluster RCT formula uses effective N
    # but with ICC=0, DEFF=1, so effective_n = total_n_per_arm
    from statsmodels.stats.power import tt_ind_solve_power
    power_individuals = tt_ind_solve_power(
        effect_size=abs(10-12)/5,
        nobs1=20*50,
        alpha=0.05,
        power=None, # Solve for power
        ratio=1.0
    )
    # Handle cases where tt_ind_solve_power might return nan for power very close to 1.0
    if np.isnan(power_individuals):
        assert result_zero_icc["power"] > 0.9999, \
            "Expected power to be > 0.9999 when tt_ind_solve_power returns NaN"
    else:
        assert np.isclose(result_zero_icc["power"], power_individuals, atol=0.001)

def test_sample_size_continuous_cluster_rct():
    """Test sample size calculation for continuous outcome in cluster RCT."""
    # Test with standard parameters
    result = analytical_continuous.sample_size_continuous(
        mean1=10,
        mean2=12,
        std_dev=5,
        icc=0.05,
        cluster_size=50,
        power=0.8,
        alpha=0.05
    )

    # Basic checks
    assert result["n_clusters"] > 0
    assert result["cluster_size"] == 50
    assert result["icc"] == 0.05
    assert result["mean1"] == 10
    assert result["mean2"] == 12
    assert result["std_dev"] == 5
    assert result["target_power"] == 0.8
    # Achieved power should be >= target power and reasonably close
    assert result["achieved_power"] >= result["target_power"]
    assert np.isclose(result["achieved_power"], result["target_power"], atol=0.05) # Allow some tolerance due to ceiling

    # Check that increasing target power increases n_clusters
    result_higher_power = analytical_continuous.sample_size_continuous(
        mean1=10,
        mean2=12,
        std_dev=5,
        icc=0.05,
        cluster_size=50,
        power=0.9, # Increased
        alpha=0.05
    )
    assert result_higher_power["n_clusters"] > result["n_clusters"]

    # Check that increasing effect size (difference between means) decreases n_clusters
    result_higher_effect = analytical_continuous.sample_size_continuous(
        mean1=10,
        mean2=13, # Increased difference
        std_dev=5,
        icc=0.05,
        cluster_size=50,
        power=0.8,
        alpha=0.05
    )
    assert result_higher_effect["n_clusters"] < result["n_clusters"]

    # Check that increasing ICC increases n_clusters
    result_higher_icc = analytical_continuous.sample_size_continuous(
        mean1=10,
        mean2=12,
        std_dev=5,
        icc=0.1, # Increased
        cluster_size=50,
        power=0.8,
        alpha=0.05
    )
    assert result_higher_icc["n_clusters"] > result["n_clusters"]

    # Test with zero ICC
    result_zero_icc = analytical_continuous.sample_size_continuous(
        mean1=10,
        mean2=12,
        std_dev=5,
        icc=0.0, # Zero ICC
        cluster_size=50,
        power=0.8,
        alpha=0.05
    )
    assert result_zero_icc["n_clusters"] < result["n_clusters"]
    # Compare with a standard sample size calculation for individuals (approximate)
    from statsmodels.stats.power import tt_ind_solve_power
    n_individuals = tt_ind_solve_power(
        effect_size=abs(10-12)/5,
        nobs1=None, # Solve for nobs1
        alpha=0.05,
        power=0.8,
        ratio=1.0
    )
    # Expected clusters = ceil(n_individuals / cluster_size)
    expected_clusters_icc0 = math.ceil(n_individuals / 50)
    assert result_zero_icc["n_clusters"] == expected_clusters_icc0

def test_min_detectable_effect_continuous_cluster_rct():
    """Test MDE calculation for continuous outcome in cluster RCT."""
    # Test with standard parameters
    result = analytical_continuous.min_detectable_effect_continuous(
        n_clusters=20,
        cluster_size=50,
        icc=0.05,
        std_dev=5,
        power=0.8,
        alpha=0.05
    )

    # Basic checks
    assert result["mde"] > 0
    assert result["standardized_mde"] > 0
    assert result["n_clusters"] == 20
    assert result["cluster_size"] == 50
    assert result["icc"] == 0.05
    assert result["std_dev"] == 5
    assert result["power"] == 0.8

    # Check that increasing n_clusters decreases MDE
    result_higher_n_clusters = analytical_continuous.min_detectable_effect_continuous(
        n_clusters=30, # Increased
        cluster_size=50,
        icc=0.05,
        std_dev=5,
        power=0.8,
        alpha=0.05
    )
    assert result_higher_n_clusters["mde"] < result["mde"]

    # Check that increasing power increases MDE (for a fixed n_clusters)
    result_higher_power = analytical_continuous.min_detectable_effect_continuous(
        n_clusters=20,
        cluster_size=50,
        icc=0.05,
        std_dev=5,
        power=0.9, # Increased
        alpha=0.05
    )
    assert result_higher_power["mde"] > result["mde"]

    # Check that increasing ICC increases MDE
    result_higher_icc = analytical_continuous.min_detectable_effect_continuous(
        n_clusters=20,
        cluster_size=50,
        icc=0.1, # Increased
        std_dev=5,
        power=0.8,
        alpha=0.05
    )
    assert result_higher_icc["mde"] > result["mde"]

    # Test with zero ICC
    result_zero_icc = analytical_continuous.min_detectable_effect_continuous(
        n_clusters=20,
        cluster_size=50,
        icc=0.0, # Zero ICC
        std_dev=5,
        power=0.8,
        alpha=0.05
    )
    assert result_zero_icc["mde"] < result["mde"]
    # Compare with a standard MDE calculation for individuals (approximate)
    from statsmodels.stats.power import tt_ind_solve_power
    mde_individuals_std = tt_ind_solve_power(
        effect_size=None, # Solve for effect_size
        nobs1=20*50,
        alpha=0.05,
        power=0.8,
        ratio=1.0
    )
    mde_individuals_raw = mde_individuals_std * 5 # Multiply by std_dev
    assert np.isclose(result_zero_icc["mde"], mde_individuals_raw, atol=0.001)
