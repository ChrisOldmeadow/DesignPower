"""
Unit tests for power and sample size calculation functions.

This module tests the correctness of the functions in the core.power module.
"""
import pytest
import numpy as np
from core.power import (
    sample_size_difference_in_means,
    power_difference_in_means,
    power_binary_cluster_rct,
    sample_size_binary_cluster_rct,
    min_detectable_effect_binary_cluster_rct
)


def test_sample_size_difference_in_means():
    """Test sample size calculation for difference in means."""
    # Test with standard parameters
    result = sample_size_difference_in_means(
        delta=0.5,
        std_dev=1.0,
        power=0.8,
        alpha=0.05,
        allocation_ratio=1.0
    )
    
    # Basic checks
    assert result["n1"] > 0
    assert result["n2"] > 0
    assert result["total_n"] == result["n1"] + result["n2"]
    assert result["parameters"]["delta"] == 0.5
    
    # Check that increasing power increases sample size
    result_higher_power = sample_size_difference_in_means(
        delta=0.5,
        std_dev=1.0,
        power=0.9,
        alpha=0.05,
        allocation_ratio=1.0
    )
    assert result_higher_power["total_n"] > result["total_n"]
    
    # Check that increasing effect size decreases sample size
    result_higher_effect = sample_size_difference_in_means(
        delta=1.0,
        std_dev=1.0,
        power=0.8,
        alpha=0.05,
        allocation_ratio=1.0
    )
    assert result_higher_effect["total_n"] < result["total_n"]


def test_power_difference_in_means():
    """Test power calculation for difference in means."""
    # Test with standard parameters
    result = power_difference_in_means(
        n1=64,
        n2=64,
        delta=0.5,
        std_dev=1.0,
        alpha=0.05
    )
    
    # Basic checks
    assert 0 <= result["power"] <= 1
    assert result["parameters"]["n1"] == 64
    assert result["parameters"]["delta"] == 0.5
    
    # Check that increasing sample size increases power
    result_higher_n = power_difference_in_means(
        n1=100,
        n2=100,
        delta=0.5,
        std_dev=1.0,
        alpha=0.05
    )
    assert result_higher_n["power"] > result["power"]
    
    # Check that increasing effect size increases power
    result_higher_effect = power_difference_in_means(
        n1=64,
        n2=64,
        delta=0.8,
        std_dev=1.0,
        alpha=0.05
    )
    assert result_higher_effect["power"] > result["power"]


def test_power_binary_cluster_rct():
    """Test power calculation for binary outcome in cluster RCT."""
    # Test with standard parameters
    result = power_binary_cluster_rct(
        n_clusters=10,
        cluster_size=20,
        icc=0.05,
        p1=0.5,
        p2=0.6,
        alpha=0.05
    )
    
    # Basic checks
    assert 0 <= result["power"] <= 1
    assert result["parameters"]["n_clusters"] == 10
    assert result["parameters"]["cluster_size"] == 20
    assert result["parameters"]["icc"] == 0.05
    
    # Check that increasing number of clusters increases power
    result_higher_n = power_binary_cluster_rct(
        n_clusters=20,
        cluster_size=20,
        icc=0.05,
        p1=0.5,
        p2=0.6,
        alpha=0.05
    )
    assert result_higher_n["power"] > result["power"]
    
    # Check that increasing effect size increases power
    result_higher_effect = power_binary_cluster_rct(
        n_clusters=10,
        cluster_size=20,
        icc=0.05,
        p1=0.5,
        p2=0.7,
        alpha=0.05
    )
    assert result_higher_effect["power"] > result["power"]
    
    # Check that increasing ICC decreases power
    result_higher_icc = power_binary_cluster_rct(
        n_clusters=10,
        cluster_size=20,
        icc=0.2,
        p1=0.5,
        p2=0.6,
        alpha=0.05
    )
    assert result_higher_icc["power"] < result["power"]


def test_sample_size_binary_cluster_rct():
    """Test sample size calculation for binary outcome in cluster RCT."""
    # Test with standard parameters
    result = sample_size_binary_cluster_rct(
        p1=0.5,
        p2=0.6,
        icc=0.05,
        cluster_size=20,
        power=0.8,
        alpha=0.05
    )
    
    # Basic checks
    assert result["n_clusters_per_arm"] > 0
    assert result["total_clusters"] == 2 * result["n_clusters_per_arm"]
    assert result["parameters"]["p1"] == 0.5
    assert result["parameters"]["p2"] == 0.6
    
    # Check that increasing power increases number of clusters
    result_higher_power = sample_size_binary_cluster_rct(
        p1=0.5,
        p2=0.6,
        icc=0.05,
        cluster_size=20,
        power=0.9,
        alpha=0.05
    )
    assert result_higher_power["n_clusters_per_arm"] > result["n_clusters_per_arm"]
    
    # Check that increasing effect size decreases number of clusters
    result_higher_effect = sample_size_binary_cluster_rct(
        p1=0.5,
        p2=0.7,
        icc=0.05,
        cluster_size=20,
        power=0.8,
        alpha=0.05
    )
    assert result_higher_effect["n_clusters_per_arm"] < result["n_clusters_per_arm"]
    
    # Check that increasing ICC increases number of clusters
    result_higher_icc = sample_size_binary_cluster_rct(
        p1=0.5,
        p2=0.6,
        icc=0.2,
        cluster_size=20,
        power=0.8,
        alpha=0.05
    )
    assert result_higher_icc["n_clusters_per_arm"] > result["n_clusters_per_arm"]


def test_min_detectable_effect_binary_cluster_rct():
    """Test minimum detectable effect calculation for binary outcome in cluster RCT."""
    # Test with standard parameters
    result = min_detectable_effect_binary_cluster_rct(
        n_clusters=10,
        cluster_size=20,
        icc=0.05,
        p1=0.5,
        power=0.8,
        alpha=0.05
    )
    
    # Basic checks
    assert 0 <= result["p2"] <= 1
    assert result["mde"] > 0
    assert result["parameters"]["n_clusters"] == 10
    assert result["parameters"]["p1"] == 0.5
    
    # Check that increasing number of clusters decreases MDE
    result_higher_n = min_detectable_effect_binary_cluster_rct(
        n_clusters=20,
        cluster_size=20,
        icc=0.05,
        p1=0.5,
        power=0.8,
        alpha=0.05
    )
    assert result_higher_n["mde"] < result["mde"]
    
    # Check that increasing power increases MDE
    result_higher_power = min_detectable_effect_binary_cluster_rct(
        n_clusters=10,
        cluster_size=20,
        icc=0.05,
        p1=0.5,
        power=0.9,
        alpha=0.05
    )
    assert result_higher_power["mde"] > result["mde"]
    
    # Check that increasing ICC increases MDE
    result_higher_icc = min_detectable_effect_binary_cluster_rct(
        n_clusters=10,
        cluster_size=20,
        icc=0.2,
        p1=0.5,
        power=0.8,
        alpha=0.05
    )
    assert result_higher_icc["mde"] > result["mde"]
