"""
Unit tests for analytical functions for binary outcomes in Cluster RCT designs.

This module tests the correctness of the functions in core.designs.cluster_rct.analytical_binary.
"""
import pytest
import numpy as np
import math
from scipy import stats # Required by one of the functions, good to have for potential extensions
from core.designs.cluster_rct import analytical_binary

def test_power_binary_cluster_rct():
    """Test power calculation for binary outcome in cluster RCT."""
    # Test with standard parameters
    result = analytical_binary.power_binary(
        n_clusters=10,
        cluster_size=20,
        icc=0.05,
        p1=0.5,
        p2=0.6,
        alpha=0.05
    )
    
    # Basic checks
    assert 0 <= result["power"] <= 1
    assert result["n_clusters"] == 10 # n_clusters is per arm
    assert result["cluster_size"] == 20
    assert result["icc"] == 0.05
    
    # Check that increasing number of clusters increases power
    result_higher_n = analytical_binary.power_binary(
        n_clusters=20,
        cluster_size=20,
        icc=0.05,
        p1=0.5,
        p2=0.6,
        alpha=0.05
    )
    assert result_higher_n["power"] > result["power"]
    
    # Check that increasing effect size increases power
    result_higher_effect = analytical_binary.power_binary(
        n_clusters=10,
        cluster_size=20,
        icc=0.05,
        p1=0.5,
        p2=0.7,
        alpha=0.05
    )
    assert result_higher_effect["power"] > result["power"]
    
    # Check that increasing ICC decreases power
    result_higher_icc = analytical_binary.power_binary(
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
    result = analytical_binary.sample_size_binary(
        p1=0.5,
        p2=0.6,
        icc=0.05,
        cluster_size=20,
        power=0.8,
        alpha=0.05
    )
    
    # Basic checks
    assert result["n_clusters"] > 0 # n_clusters is per arm
    # total_clusters is not directly returned, but n_clusters is per arm
    assert result["p1"] == 0.5
    assert result["p2"] == 0.6
    
    # Check that increasing power increases number of clusters
    result_higher_power = analytical_binary.sample_size_binary(
        p1=0.5,
        p2=0.6,
        icc=0.05,
        cluster_size=20,
        power=0.9,
        alpha=0.05
    )
    assert result_higher_power["n_clusters"] > result["n_clusters"]
    
    # Check that increasing effect size decreases number of clusters
    result_higher_effect = analytical_binary.sample_size_binary(
        p1=0.5,
        p2=0.7,
        icc=0.05,
        cluster_size=20,
        power=0.8,
        alpha=0.05
    )
    assert result_higher_effect["n_clusters"] < result["n_clusters"]
    
    # Check that increasing ICC increases number of clusters
    result_higher_icc = analytical_binary.sample_size_binary(
        p1=0.5,
        p2=0.6,
        icc=0.2,
        cluster_size=20,
        power=0.8,
        alpha=0.05
    )
    assert result_higher_icc["n_clusters"] > result["n_clusters"]


def test_min_detectable_effect_binary_cluster_rct():
    """Test minimum detectable effect calculation for binary outcome in cluster RCT."""
    # Test with standard parameters
    result = analytical_binary.min_detectable_effect_binary(
        n_clusters=10,
        cluster_size=20,
        icc=0.05,
        p1=0.5,
        power=0.8,
        alpha=0.05
    )
    
    # Basic checks
    assert result["mde"] > 0
    assert result["p2"] == result["p1"] + result["mde"]
    assert result["n_clusters"] == 10
    assert result["p1"] == 0.5
    
    # Check that increasing n_clusters decreases MDE
    result_higher_n = analytical_binary.min_detectable_effect_binary(
        n_clusters=20, # Higher n_clusters
        cluster_size=20,
        icc=0.05,
        p1=0.5,
        power=0.8,
        alpha=0.05
    )
    assert result_higher_n["mde"] < result["mde"]
    
    # Check that increasing power increases MDE (for a fixed n)
    result_higher_power = analytical_binary.min_detectable_effect_binary(
        n_clusters=10,
        cluster_size=20,
        icc=0.05,
        p1=0.5,
        power=0.9, # Higher power
        alpha=0.05
    )
    assert result_higher_power["mde"] > result["mde"]
    
    # Check that increasing ICC increases MDE
    result_higher_icc = analytical_binary.min_detectable_effect_binary(
        n_clusters=10,
        cluster_size=20,
        icc=0.2, # Higher ICC
        p1=0.5,
        power=0.8,
        alpha=0.05
    )
    assert result_higher_icc["mde"] > result["mde"]
