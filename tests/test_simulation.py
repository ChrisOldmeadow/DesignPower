"""
Unit tests for simulation-based estimation functions.

This module tests the correctness of the functions in the core.simulation module.
"""
import pytest
import numpy as np
from core.simulation import (
    simulate_parallel_rct,
    simulate_cluster_rct,
    simulate_stepped_wedge,
    simulate_binary_cluster_rct
)


def test_simulate_parallel_rct():
    """Test simulation of parallel RCT with continuous outcome."""
    # Test with a large effect size that should give high power
    result = simulate_parallel_rct(
        n1=30,
        n2=30,
        mean1=0,
        mean2=1.0,  # Large effect size (1 SD)
        std_dev=1.0,
        nsim=100,  # Small for faster tests
        alpha=0.05
    )
    
    # Basic checks
    assert 0 <= result["power"] <= 1
    assert result["parameters"]["n1"] == 30
    assert result["parameters"]["n2"] == 30
    assert result["parameters"]["mean1"] == 0
    assert result["parameters"]["mean2"] == 1.0
    
    # With such a large effect size, power should be reasonably high
    assert result["power"] > 0.7
    
    # Test with a tiny effect size that should give low power
    result_low = simulate_parallel_rct(
        n1=30,
        n2=30,
        mean1=0,
        mean2=0.1,  # Small effect size (0.1 SD)
        std_dev=1.0,
        nsim=100,
        alpha=0.05
    )
    
    # Power should be much lower with smaller effect
    assert result_low["power"] < result["power"]


def test_simulate_cluster_rct():
    """Test simulation of cluster RCT with continuous outcome."""
    # Test with a large effect size that should give high power
    result = simulate_cluster_rct(
        n_clusters=10,
        cluster_size=10,
        icc=0.05,
        mean1=0,
        mean2=0.8,  # Large effect size
        std_dev=1.0,
        nsim=100,  # Small for faster tests
        alpha=0.05
    )
    
    # Basic checks
    assert 0 <= result["power"] <= 1
    assert result["parameters"]["n_clusters"] == 10
    assert result["parameters"]["cluster_size"] == 10
    assert result["parameters"]["icc"] == 0.05
    
    # Test with increased ICC (should decrease power)
    result_high_icc = simulate_cluster_rct(
        n_clusters=10,
        cluster_size=10,
        icc=0.3,  # Higher ICC
        mean1=0,
        mean2=0.8,
        std_dev=1.0,
        nsim=100,
        alpha=0.05
    )
    
    # Higher ICC should result in lower power
    assert result_high_icc["power"] <= result["power"] + 0.1  # Allow some random variation


def test_simulate_stepped_wedge():
    """Test simulation of stepped wedge design."""
    # Test with standard parameters
    result = simulate_stepped_wedge(
        clusters=8,
        steps=4,
        individuals_per_cluster=5,
        icc=0.05,
        treatment_effect=0.5,
        std_dev=1.0,
        nsim=50,  # Very small for faster tests
        alpha=0.05
    )
    
    # Basic checks
    assert 0 <= result["power"] <= 1
    assert result["parameters"]["clusters"] == 8
    assert result["parameters"]["steps"] == 4
    assert result["parameters"]["individuals_per_cluster"] == 5
    assert result["parameters"]["icc"] == 0.05
    
    # Test with increased treatment effect (should increase power)
    result_high_effect = simulate_stepped_wedge(
        clusters=8,
        steps=4,
        individuals_per_cluster=5,
        icc=0.05,
        treatment_effect=1.0,  # Higher effect
        std_dev=1.0,
        nsim=50,
        alpha=0.05
    )
    
    # Higher effect should result in higher power
    # Due to small simulation count, we allow for some random variation
    assert result_high_effect["power"] >= result["power"] - 0.1


def test_simulate_binary_cluster_rct():
    """Test simulation of cluster RCT with binary outcome."""
    # Test with standard parameters
    result = simulate_binary_cluster_rct(
        n_clusters=10,
        cluster_size=20,
        icc=0.05,
        p1=0.5,
        p2=0.7,  # Large effect
        nsim=100,  # Small for faster tests
        alpha=0.05
    )
    
    # Basic checks
    assert 0 <= result["power"] <= 1
    assert result["parameters"]["n_clusters"] == 10
    assert result["parameters"]["cluster_size"] == 20
    assert result["parameters"]["icc"] == 0.05
    assert result["parameters"]["p1"] == 0.5
    assert result["parameters"]["p2"] == 0.7
    
    # Test with smaller effect size (should decrease power)
    result_small_effect = simulate_binary_cluster_rct(
        n_clusters=10,
        cluster_size=20,
        icc=0.05,
        p1=0.5,
        p2=0.6,  # Smaller effect
        nsim=100,
        alpha=0.05
    )
    
    # Smaller effect should result in lower power
    assert result_small_effect["power"] <= result["power"] + 0.1  # Allow some random variation
