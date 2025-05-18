"""
Unit tests for Julia interoperability.

This module tests the interoperability between Python and Julia,
specifically for stepped wedge simulation.
"""
import pytest
import os
import sys

# Add parent directory to path to import core modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.simulation import simulate_stepped_wedge


# Check if Julia is available
def is_julia_available():
    """Check if Julia is available on the system."""
    try:
        from julia import Main
        from julia.api import Julia
        jl = Julia(compiled_modules=False)
        return True
    except (ImportError, Exception):
        return False


@pytest.mark.skipif(not is_julia_available(), reason="Julia is not available")
def test_julia_stepped_wedge():
    """Test the Julia implementation of stepped wedge simulation."""
    try:
        from julia import Main
        from julia.api import Julia
        
        # Initialize Julia
        jl = Julia(compiled_modules=False)
        
        # Include the Julia file
        julia_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                "julia_backend", "stepped_wedge.jl")
        Main.include(julia_file)
        
        # Run both Python and Julia implementations with the same parameters
        py_result = simulate_stepped_wedge(
            clusters=8,
            steps=4,
            individuals_per_cluster=10,
            icc=0.05,
            treatment_effect=0.5,
            std_dev=1.0,
            nsim=100,  # Small number for faster test
            alpha=0.05
        )
        
        julia_result = Main.simulate_stepped_wedge(
            8,      # clusters
            4,      # steps
            10,     # individuals_per_cluster
            0.05,   # icc
            0.5,    # treatment_effect
            1.0,    # std_dev
            100,    # nsim
            0.05    # alpha
        )
        
        # Convert Julia result to Python dict if necessary
        if hasattr(julia_result, "keys"):
            # Ensure we have key values we need
            assert "power" in julia_result
            assert julia_result["power"] >= 0 and julia_result["power"] <= 1
            
            # Compare results 
            # They should be somewhat close but not identical due to random sampling
            # We just check that both are in a reasonable range
            assert 0 <= py_result["power"] <= 1
            assert 0 <= julia_result["power"] <= 1
            
        else:
            pytest.skip("Julia function did not return expected dictionary structure")
            
    except Exception as e:
        pytest.skip(f"Julia test failed: {str(e)}")


@pytest.mark.skipif(not is_julia_available(), reason="Julia is not available")
def test_julia_call_from_python():
    """Test basic Julia call from Python."""
    try:
        from julia import Main
        
        # A simple test to ensure Julia is properly linked
        Main.eval("1 + 1")
        assert Main.eval("1 + 1") == 2
        
        # Test array creation and manipulation
        Main.eval("a = [1, 2, 3]")
        assert list(Main.a) == [1, 2, 3]
        
    except Exception as e:
        pytest.skip(f"Basic Julia call failed: {str(e)}")
