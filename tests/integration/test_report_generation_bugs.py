#!/usr/bin/env python3
"""
Integration tests to reproduce and fix HTML report generation and CLI script bugs.

These tests specifically target the user-reported issues:
1. HTML report generation errors with None values in format strings
2. CLI script generation errors for binary power analytical parallel
"""

import unittest
import sys
import os

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

from core.utils.report_generator import generate_report, generate_power_report
from app.components.parallel_rct.cli_generation import generate_cli_code_parallel_binary
from app.components.parallel_rct.calculations import calculate_parallel_continuous, calculate_parallel_binary


class TestHTMLReportGenerationBugs(unittest.TestCase):
    """Test HTML report generation to catch format string errors."""
    
    def test_continuous_power_report_with_none_effect_size(self):
        """Test continuous power report generation when effect_size is None."""
        # Simulate params and results that might cause the None format error
        params = {
            'n1': 50,
            'n2': 50,
            'mean1': 10.0,
            'mean2': 12.0,
            'std_dev': 3.0,
            'alpha': 0.05,
            'method': 'analytical',
            'hypothesis_type': 'Superiority',
            'calculation_type': 'Power'
        }
        
        # Results that might have None effect_size
        results = {
            'power': 0.85,
            'n1': 50,
            'n2': 50,
            'effect_size': None  # This could cause the format error
        }
        
        # This should not raise "unsupported format string passed to NoneType.format" error
        try:
            report = generate_power_report(results, params, 'Parallel RCT', 'Continuous Outcome')
            self.assertIsInstance(report, str)
            self.assertGreater(len(report), 0)
        except Exception as e:
            self.fail(f"Report generation failed with error: {e}")
    
    def test_continuous_power_real_calculation(self):
        """Test with real calculation results to reproduce the actual bug."""
        params = {
            'calculation_type': 'Power',
            'hypothesis_type': 'Superiority',
            'mean1': 10.0,
            'mean2': 12.0,
            'std_dev': 3.0,
            'n1': 50,
            'n2': 50,
            'alpha': 0.05,
            'allocation_ratio': 1.0,
            'method': 'analytical'
        }
        
        # Get real calculation results
        results = calculate_parallel_continuous(params)
        
        # Try to generate HTML report with real results
        try:
            report = generate_report(results, params, 'Parallel RCT', 'Continuous Outcome')
            self.assertIsInstance(report, str)
            self.assertGreater(len(report), 0)
            # Should not contain None in the formatted text
            self.assertNotIn('None', report)
        except Exception as e:
            self.fail(f"HTML report generation failed with real calculation results: {e}")
    
    def test_binary_power_report(self):
        """Test binary power report generation."""
        params = {
            'calculation_type': 'Power',
            'hypothesis_type': 'Superiority',
            'p1': 0.3,
            'p2': 0.5,
            'n1': 50,
            'n2': 50,
            'alpha': 0.05,
            'allocation_ratio': 1.0,
            'method': 'analytical',
            'test_type': 'Normal Approximation'
        }
        
        # Get real calculation results
        results = calculate_parallel_binary(params)
        
        # Try to generate HTML report
        try:
            report = generate_report(results, params, 'Parallel RCT', 'Binary Outcome')
            self.assertIsInstance(report, str)
            self.assertGreater(len(report), 0)
        except Exception as e:
            self.fail(f"Binary HTML report generation failed: {e}")


class TestCLIScriptGenerationBugs(unittest.TestCase):
    """Test CLI script generation to catch errors."""
    
    def test_binary_power_analytical_parallel_cli(self):
        """Test CLI script generation for binary power analytical parallel."""
        params = {
            'calculation_type': 'Power',
            'hypothesis_type': 'Superiority',
            'p1': 0.3,
            'p2': 0.5,
            'n1': 50,
            'n2': 50,
            'alpha': 0.05,
            'allocation_ratio': 1.0,
            'method': 'analytical',
            'test_type': 'Normal Approximation'
        }
        
        # This should not raise an error
        try:
            cli_code = generate_cli_code_parallel_binary(params)
            self.assertIsInstance(cli_code, str)
            self.assertGreater(len(cli_code), 0)
            # Should contain valid Python code
            self.assertIn('import', cli_code)
            self.assertIn('power_binary', cli_code)
        except Exception as e:
            self.fail(f"CLI script generation failed for binary power analytical parallel: {e}")
    
    def test_binary_sample_size_cli(self):
        """Test CLI script generation for binary sample size."""
        params = {
            'calculation_type': 'Sample Size',
            'hypothesis_type': 'Superiority',
            'p1': 0.3,
            'p2': 0.5,
            'power': 0.8,
            'alpha': 0.05,
            'allocation_ratio': 1.0,
            'method': 'analytical',
            'test_type': 'Normal Approximation'
        }
        
        try:
            cli_code = generate_cli_code_parallel_binary(params)
            self.assertIsInstance(cli_code, str)
            self.assertGreater(len(cli_code), 0)
            self.assertIn('sample_size_binary', cli_code)
        except Exception as e:
            self.fail(f"CLI script generation failed for binary sample size: {e}")
    
    def test_binary_non_inferiority_cli(self):
        """Test CLI script generation for binary non-inferiority."""
        params = {
            'calculation_type': 'Sample Size',
            'hypothesis_type': 'Non-Inferiority',
            'p1': 0.7,
            'non_inferiority_margin': 0.1,
            'power': 0.8,
            'alpha': 0.05,
            'allocation_ratio': 1.0,
            'method': 'analytical',
            'non_inferiority_direction': 'higher'
        }
        
        try:
            cli_code = generate_cli_code_parallel_binary(params)
            self.assertIsInstance(cli_code, str)
            self.assertGreater(len(cli_code), 0)
            self.assertIn('non_inferiority', cli_code)
        except Exception as e:
            self.fail(f"CLI script generation failed for binary non-inferiority: {e}")


class TestReportRobustness(unittest.TestCase):
    """Test report generation with edge cases and missing values."""
    
    def test_missing_parameters(self):
        """Test report generation with missing parameters."""
        # Minimal params that might be missing some keys
        params = {
            'calculation_type': 'Power',
            'method': 'analytical'
        }
        
        results = {
            'power': 0.8
        }
        
        try:
            report = generate_report(results, params, 'Parallel RCT', 'Continuous Outcome')
            self.assertIsInstance(report, str)
        except Exception as e:
            self.fail(f"Report generation failed with minimal params: {e}")
    
    def test_none_values_in_results(self):
        """Test report generation when results contain None values."""
        params = {
            'calculation_type': 'Power',
            'n1': 50,
            'n2': 50,
            'mean1': 10.0,
            'mean2': 12.0,
            'alpha': 0.05,
            'method': 'analytical'
        }
        
        results = {
            'power': 0.8,
            'effect_size': None,  # Potential problem
            'n1': None,           # Potential problem
            'confidence_interval': None  # Potential problem
        }
        
        try:
            report = generate_report(results, params, 'Parallel RCT', 'Continuous Outcome')
            self.assertIsInstance(report, str)
            # Should handle None values gracefully
            self.assertNotIn('None.', report)  # Check for "None.2f" type errors
        except Exception as e:
            self.fail(f"Report generation failed with None values: {e}")


class TestCLIFormatStringFixes(unittest.TestCase):
    """Test that CLI generation fixes work correctly."""
    
    def test_survival_cli_with_safe_formatting(self):
        """Test survival CLI generation uses safe formatting."""
        from app.components.parallel_rct.cli_generation import generate_cli_code_parallel_survival
        
        params = {
            'calculation_type': 'Power',
            'hypothesis_type': 'Superiority',
            'median1': 12,
            'median2': 18,
            'n1': 50,
            'n2': 50,
            'enrollment_period': 12,
            'follow_up_period': 24,
            'dropout_rate': 0.1,
            'alpha': 0.05,
            'method': 'analytical'
        }
        
        try:
            cli_code = generate_cli_code_parallel_survival(params)
            self.assertIsInstance(cli_code, str)
            self.assertGreater(len(cli_code), 0)
            # Should not contain the old dangerous format patterns
            self.assertNotIn("'N/A'):.3f", cli_code)
            self.assertNotIn("'N/A'):.0f", cli_code)
        except Exception as e:
            self.fail(f"Survival CLI script generation failed: {e}")
    
    def test_continuous_cli_with_safe_formatting(self):
        """Test continuous CLI generation uses safe formatting."""
        from app.components.parallel_rct.cli_generation import generate_cli_code_parallel_continuous
        
        params = {
            'calculation_type': 'Power',
            'hypothesis_type': 'Superiority',
            'mean1': 10.0,
            'mean2': 12.0,
            'std_dev': 3.0,
            'n1': 50,
            'n2': 50,
            'alpha': 0.05,
            'method': 'analytical'
        }
        
        try:
            cli_code = generate_cli_code_parallel_continuous(params)
            self.assertIsInstance(cli_code, str)
            self.assertGreater(len(cli_code), 0)
            # Should not contain the old dangerous format patterns
            self.assertNotIn("'N/A'):.3f", cli_code)
        except Exception as e:
            self.fail(f"Continuous CLI script generation failed: {e}")
    
    def test_binary_power_simulation_cli_script(self):
        """Test binary power simulation CLI script generation specifically."""
        from app.components.parallel_rct.cli_generation import generate_cli_code_parallel_binary
        
        params = {
            'calculation_type': 'Power',
            'hypothesis_type': 'Superiority',
            'p1': 0.3,
            'p2': 0.5,
            'n1': 50,
            'n2': 50,
            'alpha': 0.05,
            'allocation_ratio': 1.0,
            'method': 'simulation',  # Specifically test simulation
            'test_type': 'Normal Approximation',
            'nsim': 1000,
            'seed': 42
        }
        
        try:
            cli_code = generate_cli_code_parallel_binary(params)
            self.assertIsInstance(cli_code, str)
            self.assertGreater(len(cli_code), 0)
            # Should contain simulation-specific elements
            self.assertIn('simulation', cli_code.lower())
            self.assertIn('nsim', cli_code)
            # Should be executable Python code
            self.assertIn('import', cli_code)
            self.assertIn('power_binary_sim', cli_code)
        except Exception as e:
            self.fail(f"Binary power simulation CLI script generation failed: {e}")


if __name__ == "__main__":
    # Run all tests
    unittest.main(verbosity=2)