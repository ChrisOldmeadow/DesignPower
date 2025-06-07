"""
Validation testing module for DesignPower.

This module provides comprehensive validation against:
1. Published literature benchmarks
2. Cross-validation with established software
3. Known analytical solutions
4. Edge case testing
"""

from .literature_benchmarks import (
    LiteratureBenchmark,
    validate_benchmark,
    run_all_benchmarks,
    FLEISS_1973_BENCHMARKS,
    COHEN_1988_BENCHMARKS,
    SCHOENFELD_1981_BENCHMARKS,
    DONNER_KLAR_2000_BENCHMARKS,
    AHERN_2001_BENCHMARKS,
    SIMON_1989_BENCHMARKS
)

__all__ = [
    'LiteratureBenchmark',
    'validate_benchmark', 
    'run_all_benchmarks',
    'FLEISS_1973_BENCHMARKS',
    'COHEN_1988_BENCHMARKS',
    'SCHOENFELD_1981_BENCHMARKS',
    'DONNER_KLAR_2000_BENCHMARKS',
    'AHERN_2001_BENCHMARKS',
    'SIMON_1989_BENCHMARKS'
]