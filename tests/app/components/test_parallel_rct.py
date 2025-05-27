import pytest
from app.components.parallel_rct import calculate_parallel_survival

# Test cases for calculate_parallel_survival

def test_calculate_parallel_survival_analytical_superiority_power_two_sided():
    """Test analytical power calculation for survival outcome, superiority, two-sided."""
    params = {
        "calculation_type": "Power",
        "outcome_type": "Survival",
        "method": "Analytical",
        "hypothesis_type": "Superiority",
        "median_survival1": 12,  # Control group median survival
        "hr": 0.75,              # Assumed hazard ratio (experimental / control)
        "accrual_time": 12,      # Enrollment period in months
        "follow_up_time": 18,    # Follow-up period in months
        "dropout_rate": 0.1,     # Annual dropout rate
        "alpha": 0.05,
        "n1": 100,               # Sample size group 1
        "n2": 100,               # Sample size group 2
        "allocation_ratio": 1.0,
        "sides": 2
    }
    result = calculate_parallel_survival(params)

    assert "power" in result
    assert 0 <= result["power"] <= 1, f"Power out of expected range: {result.get('power')}"
    assert "events" in result
    assert result.get("events", -1) >= 0, f"Events should be non-negative: {result.get('events')}"
    assert "error" not in result, f"Unexpected error: {result.get('error')}"
    assert result.get("median_survival1_param") == params["median_survival1"]
    assert result.get("hr_param") == params["hr"]
    assert result.get("median_survival2_derived") == pytest.approx(params["median_survival1"] / params["hr"])
    assert result.get("method_param") == params["method"]
    assert result.get("hypothesis_type_param") == params["hypothesis_type"]
    assert result.get("calculation_type_param") == params["calculation_type"]

# --- Add more test cases below ---

def test_calculate_parallel_survival_analytical_non_inferiority_power():
    """Test analytical power calculation for survival outcome, non-inferiority."""
    params = {
        "calculation_type": "Power",
        "outcome_type": "Survival",
        "method": "Analytical",
        "hypothesis_type": "Non-Inferiority",
        "median_survival1": 15,          # Control group median survival
        "non_inferiority_margin_hr": 1.3, # Non-inferiority margin for HR
        "assumed_true_hr": 1.0,          # Assumed true HR (experimental / control)
        "accrual_time": 12,              # Enrollment period in months
        "follow_up_time": 24,            # Follow-up period in months
        "dropout_rate": 0.05,            # Annual dropout rate
        "alpha": 0.025,                  # One-sided alpha for NI (often 0.025 for 95% CI)
        "n1": 150,                       # Sample size group 1
        "n2": 150,                       # Sample size group 2
        "allocation_ratio": 1.0,
        "sides": 1 # For NI, this should effectively be 1, even if UI passes 2, function should use 1 for NI analytical
    }
    result = calculate_parallel_survival(params)

    assert "power" in result
    assert 0 <= result["power"] <= 1, f"Power out of expected range: {result.get('power')}"
    assert "events" in result
    assert result.get("events", -1) >= 0, f"Events should be non-negative: {result.get('events')}"
    assert "error" not in result, f"Unexpected error: {result.get('error')}"
    assert result.get("median_survival1_param") == params["median_survival1"]
    assert result.get("non_inferiority_margin_hr_param") == params["non_inferiority_margin_hr"]
    assert result.get("assumed_true_hr_param") == params["assumed_true_hr"]
    # For NI, median_survival2_derived is based on assumed_true_hr, not the margin
    assert result.get("median_survival2_derived") == pytest.approx(params["median_survival1"] / params["assumed_true_hr"])
    assert result.get("method_param") == params["method"]
    assert result.get("hypothesis_type_param") == params["hypothesis_type"]
    assert result.get("calculation_type_param") == params["calculation_type"]
    assert result.get("alpha_param") == params["alpha"]
    # Check that 'sides' is not a primary output for NI, as it's handled internally
    # but alpha should be the one passed for NI (which is one-sided by definition)

def test_calculate_parallel_survival_analytical_non_inferiority_sample_size():
    """Test analytical sample size calculation for survival outcome, non-inferiority."""
    params = {
        "calculation_type": "Sample Size",
        "outcome_type": "Survival",
        "method": "Analytical",
        "hypothesis_type": "Non-Inferiority",
        "median_survival1": 20,          # Control group median survival
        "non_inferiority_margin_hr": 1.25, # Non-inferiority margin for HR
        "assumed_true_hr": 0.95,         # Assumed true HR (experimental / control)
        "accrual_time": 18,              # Enrollment period in months
        "follow_up_time": 30,            # Follow-up period in months
        "dropout_rate": 0.1,             # Annual dropout rate
        "alpha": 0.025,                  # One-sided alpha for NI
        "power": 0.9,                    # Desired power
        "allocation_ratio": 1.0,
        "sides": 1 # For NI, this should effectively be 1
    }
    result = calculate_parallel_survival(params)

    assert "n1" in result
    assert result["n1"] > 0, f"n1 should be positive: {result.get('n1')}"
    assert "n2" in result
    assert result["n2"] > 0, f"n2 should be positive: {result.get('n2')}"
    assert "total_n" in result
    assert result["total_n"] == result["n1"] + result["n2"], "Total n mismatch"
    assert "events" in result
    assert result.get("events", -1) >= 0, f"Events should be non-negative: {result.get('events')}"
    assert "error" not in result, f"Unexpected error: {result.get('error')}"
    assert result.get("median_survival1_param") == params["median_survival1"]
    assert result.get("non_inferiority_margin_hr_param") == params["non_inferiority_margin_hr"]
    assert result.get("assumed_true_hr_param") == params["assumed_true_hr"]
    assert result.get("power_param") == params["power"]
    assert result.get("method_param") == params["method"]
    assert result.get("hypothesis_type_param") == params["hypothesis_type"]
    assert result.get("calculation_type_param") == params["calculation_type"]
    assert result.get("alpha_param") == params["alpha"]

def test_calculate_parallel_survival_simulation_non_inferiority_power():
    """Test simulation power calculation for survival outcome, non-inferiority."""
    params = {
        "calculation_type": "Power",
        "outcome_type": "Survival",
        "method": "Simulation",
        "hypothesis_type": "Non-Inferiority",
        "median_survival1": 15,
        "non_inferiority_margin_hr": 1.3,
        "assumed_true_hr": 1.0,
        "accrual_time": 12,
        "follow_up_time": 24,
        "dropout_rate": 0.05,
        "alpha": 0.025,
        "n1": 200, # Adjusted sample size for simulation to likely yield power
        "n2": 200,
        "allocation_ratio": 1.0,
        "sides": 1, # NI sim functions handle one-sided alpha internally
        "nsim": 100,  # Reduced nsim for faster test execution
        "seed": 42
    }
    result = calculate_parallel_survival(params)

    assert "power" in result
    assert 0 <= result["power"] <= 1, f"Power out of expected range: {result.get('power')}"
    assert "events" in result
    # For simulations, events can sometimes be low if n is small, but should be non-negative
    assert result.get("events", -1) >= 0, f"Events should be non-negative: {result.get('events')}"
    assert "error" not in result, f"Unexpected error: {result.get('error')}"
    assert result.get("median_survival1_param") == params["median_survival1"]
    assert result.get("non_inferiority_margin_hr_param") == params["non_inferiority_margin_hr"]
    assert result.get("assumed_true_hr_param") == params["assumed_true_hr"]
    assert result.get("median_survival2_derived") == pytest.approx(params["median_survival1"] / params["assumed_true_hr"])
    assert result.get("method_param") == params["method"]
    assert result.get("hypothesis_type_param") == params["hypothesis_type"]
    assert result.get("calculation_type_param") == params["calculation_type"]
    assert result.get("alpha_param") == params["alpha"]
    assert result.get("nsim") == params["nsim"]

def test_calculate_parallel_survival_simulation_non_inferiority_sample_size():
    """Test simulation sample size calculation for survival outcome, non-inferiority."""
    params = {
        "calculation_type": "Sample Size",
        "outcome_type": "Survival",
        "method": "Simulation",
        "hypothesis_type": "Non-Inferiority",
        "median_survival1": 20,
        "non_inferiority_margin_hr": 1.25,
        "assumed_true_hr": 0.95,
        "accrual_time": 18,
        "follow_up_time": 30,
        "dropout_rate": 0.1,
        "alpha": 0.025,
        "power": 0.8, # Desired power, slightly lower for faster sim test
        "allocation_ratio": 1.0,
        "sides": 1, # NI sim functions handle one-sided alpha internally
        "nsim": 100,  # Reduced nsim for faster test execution
        "seed": 42
    }
    result = calculate_parallel_survival(params)

    assert "n1" in result
    assert result["n1"] > 0, f"n1 should be positive: {result.get('n1')}"
    assert "n2" in result
    assert result["n2"] > 0, f"n2 should be positive: {result.get('n2')}"
    assert "total_n" in result
    assert result["total_n"] == result["n1"] + result["n2"], "Total n mismatch"
    assert "events" in result
    assert result.get("events", -1) >= 0, f"Events should be non-negative: {result.get('events')}"
    assert "error" not in result, f"Unexpected error: {result.get('error')}"
    assert result.get("median_survival1_param") == params["median_survival1"]
    assert result.get("non_inferiority_margin_hr_param") == params["non_inferiority_margin_hr"]
    assert result.get("assumed_true_hr_param") == params["assumed_true_hr"]
    assert result.get("power_param") == params["power"]
    assert result.get("method_param") == params["method"]
    assert result.get("hypothesis_type_param") == params["hypothesis_type"]
    assert result.get("calculation_type_param") == params["calculation_type"]
    assert result.get("alpha_param") == params["alpha"]
    assert result.get("nsim") == params["nsim"]
