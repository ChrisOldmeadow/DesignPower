# DesignPower Usage Examples

This document provides practical examples of how to use the DesignPower library for various study designs and outcome types, including both analytical and simulation-based methods.

## Table of Contents

- [Parallel Group Designs](#parallel-group-designs)
  - [Binary Outcomes](#binary-outcomes)
  - [Continuous Outcomes](#continuous-outcomes)
  - [Survival Outcomes](#survival-outcomes)
- [Non-inferiority Designs](#non-inferiority-designs)
- [Advanced Simulation Usage](#advanced-simulation-usage)

## Parallel Group Designs

### Binary Outcomes

#### Analytical Methods

```python
from core.designs.parallel import (
    sample_size_binary, 
    power_binary, 
    min_detectable_effect_binary
)

# Sample size calculation for binary outcomes
result = sample_size_binary(
    p1=0.3,      # Proportion in control group
    p2=0.45,     # Proportion in treatment group
    power=0.8,   # Desired power
    alpha=0.05,  # Significance level
    allocation_ratio=1.0  # Equal allocation
)
print(f"Required sample size: {result['total_sample_size']} participants")
print(f"Group 1: {result['sample_size_1']}, Group 2: {result['sample_size_2']}")

# Power calculation
power_result = power_binary(
    n1=150,      # Sample size in group 1
    n2=150,      # Sample size in group 2
    p1=0.3,      # Proportion in control group
    p2=0.45,     # Proportion in treatment group
    alpha=0.05,  # Significance level
    test_type="normal_approximation"  # Statistical test to use
)
print(f"Statistical power: {power_result['power']:.2f}")

# Minimum detectable effect
mde_result = min_detectable_effect_binary(
    n1=150,      # Sample size in group 1
    n2=150,      # Sample size in group 2
    p1=0.3,      # Proportion in control group
    power=0.8,   # Desired power
    alpha=0.05   # Significance level
)
print(f"Minimum detectable proportion in group 2: {mde_result['minimum_detectable_p2']:.3f}")
print(f"Absolute difference: {mde_result['minimum_detectable_difference']:.3f}")
```

#### Simulation Methods

```python
from core.designs.parallel import (
    sample_size_binary_sim,
    power_binary_sim,
    min_detectable_effect_binary_sim
)

# Sample size calculation using simulation
sim_result = sample_size_binary_sim(
    p1=0.3,               # Proportion in control group
    p2=0.45,              # Proportion in treatment group
    power=0.8,            # Desired power
    alpha=0.05,           # Significance level
    allocation_ratio=1.0, # Equal allocation
    nsim=1000,            # Number of simulations
    min_n=10,             # Minimum sample size to try
    max_n=500,            # Maximum sample size to try
    step=10,              # Step size for incrementing sample size
    test_type="Likelihood Ratio Test"  # Statistical test to use
)
print(f"Required sample size (simulation): {sim_result['total_sample_size']} participants")
print(f"Achieved power: {sim_result['achieved_power']:.2f}")

# Power calculation using simulation
power_sim = power_binary_sim(
    n1=150,               # Sample size in group 1
    n2=150,               # Sample size in group 2
    p1=0.3,               # Proportion in control group
    p2=0.45,              # Proportion in treatment group
    alpha=0.05,           # Significance level
    nsim=1000,            # Number of simulations
    test_type="Exact Test"  # Statistical test to use
)
print(f"Statistical power (simulation): {power_sim['power']:.2f}")

# Minimum detectable effect using simulation
mde_sim = min_detectable_effect_binary_sim(
    n1=150,               # Sample size in group 1
    n2=150,               # Sample size in group 2
    p1=0.3,               # Proportion in control group
    power=0.8,            # Desired power
    alpha=0.05,           # Significance level
    nsim=1000,            # Number of simulations
    precision=0.01        # Desired precision for the effect size
)
print(f"Minimum detectable proportion (simulation): {mde_sim['p2']:.3f}")
print(f"Absolute difference: {mde_sim['absolute_mde']:.3f}")
```

### Continuous Outcomes

#### Analytical Methods

```python
from core.designs.parallel import (
    sample_size_continuous,
    power_continuous,
    min_detectable_effect_continuous
)

# Sample size calculation for continuous outcomes
result = sample_size_continuous(
    mean1=10.0,   # Mean in group 1
    mean2=12.0,   # Mean in group 2
    std_dev=5.0,  # Standard deviation
    power=0.8,    # Desired power
    alpha=0.05,   # Significance level
    allocation_ratio=1.0  # Equal allocation
)
print(f"Required sample size: {result['total_sample_size']} participants")
print(f"Group 1: {result['sample_size_1']}, Group 2: {result['sample_size_2']}")
print(f"Standardized effect size: {result['effect_size']:.2f}")

# Power calculation
power_result = power_continuous(
    n1=100,      # Sample size in group 1
    n2=100,      # Sample size in group 2
    mean1=10.0,  # Mean in group 1
    mean2=12.0,  # Mean in group 2
    std_dev=5.0, # Standard deviation
    alpha=0.05   # Significance level
)
print(f"Statistical power: {power_result['power']:.2f}")

# Minimum detectable effect
mde_result = min_detectable_effect_continuous(
    n1=100,      # Sample size in group 1
    n2=100,      # Sample size in group 2
    std_dev=5.0, # Standard deviation
    power=0.8,   # Desired power
    alpha=0.05   # Significance level
)
print(f"Minimum detectable effect: {mde_result['minimum_detectable_effect']:.2f}")
print(f"Standardized effect: {mde_result['standardized_effect']:.2f}")
```

#### Simulation Methods

```python
from core.designs.parallel import (
    sample_size_continuous_sim,
    power_continuous_sim,
    min_detectable_effect_continuous_sim
)

# Calculate effect size (delta) for sample size calculation
delta = 2.0  # Mean2 - Mean1 = 12 - 10

# Sample size calculation using simulation
sim_result = sample_size_continuous_sim(
    delta=delta,          # Difference in means
    std_dev=5.0,          # Standard deviation
    power=0.8,            # Desired power
    alpha=0.05,           # Significance level
    allocation_ratio=1.0, # Equal allocation
    nsim=1000,            # Number of simulations
    min_n=10,             # Minimum sample size to try
    max_n=500,            # Maximum sample size to try
    step=10               # Step size for incrementing sample size
)
print(f"Required sample size (simulation): {sim_result['total_sample_size']} participants")
print(f"Achieved power: {sim_result['achieved_power']:.2f}")

# Power calculation using simulation
power_sim = power_continuous_sim(
    n1=100,               # Sample size in group 1
    n2=100,               # Sample size in group 2
    mean1=10.0,           # Mean in group 1
    mean2=12.0,           # Mean in group 2
    sd1=5.0,              # Standard deviation in group 1
    alpha=0.05,           # Significance level
    nsim=1000             # Number of simulations
)
print(f"Statistical power (simulation): {power_sim['power']:.2f}")

# Minimum detectable effect using simulation
mde_sim = min_detectable_effect_continuous_sim(
    n1=100,               # Sample size in group 1
    n2=100,               # Sample size in group 2
    std_dev=5.0,          # Standard deviation
    power=0.8,            # Desired power
    alpha=0.05,           # Significance level
    nsim=1000,            # Number of simulations
    precision=0.01        # Desired precision for the effect size
)
print(f"Minimum detectable effect (simulation): {mde_sim['minimum_detectable_effect']:.2f}")
print(f"Standardized effect: {mde_sim['standardized_effect']:.2f}")
```

### Survival Outcomes

#### Analytical Methods

```python
from core.designs.parallel import (
    sample_size_survival,
    power_survival,
    min_detectable_effect_survival
)

# Sample size calculation for survival outcomes
result = sample_size_survival(
    median1=12.0,           # Median survival time in group 1 (months)
    median2=18.0,           # Median survival time in group 2 (months)
    enrollment_period=12.0, # Duration of enrollment period (months)
    follow_up_period=12.0,  # Duration of follow-up period (months)
    dropout_rate=0.1,       # Expected dropout rate
    power=0.8,              # Desired power
    alpha=0.05,             # Significance level
    allocation_ratio=1.0    # Equal allocation
)
print(f"Required sample size: {result['total_sample_size']} participants")
print(f"Group 1: {result['sample_size_1']}, Group 2: {result['sample_size_2']}")
print(f"Expected events: {result['total_events']:.0f}")
print(f"Hazard ratio: {result['hazard_ratio']:.2f}")

# Power calculation
power_result = power_survival(
    n1=150,                 # Sample size in group 1
    n2=150,                 # Sample size in group 2
    median1=12.0,           # Median survival time in group 1 (months)
    median2=18.0,           # Median survival time in group 2 (months)
    enrollment_period=12.0, # Duration of enrollment period (months)
    follow_up_period=12.0,  # Duration of follow-up period (months)
    dropout_rate=0.1,       # Expected dropout rate
    alpha=0.05              # Significance level
)
print(f"Statistical power: {power_result['power']:.2f}")
print(f"Expected events: {power_result['total_events']:.0f}")

# Minimum detectable effect
mde_result = min_detectable_effect_survival(
    n1=150,                 # Sample size in group 1
    n2=150,                 # Sample size in group 2
    median1=12.0,           # Median survival time in group 1 (months)
    enrollment_period=12.0, # Duration of enrollment period (months)
    follow_up_period=12.0,  # Duration of follow-up period (months)
    dropout_rate=0.1,       # Expected dropout rate
    power=0.8,              # Desired power
    alpha=0.05              # Significance level
)
print(f"Minimum detectable median survival in group 2: {mde_result['minimum_detectable_median']:.1f} months")
print(f"Minimum detectable hazard ratio: {mde_result['minimum_detectable_hazard_ratio']:.2f}")
```

#### Simulation Methods

```python
from core.designs.parallel import (
    sample_size_survival_sim,
    power_survival_sim,
    min_detectable_effect_survival_sim
)

# Sample size calculation using simulation
sim_result = sample_size_survival_sim(
    median1=12.0,           # Median survival time in group 1 (months)
    median2=18.0,           # Median survival time in group 2 (months)
    enrollment_period=12.0, # Duration of enrollment period (months)
    follow_up_period=12.0,  # Duration of follow-up period (months)
    dropout_rate=0.1,       # Expected dropout rate
    power=0.8,              # Desired power
    alpha=0.05,             # Significance level
    allocation_ratio=1.0,   # Equal allocation
    nsim=500,               # Number of simulations
    min_n=50,               # Minimum sample size to try
    max_n=500,              # Maximum sample size to try
    step=10                 # Step size for incrementing sample size
)
print(f"Required sample size (simulation): {sim_result['total_sample_size']} participants")
print(f"Achieved power: {sim_result['achieved_power']:.2f}")
print(f"Expected events: {sim_result['total_events']:.0f}")

# Power calculation using simulation
power_sim = power_survival_sim(
    n1=150,                 # Sample size in group 1
    n2=150,                 # Sample size in group 2
    median1=12.0,           # Median survival time in group 1 (months)
    median2=18.0,           # Median survival time in group 2 (months)
    enrollment_period=12.0, # Duration of enrollment period (months)
    follow_up_period=12.0,  # Duration of follow-up period (months)
    dropout_rate=0.1,       # Expected dropout rate
    alpha=0.05,             # Significance level
    nsim=500                # Number of simulations
)
print(f"Statistical power (simulation): {power_sim['power']:.2f}")
print(f"Mean number of events: {power_sim['mean_events']:.0f}")

# Minimum detectable effect using simulation
mde_sim = min_detectable_effect_survival_sim(
    n1=150,                 # Sample size in group 1
    n2=150,                 # Sample size in group 2
    median1=12.0,           # Median survival time in group 1 (months)
    enrollment_period=12.0, # Duration of enrollment period (months)
    follow_up_period=12.0,  # Duration of follow-up period (months)
    dropout_rate=0.1,       # Expected dropout rate
    power=0.8,              # Desired power
    alpha=0.05,             # Significance level
    nsim=500,               # Number of simulations
    precision=0.1           # Desired precision for the effect size
)
print(f"Minimum detectable median survival (simulation): {mde_sim['minimum_detectable_median']:.1f} months")
print(f"Minimum detectable hazard ratio: {mde_sim['minimum_detectable_hazard_ratio']:.2f}")
```

## Non-inferiority Designs

DesignPower supports non-inferiority trial designs for all outcome types. Here's an example for binary outcomes:

```python
from core.designs.parallel import (
    sample_size_binary_non_inferiority,
    power_binary_non_inferiority,
    sample_size_binary_non_inferiority_sim,
    min_detectable_binary_non_inferiority_margin_sim
)

# Analytical sample size calculation for non-inferiority
result = sample_size_binary_non_inferiority(
    p1=0.7,                    # Proportion in control/standard group
    non_inferiority_margin=0.1, # Non-inferiority margin
    power=0.8,                 # Desired power
    alpha=0.05,                # Significance level (one-sided for non-inferiority)
    allocation_ratio=1.0,      # Equal allocation
    assumed_difference=0.0,    # Assumed true difference (0 = treatments truly equivalent)
    direction="lower"          # Direction of non-inferiority ("lower" or "upper")
)
print(f"Required sample size for non-inferiority: {result['total_sample_size']} participants")
print(f"Group 1: {result['sample_size_1']}, Group 2: {result['sample_size_2']}")

# Simulation-based sample size calculation for non-inferiority
sim_result = sample_size_binary_non_inferiority_sim(
    p1=0.7,                    # Proportion in control/standard group
    non_inferiority_margin=0.1, # Non-inferiority margin
    power=0.8,                 # Desired power
    alpha=0.05,                # Significance level (one-sided for non-inferiority)
    allocation_ratio=1.0,      # Equal allocation
    nsim=1000,                 # Number of simulations
    assumed_difference=0.0,    # Assumed true difference (0 = treatments truly equivalent)
    direction="lower"          # Direction of non-inferiority ("lower" or "upper")
)
print(f"Required sample size for non-inferiority (simulation): {sim_result['total_sample_size']} participants")
print(f"Achieved power: {sim_result['achieved_power']:.2f}")

# Minimum detectable non-inferiority margin
mde_result = min_detectable_binary_non_inferiority_margin_sim(
    n1=200,                   # Sample size in group 1
    n2=200,                   # Sample size in group 2
    p1=0.7,                   # Proportion in control/standard group
    power=0.8,                # Desired power
    alpha=0.05,               # Significance level (one-sided for non-inferiority)
    assumed_difference=0.0,   # Assumed true difference (0 = treatments truly equivalent)
    direction="lower"         # Direction of non-inferiority ("lower" or "upper")
)
print(f"Minimum detectable non-inferiority margin: {mde_result['minimum_detectable_margin']:.3f}")
```

## Advanced Simulation Usage

Here are examples of advanced simulation configurations:

### Custom Test Methods for Binary Outcomes

```python
from core.designs.parallel import power_binary_sim

# Using Fisher's Exact Test for small sample sizes
result_exact = power_binary_sim(
    n1=30,                  # Small sample size in group 1
    n2=30,                  # Small sample size in group 2
    p1=0.3,                 # Proportion in control group
    p2=0.5,                 # Proportion in treatment group
    alpha=0.05,             # Significance level
    nsim=1000,              # Number of simulations
    test_type="fishers_exact"  # Using Fisher's Exact Test
)
print(f"Power with Fisher's Exact Test: {result_exact['power']:.2f}")

# Using Likelihood Ratio Test
result_lr = power_binary_sim(
    n1=30,
    n2=30,
    p1=0.3,
    p2=0.5,
    alpha=0.05,
    nsim=1000,
    test_type="likelihood_ratio"  # Using Likelihood Ratio Test
)
print(f"Power with Likelihood Ratio Test: {result_lr['power']:.2f}")
```

### Repeated Measures for Continuous Outcomes

```python
from core.designs.parallel import sample_size_continuous_sim

# Sample size for continuous outcomes with repeated measures
result = sample_size_continuous_sim(
    delta=2.0,              # Difference in means
    std_dev=5.0,            # Standard deviation
    power=0.8,              # Desired power
    alpha=0.05,             # Significance level
    allocation_ratio=1.0,   # Equal allocation
    nsim=500,               # Number of simulations
    repeated_measures=True, # Use repeated measures design
    correlation=0.7,        # Correlation between baseline and follow-up
    method="ancova"         # Analysis method (ANCOVA or change_score)
)
print(f"Required sample size with repeated measures: {result['total_sample_size']} participants")
print(f"Achieved power: {result['achieved_power']:.2f}")
```

### Unequal Variances in Continuous Outcomes

```python
from core.designs.parallel import power_continuous_sim

# Power calculation with unequal variances
result = power_continuous_sim(
    n1=100,                # Sample size in group 1
    n2=100,                # Sample size in group 2
    mean1=10.0,            # Mean in group 1
    mean2=12.0,            # Mean in group 2
    sd1=5.0,               # Standard deviation in group 1
    sd2=7.0,               # Standard deviation in group 2 (different from group 1)
    alpha=0.05,            # Significance level
    nsim=1000              # Number of simulations
)
print(f"Power with unequal variances: {result['power']:.2f}")
```

### Complex Survival Analysis Configuration

```python
from core.designs.parallel import simulate_survival_trial

# Direct simulation of a survival trial with specific parameters
result = simulate_survival_trial(
    n1=150,                   # Sample size in group 1
    n2=150,                   # Sample size in group 2
    median1=12.0,             # Median survival time in group 1 (months)
    median2=18.0,             # Median survival time in group 2 (months)
    enrollment_period=12.0,   # Duration of enrollment period (months)
    follow_up_period=12.0,    # Duration of follow-up period (months)
    dropout_rate=0.15,        # Higher dropout rate
    nsim=1000,                # More simulations for precision
    alpha=0.05,               # Significance level
    seed=42,                  # Random seed for reproducibility
    sides=1                   # One-sided test
)
print(f"Power from direct simulation: {result['empirical_power']:.2f}")
print(f"Mean events: {result['mean_events']:.0f}")
print(f"Mean hazard ratio estimate: {result['mean_log_hr']:.2f}")
```
