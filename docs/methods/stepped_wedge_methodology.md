# Stepped Wedge Cluster Randomized Trials: Methodology Documentation

## Overview

Stepped wedge cluster randomized trials (SW-CRT) are a specialized type of cluster randomized trial in which all clusters receive the intervention, but the timing of implementation is randomized. Clusters begin in the control condition and switch to the intervention at randomly assigned time points, creating a stepped pattern of implementation.

This document provides comprehensive methodology for both analytical and simulation-based approaches to power analysis in stepped wedge designs, implemented in the DesignPower statistical software.

## Key Features of Stepped Wedge Designs

### Design Characteristics
- **Sequential implementation**: All clusters eventually receive the intervention
- **Unidirectional switching**: Once a cluster switches to intervention, it remains in intervention
- **Built-in control**: Each cluster serves as its own control during the baseline period
- **Staggered rollout**: Intervention implementation is staggered across time steps

### Advantages
- Ethical acceptability when intervention is believed to be beneficial
- Efficiency in resource utilization (gradual rollout)
- Ability to study implementation processes over time
- Built-in replication across clusters and time

### Disadvantages
- Complex correlation structure requiring specialized analysis
- Temporal trends may confound intervention effects
- Limited flexibility once design is initiated
- Larger sample sizes often required compared to parallel group designs

## Mathematical Framework

### Design Structure

A stepped wedge design with K clusters and T time periods can be represented by a design matrix **X** of dimensions K × T, where:

- X_{kt} = 0 if cluster k is in control condition at time t
- X_{kt} = 1 if cluster k is under intervention at time t

The total number of cluster-periods is K × T, with:
- Control periods: Σ Σ (1 - X_{kt})
- Intervention periods: Σ Σ X_{kt}

### Correlation Structure

Stepped wedge designs involve complex correlation structures with three types of correlations:

1. **Intracluster Correlation (ICC, ρ)**: Correlation between individuals within the same cluster at the same time period
2. **Cluster Autocorrelation (CAC, ρ_c)**: Correlation within clusters across different time periods
3. **Individual Autocorrelation**: Correlation for the same individual across time (when applicable)

## Analytical Methods: Hussey & Hughes (2007)

### Theoretical Foundation

The Hussey & Hughes method provides closed-form analytical formulas for power calculation in stepped wedge designs. This approach assumes a linear mixed-effects model framework with the following variance-covariance structure.

#### Model Specification

For continuous outcomes, the model can be written as:

Y_{ijt} = μ + θX_{jt} + u_j + v_{jt} + e_{ijt}

Where:
- Y_{ijt} = outcome for individual i in cluster j at time t
- μ = overall mean
- θ = intervention effect (parameter of interest)
- X_{jt} = intervention indicator (0/1)
- u_j = random cluster effect
- v_{jt} = random cluster-period effect
- e_{ijt} = individual-level random error

#### Variance Components

The total variance is decomposed as:

**Total Variance**: σ² = σ²_e + σ²_c + σ²_s

Where:
- **σ²_e**: Individual-level variance = σ² × (1 - ρ)
- **σ²_c**: Cluster-period variance = σ² × ρ × (1 - ρ_c)
- **σ²_s**: Cluster-level variance = σ² × ρ × ρ_c

#### Correlation Parameters

- **ICC (ρ)**: Intracluster correlation coefficient
  - ρ = (σ²_c + σ²_s) / σ²
  - Represents correlation between individuals in same cluster-period

- **CAC (ρ_c)**: Cluster autocorrelation coefficient
  - ρ_c = σ²_s / (σ²_c + σ²_s)
  - Represents correlation between cluster-periods within the same cluster

### Variance-Covariance Matrix Calculations

#### For Cluster-Period Means

The variance of cluster-period means is:

Var(Ȳ_{jt}) = σ²_e/m + σ²_c + σ²_s

Where m is the number of individuals per cluster per time period.

#### Correlation Between Cluster-Period Means

**Same cluster, different periods**:
Corr(Ȳ_{jt}, Ȳ_{jt'}) = σ²_s / Var(Ȳ_{jt}) = ρ_between_periods

**Different clusters**:
Corr(Ȳ_{jt}, Ȳ_{j't'}) = 0

### Treatment Effect Variance

The variance of the treatment effect estimator under the Hussey & Hughes framework is:

Var(θ̂) = Var(Ȳ_{jt}) × [(1/n_c) + (1/n_i)] × ψ

Where:
- n_c = number of control cluster-periods
- n_i = number of intervention cluster-periods
- ψ = correlation adjustment factor = 1 + (T-1) × ρ_between_periods

### Power Calculation

Statistical power is calculated as:

Power = Φ(|θ|/SE(θ̂) - z_{α/2})

Where:
- Φ = standard normal cumulative distribution function
- θ = true treatment effect
- SE(θ̂) = √Var(θ̂)
- z_{α/2} = critical value for two-sided test at significance level α

### Binary Outcomes: Arcsine Transformation

For binary outcomes, the Hussey & Hughes method employs an arcsine transformation:

1. **Transform proportions**: θ_c = arcsin(√p_c), θ_i = arcsin(√p_i)
2. **Treatment effect**: δ = θ_i - θ_c
3. **Variance**: Var(arcsin(√p)) ≈ 1/(4m) where m is cluster size
4. **Apply continuous outcome formulas** with transformed parameters

This transformation:
- Stabilizes variance across different baseline rates
- Improves normal approximation
- Accounts for bounded nature of proportions (0,1)

## Simulation-Based Approaches

### Continuous Outcomes Simulation

The simulation approach generates synthetic datasets following the stepped wedge design structure:

#### Data Generation Process

1. **Generate cluster effects**: u_j ~ N(0, σ²_between)
2. **Assign intervention timing**: Random assignment of clusters to steps
3. **Generate individual outcomes**: 
   - Control periods: Y_{ijt} ~ N(μ + u_j, σ²_within)
   - Intervention periods: Y_{ijt} ~ N(μ + θ + u_j, σ²_within)

#### Variance Decomposition

- σ²_between = ρ × σ²_total (between-cluster variance)
- σ²_within = (1 - ρ) × σ²_total (within-cluster variance)

#### Statistical Analysis

For each simulated dataset:
1. **Simple comparison**: t-test between intervention and control observations
2. **Count significant results**: Proportion of p-values < α
3. **Estimate power**: Significant results / Total simulations

### Binary Outcomes Simulation

#### Beta-Binomial Model

For binary outcomes with clustering, we use a beta-binomial model:

1. **Cluster-specific probabilities**:
   - Control: p_{jc} ~ Beta(α_c, β_c)
   - Intervention: p_{ji} ~ Beta(α_i, β_i)

2. **Parameter calculation**:
   - κ = (1 - ρ) / ρ (concentration parameter)
   - α_c = p_c × κ, β_c = (1 - p_c) × κ
   - α_i = p_i × κ, β_i = (1 - p_i) × κ

3. **Individual outcomes**: Y_{ijt} ~ Bernoulli(p_{jt})

#### Statistical Testing

1. **Calculate proportions** in control and intervention periods
2. **Account for design effect**: DE = 1 + (m-1) × ρ
3. **Standard error**: SE = √[p̄(1-p̄) × (1/n_c + 1/n_i) × DE]
4. **Test statistic**: z = (p_i - p_c) / SE

## Design Effect Considerations

### Traditional Design Effect

For clustered designs, the design effect is:

DE = 1 + (m - 1) × ρ

Where:
- m = average cluster size
- ρ = intracluster correlation coefficient

### Stepped Wedge Design Effect

Stepped wedge designs have additional complexity due to:

1. **Temporal correlation**: CAC introduces additional correlation
2. **Varying exposure**: Clusters have different lengths of exposure
3. **Secular trends**: Time effects may confound intervention effects

The effective design effect accounts for:
- Standard clustering (ICC)
- Temporal clustering (CAC)
- Design efficiency (proportion of intervention periods)

## Method Selection Guidelines

### When to Use Analytical Methods (Hussey & Hughes)

**Advantages**:
- Fast computation
- Exact results (no simulation variability)
- Handles complex correlation structures (ICC and CAC)
- Suitable for sample size calculations

**Recommended when**:
- Standard stepped wedge design assumptions are met
- Complex correlation structure (CAC > 0) needs to be modeled
- Large-scale sample size calculations required
- Computational efficiency is important

**Limitations**:
- Assumes linear mixed-effects model framework
- Normal distribution assumptions
- May not handle highly complex designs or non-standard analysis approaches

### When to Use Simulation Methods

**Advantages**:
- Flexible for non-standard designs
- Can incorporate complex analysis methods
- No distributional assumptions
- Can model realistic data generation processes

**Recommended when**:
- Non-standard stepped wedge designs
- Complex outcome distributions
- Non-standard analysis approaches
- Validation of analytical methods
- Small sample situations where normal approximation may fail

**Limitations**:
- Computationally intensive
- Simulation variability in results
- Current implementation uses simplified correlation structure
- Requires careful validation of simulation model

### Practical Decision Framework

1. **Standard Design + Standard Analysis** → Hussey & Hughes Analytical
2. **Complex Correlation Structure (CAC > 0)** → Hussey & Hughes Analytical
3. **Non-standard Design/Analysis** → Simulation
4. **Small Clusters/Samples** → Consider both methods for validation
5. **Binary Outcomes with Extreme Rates** → Consider simulation validation

## Implementation Details

### Software Implementation

The DesignPower implementation provides:

1. **Analytical Methods**:
   - `hussey_hughes_power_continuous()`
   - `hussey_hughes_power_binary()`
   - `hussey_hughes_sample_size_continuous()`
   - `hussey_hughes_sample_size_binary()`

2. **Simulation Methods**:
   - `simulate_continuous()`
   - `simulate_binary()`

### Parameter Specifications

#### Required Parameters
- **clusters**: Number of clusters (K)
- **steps**: Number of time steps including baseline (T)
- **individuals_per_cluster**: Cluster size (m)
- **icc**: Intracluster correlation coefficient (ρ)
- **alpha**: Significance level (α)

#### Outcome-Specific Parameters

**Continuous Outcomes**:
- **treatment_effect**: Mean difference (θ)
- **std_dev**: Standard deviation (σ)
- **cluster_autocorr**: Cluster autocorrelation (ρ_c) [analytical only]

**Binary Outcomes**:
- **p_control**: Control proportion
- **p_intervention**: Intervention proportion
- **cluster_autocorr**: Cluster autocorrelation (ρ_c) [analytical only]

#### Simulation Parameters
- **nsim**: Number of simulations (default: 1,000)

### Input Validation

The implementation includes validation for:
- Minimum number of clusters ≥ intervention steps
- Valid correlation coefficients (0 ≤ ρ, ρ_c ≤ 1)
- Valid proportions for binary outcomes (0 < p < 1)
- Positive effect sizes and standard deviations

## Assumptions and Limitations

### Model Assumptions

1. **Linear mixed-effects model**: Appropriate for the outcome type
2. **Normal distribution**: For continuous outcomes and transformed binary outcomes
3. **Constant variance**: Homoscedasticity across clusters and time
4. **Independence**: Between-cluster independence
5. **Missing data**: Complete data assumed (MCAR if missing)

### Design Assumptions

1. **Unidirectional switching**: No contamination or switch-back
2. **Stable intervention**: Intervention effect remains constant over time
3. **No secular trends**: Or secular trends are accounted for in analysis
4. **Balanced design**: Equal cluster sizes and measurement intervals

### Limitations

1. **Secular trends**: Method assumes no confounding time trends
2. **Implementation fidelity**: Assumes consistent intervention implementation
3. **Carryover effects**: Not explicitly modeled
4. **Complex interactions**: Time-varying treatment effects not modeled

## Validation and Quality Assurance

### Analytical Validation

The Hussey & Hughes implementation has been validated against:
- Published examples from Hussey & Hughes (2007)
- Simulation studies in the literature
- Cross-validation with R packages (e.g., `steppedwedge`, `swCRTdesign`)

### Simulation Validation

Simulation methods validated through:
- Convergence studies (varying nsim)
- Comparison with analytical methods for standard designs
- Type I error rate validation (power = α when effect = 0)
- Effect size sensitivity analysis

### Known Limitations

1. **Simplified simulation correlation structure**: Current simulation implementation uses basic clustering model
2. **No period effects**: Explicit period effects not modeled in current simulation
3. **Balanced designs**: Optimization for balanced designs primarily

## References

### Primary References

1. **Hussey MA, Hughes JP** (2007). Design and analysis of stepped wedge cluster randomized trials. *Contemporary Clinical Trials*, 28(2), 182-191.

2. **Hemming K, Haines TP, Chilton PJ, Girling AJ, Lilford RJ** (2015). The stepped wedge cluster randomised trial: rationale, design, analysis, and reporting. *BMJ*, 350, h391.

3. **Copas AJ, Lewis JJ, Thompson JA, Davey C, Baio G, Hargreaves JR** (2015). Designing a stepped wedge trial: three main designs, carry-over effects and randomisation approaches. *Trials*, 16(1), 352.

### Methodological References

4. **Hooper R, Teerenstra S, de Hoop E, Eldridge S** (2016). Sample size calculation for stepped wedge and other longitudinal cluster randomised trials. *Statistics in Medicine*, 35(26), 4718-4728.

5. **Li F, Hughes JP, Hemming K, Taljaard M, Melnick ER, Heagerty PJ** (2021). Mixed-effects models for the design and analysis of stepped wedge cluster randomized trials: An overview. *Statistical Methods in Medical Research*, 30(2), 612-639.

6. **Thompson JA, Davey C, Fielding K, Hargreaves JR, Hayes RJ** (2018). Robust analysis of stepped wedge trials using cluster-level summaries within periods. *Statistics in Medicine*, 37(16), 2487-2493.

### Software References

7. **Hemming K, Kasza J, Hooper R, Forbes A, Taljaard M** (2020). A tutorial on sample size calculation for multiple-period cluster randomized parallel, cross-over and stepped-wedge trials using the Shiny CRT Calculator. *International Journal of Epidemiology*, 49(3), 979-995.

8. **Kasza J, Hemming K, Hooper R, Matthews JN, Forbes AB** (2019). Impact of non-uniform correlation structure on sample size and power in multiple-period cluster randomised trials. *Statistical Methods in Medical Research*, 28(3), 703-716.

## Appendix: Mathematical Derivations

### A.1 Variance of Treatment Effect Estimator

Under the Hussey & Hughes framework, the treatment effect is estimated as the difference between intervention and control period means:

θ̂ = Ȳ_I - Ȳ_C

Where:
- Ȳ_I = mean of all intervention cluster-periods
- Ȳ_C = mean of all control cluster-periods

The variance is:
Var(θ̂) = Var(Ȳ_I) + Var(Ȳ_C) - 2Cov(Ȳ_I, Ȳ_C)

For stepped wedge designs with proper randomization, Cov(Ȳ_I, Ȳ_C) accounts for the correlation between intervention and control periods within the same clusters.

### A.2 Design Matrix Properties

For K clusters and T time periods, the design matrix X has special properties:

1. **Row sums**: Each row sums to the number of intervention periods for that cluster
2. **Column sums**: First column (baseline) always sums to 0, subsequent columns increase
3. **Total exposure**: ΣΣX_{kt} represents total intervention cluster-periods

These properties ensure the stepped wedge structure and affect the precision of treatment effect estimation.

### A.3 Arcsine Transformation Properties

For proportion p, the arcsine transformation θ = arcsin(√p) has:

1. **Domain**: [0,π/2] for p ∈ [0,1]
2. **Variance stabilization**: Var(θ) ≈ 1/(4n) regardless of p
3. **Normal approximation**: Better approximation for extreme proportions
4. **Inverse transformation**: p = sin²(θ)

This transformation is particularly useful for binary outcomes in stepped wedge designs where control and intervention proportions may differ substantially.