# Interrupted Time Series Analysis: Methodology Documentation

## Overview

Interrupted Time Series (ITS) analysis is a quasi-experimental design used to evaluate the impact of interventions or policy changes when randomized controlled trials are not feasible. The design involves collecting data at multiple time points before and after an intervention, allowing researchers to assess whether the intervention caused changes in the level and/or trend of the outcome over time.

This document provides comprehensive methodology for both analytical and simulation-based approaches to power analysis in interrupted time series designs, implemented in the DesignPower statistical software.

## Key Features of ITS Designs

### Design Characteristics
- **Longitudinal structure**: Multiple observations before and after intervention
- **Quasi-experimental**: No randomization, but strong internal validity through design
- **Temporal control**: Pre-intervention period serves as control for post-intervention period
- **Trend analysis**: Can detect both immediate (level) and gradual (slope) changes

### Advantages
- Feasible when randomization is not possible or ethical
- Strong causal inference when assumptions are met
- Can evaluate both immediate and long-term effects
- Controls for stable confounders through temporal comparison
- Suitable for population-level interventions

### Disadvantages
- Vulnerable to confounding from time-varying factors
- Requires sufficient pre-intervention time points
- Susceptible to seasonal and secular trends
- Autocorrelation complicates statistical inference
- Cannot control for unmeasured time-varying confounders

## Mathematical Framework

### Segmented Regression Model

The fundamental model for ITS analysis is segmented regression, which models the outcome as a function of time with different parameters before and after the intervention:

**Basic Model**:
Y_t = β₀ + β₁T_t + β₂X_t + β₃TX_t + ε_t

Where:
- Y_t = outcome at time t
- T_t = time since start of study (continuous, centered at intervention)
- X_t = intervention indicator (0 = pre-intervention, 1 = post-intervention)
- TX_t = interaction term (time since intervention, 0 for pre-intervention periods)
- ε_t = error term with potential autocorrelation

### Parameter Interpretation

- **β₀**: Baseline level at the intervention point
- **β₁**: Pre-intervention trend (slope)
- **β₂**: Immediate level change at intervention point
- **β₃**: Change in trend (slope change) after intervention

### Extended Model with Seasonality

For data with seasonal patterns:

Y_t = β₀ + β₁T_t + β₂X_t + β₃TX_t + Σ(γₛS_{st}) + ε_t

Where:
- S_{st} = seasonal indicator variables (e.g., monthly dummies)
- γₛ = seasonal effects

### Autocorrelation Structure

The error term ε_t often exhibits autocorrelation:

**First-order autoregressive AR(1)**:
ε_t = ρε_{t-1} + u_t

Where:
- ρ = autocorrelation coefficient (−1 < ρ < 1)
- u_t ~ N(0, σ²) independent white noise

**Higher-order models** (AR(p), MA(q), ARIMA) may be needed for complex temporal patterns.

## Statistical Methods

### Power Calculation Framework

Power calculations for ITS designs focus on detecting the intervention effect parameters (β₂ for level change, β₃ for trend change) using the segmented regression model.

#### Level Change Detection (β₂)

For detecting an immediate level change, power depends on:

**Test statistic**: t = β̂₂ / SE(β̂₂)

**Standard error**: SE(β̂₂) = σ√(1/n₀ + 1/n₁ + (T̄₀ - T̄₁)²/SST)

Where:
- n₀, n₁ = number of pre- and post-intervention observations
- T̄₀, T̄₁ = mean time in pre- and post-intervention periods
- SST = total sum of squares of time variable
- σ = residual standard deviation (adjusted for autocorrelation)

#### Trend Change Detection (β₃)

For detecting a change in trend:

**Test statistic**: t = β̂₃ / SE(β̂₃)

**Standard error**: SE(β̂₃) = σ√(1/SST₁)

Where SST₁ = sum of squares of post-intervention time variable.

#### Autocorrelation Adjustment

When autocorrelation is present (ρ ≠ 0), the effective sample size is reduced:

**Design Effect**: DE = (1 + ρ)/(1 - ρ) for strong autocorrelation

**Adjusted variance**: σ²_adj = σ² × (1 + 2ρΣ(n-k)ρᵏ⁻¹) for AR(1) errors

### Effect Size Specifications

#### Standardized Effect Sizes

**Level change effect size**: d₂ = β₂ / σ
**Trend change effect size**: d₃ = β₃ / (σ/√12) for yearly trend change

#### Clinical Significance Thresholds

Effect sizes should be based on:
- Clinical or policy relevance
- Expected magnitude of intervention effect
- Baseline variability in the outcome
- Measurement precision

### Sample Size Determination

#### Minimum Time Points

**Rule of thumb**: 
- Minimum 8 observations (4 pre, 4 post)
- Recommended 20+ observations for stable estimates
- At least 3 years of data when seasonal patterns expected

#### Power-Based Sample Size

For detecting level change β₂ with power 1-β at significance α:

n = 2(z_{α/2} + z_β)² × σ²_adj / β₂²

Where:
- z_{α/2} = critical value for two-sided test
- z_β = critical value for desired power
- σ²_adj = autocorrelation-adjusted variance

#### Optimal Allocation

For fixed total sample size n:
- **Equal allocation**: n₀ = n₁ = n/2 is often optimal for level change
- **Trend detection**: Longer post-intervention period may improve power for trend changes

## Implementation Details

### Continuous Outcomes

#### Model Fitting
1. **Ordinary Least Squares (OLS)**: Valid when ρ = 0
2. **Generalized Least Squares (GLS)**: For known autocorrelation structure
3. **Feasible GLS**: Estimate ρ, then apply GLS
4. **Maximum Likelihood**: For complex ARIMA error structures

#### Assumption Checking
- **Linearity**: Examine residual plots, consider non-linear trends
- **Normality**: Q-Q plots, Shapiro-Wilk test
- **Homoscedasticity**: Plot residuals vs. fitted values
- **Independence**: Durbin-Watson test, ACF/PACF of residuals

### Binary Outcomes

#### Logistic Segmented Regression

For binary outcomes Y_t ~ Bernoulli(p_t):

logit(p_t) = β₀ + β₁T_t + β₂X_t + β₃TX_t + ε_t

#### Overdispersion and Autocorrelation

**Quasi-likelihood methods**: Account for overdispersion
**GEE approach**: Model correlation structure explicitly
**Mixed-effects models**: Random effects for unmeasured heterogeneity

#### Power Approximation

Using normal approximation to logistic regression:

SE(β̂₂) ≈ 1/√(np̄(1-p̄))

Where p̄ is the average probability across all time points.

### Autocorrelation Modeling

#### AR(1) Model Estimation

**Durbin-Watson statistic**: DW = Σ(ε_t - ε_{t-1})²/Σε_t²
**Approximate relationship**: ρ ≈ 1 - DW/2

**Cochrane-Orcutt procedure**:
1. Estimate OLS model, obtain residuals
2. Estimate ρ from residuals: ρ̂ = Σε_tε_{t-1}/Σε²_{t-1}
3. Transform data: Y*_t = Y_t - ρ̂Y_{t-1}
4. Re-estimate with transformed data
5. Iterate until convergence

#### Model Selection

**Information criteria**: AIC, BIC for ARIMA model selection
**Residual diagnostics**: Ljung-Box test for remaining autocorrelation
**Cross-validation**: Out-of-sample prediction accuracy

## Practical Considerations

### Minimum Number of Time Points

#### General Guidelines
- **Absolute minimum**: 8 time points (4 pre, 4 post)
- **Recommended minimum**: 12 time points (6 pre, 6 post)
- **Optimal for power**: 20+ time points with balanced allocation

#### Special Considerations
- **Seasonal data**: Minimum 2-3 complete cycles
- **High autocorrelation**: More time points needed
- **Multiple interventions**: Additional points for each segment

### Seasonality and Trends

#### Seasonal Adjustment Methods

**Harmonic terms**: cos(2πkt/s) + sin(2πkt/s) for season length s
**Monthly indicators**: Dummy variables for each month
**STL decomposition**: Seasonal and Trend decomposition using Loess

#### Secular Trend Control

**Linear trend**: Included in basic segmented model
**Non-linear trends**: Polynomial or spline terms
**Structural breaks**: Test for pre-existing trend changes

### Confounding Factors

#### Types of Confounding
1. **History**: Co-occurring events affecting outcome
2. **Maturation**: Natural evolution of outcome over time
3. **Instrumentation**: Changes in measurement or data collection
4. **Selection**: Changes in population composition

#### Control Strategies
- **Multiple comparison series**: Control groups or regions
- **Sensitivity analysis**: Test robustness to model specifications
- **Lag analysis**: Examine timing of effect onset
- **Dose-response**: Evaluate gradient of exposure

## Simulation-Based Approaches

### Continuous Outcomes Simulation

#### Data Generation Process

1. **Generate baseline time series**: Y_t^(0) following AR(1) process
2. **Add intervention effects**:
   - Level change: Y_t^(1) = Y_t^(0) + β₂ × X_t
   - Trend change: Y_t^(1) = Y_t^(0) + β₃ × TX_t
3. **Add seasonal components**: Y_t^(2) = Y_t^(1) + seasonal_effects
4. **Add measurement error**: Y_t = Y_t^(2) + ε_t

#### Parameter Specification

**Autocorrelation generation**: ε_{t+1} = ρε_t + √(1-ρ²)u_t
**Effect size calibration**: Based on standardized differences
**Noise level**: Realistic signal-to-noise ratios

### Binary Outcomes Simulation

#### Logistic Model Simulation

1. **Generate linear predictor**: η_t = β₀ + β₁T_t + β₂X_t + β₃TX_t + ε_t
2. **Apply logistic transformation**: p_t = 1/(1 + exp(-η_t))
3. **Generate binary outcomes**: Y_t ~ Bernoulli(p_t)

#### Autocorrelation in Binary Series

**Markov chain approach**: P(Y_t = 1|Y_{t-1} = 1) ≠ P(Y_t = 1|Y_{t-1} = 0)
**Beta-binomial model**: Account for overdispersion and correlation

### Power Estimation

#### Monte Carlo Procedure

1. **Generate nsim datasets** under specified parameters
2. **Fit segmented regression** to each dataset
3. **Test intervention parameters**: Count significant results
4. **Estimate power**: Proportion of significant tests

#### Bias and Coverage Assessment

- **Parameter bias**: E[β̂] - β
- **Coverage probability**: Proportion of confidence intervals containing true parameter
- **Type I error**: Power when β = 0 should equal α

## Assumptions and Limitations

### Model Assumptions

1. **Functional form**: Linear relationship between predictors and outcome
2. **No unmeasured confounders**: All time-varying confounders controlled
3. **Stable variance**: Homoscedasticity over time
4. **Correct autocorrelation structure**: AR(1) may be insufficient
5. **No structural breaks**: Pre-intervention trend is stable

### Design Assumptions

1. **Intervention timing**: Precisely known and implemented
2. **No anticipation effects**: No changes before official intervention
3. **Stable population**: No major demographic shifts
4. **Consistent measurement**: No changes in data collection methods
5. **No co-interventions**: Other policy changes at same time

### Key Limitations

1. **Causal inference**: Weaker than randomized experiments
2. **External validity**: Results may not generalize to other settings
3. **Complex interventions**: Difficult to isolate specific components
4. **Long-term effects**: May require extended follow-up
5. **Statistical power**: Often lower than parallel group designs

## Method Selection Guidelines

### When to Use ITS Analysis

**Recommended when**:
- Randomization is not feasible or ethical
- Population-level intervention with clear implementation date
- Sufficient historical data available (8+ time points)
- Strong theoretical basis for immediate or gradual effects
- Limited risk of major confounding events

**Not recommended when**:
- Insufficient pre-intervention data (<4 time points)
- Multiple simultaneous interventions or policy changes
- Highly unstable baseline trends
- Major confounding events likely
- Individual-level randomization is feasible

### Analytical vs. Simulation Approaches

#### Use Analytical Methods When:
- Standard segmented regression assumptions met
- Simple autocorrelation structure (AR(1))
- Continuous outcomes with normal errors
- Sample size planning for standard designs
- Computational efficiency required

#### Use Simulation Methods When:
- Complex autocorrelation or seasonal patterns
- Non-normal outcomes (binary, count, time-to-event)
- Non-standard effect patterns (delayed, temporary effects)
- Sensitivity analysis for model assumptions
- Small sample sizes where normal approximation questionable

### Practical Decision Framework

1. **Design feasibility** → ITS vs. experimental design
2. **Outcome type** → Analytical vs. simulation approach
3. **Temporal complexity** → Simple vs. complex modeling
4. **Sample size** → Normal approximation vs. exact methods
5. **Computational resources** → Analytical vs. simulation intensity

## Software Implementation

### Core Functions

The DesignPower ITS implementation provides:

1. **Analytical Methods**:
   - `its_power_continuous()` - Power for continuous outcomes
   - `its_power_binary()` - Power for binary outcomes  
   - `its_sample_size()` - Sample size calculation
   - `its_minimum_detectable_effect()` - MDE calculation

2. **Simulation Methods**:
   - `its_simulate_continuous()` - Monte Carlo power estimation
   - `its_simulate_binary()` - Binary outcome simulation
   - `its_sensitivity_analysis()` - Robustness testing

### Parameter Specifications

#### Required Parameters
- **pre_periods**: Number of pre-intervention time points
- **post_periods**: Number of post-intervention time points
- **effect_size**: Standardized effect size for level or trend change
- **alpha**: Significance level (default: 0.05)
- **autocorr**: Autocorrelation coefficient (ρ)

#### Outcome-Specific Parameters

**Continuous Outcomes**:
- **std_dev**: Outcome standard deviation
- **level_change**: Immediate level change (β₂)
- **trend_change**: Change in slope (β₃)

**Binary Outcomes**:
- **baseline_prob**: Pre-intervention probability
- **effect_type**: "level_change", "trend_change", or "both"

#### Advanced Parameters
- **seasonal_period**: Length of seasonal cycle
- **seasonal_amplitude**: Magnitude of seasonal variation
- **trend_shape**: "linear", "quadratic", "exponential"
- **nsim**: Number of simulations (default: 1,000)

### Input Validation

The implementation validates:
- Minimum time points (≥8 total, ≥3 pre and post)
- Valid autocorrelation coefficients (-1 < ρ < 1)
- Valid probability values for binary outcomes (0 < p < 1)
- Reasonable effect sizes and standard deviations
- Balanced design recommendations

## Validation and Quality Assurance

### Analytical Validation

The ITS implementation has been validated against:
- Published power calculation examples
- Simulation studies from the literature
- Comparison with R packages (`its.analysis`, `bcp`, `segmented`)
- Cross-validation with SAS and Stata procedures

### Simulation Validation

Simulation methods validated through:
- Type I error rate verification (power = α when effect = 0)
- Convergence studies with varying nsim
- Comparison with analytical methods for simple cases
- Sensitivity analysis for autocorrelation assumptions

### Known Limitations

1. **Simplified autocorrelation**: Primary focus on AR(1) models
2. **Linear trends**: Limited support for complex non-linear patterns
3. **Balanced designs**: Optimization for equal pre/post periods
4. **Single intervention**: Multiple intervention points not fully supported

## References

### Primary Methodological References

1. **Wagner AK, Soumerai SB, Zhang F, Ross-Degnan D** (2002). Segmented regression analysis of interrupted time series studies in medication use research. *Journal of Clinical Pharmacy and Therapeutics*, 27(4), 299-309.

2. **Penfold RB, Zhang F** (2013). Use of interrupted time series analysis in evaluating health care quality improvements. *Academic Pediatrics*, 13(6), S38-S44.

3. **Bernal JL, Cummins S, Gasparrini A** (2017). Interrupted time series regression for the evaluation of public health interventions: a tutorial. *International Journal of Epidemiology*, 46(1), 348-355.

4. **Lopez Bernal J, Cummins S, Gasparrini A** (2018). The use of controls in interrupted time series studies of public health interventions. *International Journal of Epidemiology*, 47(6), 2082-2093.

### Statistical Theory References

5. **Box GE, Tiao GC** (1975). Intervention analysis with applications to economic and environmental problems. *Journal of the American Statistical Association*, 70(349), 70-79.

6. **McDowall D, McCleary R, Meidinger EE, Hay Jr RA** (1980). *Interrupted Time Series Analysis*. Sage Publications: Beverly Hills, CA.

7. **Harvey AC** (1989). *Forecasting, Structural Time Series Models and the Kalman Filter*. Cambridge University Press.

### Power Analysis References

8. **Zhang F, Wagner AK, Ross-Degnan D** (2011). Simulation-based power calculation for designing interrupted time series analyses of health policy interventions. *Journal of Clinical Epidemiology*, 64(11), 1252-1261.

9. **Fretheim A, Zhang F, Ross-Degnan D, Oxman AD, Cheyne H, Foy R, Goodacre S** (2015). A reanalysis of cluster randomized trials showed interrupted time-series studies were valuable in health system evaluation. *Journal of Clinical Epidemiology*, 68(3), 324-333.

### Design and Implementation References

10. **Shadish WR, Cook TD, Campbell DT** (2002). *Experimental and Quasi-Experimental Designs for Generalized Causal Inference*. Houghton Mifflin: Boston, MA.

11. **Cook TD, Campbell DT** (1979). *Quasi-Experimentation: Design and Analysis Issues for Field Settings*. Houghton Mifflin: Boston, MA.

12. **Ramsay CR, Matowe L, Grilli R, Grimshaw JM, Thomas RE** (2003). Interrupted time series designs in health technology assessment: lessons from two systematic reviews of behavior change strategies. *International Journal of Technology Assessment in Health Care*, 19(4), 613-623.

### Software and Computational References

13. **Linden A** (2015). Conducting interrupted time-series analysis for single-and multiple-group comparisons. *The Stata Journal*, 15(2), 480-500.

14. **Huitema BE, McKean JW** (2000). Design specification issues in time-series intervention models. *Educational and Psychological Measurement*, 60(1), 38-58.

15. **Prais SJ, Winsten CB** (1954). Trend estimators and serial correlation. *Cowles Commission Discussion Paper*, No. 383.

## Appendix: Mathematical Derivations

### A.1 Standard Error Derivations

For the segmented regression model with AR(1) errors, the variance-covariance matrix is:

**Var(β̂) = σ²(X'Ω⁻¹X)⁻¹**

Where Ω is the autocorrelation matrix:
- Ω_{ii} = 1
- Ω_{ij} = ρ^|i-j| for AR(1) structure

### A.2 Design Effect for Autocorrelated Data

The design effect for autocorrelated time series:

**DE = (1 + 2ρ∑_{k=1}^{n-1}(1-k/n)ρ^{k-1})**

For large n and moderate ρ:
**DE ≈ (1 + ρ)/(1 - ρ)**

### A.3 Power Function Derivation

For testing H₀: β₂ = 0 vs H₁: β₂ ≠ 0:

**Power = P(|t| > t_{α/2,df} | β₂ ≠ 0)**
**     = P(|β̂₂/SE(β̂₂)| > t_{α/2,df})**
**     = Φ(|β₂|/SE(β̂₂) - t_{α/2,df}) + Φ(|β₂|/SE(β̂₂) + t_{α/2,df})**

Where Φ is the standard normal CDF when df is large.

### A.4 Optimal Allocation for Trend Detection

For detecting trend change β₃, the optimal allocation maximizes:

**1/Var(β̂₃) = 1/(σ²/∑(TX_t - T̄X)²)**

This is maximized when post-intervention observations are spread over the longest possible time period, suggesting longer post-intervention follow-up improves power for trend detection.