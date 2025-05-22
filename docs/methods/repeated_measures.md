# Repeated Measures Designs

This document details the statistical methodology implemented in DesignPower for calculating sample size, power, and minimum detectable effect for studies with repeated measures designs.

## Background

Repeated measures designs involve multiple measurements on the same subject over time or under different conditions. These designs are more efficient than between-subjects designs because they control for individual variability, but they introduce correlation between measurements that must be accounted for in the statistical analysis.

## Types of Repeated Measures Designs

DesignPower supports several repeated measures designs:

### Pre-Post Design with Control Group

In this design, measurements are taken at baseline and follow-up in both treatment and control groups. The primary interest is in the difference between groups in the change from baseline to follow-up.

### Crossover Design

In a crossover design, each subject receives both treatments in sequence, with a washout period between treatments to minimize carryover effects. This design is highly efficient but may not be suitable when carryover effects are substantial or the condition is progressive.

### Longitudinal Design

Longitudinal designs involve multiple measurements over time, often with more than two time points. These designs allow for the study of trajectories and time-varying effects.

## Statistical Framework

### Correlation Structure

The correlation between repeated measurements is characterized by various correlation structures:

1. **Compound Symmetry**: Assumes equal correlation between any two measurements
2. **Autoregressive**: Correlation decreases with increasing time between measurements
3. **Unstructured**: Each pair of measurements has a unique correlation

The choice of correlation structure affects the sample size and power calculations.

## Analytical Methods

### Sample Size Calculation for Pre-Post Design

The sample size calculation for a pre-post design with a control group uses the formula:

$$n = \frac{2(z_{1-\alpha/2} + z_{1-\beta})^2 \sigma_d^2}{\Delta^2}$$

Where:
- $n$ = sample size per group
- $\sigma_d^2$ = variance of the difference scores (taking into account the correlation)
- $\Delta$ = minimal clinically important difference in the change from baseline between groups
- $\alpha$ = significance level (typically 0.05)
- $\beta$ = type II error rate (1 - power)
- $z_{1-\alpha/2}$ = critical value from the standard normal distribution for a two-sided test
- $z_{1-\beta}$ = critical value from the standard normal distribution corresponding to the desired power

The variance of the difference scores is calculated as:

$$\sigma_d^2 = \sigma^2(2 - 2\rho)$$

Where:
- $\sigma^2$ = variance of the individual measurements
- $\rho$ = correlation between repeated measurements

### Sample Size Calculation for Crossover Design

For a 2×2 crossover design (two treatments, two periods), the sample size is:

$$n = \frac{2(z_{1-\alpha/2} + z_{1-\beta})^2 \sigma_w^2}{\Delta^2}$$

Where:
- $\sigma_w^2$ = within-subject variance
- Other parameters are as defined above

The within-subject variance is typically much smaller than the total variance, making crossover designs more efficient than parallel group designs.

### Sample Size Calculation for Longitudinal Design

For longitudinal designs with more than two time points, the sample size calculation depends on the specific hypothesis being tested (e.g., group difference at a specific time point, overall group difference across time, group-by-time interaction).

For testing the group-by-time interaction in a design with $t$ time points, the formula becomes more complex and often requires specialized software or simulation methods.

### Power Calculation

Power for a fixed sample size in a pre-post design can be calculated as:

$$1-\beta = \Phi\left(\frac{\Delta\sqrt{n}}{\sqrt{2\sigma_d^2}} - z_{1-\alpha/2}\right)$$

Where:
- $\Phi$ = cumulative distribution function of the standard normal distribution
- Other parameters are as defined above

### Minimum Detectable Effect Calculation

The minimum detectable effect for a given sample size in a pre-post design is:

$$\Delta = \frac{\sqrt{2\sigma_d^2}(z_{1-\alpha/2} + z_{1-\beta})}{\sqrt{n}}$$

## Analysis Methods

DesignPower provides sample size and power calculations for several analysis methods for repeated measures designs:

### ANCOVA (Analysis of Covariance)

For pre-post designs, ANCOVA with the baseline measurement as a covariate is often more powerful than analyzing change scores:

$$Y_{post,i} = \beta_0 + \beta_1 X_i + \beta_2 Y_{pre,i} + \epsilon_i$$

Where:
- $Y_{post,i}$ = post-treatment measurement for subject $i$
- $X_i$ = treatment indicator for subject $i$
- $Y_{pre,i}$ = pre-treatment measurement for subject $i$
- $\beta_0$ = intercept
- $\beta_1$ = treatment effect
- $\beta_2$ = coefficient for the baseline measurement
- $\epsilon_i$ = residual error for subject $i$

### Linear Mixed Models

For designs with more than two time points, linear mixed models account for the correlation structure and can handle missing data:

$$Y_{ij} = \beta_0 + \beta_1 X_i + \beta_2 t_j + \beta_3 X_i t_j + u_i + \epsilon_{ij}$$

Where:
- $Y_{ij}$ = measurement for subject $i$ at time $j$
- $X_i$ = treatment indicator for subject $i$
- $t_j$ = time point $j$
- $\beta_0$ = intercept
- $\beta_1$ = main effect of treatment
- $\beta_2$ = main effect of time
- $\beta_3$ = treatment-by-time interaction
- $u_i$ = random effect for subject $i$
- $\epsilon_{ij}$ = residual error for subject $i$ at time $j$

## Simulation Methods

DesignPower implements simulation-based approaches for repeated measures designs, which are particularly useful when:
- Complex correlation structures are present
- Missing data is anticipated
- Non-linear trajectories are expected

### Simulation Algorithm

1. For each simulated study:
   - Generate correlated measurements for each subject using the specified correlation structure
   - Apply the treatment effect according to the design
   - Perform the selected analysis method (ANCOVA, mixed model, etc.)
   - Record whether the null hypothesis was rejected

2. Sample size calculation:
   - Incrementally increase the sample size until the desired power is achieved
   - For each sample size, run multiple simulations and calculate the proportion of rejections

3. Power calculation:
   - For a fixed sample size, run multiple simulations
   - Calculate the proportion of simulations that reject the null hypothesis

4. MDE calculation:
   - Using binary search, find the smallest effect size that achieves the desired power
   - For each effect size, run multiple simulations

## Practical Considerations

### Correlation Estimation

The correlation between repeated measurements is a critical parameter. DesignPower provides guidance on:
- Using published correlations from similar studies
- Performing sensitivity analyses across a range of plausible correlation values
- How correlation affects efficiency (higher correlation generally leads to smaller required sample size)

### Missing Data

Repeated measures designs are susceptible to missing data due to dropout or missed visits. DesignPower accounts for this by:
- Allowing for specification of anticipated dropout rates
- Adjusting sample size calculations accordingly
- Recommending analysis methods that can handle missing data (e.g., mixed models)

### Period and Carryover Effects in Crossover Designs

Crossover designs must account for potential period effects and carryover effects. DesignPower provides:
- Sample size calculations that account for these effects
- Recommendations for washout periods
- Guidelines for when a crossover design is appropriate

## References

1. Fitzmaurice GM, Laird NM, Ware JH. Applied Longitudinal Analysis. 2nd ed. Wiley; 2011.

2. Diggle PJ, Heagerty P, Liang K-Y, Zeger SL. Analysis of Longitudinal Data. 2nd ed. Oxford University Press; 2002.

3. Chow S-C, Liu J-P, Wang H. Design and Analysis of Bioavailability and Bioequivalence Studies. 3rd ed. Chapman & Hall/CRC; 2008.

4. Lu K, Mehrotra DV, Liu G. Sample size determination for constrained longitudinal data analysis. Stat Med. 2009;28(4):679-699.
