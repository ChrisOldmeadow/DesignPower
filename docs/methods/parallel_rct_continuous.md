# Parallel RCT with Continuous Outcomes

This document details the statistical methodology implemented in DesignPower for calculating sample size, power, and minimum detectable effect for parallel randomized controlled trials with continuous outcomes.

## Analytical Methods

### Sample Size Calculation

The sample size calculation for a parallel group trial with continuous outcomes uses the formula:

$$n = \frac{2\sigma^2(z_{1-\alpha/2} + z_{1-\beta})^2}{\Delta^2}$$

Where:
- $n$ = sample size per group
- $\sigma$ = standard deviation (assumed equal in both groups)
- $\Delta$ = minimal clinically important difference between means
- $\alpha$ = significance level (typically 0.05)
- $\beta$ = type II error rate (1 - power)
- $z_{1-\alpha/2}$ = critical value from the standard normal distribution for a two-sided test
- $z_{1-\beta}$ = critical value from the standard normal distribution corresponding to the desired power

#### Implementation Details

For unequal allocation ratios ($r = n_1/n_2$), the formula is adjusted to:

$$n_1 = \frac{(1+\frac{1}{r})\sigma^2(z_{1-\alpha/2} + z_{1-\beta})^2}{\Delta^2}$$
$$n_2 = n_1/r$$

### Power Calculation

Power is calculated using the formula:

$$1-\beta = \Phi\left(\frac{\Delta\sqrt{n}}{\sqrt{2}\sigma} - z_{1-\alpha/2}\right)$$

Where:
- $\Phi$ = cumulative distribution function of the standard normal distribution
- Other parameters are as defined above

### Minimum Detectable Effect Calculation

The minimum detectable effect (MDE) is calculated as:

$$\Delta = \frac{\sqrt{2}\sigma(z_{1-\alpha/2} + z_{1-\beta})}{\sqrt{n}}$$

Where all parameters are as defined above.

## Simulation Methods

In addition to analytical formulas, DesignPower offers simulation-based calculations that can provide more accurate results, especially in complex scenarios.

### Simulation Algorithm

1. For each simulated trial:
   - Generate samples from normal distributions with specified means and standard deviations
   - Perform a two-sample t-test
   - Record whether the null hypothesis was rejected

2. Sample size calculation:
   - Incrementally increase sample size until the desired power is achieved
   - For each sample size, run multiple simulations and calculate the proportion of rejections

3. Power calculation:
   - For a fixed sample size, run multiple simulations
   - Calculate the proportion of simulations that reject the null hypothesis

4. MDE calculation:
   - Using binary search, find the smallest effect size that achieves the desired power
   - For each effect size, run multiple simulations

## References

1. Chow S-C, Shao J, Wang H. Sample Size Calculations in Clinical Research. 2nd ed. Chapman & Hall/CRC; 2008.

2. Cohen J. Statistical Power Analysis for the Behavioral Sciences. 2nd ed. Lawrence Erlbaum Associates; 1988.

3. Lachin JM. Introduction to sample size determination and power analysis for clinical trials. Control Clin Trials. 1981;2(2):93-113.
