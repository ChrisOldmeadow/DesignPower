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

## Non-Inferiority Trials

Non-inferiority trials aim to show that a new treatment is not unacceptably worse than a standard treatment. The hypothesis setup differs from superiority trials:

- Null Hypothesis ($H_0$): The new treatment is worse than the standard treatment by at least the non-inferiority margin ($\delta$).
  - For a 'lower is better' outcome: $H_0: \mu_T - \mu_S \ge \delta$
  - For an 'upper is better' outcome: $H_0: \mu_S - \mu_T \ge \delta$ (or $\mu_T - \mu_S \le -\delta$)
- Alternative Hypothesis ($H_1$): The new treatment is not worse than the standard treatment by the non-inferiority margin.
  - For a 'lower is better' outcome: $H_1: \mu_T - \mu_S < \delta$
  - For an 'upper is better' outcome: $H_1: \mu_S - \mu_T < \delta$ (or $\mu_T - \mu_S > -\delta$)

Key considerations for non-inferiority trials:
- **Non-Inferiority Margin ($\delta$)**: This is a pre-specified, clinically meaningful threshold. It must be positive.
- **One-Sided Test**: Non-inferiority tests are inherently one-sided, so the significance level $\alpha$ is used directly (e.g., $z_{1-\alpha}$ instead of $z_{1-\alpha/2}$). DesignPower's non-inferiority functions handle this internally.
- **Assumed Difference**: The power of a non-inferiority trial is sensitive to the assumed true difference between the treatments. If the true difference is close to the non-inferiority margin, power will be low.

### Sample Size for Non-Inferiority

The sample size formula (assuming $\mu_T - \mu_S = 0$ for simplicity, i.e., true equivalence) is:

$$n = \frac{2\sigma^2(z_{1-\alpha} + z_{1-\beta})^2}{(\delta - (\mu_T - \mu_S)_{assumed})^2}$$ 

More generally, if an `assumed_difference` $(\mu_T - \mu_S)_{assumed}$ is specified:

$$n_{per\_group} = \frac{(1+1/r)\sigma^2(z_{1-\alpha} + z_{1-\beta})^2}{(\delta - |(\mu_T - \mu_S)_{assumed}|)^2}$$ 

(Note: The exact formulation depends on the direction of the margin and assumed difference. DesignPower's `sample_size_continuous_non_inferiority` and `sample_size_continuous_non_inferiority_sim` implement the appropriate logic.)

### Power for Non-Inferiority

Power is calculated based on the distance between the assumed true difference and the non-inferiority margin, using a one-sided critical value $z_{1-\alpha}$.

### Combining Non-Inferiority with Repeated Measures

When a non-inferiority trial for continuous outcomes utilizes a repeated measures design (e.g., pre-post measurements), the standard deviation ($\sigma$) used in the analytical formulas, or the `sd1` and `sd2` parameters in simulation functions, should represent the *effective standard deviation*.

This effective standard deviation is calculated based on the raw standard deviation of the outcome, the correlation between repeated measurements, and the chosen analysis method (e.g., 'change_score' or 'ANCOVA'). For details on how the effective standard deviation is derived in repeated measures designs, please refer to the `docs/methods/repeated_measures.md` document.

DesignPower's simulation functions (`power_continuous_non_inferiority_sim` and `sample_size_continuous_non_inferiority_sim`) automatically handle this when `repeated_measures=True` by taking the raw outcome SDs and applying the correlation and analysis method to compute the effective SDs for the simulation.

## Simulation Methods

In addition to analytical formulas, DesignPower offers simulation-based calculations that can provide more accurate results, especially in complex scenarios such as non-inferiority trials or when incorporating repeated measures.

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
