# Single-Arm Trials with Continuous Outcomes

This document details the statistical methodology implemented in DesignPower for calculating sample size, power, and minimum detectable effect for single-arm trials with continuous outcomes.

## Statistical Framework

Single-arm trials compare a single treatment group to a fixed reference value or historical control. For continuous outcomes, this typically involves comparing the mean of a measured variable to a pre-specified value.

### Hypothesis Testing

The hypothesis tests are formulated as:

- $H_0$: $\mu = \mu_0$ (The treatment has no effect)
- $H_1$: $\mu \neq \mu_0$ (two-sided) or $\mu > \mu_0$ or $\mu < \mu_0$ (one-sided)

Where:
- $\mu$ = true mean of the outcome in the treatment group
- $\mu_0$ = reference value (e.g., historical control mean)

## Analytical Methods

### Sample Size Calculation

The sample size calculation for a single-arm trial with continuous outcomes uses the formula:

$$n = \frac{\sigma^2(z_{1-\alpha/2} + z_{1-\beta})^2}{(\mu - \mu_0)^2}$$

For a one-sided test:

$$n = \frac{\sigma^2(z_{1-\alpha} + z_{1-\beta})^2}{(\mu - \mu_0)^2}$$

Where:
- $n$ = sample size
- $\sigma$ = standard deviation of the outcome
- $\mu - \mu_0$ = minimal clinically important difference
- $\alpha$ = significance level (typically 0.05)
- $\beta$ = type II error rate (1 - power)
- $z_{1-\alpha/2}$ = critical value from the standard normal distribution for a two-sided test
- $z_{1-\alpha}$ = critical value from the standard normal distribution for a one-sided test
- $z_{1-\beta}$ = critical value from the standard normal distribution corresponding to the desired power

### Power Calculation

Power for a fixed sample size is calculated as:

$$1-\beta = \Phi\left(\frac{|\mu - \mu_0|\sqrt{n}}{\sigma} - z_{1-\alpha/2}\right)$$

For a one-sided test:

$$1-\beta = \Phi\left(\frac{|\mu - \mu_0|\sqrt{n}}{\sigma} - z_{1-\alpha}\right)$$

Where:
- $\Phi$ = cumulative distribution function of the standard normal distribution
- Other parameters are as defined above

### Minimum Detectable Effect Calculation

The minimum detectable effect for a given sample size is:

$$|\mu - \mu_0| = \frac{\sigma(z_{1-\alpha/2} + z_{1-\beta})}{\sqrt{n}}$$

For a one-sided test:

$$|\mu - \mu_0| = \frac{\sigma(z_{1-\alpha} + z_{1-\beta})}{\sqrt{n}}$$

## Adjustments for Known Population Parameters

When the population standard deviation is known (rather than estimated), the sample size formula uses the standard normal distribution (z-test) rather than the t-distribution.

When the population standard deviation is unknown and must be estimated from the data, the sample size formula may be adjusted to use the t-distribution, particularly for small sample sizes:

$$n = \frac{\sigma^2(t_{n-1,1-\alpha/2} + z_{1-\beta})^2}{(\mu - \mu_0)^2}$$

This requires an iterative solution since the degrees of freedom depend on the sample size.

## Simulation Methods

DesignPower also offers simulation-based calculations for single-arm trials with continuous outcomes.

### Simulation Algorithm

1. For each simulated trial:
   - Generate samples from a normal distribution with specified mean and standard deviation
   - Perform a one-sample t-test (or z-test if the standard deviation is known)
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

## Considerations for Single-Arm Trials

Single-arm trials have several limitations compared to randomized controlled trials:
- Cannot control for confounding factors, placebo effects, or selection bias
- Historical controls may not be directly comparable to the current treatment group
- Results may be influenced by changes in supportive care or diagnostic criteria over time

Despite these limitations, single-arm trials are useful in several contexts:
- Early-phase studies (e.g., phase II trials in oncology)
- Rare diseases where recruitment for a controlled trial is challenging
- Ethical considerations that make randomization problematic

## References

1. Chow S-C, Shao J, Wang H. Sample Size Calculations in Clinical Research. 2nd ed. Chapman & Hall/CRC; 2008.

2. Lachin JM. Introduction to sample size determination and power analysis for clinical trials. Control Clin Trials. 1981;2(2):93-113.

3. Vickers AJ. How to design a phase II clinical trial. J Natl Cancer Inst. 2006;98(16):1095-1096.
