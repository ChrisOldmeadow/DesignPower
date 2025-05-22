# Parallel RCT with Binary Outcomes

This document details the statistical methodology implemented in DesignPower for calculating sample size, power, and minimum detectable effect for parallel randomized controlled trials with binary outcomes.

## Analytical Methods

### Sample Size Calculation

The sample size calculation for a parallel group trial with binary outcomes can be performed using several methods:

#### Normal Approximation Method

For a two-sided test of proportions, the sample size per group is calculated as:

$$n = \frac{(z_{1-\alpha/2} + z_{1-\beta})^2 [p_1(1-p_1) + p_2(1-p_2)]}{(p_1 - p_2)^2}$$

Where:
- $n$ = sample size per group
- $p_1$ = event rate in the control group
- $p_2$ = event rate in the treatment group
- $\alpha$ = significance level (typically 0.05)
- $\beta$ = type II error rate (1 - power)
- $z_{1-\alpha/2}$ = critical value from the standard normal distribution for a two-sided test
- $z_{1-\beta}$ = critical value from the standard normal distribution corresponding to the desired power

#### With Continuity Correction

For small sample sizes or small differences in proportions, a continuity correction can improve accuracy:

$$n = \frac{(z_{1-\alpha/2} + z_{1-\beta})^2 [p_1(1-p_1) + p_2(1-p_2)] + z_{1-\alpha/2}^2/2}{(p_1 - p_2)^2}$$

#### For Unequal Allocation

When allocation ratio is not 1:1 ($r = n_1/n_2$), the sample sizes are:

$$n_1 = \frac{(z_{1-\alpha/2} + z_{1-\beta})^2 [p_1(1-p_1) + p_2(1-p_2)/r]}{(p_1 - p_2)^2}$$
$$n_2 = n_1/r$$

### Power Calculation

Power for a fixed sample size can be calculated as:

$$1-\beta = \Phi\left(\frac{|p_1 - p_2|\sqrt{n}}{\sqrt{p_1(1-p_1) + p_2(1-p_2)}} - z_{1-\alpha/2}\right)$$

Where:
- $\Phi$ = cumulative distribution function of the standard normal distribution
- Other parameters are as defined above

### Minimum Detectable Effect Calculation

The minimum detectable difference in proportions for a given sample size is:

$$|p_1 - p_2| = (z_{1-\alpha/2} + z_{1-\beta})\sqrt{\frac{p_1(1-p_1) + p_2(1-p_2)}{n}}$$

This is typically solved iteratively because $p_2$ appears on both sides of the equation.

## Alternative Test Methods

DesignPower offers several statistical test options for binary outcomes:

### Fisher's Exact Test

For small sample sizes, Fisher's exact test provides a more accurate Type I error rate than the normal approximation. The sample size and power calculations are based on exact binomial probabilities.

### Likelihood Ratio Test

The likelihood ratio test provides good statistical properties and is implemented as an alternative to the normal approximation and Fisher's exact test.

## Simulation Methods

DesignPower implements simulation-based approaches for binary outcomes, which are particularly useful when:
- Sample sizes are small
- The normal approximation may not be valid
- Complex designs or adjustments are needed

### Simulation Algorithm

1. For each simulated trial:
   - Generate binary outcomes using Bernoulli random variables with specified probabilities
   - Perform the selected statistical test (normal approximation, Fisher's exact, or likelihood ratio)
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

1. Fleiss JL, Levin B, Paik MC. Statistical Methods for Rates and Proportions. 3rd ed. Wiley; 2003.

2. Chow S-C, Shao J, Wang H. Sample Size Calculations in Clinical Research. 2nd ed. Chapman & Hall/CRC; 2008.

3. Lachin JM. Biostatistical Methods: The Assessment of Relative Risks. 2nd ed. Wiley; 2011.
