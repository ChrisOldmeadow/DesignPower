# Single-Arm Trials with Binary Outcomes

This document details the statistical methodology implemented in DesignPower for calculating sample size, power, and minimum detectable effect for single-arm trials with binary outcomes.

## Statistical Framework

Single-arm trials with binary outcomes compare the response rate in a treatment group to a pre-specified target value or historical control rate.

### Hypothesis Testing

The hypothesis tests are formulated as:

- $H_0$: $p = p_0$ (The treatment has the reference response rate)
- $H_1$: $p \neq p_0$ (two-sided) or $p > p_0$ or $p < p_0$ (one-sided)

Where:
- $p$ = true response probability in the treatment group
- $p_0$ = reference value (e.g., historical control response rate)

## Analytical Methods

### Sample Size Calculation

#### Normal Approximation Method

For a two-sided test using the normal approximation:

$$n = \frac{z_{1-\alpha/2}^2 p_0(1-p_0) + z_{1-\beta}^2 p_1(1-p_1)}{(p_1 - p_0)^2}$$

For a one-sided test:

$$n = \frac{z_{1-\alpha}^2 p_0(1-p_0) + z_{1-\beta}^2 p_1(1-p_1)}{(p_1 - p_0)^2}$$

Where:
- $n$ = sample size
- $p_0$ = reference response rate (null hypothesis)
- $p_1$ = anticipated response rate under the alternative hypothesis
- $\alpha$ = significance level (typically 0.05)
- $\beta$ = type II error rate (1 - power)
- $z_{1-\alpha/2}$ = critical value from the standard normal distribution for a two-sided test
- $z_{1-\alpha}$ = critical value from the standard normal distribution for a one-sided test
- $z_{1-\beta}$ = critical value from the standard normal distribution corresponding to the desired power

#### With Continuity Correction

For small sample sizes or small differences in proportions, a continuity correction improves accuracy:

$$n = \frac{z_{1-\alpha/2}^2 p_0(1-p_0) + z_{1-\beta}^2 p_1(1-p_1) + (z_{1-\alpha/2}\sqrt{p_0(1-p_0)} + z_{1-\beta}\sqrt{p_1(1-p_1)})^2/4(p_1-p_0)^2}{(p_1 - p_0)^2}$$

### Power Calculation

Power for a fixed sample size is calculated as:

$$1-\beta = \Phi\left(\frac{|p_1 - p_0|\sqrt{n} - z_{1-\alpha/2}\sqrt{p_0(1-p_0)}}{\sqrt{p_1(1-p_1)}}\right)$$

For a one-sided test:

$$1-\beta = \Phi\left(\frac{|p_1 - p_0|\sqrt{n} - z_{1-\alpha}\sqrt{p_0(1-p_0)}}{\sqrt{p_1(1-p_1)}}\right)$$

Where:
- $\Phi$ = cumulative distribution function of the standard normal distribution
- Other parameters are as defined above

### Minimum Detectable Effect Calculation

The minimum detectable effect for a given sample size must typically be solved iteratively, as the value of $p_1$ affects both sides of the equation.

## Exact Methods

### Exact Binomial Test

For small sample sizes, an exact binomial test is more appropriate than the normal approximation. This approach uses the binomial probability mass function directly:

$$P(X \geq k | n, p_0) = \sum_{i=k}^{n} {n \choose i} p_0^i (1-p_0)^{n-i}$$

Where:
- $X$ = number of responses
- $k$ = critical value for rejecting the null hypothesis
- $n$ = sample size
- $p_0$ = reference response rate

Sample size calculation involves finding the smallest $n$ such that:
1. $P(X \geq k | n, p_0) \leq \alpha$ (Type I error control)
2. $P(X < k | n, p_1) \leq \beta$ (Type II error control)

This requires searching over combinations of $n$ and $k$ to find the smallest $n$ that satisfies both conditions.

## Specialized Designs for Phase II Trials

DesignPower implements specialized designs for phase II trials with binary outcomes:

1. **A'Hern Design**: A single-stage design using exact binomial probabilities
2. **Simon's Two-Stage Design**: A two-stage design that allows early stopping for futility

These designs are detailed in separate methodology documents:
- [A'Hern Design](ahern_design.md)
- Simon's Two-Stage Design (under development)

## Simulation Methods

DesignPower offers simulation-based calculations for single-arm trials with binary outcomes.

### Simulation Algorithm

1. For each simulated trial:
   - Generate binary outcomes using Bernoulli random variables with specified probability
   - Perform the selected statistical test (normal approximation, exact binomial, etc.)
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

Single-arm trials with binary outcomes have several limitations:
- Cannot control for confounding factors, placebo effects, or selection bias
- Historical controls may not be directly comparable to the current treatment group
- Results may be influenced by changes in supportive care or diagnostic criteria over time

Despite these limitations, they are commonly used in:
- Early-phase oncology trials where the primary endpoint is objective response rate
- Rare diseases where recruitment for a controlled trial is challenging
- Situations where randomization to placebo is ethically problematic

## References

1. Fleiss JL, Levin B, Paik MC. Statistical Methods for Rates and Proportions. 3rd ed. Wiley; 2003.

2. Jung SH. Statistical issues for design and analysis of single-arm multi-stage phase II cancer clinical trials. Contemp Clin Trials. 2015;42:9-17.

3. A'Hern RP. Sample size tables for exact single-stage phase II designs. Stat Med. 2001;20(6):859-866.
