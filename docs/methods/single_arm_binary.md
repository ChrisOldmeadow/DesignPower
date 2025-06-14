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
2. **Simon's Two-Stage Design**: A two-stage design that allows early stopping for futility *[See dedicated documentation](simons_two_stage_design.md)*

### A'Hern Design Implementation

DesignPower's A'Hern design implementation uses a **hybrid approach**:

#### Lookup Table Method (Standard Cases)
For commonly used parameter combinations (α = 0.05, β = 0.1, 0.2), the implementation uses pre-computed lookup tables from A'Hern (2001) Table 1, ensuring exact matches to published values.

#### Enhanced Algorithm (Non-Standard Cases)
For parameter combinations not in the lookup tables, an improved search algorithm finds optimal (n, r) pairs:

```python
# Balanced search approach
for n in range(1, max_n):
    for r in range(n+1):
        # Check Type I error constraint
        type_1_error = 1 - binom.cdf(r-1, n, p0)
        if type_1_error > alpha:
            continue
            
        # Check Type II error constraint  
        type_2_error = binom.cdf(r-1, n, p1)
        if type_2_error <= beta:
            return n, r  # Found valid design
```

#### Implementation Features
The enhanced implementation provides reliable results for both standard and custom parameter combinations.

### Simon's Two-Stage Design Implementation

DesignPower implements Simon's optimal and minimax two-stage designs using a **hybrid optimization approach** that combines computational efficiency with unlimited flexibility.

#### Mathematical Framework

Simon's two-stage design allows early stopping for futility after an interim analysis. The design parameters are:

- **n₁**: Stage 1 sample size
- **r₁**: Stage 1 rejection threshold (stop if responses ≤ r₁)
- **n**: Total sample size (both stages)
- **r**: Final rejection threshold (reject H₀ if total responses > r)

#### Decision Rules

1. **Stage 1**: Enroll n₁ patients
   - If responses ≤ r₁: Stop for futility (accept H₀)
   - If responses > r₁: Continue to stage 2

2. **Stage 2**: Enroll additional (n - n₁) patients
   - If total responses > r: Reject H₀ (promising treatment)
   - If total responses ≤ r: Accept H₀ (ineffective treatment)

#### Optimization Criteria

**Optimal Design**: Minimize expected sample size under H₀
$$\text{minimize } E[N|H_0] = n_1 + (n-n_1) \cdot P(\text{continue}|H_0)$$

**Minimax Design**: Minimize maximum sample size
$$\text{minimize } \max(N) = n$$

Both subject to constraints:
- $P(\text{Type I error}) \leq \alpha$
- $P(\text{Type II error}) \leq \beta$

#### Implementation Strategy

DesignPower uses a **three-tier approach** for complete coverage:

##### Tier 1: Lookup Table (Instant Results)
Pre-computed designs from Simon (1989):

```python
# Simon (1989) Table 1-4 examples
SIMON_DESIGNS = {
    (0.05, 0.25, 0.05, 0.2): {
        'optimal': (n1=9, r1=0, n=17, r=2, EN0=11.9),
        'minimax': (n1=12, r1=0, n=16, r=2, EN0=12.7)
    },
    (0.10, 0.30, 0.05, 0.2): {
        'optimal': (n1=10, r1=0, n=29, r=4, EN0=15.0),
        'minimax': (n1=15, r1=1, n=25, r=4, EN0=17.3)
    },
    # ... additional standard designs
}
```

##### Tier 2: Approximate Matching (Near-Instant)
For parameters within tolerance (±0.005) of standard cases, return closest standard design.

##### Tier 3: Full Optimization Algorithm (1-10 seconds)
Exhaustive search over discrete parameter space for custom requirements:

```python
def simon_optimization(p0, p1, alpha, beta, design_type):
    admissible_designs = []
    
    # Search space
    for n in range(10, max_n):
        for n1 in range(5, n):
            for r1 in range(0, n1):
                for r in range(r1, n):
                    # Calculate exact error rates
                    alpha_actual = calculate_type1_error(n1, n, r1, r, p0)
                    beta_actual = calculate_type2_error(n1, n, r1, r, p1)
                    
                    # Check constraints
                    if alpha_actual <= alpha and beta_actual <= beta:
                        en0 = calculate_expected_n(n1, n, r1, p0)
                        admissible_designs.append((n1, r1, n, r, en0))
    
    # Select optimal design
    if design_type == 'optimal':
        return min(admissible_designs, key=lambda x: x[4])  # min EN0
    else:  # minimax
        return min(admissible_designs, key=lambda x: x[2])  # min n
```

#### Error Rate Calculations

**Type I Error** (exact binomial calculation):
$$\alpha = P(\text{reject } H_0 | H_0) = \sum_{x_1=r_1+1}^{n_1} P(X_1=x_1|p_0) \cdot P(X_2 > r-x_1|p_0)$$

**Type II Error**:
$$\beta = P(\text{accept } H_0 | H_1) = 1 - \sum_{x_1=r_1+1}^{n_1} P(X_1=x_1|p_1) \cdot P(X_2 > r-x_1|p_1)$$

#### Implementation Coverage

- **Standard cases**: Instant lookup from published literature
- **Custom cases**: Full optimization in 0.1-10 seconds
- **Edge cases**: Graceful error handling for impossible combinations
- **Unlimited flexibility**: Handles any valid (p₀, p₁, α, β) combination

These designs are detailed in separate methodology documents:
- [A'Hern Design](ahern_design.md)
- [Simon's Two-Stage Design](simons_two_stage_design.md)

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

4. Simon R. Optimal two-stage designs for phase II clinical trials. Control Clin Trials. 1989;10(1):1-10.
