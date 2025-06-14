# A'Hern Design for Single-Arm Binary Outcome Trials

This document details the methodology implemented in DesignPower for A'Hern designs, which are used for single-arm phase II trials with binary outcomes.

## Background

The A'Hern design is a single-stage design for phase II trials that uses exact binomial probabilities rather than normal approximations. This approach is more appropriate for smaller sample sizes, which are common in phase II trials.

## Statistical Framework

### Hypothesis Testing

The A'Hern design tests the following hypotheses:

- $H_0$: $p \leq p_0$ (The treatment is ineffective)
- $H_1$: $p \geq p_1$ (The treatment is effective)

Where:
- $p$ = true response probability
- $p_0$ = maximum response probability that would indicate the treatment is ineffective (null hypothesis)
- $p_1$ = minimum response probability that would indicate the treatment is promising (alternative hypothesis)

### Sample Size and Rejection Threshold Calculation

The A'Hern design determines both the required sample size ($n$) and the rejection threshold ($r$). If $r$ or more responses are observed in $n$ patients, the null hypothesis is rejected, and the treatment is considered promising.

The calculation uses exact binomial probabilities to:

1. Control the Type I error at level $\alpha$:
   $P(X \geq r | p = p_0) \leq \alpha$

2. Control the Type II error at level $\beta$:
   $P(X < r | p = p_1) \leq \beta$

Where $X$ follows a binomial distribution $B(n, p)$.

### Implementation Algorithm

DesignPower uses a **hybrid approach** for optimal performance:

#### Lookup Table Method (Standard Cases)
For commonly used parameter combinations (α = 0.05, β = 0.1, 0.2), pre-computed values from A'Hern (2001) Table 1 ensure exact matches:

```python
table_key = (round(p0, 2), round(p1, 2), round(alpha, 2), round(beta, 1))
if table_key in AHERN_TABLE:
    return AHERN_TABLE[table_key]  # (n, r)
```

#### Enhanced Search Algorithm (Non-Standard Cases)
For other parameter combinations, a balanced search algorithm finds optimal (n, r) pairs:

1. For each possible sample size $n$, starting from a minimum value:
   - Calculate the smallest value of $r$ such that $P(X \geq r | p = p_0) \leq \alpha$
   - Calculate $P(X < r | p = p_1)$ to determine if the Type II error is controlled

2. Select the smallest $n$ for which both error constraints are satisfied

3. Return the sample size $n$ and rejection threshold $r$

#### Floating-Point Precision Handling
The lookup table uses rounded keys to handle floating-point precision issues:
- p0, p1: Rounded to 2 decimal places
- alpha: Rounded to 2 decimal places  
- beta: Rounded to 1 decimal place

### Actual Error Rates

Because the binomial distribution is discrete, the actual error rates are typically lower than the specified levels:

- Actual Type I error: $P(X \geq r | p = p_0)$
- Actual Type II error: $P(X < r | p = p_1)$
- Actual power: $1 - P(X < r | p = p_1)$

## Interpretation of Results

For an A'Hern design with sample size $n$ and rejection threshold $r$:

- If $r$ or more responses are observed, reject $H_0$ and consider the treatment promising
- If fewer than $r$ responses are observed, do not reject $H_0$ and consider the treatment not promising enough for further investigation

## Advantages

1. Uses exact binomial probabilities, which is more appropriate for small sample sizes
2. Does not rely on normal approximations that may be invalid for small samples
3. Simple to implement and interpret
4. Provides clear decision rules for proceeding with treatment development

## Algorithm Implementation

The DesignPower implementation uses a hybrid approach:

1. **Hybrid Strategy**: Combines lookup tables with enhanced search algorithm
2. **Precision Handling**: Robust floating-point arithmetic for parameter matching  
3. **Performance**: Instant results for standard cases, fast computation for non-standard cases

## References

1. A'Hern RP. Sample size tables for exact single-stage phase II designs. Stat Med. 2001;20(6):859-866.

2. Jung SH. Statistical issues for design and analysis of single-arm multi-stage phase II cancer clinical trials. Contemp Clin Trials. 2015;42:9-17.

3. Machin D, Campbell MJ, Tan SB, Tan SH. Sample Size Tables for Clinical Studies. 3rd ed. Wiley-Blackwell; 2009.
