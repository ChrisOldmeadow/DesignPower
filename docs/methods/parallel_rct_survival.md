# Parallel RCT with Survival Outcomes

This document details the statistical methodology implemented in DesignPower for calculating sample size, power, and minimum detectable effect for parallel randomized controlled trials with time-to-event (survival) outcomes.

## Analytical Methods

### Assumptions

The methods implemented in DesignPower for survival outcomes are based on the following assumptions:

1. Exponential survival distribution in both groups
2. Proportional hazards between treatment and control groups
3. Uniform accrual of patients over the enrollment period
4. Potential dropouts occurring at a constant rate
5. Analysis using the log-rank test

### Sample Size Calculation

The sample size calculation for a parallel group trial with survival outcomes uses the formula derived by Schoenfeld (1983):

$$n = \frac{4(z_{1-\alpha/2} + z_{1-\beta})^2}{[\ln(HR)]^2 \times P_E}$$

Where:
- $n$ = total sample size (both groups combined)
- $HR$ = hazard ratio ($\lambda_2 / \lambda_1$, where $\lambda$ is the hazard rate)
- $P_E$ = probability of observing an event
- $\alpha$ = significance level (typically 0.05)
- $\beta$ = type II error rate (1 - power)
- $z_{1-\alpha/2}$ = critical value from the standard normal distribution for a two-sided test
- $z_{1-\beta}$ = critical value from the standard normal distribution corresponding to the desired power

#### Calculating Probability of Event

For trials with specified accrual and follow-up periods, the probability of observing an event depends on:
- Median survival in each group
- Accrual (enrollment) period duration
- Follow-up period duration after the last patient is enrolled
- Dropout rates

The probability of event ($P_E$) is calculated taking into account all these factors and the exponential survival distribution.

#### Parameterization with Median Survival Times

Since median survival is often more clinically interpretable than hazard rates, DesignPower uses median survival as input. The relationship between median survival and hazard rate is:

$$\lambda = \frac{\ln(2)}{median}$$

And the relationship between median survival times in the two groups is:

$$median_2 = \frac{median_1}{HR}$$

Where $HR$ is the hazard ratio.

### Power Calculation

Power for a fixed sample size is calculated as:

$$1-\beta = \Phi\left(\frac{|\ln(HR)| \times \sqrt{n \times P_E}}{2} - z_{1-\alpha/2}\right)$$

Where:
- $\Phi$ = cumulative distribution function of the standard normal distribution
- Other parameters are as defined above

### Minimum Detectable Effect Calculation

The minimum detectable hazard ratio for a given sample size is:

$$\ln(HR) = \frac{2(z_{1-\alpha/2} + z_{1-\beta})}{\sqrt{n \times P_E}}$$

This can be solved for $HR$ to get the minimum detectable hazard ratio.

## Impact of Study Design Parameters

### Accrual and Follow-up Periods

The length of accrual and follow-up periods affects the number of events observed and therefore the required sample size:
- Longer follow-up periods generally lead to more events and smaller required sample sizes
- Slow accrual rates may increase the total study duration but can reduce the required sample size

### Dropout Rate

Accounting for anticipated dropouts increases the required sample size. DesignPower applies the formula:

$$n_{adjusted} = \frac{n}{(1 - d)^2}$$

Where $d$ is the anticipated dropout rate.

## Simulation Methods

For more complex scenarios or when the assumptions of the analytical method may not hold, DesignPower offers simulation-based calculations.

### Simulation Algorithm

1. For each simulated trial:
   - Generate survival times from exponential distributions with the specified median survival times
   - Generate accrual times uniformly over the accrual period
   - Calculate follow-up time for each subject
   - Apply censoring based on end of study and dropout rate
   - Perform a log-rank test to compare survival curves
   - Record whether the null hypothesis was rejected

2. Sample size calculation:
   - Incrementally increase sample size until the desired power is achieved
   - For each sample size, run multiple simulations and calculate the proportion of rejections

3. Power calculation:
   - For a fixed sample size, run multiple simulations
   - Calculate the proportion of simulations that reject the null hypothesis

4. MDE calculation:
   - Using binary search, find the smallest hazard ratio that achieves the desired power
   - For each hazard ratio, run multiple simulations

## References

1. Schoenfeld DA. Sample-size formula for the proportional-hazards regression model. Biometrics. 1983;39(2):499-503.

2. Lachin JM, Foulkes MA. Evaluation of sample size and power for analyses of survival with allowance for nonuniform patient entry, losses to follow-up, noncompliance, and stratification. Biometrics. 1986;42(3):507-519.

3. Freedman LS. Tables of the number of patients required in clinical trials using the logrank test. Stat Med. 1982;1(2):121-129.
