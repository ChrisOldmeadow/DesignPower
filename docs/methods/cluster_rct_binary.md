# Cluster RCT with Binary Outcomes

This document details the statistical methodology implemented in DesignPower for calculating sample size, power, and minimum detectable effect for cluster randomized controlled trials (cRCTs) with binary outcomes.

## Background

In cluster randomized trials with binary outcomes, the outcome for each individual is a binary variable (e.g., success/failure, event/no event), but randomization occurs at the cluster level. The correlation among individuals within the same cluster must be accounted for in the design and analysis.

## Statistical Framework

### Intracluster Correlation Coefficient

For binary outcomes, the intracluster correlation coefficient (ICC, denoted as $\rho$) measures the degree of similarity in responses among individuals within the same cluster.

For a binary outcome, the ICC can be defined as:

$$\rho = \frac{\sigma_b^2}{\sigma_b^2 + \pi(1-\pi)}$$

Where:
- $\sigma_b^2$ = between-cluster variance on the latent scale
- $\pi$ = overall probability of the outcome
- $\pi(1-\pi)$ = total variance for a binary outcome under independence

### Design Effect

The design effect (DE) quantifies how much the sample size needs to be inflated compared to an individually randomized trial:

$$DE = 1 + (m - 1) \times \rho$$

Where:
- $m$ = average cluster size
- $\rho$ = intracluster correlation coefficient

## Analytical Methods

### Sample Size Calculation

The sample size calculation for a cluster randomized trial with binary outcomes builds on the formula for individually randomized trials:

$$n_{individual} = \frac{(z_{1-\alpha/2} + z_{1-\beta})^2 [p_1(1-p_1) + p_2(1-p_2)]}{(p_1 - p_2)^2}$$

Applying the design effect, the number of individuals required becomes:

$$n_{cluster} = n_{individual} \times DE = n_{individual} \times [1 + (m - 1) \times \rho]$$

The number of clusters required per arm is:

$$k = \frac{n_{cluster}}{m}$$

#### With Unequal Cluster Sizes

When cluster sizes vary, the design effect can be adjusted as:

$$DE = 1 + [(1 + CV^2) \times m - 1] \times \rho$$

Where:
- $CV$ = coefficient of variation of cluster sizes
- $m$ = average cluster size

### Power Calculation

Power for a fixed number of clusters and cluster size can be calculated as:

$$1-\beta = \Phi\left(\frac{|p_1 - p_2|\sqrt{k \times m}}{\sqrt{[p_1(1-p_1) + p_2(1-p_2)] \times DE}} - z_{1-\alpha/2}\right)$$

Where:
- $\Phi$ = cumulative distribution function of the standard normal distribution
- $k$ = number of clusters per arm
- Other parameters are as defined above

### Minimum Detectable Effect Calculation

The minimum detectable difference in proportions for a given number of clusters and cluster size is:

$$|p_1 - p_2| = \frac{\sqrt{[p_1(1-p_1) + p_2(1-p_2)] \times DE} \times (z_{1-\alpha/2} + z_{1-\beta})}{\sqrt{k \times m}}$$

This is typically solved iteratively because $p_2$ appears on both sides of the equation.

## Analysis Methods

DesignPower provides sample size and power calculations for several analysis methods for binary outcomes in cluster randomized trials:

### Cluster-Level Analysis

In cluster-level analysis, summary statistics (e.g., proportions) are calculated for each cluster, and the analysis is performed on these summary measures. For binary outcomes, this might involve:
- Comparing cluster-level proportions using a t-test
- Weighting cluster-level proportions by cluster size

### Generalized Linear Mixed Models (GLMM)

GLMMs extend linear mixed models to binary outcomes using a link function (typically logit):

$$logit(P(Y_{ij} = 1)) = \beta_0 + \beta_1X_j + u_j$$

Where:
- $Y_{ij}$ = binary outcome for individual $i$ in cluster $j$
- $X_j$ = treatment indicator for cluster $j$
- $\beta_0$ = intercept
- $\beta_1$ = treatment effect (log odds ratio)
- $u_j$ = random effect for cluster $j$

### Generalized Estimating Equations (GEE)

GEE is an alternative approach that focuses on estimating population-averaged effects while accounting for the correlation structure within clusters.

## Simulation Methods

DesignPower implements simulation-based approaches for cluster randomized trials with binary outcomes.

### Simulation Algorithm

1. For each simulated trial:
   - Generate cluster-level random effects
   - Generate individual-level binary outcomes with the appropriate correlation structure
   - Perform the selected analysis method (cluster-level, GLMM, or GEE)
   - Record whether the null hypothesis was rejected

2. Sample size calculation:
   - Incrementally increase the number of clusters until the desired power is achieved
   - For each number of clusters, run multiple simulations and calculate the proportion of rejections

3. Power calculation:
   - For a fixed number of clusters and cluster size, run multiple simulations
   - Calculate the proportion of simulations that reject the null hypothesis

4. MDE calculation:
   - Using binary search, find the smallest effect size that achieves the desired power
   - For each effect size, run multiple simulations

## Practical Considerations

### ICC Estimation for Binary Outcomes

Estimating the ICC for binary outcomes is more challenging than for continuous outcomes. DesignPower provides guidance on:
- Using published ICCs from similar studies
- Converting ICCs from different scales (e.g., from logit to linear scale)
- Performing sensitivity analyses across a range of plausible ICC values

### Effect Size Specification

For binary outcomes, the effect size can be specified in various ways:
- Difference in proportions (risk difference)
- Risk ratio
- Odds ratio

DesignPower allows users to specify the effect size using any of these measures and converts appropriately for calculations.

### Small Number of Clusters

When the number of clusters is small, standard asymptotic methods may not be valid. DesignPower provides:
- Warnings when the calculated number of clusters falls below recommended thresholds
- Adjusted methods for small numbers of clusters
- Recommendations for alternative analysis approaches

## References

1. Donner A, Klar N. Design and Analysis of Cluster Randomization Trials in Health Research. Arnold; 2000.

2. Hayes RJ, Moulton LH. Cluster Randomised Trials, Second Edition. Chapman and Hall/CRC; 2017.

3. Eldridge SM, Ukoumunne OC, Carlin JB. The intra-cluster correlation coefficient in cluster randomized trials: A review of definitions. Int Stat Rev. 2009;77(3):378-394.

4. Gao F, Earnest A, Matchar DB, Campbell MJ, Machin D. Sample size calculations for the design of cluster randomized trials: A summary of methodology. Contemp Clin Trials. 2015;42:41-50.
