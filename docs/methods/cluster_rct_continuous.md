# Cluster RCT with Continuous Outcomes

This document details the statistical methodology implemented in DesignPower for calculating sample size, power, and minimum detectable effect for cluster randomized controlled trials (cRCTs) with continuous outcomes.

## Background

In cluster randomized trials, intact social units (clusters) such as clinics, schools, or communities are randomized to intervention or control conditions, but outcomes are measured on individuals within those clusters. This design introduces additional complexity due to the correlation among individuals within the same cluster.

## Statistical Framework

### Intracluster Correlation Coefficient

The key parameter that characterizes the correlation structure in cluster randomized trials is the intracluster correlation coefficient (ICC, denoted as $\rho$), which measures the degree of similarity among individuals within the same cluster.

The ICC is defined as:

$$\rho = \frac{\sigma_b^2}{\sigma_b^2 + \sigma_w^2}$$

Where:
- $\sigma_b^2$ = between-cluster variance
- $\sigma_w^2$ = within-cluster variance
- $\sigma_b^2 + \sigma_w^2$ = total variance

### Design Effect

The design effect (DE) quantifies how much the sample size needs to be inflated compared to an individually randomized trial:

$$DE = 1 + (m - 1) \times \rho$$

Where:
- $m$ = average cluster size
- $\rho$ = intracluster correlation coefficient

## Analytical Methods

### Sample Size Calculation

The sample size calculation for a cluster randomized trial with continuous outcomes builds on the formula for individually randomized trials:

$$n_{individual} = \frac{2\sigma^2(z_{1-\alpha/2} + z_{1-\beta})^2}{\Delta^2}$$

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

$$1-\beta = \Phi\left(\frac{\Delta\sqrt{k \times m}}{\sqrt{2\sigma^2 \times DE}} - z_{1-\alpha/2}\right)$$

Where:
- $\Phi$ = cumulative distribution function of the standard normal distribution
- $k$ = number of clusters per arm
- Other parameters are as defined above

### Minimum Detectable Effect Calculation

The minimum detectable effect for a given number of clusters and cluster size is:

$$\Delta = \frac{\sqrt{2\sigma^2 \times DE} \times (z_{1-\alpha/2} + z_{1-\beta})}{\sqrt{k \times m}}$$

## Analysis Methods

DesignPower provides sample size and power calculations for several analysis methods:

### Cluster-Level Analysis

In cluster-level analysis, summary statistics (e.g., means) are calculated for each cluster, and the analysis is performed on these summary measures. This approach is simple but may lose efficiency.

### Mixed-Effects Models

Mixed-effects models account for the hierarchical data structure by incorporating both fixed effects (treatment effect) and random effects (cluster effects):

$$Y_{ij} = \beta_0 + \beta_1X_j + u_j + \epsilon_{ij}$$

Where:
- $Y_{ij}$ = outcome for individual $i$ in cluster $j$
- $X_j$ = treatment indicator for cluster $j$
- $\beta_0$ = intercept
- $\beta_1$ = treatment effect
- $u_j$ = random effect for cluster $j$
- $\epsilon_{ij}$ = residual error for individual $i$ in cluster $j$

### Generalized Estimating Equations (GEE)

GEE is an alternative approach that focuses on estimating population-averaged effects while accounting for the correlation structure within clusters.

## Simulation Methods

DesignPower implements simulation-based approaches for cluster randomized trials, which are particularly useful when:
- Cluster sizes are highly variable
- Complex correlation structures are present
- Standard assumptions may not hold

### Simulation Algorithm

1. For each simulated trial:
   - Generate cluster-level random effects
   - Generate individual-level outcomes with the appropriate correlation structure
   - Perform the selected analysis method (cluster-level, mixed-effects, or GEE)
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

### ICC Estimation

The ICC is a critical parameter but is often difficult to estimate precisely. DesignPower provides guidance on:
- Using published ICCs from similar studies
- Performing sensitivity analyses across a range of plausible ICC values
- Accounting for uncertainty in ICC estimates

### Cluster Size Variability

When cluster sizes vary substantially, several approaches are available:
- Using the design effect adjustment for unequal cluster sizes
- Incorporating minimum and maximum cluster sizes in the simulation
- Weighting strategies in the analysis

### Minimum Number of Clusters

Statistical theory suggests that a minimum number of clusters (typically 40-60 total) is needed for robust inference, regardless of the cluster sizes. DesignPower provides warnings when the calculated number of clusters falls below recommended thresholds.

## References

1. Donner A, Klar N. Design and Analysis of Cluster Randomization Trials in Health Research. Arnold; 2000.

2. Hayes RJ, Moulton LH. Cluster Randomised Trials, Second Edition. Chapman and Hall/CRC; 2017.

3. Eldridge SM, Ukoumunne OC, Carlin JB. The intra-cluster correlation coefficient in cluster randomized trials: A review of definitions. Int Stat Rev. 2009;77(3):378-394.
