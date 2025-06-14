# Statistical Methods Documentation

This directory contains detailed methodology documentation for the statistical methods implemented in DesignPower.

## Organization

The documentation is organized by study design and outcome type, matching the structure of the application:

1. **Parallel RCT**
   - [Continuous Outcomes](parallel_rct_continuous.md)
   - [Binary Outcomes](parallel_rct_binary.md)
   - [Survival Outcomes](parallel_rct_survival.md)

2. **Single-Arm Trials**
   - [Continuous Outcomes](single_arm_continuous.md)
   - [Binary Outcomes](single_arm_binary.md)
   - [A'Hern Design for Phase II Trials](ahern_design.md)
   - [Simon's Two-Stage Design for Phase II Trials](simons_two_stage_design.md)

3. **Cluster RCT**
   - [Continuous Outcomes](cluster_rct_continuous.md)
   - [Binary Outcomes](cluster_rct_binary.md)

4. **Stepped Wedge Cluster RCT**
   - [Stepped Wedge Methodology](stepped_wedge_methodology.md)

5. **Interrupted Time Series**
   - [Interrupted Time Series Methodology](interrupted_time_series_methodology.md)

6. **Special Topics**
   - [Repeated Measures Designs](repeated_measures.md)
   - [Simulation Methods](simulation_methods.md)
   - [Bayesian Inference Methods](bayesian_inference.md)

## Implementation Features

### Advanced Statistical Methods
- **Exact calculations** where analytically feasible
- **Simulation-based methods** for complex scenarios
- **Hybrid approaches** combining speed and accuracy
- **Iterative algorithms** for precision optimization

## Calculation Types

Each document describes the methodology for all three calculation types:
- Sample Size Calculation
- Power Calculation
- Minimum Detectable Effect (MDE) Calculation

## Hypothesis Testing Support

- **Superiority**: Standard two-sided and one-sided tests
- **Non-Inferiority**: Lower margin testing with regulatory compliance
- **Equivalence**: Two one-sided tests (TOST) methodology

## Contributing

To add or modify methodology documentation, please submit a pull request with your changes. All new methods must include complete mathematical specifications and literature references.

## Validation

For information about validation status, test results, and quality assurance, see the [Validation Report](../validation_report.md).
