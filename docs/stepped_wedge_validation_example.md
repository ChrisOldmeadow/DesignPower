# Stepped Wedge Methods Validation Example

This document demonstrates the validation and comparison between analytical (Hussey & Hughes) and simulation methods for stepped wedge cluster randomized trials.

## Test Case: Continuous Outcome

### Design Parameters
- **Clusters**: 12
- **Time Steps**: 4 (including baseline)
- **Individuals per Cluster**: 25
- **ICC**: 0.05
- **Cluster Autocorrelation (CAC)**: 0.0 (simplified for comparison)
- **Treatment Effect**: 0.5
- **Standard Deviation**: 2.0
- **Significance Level**: 0.05

### Method Comparison

```python
from core.designs.stepped_wedge.analytical import hussey_hughes_power_continuous
from core.designs.stepped_wedge.simulation import simulate_continuous

# Analytical Method
analytical_result = hussey_hughes_power_continuous(
    clusters=12, steps=4, individuals_per_cluster=25,
    icc=0.05, cluster_autocorr=0.0, treatment_effect=0.5,
    std_dev=2.0, alpha=0.05
)

# Simulation Method
simulation_result = simulate_continuous(
    clusters=12, steps=4, individuals_per_cluster=25,
    icc=0.05, treatment_effect=0.5, std_dev=2.0,
    nsim=5000, alpha=0.05
)

print(f"Analytical Power: {analytical_result['power']:.3f}")
print(f"Simulation Power: {simulation_result['power']:.3f}")
print(f"Difference: {abs(analytical_result['power'] - simulation_result['power']):.3f}")
```

Expected output shows close agreement between methods when cluster autocorrelation is zero.

## Key Validation Points

1. **Type I Error Rate**: When treatment effect = 0, power should approximate α
2. **Effect Size Sensitivity**: Larger treatment effects should yield higher power
3. **Sample Size Relationships**: More clusters or larger cluster sizes should increase power
4. **ICC Impact**: Higher ICC should generally reduce power (design effect)

## Recommendations

- Use **Analytical Methods** when CAC > 0 or for rapid calculations
- Use **Simulation Methods** for validation or non-standard designs
- Consider both methods for critical sample size calculations

This validation framework ensures the reliability of both methodological approaches implemented in DesignPower.