# Experiment EXP-SEN-001: Uncertainty Penalty & Tipping Point Bifurcation Sweep

- **Component**: Evolutionary Game Theory
- **Category**: Sensitivity Analysis
- **Status**: Validated

## Description
Systematic parameter sweep over uncertainty penalty multipliers (0.0 to 15.0) and initial sharing conditions (0.20, 0.50, 0.80) to detect behavioral phase transitions.

## Key Empirical Metrics
```json
{
  "Max_Sensitivity_Uncertainty_pct": 3.48,
  "Peak_Gradient": -0.04,
  "Final_Share_Multiplier_0": 0.5052,
  "Final_Share_Multiplier_15": 0.4015
}
```

## Methodological Notes
Empirically disproves sharp catastrophic collapse; proves smooth monotonic decay under high risk penalty.

## Associated Result Files
- `results/corrected/tables/table_4_tipping_point_sweep.csv`
- `results/corrected/figures/figure_5_tipping_point.png`
