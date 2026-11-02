# Experiment EXP-UNC-002: Corrected Split Conformal UQ & Calibration

- **Component**: Uncertainty Quantification
- **Category**: Corrected Primary
- **Status**: Validated

## Description
Studentized split conformal prediction providing both pooled marginal coverage and joint trajectory coverage with physical zeroing.

## Key Empirical Metrics
```json
{
  "Pooled_PICP_pct": 88.7,
  "Pooled_MPIW_kW": 6310.96,
  "Pooled_Winkler_Score": 6711.66,
  "Trajectory_Joint_PICP_pct": 67.85,
  "Conformal_Quantile_q": 35.618
}
```

## Methodological Notes
Guarantees finite-sample distribution-free coverage on unseen test windows.

## Associated Result Files
- `results/corrected/tables/table_2_uncertainty_calibration.csv`
- `results/corrected/figures/figure_3_uncertainty_calibration.png`
