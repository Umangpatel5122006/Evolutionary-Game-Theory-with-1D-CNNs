# Experiment EXP-FC-002: Corrected Diurnal Residual TCN Forecast

- **Component**: Solar Forecasting
- **Category**: Corrected Primary
- **Status**: Validated

## Description
Multi-scale residual TCN with strict validation-based checkpoint selection and multi-seed evaluation (42, 123, 456).

## Key Empirical Metrics
```json
{
  "Best_Test_MAE_kW": 1783.29,
  "Best_Test_nMAE_pct": 6.12,
  "Mean_Test_MAE_kW": 1815.39,
  "MAE_std": 33.86
}
```

## Methodological Notes
Selected best model by validation MAE (1661.33 kW). Physical non-negativity and astronomical night constraints enforced.

## Associated Result Files
- `results/corrected/tables/table_1_forecast_metrics.csv`
- `results/corrected/figures/figure_1_actual_vs_predicted.png`
- `results/corrected/figures/figure_2_horizon_error.png`
