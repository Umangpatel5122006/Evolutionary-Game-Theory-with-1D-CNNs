# Experiment EXP-FC-001: Original 1D-CNN Forecast (Historical)

- **Component**: Solar Forecasting
- **Category**: Historical Baseline
- **Status**: Historical Record (Deprecated)

## Description
Initial 1D-CNN point forecast using single seed (42) and sliding window evaluation with test-set checkpoint selection.

## Key Empirical Metrics
```json
{
  "Test_MAE_kW": 1813.4,
  "Test_nMAE_pct": 6.22,
  "Test_RMSE_kW": 3450.1
}
```

## Methodological Notes
Had test-set leakage in checkpoint selection and lacked true time-of-day diurnal profile baseline.

## Associated Result Files
- `results/original/tables/table_1_forecast_metrics.csv`
- `results/original/figures/figure_1_actual_vs_predicted.png`
