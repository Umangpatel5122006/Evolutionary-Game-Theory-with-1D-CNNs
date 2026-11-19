# Experiment EXP-FC-003: Multi-Horizon Baselines Benchmark

- **Component**: Solar Forecasting
- **Category**: Benchmark
- **Status**: Validated

## Description
Rigorous comparison against Persistence (Lag-96), Seasonal Naive (Lag 96+192), and True Time-of-Day Diurnal Profile, across sliding and non-overlapping test sets.

## Key Empirical Metrics
```json
{
  "Persistence_MAE_kW": 1881.66,
  "Seasonal_Naive_MAE_kW": 1722.23,
  "Historical_Diurnal_MAE_kW": 1527.04,
  "TCN_Non_Overlapping_MAE_kW": 1955.15,
  "Persistence_Non_Overlapping_MAE_kW": 1955.43
}
```

## Methodological Notes
Historical diurnal profile indexes genuine time-of-day 15-min slot means from training data.

## Associated Result Files
- `results/corrected/tables/table_1_forecast_metrics.csv`
