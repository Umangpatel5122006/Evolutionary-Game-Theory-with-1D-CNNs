# Experiment EXP-WS-002: Corrected Peak-Aligned Severe Weather Shock Experiment

- **Component**: Resilience Testing
- **Category**: Corrected Primary
- **Status**: Validated

## Description
Daytime peak-aligned weather shock (interval 150 = 11:00 AM) with 75% irradiance loss over 6 hours.

## Key Empirical Metrics
```json
{
  "Shock_Start_Interval": 150,
  "Shock_Duration_Hours": 6.0,
  "Pre_Shock_Share": 0.87,
  "Min_Shock_Share": 0.87,
  "Post_Shock_Final_Share": 0.89,
  "Cost_Increase_USD": 124.48,
  "Recovered_95pct": true
}
```

## Methodological Notes
Quantifies exact financial impact ($124.48 cost increase) and confirms prosumer strategy resilience.

## Associated Result Files
- `results/corrected/tables/table_5_weather_shock_metrics.csv`
- `results/corrected/figures/figure_6_weather_shock.png`
