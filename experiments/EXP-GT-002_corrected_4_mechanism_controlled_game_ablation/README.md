# Experiment EXP-GT-002: Corrected 4-Mechanism Controlled Game Ablation

- **Component**: Evolutionary Game Theory
- **Category**: Corrected Primary
- **Status**: Validated

## Description
Controlled comparison across B0 (Persistence), B1 (Static Diurnal), B2 (CNN Risk-Neutral), and B3 (Forecast-Aware Conformal EGT) under identical load profiles and shortfall penalties.

## Key Empirical Metrics
```json
{
  "B0_Final_Share": 0.50476,
  "B1_Final_Share": 0.50587,
  "B2_Final_Share": 0.50521,
  "B3_Final_Share": 0.50474,
  "B1_Realized_Savings_USD": 5.67,
  "B2_Realized_Savings_USD": 3.62,
  "B3_Realized_Savings_USD": 3.62
}
```

## Methodological Notes
Demonstrates that risk-penalty reduces aggressive over-commitment, balancing traded kWh against shortfall exposure.

## Associated Result Files
- `results/corrected/tables/table_3_model_comparison.csv`
- `results/corrected/figures/figure_4_comparative_trajectories.png`
