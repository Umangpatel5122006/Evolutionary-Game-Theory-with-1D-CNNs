# Experiment EXP-TR-002: Genuine Zero-Shot Cross-Plant Generalization (Plant 2)

- **Component**: Domain Generalization
- **Category**: Corrected Primary
- **Status**: Validated

## Description
Strict zero-shot transfer applying frozen Plant 1 feature and target scalers directly to unseen Plant 2 data without any target fine-tuning.

## Key Empirical Metrics
```json
{
  "Plant2_Persistence_MAE_kW": 1622.74,
  "Plant2_Persistence_nMAE_pct": 6.25,
  "Plant2_ZeroShot_MAE_kW": 1530.17,
  "Plant2_ZeroShot_nMAE_pct": 5.89,
  "Plant2_CapacityHarmonized_MAE_kW": 1598.69,
  "Plant2_CapacityHarmonized_nMAE_pct": 6.15
}
```

## Methodological Notes
Zero-shot model outperforms Plant 2 persistence baseline by 0.36% nMAE (92.57 kW MAE reduction).

## Associated Result Files
- `results/corrected/tables/table_6_plant2_generalization.csv`
