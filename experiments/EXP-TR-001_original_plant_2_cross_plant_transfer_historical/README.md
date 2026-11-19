# Experiment EXP-TR-001: Original Plant 2 Cross-Plant Transfer (Historical)

- **Component**: Domain Generalization
- **Category**: Historical Baseline
- **Status**: Historical Record (Deprecated)

## Description
Cross-plant evaluation where feature and target scalers were erroneously re-fitted on target Plant 2 data.

## Key Empirical Metrics
```json
{
  "Reported_nMAE_pct": 4.1
}
```

## Methodological Notes
Violated genuine zero-shot transfer protocol due to target-set re-scaling.

## Associated Result Files
- `results/original/tables/table_6_plant2_generalization.csv`
