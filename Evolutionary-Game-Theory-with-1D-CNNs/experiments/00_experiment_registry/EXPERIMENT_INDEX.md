# Central Master Experiment Registry

This registry defines and tracks all experiments in the Forecast-Aware Evolutionary Game Theory repository.

| Experiment ID | Name | Category | Status | Primary Metric | Result Artifacts |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`EXP-FC-001`** | Original 1D-CNN Forecast (Historical) | Historical Baseline | `Historical Record (Deprecated)` | `Test_MAE_kW: 1813.4` | [Details](file:///EXP-FC-001) |
| **`EXP-FC-002`** | Corrected Diurnal Residual TCN Forecast | Corrected Primary | `Validated` | `Best_Test_MAE_kW: 1783.29` | [Details](file:///EXP-FC-002) |
| **`EXP-FC-003`** | Multi-Horizon Baselines Benchmark | Benchmark | `Validated` | `Persistence_MAE_kW: 1881.66` | [Details](file:///EXP-FC-003) |
| **`EXP-UNC-001`** | Original Conformal Uncertainty Quantification (Historical) | Historical Baseline | `Historical Record (Deprecated)` | `Empirical_Coverage_pct: 90.1` | [Details](file:///EXP-UNC-001) |
| **`EXP-UNC-002`** | Corrected Split Conformal UQ & Calibration | Corrected Primary | `Validated` | `Pooled_PICP_pct: 88.7` | [Details](file:///EXP-UNC-002) |
| **`EXP-GT-001`** | Original 4-Mechanism Replicator Dynamics (Historical) | Historical Baseline | `Historical Record (Deprecated)` | `M0_Final_Share: 0.505` | [Details](file:///EXP-GT-001) |
| **`EXP-GT-002`** | Corrected 4-Mechanism Controlled Game Ablation | Corrected Primary | `Validated` | `B0_Final_Share: 0.50476` | [Details](file:///EXP-GT-002) |
| **`EXP-SEN-001`** | Uncertainty Penalty & Tipping Point Bifurcation Sweep | Sensitivity Analysis | `Validated` | `Max_Sensitivity_Uncertainty_pct: 3.48` | [Details](file:///EXP-SEN-001) |
| **`EXP-WS-001`** | Original Weather Shock Experiment (Historical) | Historical Baseline | `Historical Record (Deprecated)` | `Reported_Collapse: Severe` | [Details](file:///EXP-WS-001) |
| **`EXP-WS-002`** | Corrected Peak-Aligned Severe Weather Shock Experiment | Corrected Primary | `Validated` | `Shock_Start_Interval: 150` | [Details](file:///EXP-WS-002) |
| **`EXP-TR-001`** | Original Plant 2 Cross-Plant Transfer (Historical) | Historical Baseline | `Historical Record (Deprecated)` | `Reported_nMAE_pct: 4.1` | [Details](file:///EXP-TR-001) |
| **`EXP-TR-002`** | Genuine Zero-Shot Cross-Plant Generalization (Plant 2) | Corrected Primary | `Validated` | `Plant2_Persistence_MAE_kW: 1622.74` | [Details](file:///EXP-TR-002) |