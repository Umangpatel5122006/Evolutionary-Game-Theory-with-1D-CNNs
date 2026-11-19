# Complete Research Repository Inventory & Asset Manifest

**Repository**: `Evolutionary Game Theory with 1D-CNNs — Forecast-Aware Evolutionary Game Theory for Adaptive Energy Sharing in Renewable Energy Communities`  
**Archival Date**: 2026-11-01  
**Git Baseline Commit**: `7a0588cbb50c7faef8379e0ec78eefed5e17f756` (Author: Umang Patel, Oct 10 2026)  
**Total Tracked Files**: 228  

---

## 1. Executive Summary of Archival & Inventory

In accordance with strict research preservation principles, all original raw datasets, baseline model weights, initial experimental outputs, and preliminary documents have been inventoried, SHA-256 hashed, and frozen. An immutable snapshot of the baseline commit is stored in `research_archive/original_snapshot/`.

| Category | File Count | Total Size (MB) | Integrity Policy |
| :--- | :--- | :--- | :--- |
| **Project Infrastructure** | 3 | 0.01 MB | Cryptographically hashed with SHA-256 |
| **Reproducible Configuration** | 4 | 0.00 MB | Cryptographically hashed with SHA-256 |
| **Script / Pipeline Driver** | 4 | 0.10 MB | Cryptographically hashed with SHA-256 |
| **Model Weights** | 1 | 0.35 MB | Cryptographically hashed with SHA-256 |
| **Processed Data** | 2 | 1.16 MB | Cryptographically hashed with SHA-256 |
| **Historical Experiment Output** | 36 | 17.07 MB | Cryptographically hashed with SHA-256 |
| **Raw Dataset** | 8 | 21.57 MB | Cryptographically hashed with SHA-256 |
| **Documentation / Research Paper Source** | 6 | 0.15 MB | Cryptographically hashed with SHA-256 |
| **Experiment Registry & Execution Records** | 39 | 0.04 MB | Cryptographically hashed with SHA-256 |
| **Notebook** | 1 | 0.03 MB | Cryptographically hashed with SHA-256 |
| **Validated Corrected Output** | 21 | 2.27 MB | Cryptographically hashed with SHA-256 |
| **Academic Paper / Template** | 9 | 2.42 MB | Cryptographically hashed with SHA-256 |
| **Academic Report / Deliverable** | 22 | 3.64 MB | Cryptographically hashed with SHA-256 |
| **Research Archive** | 2 | 0.05 MB | Cryptographically hashed with SHA-256 |
| **Snapshot / Immutable Archive** | 42 | 20.88 MB | Cryptographically hashed with SHA-256 |
| **Generated Figure** | 6 | 1.29 MB | Cryptographically hashed with SHA-256 |
| **Generated Table** | 14 | 0.63 MB | Cryptographically hashed with SHA-256 |
| **Source Code** | 7 | 0.06 MB | Cryptographically hashed with SHA-256 |
| **Test Suite** | 1 | 0.01 MB | Cryptographically hashed with SHA-256 |

---

## 2. Granular Inventory by Functional Component

### 2.1 Project Infrastructure

| Relative Path | Size | SHA-256 Digest | Status & Description |
| :--- | :--- | :--- | :--- |
| `.gitignore` | 0 B | `e3b0c44298fc...` | Repository configuration |
| `data/README.md` | 1989 B | `768d557175dd...` | Repository configuration |
| `README.md` | 10765 B | `408a8ddeb600...` | Repository configuration |

### 2.2 Reproducible Configuration

| Relative Path | Size | SHA-256 Digest | Status & Description |
| :--- | :--- | :--- | :--- |
| `configs/data_config.yaml` | 986 B | `2d90b006fb68...` | System hyperparameter & pipeline configs |
| `configs/game_config.yaml` | 1246 B | `a8aa22391326...` | System hyperparameter & pipeline configs |
| `configs/model_config.yaml` | 705 B | `381749ff7104...` | System hyperparameter & pipeline configs |
| `configs/uncertainty_config.yaml` | 510 B | `02519be917c8...` | System hyperparameter & pipeline configs |

### 2.3 Script / Pipeline Driver

| Relative Path | Size | SHA-256 Digest | Status & Description |
| :--- | :--- | :--- | :--- |
| `create_ieee_artifacts.py` | 51856 B | `082612d82305...` | Execution orchestration |
| `create_notebook.py` | 23743 B | `e79c203c9ca9...` | Execution orchestration |
| `refresh_manifest.py` | 6962 B | `58de98d8c31c...` | Execution orchestration |
| `run_pipeline.py` | 20918 B | `82ad0445662d...` | Execution orchestration |

### 2.4 Model Weights

| Relative Path | Size | SHA-256 Digest | Status & Description |
| :--- | :--- | :--- | :--- |
| `data/processed/cnn_model_weights.pt` | 0.354 MB | `b5abaaf218b1...` | Trained neural network checkpoint |

### 2.5 Processed Data

| Relative Path | Size | SHA-256 Digest | Status & Description |
| :--- | :--- | :--- | :--- |
| `data/processed/plant_1_cleaned.csv` | 0.562 MB | `f24b3d38b1ed...` | Standardized continuous time-series arrays |
| `data/processed/plant_2_cleaned.csv` | 0.602 MB | `ddf66b5609a9...` | Standardized continuous time-series arrays |

### 2.6 Historical Experiment Output

| Relative Path | Size | SHA-256 Digest | Status & Description |
| :--- | :--- | :--- | :--- |
| `data/processed_original/cnn_model_weights.pt` | 6.398 MB | `97dacc3e090f...` | Original run experimental records |
| `data/processed_original/plant_1_cleaned.csv` | 0.559 MB | `d1b04b35a8f2...` | Original run experimental records |
| `data/processed_original/plant_2_cleaned.csv` | 0.602 MB | `ddf66b5609a9...` | Original run experimental records |
| `models/original/cnn_model_weights_original.pt` | 6.398 MB | `97dacc3e090f...` | Original run experimental records |
| `results/original/figures/figure_1_actual_vs_predicted.png` | 0.305 MB | `1b5aa7c1546b...` | Original run experimental records |
| `results/original/figures/figure_2_horizon_error.png` | 0.108 MB | `ca2f2d23e0f5...` | Original run experimental records |
| `results/original/figures/figure_3_uncertainty_calibration.png` | 0.152 MB | `5dc246914848...` | Original run experimental records |
| `results/original/figures/figure_4_comparative_trajectories.png` | 0.195 MB | `d65959914bb5...` | Original run experimental records |
| `results/original/figures/figure_5_tipping_point.png` | 0.22 MB | `b1ce490062f2...` | Original run experimental records |
| `results/original/figures/figure_6_weather_shock.png` | 0.291 MB | `86d7476651a1...` | Original run experimental records |
| `results/original/tables/simulation_M0.csv` | 0.066 MB | `f73afc001116...` | Original run experimental records |
| `results/original/tables/simulation_M1.csv` | 0.078 MB | `3f1916be512b...` | Original run experimental records |
| `results/original/tables/simulation_M2.csv` | 0.068 MB | `881c0b10875b...` | Original run experimental records |
| `results/original/tables/simulation_M3.csv` | 0.074 MB | `6fbed12aae53...` | Original run experimental records |
| `results/original/tables/table_1_forecast_metrics.csv` | 351 B | `69a7bb050e7c...` | Original run experimental records |
| `results/original/tables/table_2_uncertainty_calibration.csv` | 157 B | `4482031dc161...` | Original run experimental records |
| `results/original/tables/table_3_model_comparison.csv` | 713 B | `e36ed42ae9fb...` | Original run experimental records |
| `results/original/tables/table_4_tipping_point_sweep.csv` | 1224 B | `02560b04d0ad...` | Original run experimental records |
| `results/original/tables/table_5_weather_shock_metrics.csv` | 232 B | `03d49f19ab87...` | Original run experimental records |
| `results/original/tables/table_6_plant2_generalization.csv` | 215 B | `45b90690ad3c...` | Original run experimental records |
| `results_original/figures/figure_1_actual_vs_predicted.png` | 0.305 MB | `1b5aa7c1546b...` | Original run experimental records |
| `results_original/figures/figure_2_horizon_error.png` | 0.108 MB | `ca2f2d23e0f5...` | Original run experimental records |
| `results_original/figures/figure_3_uncertainty_calibration.png` | 0.152 MB | `5dc246914848...` | Original run experimental records |
| `results_original/figures/figure_4_comparative_trajectories.png` | 0.195 MB | `d65959914bb5...` | Original run experimental records |
| `results_original/figures/figure_5_tipping_point.png` | 0.22 MB | `b1ce490062f2...` | Original run experimental records |
| `results_original/figures/figure_6_weather_shock.png` | 0.291 MB | `86d7476651a1...` | Original run experimental records |
| `results_original/tables/simulation_M0.csv` | 0.066 MB | `f73afc001116...` | Original run experimental records |
| `results_original/tables/simulation_M1.csv` | 0.078 MB | `3f1916be512b...` | Original run experimental records |
| `results_original/tables/simulation_M2.csv` | 0.068 MB | `881c0b10875b...` | Original run experimental records |
| `results_original/tables/simulation_M3.csv` | 0.074 MB | `6fbed12aae53...` | Original run experimental records |
| `results_original/tables/table_1_forecast_metrics.csv` | 351 B | `69a7bb050e7c...` | Original run experimental records |
| `results_original/tables/table_2_uncertainty_calibration.csv` | 157 B | `4482031dc161...` | Original run experimental records |
| `results_original/tables/table_3_model_comparison.csv` | 713 B | `e36ed42ae9fb...` | Original run experimental records |
| `results_original/tables/table_4_tipping_point_sweep.csv` | 1224 B | `02560b04d0ad...` | Original run experimental records |
| `results_original/tables/table_5_weather_shock_metrics.csv` | 232 B | `03d49f19ab87...` | Original run experimental records |
| `results_original/tables/table_6_plant2_generalization.csv` | 215 B | `45b90690ad3c...` | Original run experimental records |

### 2.7 Raw Dataset

| Relative Path | Size | SHA-256 Digest | Status & Description |
| :--- | :--- | :--- | :--- |
| `data/raw/Plant_1_Generation_Data.csv` | 4.615 MB | `882f75f1c295...` | Original unaltered solar and weather sensor data |
| `data/raw/Plant_1_Weather_Sensor_Data.csv` | 0.278 MB | `5a16949aff67...` | Original unaltered solar and weather sensor data |
| `data/raw/Plant_2_Generation_Data.csv` | 5.601 MB | `6b635a8ce399...` | Original unaltered solar and weather sensor data |
| `data/raw/Plant_2_Weather_Sensor_Data.csv` | 0.291 MB | `b7e40abc6b0a...` | Original unaltered solar and weather sensor data |
| `Plant_1_Generation_Data.csv` | 4.615 MB | `882f75f1c295...` | Kaggle Solar Power Generation and Weather dataset |
| `Plant_1_Weather_Sensor_Data.csv` | 0.278 MB | `5a16949aff67...` | Kaggle Solar Power Generation and Weather dataset |
| `Plant_2_Generation_Data.csv` | 5.601 MB | `6b635a8ce399...` | Kaggle Solar Power Generation and Weather dataset |
| `Plant_2_Weather_Sensor_Data.csv` | 0.291 MB | `b7e40abc6b0a...` | Kaggle Solar Power Generation and Weather dataset |

### 2.8 Documentation / Research Paper Source

| Relative Path | Size | SHA-256 Digest | Status & Description |
| :--- | :--- | :--- | :--- |
| `doc1_extracted.txt` | 9140 B | `5671d93411ab...` | Foundational research notes and papers |
| `doc2_extracted.txt` | 11359 B | `1956ff6c2188...` | Foundational research notes and papers |
| `Forecast_Aware_Evolutionary_Game_Novelty_Full_Chat_Version.docx` | 41247 B | `5320639c3c07...` | Foundational research notes and papers |
| `Implementation_Plan_1D_CNN_Evolutionary_Game_Renewable_Energy.pdf` | 0.055 MB | `5bfcb105974d...` | Foundational research notes and papers |
| `Papers_Combined_for_Forecast_Aware_Evolutionary_Game.docx` | 41721 B | `6f4669c7c3f5...` | Foundational research notes and papers |
| `requirements.txt` | 266 B | `97840915aeab...` | Foundational research notes and papers |

### 2.9 Experiment Registry & Execution Records

| Relative Path | Size | SHA-256 Digest | Status & Description |
| :--- | :--- | :--- | :--- |
| `experiments/00_experiment_registry/EXPERIMENT_INDEX.md` | 2416 B | `73fa3e2bc4f8...` | Registered experiment metadata & metrics |
| `experiments/00_experiment_registry/experiment_registry.csv` | 4044 B | `7bf81e1b6613...` | Registered experiment metadata & metrics |
| `experiments/00_experiment_registry/experiment_registry.json` | 9599 B | `42b86f59e806...` | Registered experiment metadata & metrics |
| `experiments/EXP-FC-001_original_1d_cnn_forecast_historical/metadata.json` | 706 B | `47c6cfb29786...` | Registered experiment metadata & metrics |
| `experiments/EXP-FC-001_original_1d_cnn_forecast_historical/metrics.json` | 82 B | `da5b8680b79b...` | Registered experiment metadata & metrics |
| `experiments/EXP-FC-001_original_1d_cnn_forecast_historical/README.md` | 727 B | `9617bd0ced15...` | Registered experiment metadata & metrics |
| `experiments/EXP-FC-002_corrected_diurnal_residual_tcn_forecast/metadata.json` | 807 B | `c5ebc286f83b...` | Registered experiment metadata & metrics |
| `experiments/EXP-FC-002_corrected_diurnal_residual_tcn_forecast/metrics.json` | 119 B | `474f21dfcb7e...` | Registered experiment metadata & metrics |
| `experiments/EXP-FC-002_corrected_diurnal_residual_tcn_forecast/README.md` | 823 B | `136798c56f50...` | Registered experiment metadata & metrics |
| `experiments/EXP-FC-003_multi_horizon_baselines_benchmark/metadata.json` | 773 B | `3d3c3e1434b6...` | Registered experiment metadata & metrics |
| `experiments/EXP-FC-003_multi_horizon_baselines_benchmark/metrics.json` | 207 B | `2798f3b997fd...` | Registered experiment metadata & metrics |
| `experiments/EXP-FC-003_multi_horizon_baselines_benchmark/README.md` | 793 B | `664890508250...` | Registered experiment metadata & metrics |
| `experiments/EXP-GT-001_original_4_mechanism_replicator_dynamics_historical/metadata.json` | 731 B | `3095c65f596e...` | Registered experiment metadata & metrics |
| `experiments/EXP-GT-001_original_4_mechanism_replicator_dynamics_historical/metrics.json` | 115 B | `e6807b9ae85c...` | Registered experiment metadata & metrics |
| `experiments/EXP-GT-001_original_4_mechanism_replicator_dynamics_historical/README.md` | 750 B | `04985ee242c8...` | Registered experiment metadata & metrics |
| `experiments/EXP-GT-002_corrected_4_mechanism_controlled_game_ablation/metadata.json` | 945 B | `6398a9525f5b...` | Registered experiment metadata & metrics |
| `experiments/EXP-GT-002_corrected_4_mechanism_controlled_game_ablation/metrics.json` | 231 B | `5add4f94ca11...` | Registered experiment metadata & metrics |
| `experiments/EXP-GT-002_corrected_4_mechanism_controlled_game_ablation/README.md` | 958 B | `c89d1d054672...` | Registered experiment metadata & metrics |
| `experiments/EXP-SEN-001_uncertainty_penalty_&_tipping_point_bifurcation_sweep/metadata.json` | 838 B | `54e90634aa0e...` | Registered experiment metadata & metrics |
| `experiments/EXP-SEN-001_uncertainty_penalty_&_tipping_point_bifurcation_sweep/metrics.json` | 153 B | `df4514cf1a46...` | Registered experiment metadata & metrics |
| `experiments/EXP-SEN-001_uncertainty_penalty_&_tipping_point_bifurcation_sweep/README.md` | 857 B | `a29a2620d480...` | Registered experiment metadata & metrics |
| `experiments/EXP-TR-001_original_plant_2_cross_plant_transfer_historical/metadata.json` | 572 B | `064b0c41d28c...` | Registered experiment metadata & metrics |
| `experiments/EXP-TR-001_original_plant_2_cross_plant_transfer_historical/metrics.json` | 32 B | `9728aeec3dec...` | Registered experiment metadata & metrics |
| `experiments/EXP-TR-001_original_plant_2_cross_plant_transfer_historical/README.md` | 600 B | `1d7f521b503e...` | Registered experiment metadata & metrics |
| `experiments/EXP-TR-002_genuine_zero_shot_cross_plant_generalization_plant_2/metadata.json` | 842 B | `a54cfa621e49...` | Registered experiment metadata & metrics |
| `experiments/EXP-TR-002_genuine_zero_shot_cross_plant_generalization_plant_2/metrics.json` | 254 B | `4d2655d21354...` | Registered experiment metadata & metrics |
| `experiments/EXP-TR-002_genuine_zero_shot_cross_plant_generalization_plant_2/README.md` | 860 B | `51fcac5b36c6...` | Registered experiment metadata & metrics |
| `experiments/EXP-UNC-001_original_conformal_uncertainty_quantification_historical/metadata.json` | 642 B | `2ccecff1a8f8...` | Registered experiment metadata & metrics |
| `experiments/EXP-UNC-001_original_conformal_uncertainty_quantification_historical/metrics.json` | 84 B | `f103fcb44c3f...` | Registered experiment metadata & metrics |
| `experiments/EXP-UNC-001_original_conformal_uncertainty_quantification_historical/README.md` | 663 B | `f84f81b804c6...` | Registered experiment metadata & metrics |
| `experiments/EXP-UNC-002_corrected_split_conformal_uq_&_calibration/metadata.json` | 794 B | `8c278a52586b...` | Registered experiment metadata & metrics |
| `experiments/EXP-UNC-002_corrected_split_conformal_uq_&_calibration/metrics.json` | 171 B | `d31ae5704437...` | Registered experiment metadata & metrics |
| `experiments/EXP-UNC-002_corrected_split_conformal_uq_&_calibration/README.md` | 811 B | `1ebd8f6111ca...` | Registered experiment metadata & metrics |
| `experiments/EXP-WS-001_original_weather_shock_experiment_historical/metadata.json` | 623 B | `814232fe2252...` | Registered experiment metadata & metrics |
| `experiments/EXP-WS-001_original_weather_shock_experiment_historical/metrics.json` | 73 B | `cb6235219bf4...` | Registered experiment metadata & metrics |
| `experiments/EXP-WS-001_original_weather_shock_experiment_historical/README.md` | 646 B | `149f3291ea6b...` | Registered experiment metadata & metrics |
| `experiments/EXP-WS-002_corrected_peak_aligned_severe_weather_shock_experiment/metadata.json` | 829 B | `8ece718daa6c...` | Registered experiment metadata & metrics |
| `experiments/EXP-WS-002_corrected_peak_aligned_severe_weather_shock_experiment/metrics.json` | 218 B | `75cdf980d9e4...` | Registered experiment metadata & metrics |
| `experiments/EXP-WS-002_corrected_peak_aligned_severe_weather_shock_experiment/README.md` | 842 B | `687da5f69df1...` | Registered experiment metadata & metrics |

### 2.10 Notebook

| Relative Path | Size | SHA-256 Digest | Status & Description |
| :--- | :--- | :--- | :--- |
| `Forecast_Aware_Evolutionary_Game_Theory.ipynb` | 27944 B | `9fad1a50b371...` | Interactive presentation of methodology |

### 2.11 Validated Corrected Output

| Relative Path | Size | SHA-256 Digest | Status & Description |
| :--- | :--- | :--- | :--- |
| `models/corrected/cnn_model_weights_corrected.pt` | 0.354 MB | `b5abaaf218b1...` | Validated experimental records |
| `results/corrected/figures/figure_1_actual_vs_predicted.png` | 0.323 MB | `f264cdf34473...` | Validated experimental records |
| `results/corrected/figures/figure_2_horizon_error.png` | 0.099 MB | `c9f992f54051...` | Validated experimental records |
| `results/corrected/figures/figure_3_uncertainty_calibration.png` | 0.157 MB | `b876d7898e3a...` | Validated experimental records |
| `results/corrected/figures/figure_4_comparative_trajectories.png` | 0.174 MB | `2ec0188d3ff4...` | Validated experimental records |
| `results/corrected/figures/figure_5_tipping_point.png` | 0.258 MB | `e27c8f4c35d9...` | Validated experimental records |
| `results/corrected/figures/figure_6_weather_shock.png` | 0.278 MB | `0cfb964640dc...` | Validated experimental records |
| `results/corrected/tables/simulation_B0.csv` | 0.083 MB | `c319ffc89ce3...` | Validated experimental records |
| `results/corrected/tables/simulation_B1.csv` | 0.083 MB | `dff154da4607...` | Validated experimental records |
| `results/corrected/tables/simulation_B2.csv` | 0.083 MB | `4705afdbebf4...` | Validated experimental records |
| `results/corrected/tables/simulation_B3.csv` | 0.085 MB | `a2b11ce0a801...` | Validated experimental records |
| `results/corrected/tables/simulation_M0.csv` | 0.066 MB | `f73afc001116...` | Validated experimental records |
| `results/corrected/tables/simulation_M1.csv` | 0.078 MB | `3f1916be512b...` | Validated experimental records |
| `results/corrected/tables/simulation_M2.csv` | 0.068 MB | `881c0b10875b...` | Validated experimental records |
| `results/corrected/tables/simulation_M3.csv` | 0.074 MB | `6fbed12aae53...` | Validated experimental records |
| `results/corrected/tables/table_1_forecast_metrics.csv` | 952 B | `62175b8a5661...` | Validated experimental records |
| `results/corrected/tables/table_2_uncertainty_calibration.csv` | 372 B | `d68195cfd514...` | Validated experimental records |
| `results/corrected/tables/table_3_model_comparison.csv` | 1188 B | `f3a76f34580b...` | Validated experimental records |
| `results/corrected/tables/table_4_tipping_point_sweep.csv` | 4445 B | `f5520376123b...` | Validated experimental records |
| `results/corrected/tables/table_5_weather_shock_metrics.csv` | 274 B | `b046c124b057...` | Validated experimental records |
| `results/corrected/tables/table_6_plant2_generalization.csv` | 519 B | `bf3efcf08db3...` | Validated experimental records |

### 2.12 Academic Paper / Template

| Relative Path | Size | SHA-256 Digest | Status & Description |
| :--- | :--- | :--- | :--- |
| `paper/docx/IEEE_Project_Report.docx` | 1.118 MB | `1da7eecfab08...` | IEEE Conference paper (Word & LaTeX) |
| `paper/latex/figures/figure_1_actual_vs_predicted.png` | 0.323 MB | `f264cdf34473...` | IEEE Conference paper (Word & LaTeX) |
| `paper/latex/figures/figure_2_horizon_error.png` | 0.099 MB | `c9f992f54051...` | IEEE Conference paper (Word & LaTeX) |
| `paper/latex/figures/figure_3_uncertainty_calibration.png` | 0.157 MB | `b876d7898e3a...` | IEEE Conference paper (Word & LaTeX) |
| `paper/latex/figures/figure_4_comparative_trajectories.png` | 0.174 MB | `2ec0188d3ff4...` | IEEE Conference paper (Word & LaTeX) |
| `paper/latex/figures/figure_5_tipping_point.png` | 0.258 MB | `e27c8f4c35d9...` | IEEE Conference paper (Word & LaTeX) |
| `paper/latex/figures/figure_6_weather_shock.png` | 0.278 MB | `0cfb964640dc...` | IEEE Conference paper (Word & LaTeX) |
| `paper/latex/main.tex` | 11173 B | `1937c80149f8...` | IEEE Conference paper (Word & LaTeX) |
| `paper/latex/references.bib` | 1931 B | `fb6d1f72a386...` | IEEE Conference paper (Word & LaTeX) |

### 2.13 Academic Report / Deliverable

| Relative Path | Size | SHA-256 Digest | Status & Description |
| :--- | :--- | :--- | :--- |
| `reports/CLAIMS_AND_EVIDENCE.md` | 3598 B | `a5e24e5673c0...` | Scientific reports, audit, and documentation |
| `reports/DATA_DICTIONARY.md` | 4339 B | `fc7c893962f9...` | Scientific reports, audit, and documentation |
| `reports/EXPERIMENT_INDEX.md` | 2416 B | `73fa3e2bc4f8...` | Scientific reports, audit, and documentation |
| `reports/EXPERIMENT_PROTOCOL.md` | 8446 B | `764b8fe1b05f...` | Scientific reports, audit, and documentation |
| `reports/FINAL_RESULTS.md` | 5225 B | `a98a3ad3ad03...` | Scientific reports, audit, and documentation |
| `reports/IEEE_Project_Report.docx` | 1.118 MB | `1da7eecfab08...` | Scientific reports, audit, and documentation |
| `reports/IMPROVEMENTS_IMPLEMENTED.md` | 9202 B | `b0657aee36b3...` | Scientific reports, audit, and documentation |
| `reports/LIMITATIONS.md` | 4406 B | `96d55bd57b5a...` | Scientific reports, audit, and documentation |
| `reports/overleaf_package/figures/figure_1_actual_vs_predicted.png` | 0.323 MB | `f264cdf34473...` | Scientific reports, audit, and documentation |
| `reports/overleaf_package/figures/figure_2_horizon_error.png` | 0.099 MB | `c9f992f54051...` | Scientific reports, audit, and documentation |
| `reports/overleaf_package/figures/figure_3_uncertainty_calibration.png` | 0.157 MB | `b876d7898e3a...` | Scientific reports, audit, and documentation |
| `reports/overleaf_package/figures/figure_4_comparative_trajectories.png` | 0.174 MB | `2ec0188d3ff4...` | Scientific reports, audit, and documentation |
| `reports/overleaf_package/figures/figure_5_tipping_point.png` | 0.258 MB | `e27c8f4c35d9...` | Scientific reports, audit, and documentation |
| `reports/overleaf_package/figures/figure_6_weather_shock.png` | 0.278 MB | `0cfb964640dc...` | Scientific reports, audit, and documentation |
| `reports/overleaf_package/main.tex` | 11173 B | `1937c80149f8...` | Scientific reports, audit, and documentation |
| `reports/overleaf_package/references.bib` | 1931 B | `fb6d1f72a386...` | Scientific reports, audit, and documentation |
| `reports/overleaf_package.zip` | 1.163 MB | `c4a1604f900d...` | Scientific reports, audit, and documentation |
| `reports/REPOSITORY_AUDIT.md` | 7645 B | `9a757aed0ab9...` | Scientific reports, audit, and documentation |
| `reports/REPRODUCIBILITY_GUIDE.md` | 3058 B | `6108ff4c5e6a...` | Scientific reports, audit, and documentation |
| `reports/RESEARCH_HISTORY.md` | 4276 B | `91ad74226abd...` | Scientific reports, audit, and documentation |
| `reports/RESULTS_EVOLUTION.md` | 4177 B | `42de64be5d49...` | Scientific reports, audit, and documentation |
| `reports/SUBMISSION_READINESS.md` | 2739 B | `b4fd435f28e6...` | Scientific reports, audit, and documentation |

### 2.14 Research Archive

| Relative Path | Size | SHA-256 Digest | Status & Description |
| :--- | :--- | :--- | :--- |
| `research_archive/file_manifest.csv` | 34852 B | `e0f1bc83245f...` | Repository management |
| `research_archive/INVENTORY.md` | 22155 B | `618d38a2b040...` | Repository management |

### 2.15 Snapshot / Immutable Archive

| Relative Path | Size | SHA-256 Digest | Status & Description |
| :--- | :--- | :--- | :--- |
| `research_archive/original_snapshot/.gitignore` | 0 B | `e3b0c44298fc...` | Preserved original commit state |
| `research_archive/original_snapshot/create_notebook.py` | 23743 B | `e79c203c9ca9...` | Preserved original commit state |
| `research_archive/original_snapshot/data/processed/cnn_model_weights.pt` | 6.398 MB | `97dacc3e090f...` | Preserved original commit state |
| `research_archive/original_snapshot/data/processed/plant_1_cleaned.csv` | 0.559 MB | `d1b04b35a8f2...` | Preserved original commit state |
| `research_archive/original_snapshot/data/processed/plant_2_cleaned.csv` | 0.602 MB | `ddf66b5609a9...` | Preserved original commit state |
| `research_archive/original_snapshot/doc1_extracted.txt` | 9140 B | `5671d93411ab...` | Preserved original commit state |
| `research_archive/original_snapshot/doc2_extracted.txt` | 11359 B | `1956ff6c2188...` | Preserved original commit state |
| `research_archive/original_snapshot/Forecast_Aware_Evolutionary_Game_Novelty_Full_Chat_Version.docx` | 41247 B | `5320639c3c07...` | Preserved original commit state |
| `research_archive/original_snapshot/Forecast_Aware_Evolutionary_Game_Theory.ipynb` | 0.732 MB | `d1fd905a26ce...` | Preserved original commit state |
| `research_archive/original_snapshot/Implementation_Plan_1D_CNN_Evolutionary_Game_Renewable_Energy.pdf` | 0.055 MB | `5bfcb105974d...` | Preserved original commit state |
| `research_archive/original_snapshot/Papers_Combined_for_Forecast_Aware_Evolutionary_Game.docx` | 41721 B | `6f4669c7c3f5...` | Preserved original commit state |
| `research_archive/original_snapshot/Plant_1_Generation_Data.csv` | 4.615 MB | `882f75f1c295...` | Preserved original commit state |
| `research_archive/original_snapshot/Plant_1_Weather_Sensor_Data.csv` | 0.278 MB | `5a16949aff67...` | Preserved original commit state |
| `research_archive/original_snapshot/Plant_2_Generation_Data.csv` | 5.601 MB | `6b635a8ce399...` | Preserved original commit state |
| `research_archive/original_snapshot/Plant_2_Weather_Sensor_Data.csv` | 0.291 MB | `b7e40abc6b0a...` | Preserved original commit state |
| `research_archive/original_snapshot/README.md` | 7879 B | `8e6ba42bcb3a...` | Preserved original commit state |
| `research_archive/original_snapshot/requirements.txt` | 266 B | `97840915aeab...` | Preserved original commit state |
| `research_archive/original_snapshot/results/figures/figure_1_actual_vs_predicted.png` | 0.305 MB | `1b5aa7c1546b...` | Preserved original commit state |
| `research_archive/original_snapshot/results/figures/figure_2_horizon_error.png` | 0.108 MB | `ca2f2d23e0f5...` | Preserved original commit state |
| `research_archive/original_snapshot/results/figures/figure_3_uncertainty_calibration.png` | 0.152 MB | `5dc246914848...` | Preserved original commit state |
| `research_archive/original_snapshot/results/figures/figure_4_comparative_trajectories.png` | 0.195 MB | `d65959914bb5...` | Preserved original commit state |
| `research_archive/original_snapshot/results/figures/figure_5_tipping_point.png` | 0.22 MB | `b1ce490062f2...` | Preserved original commit state |
| `research_archive/original_snapshot/results/figures/figure_6_weather_shock.png` | 0.291 MB | `86d7476651a1...` | Preserved original commit state |
| `research_archive/original_snapshot/results/tables/simulation_M0.csv` | 0.066 MB | `f73afc001116...` | Preserved original commit state |
| `research_archive/original_snapshot/results/tables/simulation_M1.csv` | 0.078 MB | `3f1916be512b...` | Preserved original commit state |
| `research_archive/original_snapshot/results/tables/simulation_M2.csv` | 0.068 MB | `881c0b10875b...` | Preserved original commit state |
| `research_archive/original_snapshot/results/tables/simulation_M3.csv` | 0.074 MB | `6fbed12aae53...` | Preserved original commit state |
| `research_archive/original_snapshot/results/tables/table_1_forecast_metrics.csv` | 351 B | `69a7bb050e7c...` | Preserved original commit state |
| `research_archive/original_snapshot/results/tables/table_2_uncertainty_calibration.csv` | 157 B | `4482031dc161...` | Preserved original commit state |
| `research_archive/original_snapshot/results/tables/table_3_model_comparison.csv` | 713 B | `e36ed42ae9fb...` | Preserved original commit state |
| `research_archive/original_snapshot/results/tables/table_4_tipping_point_sweep.csv` | 1224 B | `02560b04d0ad...` | Preserved original commit state |
| `research_archive/original_snapshot/results/tables/table_5_weather_shock_metrics.csv` | 232 B | `03d49f19ab87...` | Preserved original commit state |
| `research_archive/original_snapshot/results/tables/table_6_plant2_generalization.csv` | 215 B | `45b90690ad3c...` | Preserved original commit state |
| `research_archive/original_snapshot/run_pipeline.py` | 13648 B | `c8ac9aeeb416...` | Preserved original commit state |
| `research_archive/original_snapshot/snapshot_manifest.csv` | 6433 B | `450f90be2cd3...` | Preserved original commit state |
| `research_archive/original_snapshot/src/__init__.py` | 795 B | `a39ddae6d40a...` | Preserved original commit state |
| `research_archive/original_snapshot/src/experiments.py` | 13760 B | `326864af3aae...` | Preserved original commit state |
| `research_archive/original_snapshot/src/payoffs.py` | 6581 B | `93c68f542db8...` | Preserved original commit state |
| `research_archive/original_snapshot/src/preprocess.py` | 8023 B | `616aa75fb0a9...` | Preserved original commit state |
| `research_archive/original_snapshot/src/replicator.py` | 4884 B | `ff766fe87122...` | Preserved original commit state |
| `research_archive/original_snapshot/src/train_cnn.py` | 6600 B | `df542bf41691...` | Preserved original commit state |
| `research_archive/original_snapshot/src/uncertainty.py` | 4538 B | `aab49470820e...` | Preserved original commit state |

### 2.16 Generated Figure

| Relative Path | Size | SHA-256 Digest | Status & Description |
| :--- | :--- | :--- | :--- |
| `results/figures/figure_1_actual_vs_predicted.png` | 0.323 MB | `f264cdf34473...` | Publication plots |
| `results/figures/figure_2_horizon_error.png` | 0.099 MB | `c9f992f54051...` | Publication plots |
| `results/figures/figure_3_uncertainty_calibration.png` | 0.157 MB | `b876d7898e3a...` | Publication plots |
| `results/figures/figure_4_comparative_trajectories.png` | 0.174 MB | `2ec0188d3ff4...` | Publication plots |
| `results/figures/figure_5_tipping_point.png` | 0.258 MB | `e27c8f4c35d9...` | Publication plots |
| `results/figures/figure_6_weather_shock.png` | 0.278 MB | `0cfb964640dc...` | Publication plots |

### 2.17 Generated Table

| Relative Path | Size | SHA-256 Digest | Status & Description |
| :--- | :--- | :--- | :--- |
| `results/tables/simulation_B0.csv` | 0.083 MB | `c319ffc89ce3...` | Tabulated performance metrics |
| `results/tables/simulation_B1.csv` | 0.083 MB | `dff154da4607...` | Tabulated performance metrics |
| `results/tables/simulation_B2.csv` | 0.083 MB | `4705afdbebf4...` | Tabulated performance metrics |
| `results/tables/simulation_B3.csv` | 0.085 MB | `a2b11ce0a801...` | Tabulated performance metrics |
| `results/tables/simulation_M0.csv` | 0.066 MB | `f73afc001116...` | Tabulated performance metrics |
| `results/tables/simulation_M1.csv` | 0.078 MB | `3f1916be512b...` | Tabulated performance metrics |
| `results/tables/simulation_M2.csv` | 0.068 MB | `881c0b10875b...` | Tabulated performance metrics |
| `results/tables/simulation_M3.csv` | 0.074 MB | `6fbed12aae53...` | Tabulated performance metrics |
| `results/tables/table_1_forecast_metrics.csv` | 952 B | `62175b8a5661...` | Tabulated performance metrics |
| `results/tables/table_2_uncertainty_calibration.csv` | 372 B | `d68195cfd514...` | Tabulated performance metrics |
| `results/tables/table_3_model_comparison.csv` | 1188 B | `f3a76f34580b...` | Tabulated performance metrics |
| `results/tables/table_4_tipping_point_sweep.csv` | 4445 B | `f5520376123b...` | Tabulated performance metrics |
| `results/tables/table_5_weather_shock_metrics.csv` | 274 B | `b046c124b057...` | Tabulated performance metrics |
| `results/tables/table_6_plant2_generalization.csv` | 519 B | `bf3efcf08db3...` | Tabulated performance metrics |

### 2.18 Source Code

| Relative Path | Size | SHA-256 Digest | Status & Description |
| :--- | :--- | :--- | :--- |
| `src/__init__.py` | 795 B | `a39ddae6d40a...` | Pipeline components (preprocess, TCN, UQ, EGT) |
| `src/experiments.py` | 16081 B | `7d8ef614b370...` | Pipeline components (preprocess, TCN, UQ, EGT) |
| `src/payoffs.py` | 8890 B | `a17272e21abe...` | Pipeline components (preprocess, TCN, UQ, EGT) |
| `src/preprocess.py` | 9695 B | `007155eb8701...` | Pipeline components (preprocess, TCN, UQ, EGT) |
| `src/replicator.py` | 7128 B | `1f91065a761b...` | Pipeline components (preprocess, TCN, UQ, EGT) |
| `src/train_cnn.py` | 10264 B | `a6ef9165c181...` | Pipeline components (preprocess, TCN, UQ, EGT) |
| `src/uncertainty.py` | 8056 B | `ff6335df7dd6...` | Pipeline components (preprocess, TCN, UQ, EGT) |

### 2.19 Test Suite

| Relative Path | Size | SHA-256 Digest | Status & Description |
| :--- | :--- | :--- | :--- |
| `tests/test_pipeline.py` | 11927 B | `47132cec85ee...` | Automated regression and verification tests |
