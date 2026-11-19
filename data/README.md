# Data Directory Organization & Provenance

This directory contains all data assets utilized in the **Forecast-Aware Evolutionary Game Theory** research project.

## Directory Structure

```text
data/
├── raw/                      # Original, unaltered Kaggle Solar Power Generation and Weather datasets
│   ├── Plant_1_Generation_Data.csv
│   ├── Plant_1_Weather_Sensor_Data.csv
│   ├── Plant_2_Generation_Data.csv
│   └── Plant_2_Weather_Sensor_Data.csv
├── processed/                # Standardized, continuous 15-minute aligned datasets
│   ├── original/             # Historical processed files from initial experiments
│   │   ├── plant_1_cleaned.csv
│   │   └── plant_2_cleaned.csv
│   ├── plant_1_cleaned.csv   # Unit-harmonized, zero-leakage cleaned Plant 1 data
│   ├── plant_2_cleaned.csv   # Unit-harmonized cleaned Plant 2 data
│   └── cnn_model_weights.pt  # Trained Diurnal Residual TCN PyTorch checkpoint
└── metadata/                 # Schema definitions, physical unit audits, and data dictionaries
    ├── data_dictionary.json
    └── sensor_specifications.md
```

## Physical Measurement Units & Sensor Calibration

### Plant 1 vs. Plant 2 DC Power Multiplier
- **Plant 1 Raw Data**: The `DC_POWER` column was logged in 10-Watt units, resulting in a median DC/AC ratio of $\approx 10.18$ relative to AC power in kW.
- **Plant 2 Raw Data**: The `DC_POWER` column was logged in true kW units, with a median DC/AC ratio of $\approx 1.02$.
- **Harmonization**: Plant 1 DC power is scaled by $0.10$ during preprocessing to align both plants to identical kW physical units.

### Astronomical Nighttime Constraint
During nighttime hours ($[19:00, 05:00)$ local time), solar irradiance is physically $0.00\text{ kW/m}^2$ and inverter output is $0.00\text{ kW}$. Strict zero-masking is enforced to prevent unphysical non-zero solar generation predictions at night.

