# End-to-End Reproducibility Guide

**Repository:** `Evolutionary-Game-Theory-with-1D-CNNs`  
**Purpose:** Deterministic step-by-step reproduction of all experiments, metrics, figures, and tables from a clean environment.  
**Platform Tested:** Windows 11 / Linux Ubuntu 22.04 LTS (Python 3.10 - 3.14, PyTorch 2.0+)  

---

## 1. Environment Setup & Prerequisites

```bash
# Clone the repository
git clone https://github.com/Umangpatel5122006/Evolutionary-Game-Theory-with-1D-CNNs.git
cd Evolutionary-Game-Theory-with-1D-CNNs

# Create and activate a clean Python virtual environment
python -m venv venv
# On Windows PowerShell:
.\venv\Scripts\Activate.ps1
# On Linux / macOS:
source venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

---

## 2. Automated Test Suite Verification

Before running experiments, execute the 13-test regression suite to verify unit harmonization, zero leakage, model checkpointing, conformal coverage, and game dynamics:

```bash
python -m unittest tests/test_pipeline.py
```
*Expected output:* `Ran 13 tests in ~0.37s ... OK`

---

## 3. Running the Complete Research Pipeline

To reproduce all 12 experiments, train models across multiple seeds, calibrate conformal prediction intervals, execute evolutionary game ablations, and generate all tables and figures:

```bash
python run_pipeline.py
```

### Generated Artifacts
1. **Figures** (`results/figures/` and `results/corrected/figures/`):
   - `figure_1_actual_vs_predicted.png`: Actual vs TCN forecast with 90% split conformal prediction band.
   - `figure_2_horizon_error.png`: Horizon error degradation curve across 15-min to 24-hr lead times.
   - `figure_3_uncertainty_calibration.png`: Conformal calibration curve (empirical vs nominal coverage).
   - `figure_4_comparative_trajectories.png`: Evolutionary strategy trajectories for baselines B0–B3.
   - `figure_5_tipping_point.png`: Bifurcation curves and sensitivity across risk penalties.
   - `figure_6_weather_shock.png`: Community response and recovery under severe daytime weather shock.

2. **Tables** (`results/tables/` and `results/corrected/tables/`):
   - `table_1_forecast_metrics.csv`: Forecast comparison (Persistence, Seasonal Naive, Diurnal Profile, TCN).
   - `table_2_uncertainty_calibration.csv`: Split conformal calibration metrics (PICP, MPIW, Winkler).
   - `table_3_model_comparison.csv`: Evolutionary game model comparison and realized savings.
   - `table_4_tipping_point_sweep.csv`: Sensitivity sweep across uncertainty multipliers.
   - `table_5_weather_shock_metrics.csv`: Weather shock operational diagnostics and recovery metrics.
   - `table_6_plant2_generalization.csv`: Genuine zero-shot cross-plant evaluation on Plant 2.

---

## 4. Re-Generating Academic Deliverables

To re-generate the IEEE-formatted Word report (`reports/IEEE_Project_Report.docx`) and the Overleaf-ready LaTeX package (`reports/overleaf_package.zip`):

```bash
python create_ieee_artifacts.py
```
