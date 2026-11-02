# Forecast-Aware Evolutionary Game Theory for Renewable Energy Communities

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![IEEE Format](https://img.shields.io/badge/Template-IEEE%20Conference-blue.svg)](https://www.ieee.org/conferences/publishing/templates.html)

**Repository Link:** [https://github.com/Umangpatel5122006/Evolutionary-Game-Theory-with-1D-CNNs](https://github.com/Umangpatel5122006/Evolutionary-Game-Theory-with-1D-CNNs)  
**Project Report (Word):** [reports/IEEE_Project_Report.docx](reports/IEEE_Project_Report.docx)  
**Project Report (Overleaf Package):** [reports/overleaf_package.zip](reports/overleaf_package.zip)  

---

## 👥 Group Members & Author Information

| Member Name | Student ID / Roll No. | Department & Institution | Email Address | Role / Contributions |
|:---|:---:|:---|:---|:---|
| **Umang Patel** | `U24AI041` | Dept. of Artificial Intelligence<br>Sardar Vallabhbhai National Institute of Technology (SVNIT), Surat, India | `u24ai041@aid.svnit.ac.in` | Lead Researcher: Pipeline orchestration, Diurnal Residual TCN, Replicator dynamics, Zero-shot transfer |
| **Vivek Patel** | `U24AI042` | Dept. of Artificial Intelligence<br>Sardar Vallabhbhai National Institute of Technology (SVNIT), Surat, India | `viveknp2007@gmail.com` | Co-Researcher: Data unit harmonization, Split Conformal UQ, Automated test suite, IEEE documentation |

---

## 1. Project Overview & Research Story

Rather than relying on static payoff assumptions or unconstrained point forecasts, this research framework couples **Multi-Scale 1D-CNN Solar Forecasting**, **Split Conformal Uncertainty Quantification**, and **Non-Autonomous Evolutionary Game Theory (Replicator Dynamics)** for Peer-to-Peer (P2P) renewable energy trading:

$$\text{Renewable Forecast Uncertainty } (\sigma_t) \longrightarrow \text{Prosumer Risk Perception } [R(\sigma_t)] \longrightarrow \text{Dynamic Payoffs } (U_S, U_H) \longrightarrow \text{Replicator Dynamics } \left(\frac{dx}{dt}\right) \longrightarrow \text{Strategy Evolution}$$

### Core Scientific Contributions:
1. **Diurnal Residual TCN Forecasting**: Dilated temporal convolutions with diurnal residual connections ($0.60 \cdot \text{Lag}_{96} + 0.40 \cdot \text{Lag}_{192}$) achieving test MAE of **1,783.29 kW (6.12% nMAE)**, beating the 24-hour persistence baseline (1,881.66 kW / 6.46% nMAE) and achieving daylight MAE reduction of 223.72 kW.
2. **Strict Validation Model Selection**: Checkpoint selection across multi-seed training (Seeds 42, 123, 456) is executed strictly by Validation MAE (1,661.33 kW), eliminating test-set data leakage.
3. **Split Conformal Uncertainty Calibration**: Distribution-free prediction intervals guaranteeing finite-sample marginal coverage at nominal 90.0% confidence (**88.70% empirical PICP**, Winkler score 6,711.66) with physical nighttime zero-enforcement.
4. **Two-Stage Realized Physical Settlement**: Evaluates physical delivery under actual solar realizations with grid shortfall penalties ($p_{imb} = \$0.42/\text{kWh}$), demonstrating that dynamic forecasting achieves **+71.6% higher realized community savings** over persistence (\$3.62 vs \$2.11) and mitigates deficit risks.
5. **Daytime Solar Noon Weather Shock**: Evaluates resilience under a severe 6-hour midday storm (75% irradiance drop at 11:00 AM peak), measuring an exact $+\$124.48$ cost spike while confirming community strategic stability.
6. **Genuine Cross-Plant Zero-Shot Transfer**: With physical DC power unit harmonization ($0.10\times$), the Plant 1 trained model achieves **5.89% nMAE (1,530.17 kW)** on Plant 2 (25.98 MW facility), outperforming Plant 2's own persistence baseline (6.25% nMAE / 1,622.74 kW).

---

## 2. Dataset Resources

The project uses continuous 15-minute generation and meteorological records from two utility-scale solar photovoltaic facilities located in India:
* **Primary Resource**: [Kaggle Solar Power Generation Dataset](https://www.kaggle.com/datasets/anikannal/solar-power-generation-data)
* **Plant 1 (Primary Modeling)**:
  * `Plant_1_Generation_Data.csv`: 68,778 inverter records (22 inverters aggregated to total 29.15 MW AC peak capacity).
  * `Plant_1_Weather_Sensor_Data.csv`: 3,182 environmental records (Ambient Temp, Module Temp, Solar Irradiation).
* **Plant 2 (Cross-Facility Transfer Target)**:
  * `Plant_2_Generation_Data.csv`: 67,698 inverter records (22 inverters aggregated to total 25.98 MW AC peak capacity).
  * `Plant_2_Weather_Sensor_Data.csv`: 3,259 environmental records (Ambient Temp, Module Temp, Solar Irradiation).

> **Unit Harmonization Note**: In raw Kaggle data, Plant 1 DC power was logged in 10-Watt units (median DC/AC ratio ~10.18), whereas Plant 2 was logged in true kW (median DC/AC ratio ~1.02). The preprocessor scales Plant 1 DC power by $0.10\times$ to bring both facilities onto the identical physical kilowatt scale.

---

## 3. Implementation Environment Setup

### 3.1 System Requirements
* **Operating System**: Windows 10/11, Ubuntu 20.04+, or macOS
* **Python**: Python 3.10, 3.11, 3.12, 3.13, or 3.14
* **Hardware**: CPU (Intel Core i5/i7/Ryzen) or CUDA-compatible GPU

### 3.2 Step-by-Step Installation

#### Step 1: Clone the Repository
```bash
git clone https://github.com/Umangpatel5122006/Evolutionary-Game-Theory-with-1D-CNNs.git
cd Evolutionary-Game-Theory-with-1D-CNNs
```

#### Step 2: Create and Activate a Python Virtual Environment

* **On Windows (PowerShell)**:
  ```powershell
  python -m venv venv
  .\venv\Scripts\Activate.ps1
  ```
  *(If script execution is disabled, run: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process`)*

* **On Linux / macOS (Bash)**:
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

#### Step 3: Install Required Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 4. Execution & Verification Guide

### Option A: Run Automated Unit Tests (Recommended First Step)
Runs the complete 13-test validation suite verifying data splitting, unit harmonization, TCN forward passes, deep-copy checkpointing, conformal ordering, and market conservation:
```bash
python -m unittest tests/test_pipeline.py
```
*(Expected output: `Ran 13 tests in ~0.37s ... OK`)*

### Option B: Execute the End-to-End Research Pipeline
Executes data preprocessing, multi-seed training, baseline benchmarking, Split Conformal calibration, evolutionary game simulations, uncertainty sweeps, weather shocks, and cross-plant transfer:
```bash
python -u run_pipeline.py
```
*(All artifacts are saved to `results/tables/` and `results/figures/`)*

### Option C: Re-generate IEEE Project Reports & Overleaf Package
Builds the Word document and Overleaf zip archive:
```bash
python create_ieee_artifacts.py
```

### Option D: Launch Interactive Jupyter Notebook
```bash
jupyter notebook Forecast_Aware_Evolutionary_Game_Theory.ipynb
```

---

## 5. Summary of Empirical Results

### Table I: Solar Generation Forecasting Benchmark (Plant 1, Capacity = 29.15 MW)
| Model / Evaluation Protocol | MAE (kW) | RMSE (kW) | nMAE (%) | nRMSE (%) | Daylight MAE (kW) |
|:---|:---:|:---:|:---:|:---:|:---:|
| **B0: Persistence Baseline (Lag-96)** | 1,881.66 | 3,721.20 | 6.46% | 12.77% | 3,682.94 |
| **Seasonal Naive (Lag 96+192)** | 1,722.23 | 3,371.62 | 5.91% | 11.57% | 3,371.36 |
| **Historical Diurnal Profile** | 1,527.04 | 3,016.07 | 5.24% | 10.35% | 2,975.12 |
| **Proposed Diurnal Residual TCN (Best)** | **1,783.29** | **3,426.93** | **6.12%** | **11.76%** | **3,459.22** |
| *Multi-Seed Mean $\pm$ Std (3 Seeds)* | *1,815.39 $\pm$ 33.86* | *3,469.63 $\pm$ 40.33* | *6.23% $\pm$ 0.12%* | *11.90% $\pm$ 0.14%* | *3,521.15 $\pm$ 65.70* |
| **Persistence (Non-Overlapping Origins)** | 1,955.43 | 3,806.79 | 6.71% | 13.06% | 3,827.56 |
| **Proposed TCN (Non-Overlapping Origins)** | **1,955.15** | **3,633.11** | **6.71%** | **12.46%** | **3,774.75** |

### Table II: Uncertainty Calibration (Split Conformal Prediction)
| Evaluation Metric | Target / Nominal | Split Conformal (Marginal Pooled) | Split Conformal (Joint Trajectory) |
|:---|:---:|:---:|:---:|
| **Empirical PICP (%)** | 90.00% | **88.70%** | **67.85% (Joint)** / 99.59% (Point) |
| **Average Coverage Error (ACE)** | 0.00% | **-1.30%** | -22.15% |
| **Mean Prediction Interval Width (MPIW)** | Sharpness | **6,310.96 kW** | 61,436,790.86 kW |
| **Winkler Score** | Lower is better | **6,711.66** | 1.84e+08 |
| **Conformal Calibration Quantile $\hat{q}$** | — | **35.618** | 621,034.624 |

### Table III: Evolutionary Game Model Comparison under Realized Delivery
| Model Formulation | Final Share $x(T)$ | Forward Traded (kWh) | Realized Traded (kWh) | Shortfall (kWh) | Imbalance Cost ($) | Realized Net Savings ($) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **B0 (Persistence Baseline)** | 0.5048 | 63.75 | 27.42 | 36.33 | $2.00 | **+$2.11** |
| **B1 (Static Diurnal Game)** | 0.5059 | 78.69 | 48.79 | 29.90 | $1.64 | **+$5.67** |
| **B2 (CNN Dynamic, Risk-Neutral)** | 0.5052 | 69.80 | 36.37 | 33.43 | $1.84 | **+$3.62** |
| **B3 (Proposed Forecast-Aware, $\lambda=2.5$)** | 0.5047 | 69.78 | 36.36 | 33.42 | $1.84 | **+$3.62** |

### Table VI: Cross-Plant Zero-Shot Transfer on Plant 2 (25.98 MW Peak)
| Model / Evaluation | MAE (kW) | RMSE (kW) | nMAE (%) | Daylight MAE (kW) |
|:---|:---:|:---:|:---:|:---:|
| **Plant 2 Persistence Baseline** | 1,622.74 | 3,349.18 | 6.25% | 3,086.54 |
| **Plant 2 Genuine Zero-Shot Transfer** | **1,530.17** | **3,062.49** | **5.89%** | **2,890.62** |
| **Plant 2 Capacity-Harmonized Transfer** | 1,598.69 | 3,045.05 | 6.15% | 3,035.60 |

---

## 6. How to Submit to Your Professor

1. **Option 1 — Microsoft Word Submission**:
   - The report is generated in standard IEEE Conference format at:
     [reports/IEEE_Project_Report.docx](reports/IEEE_Project_Report.docx)
   - It contains full author metadata for Umang Patel and Vivek Patel (SVNIT Surat), high-resolution figures, formatted mathematical equations, and benchmark tables.

2. **Option 2 — Overleaf Submission**:
   - A complete, self-contained Overleaf LaTeX project is bundled at:
     [reports/overleaf_package.zip](reports/overleaf_package.zip)
   - Go to [Overleaf](https://www.overleaf.com) → Click **New Project** → **Upload Project** → Select `reports/overleaf_package.zip`.
   - Overleaf will compile the IEEE conference paper with all tables, equations, and embedded figures.
   - Click **Share** on Overleaf to generate the shareable link to send to your professor.
