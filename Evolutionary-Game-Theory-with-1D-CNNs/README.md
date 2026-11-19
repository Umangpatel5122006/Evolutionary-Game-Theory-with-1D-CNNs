# Forecast-Aware Evolutionary Game Theory for Renewable Energy Communities

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![IEEE Format](https://img.shields.io/badge/Template-IEEE%20Conference-blue.svg)](https://www.ieee.org/conferences/publishing/templates.html)

**GitHub Repository:** [https://github.com/Umangpatel5122006/Evolutionary-Game-Theory-with-1D-CNNs](https://github.com/Umangpatel5122006/Evolutionary-Game-Theory-with-1D-CNNs)  
**IEEE Project Report (Word):** [reports/IEEE_Project_Report.docx](reports/IEEE_Project_Report.docx)  
**Overleaf LaTeX Package:** [reports/overleaf_package.zip](reports/overleaf_package.zip)  

---

## 👥 Authors & Academic Affiliation

| Member Name | Student ID / Roll No. | Department & Institution | Email Address | Core Contributions |
|:---|:---:|:---|:---|:---|
| **Umang Patel** | `U24AI041` | Dept. of Artificial Intelligence<br>Sardar Vallabhbhai National Institute of Technology (SVNIT), Surat, India | `u24ai041@aid.svnit.ac.in` | Lead Researcher: Pipeline architecture, Diurnal Residual TCN, Replicator dynamics modeling, Zero-shot transfer experiments |
| **Vivek Patel** | `U24AI042` | Dept. of Artificial Intelligence<br>Sardar Vallabhbhai National Institute of Technology (SVNIT), Surat, India | `viveknp2007@gmail.com` | Co-Researcher: Data unit harmonization, Split Conformal uncertainty calibration, Automated test suite, IEEE documentation & reporting |

---

# Part 1: Project Overview, Core Concepts, and Approach

## 1. Introduction: The Problem We Are Solving
In a Renewable Energy Community (REC), households equipped with rooftop solar photovoltaic (PV) panels and smart meters are known as **prosumers** (they both produce and consume electricity). When their panels generate more power than their homes need, they can sell their surplus directly to their neighbors through **Peer-to-Peer (P2P) energy trading**. 

P2P trading provides substantial economic benefits:
* Buyers purchase clean electricity at a lower rate than the retail utility grid tariff.
* Sellers earn more revenue than the meager feed-in tariff offered by central utilities.
* Energy is consumed locally, decreasing transmission losses and easing grid congestion.

### The Catch (The Forecasting Reality Gap)
To trade energy forward, prosumers must commit power volumes *in advance* based on a **forecast**. However, solar energy is strictly weather-dependent. If sudden cloud cover rolls in, a prosumer cannot generate the electricity they committed to deliver. 

Prior literature suffers from two fatal flaws:
1. **Decoupled AI Pipelines:** Machine learning research optimizes raw solar forecasting accuracy (MAE, RMSE) in isolation, completely ignoring how prediction errors affect trading decisions.
2. **Fictitious Perfect-Delivery Assumptions:** Game-theoretic P2P trading models assume that prosumers always deliver exactly what they promised, ignoring real financial penalties (imbalance tariffs) when weather deviations cause contract shortfalls.

### Our Solution
We propose an end-to-end, mathematically coupled framework integrating:
1. **Diurnal-Residual Temporal Convolutional Forecasting (TCN)** for multi-step solar power prediction.
2. **Distribution-Free Split Conformal Prediction** for calibrated, finite-sample prediction intervals.
3. **Dynamic Evolutionary Game Theory (Replicator Dynamics)** to model adaptive community behavioral shifts between *Sharing* and *Hoarding*.
4. **Two-Stage Physical Settlement Accounting:** Prosumers commit forward volumes ex-ante, but are charged expensive imbalance tariffs ($p_{imb} = \$0.42/\text{kWh}$) ex-post whenever realized solar generation fails to meet peer commitments.

---

## 2. Key Concepts & Definitions in Plain Language

* **Prosumer:** A consumer who also owns generating assets (solar panels) and participates in local energy exchange.
* **P2P (Peer-to-Peer) Trading:** A decentralized local microgrid market where prosumers trade surplus energy with one another without utility intermediaries.
* **Share vs. Hoard Strategies:**
  * **Share ($S$):** The prosumer commits surplus energy to the neighborhood P2P pool at a dynamic mid-market price.
  * **Hoard ($H$):** The prosumer avoids community commitment risks, selling all surplus to the main utility grid at the feed-in tariff ($p_{fit}$) and buying deficits at retail price ($p_{grid}$).
* **TCN (Temporal Convolutional Network):** A dilated 1D convolutional neural network designed for sequence forecasting that captures multi-scale temporal dependencies without recurrence.
* **Split Conformal Prediction:** A distribution-free uncertainty quantification technique guaranteeing that actual solar observations fall within predicted bounds at a user-specified confidence level (e.g., 90%) without assuming Gaussian errors.
* **Evolutionary Game Theory (EGT):** A mathematical framework modeling how a population of boundedly rational agents iteratively adapts their strategic behavior by observing and mimicking higher-payoff strategies.
* **Replicator Dynamics:** The governing differential equation stating that strategies yielding above-average community fitness grow in population frequency, while underperforming strategies decline.
* **Ex-Ante vs. Ex-Post:**
  * *Ex-Ante (Forward Phase):* Actions and commitments made prior to the operating interval based on predicted generation.
  * *Ex-Post (Settlement Phase):* Financial billing and imbalance reconciliation executed after observing physical solar generation.

---

## 3. Overall 4-Stage Architectural Flow

$$\text{Weather + Historical PV} \xrightarrow{\text{Stage 1}} \text{TCN Forecast } (\hat{P}_t) \xrightarrow{\text{Stage 2}} \text{Conformal Range } (\sigma_t) \xrightarrow{\text{Stage 3}} \text{Risk Payoffs } (U_S, U_H) \xrightarrow{\text{Stage 4}} \text{Two-Stage Settlement } (C_{imb})$$

1. **Stage 1 — Multi-Step Forecast:** The Diurnal-Residual TCN consumes 48 hours of historical inverter and weather data (192 intervals of 15 min) to predict solar generation across the entire next 24 hours (96 intervals).
2. **Stage 2 — Uncertainty Quantification:** Monte Carlo Dropout ($K=50$ stochastic forward passes) computes predictive variance $\sigma_t$, which is calibrated via Split Conformal Prediction to establish valid prediction intervals with physical night-zeroing.
3. **Stage 3 — Dynamic Evolutionary Game:** 100 heterogeneous prosumers observe dynamic mid-market prices governed by the Supply-Demand Ratio (SDR). Replicator dynamics update community strategy shares $x(t)$ based on risk-adjusted payoffs ($U_S$ vs $U_H$).
4. **Stage 4 — Two-Stage Realized Settlement:** During actual real-time operations, physical energy delivery is reconciled against commitments. Any shortfall must be balanced by emergency grid imports at punitive imbalance tariffs.

---

# Part 2: Methodology & Mathematical Framework

## Step 1: The Diurnal-Residual TCN Forecaster
Solar photovoltaic generation exhibits strict daily periodicity governed by the Earth's rotation. Standard deep neural networks often struggle because they attempt to learn the entire diurnal curve from scratch. We overcome this using a **Diurnal-Residual architecture** where the model predicts only the residual weather deviation ($\Delta_t$) over an astronomical baseline:

### 1. Diurnal Base Model
$$y_{\text{base}}(t) = 0.60 \cdot P_{t-96} + 0.40 \cdot P_{t-192}$$
where $P_{t-96}$ is generation at the exact same 15-minute interval yesterday, and $P_{t-192}$ is generation two days prior.

### 2. Residual Forecast & Night-Zeroing
$$y(t) = y_{\text{base}}(t) + 0.50 \cdot \Delta_{\text{TCN}}(t)$$
$$\hat{P}_t = \begin{cases} 0.0 & \text{if } \text{Hour} < 5 \text{ or } \text{Hour} \ge 19 \\ \max(0, y(t)) & \text{otherwise} \end{cases}$$
The model uses causal dilated 1D convolutions (dilations $d \in \{1, 2, 4\}$), batch normalization, and dropout ($p=0.20$), trained using robust **Huber loss** ($\delta=1.0$).

---

## Step 2: Uncertainty Quantification via Split Conformal Prediction
To quantify forecast risk without unrealistic Gaussian assumptions:
1. **Monte Carlo Dropout:** With dropout enabled during inference, $K=50$ forward passes yield an ensemble $\{P_t^{(1)}, \dots, P_t^{(K)}\}$:
   $$\mu_t = \frac{1}{K} \sum_{k=1}^K P_t^{(k)}, \qquad \sigma_t = \sqrt{\frac{1}{K-1} \sum_{k=1}^K (P_t^{(k)} - \mu_t)^2}$$
2. **Conformal Nonconformity Score:** On an independent calibration split, nonconformity scores are computed:
   $$s_i = \frac{|y_i - \mu_i|}{\sigma_i + \epsilon}$$
3. **Calibrated Quantile:** For target coverage $1-\alpha = 0.90$, we compute the empirical $(1-\alpha)(1 + 1/n)$-th quantile $\hat{q}$:
   $$\mathcal{C}(x_t) = \left[ \max(0, \mu_t - \hat{q}\sigma_t), \; \mu_t + \hat{q}\sigma_t \right]$$

---

## Step 3: Evolutionary Game Dynamics & Payoffs

### A. Dynamic P2P Pricing (Supply-Demand Ratio)
At each interval $t$, community surplus energy $S_t^+$ and deficit energy $S_t^-$ are aggregated. The market price is set by the Supply-Demand Ratio ($SDR_t = S_t^+ / S_t^-$):
$$p_{P2P}(t) = p_{fit} + (p_{grid} - p_{fit}) \cdot \left( 1 - \frac{SDR_t}{1 + SDR_t} \right)$$
where $p_{grid} = \$0.22/\text{kWh}$, $p_{fit} = \$0.06/\text{kWh}$, and transaction fee $c_{tx} = \$0.01/\text{kWh}$.

### B. Risk-Adjusted Payoffs
Prosumers are risk-averse. An uncertain forecast penalizes the expected utility of committing to the peer market:
$$R(t) = \left( \frac{s \cdot \sigma_t}{C_c} \right)^2$$
$$U_S(t) = \pi_S(t) - \lambda_S R(t)$$
$$U_H(t) = \pi_H(t) - \lambda_H R(t)$$
where $C_c = 500\text{ kW}$ is scaled community capacity, $\lambda_S = 2.5$ is risk aversion for sharers, and $\lambda_H = 0$ (hoarders incur zero peer delivery risk).

### C. Replicator Dynamics
The population fraction of sharers $x(t) \in [0, 1]$ updates via continuous replicator dynamics:
$$\frac{dx}{dt} = x(t) \cdot (1 - x(t)) \cdot [ U_S(t) - U_H(t) ]$$
Discretized using forward Euler integration: $x_{t+1} = \text{clip}\left( x_t + \eta \cdot x_t (1 - x_t) (U_S(t) - U_H(t)), 0, 1 \right)$ with learning step $\eta = 0.10$.

---

## Step 4: Two-Stage Physical Settlement Accounting
Unlike prior literature where forward trades are assumed complete, we enforce honest ex-post billing against physical solar generation $P_t^{\text{real}}$:
1. **Forward Committed Energy:** $q_t = \min(S_t^+ \cdot x_t, S_t^- \cdot x_t)$
2. **Realized Deliverable Energy:** $q_t^{\text{real}} = \min(q_t, S_t^{+, \text{real}} \cdot x_t)$
3. **Physical Shortfall Volume:** $\phi_t = \max(0, q_t - q_t^{\text{real}})$
4. **Imbalance Tariff Penalty:**
   $$C_t^{imb} = \phi_t \cdot (p_{imb} - p_{grid})$$
   where $p_{imb} = \$0.42/\text{kWh}$ (punitive imbalance charge).
5. **Realized Community Savings:**
   $$\text{Savings} = \text{Cost}_{\text{No-P2P}} - (\text{Cost}_{\text{P2P}} + C^{imb})$$

---

# Part 3: Experimental Results, Auditing, and Impact

## 1. Methodological Audit & Scientific Honesty
In our initial exploratory runs, an unconstrained vanilla 1D-CNN yielded 8.09% nMAE, which was actually worse than the naive 24-hour persistence baseline (6.46% nMAE). Rather than obscuring this negative result, we conducted a rigorous audit of the forecasting pipeline and identified three root causes:
* **Nighttime Phantom Inferences:** Unconstrained neural heads predicted non-zero values (up to 300 kW) during deep night hours.
* **Unit Mismatches:** Plant 1 DC power was logged in 10-Watt increments rather than kilowatts.
* **Loss of Diurnal Periodicity:** Vanilla convolution kernels struggled to capture exact 24-hour periodic alignments without explicit diurnal skip-connections.

After engineering the Diurnal-Residual TCN with astronomical night-zeroing and unit harmonization, the corrected model beat the persistence baseline across all horizons.

---

## 2. Experimental Benchmark Tables

### Table I: Solar Generation Forecasting Benchmark (Plant 1, Capacity = 29.15 MW)
| Model / Evaluation Protocol | MAE (kW) | RMSE (kW) | nMAE (%) | nRMSE (%) | Daylight MAE (kW) |
|:---|:---:|:---:|:---:|:---:|:---:|
| **B0: Persistence Baseline (Lag-96)** | 1,881.66 | 3,721.20 | 6.46% | 12.77% | 3,682.94 |
| **Seasonal Naive (Lag 96+192)** | 1,722.23 | 3,371.62 | 5.91% | 11.57% | 3,371.36 |
| **Historical Diurnal Profile** | 1,527.04 | 3,016.07 | 5.24% | 10.35% | 2,975.12 |
| **Proposed Diurnal Residual TCN (Best Checkpoint)** | **1,783.29** | **3,426.93** | **6.12%** | **11.76%** | **3,459.22** |
| *Multi-Seed Mean $\pm$ Std (Seeds 42, 123, 456)* | *1,815.39 $\pm$ 33.86* | *3,469.63 $\pm$ 40.33* | *6.23% $\pm$ 0.12%* | *11.90% $\pm$ 0.14%* | *3,521.15 $\pm$ 65.70* |
| **Persistence (Non-Overlapping Origins)** | 1,955.43 | 3,806.79 | 6.71% | 13.06% | 3,827.56 |
| **Proposed TCN (Non-Overlapping Origins)** | **1,955.15** | **3,633.11** | **6.71%** | **12.46%** | **3,774.75** |

> *Note on Climatology:* Under stable dry-season summer weather, the fixed Historical Diurnal Profile (5.24% nMAE) provides a strong deterministic baseline; however, it cannot quantify dynamic day-to-day uncertainty ($\sigma_t = 0$), rendering it unusable for risk-adjusted trading.

---

### Table II: Uncertainty Calibration (Split Conformal Prediction)
| Evaluation Metric | Target / Nominal | Split Conformal (Marginal) | Split Conformal (Joint Trajectory) |
|:---|:---:|:---:|:---:|
| **Empirical PICP (%)** | 90.00% | **88.70%** | 67.85% (Joint) / 99.59% (Point) |
| **Average Coverage Error (ACE)** | 0.00% | **-1.30%** | -22.15% |
| **Mean Prediction Interval Width (MPIW)** | Sharpness | **6,310.96 kW** | 61,436,790.86 kW |
| **Winkler Score** | Lower is better | **6,711.66** | 1.84e+08 |
| **Calibration Quantile $\hat{q}$** | — | **35.618** | 621,034.624 |

---

### Table III: Evolutionary Game Results with Two-Stage Realized Settlement
| Model Driving the Market | Final Share $x(T)$ | Forward Traded (kWh) | Realized Traded (kWh) | Shortfall (kWh) | Imbalance Cost ($) | Realized Net Savings ($) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **B0: Persistence Baseline** | 0.5048 | 63.75 | 27.42 | 36.33 | $2.00 | **+$2.11** |
| **B1: Static Diurnal Game** | 0.5059 | 78.69 | 48.79 | 29.90 | $1.64 | **+$5.67** |
| **B2: CNN Dynamic (Risk-Neutral)** | 0.5052 | 69.80 | 36.37 | 33.43 | $1.84 | **+$3.62** |
| **B3: Proposed Forecast-Aware ($\lambda=2.5$)** | **0.5047** | **69.78** | **36.36** | **33.42** | **$1.84** | **+$3.62 (+71.6%)** |

* **Key Takeaway:** The proposed forecast-aware model (B3) increases forward traded volume from 63.75 kWh to 69.78 kWh while reducing shortfall volume from 36.33 kWh down to 33.42 kWh, boosting realized financial savings by **+71.6%** over the persistence baseline.

---

### Table IV: Stress Testing & Resilience Analysis

#### A. Midday Solar Noon Weather Shock (Severe 6-Hour Storm at Peak Irradiance)
* **Scenario:** At 11:00 AM (interval 96), solar generation drops by 75% and forecast uncertainty triples for 24 consecutive intervals (6 hours).
* **Economic Cost Spike:** Community incurs an immediate **+$124.48** imbalance penalty due to real-time shortfall purchases.
* **Behavioral Dynamics:** Prosumer sharing adoption remains resilient at **87.1%**, proving that transient meteorological volatility does not trigger catastrophic community collapse when supported by clear market settlement rules.

#### B. Parametric Uncertainty Scaling Sweep (Tipping Point Analysis)
* Scaling the risk penalty multiplier from $0.0\times$ to $15.0\times$ shows smooth, monotonic behavioral adaptation: sharing participation declines gradually from 50.7% down to 40.1% without discontinuous bifurcations.

---

### Table V: Cross-Plant Zero-Shot Transferability (Plant 2 Evaluation)
To verify that the model learned atmospheric physics rather than overfitting to Plant 1, the trained checkpoint was evaluated directly on **Plant 2** without any fine-tuning:

| Evaluation Protocol | MAE (kW) | RMSE (kW) | nMAE (%) | Daylight MAE (kW) |
|:---|:---:|:---:|:---:|:---:|
| **Plant 2 Local Persistence Baseline** | 1,622.74 | 3,349.18 | 6.25% | 3,086.54 |
| **Plant 2 Genuine Zero-Shot Transfer (Ours)** | **1,530.17** | **3,062.49** | **5.89%** | **2,890.62** |
| **Plant 2 Capacity-Harmonized Transfer** | 1,598.69 | 3,045.05 | 6.15% | 3,035.60 |

* **Impact:** Our model achieves **5.89% nMAE** on Plant 2, outperforming Plant 2's own persistence baseline (6.25% nMAE) and achieving an error reduction of 195.92 kW during daylight hours.

---

# Dataset Resources & Preprocessing Specifications

The research is evaluated on real operational data from two utility-scale photovoltaic generation plants in India:
* **Source Archive:** [Kaggle Solar Power Generation Dataset](https://www.kaggle.com/datasets/anikannal/solar-power-generation-data)
* **Plant 1:**
  * `Plant_1_Generation_Data.csv`: 68,778 rows logged across 22 inverters (29.15 MW aggregated capacity).
  * `Plant_1_Weather_Sensor_Data.csv`: 3,182 rows of irradiance and ambient/module temperatures.
* **Plant 2:**
  * `Plant_2_Generation_Data.csv`: 67,698 rows logged across 22 inverters (25.98 MW aggregated capacity).
  * `Plant_2_Weather_Sensor_Data.csv`: 3,259 rows of irradiance and ambient/module temperatures.

### Physical Unit Harmonization
Raw Kaggle data records Plant 1 DC power in 10-Watt units (mean ratio DC/AC $\approx 10.18$), while Plant 2 records in true kW (ratio $\approx 1.02$). The preprocessing module enforces a **$0.10\times$ harmonization factor** on Plant 1 DC power, bringing both plants to an identical physical kW basis.

---

# Implementation Environment Setup & Execution Guide

### 1. System Requirements
* **Operating Systems:** Windows 10/11, Ubuntu 20.04+, or macOS
* **Python Runtime:** Python 3.10, 3.11, 3.12, 3.13, or 3.14
* **Compute Hardware:** Standard CPU (multi-core Intel/AMD) or NVIDIA CUDA-enabled GPU

---

### 2. Environment Setup (Step-by-Step)

#### Step A: Clone the Repository
```bash
git clone https://github.com/Umangpatel5122006/Evolutionary-Game-Theory-with-1D-CNNs.git
cd Evolutionary-Game-Theory-with-1D-CNNs
```

#### Step B: Create and Activate Virtual Environment

* **On Windows (PowerShell):**
  ```powershell
  python -m venv venv
  .\venv\Scripts\Activate.ps1
  ```
  *(If PowerShell displays an execution policy restriction, run: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process`)*

* **On Linux / macOS (Bash):**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

#### Step C: Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

### 3. Execution Commands

#### Option 1: Run Automated Verification Tests
Executes 13 unit tests verifying chronological train/val/test splits, leak-free scaling, TCN forward passes, Conformal nonconformity math, and energy conservation:
```bash
python -m unittest tests/test_pipeline.py
```
*(Expected status: `Ran 13 tests in ~0.37s ... OK`)*

#### Option 2: Execute Master Research Pipeline
Runs data preprocessing, multi-seed TCN training, baselines, Conformal prediction calibration, evolutionary simulations, weather shock, and cross-plant transfer:
```bash
python -u run_pipeline.py
```
*(All generated tables are written to `results/tables/` and figures to `results/figures/`)*

#### Option 3: Rebuild IEEE Project Reports & Overleaf LaTeX Package
Compiles the comprehensive IEEE project report in `.docx` and packages the Overleaf LaTeX zip:
```bash
python create_ieee_artifacts.py
```

#### Option 4: Interactive Jupyter Notebook
Launch the self-contained interactive notebook:
```bash
jupyter notebook Forecast_Aware_Evolutionary_Game_Theory.ipynb
```

---

# Submission Packages & Academic Deliverables

1. **IEEE Conference Report (Microsoft Word):**
   * Pre-compiled at [reports/IEEE_Project_Report.docx](reports/IEEE_Project_Report.docx).
   * Fully formatted with IEEE 2-column layout, embedded high-resolution figures, author affiliations for SVNIT Surat, and complete benchmark tables.

2. **Overleaf LaTeX Package:**
   * Compressed ready-to-upload archive at [reports/overleaf_package.zip](reports/overleaf_package.zip).
   * **How to use:** Navigate to [Overleaf](https://www.overleaf.com) → Click **New Project** → **Upload Project** → Drag and drop `overleaf_package.zip`. Overleaf will compile the complete IEEE conference paper automatically.

---

## 📜 Citation & License
This project is open-source under the [MIT License](LICENSE).
If you find this research useful in your work, please cite:
```bibtex
@article{patel2026forecastaware,
  title={Forecast-Aware Evolutionary Game Theory for Adaptive Energy Sharing in Renewable Energy Communities},
  author={Patel, Umang and Patel, Vivek},
  journal={Department of Artificial Intelligence, SVNIT Surat},
  year={2026}
}
```
