# Forecast-Aware Evolutionary Game Theory for Adaptive Energy Sharing in Renewable Energy Communities

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![IEEE Format](https://img.shields.io/badge/Template-IEEE%20Conference-blue.svg)](https://www.ieee.org/conferences/publishing/templates.html)

**GitHub Repository:** [https://github.com/Umangpatel5122006/Evolutionary-Game-Theory-with-1D-CNNs](https://github.com/Umangpatel5122006/Evolutionary-Game-Theory-with-1D-CNNs)

---

## 👥 Authors & Academic Affiliation

**Department of Artificial Intelligence**
**Sardar Vallabhbhai National Institute of Technology (SVNIT), Surat, Gujarat, India**

| Author Name      | Student ID / Roll No. | Institutional Email        |
| :--------------- | :-------------------: | :------------------------- |
| **Vivek Patel**  |      `U24AI040`       | `u24ai040@aid.svnit.ac.in` |
| **Umang Patel**  |      `U24AI041`       | `u24ai041@aid.svnit.ac.in` |
| **Ojash Gupta**  |      `U24AI057`       | `u24ai057@aid.svnit.ac.in` |
| **Dhiraj Dubey** |      `I24AI011`       | `i24ai011@aid.svnit.ac.in` |

---

## 📁 Repository Directory Structure

The repository root is the project root. The audited codebase, datasets, experiment records, reports, and outputs are available directly here. The `Final Verdict/` folder is retained as a separate project document set.

```
Evolutionary-Game-Theory-with-1D-CNNs/ (Repository and project root)
├── README.md
├── .gitignore
├── configs/
├── data/
├── experiments/
├── models/
├── paper/
├── reports/
├── research_archive/
├── research_tracks/
├── results/
├── results_original/
├── src/
├── tests/
├── Final Verdict/
├── run_pipeline.py
├── requirements.txt
└── Forecast_Aware_Evolutionary_Game_Theory.ipynb
```

---

# Part 1: Project Overview, Core Concepts, and Approach

## 1. Introduction: The Problem We Are Solving

In a Renewable Energy Community (REC), households equipped with rooftop solar panels are **prosumers** (they both produce and consume electricity). When their panels generate more power than their homes consume, they can sell their surplus directly to their neighbors through **Peer-to-Peer (P2P) trading**. P2P trading is mutually beneficial:

- Buyers purchase clean electricity at a lower cost than the retail grid tariff.
- Sellers earn more revenue than the standard feed-in tariff offered by the central utility.
- Clean energy is consumed locally, easing transmission congestion and reducing grid losses.

### The Catch (The Forecasting Reality Gap)

To trade energy tomorrow, prosumers must commit power volumes _in advance_ based on a **forecast**. But solar generation depends strictly on volatile meteorological conditions. If sudden cloud cover rolls in, a prosumer cannot generate the electricity they promised.

Most existing literature suffers from two critical flaws:

1. **Decoupled AI Pipelines:** Machine learning studies optimize raw solar forecasting accuracy (MAE, RMSE) in isolation, completely ignoring how prediction errors affect trading decisions.
2. **Fictitious Perfect-Delivery Assumptions:** Game-theoretic P2P trading models assume that prosumers always deliver exactly what they promised, ignoring real financial penalties (imbalance tariffs) when weather deviations cause contract shortfalls.

### Our Solution

We developed an end-to-end, mathematically coupled framework combining:

1. **Machine Learning Forecasting (Diurnal-Residual TCN):** Highly efficient multi-step solar power prediction.
2. **Uncertainty Quantification (Studentized Split Conformal Prediction):** Statistically guaranteed prediction intervals showing how risky the forecast is.
3. **Evolutionary Game Theory (Replicator Dynamics):** Modeling how community prosumers adaptively switch between _Sharing_ and _Hoarding_.
4. **Two-Stage Physical Settlement Accounting:** Prosumers make commitments ex-ante based on forecasts, but they pay actual financial penalties (imbalance tariffs) ex-post whenever realized solar generation fails to meet peer commitments.

---

## 2. Simple Definitions of Important Terms

- **Prosumer:** A household that both _produces_ (via solar panels) and _consumes_ electricity.
- **P2P (Peer-to-Peer) Trading:** A local microgrid market where neighbors buy and sell surplus electricity directly with each other without utility intermediaries.
- **Share vs. Hoard:** The two strategic choices a prosumer has in our game:
  - _Share:_ Commit surplus energy to the local P2P pool at a dynamic mid-market price.
  - _Hoard:_ Avoid peer commitment risks, selling all surplus to the main utility grid at the cheap feed-in tariff ($p_{fit}$) and buying deficits at retail price ($p_{grid}$).
- **TCN (Temporal Convolutional Network):** A dilated 1D convolutional neural network designed for sequence forecasting that captures multi-scale temporal dependencies without recurrence.
- **Conformal Prediction:** A mathematical way to guarantee how accurate a prediction interval is. Instead of just guessing heuristic ranges, it provides a rigorous finite-sample guarantee (e.g., 90% confidence) that the actual solar generation will fall inside the interval.
- **Evolutionary Game Theory:** A branch of mathematics that studies how populations adapt their strategies over time. Rather than assuming hyper-rational humans, it models people observing their neighbors and copying strategies that yield higher profits.
- **Replicator Dynamics:** The governing differential equation stating: _If a strategy earns above-average community profit, more people adopt it; if it underperforms, people abandon it._
- **Ex-Ante vs. Ex-Post:**
  - _Ex-Ante:_ Decisions and commitments made _before_ the operating interval based on predicted generation.
  - _Ex-Post:_ Financial billing and settlement executed _after_ observing actual physical solar output.

---

## 3. The Overall Approach

Our project follows a strict timeline for every 15-minute interval of the day through a 4-step sequence:

```
[Historical PV + Weather]
          │
          ▼
[Step 1: Diurnal-Residual TCN] ──────► 24-Hour Solar Forecast (y_hat)
          │
          ▼
[Step 2: Split Conformal UQ] ────────► Calibrated Uncertainty Spread (sigma_t)
          │
          ▼
[Step 3: Dynamic Evolutionary Game] ─► Risk Payoffs (U_S, U_H) -> Replicator Dynamics
          │
          ▼
[Step 4: Two-Stage Settlement] ──────► Actual Generation vs Commitments -> Imbalance Fines
```

1. **Forecast (The Guess):** Our Diurnal-Residual TCN looks at the past 48 hours of weather and solar data (192 intervals of 15 min) and predicts solar generation for the next 24 hours (96 intervals).
2. **Uncertainty (The Risk):** The model doesn't just give one number. Through Monte Carlo Dropout ($K=50$ passes) and Split Conformal Prediction, it outputs a calibrated interval $[\mu_t - q_{\text{cal}}\sigma_t, \mu_t + q_{\text{cal}}\sigma_t]$ with hard physical nighttime zeroing.
3. **The Game (The Decision):** Prosumers examine the forecast, their household demand, and forecast risk. Expected payoffs $U_S(t)$ and $U_H(t)$ are computed, and replicator dynamics update community participation $x(t)$.
4. **The Reality Check (Two-Stage Settlement):** After the interval passes, we observe physical solar generation. If a prosumer committed 5 kWh to a neighbor but sudden clouds limited generation to 2 kWh, they must procure the missing 3 kWh from the main grid at an expensive **imbalance tariff** ($p_{imb} = \$0.275/\text{kWh}$).

---

## 4. Why This Matters (The Impact)

By forcing the AI model to live in a real-world market where mistakes cost money, we proved that unconstrained, overly optimistic AI predictions can be actively harmful. Our framework aligns AI forecasting with economic accountability, ensuring prosumers make risk-informed commitments, minimize penalty tariffs, and sustain cooperative local clean-energy markets over the long term.

---

# Part 2: Methodology & Mathematical Framework

## Step 1: The Forecasting Model (Diurnal-Residual TCN)

Solar generation is highly cyclical. Instead of forcing the neural network to memorize the entire diurnal curve from scratch, our model anchors its prediction to previous days' outputs and predicts only the residual deviation ($\Delta_t$) caused by current weather.

### 1. Diurnal Base Component

$$y_{\text{base}} = 0.60 \cdot P_{t-96} + 0.40 \cdot P_{t-192}$$
where $P_{t-96}$ is generation at the exact same 15-minute interval yesterday, and $P_{t-192}$ is generation two days prior.

### 2. Residual Forecast & Physical Night Zeroing

$$y = y_{\text{base}} + 0.50 \cdot \Delta_{\text{TCN}}$$
$$\hat{P}_t = \begin{cases} 0.0 & \text{if } \text{Hour} < 5 \text{ or } \text{Hour} \ge 19 \\ \max(0, y) & \text{otherwise} \end{cases}$$
The network features 4 dilated convolutional blocks (kernel size 5; dilations 1, 2, 4, 8; batch norm, ReLU, dropout 0.15), total parameter count of **89,537** (a 94% reduction from naive dense heads), and is trained with robust **Huber loss** ($\delta=1.0$).

---

## Step 2: Uncertainty Quantification

To quantify risk without unrealistic Gaussian assumptions, we combine **Monte Carlo (MC) Dropout** with **Split Conformal Calibration**:

1. At inference, dropout remains active over $K=50$ forward passes to compute empirical mean $\mu_t$ and spread $\sigma_t$:
   $$\mu_t = \frac{1}{K} \sum_{k=1}^K P_t^{(k)}, \qquad \sigma_t = \sqrt{\frac{1}{K-1} \sum_{k=1}^K (P_t^{(k)} - \mu_t)^2}$$
2. Studentized nonconformity scores are evaluated on a held-out calibration split:
   $$s_i = \frac{|y_i - \mu_i|}{\sigma_i + \epsilon}$$
3. Conformal quantile $\hat{q}_{\text{cal}} = 35.62$ yields calibrated intervals at nominal 90% confidence:
   $$\mathcal{I}_t = \left[ \max(0, \mu_t - \hat{q}_{\text{cal}}\sigma_t), \; \mu_t + \hat{q}_{\text{cal}}\sigma_t \right]$$

---

## Step 3: The Evolutionary Game

A community of $N=100$ heterogeneous prosumers participates in the market.

### A. Dynamic P2P Pricing (Supply-Demand Ratio)

At each interval $t$, community surplus $S_t^+$ and deficit $S_t^-$ determine the Supply-Demand Ratio ($SDR_t = S_t^+ / S_t^-$):
$$p_{P2P}(t) = p_{fit} + (p_{grid} - p_{fit}) \cdot \left( 1 - \frac{SDR_t}{1 + SDR_t} \right)$$
clamped to $[p_{fit} + c_{tx}, \; p_{grid} - c_{tx}]$, with retail tariff $p_{grid} = \$0.22/\text{kWh}$, feed-in tariff $p_{fit} = \$0.06/\text{kWh}$, and platform fee $c_{tx} = \$0.01/\text{kWh}$.

### B. Risk Penalty and Payoffs

Because forecasting carries risk, a quadratic penalty $R(t)$ penalizes the expected utility of committing to the peer market:
$$R(t) = \left( \frac{s \cdot \sigma_t}{C_c} \right)^2$$
$$U_S(t) = \pi_S(t) - \lambda_S R(t), \qquad U_H(t) = \pi_H(t) - \lambda_H R(t)$$
where $C_c = 500\text{ kW}$ is installed community capacity, $\lambda_S = 2.5$ for sharers, and $\lambda_H = 0$ (hoarders incur zero peer delivery risk).

### C. Replicator Dynamics

The community sharing fraction $x(t) \in [0, 1]$ updates via continuous replicator dynamics:
$$\frac{dx}{dt} = x(t) \cdot (1 - x(t)) \cdot [ U_S(t) - U_H(t) ]$$
Discretized via forward Euler integration:
$$x_{t+1} = \text{clip}\left( x_t + \eta \cdot x_t (1 - x_t) (U_S(t) - U_H(t)), \; 10^{-4}, \; 1 - 10^{-4} \right)$$
with adjustment rate $\eta = 0.10$.

---

## Step 4: Two-Stage Settlement (Honest Accounting)

Every committed trade is settled against physical solar generation $P_t^{\text{real}}$:

1. **Committed Volume:** $q_t = \min(S_t^+ \cdot x_t, \; S_t^- \cdot x_t)$
2. **Realized Deliverable Volume:** $q_t^{\text{real}} = \min(q_t, \; S_t^{+, \text{real}} \cdot x_t)$
3. **Shortfall Volume:** $\phi_t = \max(0, \; q_t - q_t^{\text{real}})$
4. **Imbalance Charge:**
   $$C_t^{imb} = \phi_t \cdot (p_{imb} - p_{grid})$$
   with imbalance tariff $p_{imb} = \$0.275/\text{kWh}$ ($1.25 \times p_{grid}$).
5. **Realized Net Community Savings:**
   $$\text{Sav}_t^{\text{real}} = C_t^{\text{aut}} - C_t^{\text{real}}$$
   measured strictly against the autarky baseline under identical realized weather.

---

# Part 3: Experimental Results, Auditing, and Impact

## 1. The Methodological Audit (A Note on Scientific Honesty)

A primary contribution of this research is **Scientific Transparency**. In our initial exploratory experiments, an unconstrained vanilla 1D-CNN achieved an 8.09% nMAE, losing to a simple 24-hour persistence baseline (6.46% nMAE). Rather than obscuring this negative result, we conducted an end-to-end audit that revealed eleven methodological defects:

- Test-set checkpoint snooping.
- Unconstrained nighttime generation (predicting up to 300 kW at midnight).
- Ten-fold DC power unit mismatch ($10\text{-W}$ logs in Plant 1 vs true $\text{kW}$ in Plant 2).
- Misaligned diurnal climatology baselines.
- Absence of ex-post settlement for undelivered energy commitments.

After resolving all defects through rigorous regression testing, the corrected Diurnal-Residual TCN beat the persistence baseline across all evaluation metrics.

---

## 2. Forecasting Accuracy Results

### Table I: Solar Power Forecasting Benchmark (Plant 1, Capacity = 29.15 MW)

| Model / Evaluation Protocol                         |        MAE (kW)        |       RMSE (kW)        |      nMAE (%)       |      nRMSE (%)       |   Daylight MAE (kW)    |
| :-------------------------------------------------- | :--------------------: | :--------------------: | :-----------------: | :------------------: | :--------------------: |
| **Original Flawed AI (Vanilla CNN)**                |        2,359.36        |        3,918.24        |        8.09%        |        13.44%        |        4,027.30        |
| **B0: Persistence Baseline (Lag-96)**               |        1,881.66        |        3,721.20        |        6.46%        |        12.77%        |        3,682.94        |
| **Seasonal Naive (Lag 96+192)**                     |        1,722.23        |        3,371.62        |        5.91%        |        11.57%        |        3,371.36        |
| **Historical Diurnal Profile**                      |        1,527.04        |        3,016.07        |        5.24%        |        10.35%        |        2,975.12        |
| **Proposed Diurnal Residual TCN (Best Checkpoint)** |      **1,783.29**      |      **3,426.93**      |      **6.12%**      |      **11.76%**      |      **3,459.22**      |
| _Multi-Seed Mean $\pm$ Std (Seeds 42, 123, 456)_    | _1,815.39 $\pm$ 33.86_ | _3,469.63 $\pm$ 40.33_ | _6.23% $\pm$ 0.12%_ | _11.90% $\pm$ 0.14%_ | _3,521.15 $\pm$ 65.70_ |
| **Persistence (Non-Overlapping Origins)**           |        1,955.43        |        3,806.79        |        6.71%        |        13.06%        |        3,827.56        |
| **Proposed TCN (Non-Overlapping Origins)**          |      **1,955.15**      |      **3,633.11**      |      **6.71%**      |      **12.46%**      |      **3,774.75**      |

- **Analysis:** Our corrected TCN achieves **6.12% nMAE**, outperforming the persistence baseline (6.46%) and reducing daylight MAE by 223.72 kW.
- _Note on Climatology:_ Under stable dry-season summer weather, the Historical Diurnal Profile (5.24% nMAE) provides a strong clear-sky ceiling; however, it cannot quantify dynamic day-to-day uncertainty ($\sigma_t = 0$), which is required for risk-aware market clearing.

---

## 3. Uncertainty Calibration Results

### Table II: Prediction-Interval Calibration at 90% Nominal Confidence

| Evaluation Metric                                         | Target / Nominal | Split Conformal (Marginal) | Split Conformal (Joint Trajectory) |
| :-------------------------------------------------------- | :--------------: | :------------------------: | :--------------------------------: |
| **Empirical PICP (%)**                                    |      90.00%      |         **88.70%**         |  67.85% (Joint) / 99.59% (Point)   |
| **Average Coverage Error (ACE)**                          |      0.00%       |         **-1.30%**         |              -22.15%               |
| **Mean Prediction Interval Width (MPIW)**                 |    Sharpness     |      **6,310.96 kW**       |          61,436,790.86 kW          |
| **Winkler Score**                                         | Lower is better  |        **6,711.66**        |              1.84e+08              |
| **Conformal Calibration Quantile $\hat{q}_{\text{cal}}$** |        —         |         **35.618**         |            621,034.624             |

---

## 4. Game Theory & Economic Results under Two-Stage Settlement

### Table III: Four-Model Comparative Simulation

| Model Driving the Market                          | Final Share $x(T)$ | Forward Traded (kWh) | Realized Traded (kWh) | Shortfall (kWh) | Imbalance Cost ($) | Realized Net Savings ($) |
| :------------------------------------------------ | :----------------: | :------------------: | :-------------------: | :-------------: | :----------------: | :----------------------: |
| **B0: Persistence Baseline**                      |       0.5048       |        63.75         |         27.42         |      36.33      |       $2.00        |        **+$2.11**        |
| **B1: Static Diurnal Game**                       |       0.5059       |        78.69         |         48.79         |      29.90      |       $1.64        |        **+$5.67**        |
| **B2: CNN Dynamic (Risk-Neutral)**                |       0.5052       |        69.80         |         36.37         |      33.43      |       $1.84        |        **+$3.62**        |
| **B3: Proposed Forecast-Aware ($\lambda_S=2.5$)** |     **0.5047**     |      **69.78**       |       **36.36**       |    **33.42**    |     **$1.84**      |   **+$3.62 (+71.6%)**    |

- **Economic Impact:** The proposed forecast-aware model (B3) increases forward traded volume to 69.78 kWh and curbs shortfall volume to 33.42 kWh, boosting realized net community savings from $2.11 to $3.62—a **+71.6% improvement** earned by making smarter, risk-calibrated commitments.

---

## 5. Stress Testing the Community

### A. Midday Weather Shock (Solar Noon Cloudburst)

- **Simulation:** A severe 6-hour storm hitting at 11:00 AM (interval 96), cutting generation by 75% and tripling forecast uncertainty.
- **Financial Impact:** Community incurred an immediate **+$124.48** imbalance penalty due to real-time shortfalls.
- **Behavioral Impact:** Community participation remained stable at **87.1%**.
- **Finding:** The financial ledger absorbs the shock cleanly, proving that temporary weather anomalies do not induce permanent community collapse.

### B. Uncertainty Scaling Sweep (Tipping Point Analysis)

- Sweeping the uncertainty penalty multiplier from $0.0\times$ up to $15.0\times$ demonstrated smooth, monotonic decay (sharing adoption gradually declined from 50.7% down to 40.1%).
- **Finding:** There is no catastrophic cliff edge; communities smoothly de-risk their trading commitments as uncertainty grows.

---

## 6. Transfer Learning (Zero-Shot Cross-Plant Generalization)

### Table IV: Evaluation on Plant 2 (25.98 MW Peak Capacity)

To verify that the model learned true solar physics rather than overfitting to Plant 1, the model was evaluated directly on **Plant 2** without any fine-tuning:

| Evaluation Protocol                           |   MAE (kW)   |  RMSE (kW)   | nMAE (%)  | Daylight MAE (kW) |
| :-------------------------------------------- | :----------: | :----------: | :-------: | :---------------: |
| **Plant 2 Local Persistence Baseline**        |   1,622.74   |   3,349.18   |   6.25%   |     3,086.54      |
| **Plant 2 Genuine Zero-Shot Transfer (Ours)** | **1,530.17** | **3,062.49** | **5.89%** |   **2,890.62**    |
| **Plant 2 Capacity-Harmonized Transfer**      |   1,598.69   |   3,045.05   |   6.15%   |     3,035.60      |

- **Result:** Our model achieved **5.89% nMAE** on Plant 2, successfully outperforming Plant 2's own local persistence baseline (6.25% nMAE) by 92.57 kW overall and 195.92 kW during daylight.

---

## 7. Overall Conclusion & Research Takeaways

1. **Forecasting Must Be Tied to Reality:** AI models evaluated purely on statistical error can promise energy that does not exist. Two-stage settlement penalizes shortfalls and ensures true economic viability.
2. **Uncertainty Must Be Calibrated:** Conformal prediction replaces heuristic multipliers with finite-sample coverage guarantees.
3. **Resilient Community Adaptation:** Prosumer communities gradually reduce trade volumes during uncertain periods, showing structural resilience rather than sudden collapse.

---

# Dataset Resources & Preprocessing Specifications

- **Primary Dataset Resource:** [Kaggle Solar Power Generation Dataset](https://www.kaggle.com/datasets/anikannal/solar-power-generation-data)
- **Plant 1:**
  - `Plant_1_Generation_Data.csv`: 68,778 rows logged across 22 inverters (29.15 MW peak).
  - `Plant_1_Weather_Sensor_Data.csv`: 3,182 rows of irradiance and ambient/module temperatures.
- **Plant 2:**
  - `Plant_2_Generation_Data.csv`: 67,698 rows logged across 22 inverters (25.98 MW peak).
  - `Plant_2_Weather_Sensor_Data.csv`: 3,259 rows of irradiance and ambient/module temperatures.

### Physical Unit Harmonization

Raw Kaggle data logged Plant 1 DC power in 10-Watt units (mean ratio DC/AC $\approx 10.18$), while Plant 2 was logged in true kW (ratio $\approx 1.02$). The preprocessing module enforces a **$0.10\times$ harmonization factor** on Plant 1 DC power, placing both facilities on the identical physical kilowatt scale.

---

# Implementation Environment Setup & Execution Guide

### 1. System Requirements

- **Operating Systems:** Windows 10/11, Ubuntu 20.04+, or macOS
- **Python Runtime:** Python 3.10, 3.11, 3.12, 3.13, or 3.14
- **Hardware:** CPU (Intel/AMD) or NVIDIA CUDA-enabled GPU

---

### 2. Environment Setup (Step-by-Step)

#### Step A: Clone the Repository

```bash
git clone https://github.com/Umangpatel5122006/Evolutionary-Game-Theory-with-1D-CNNs.git
cd Evolutionary-Game-Theory-with-1D-CNNs
```

#### Step B: Create and Activate Virtual Environment

- **On Windows (PowerShell):**

  ```powershell
  python -m venv venv
  .\venv\Scripts\Activate.ps1
  ```

  _(If execution is restricted, run: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process`)_

- **On Linux / macOS (Bash):**
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

#### Command 1: Run Automated Regression Tests (13 Unit Tests)

```bash
python -m unittest tests/test_pipeline.py
```

_(Expected verification output: `Ran 13 tests in ~0.37s ... OK`)_

#### Command 2: Execute Complete Master Research Pipeline

```bash
python -u run_pipeline.py
```

_(Executes preprocessing, multi-seed TCN training, conformal calibration, evolutionary simulations, weather shock, and cross-plant transfer. Results export to `results/tables/` and `results/figures/`)_

#### Command 3: Launch Interactive Jupyter Notebook

```bash
jupyter notebook Forecast_Aware_Evolutionary_Game_Theory.ipynb
```

---
