# Rigorous Experimental Protocol
**Project:** Forecast-Aware Evolutionary Game Theory for Renewable Energy Communities  
**Version:** 2.0 (Post-Audit Research Protocol)  
**Date:** November 1, 2026  

---

## 1. Datasets & Physical Specifications

Experiments utilize continuous 15-minute generation and meteorological data from two distinct utility-scale solar photovoltaic facilities located in India (Kaggle Solar Power Generation Dataset):

| Parameter | Plant 1 (Primary Testbed) | Plant 2 (Transfer Target) | Notes / Harmonization |
|:---|:---:|:---:|:---|
| **Monitoring Window** | May 15, 2020 – June 17, 2020 | May 15, 2020 – June 17, 2020 | 34 contiguous calendar days |
| **Total Intervals** | 3,264 intervals (15-min grid) | 3,264 intervals (15-min grid) | Continuous time grid without temporal gaps |
| **Number of Inverters** | 22 inverters | 22 inverters | Aggregated to plant-level total power |
| **Observed Peak AC Power** | 29,150.21 kW (29.15 MW) | 25,979.76 kW (25.98 MW) | Used for nMAE and nRMSE normalization |
| **Raw DC Power Scale** | 10-Watt units (Peak ~298.9 MW) | Kilowatts (Peak ~26.6 MW) | **Harmonized**: Plant 1 DC scaled by $0.10\times$ |
| **Weather Sensors** | Ambient Temp, Module Temp, Irradiation | Ambient Temp, Module Temp, Irradiation | Co-located pyranometer & thermal sensors |

---

## 2. Chronological Data Splitting & Leak-Free Windowing

To prevent temporal data leakage, all splits are strictly chronological:

| Split Partition | Ratio | Interval Range | Timestamp Extent | Purpose |
|:---|:---:|:---:|:---|:---|
| **Training Set** | 70% | Intervals 0 – 2,284 (2,285 intervals) | 2020-05-15 00:00 to 2020-06-07 18:45 | Model training & parameter estimation |
| **Validation Set** | 15% | Intervals 2,093 – 2,774 (682 intervals)* | 2020-06-07 19:00 to 2020-06-12 21:15 | Model selection, early stopping, conformal calibration |
| **Test Set** | 15% | Intervals 2,583 – 3,263 (681 intervals)* | 2020-06-12 21:30 to 2020-06-17 23:45 | Unseen evaluation of models & game dynamics |

*\*Note: Validation and test splits include a 192-step history buffer from the preceding partition to ensure that target forecasting horizons start strictly at the partition boundary.*

### Sliding Window Configuration
- **Input History Length ($L_{in}$)**: 192 time steps (48 continuous hours / 2 days).
- **Forecast Horizon ($H$)**: 96 time steps (24 continuous hours / 1 day ahead).
- **Evaluation Protocols**:
  1. *Rolling Sliding Window*: Step size $S = 1$ interval (15 minutes), producing 395 evaluation windows.
  2. *Non-Overlapping Daily Rollouts*: Step size $S = 96$ intervals (24 hours), producing 5 statistically independent 24-hour test days.

---

## 3. Forecasting Models & Baseline Specifications

### 3.1 Proposed Model: Diurnal Residual TCN (`DiurnalResidualTCN`)
- **Input Channels (9)**: `[AC_POWER, DC_POWER, AMBIENT_TEMP, MODULE_TEMP, IRRADIATION, sin_tod, cos_tod, sin_dow, cos_dow]`.
- **Architecture**:
  - Input Projection: `Conv1d(9, 64, kernel_size=1)`.
  - Dilated Residual Blocks: 4 layers with `kernel_size=5`, dilations $d \in \{1, 2, 4, 8\}$, batch normalization, ReLU activation, and residual skip connections.
  - Dropout: $p = 0.15$ retained during inference for Monte Carlo Dropout.
  - Adaptive Pooling: Maps 192 feature maps to 96 forecast horizon steps.
  - Prediction Head: `Conv1d(64, 32, kernel_size=3) -> ReLU -> Dropout(0.15) -> Conv1d(32, 1, kernel_size=1)`.
  - Diurnal Residual Connection: $\hat{y} = (0.60 \cdot \text{Lag}_{96} + 0.40 \cdot \text{Lag}_{192}) + 0.50 \cdot \delta_{CNN}$.
  - Total Parameters: 71,105 (compared to 1,570,000 in original model).
  - Astronomical Night Zeroing: Hard constraints enforcing 0.0 kW during hours $< 5$ and $\ge 19$.

### 3.2 Benchmark Baselines
1. **B0: Persistence Baseline (Lag-96)**: Takes identical 15-minute time steps from the previous 24-hour cycle ($\hat{y}_{t+h} = y_{t-96+h}$).
2. **Seasonal Naive (Lag-96 + Lag-192)**: Averages generation at the same time from 1 day ago and 2 days ago.
3. **Historical Diurnal Profile**: Averages historical observations across all training days for each of the 96 time-of-day slots.

### 3.3 Training Hyperparameters
- Loss Function: Smooth L1 (Huber) Loss ($\beta = 1.0$).
- Optimizer: AdamW ($\text{lr} = 10^{-3}$, $\text{weight\_decay} = 10^{-4}$).
- Scheduler: `ReduceLROnPlateau(mode='min', factor=0.5, patience=2)`.
- Batch Size: 32; Maximum Epochs: 25; Early Stopping Patience: 4 epochs.
- Checkpointing: Deep-copied tensor state dictionary `{k: v.cpu().clone()}`.
- Random Seeds: Evaluated across independent seeds `[42, 123, 456]`.

---

## 4. Split Conformal Uncertainty Protocol

1. **MC Dropout Sampling**: $K = 50$ stochastic forward passes with dropout active.
2. **Non-Conformity Scoring**: Computed on the validation partition:
   $$s_i = \max_{h \in [1, 96]} \frac{|y_{val, i, h} - \hat{y}_{val, i, h}|}{\hat{\sigma}_{val, i, h} + \epsilon}$$
3. **Finite-Sample Conformal Quantile**:
   $$\hat{q} = \text{Quantile}_{\left\lceil \frac{(n_{val}+1)(1-\alpha)}{n_{val}} \right\rceil}(s_i) \quad \text{for } 1 - \alpha = 0.90$$
4. **Prediction Intervals**:
   $$L_{t+h} = \max\left(0, \hat{y}_{t+h} - \hat{q} \hat{\sigma}_{t+h}\right), \quad U_{t+h} = \max\left(0, \hat{y}_{t+h} + \hat{q} \hat{\sigma}_{t+h}\right)$$
   with $L_{t+h} = U_{t+h} = 0$ enforced during nighttime.
5. **Evaluation Metrics**:
   - Prediction Interval Coverage Probability (PICP, target $\ge 90.0\%$).
   - Mean Prediction Interval Width (MPIW, kW).
   - Average Coverage Error (ACE = PICP - nominal).
   - Winkler Score (proper scoring rule penalizing width and interval violations).

---

## 5. Microeconomic Market & Evolutionary Game Dynamics

### 5.1 Community & Tariff Parameters
- Prosumer Population: $N = 100$ heterogeneous households.
- Scaled Community Peak Solar: $500$ kW (scale factor $\gamma = 500 / 29,150.21 \approx 0.01715$).
- Solar Allocations: Dirichlet distributed $\alpha \sim \text{Dirichlet}(\mathbf{1}_{100})$, fixed seed 42.
- Demand Profiles: Dual-Gaussian diurnal residential curve (morning peak 8:00, evening peak 19:30, baseload 0.8 kW, household multipliers $\in [0.6, 1.5]$).
- Grid Retail Tariff: $p_{grid} = \$0.22/\text{kWh}$.
- Feed-in Tariff: $p_{fit} = \$0.06/\text{kWh}$.
- P2P Transaction Fee: $c_{tx} = \$0.01/\text{kWh}$.
- Imbalance Shortfall Tariff: $p_{imb} = p_{grid} \times 1.25 = \$0.275/\text{kWh}$ (25% penalty above retail).

### 5.2 Dynamic P2P Pricing (Supply-Demand Ratio)
$$SDR(t) = \frac{x(t) \sum S_i(t)}{x(t) \sum D_i(t) + \epsilon}$$
$$p_{p2p}(t) = \text{clip}\left(p_{fit} + (p_{grid} - p_{fit})\left[1 - \frac{SDR(t)}{1 + SDR(t)}\right], p_{fit} + c_{tx}, p_{grid} - c_{tx}\right)$$

### 5.3 Risk-Adjusted Payoffs & Replicator Dynamics
- Normalized Uncertainty: $R_t = \left(\frac{\sigma_{comm}(t)}{500}\right)^2$.
- Strategy Payoffs: $U_S(t) = \pi_S(t) - \lambda_S R_t$, $U_H(t) = \pi_H(t) - \lambda_H R_t$, with $\lambda_S = 2.5, \lambda_H = 0.0$.
- Replicator ODE: $\frac{dx}{dt} = \eta x(t)(1 - x(t))(U_S(t) - U_H(t))$, clamped to $[10^{-4}, 1 - 10^{-4}]$.

### 5.4 Two-Stage Settlement (Ex-Ante vs Ex-Post Realized)
- *Ex-Ante Simulation*: Assumes physical delivery matches forecast commitment.
- *Ex-Post Realized Settlement*: Evaluates physical delivery under actual realized solar generation $P_{actual}$. Any shortfall in committed export ($Q_{comm} - Q_{act}$) is procured from the utility grid at $p_{imb} = \$0.275/\text{kWh}$.
- Baseline Cost: Autarky cost ($x = 0$) under actual solar generation:
  $$Cost_{base} = \sum D_{act, i} \cdot p_{grid} - \sum S_{act, i} \cdot p_{fit}$$
- Realized Net Savings: $Savings_{realized} = Cost_{base} - Cost_{community, realized}$.

---

## 6. Sensitivity, Shock & Generalization Protocols

1. **Uncertainty Sensitivity Sweep**: Multipliers $\kappa \in [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0, 15.0]$ tested across initial states $x_0 \in \{0.20, 0.50, 0.80\}$.
2. **Daytime Weather Shock Experiment**: Starts at interval 150 (Day 2, 11:00 AM), lasting 24 intervals (6 hours until 17:00). Solar drops by 75%, uncertainty surges $3\times$. Learning step size $\eta = 2.0$. Formal recovery: steps to reach $x(t) \ge 0.95 \times x_{pre}$.
3. **Genuine Zero-Shot Plant 2 Transfer**: Transforms Plant 2 features using Plant 1's fitted `StandardScaler` without target adaptation; evaluates against Plant 2 ground truth and Plant 2's own persistence baseline.

