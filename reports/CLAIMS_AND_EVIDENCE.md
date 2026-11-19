# Scientific Claims and Empirical Evidence Matrix

**Project:** Forecast-Aware Evolutionary Game Theory for Renewable Energy Communities  
**Methodology:** Evidence-Based Defense of Claims & Rigorous Disclosure of Limitations  
**Date:** November 1, 2026  

---

## 1. Verified Defensible Claims

### Claim 1: Multi-Scale 1D-CNN (TCN) Outperforms Persistence Forecasters
- **Empirical Evidence**: The proposed `DiurnalResidualTCN` achieves a test nMAE of $6.12\%$ ($1783.29\text{ kW}$ MAE), outperforming the 24-hour persistence baseline at $6.46\%$ nMAE ($1881.66\text{ kW}$ MAE) and achieving a daylight MAE reduction of $223.72\text{ kW}$ ($3459.22\text{ kW}$ vs $3682.94\text{ kW}$).
- **Multi-Seed Stability**: Evaluated across seeds 42, 123, and 456, yielding consistent performance ($1815.39 \pm 33.86\text{ kW}$ MAE, $6.23\% \pm 0.12\%$ nMAE).

### Claim 2: Split Conformal Prediction Delivers Valid Empirical Coverage
- **Empirical Evidence**: Tested on unseen test windows, studentized split conformal calibration achieved an empirical PICP of $88.70\%$, closely matching the nominal $90.0\%$ confidence level ($	ext{ACE} = -1.30\%$) with an average interval width of $6310.96\text{ kW}$ ($21.65\%$ of peak capacity) and Winkler Score of $6711.66$.

### Claim 3: Dual-Stage Evolutionary Game Balances Traded Volume Against Shortfall Risk
- **Empirical Evidence**: When real-time grid imbalance penalties ($p_{\text{imb}} = \$0.42/\text{kWh}$) are enforced, the forecast-aware conformal mechanism (B3) and CNN dynamic mechanism (B2) reduce shortfall penalties compared to the persistence baseline ($33.42\text{ kWh}$ shortfall vs $36.33\text{ kWh}$), generating $+\$1.51$ higher realized community savings (\$3.62 vs \$2.11).

### Claim 4: The Forecasting Backbone Demonstrates Genuine Zero-Shot Transferability
- **Empirical Evidence**: When evaluated on unseen Plant 2 using strictly frozen Plant 1 scalers without any fine-tuning, the zero-shot model achieves $1530.17\text{ kW}$ MAE ($5.89\%$ nMAE), outperforming Plant 2's local persistence baseline ($1622.74\text{ kW}$ MAE, $6.25\%$ nMAE).

---

## 2. Refuted Claims & Honest Disclosure of Negative Findings

### Refuted Claim 1: "1D-CNN Outperforms All Statistical Baselines"
- **Empirical Finding**: **REFUTED**. Under clear-sky conditions, the Historical Diurnal Profile baseline achieves an MAE of $1527.04\text{ kW}$ ($5.24\%$ nMAE), outperforming the TCN ($1783.29\text{ kW}$, $6.12\%$ nMAE). In low-variance desert/semi-arid summer environments, historical diurnal averages represent an exceptionally difficult benchmark to surpass. Deep learning models provide value during dynamic weather transitions and multi-variable sensor fusion, but should not be claimed to beat diurnal profiling on clear days.

### Refuted Claim 2: "Uncertainty Causes Catastrophic Sudden Collapse of Energy Sharing"
- **Empirical Finding**: **REFUTED**. A comprehensive bifurcation sweep across risk penalty multipliers ($0.0$ to $15.0$) revealed a smooth, monotonic decline in sharing proportion ($50.52\% \to 40.15\%$) rather than a discontinuous first-order tipping point. The community exhibits high structural resilience.

### Refuted Claim 3: "Prior Zero-Shot Transfer Achieved 4.10% nMAE"
- **Empirical Finding**: **INVALID PROTOCOL**. The prior claim was an artifact of re-fitting scalers on Plant 2 test data and ignoring an uncalibrated $10\times$ DC sensor scale. The true zero-shot transfer nMAE is $5.89\%$, which legitimately beats the Plant 2 persistence baseline ($6.25\%$) without data leakage.
