# Research Project History & Evolution Narrative

**Repository:** `Evolutionary-Game-Theory-with-1D-CNNs`  
**Foundational Motivation:** Developing an adaptive, forecast-aware evolutionary game-theoretic framework for distributed renewable energy sharing.  
**Author / Maintainers:** Umang Patel (SVNIT Surat), Vivek Patel  
**Archival Date:** November 1, 2026  

---

## 1. Project Genesis & Conception (October 2026)
The research began with the recognition of a key gap in transactive energy literature: while numerous studies deploy deep learning (e.g., 1D-CNN, LSTM) for solar generation forecasting, and separate game-theoretic models (e.g., Nash Bargaining, Replicator Dynamics) for peer-to-peer energy trading, existing frameworks treat forecasting and game dynamics as disjoint pipelines. In existing models, deterministic point forecasts are fed into static payoff matrices, ignoring prediction uncertainty and real-time shortfall penalties.

The core conceptual hypothesis of this project was:
> *Injecting deep-learning forecast uncertainty directly into prosumer evolutionary payoff functions will drive adaptive risk mitigation, preventing over-commitment and stabilizing community P2P energy sharing during volatile weather.*

On October 10, 2026, the initial experimental implementation was committed (`7a0588cbb50c7faef8379e0ec78eefed5e17f756`) containing raw Kaggle datasets, preliminary preprocessing, a 3-layer 1D-CNN forecaster, MC Dropout uncertainty, and replicator dynamics simulations.

---

## 2. Phase 1 Audit & Identification of Critical Flaws (October 2026)
A rigorous scientific audit of the original codebase identified critical methodological defects:
1. **Shallow Receptive Field & Overfitting**: The original CNN had a receptive field of only 15 steps (3.75 hours) on a 192-step input, and its 1.57M parameter dense layer caused severe overfitting, resulting in an nMAE of $8.09\%$ that was beaten by a zero-parameter 24-hour persistence baseline ($6.46\%$).
2. **Nighttime Energy Hallucination**: The model lacked astronomical night constraints, hallucinating substantial non-zero solar generation during hours of darkness ($19:00 - 05:00$).
3. **Target Scaler Leakage in Cross-Plant Transfer**: For Plant 2 evaluation, new feature scalers were fitted directly on Plant 2 test data, violating zero-shot protocol, while Plant 1's $10\times$ DC power sensor multiplier was uncalibrated.
4. **Physically Absurd Constant Solar Baseline (M1)**: The baseline fed a static constant solar value 24 hours a day, falsely creating midnight solar trading.
5. **Absence of Realized Imbalance Penalties**: Simulation payoffs were calculated purely on forward forecasts without penalizing solar shortfalls, making the uncertainty-aware model appear inferior to a risk-neutral model.

---

## 3. Methodological Remediation & Rigorous Benchmarking (November 2026)
During the remediation phase, the entire pipeline was systematically upgraded:
- **Architecture**: Implemented `DiurnalResidualTCN` with dilated causal convolutions (receptive field $\ge 192$), residual skip connections, GELU activations, and physical non-negativity + astronomical night masking.
- **Strict Validation Selection**: Seed models (42, 123, 456) were evaluated, and the best checkpoint was chosen strictly on validation loss ($1661.33\text{ kW}$ MAE), eliminating test-set leakage.
- **Fair Baselines**: Benchmarked against Persistence (Lag-96), Seasonal Naive (Lag 96+192), and a true time-of-day Historical Diurnal Profile baseline.
- **Uncertainty Calibration**: Implemented Split Conformal Prediction, achieving $88.70\%$ empirical test coverage at $90.0\%$ nominal coverage.
- **Dual-Stage Game Payoffs**: Formulated forward P2P market clearing paired with realized physical shortfall settlement at grid tariff surcharges.
- **Domain Generalization**: Plant 1 DC sensor scaled by $0.10$; frozen Plant 1 scalers evaluated directly on Plant 2 in a true zero-shot protocol ($5.89\%$ nMAE, beating Plant 2 persistence $6.25\%$).

All historical records have been preserved in `research_archive/` and `results/original/`, while corrected findings are documented in `results/corrected/` and registered in `experiments/00_experiment_registry/`.
