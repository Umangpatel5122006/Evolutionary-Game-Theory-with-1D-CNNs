# Comprehensive Limitations & Scientific Boundaries

**Project:** Forecast-Aware Evolutionary Game Theory for Renewable Energy Communities  
**Document:** Methodological, Empirical, and Systemic Limitations  
**Date:** November 1, 2026  

---

## 1. Geographic and Climatic Scope
1. **Single Semi-Arid Geographic Region**: The experimental dataset originates from two utility-scale solar PV installations situated in Gandikota, Andhra Pradesh, India (latitude $\approx 14.81^\circ$N, longitude $\approx 78.28^\circ$E). Consequently, meteorological phenomena reflect high direct-normal irradiance (DNI), minimal cloud cover during summer months, and high ambient temperatures ($25^\circ\text{C}$ to $40^\circ\text{C}$).
2. **Seasonal Duration**: The dataset covers a continuous 34-day operational monitoring window (May 15, 2020 to June 17, 2020), capturing early monsoon transition dynamics. While adequate for short-term rolling forecasting benchmarks, it does not encompass deep winter low-zenith-angle irradiance or monsoon-induced heavy precipitation overcast patterns.

## 2. Microgrid Scale and Physical Granularity
1. **Aggregated Plant-to-Prosumer Scaling**: Raw sensor measurements reflect 22 utility-scale centralized string inverters totaling $\approx 29.15\text{ MW}$ (Plant 1) and $\approx 25.98\text{ MW}$ (Plant 2). In our simulation framework, total solar generation is downscaled to represent a 100-member residential microgrid ($N=100$, 50 prosumers each owning a $5.0\text{ kW}$ rooftop PV installation, total community PV capacity $250\text{ kW}$). While this maintains the physical cloud-passage dynamics of real solar generation, rooftop distributed installations experience higher spatial variance, localized rooftop shading, and orientation differences that are smoothed out in utility solar farms.
2. **Synthetic Demand Profiles**: Prosumer and consumer residential demand was modeled using parameterized diurnal sinusoidal and Gaussian-peak load models ($18.0\text{ kWh/day}$ per household, peak load 19:00-22:00, randomized seed $42$). Real-world smart meter residential datasets (e.g., Pecan Street or Ausgrid) would introduce higher stochastic volatility in local consumer demand.
3. **Distribution Grid Power-Flow Approximations**: The microeconomic energy sharing model enforces conservation of energy at each 15-minute time step ($Q_{\text{traded}} \le \min(\sum \text{surplus}, \sum \text{deficit})$) and applies transmission transaction costs ($c_{\text{tx}} = \$0.01/\text{kWh}$). However, AC optimal power flow (ACOPF) constraints, bus voltage limits, transformer thermal ratings, and reactive power compensation were omitted.

## 3. Machine Learning and Uncertainty Quantification
1. **Forecasting Model vs. Historical Diurnal Baseline**: While the proposed multi-scale `DiurnalResidualTCN` achieves a competitive test nMAE of $6.12\%$ ($1783.29\text{ kW}$ MAE) and beats the 24-hour persistence baseline ($6.46\%$ nMAE, $1881.66\text{ kW}$ MAE), it does not outperform the empirical Historical Diurnal Profile baseline ($5.24\%$ nMAE, $1527.04\text{ kW}$ MAE) under stable clear-sky summer conditions. Under low weather variability, repeating the historical diurnal mean is an exceptionally strong benchmark that deep learning models only match or slightly trail unless sudden severe cloud transients occur.
2. **Conformal Trajectory Coverage**: Split conformal prediction guarantees finite-sample marginal coverage ($88.70\%$ vs $90.0\%$ nominal) at each discrete 15-minute step. However, simultaneous joint trajectory coverage over the entire 96-step 24-hour horizon yields $67.85\%$, reflecting multi-step temporal correlation across the lead steps.

## 4. Game-Theoretic Assumptions
1. **Replicator Dynamics Rationality**: Replicator dynamics models boundedly rational agents adapting their strategy proportions $x(t)$ proportionally to payoff differentials. It assumes continuous well-mixed population interactions and synchronous strategy revision, whereas real residential prosumers make discrete, contractual, or automated smart-contract trading choices.
2. **Absence of Strategic Bidding Collusion**: Prosumers are assumed to share power at the regulated mid-market P2P tariff ($p_{\text{p2p}} = \$0.175/\text{kWh}$). Strategic price bidding, Bertrand oligopolistic competition, or non-cooperative coalition formation are not modeled.
