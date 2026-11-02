"""
payoffs.py
Microeconomic P2P Energy Market and Dynamic Payoff Formulation for Renewable Energy Communities.
Includes:
- Physical energy balance (kW -> kWh) with strict conservation.
- Supply-Demand Ratio (SDR) dynamic mid-market P2P pricing.
- Risk-adjusted strategy utilities (Replicator Dynamics fitness).
- Real-time physical delivery and grid imbalance settlement under actual realized solar generation.
- Deterministic residential load generation with household heterogeneity.
"""

import numpy as np
import pandas as pd


class ProsumerCommunity:
    """
    Simulates a heterogeneous prosumer community of N households with
    solar allocations and diurnal residential demand profiles.
    """
    def __init__(self, num_prosumers=100, plant_capacity_kw=29150.0,
                 p_grid=0.22, p_fit=0.06, c_tx=0.01, p_imbalance_penalty_pct=0.25,
                 seed=42):
        self.num_prosumers = num_prosumers
        self.p_grid = p_grid                                # Grid import retail tariff ($/kWh)
        self.p_fit = p_fit                                  # Feed-in tariff ($/kWh)
        self.c_tx = c_tx                                    # Platform transaction fee ($/kWh)
        self.p_imb = p_grid * (1.0 + p_imbalance_penalty_pct) # Real-time imbalance grid tariff ($/kWh)
        self.plant_capacity = plant_capacity_kw

        # Scaled community capacity: ~500 kW peak community solar (~5 kW per household)
        self.comm_scale_factor = 500.0 / plant_capacity_kw

        # Heterogeneous solar share allocations (Dirichlet distribution with fixed seed)
        rng = np.random.RandomState(seed)
        alpha = rng.uniform(0.6, 1.4, size=num_prosumers)
        self.solar_weights = alpha / np.sum(alpha)

        # Heterogeneous base demand multipliers (mean ~1.0, range 0.6 to 1.5)
        self.demand_multipliers = rng.uniform(0.6, 1.5, size=num_prosumers)

    def generate_diurnal_load(self, timestamps, seed=42):
        """
        Synthesizes realistic 15-minute residential load profile (kW)
        with morning peak (7:00-9:00), evening peak (18:00-21:00), and base load.
        Uses deterministic seed for rigorous, fair multi-model comparison.
        """
        rng = np.random.RandomState(seed)
        hours = np.array([ts.hour + ts.minute / 60.0 for ts in timestamps])
        morning_peak = 1.2 * np.exp(-((hours - 8.0) ** 2) / (2 * (1.5 ** 2)))
        evening_peak = 2.2 * np.exp(-((hours - 19.5) ** 2) / (2 * (2.0 ** 2)))
        base_load = 0.8 + 0.3 * np.sin(np.pi * (hours - 6.0) / 12.0)
        base_curve = base_load + morning_peak + evening_peak  # Shape: (T,)

        loads = np.outer(base_curve, self.demand_multipliers)
        noise = rng.normal(0, 0.05, size=loads.shape)
        return np.clip(loads + noise, 0.3, None)

    def calculate_step_payoffs(self, gen_kw, sigma_kw, loads_kw, x_share,
                               lambda_s=2.5, lambda_h=0.0, dt_hours=0.25):
        """
        Calculates Stage 1 expected payoffs U_S and U_H for Replicator Dynamics:
        - gen_kw: forecast solar generation (kW)
        - sigma_kw: forecast uncertainty (predictive std in kW)
        - loads_kw: prosumer load vector (kW) [N]
        - x_share: fraction choosing Share in [0, 1]
        - lambda_s: risk penalty sensitivity for Sharers
        - lambda_h: risk penalty sensitivity for Hoarders
        - dt_hours: interval duration (0.25 h for 15-min)
        """
        comm_solar = gen_kw * self.comm_scale_factor
        solar_i = comm_solar * self.solar_weights  # shape (N,)
        load_i = loads_kw                          # shape (N,)

        # Energy quantities (kWh)
        e_solar_i = solar_i * dt_hours
        e_load_i = load_i * dt_hours
        net_i = e_solar_i - e_load_i

        surplus_i = np.maximum(0.0, net_i)
        deficit_i = np.maximum(0.0, -net_i)

        total_surplus = float(np.sum(surplus_i))
        total_deficit = float(np.sum(deficit_i))

        # P2P trading market quantities available from Sharers
        p2p_surplus_avail = x_share * total_surplus
        p2p_deficit_avail = x_share * total_deficit

        # Supply-Demand Ratio (SDR)
        eps = 1e-6
        sdr = p2p_surplus_avail / (p2p_deficit_avail + eps)

        # Dynamic P2P clearing price: mid-market SDR formula
        price_spread = self.p_grid - self.p_fit
        sdr_factor = 1.0 - (sdr / (1.0 + sdr))
        p_p2p_raw = self.p_fit + price_spread * sdr_factor
        p_p2p = float(np.clip(p_p2p_raw, self.p_fit + self.c_tx, self.p_grid - self.c_tx))

        # Traded P2P energy volume
        q_p2p = min(p2p_surplus_avail, p2p_deficit_avail)

        # Fulfillment ratios
        rho_s = q_p2p / (p2p_surplus_avail + eps) if p2p_surplus_avail > 0 else 0.0
        rho_d = q_p2p / (p2p_deficit_avail + eps) if p2p_deficit_avail > 0 else 0.0

        # --- Payoff for HOARD Strategy ---
        pi_h_i = surplus_i * self.p_fit - deficit_i * self.p_grid
        pi_h = float(np.mean(pi_h_i))

        # --- Payoff for SHARE Strategy ---
        rev_s_i = (rho_s * surplus_i) * (p_p2p - self.c_tx) + ((1.0 - rho_s) * surplus_i) * self.p_fit
        cost_s_i = (rho_d * deficit_i) * p_p2p + ((1.0 - rho_d) * deficit_i) * self.p_grid
        pi_s_i = rev_s_i - cost_s_i
        pi_s = float(np.mean(pi_s_i))

        # --- Risk Penalty Term ---
        # Normalized uncertainty measure R_t = (sigma_comm / Comm_Peak)^2
        sigma_comm = sigma_kw * self.comm_scale_factor
        r_t = float((sigma_comm / 500.0) ** 2)

        u_s = pi_s - lambda_s * r_t
        u_h = pi_h - lambda_h * r_t

        grid_import_kwh = (1.0 - x_share) * total_deficit + (p2p_deficit_avail - q_p2p)
        total_demand_kwh = float(np.sum(e_load_i))
        grid_dep_ratio = float(grid_import_kwh / (total_demand_kwh + eps))

        return u_s, u_h, {
            'p_p2p': p_p2p,
            'sdr': sdr,
            'q_p2p_kwh': q_p2p,
            'grid_import_kwh': grid_import_kwh,
            'grid_dep_ratio': grid_dep_ratio,
            'pi_s': pi_s,
            'pi_h': pi_h,
            'r_t': r_t,
            'u_s': u_s,
            'u_h': u_h
        }

    def evaluate_realized_step(self, gen_act_kw, q_p2p_comm, x_share, loads_kw, p_p2p, dt_hours=0.25):
        """
        Stage 2 Realized physical settlement under actual solar generation:
        - Evaluates physical delivery against committed P2P trade q_p2p_comm.
        - If actual solar generation falls short of committed export, prosumers must procure
          the shortfall from the grid at the imbalance penalty tariff p_imb ($/kWh).
        - Computes genuine realized community cost and savings.
        """
        comm_solar = gen_act_kw * self.comm_scale_factor
        solar_i = comm_solar * self.solar_weights
        load_i = loads_kw

        e_solar_i = solar_i * dt_hours
        e_load_i = load_i * dt_hours
        net_i = e_solar_i - e_load_i

        surplus_i = np.maximum(0.0, net_i)
        deficit_i = np.maximum(0.0, -net_i)

        total_surplus_act = float(np.sum(surplus_i))
        total_deficit_act = float(np.sum(deficit_i))

        # Realized P2P capability
        sharer_surplus_act = x_share * total_surplus_act
        sharer_deficit_act = x_share * total_deficit_act

        # Feasible physical delivery: cannot exceed actual available surplus or deficit
        q_p2p_realized = min(q_p2p_comm, sharer_surplus_act, sharer_deficit_act)
        q_shortfall = max(0.0, q_p2p_comm - q_p2p_realized)

        # Realized trading financial settlement:
        # P2P revenue and cost:
        p2p_rev = q_p2p_realized * (p_p2p - self.c_tx)
        p2p_cost = q_p2p_realized * p_p2p

        # Remaining physical surplus sold to grid at p_fit
        grid_export_kwh = max(0.0, total_surplus_act - q_p2p_realized)
        grid_export_rev = grid_export_kwh * self.p_fit

        # Remaining deficit imported from grid at p_grid
        grid_import_kwh = max(0.0, total_deficit_act - q_p2p_realized)
        grid_import_cost = grid_import_kwh * self.p_grid

        # Imbalance penalty for shortfall
        imbalance_cost = q_shortfall * (self.p_imb - self.p_grid)

        # Total community cost under realized physical conditions:
        realized_community_cost = (grid_import_cost + imbalance_cost + p2p_cost) - (p2p_rev + grid_export_rev)

        # Autarky / No-P2P baseline under actual generation (x=0):
        baseline_cost = total_deficit_act * self.p_grid - total_surplus_act * self.p_fit

        realized_savings = baseline_cost - realized_community_cost

        return {
            'q_p2p_realized_kwh': q_p2p_realized,
            'q_shortfall_kwh': q_shortfall,
            'imbalance_cost ($)': imbalance_cost,
            'realized_cost ($)': realized_community_cost,
            'realized_baseline_cost ($)': baseline_cost,
            'realized_savings ($)': realized_savings
        }
