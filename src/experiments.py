"""
experiments.py
Comprehensive experimental suite for Forecast-Aware Evolutionary Game Theory:
- Controlled model comparisons (B0: Persistence, B1: Static Diurnal, B2: CNN Dynamic, B3: Proposed Forecast-Aware).
- Dual evaluation: Expected market dynamics and Realized physical delivery with grid imbalance settlement.
- Fine-grained uncertainty sensitivity and tipping-point bifurcation sweeps across multiple initial states.
- Severe weather shock resilience with formal recovery time quantification.
- Publication-quality 300 DPI figures and summary tables.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def set_plot_style():
    """Sets clean academic plotting styles with seaborn."""
    sns.set_theme(style="whitegrid", font_scale=1.1)
    plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
    plt.rcParams['axes.edgecolor'] = '#333333'
    plt.rcParams['axes.linewidth'] = 0.8


def run_comparative_experiment(community, test_dates, y_true, y_cnn, sigma_cnn,
                               y_persist, y_diurnal_static, x0=0.5, lambda_s=2.5):
    """
    Runs controlled comparison between models B0, B1, B2, and B3:
    - B0: Persistence Baseline Forecast (Lag-96)
    - B1: Static Diurnal Forecast Baseline (Historical mean profile per 15-min slot, 0 at night)
    - B2: 1D-CNN Dynamic Payoff without Uncertainty Penalty (lambda=0)
    - B3: Proposed 1D-CNN + Uncertainty-Aware Dynamic Payoffs (lambda=lambda_s)
    Evaluates both Stage 1 expected market dynamics and Stage 2 realized physical settlement.
    """
    from .replicator import ReplicatorEngine

    engine = ReplicatorEngine(step_size=0.1)
    loads = community.generate_diurnal_load(test_dates, seed=42)

    gen_actual = y_true[:, 0]
    gen_cnn = y_cnn[:, 0]
    sigma = sigma_cnn[:, 0]
    gen_persist = y_persist[:, 0]
    gen_static = y_diurnal_static[:, 0]
    zero_sigma = np.zeros_like(sigma)

    models = {}

    # B0: Persistence baseline
    df_b0, sum_b0 = engine.simulate(
        community, gen_series=gen_persist, sigma_series=zero_sigma,
        load_matrix=loads, timestamps=test_dates, x0=x0, lambda_s=0.0,
        gen_actual_series=gen_actual
    )
    models['B0 (Persistence Baseline)'] = (df_b0, sum_b0)

    # B1: Static Diurnal Game Baseline (physically 0 at night)
    df_b1, sum_b1 = engine.simulate(
        community, gen_series=gen_static, sigma_series=zero_sigma,
        load_matrix=loads, timestamps=test_dates, x0=x0, lambda_s=0.0,
        gen_actual_series=gen_actual
    )
    models['B1 (Static Diurnal Game)'] = (df_b1, sum_b1)

    # B2: 1D-CNN + Dynamic Payoff (No Uncertainty Penalty)
    df_b2, sum_b2 = engine.simulate(
        community, gen_series=gen_cnn, sigma_series=zero_sigma,
        load_matrix=loads, timestamps=test_dates, x0=x0, lambda_s=0.0,
        gen_actual_series=gen_actual
    )
    models['B2 (CNN Dynamic, Risk-Neutral)'] = (df_b2, sum_b2)

    # B3: Proposed 1D-CNN + Uncertainty-Aware Dynamic Payoff
    df_b3, sum_b3 = engine.simulate(
        community, gen_series=gen_cnn, sigma_series=sigma,
        load_matrix=loads, timestamps=test_dates, x0=x0, lambda_s=lambda_s,
        gen_actual_series=gen_actual
    )
    models['B3 (Proposed Forecast-Aware Game)'] = (df_b3, sum_b3)

    return models


def run_uncertainty_sweep(community, test_dates, gen_series, sigma_series,
                          loads, lambda_s=2.5,
                          multipliers=[0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5, 4.0],
                          initial_shares=[0.20, 0.50, 0.80]):
    """
    Sweeps uncertainty scaling factor across fine grid and multiple initial states x0
    to distinguish gradual continuous sensitivity from discrete bifurcation.
    """
    from .replicator import ReplicatorEngine
    engine = ReplicatorEngine(step_size=0.1)

    records = []
    trajectories = {}

    for x0 in initial_shares:
        trajectories[x0] = {}
        for mult in multipliers:
            scaled_sigma = sigma_series * mult
            df_sim, summary = engine.simulate(
                community, gen_series=gen_series, sigma_series=scaled_sigma,
                load_matrix=loads, timestamps=test_dates, x0=x0, lambda_s=lambda_s
            )
            norm_unc_pct = (np.mean(scaled_sigma) / community.plant_capacity) * 100.0
            records.append({
                'Initial_Share': x0,
                'Uncertainty_Multiplier': mult,
                'Uncertainty_Scale_Pct': norm_unc_pct,
                'Final_Share': summary['Final_Share'],
                'Mean_Share': summary['Mean_Share'],
                'Time_Above_Threshold': summary['Time_Above_Threshold'],
                'P2P_Traded_kWh': summary['Total_P2P_Traded_kWh'],
                'Community_Savings ($)': summary['Total_Savings ($)'],
                'Savings_Pct (%)': summary['Savings_Pct (%)']
            })
            trajectories[x0][mult] = df_sim['x_share'].values

    sweep_df = pd.DataFrame(records)

    # Compute numerical derivative dx(T) / d(Uncertainty) for x0 = 0.50
    sub_df = sweep_df[sweep_df['Initial_Share'] == 0.50].sort_values('Uncertainty_Multiplier')
    d_unc = np.diff(sub_df['Uncertainty_Scale_Pct'].values)
    d_x = np.diff(sub_df['Final_Share'].values)
    gradients = d_x / (d_unc + 1e-6)

    # Max sensitivity point
    max_sens_idx = int(np.argmin(gradients))
    tipping_point_info = {
        'max_sensitivity_uncertainty_pct': float(sub_df['Uncertainty_Scale_Pct'].iloc[max_sens_idx]),
        'max_gradient': float(gradients[max_sens_idx]),
        'collapse_threshold_pct': float(sub_df[sub_df['Final_Share'] < 0.20]['Uncertainty_Scale_Pct'].min()) if np.any(sub_df['Final_Share'] < 0.20) else None
    }

    return sweep_df, trajectories, tipping_point_info


def run_weather_shock_experiment(community, test_dates, gen_series, sigma_series,
                                 loads, x0=0.85, lambda_s=2.5,
                                 shock_start_idx=150, shock_duration=24, shock_drop=0.75,
                                 step_size=2.0):
    """
    Simulates a sudden severe weather shock (e.g. cloudburst / sudden storm):
    Generation drops by 75% and forecast uncertainty surges by 3x for 24 steps (6 hours).
    Measures behavioral withdrawal, lost trading, and formally quantifies recovery time.
    """
    from .replicator import ReplicatorEngine
    engine = ReplicatorEngine(step_size=step_size)

    shock_gen = gen_series.copy()
    shock_sigma = sigma_series.copy()

    end_idx = min(len(gen_series), shock_start_idx + shock_duration)
    shock_gen[shock_start_idx:end_idx] *= (1.0 - shock_drop)
    shock_sigma[shock_start_idx:end_idx] *= 3.0

    # Normal simulation
    df_normal, sum_normal = engine.simulate(
        community, gen_series=gen_series, sigma_series=sigma_series,
        load_matrix=loads, timestamps=test_dates, x0=x0, lambda_s=lambda_s
    )

    # Shock simulation
    df_shock, sum_shock = engine.simulate(
        community, gen_series=shock_gen, sigma_series=shock_sigma,
        load_matrix=loads, timestamps=test_dates, x0=x0, lambda_s=lambda_s
    )

    trade_lost = sum_normal['Total_P2P_Traded_kWh'] - sum_shock['Total_P2P_Traded_kWh']
    cost_increase = sum_shock['Total_Cost ($)'] - sum_normal['Total_Cost ($)']

    pre_shock_share = float(df_shock['x_share'].iloc[shock_start_idx - 1])
    min_shock_share = float(df_shock['x_share'].iloc[shock_start_idx:end_idx].min())
    post_shock_final = float(df_shock['x_share'].iloc[-1])

    # Formal recovery criterion: interval where x(t) >= 0.95 * pre_shock_share
    post_shock_shares = df_shock['x_share'].iloc[end_idx:].values
    recovery_target = 0.95 * pre_shock_share
    recovery_indices = np.where(post_shock_shares >= recovery_target)[0]

    if len(recovery_indices) > 0:
        recovery_intervals = int(recovery_indices[0])
        recovery_hours = float(recovery_intervals * 0.25)
        recovered = True
    else:
        recovery_intervals = -1
        recovery_hours = -1.0
        recovered = False

    metrics = {
        'Shock_Start_Interval': shock_start_idx,
        'Shock_Duration_Hours': shock_duration * 0.25,
        'Pre_Shock_Share': pre_shock_share,
        'Min_Shock_Share': min_shock_share,
        'Post_Shock_Final_Share': post_shock_final,
        'Recovered_95pct': recovered,
        'Recovery_Hours': recovery_hours,
        'Trade_Lost_kWh': float(trade_lost),
        'Cost_Increase ($)': float(cost_increase)
    }

    return df_normal, df_shock, metrics, (shock_start_idx, end_idx)


# --- Plotting Functions ---

def plot_forecast_actual_vs_pred(timestamps, y_true_day, y_pred_day, lower_day, upper_day,
                                save_path=None):
    """Figure 1: Actual vs Predicted Plant AC Power with Conformal Prediction Intervals."""
    set_plot_style()
    fig, ax = plt.subplots(figsize=(12, 5))
    x_axis = np.arange(len(y_true_day))

    ax.plot(x_axis, y_true_day, label='Actual AC Power', color='#1f77b4', linewidth=2.2)
    ax.plot(x_axis, y_pred_day, label='Diurnal Residual TCN Forecast', color='#d62728', linestyle='--', linewidth=2.0)
    ax.fill_between(x_axis, lower_day, upper_day, color='#ff7f0e', alpha=0.30,
                    label='90% Conformalized Uncertainty Interval')

    ax.set_title('Figure 1: Multi-Scale 1D-CNN 24-Hour Solar AC Power Forecast with Conformal Uncertainty Band',
                 fontweight='bold', fontsize=13)
    ax.set_xlabel('15-Minute Forecast Lead Steps (0 to 95 = 24 Hours)')
    ax.set_ylabel('Plant AC Power (kW)')
    ax.legend(loc='upper right', frameon=True)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300)
    return fig


def plot_error_by_horizon(horizon_dict, save_path=None):
    """Figure 2: Forecast Error by Lead Time (15m, 1h, 6h, 12h, 24h)."""
    set_plot_style()
    fig, ax = plt.subplots(figsize=(8, 4.5))

    names = list(horizon_dict.keys())
    values = list(horizon_dict.values())

    bars = ax.bar(names, values, color='#2ca02c', alpha=0.85, edgecolor='black', width=0.55)
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f'{h:.1f} kW',
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 4), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_title('Figure 2: Forecast Error (MAE) Across Forecasting Horizons', fontweight='bold', fontsize=13)
    ax.set_ylabel('Mean Absolute Error (kW)')
    ax.set_ylim(0, max(values) * 1.25)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300)
    return fig


def plot_uncertainty_calibration(calib_results, save_path=None):
    """Figure 3: Prediction Interval Calibration (Empirical PICP vs Nominal Coverage)."""
    set_plot_style()
    fig, ax = plt.subplots(figsize=(7, 5))

    nominal_levels = [50, 60, 70, 80, 90, 95]
    picp_levels = [calib_results.get(lvl, lvl) for lvl in nominal_levels]

    ax.plot([50, 95], [50, 95], 'k--', label='Ideal Calibration Line (y = x)', linewidth=1.8)
    ax.plot(nominal_levels, picp_levels, 'o-', color='#e377c2',
            linewidth=2.5, markersize=8, label='Empirical Coverage (Split Conformal)')

    ax.set_title('Figure 3: Conformal Prediction Interval Calibration Curve (PICP vs Nominal)',
                 fontweight='bold', fontsize=13)
    ax.set_xlabel('Nominal Confidence Level (%)')
    ax.set_ylabel('Empirical Coverage PICP (%)')
    ax.legend(loc='lower right', frameon=True)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300)
    return fig


def plot_comparative_trajectories(models_dict, save_path=None):
    """Figure 4: Comparative Evolutionary Trajectories x(t) for B0, B1, B2, and B3."""
    set_plot_style()
    fig, ax = plt.subplots(figsize=(12, 5.5))

    colors = {
        'B0 (Persistence Baseline)': '#7f7f7f',
        'B1 (Static Diurnal Game)': '#1f77b4',
        'B2 (CNN Dynamic, Risk-Neutral)': '#2ca02c',
        'B3 (Proposed Forecast-Aware Game)': '#d62728'
    }

    for name, (df_sim, _) in models_dict.items():
        time_hours = np.arange(len(df_sim)) * 0.25
        ax.plot(time_hours, df_sim['x_share'], label=name, color=colors.get(name, 'black'),
                linewidth=2.2, alpha=0.9)

    ax.axhline(0.80, color='gray', linestyle=':', label='High-Cooperation Threshold (80%)')
    ax.set_title('Figure 4: Prosumer Strategy Evolution x(t) Under Controlled Model Ablations',
                 fontweight='bold', fontsize=13)
    ax.set_xlabel('Simulation Time (Hours)')
    ax.set_ylabel('Population Fraction Choosing "Share" x(t)')
    ax.set_ylim(-0.02, 1.05)
    ax.legend(loc='lower left', frameon=True)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300)
    return fig


def plot_tipping_point(sweep_df, save_path=None):
    """Figure 5: Strategy Adoption vs Uncertainty Level across Multiple Initial States."""
    set_plot_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    colors = {0.20: '#2ca02c', 0.50: '#9467bd', 0.80: '#d62728'}
    for x0, grp in sweep_df.groupby('Initial_Share'):
        ax1.plot(grp['Uncertainty_Scale_Pct'], grp['Final_Share'], 'o-',
                 label=f'x0 = {x0:.2f}', color=colors.get(x0, 'black'), linewidth=2.2, markersize=6)

    ax1.axhline(0.20, color='red', linestyle=':', label='Collapse Threshold (20%)')
    ax1.set_title('(A) Equilibrium Adoption x(T) vs Uncertainty', fontweight='bold', fontsize=12)
    ax1.set_xlabel('Normalized Forecast Uncertainty (%)')
    ax1.set_ylabel('Final Share Adoption x(T)')
    ax1.set_ylim(-0.05, 1.05)
    ax1.legend(loc='upper right', frameon=True)

    # Community savings for x0=0.50
    sub_50 = sweep_df[sweep_df['Initial_Share'] == 0.50]
    ax2.plot(sub_50['Uncertainty_Scale_Pct'], sub_50['Community_Savings ($)'], 's-',
             color='#8c564b', linewidth=2.5, markersize=7)
    ax2.set_title('(B) Community Cumulative Savings vs Uncertainty (x0 = 0.50)', fontweight='bold', fontsize=12)
    ax2.set_xlabel('Normalized Forecast Uncertainty (%)')
    ax2.set_ylabel('Cumulative P2P Savings ($)')

    plt.suptitle('Figure 5: Uncertainty Sensitivity & Critical Transition Analysis',
                 fontweight='bold', fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300)
    return fig


def plot_weather_shock_response(df_normal, df_shock, shock_range, save_path=None):
    """Figure 6: Weather Shock Response and Behavioral Recovery."""
    set_plot_style()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    time_hours = np.arange(len(df_shock)) * 0.25
    start_h = shock_range[0] * 0.25
    end_h = shock_range[1] * 0.25

    # Share trajectory
    ax1.plot(time_hours, df_normal['x_share'], label='Baseline Condition (No Shock)',
             color='#2ca02c', linestyle='--', linewidth=2.0)
    ax1.plot(time_hours, df_shock['x_share'], label='Severe Weather Shock Response',
             color='#d62728', linewidth=2.5)
    ax1.axvspan(start_h, end_h, color='red', alpha=0.15, label='Shock Duration (6h Cloudburst & Uncertainty Surge)')
    ax1.set_ylabel('Share Adoption x(t)')
    ax1.set_title('Figure 6: Community Behavioral Resilience and Strategy Dynamics During Weather Shock',
                  fontweight='bold', fontsize=13)
    ax1.legend(loc='lower left', frameon=True)
    ax1.set_ylim(-0.02, 1.05)

    # P2P traded energy
    ax2.plot(time_hours, df_normal['Q_P2P_kWh'], label='Normal Traded Energy',
             color='#1f77b4', linestyle='--', linewidth=1.8)
    ax2.plot(time_hours, df_shock['Q_P2P_kWh'], label='Traded Under Shock',
             color='#ff7f0e', linewidth=2.0)
    ax2.axvspan(start_h, end_h, color='red', alpha=0.15)
    ax2.set_ylabel('P2P Traded Energy (kWh)')
    ax2.set_xlabel('Simulation Time (Hours)')
    ax2.legend(loc='upper right', frameon=True)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300)
    return fig
