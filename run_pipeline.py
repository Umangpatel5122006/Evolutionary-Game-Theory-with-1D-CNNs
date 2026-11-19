"""
run_pipeline.py
Master execution script running the full research-grade end-to-end pipeline:
1. Data Ingestion & Preprocessing with DC/AC unit harmonization (Plant 1 & Plant 2)
2. Chronological Splitting (train/val/test) & Non-overlapping window creation
3. Multi-Scale Dilated 1D-CNN (DiurnalResidualTCN) training with multi-seed evaluation
4. Comprehensive Baseline Evaluation (Persistence, Seasonal Naive, Historical Diurnal, 1D-CNN)
5. Split Conformal Uncertainty Quantification (PICP, MPIW, ACE, Winkler Score, Horizon Breakdown)
6. Controlled Evolutionary Game Ablations (B0, B1, B2, B3) with Realized Imbalance Settlement
7. Fine-Grained Uncertainty Sensitivity Sweep & Transition Analysis
8. Weather Shock Resilience with Formal Recovery Time Quantification
9. Genuine Zero-Shot Transfer Evaluation on Plant 2
10. Export of Publication-Quality 300 DPI Figures and Research CSV Tables
"""

import os
import sys
import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd

from src.preprocess import (load_and_clean_plant_data, prepare_chronological_splits,
                            prepare_genuine_zero_shot_transfer, get_astronomical_night_mask)
from src.train_cnn import (DiurnalResidualTCN, train_model, compute_forecast_metrics,
                           persistence_baseline, seasonal_naive_baseline, historical_diurnal_profile_baseline)
from src.uncertainty import (predict_mc_dropout, compute_prediction_intervals,
                             compute_split_conformal_intervals, evaluate_uncertainty_calibration)
from src.payoffs import ProsumerCommunity
from src.replicator import ReplicatorEngine
from src.experiments import (run_comparative_experiment, run_uncertainty_sweep,
                             run_weather_shock_experiment, plot_forecast_actual_vs_pred,
                             plot_error_by_horizon, plot_uncertainty_calibration,
                             plot_comparative_trajectories, plot_tipping_point,
                             plot_weather_shock_response)


def main():
    print("=" * 80)
    print("RESEARCH-GRADE FORECAST-AWARE EVOLUTIONARY GAME THEORY PIPELINE")
    print("Multi-Scale 1D-CNN + Split Conformal UQ + Dynamic Replicator Dynamics")
    print("=" * 80)

    # Ensure output directories exist
    os.makedirs('results/tables', exist_ok=True)
    os.makedirs('results/figures', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Execution Device: {device}")

    # Paths
    p1_gen_path = 'Plant_1_Generation_Data.csv'
    p1_weather_path = 'Plant_1_Weather_Sensor_Data.csv'
    p2_gen_path = 'Plant_2_Generation_Data.csv'
    p2_weather_path = 'Plant_2_Weather_Sensor_Data.csv'

    # --- STEP 1: Data Ingestion & Preprocessing (Plant 1) ---
    print("\n--- STEP 1: Data Ingestion & Preprocessing (Plant 1) ---")
    df_p1 = load_and_clean_plant_data(p1_gen_path, p1_weather_path, is_plant1=True)
    print(f"Plant 1 continuous series: {len(df_p1)} intervals ({df_p1['DATE_TIME'].min()} to {df_p1['DATE_TIME'].max()})")
    df_p1.to_csv('data/processed/plant_1_cleaned.csv', index=False)

    feature_cols = ['AC_POWER', 'DC_POWER', 'AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION',
                    'sin_tod', 'cos_tod', 'sin_dow', 'cos_dow']
    ac_idx = feature_cols.index('AC_POWER')

    # --- STEP 2: Chronological Splitting & Windowing ---
    print("\n--- STEP 2: Chronological Splitting & Windowing (192 in -> 96 out) ---")
    data_dict = prepare_chronological_splits(df_p1, feature_cols, history_len=192, horizon_len=96)

    X_train, y_train, dates_train = data_dict['train']
    X_val, y_val, dates_val = data_dict['val']
    X_test, y_test, dates_test = data_dict['test']
    X_test_no, y_test_no, dates_test_no = data_dict['test_nonoverlap']
    scalers = data_dict['scalers']
    rated_capacity = scalers['rated_capacity']

    print(f"Train samples: {len(X_train)} | Val samples: {len(X_val)} | Test sliding: {len(X_test)} | Test non-overlapping: {len(X_test_no)}")
    print(f"Plant Peak Observed AC Capacity: {rated_capacity:.2f} kW ({rated_capacity / 1000.0:.2f} MW)")

    # Unscaled targets
    y_train_unscaled = y_train * scalers['target_std'] + scalers['target_mean']
    y_val_unscaled = y_val * scalers['target_std'] + scalers['target_mean']
    y_test_unscaled = y_test * scalers['target_std'] + scalers['target_mean']
    y_test_no_unscaled = y_test_no * scalers['target_std'] + scalers['target_mean']

    # DataLoaders
    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    test_tensor_X = torch.tensor(X_test, dtype=torch.float32)

    # --- STEP 3: Multi-Seed Training of Diurnal Residual TCN ---
    print("\n--- STEP 3: Training Diurnal Residual TCN with Multi-Seed Evaluation ---")
    seeds = [42, 123, 456]
    seed_models = []
    seed_metrics = []
    val_tensor_X = torch.tensor(X_val, dtype=torch.float32)

    for seed in seeds:
        print(f"\nTraining with Seed {seed}...")
        torch.manual_seed(seed)
        np.random.seed(seed)
        m = DiurnalResidualTCN(in_channels=len(feature_cols), horizon=96, dropout_rate=0.15, ac_idx=ac_idx)
        m, _ = train_model(m, train_loader, val_loader, epochs=25, lr=1e-3, device=device, patience=4)

        # STRICT VALIDATION-BASED CHECKPOINT EVALUATION (NO TEST LEAKAGE)
        m.eval()
        with torch.no_grad():
            val_pred_s = m(val_tensor_X.to(device)).cpu().numpy()
        val_pred_unscaled = np.clip(val_pred_s * scalers['target_std'] + scalers['target_mean'], 0.0, None)
        night_mask_val = get_astronomical_night_mask(dates_val)
        val_pred_unscaled[night_mask_val] = 0.0
        val_mae = float(np.mean(np.abs(y_val_unscaled - val_pred_unscaled)))

        with torch.no_grad():
            pred_s = m(test_tensor_X.to(device)).cpu().numpy()
        pred_unscaled = np.clip(pred_s * scalers['target_std'] + scalers['target_mean'], 0.0, None)
        # Astronomical night zeroing
        night_mask = get_astronomical_night_mask(dates_test)
        pred_unscaled[night_mask] = 0.0

        met_s = compute_forecast_metrics(y_test_unscaled, pred_unscaled, rated_capacity, name=f"Seed {seed}")
        seed_metrics.append(met_s)
        seed_models.append((m, val_mae, met_s['MAE (kW)']))
        print(f"Seed {seed} -> Validation MAE: {val_mae:.2f} kW | Test MAE: {met_s['MAE (kW)']:.2f} kW")

    # Select best checkpoint strictly by Validation MAE (zero test-set leakage)
    best_tuple = sorted(seed_models, key=lambda x: x[1])[0]
    best_model = best_tuple[0]
    print(f"\nSelected best checkpoint by Validation MAE: {best_tuple[1]:.2f} kW (Corresponding Test MAE: {best_tuple[2]:.2f} kW)")
    torch.save(best_model.state_dict(), 'data/processed/cnn_model_weights.pt')
    print("Best model checkpoint saved to data/processed/cnn_model_weights.pt")

    mae_vals = [m['MAE (kW)'] for m in seed_metrics]
    rmse_vals = [m['RMSE (kW)'] for m in seed_metrics]
    nmae_vals = [m['nMAE (%)'] for m in seed_metrics]
    print(f"Multi-Seed Summary across {len(seeds)} runs:")
    print(f"  Test MAE:  {np.mean(mae_vals):.2f} +/- {np.std(mae_vals):.2f} kW")
    print(f"  Test RMSE: {np.mean(rmse_vals):.2f} +/- {np.std(rmse_vals):.2f} kW")
    print(f"  Test nMAE: {np.mean(nmae_vals):.2f}% +/- {np.std(nmae_vals):.2f}%")

    # --- STEP 4: Fair Baseline Forecasting Comparison ---
    print("\n--- STEP 4: Fair Baseline Forecasting Comparison ---")
    # Baselines on sliding test set
    y_persist = persistence_baseline(X_test, scalers['target_scaler'], feature_cols)
    y_snaive = seasonal_naive_baseline(X_test, scalers['target_scaler'], feature_cols)
    y_diurnal = historical_diurnal_profile_baseline(y_train_unscaled, dates_train, dates_test)

    best_model.eval()
    with torch.no_grad():
        y_cnn_scaled = best_model(test_tensor_X.to(device)).cpu().numpy()
    y_cnn = np.clip(y_cnn_scaled * scalers['target_std'] + scalers['target_mean'], 0.0, None)
    night_mask_test = get_astronomical_night_mask(dates_test)
    y_cnn[night_mask_test] = 0.0

    met_b0 = compute_forecast_metrics(y_test_unscaled, y_persist, rated_capacity, name="B0: Persistence Baseline (Lag-96)")
    met_snaive = compute_forecast_metrics(y_test_unscaled, y_snaive, rated_capacity, name="Seasonal Naive (Lag 96+192)")
    met_diurnal = compute_forecast_metrics(y_test_unscaled, y_diurnal, rated_capacity, name="Historical Diurnal Profile")
    met_cnn = compute_forecast_metrics(y_test_unscaled, y_cnn, rated_capacity, name="Proposed Multi-Scale 1D-CNN (TCN)", dates=dates_test)

    # Non-overlapping evaluation for rigorous statistical independence
    test_tensor_no = torch.tensor(X_test_no, dtype=torch.float32)
    y_persist_no = persistence_baseline(X_test_no, scalers['target_scaler'], feature_cols)
    with torch.no_grad():
        y_cnn_no_scaled = best_model(test_tensor_no.to(device)).cpu().numpy()
    y_cnn_no = np.clip(y_cnn_no_scaled * scalers['target_std'] + scalers['target_mean'], 0.0, None)
    y_cnn_no[get_astronomical_night_mask(dates_test_no)] = 0.0

    met_persist_no = compute_forecast_metrics(y_test_no_unscaled, y_persist_no, rated_capacity, name="Persistence (Non-Overlapping Origins)")
    met_cnn_no = compute_forecast_metrics(y_test_no_unscaled, y_cnn_no, rated_capacity, name="Proposed 1D-CNN (Non-Overlapping Origins)")

    display_cols = ['Model', 'MAE (kW)', 'RMSE (kW)', 'nMAE (%)', 'nRMSE (%)', 'Daylight MAE (kW)', 'Daylight RMSE (kW)']
    metrics_df = pd.DataFrame([met_b0, met_snaive, met_diurnal, met_cnn, met_persist_no, met_cnn_no])[display_cols]
    print("\nForecast Performance Comparison Table:")
    print(metrics_df.to_string(index=False))
    metrics_df.to_csv('results/tables/table_1_forecast_metrics.csv', index=False)

    plot_error_by_horizon(met_cnn['Horizon_MAE'], save_path='results/figures/figure_2_horizon_error.png')
    print("Saved Figure 2: Horizon error breakdown")

    # --- STEP 5: Split Conformal Uncertainty Quantification ---
    print("\n--- STEP 5: Split Conformal Prediction & Uncertainty Quantification ---")
    val_mean, val_std, _ = predict_mc_dropout(best_model, val_tensor_X, scalers['target_scaler'], K=30, device=device, dates=dates_val)
    mean_pred, std_pred, all_runs = predict_mc_dropout(best_model, test_tensor_X, scalers['target_scaler'], K=50, device=device, dates=dates_test)

    # 1. Marginal Pooled Split Conformal Calibration
    lower_90_pool, upper_90_pool, q_hat_pool = compute_split_conformal_intervals(
        y_val_unscaled, val_mean, val_std, mean_pred, std_pred, nominal_coverage=0.90, dates_test=dates_test, method='pooled'
    )
    eval_unc_pool = evaluate_uncertainty_calibration(y_test_unscaled, lower_90_pool, upper_90_pool, nominal_coverage=0.90)

    # 2. Trajectory-wise (Joint 24h Horizon) Split Conformal Calibration
    lower_90_traj, upper_90_traj, q_hat_traj = compute_split_conformal_intervals(
        y_val_unscaled, val_mean, val_std, mean_pred, std_pred, nominal_coverage=0.90, dates_test=dates_test, method='trajectory'
    )
    eval_unc_traj = evaluate_uncertainty_calibration(y_test_unscaled, lower_90_traj, upper_90_traj, nominal_coverage=0.90)
    # Check simultaneous trajectory coverage
    traj_covered = np.all((y_test_unscaled >= lower_90_traj) & (y_test_unscaled <= upper_90_traj), axis=1)
    joint_traj_picp = float(np.mean(traj_covered) * 100.0)

    print(f"Marginal Pooled Conformal (90%): PICP = {eval_unc_pool['PICP (%)']:.2f}% | MPIW = {eval_unc_pool['MPIW (kW)']:.2f} kW | q = {q_hat_pool:.3f}")
    print(f"Trajectory Joint Conformal (90%): Joint Traj PICP = {joint_traj_picp:.2f}% | Point PICP = {eval_unc_traj['PICP (%)']:.2f}% | MPIW = {eval_unc_traj['MPIW (kW)']:.2f} kW | q = {q_hat_traj:.3f}")

    # Use marginal pooled for standard point-step trading payoffs, while documenting trajectory performance
    lower_90 = lower_90_pool
    upper_90 = upper_90_pool
    q_hat = q_hat_pool

    unc_df = pd.DataFrame([
        {
            'Conformal Method': 'Split Conformal (Pooled Marginal)',
            'Nominal Coverage (%)': 90.0,
            'Empirical PICP (%)': eval_unc_pool['PICP (%)'],
            'MPIW (kW)': eval_unc_pool['MPIW (kW)'],
            'ACE (%)': eval_unc_pool['ACE (%)'],
            'Winkler_Score': eval_unc_pool['Winkler_Score'],
            'Conformal_Quantile': q_hat_pool
        },
        {
            'Conformal Method': 'Split Conformal (Joint Trajectory)',
            'Nominal Coverage (%)': 90.0,
            'Empirical PICP (%)': eval_unc_traj['PICP (%)'],
            'MPIW (kW)': eval_unc_traj['MPIW (kW)'],
            'ACE (%)': eval_unc_traj['ACE (%)'],
            'Winkler_Score': eval_unc_traj['Winkler_Score'],
            'Conformal_Quantile': q_hat_traj
        }
    ])
    unc_df.to_csv('results/tables/table_2_uncertainty_calibration.csv', index=False)

    # Calibration across nominal confidence levels [50, 60, 70, 80, 90, 95] for Figure 3
    calib_curve = {}
    for nom in [50, 60, 70, 80, 90, 95]:
        low, upp, _ = compute_split_conformal_intervals(
            y_val_unscaled, val_mean, val_std, mean_pred, std_pred, nominal_coverage=(nom / 100.0), dates_test=dates_test
        )
        cov = float(np.mean((y_test_unscaled >= low) & (y_test_unscaled <= upp)) * 100.0)
        calib_curve[nom] = cov
    plot_uncertainty_calibration(calib_curve, save_path='results/figures/figure_3_uncertainty_calibration.png')
    print("Saved Figure 3: Uncertainty calibration curve")

    # Figure 1: Representative 24-hour test day with calibrated conformal band
    sample_day_idx = min(96, len(y_test_unscaled) - 1)
    plot_forecast_actual_vs_pred(
        dates_test[sample_day_idx],
        y_test_unscaled[sample_day_idx],
        mean_pred[sample_day_idx],
        lower_90[sample_day_idx],
        upper_90[sample_day_idx],
        save_path='results/figures/figure_1_actual_vs_predicted.png'
    )
    print("Saved Figure 1: Actual vs Predicted with Uncertainty Band")

    # --- STEP 6: Controlled Evolutionary Game Ablations ---
    print("\n--- STEP 6: Controlled Evolutionary Game Ablations (B0, B1, B2, B3) ---")
    community = ProsumerCommunity(num_prosumers=100, plant_capacity_kw=rated_capacity, p_grid=0.22, p_fit=0.06, c_tx=0.01)
    test_dates_flat = pd.to_datetime(dates_test[:, 0])

    models_dict = run_comparative_experiment(
        community,
        test_dates=test_dates_flat,
        y_true=y_test_unscaled,
        y_cnn=mean_pred,
        sigma_cnn=std_pred,
        y_persist=y_persist,
        y_diurnal_static=y_diurnal,
        x0=0.50,
        lambda_s=2.5
    )

    summary_rows = []
    for name, (df_sim, summary) in models_dict.items():
        summary['Model'] = name
        summary_rows.append(summary)
        clean_name = name.split(' ')[0].replace('(', '').replace(')', '')
        df_sim.to_csv(f'results/tables/simulation_{clean_name}.csv', index=False)

    summary_df = pd.DataFrame(summary_rows)
    cols_order = ['Model', 'Final_Share', 'Mean_Share', 'Time_Above_Threshold',
                  'Total_P2P_Traded_kWh', 'Total_Cost ($)', 'Total_Savings ($)', 'Savings_Pct (%)',
                  'Realized_Traded_kWh', 'Total_Shortfall_kWh', 'Total_Imbalance_Cost ($)',
                  'Realized_Community_Cost ($)', 'Realized_Savings ($)']
    print("\nEvolutionary Game Model Comparison:")
    print(summary_df[cols_order].to_string(index=False))
    summary_df[cols_order].to_csv('results/tables/table_3_model_comparison.csv', index=False)

    plot_comparative_trajectories(models_dict, save_path='results/figures/figure_4_comparative_trajectories.png')
    print("Saved Figure 4: Comparative evolutionary trajectories")

    # --- STEP 7: Sensitivity & Critical Tipping Point Sweep ---
    print("\n--- STEP 7: Sensitivity & Critical Transition Analysis ---")
    loads = community.generate_diurnal_load(test_dates_flat, seed=42)
    gen_cnn_1step = mean_pred[:, 0]
    sigma_cnn_1step = std_pred[:, 0]

    sweep_df, _, tipping_info = run_uncertainty_sweep(
        community, test_dates=test_dates_flat,
        gen_series=gen_cnn_1step, sigma_series=sigma_cnn_1step,
        loads=loads, lambda_s=2.5,
        multipliers=[0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0, 15.0],
        initial_shares=[0.20, 0.50, 0.80]
    )
    print("\nUncertainty Sweep Summary (for x0 = 0.50):")
    sub_50 = sweep_df[sweep_df['Initial_Share'] == 0.50]
    print(sub_50[['Uncertainty_Multiplier', 'Uncertainty_Scale_Pct', 'Final_Share', 'Community_Savings ($)']].to_string(index=False))
    print(f"\nOperational Transition Diagnostics:")
    print(f"  Maximum Sensitivity Point: {tipping_info['max_sensitivity_uncertainty_pct']:.2f}% Normalized Uncertainty (Peak gradient: {tipping_info['max_gradient']:.2f})")
    print(f"  Severe Collapse Threshold (<20% cooperation): {tipping_info['collapse_threshold_pct']}% Normalized Uncertainty")
    sweep_df.to_csv('results/tables/table_4_tipping_point_sweep.csv', index=False)

    plot_tipping_point(sweep_df, save_path='results/figures/figure_5_tipping_point.png')
    print("Saved Figure 5: Sensitivity and bifurcation curves")

    # --- STEP 8: Weather Shock Response ---
    print("\n--- STEP 8: Severe Weather Shock Resilience Experiment ---")
    df_normal, df_shock, shock_metrics, shock_range = run_weather_shock_experiment(
        community, test_dates=test_dates_flat,
        gen_series=gen_cnn_1step, sigma_series=sigma_cnn_1step,
        loads=loads, x0=0.85, lambda_s=2.5,
        shock_start_idx=150, shock_duration=24, shock_drop=0.75
    )
    print("\nWeather Shock Response Metrics:")
    for k, v in shock_metrics.items():
        print(f"  {k}: {v:.2f}" if isinstance(v, float) else f"  {k}: {v}")

    pd.DataFrame([shock_metrics]).to_csv('results/tables/table_5_weather_shock_metrics.csv', index=False)
    plot_weather_shock_response(df_normal, df_shock, shock_range, save_path='results/figures/figure_6_weather_shock.png')
    print("Saved Figure 6: Weather shock response and recovery dynamics")

    # --- STEP 9: Genuine Zero-Shot Transfer Test (Plant 2) ---
    print("\n--- STEP 9: Genuine Zero-Shot Cross-Plant Generalization Test (Plant 2) ---")
    df_p2 = load_and_clean_plant_data(p2_gen_path, p2_weather_path, is_plant1=False)
    print(f"Loaded Plant 2: {len(df_p2)} intervals ({df_p2['DATE_TIME'].min()} to {df_p2['DATE_TIME'].max()})")
    df_p2.to_csv('data/processed/plant_2_cleaned.csv', index=False)

    # Genuine zero-shot transfer: strictly apply Plant 1 scalers to Plant 2 features
    X_p2, y_p2, dates_p2, y_p2_actual, p2_peak_capacity = prepare_genuine_zero_shot_transfer(
        df_p2, feature_cols, source_scalers=scalers, history_len=192, horizon_len=96
    )

    test_tensor_p2 = torch.tensor(X_p2, dtype=torch.float32)
    with torch.no_grad():
        p2_pred_scaled = best_model(test_tensor_p2.to(device)).cpu().numpy()

    # Inverse transform using Plant 1 target scaler
    p2_pred_unscaled = np.clip(p2_pred_scaled * scalers['target_std'] + scalers['target_mean'], 0.0, None)
    # Astronomical night zeroing
    night_mask_p2 = get_astronomical_night_mask(dates_p2)
    p2_pred_unscaled[night_mask_p2] = 0.0

    # Evaluate zero-shot against actual Plant 2 capacity
    met_p2_raw = compute_forecast_metrics(y_p2_actual, p2_pred_unscaled, p2_peak_capacity, name="Plant 2 Genuine Zero-Shot Transfer")

    # Capacity-normalized transfer: scales prediction by Plant 2 / Plant 1 capacity ratio (25.98 MW / 29.15 MW)
    cap_ratio = p2_peak_capacity / rated_capacity
    p2_pred_adapted = p2_pred_unscaled * cap_ratio
    met_p2_adapted = compute_forecast_metrics(y_p2_actual, p2_pred_adapted, p2_peak_capacity, name="Plant 2 Capacity-Harmonized Transfer")

    # Plant 2 Persistence baseline for fair reference
    p2_persist = persistence_baseline(X_p2, scalers['target_scaler'], feature_cols)
    met_p2_persist = compute_forecast_metrics(y_p2_actual, p2_persist, p2_peak_capacity, name="Plant 2 Persistence Baseline")

    p2_summary_df = pd.DataFrame([met_p2_persist, met_p2_raw, met_p2_adapted])[display_cols]
    print("\nPlant 2 Generalization Results:")
    print(p2_summary_df.to_string(index=False))
    p2_summary_df.to_csv('results/tables/table_6_plant2_generalization.csv', index=False)

    print("\n" + "=" * 80)
    print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print("Generated Artifacts:")
    print("  Figures: results/figures/ (Figures 1 to 6)")
    print("  Tables:  results/tables/  (Tables 1 to 6)")
    print("=" * 80)


if __name__ == '__main__':
    main()
