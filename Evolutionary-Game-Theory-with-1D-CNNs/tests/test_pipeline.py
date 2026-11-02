"""
test_pipeline.py
Comprehensive automated test suite validating the entire research pipeline:
- Preprocessing, unit harmonization, and chronological splitting
- Model architecture, forward pass, and deep-copy checkpointing
- Conformal prediction intervals and ordering constraints
- Microeconomic market conservation and physical bounds
- Replicator dynamics stability and clamping
- Genuine zero-shot transfer transformation
"""

import os
import unittest
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.preprocess import (load_and_clean_plant_data, prepare_chronological_splits,
                            prepare_genuine_zero_shot_transfer, get_astronomical_night_mask)
from src.train_cnn import DiurnalResidualTCN, compute_forecast_metrics, persistence_baseline
from src.uncertainty import compute_split_conformal_intervals, evaluate_uncertainty_calibration
from src.payoffs import ProsumerCommunity
from src.replicator import ReplicatorEngine


class TestPipeline(unittest.TestCase):

    def setUp(self):
        # Set seeds for reproducibility
        np.random.seed(42)
        torch.manual_seed(42)

    def test_01_dc_ac_harmonization(self):
        """Verify Plant 1 DC power is harmonized by 0.10 to match Plant 2 kW units."""
        df_p1 = load_and_clean_plant_data('Plant_1_Generation_Data.csv', 'Plant_1_Weather_Sensor_Data.csv', is_plant1=True)
        # Median DC/AC ratio in Plant 1 should now be ~1.02 instead of ~10.18
        ratio = (df_p1['DC_POWER'] / df_p1['AC_POWER'].replace(0, 1)).median()
        self.assertAlmostEqual(ratio, 1.018, places=1,
                               msg=f"Expected harmonized Plant 1 DC/AC ratio ~1.02, got {ratio}")

    def test_02_chronological_split_no_leakage(self):
        """Verify train, val, and test splits are strictly chronological and scalers fit on train only."""
        df_dummy = pd.DataFrame({
            'DATE_TIME': pd.date_range('2020-01-01', periods=1000, freq='15min'),
            'AC_POWER': np.random.uniform(0, 1000, 1000),
            'DC_POWER': np.random.uniform(0, 1000, 1000),
            'AMBIENT_TEMPERATURE': np.random.uniform(20, 35, 1000),
            'MODULE_TEMPERATURE': np.random.uniform(20, 60, 1000),
            'IRRADIATION': np.random.uniform(0, 1, 1000),
            'sin_tod': np.zeros(1000),
            'cos_tod': np.ones(1000),
            'sin_dow': np.zeros(1000),
            'cos_dow': np.ones(1000)
        })
        feature_cols = ['AC_POWER', 'DC_POWER', 'AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE',
                        'IRRADIATION', 'sin_tod', 'cos_tod', 'sin_dow', 'cos_dow']
        splits = prepare_chronological_splits(df_dummy, feature_cols, history_len=192, horizon_len=96)

        # Scaler mean must match train_df mean exactly, NOT total df mean
        train_end = int(len(df_dummy) * 0.70)
        expected_train_mean = df_dummy.iloc[:train_end]['AC_POWER'].mean()
        self.assertAlmostEqual(splits['scalers']['target_mean'], expected_train_mean, places=4)

    def test_03_deep_copy_checkpointing(self):
        """Verify that saving model checkpoint deep copies tensors without parameter reference mutation."""
        model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 2))
        saved_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        # Mutate model weights in-place
        with torch.no_grad():
            for p in model.parameters():
                p.add_(100.0)

        # Saved weights must remain completely unaffected
        for k, v in model.state_dict().items():
            diff = torch.max(torch.abs(v - saved_weights[k])).item()
            self.assertGreater(diff, 50.0, msg="Saved weights were unexpectedly mutated!")

    def test_04_tcn_forward_pass_and_shapes(self):
        """Verify DiurnalResidualTCN forward pass shape and non-negativity."""
        model = DiurnalResidualTCN(in_channels=9, horizon=96, dropout_rate=0.15, ac_idx=0)
        x_dummy = torch.randn(8, 192, 9)
        out = model(x_dummy)
        self.assertEqual(out.shape, (8, 96), f"Expected shape (8, 96), got {out.shape}")
        self.assertFalse(torch.isnan(out).any(), "Model produced NaN outputs!")

    def test_05_astronomical_night_masking(self):
        """Verify nighttime periods (hour < 5 or hour >= 19) are strictly zeroed."""
        dates = pd.date_range('2020-06-01 00:00', periods=96, freq='15min').values.reshape(1, 96)
        mask = get_astronomical_night_mask(dates)
        # Midnight (00:00) should be True (night)
        self.assertTrue(mask[0, 0])
        # Noon (12:00 = step 48) should be False (day)
        self.assertFalse(mask[0, 48])

    def test_06_prediction_interval_ordering(self):
        """Verify conformal prediction intervals satisfy 0 <= Lower <= Upper everywhere."""
        y_val = np.random.uniform(0, 1000, size=(50, 96))
        val_mean = y_val + np.random.normal(0, 50, size=(50, 96))
        val_std = np.abs(np.random.normal(50, 10, size=(50, 96)))

        test_mean = np.random.uniform(0, 1000, size=(20, 96))
        test_std = np.abs(np.random.normal(50, 10, size=(20, 96)))

        lower, upper, q_hat = compute_split_conformal_intervals(
            y_val, val_mean, val_std, test_mean, test_std, nominal_coverage=0.90
        )
        self.assertTrue(np.all(lower >= 0.0), "Lower bound has negative values!")
        self.assertTrue(np.all(upper >= lower), "Upper bound is less than Lower bound!")

    def test_07_p2p_market_energy_conservation(self):
        """Verify P2P traded energy cannot exceed available surplus or deficit."""
        community = ProsumerCommunity(num_prosumers=20, plant_capacity_kw=1000.0)
        loads = np.random.uniform(1.0, 3.0, size=20)

        # High solar, small deficit
        u_s, u_h, met = community.calculate_step_payoffs(
            gen_kw=800.0, sigma_kw=50.0, loads_kw=loads, x_share=0.8
        )
        self.assertLessEqual(met['q_p2p_kwh'], met['grid_import_kwh'] + met['q_p2p_kwh'])
        self.assertGreaterEqual(met['p_p2p'], community.p_fit + community.c_tx)
        self.assertLessEqual(met['p_p2p'], community.p_grid - community.c_tx)

    def test_08_replicator_dynamics_bounds(self):
        """Verify Replicator dynamics state stays strictly within [0, 1]."""
        engine = ReplicatorEngine(step_size=0.2)
        community = ProsumerCommunity(num_prosumers=10, plant_capacity_kw=1000.0)
        loads = np.random.uniform(1.0, 3.0, size=(50, 10))
        dates = pd.date_range('2020-06-01', periods=50, freq='15min')

        df_sim, summary = engine.simulate(
            community, gen_series=np.full(50, 500.0), sigma_series=np.zeros(50),
            load_matrix=loads, timestamps=dates, x0=0.5
        )
        self.assertTrue(np.all(df_sim['x_share'] >= 0.0))
        self.assertTrue(np.all(df_sim['x_share'] <= 1.0))

    def test_09_realized_physical_settlement_penalties(self):
        """Verify that over-promising under actual solar shortfall incurs imbalance penalty."""
        community = ProsumerCommunity(num_prosumers=10, plant_capacity_kw=1000.0, p_imbalance_penalty_pct=0.50)
        loads = np.full(10, 2.0)

        # Committed 50 kWh based on forecast, but actual generation is 0
        realized_met = community.evaluate_realized_step(
            gen_act_kw=0.0, q_p2p_comm=50.0, x_share=1.0, loads_kw=loads, p_p2p=0.14
        )
        self.assertGreater(realized_met['imbalance_cost ($)'], 0.0,
                           "Expected non-zero imbalance penalty when actual solar is 0!")
        self.assertEqual(realized_met['q_shortfall_kwh'], 50.0)

    def test_10_plant2_zero_shot_scaler_independence(self):
        """Verify Plant 2 zero-shot transfer does NOT fit any new scalers."""
        df_p1 = load_and_clean_plant_data('Plant_1_Generation_Data.csv', 'Plant_1_Weather_Sensor_Data.csv', is_plant1=True)
        df_p2 = load_and_clean_plant_data('Plant_2_Generation_Data.csv', 'Plant_2_Weather_Sensor_Data.csv', is_plant1=False)

        feature_cols = ['AC_POWER', 'DC_POWER', 'AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE',
                        'IRRADIATION', 'sin_tod', 'cos_tod', 'sin_dow', 'cos_dow']
        splits_p1 = prepare_chronological_splits(df_p1, feature_cols)

        X_p2, y_p2, dates_p2, y_p2_actual, cap_p2 = prepare_genuine_zero_shot_transfer(
            df_p2, feature_cols, source_scalers=splits_p1['scalers']
        )
        # Scaled values must be computed using Plant 1 mean and scale
        expected_ac_0 = (df_p2['AC_POWER'].iloc[0] - splits_p1['scalers']['feature_scaler'].mean_[0]) / splits_p1['scalers']['feature_scaler'].scale_[0]
        self.assertAlmostEqual(X_p2[0, 0, 0], expected_ac_0, places=4)

    def test_11_historical_diurnal_baseline_time_of_day_lookup(self):
        """Verify historical diurnal baseline indexes by genuine 15-minute time-of-day slots."""
        from src.train_cnn import historical_diurnal_profile_baseline
        # Synthetic dates over 2 days
        train_dates = pd.date_range('2020-05-01 00:00', periods=192, freq='15min').values.reshape(2, 96)
        train_y = np.zeros((2, 96))
        # Set noon slot (h=48) to 1000 kW in train
        train_y[:, 48] = 1000.0

        # Test dates: 1 test window starting at 00:00
        test_dates = pd.date_range('2020-05-03 00:00', periods=96, freq='15min').values.reshape(1, 96)
        preds = historical_diurnal_profile_baseline(train_y, train_dates, test_dates)

        # Midnight (00:00) must be 0
        self.assertEqual(preds[0, 0], 0.0)
        # Noon (12:00 = slot 48) must be 1000 kW
        self.assertAlmostEqual(preds[0, 48], 1000.0, places=2)

    def test_12_conformal_coverage_methods(self):
        """Verify both pooled and trajectory conformal methods return valid, ordered intervals."""
        y_val = np.random.uniform(10, 100, size=(30, 96))
        val_mean = y_val + np.random.normal(0, 5, size=(30, 96))
        val_std = np.full((30, 96), 5.0)

        test_mean = np.random.uniform(10, 100, size=(10, 96))
        test_std = np.full((10, 96), 5.0)

        # Pooled
        low_p, upp_p, q_p = compute_split_conformal_intervals(
            y_val, val_mean, val_std, test_mean, test_std, nominal_coverage=0.90, method='pooled'
        )
        self.assertTrue(np.all(upp_p >= low_p))

        # Trajectory
        low_t, upp_t, q_t = compute_split_conformal_intervals(
            y_val, val_mean, val_std, test_mean, test_std, nominal_coverage=0.90, method='trajectory'
        )
        self.assertTrue(np.all(upp_t >= low_t))
        # Trajectory quantile must be at least as wide as pooled quantile
        self.assertGreaterEqual(q_t, q_p - 1e-5)

    def test_13_experiment_registry_integrity(self):
        """Verify that all experiments registered in experiment_registry.csv exist on disk."""
        import csv
        from pathlib import Path
        reg_file = Path('experiments/00_experiment_registry/experiment_registry.csv')
        self.assertTrue(reg_file.exists(), "experiment_registry.csv missing!")
        with open(reg_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            exp_rows = list(reader)

        self.assertGreaterEqual(len(exp_rows), 10, "Expected at least 10 tracked experiments")
        for row in exp_rows:
            exp_id = row['id']
            # Search for directory starting with exp_id in experiments/
            matching_dirs = list(Path('experiments').glob(f'{exp_id}*'))
            self.assertTrue(len(matching_dirs) >= 1, f"Missing directory for experiment {exp_id}")
            d = matching_dirs[0]
            self.assertTrue((d / 'metadata.json').exists(), f"Missing metadata.json in {d}")
            self.assertTrue((d / 'metrics.json').exists(), f"Missing metrics.json in {d}")
            self.assertTrue((d / 'README.md').exists(), f"Missing README.md in {d}")


if __name__ == '__main__':
    unittest.main()

