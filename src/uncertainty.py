"""
uncertainty.py
Forecast uncertainty quantification using Monte Carlo Dropout (Gal & Ghahramani)
and Split Conformal Prediction (Vovk et al., Angelopoulos & Bates).
Generates predictive standard deviations (uncertainty proxy), calibrated prediction intervals,
evaluates PICP, MPIW, ACE, Winkler Score, and lead-time coverage distributions.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from scipy.stats import norm
from .preprocess import get_astronomical_night_mask


def enable_mc_dropout(model):
    """
    Sets Dropout layers to training mode while keeping BatchNorm layers in eval mode.
    Enables stochastic forward passes without altering running mean/variance statistics.
    """
    model.eval()
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout1d, nn.Dropout2d)):
            m.train()


def predict_mc_dropout(model, X_tensor, target_scaler, K=50, device='cpu', batch_size=64, dates=None):
    """
    Performs K stochastic forward passes using Monte Carlo Dropout.
    Returns:
    - mean_pred: Point forecast in physical power units (kW) [num_samples, horizon]
    - std_pred: Predictive standard deviation / uncertainty proxy in physical power units (kW) [num_samples, horizon]
    - all_preds: Full ensemble of K predictions [K, num_samples, horizon]
    """
    enable_mc_dropout(model)
    model.to(device)

    all_runs = []
    num_samples = len(X_tensor)

    with torch.no_grad():
        for k in range(K):
            preds_k = []
            for start in range(0, num_samples, batch_size):
                end = min(start + batch_size, num_samples)
                x_batch = X_tensor[start:end].to(device)
                out = model(x_batch).cpu().numpy()
                preds_k.append(out)
            preds_k = np.concatenate(preds_k, axis=0)
            all_runs.append(preds_k)

    # Shape: (K, num_samples, horizon)
    all_runs = np.array(all_runs)

    # Unscale predictions back to actual physical units (kW)
    mean = target_scaler.mean_[0]
    std = target_scaler.scale_[0]
    all_runs_unscaled = all_runs * std + mean
    all_runs_unscaled = np.clip(all_runs_unscaled, 0.0, None)

    # Enforce strict 0 at nighttime if dates available
    if dates is not None:
        night_mask = get_astronomical_night_mask(dates)
        for k in range(K):
            all_runs_unscaled[k][night_mask] = 0.0

    mean_pred = np.mean(all_runs_unscaled, axis=0)
    std_pred = np.std(all_runs_unscaled, axis=0, ddof=1)

    return mean_pred, std_pred, all_runs_unscaled


def compute_prediction_intervals(mean_pred, std_pred, confidence_level=0.90, calibration_factor=1.0, dates=None):
    """
    Calculates symmetric prediction intervals around point forecast:
    [max(0, mean - z * sigma * calib), mean + z * sigma * calib]
    with physical non-negativity and nighttime zero-enforcement.
    """
    alpha = 1.0 - confidence_level
    z_score = norm.ppf(1.0 - alpha / 2.0)

    half_width = z_score * std_pred * calibration_factor
    lower_bound = np.clip(mean_pred - half_width, 0.0, None)
    upper_bound = np.clip(mean_pred + half_width, 0.0, None)

    if dates is not None:
        night_mask = get_astronomical_night_mask(dates)
        lower_bound[night_mask] = 0.0
        upper_bound[night_mask] = 0.0

    return lower_bound, upper_bound


def compute_split_conformal_intervals(val_y_true, val_mean, val_std, test_mean, test_std,
                                      nominal_coverage=0.90, dates_test=None, method='pooled'):
    r"""
    Constructs Split Conformal Prediction Intervals guaranteeing finite-sample coverage:
    1. Computes studentized non-conformity scores on validation set:
       s_{i,h} = |y_{val, i, h} - \hat{y}_{val, i, h}| / (\hat{\sigma}_{val, i, h} + eps)
    2. Depending on method:
       - 'pooled': computes empirical quantile across all points (marginal coverage)
       - 'trajectory': takes max score per trajectory s_i^{max} = max_h s_{i,h} (joint trajectory coverage)
       - 'per_horizon': computes quantile q_h separately for each lead step h
    3. Produces test intervals: \hat{y}_{test} +/- q_hat * \hat{\sigma}_{test}.
    """
    eps = 1e-4
    val_residuals = np.abs(val_y_true - val_mean)
    val_scores = val_residuals / (val_std + eps)

    alpha = 1.0 - nominal_coverage

    if method == 'trajectory':
        # Max-score across 96 lead steps for each validation window
        traj_scores = np.max(val_scores, axis=1) # shape (N_val,)
        n_val = len(traj_scores)
        quantile_level = min(1.0, (1.0 - alpha) * (1.0 + 1.0 / n_val))
        q_hat = np.quantile(traj_scores, quantile_level)
        half_width = q_hat * (test_std + eps)
        lower_bound = np.clip(test_mean - half_width, 0.0, None)
        upper_bound = np.clip(test_mean + half_width, 0.0, None)
    elif method == 'per_horizon':
        n_val = len(val_scores)
        quantile_level = min(1.0, (1.0 - alpha) * (1.0 + 1.0 / n_val))
        q_hat = np.quantile(val_scores, quantile_level, axis=0) # shape (96,)
        half_width = q_hat * (test_std + eps)
        lower_bound = np.clip(test_mean - half_width, 0.0, None)
        upper_bound = np.clip(test_mean + half_width, 0.0, None)
    else: # 'pooled' (default)
        n_val = len(val_scores.flatten())
        quantile_level = min(1.0, (1.0 - alpha) * (1.0 + 1.0 / n_val))
        q_hat = np.quantile(val_scores, quantile_level)
        half_width = q_hat * (test_std + eps)
        lower_bound = np.clip(test_mean - half_width, 0.0, None)
        upper_bound = np.clip(test_mean + half_width, 0.0, None)

    if dates_test is not None:
        night_mask = get_astronomical_night_mask(dates_test)
        lower_bound[night_mask] = 0.0
        upper_bound[night_mask] = 0.0

    return lower_bound, upper_bound, float(np.mean(q_hat))


def evaluate_uncertainty_calibration(y_true, lower_bound, upper_bound, nominal_coverage=0.90):
    """
    Computes comprehensive probabilistic forecasting metrics:
    - PICP: Prediction Interval Coverage Probability (fraction of actuals in interval)
    - MPIW: Mean Prediction Interval Width (sharpness)
    - ACE: Average Coverage Error (PICP - nominal_coverage)
    - Winkler Score: proper scoring rule penalizing interval width and coverage violations
    - Coverage by lead time (15-min, 1-hr, 6-hr, 12-hr, 24-hr)
    """
    in_interval = (y_true >= lower_bound) & (y_true <= upper_bound)
    picp = float(np.mean(in_interval) * 100.0)
    mpiw = float(np.mean(upper_bound - lower_bound))
    ace = float(picp - (nominal_coverage * 100.0))

    # Winkler Score
    alpha = 1.0 - nominal_coverage
    delta_width = upper_bound - lower_bound
    under_penalty = (2.0 / alpha) * np.maximum(0.0, lower_bound - y_true)
    over_penalty = (2.0 / alpha) * np.maximum(0.0, y_true - upper_bound)
    winkler_score = float(np.mean(delta_width + under_penalty + over_penalty))

    # Coverage by horizon
    horizons = {'15-min': 0, '1-hr': 3, '6-hr': 23, '12-hr': 47, '24-hr': 95}
    coverage_by_lead = {}
    for h_name, h_idx in horizons.items():
        if h_idx < y_true.shape[1]:
            coverage_by_lead[h_name] = float(np.mean(in_interval[:, h_idx]) * 100.0)

    return {
        'Nominal Coverage (%)': nominal_coverage * 100.0,
        'PICP (%)': picp,
        'MPIW (kW)': mpiw,
        'ACE (%)': ace,
        'Winkler_Score': winkler_score,
        'Coverage_by_Lead': coverage_by_lead
    }


def calibrate_intervals_on_validation(val_y_true, val_mean, val_std, nominal_coverage=0.90):
    """
    Calibrates prediction interval scale factor on validation residuals
    so that empirical PICP matches nominal coverage.
    """
    from scipy.optimize import minimize_scalar

    def objective(factor):
        low, high = compute_prediction_intervals(val_mean, val_std, nominal_coverage, factor)
        picp = np.mean((val_y_true >= low) & (val_y_true <= high))
        return (picp - nominal_coverage) ** 2

    res = minimize_scalar(objective, bounds=(0.5, 5.0), method='bounded')
    return float(res.x)
