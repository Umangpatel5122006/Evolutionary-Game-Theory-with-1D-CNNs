"""
ensemble.py - Multi-model day-ahead TCN ensemble training and probabilistic prediction.
"""

import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from .day_ahead_tcn import DayAheadTCN
from .data.loader import get_fixed_clock_dark_hours_mask

def train_single_tcn(model, train_loader, val_loader, epochs=25, lr=1e-3, device='cpu', patience=4):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.SmoothL1Loss(beta=1.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val = float('inf')
    best_state = None
    patience_cnt = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            out = model(X_b)
            loss = criterion(out, y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        val_l = 0.0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                val_l += criterion(model(X_b), y_b).item() * len(X_b)
        val_l /= len(val_loader.dataset)
        scheduler.step()

        if val_l < best_val:
            best_val = val_l
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val


class DayAheadEnsemble:
    def __init__(self, in_channels=9, horizon=96, ensemble_size=3, seeds=(101, 202, 303), dropout=0.15):
        self.in_channels = in_channels
        self.horizon = horizon
        self.ensemble_size = ensemble_size
        self.seeds = seeds
        self.dropout = dropout
        self.models = []

    def fit(self, train_X, train_y, val_X, val_y, batch_size=32, epochs=25, lr=1e-3, device='cpu'):
        train_ds = TensorDataset(torch.tensor(train_X, dtype=torch.float32), torch.tensor(train_y, dtype=torch.float32))
        val_ds = TensorDataset(torch.tensor(val_X, dtype=torch.float32), torch.tensor(val_y, dtype=torch.float32))

        train_ldr = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_ldr = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        self.models = []
        for seed in self.seeds[:self.ensemble_size]:
            torch.manual_seed(seed)
            np.random.seed(seed)
            m = DayAheadTCN(in_channels=self.in_channels, horizon=self.horizon, dropout=self.dropout)
            m_trained, b_val = train_single_tcn(m, train_ldr, val_ldr, epochs=epochs, lr=lr, device=device)
            self.models.append(m_trained)
            print(f"Ensemble member (seed {seed}) fitted with best Val Huber Loss: {b_val:.4f}")

    def predict(self, X_array, target_scaler, dates=None, device='cpu'):
        X_t = torch.tensor(X_array, dtype=torch.float32).to(device)
        member_preds = []

        if isinstance(target_scaler, dict):
            scale_val = target_scaler.get('target_std', target_scaler.get('target_scaler').scale_[0] if 'target_scaler' in target_scaler else 1.0)
            mean_val = target_scaler.get('target_mean', target_scaler.get('target_scaler').mean_[0] if 'target_scaler' in target_scaler else 0.0)
        else:
            scale_val = target_scaler.scale_[0]
            mean_val = target_scaler.mean_[0]

        for m in self.models:
            m.eval()
            with torch.no_grad():
                pred_s = m(X_t).cpu().numpy()
                unscaled = pred_s * scale_val + mean_val
                unscaled = np.clip(unscaled, 0.0, None)
                if dates is not None:
                    dark_mask = get_fixed_clock_dark_hours_mask(dates)
                    unscaled[dark_mask] = 0.0
                member_preds.append(unscaled)

        all_m = np.array(member_preds) # (M, N, 96)
        ensemble_mean = np.mean(all_m, axis=0) # (N, 96)
        ensemble_std = np.std(all_m, axis=0, ddof=1) if len(self.models) > 1 else np.zeros_like(ensemble_mean)
        return ensemble_mean, ensemble_std, all_m