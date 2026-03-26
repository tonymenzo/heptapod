"""
Tool 3: DQMAutoencoderTrainTool
--------------------------------
Trains a fully-connected autoencoder on normalised DQM histogram data.
The autoencoder learns to reconstruct "good" (normal) runs; runs that
cannot be well-reconstructed are flagged as anomalies at inference time.

Architecture (configurable via RuntimeFields):
    Input(n_bins) -> Linear(hidden) -> ReLU -> Linear(latent)
                  -> Linear(hidden) -> ReLU -> Linear(n_bins) -> Sigmoid

Input:   dqm_processed.npy + dqm_meta.json  (from DQMPreprocessorTool)
Output:  dqm_autoencoder.pt   — trained PyTorch model state dict
         dqm_train_log.json   — per-epoch loss, threshold, training metadata
"""

import json
import os
from typing import Optional

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .._compat import BaseTool, RuntimeField, StateField


# ── Model definition (only instantiated when torch is available) ──────────────
if TORCH_AVAILABLE:
    class _DQMAutoencoder(nn.Module):
        def __init__(self, n_bins, hidden_dim, latent_dim):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(n_bins,    hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, latent_dim),
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim,  hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim,  n_bins),     nn.Sigmoid(),
            )

        def forward(self, x):
            return self.decoder(self.encoder(x))

        def reconstruction_error(self, x):
            with torch.no_grad():
                x_hat = self.forward(x)
                return ((x - x_hat) ** 2).mean(dim=1)


# ── Tool ──────────────────────────────────────────────────────────────────────
class DQMAutoencoderTrainTool(BaseTool):
    """
    Trains a fully-connected autoencoder on preprocessed DQM histogram data
    for unsupervised anomaly detection.

    The anomaly threshold is set at (mean + threshold_sigma * std) of
    per-run reconstruction errors on the training set, following standard
    practice in HEP DQM literature.
    """

    # --- RuntimeFields ---
    processed_npy: str = RuntimeField(
        description="Path to dqm_processed.npy (relative to sandbox_dir)")
    meta_json: str = RuntimeField(
        description="Path to dqm_meta.json (relative to sandbox_dir)")
    output_dir: str = RuntimeField(
        description="Subdirectory (inside sandbox) for model + training log")
    hidden_dim: int = RuntimeField(
        default=32,
        description="Number of neurons in encoder/decoder hidden layer")
    latent_dim: int = RuntimeField(
        default=8,
        description="Dimensionality of the bottleneck latent space")
    epochs: int = RuntimeField(
        default=50,
        description="Number of training epochs")
    batch_size: int = RuntimeField(
        default=32,
        description="Mini-batch size for SGD")
    learning_rate: float = RuntimeField(
        default=1e-3,
        description="Adam optimiser learning rate")
    threshold_sigma: float = RuntimeField(
        default=3.0,
        description=(
            "Controls anomaly threshold percentile of training reconstruction errors: "
            "1.0→75th pct, 2.0→90th, 3.0→95th (default), 4.0→99th, 5.0→99.9th. "
            "Higher values = more lenient (fewer false positives)."
        ))
    val_fraction: float = RuntimeField(
        default=0.1,
        description="Fraction of data held out for validation loss tracking")

    # --- StateField ---
    sandbox_dir: str = StateField(
        description="Root sandbox directory for this HEPTAPOD session")

    # ------------------------------------------------------------------ #
    def run(self) -> dict:
        # Fix random seed for reproducible threshold computation
        import torch as _torch; import numpy as _np_seed; import random as _random
        _torch.manual_seed(42); _np_seed.random.seed(42); _random.seed(42)
        if not NUMPY_AVAILABLE:
            return {"status": "error", "message": "pip install numpy"}
        if not TORCH_AVAILABLE:
            return {"status": "error",
                    "message": "pip install torch"}

        npy_path  = os.path.join(self.sandbox_dir, self.processed_npy)
        meta_path = os.path.join(self.sandbox_dir, self.meta_json)
        out_dir   = os.path.join(self.sandbox_dir, self.output_dir)
        os.makedirs(out_dir, exist_ok=True)

        for p in [npy_path, meta_path]:
            if not os.path.exists(p):
                return {"status": "error", "message": f"File not found: {p}"}

        X = np.load(npy_path).astype(np.float32)          # (N, n_bins)
        with open(meta_path) as f:
            meta = json.load(f)
        n_bins = X.shape[1]

        # ── train / val split (robust to tiny datasets) ──
        n_samples = len(X)
        if n_samples == 0:
            return {"status": "error",
                    "message": "Empty input array: no samples available for training."}
        if n_samples == 1:
            idx  = np.array([0])
            X_tr = torch.tensor(X[idx])
            X_val = X_tr.clone()
        else:
            n_val = int(n_samples * self.val_fraction)
            if n_val <= 0:
                n_val = 1
            if n_val >= n_samples:
                n_val = n_samples - 1
            idx   = np.random.permutation(n_samples)
            X_val = torch.tensor(X[idx[:n_val]])
            X_tr  = torch.tensor(X[idx[n_val:]])

        loader = DataLoader(
            TensorDataset(X_tr),
            batch_size=self.batch_size,
            shuffle=True,
        )

        model     = _DQMAutoencoder(n_bins, self.hidden_dim, self.latent_dim)
        criterion = nn.MSELoss()
        optimiser = torch.optim.Adam(model.parameters(), lr=self.learning_rate)

        train_log = []
        for epoch in range(1, self.epochs + 1):
            model.train()
            ep_loss = 0.0
            for (batch,) in loader:
                optimiser.zero_grad()
                loss = criterion(model(batch), batch)
                loss.backward()
                optimiser.step()
                ep_loss += loss.item() * len(batch)
            ep_loss /= len(X_tr)

            model.eval()
            with torch.no_grad():
                val_loss = criterion(model(X_val), X_val).item()

            train_log.append({"epoch": epoch,
                               "train_loss": round(ep_loss, 6),
                               "val_loss":   round(val_loss, 6)})

        # ── compute anomaly threshold on full training set ──
        model.eval()
        errors = model.reconstruction_error(torch.tensor(X)).numpy()
        # Percentile-based threshold: more robust than mean+k*std when model
        # hasn't fully converged. threshold_sigma maps to percentile:
        # 1.0→75th, 2.0→90th, 3.0→95th, 4.0→99th, 5.0→99.9th
        import numpy as _np
        sigma_to_pct = {1.0: 75, 2.0: 90, 3.0: 95, 4.0: 99, 5.0: 99.9}
        pct = sigma_to_pct.get(float(self.threshold_sigma),
                               min(99.9, 50 + self.threshold_sigma * 16))
        threshold = float(_np.percentile(errors, pct))
        # Ensure threshold is always above mean so normal runs aren't flagged
        threshold = max(threshold, float(errors.mean() + 0.5 * errors.std()))

        # ── save ──
        model_path = os.path.join(out_dir, "dqm_autoencoder.pt")
        log_path   = os.path.join(out_dir, "dqm_train_log.json")

        torch.save({
            "state_dict":     model.state_dict(),
            "n_bins":         n_bins,
            "hidden_dim":     self.hidden_dim,
            "latent_dim":     self.latent_dim,
            "threshold":      threshold,
            "scaler_min":     meta["scaler_min"],
            "scaler_max":     meta["scaler_max"],
            "subsystem":      meta.get("subsystem", "unknown"),
        }, model_path)

        with open(log_path, "w") as f:
            json.dump({
                "epochs":         self.epochs,
                "final_train_loss": train_log[-1]["train_loss"],
                "final_val_loss":   train_log[-1]["val_loss"],
                "anomaly_threshold": threshold,
                "threshold_sigma":  self.threshold_sigma,
                "per_epoch":        train_log,
            }, f, indent=2)

        return {
            "status":            "ok",
            "model_path":        model_path,
            "train_log":         log_path,
            "n_training_runs":   int(len(X_tr)),
            "n_val_runs":        int(len(X_val)),
            "final_train_loss":  train_log[-1]["train_loss"],
            "final_val_loss":    train_log[-1]["val_loss"],
            "anomaly_threshold": round(threshold, 6),
        }
