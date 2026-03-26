"""
tools/dqm/training/transformer_train.py
----------------------------------------
Tool: DQMTransformerTrainTool
----------------------------------------
Trains a lightweight Transformer encoder for unsupervised DQM anomaly detection.

Why a Transformer for histogram data?
  - CMS DQM histograms are sequential bin arrays; Transformers can capture
    long-range bin correlations that fully-connected autoencoders miss.
  - Self-attention learns which bin regions co-vary — useful for detecting
    subtle structured degradation (e.g. dead readout groups, partial HV trips).
  - The [CLS] token embedding provides a compact run representation suitable
    for downstream clustering or few-shot classification.

Architecture:
  Embedding layer (n_bins -> d_model)
  + sinusoidal positional encoding
  -> N x TransformerEncoderLayer (multi-head self-attention + FFN)
  -> [CLS] token -> reconstruction head -> n_bins output
  Anomaly score = MSE(input, reconstruction)

Input:  dqm_processed.npy + dqm_meta.json  (identical to autoencoder inputs)
Output: dqm_transformer.pt   -- model state dict + metadata
        dqm_transformer_log.json -- per-epoch losses + threshold
"""

import json
import math
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


# ── Model classes (only defined when torch is available) ─────────────────────

def _build_model_classes():
    """Return model classes. Called lazily so this file imports without torch."""

    class SinusoidalPositionalEncoding(nn.Module):
        """Standard sinusoidal positional encoding (Vaswani et al. 2017)."""

        def __init__(self, d_model, max_len=512, dropout=0.1):
            super().__init__()
            self.dropout = nn.Dropout(p=dropout)
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(max_len).unsqueeze(1).float()
            div_term = torch.exp(
                torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
            )
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])
            self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

        def forward(self, x):
            x = x + self.pe[:, : x.size(1)]
            return self.dropout(x)

    class DQMTransformer(nn.Module):
        """
        Transformer autoencoder for DQM histogram anomaly detection.

        The histogram is treated as a sequence of bins (each bin is a 1-D token).
        A [CLS] token is prepended; its final hidden state drives reconstruction.
        """

        def __init__(self, n_bins, d_model=32, nhead=4, num_layers=2,
                     dim_feedforward=64, dropout=0.1):
            super().__init__()
            self.n_bins = n_bins
            self.d_model = d_model
            self.bin_embed = nn.Linear(1, d_model)
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
            self.pos_enc = SinusoidalPositionalEncoding(
                d_model, max_len=n_bins + 1, dropout=dropout)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout, batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer,
                                                      num_layers=num_layers)
            self.reconstruction_head = nn.Sequential(
                nn.Linear(d_model, dim_feedforward),
                nn.GELU(),
                nn.Linear(dim_feedforward, n_bins),
                nn.Sigmoid(),
            )

        def forward(self, x):
            B = x.size(0)
            tokens = self.bin_embed(x.unsqueeze(-1))          # (B, n_bins, d_model)
            cls = self.cls_token.expand(B, -1, -1)            # (B, 1, d_model)
            tokens = torch.cat([cls, tokens], dim=1)           # (B, n_bins+1, d_model)
            tokens = self.pos_enc(tokens)
            encoded = self.transformer(tokens)                 # (B, n_bins+1, d_model)
            cls_out = encoded[:, 0]                            # (B, d_model)
            return self.reconstruction_head(cls_out)           # (B, n_bins)

        def reconstruction_error(self, x):
            with torch.no_grad():
                x_hat = self.forward(x)
                return ((x - x_hat) ** 2).mean(dim=1)

    return DQMTransformer


# ── Tool ──────────────────────────────────────────────────────────────────────

class DQMTransformerTrainTool(BaseTool):
    """
    Trains a Transformer-based autoencoder on preprocessed DQM histogram data.

    The Transformer architecture captures bin-level correlations via
    multi-head self-attention, making it sensitive to structured anomalies
    (e.g. dead readout groups that affect specific eta-phi regions) that
    simple fully-connected autoencoders may overlook.

    Outputs are compatible with DQMRealtimeMonitorTool and DQMModelEvaluatorTool
    (same checkpoint format as DQMAutoencoderTrainTool).
    """

    # --- RuntimeFields ---
    processed_npy: str = RuntimeField(
        description="Path to dqm_processed.npy (relative to sandbox_dir)")
    meta_json: str = RuntimeField(
        description="Path to dqm_meta.json (relative to sandbox_dir)")
    output_dir: str = RuntimeField(
        description="Subdirectory for trained model + log")
    d_model: int = RuntimeField(
        default=32,
        description="Transformer embedding dimension (must be divisible by nhead)")
    nhead: int = RuntimeField(
        default=4,
        description="Number of self-attention heads")
    num_layers: int = RuntimeField(
        default=2,
        description="Number of Transformer encoder layers")
    dim_feedforward: int = RuntimeField(
        default=64,
        description="Inner dimension of Transformer FFN sublayer")
    dropout: float = RuntimeField(
        default=0.1,
        description="Dropout probability in Transformer layers")
    epochs: int = RuntimeField(
        default=50,
        description="Number of training epochs")
    batch_size: int = RuntimeField(
        default=32,
        description="Mini-batch size")
    learning_rate: float = RuntimeField(
        default=1e-3,
        description="Adam optimiser learning rate")
    threshold_sigma: float = RuntimeField(
        default=3.0,
        description="Anomaly threshold = mean + sigma x std of training errors")
    val_fraction: float = RuntimeField(
        default=0.1,
        description="Fraction of data held out for validation")

    # --- StateField ---
    sandbox_dir: str = StateField(
        description="Root sandbox directory for this HEPTAPOD session")

    # ------------------------------------------------------------------ #
    def run(self) -> dict:
        if not NUMPY_AVAILABLE:
            return {"status": "error", "message": "pip install numpy"}
        if not TORCH_AVAILABLE:
            return {"status": "error", "message": "pip install torch"}

        DQMTransformer = _build_model_classes()

        npy_path  = os.path.join(self.sandbox_dir, self.processed_npy)
        meta_path = os.path.join(self.sandbox_dir, self.meta_json)
        out_dir   = os.path.join(self.sandbox_dir, self.output_dir)
        os.makedirs(out_dir, exist_ok=True)

        for p in [npy_path, meta_path]:
            if not os.path.exists(p):
                return {"status": "error", "message": f"File not found: {p}"}

        if self.d_model % self.nhead != 0:
            return {
                "status": "error",
                "message": (f"d_model ({self.d_model}) must be divisible by "
                            f"nhead ({self.nhead})"),
            }

        X = np.load(npy_path).astype(np.float32)
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
            if n_val <= 0 and self.val_fraction > 0.0:
                n_val = 1
            if n_val >= n_samples:
                n_val = n_samples - 1
            idx   = np.random.permutation(n_samples)
            X_val = torch.tensor(X[idx[:n_val]])
            X_tr  = torch.tensor(X[idx[n_val:]])

        loader = DataLoader(TensorDataset(X_tr),
                            batch_size=self.batch_size, shuffle=True)

        model = DQMTransformer(
            n_bins=n_bins, d_model=self.d_model, nhead=self.nhead,
            num_layers=self.num_layers, dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
        )
        criterion = nn.MSELoss()
        optimiser = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimiser, T_max=self.epochs)

        train_log = []
        for epoch in range(1, self.epochs + 1):
            model.train()
            ep_loss = 0.0
            for (batch,) in loader:
                optimiser.zero_grad()
                loss = criterion(model(batch), batch)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimiser.step()
                ep_loss += loss.item() * len(batch)
            ep_loss /= len(X_tr)
            scheduler.step()

            model.eval()
            with torch.no_grad():
                val_loss = criterion(model(X_val), X_val).item()

            train_log.append({
                "epoch":      epoch,
                "train_loss": round(ep_loss, 6),
                "val_loss":   round(val_loss, 6),
            })

        # ── anomaly threshold ──
        model.eval()
        errors    = model.reconstruction_error(torch.tensor(X)).numpy()
        threshold = float(errors.mean() + self.threshold_sigma * errors.std())

        # ── save (same checkpoint schema as DQMAutoencoderTrainTool) ──
        model_path = os.path.join(out_dir, "dqm_transformer.pt")
        log_path   = os.path.join(out_dir, "dqm_transformer_log.json")

        torch.save({
            "state_dict":      model.state_dict(),
            "model_type":      "transformer",
            "n_bins":          n_bins,
            "d_model":         self.d_model,
            "nhead":           self.nhead,
            "num_layers":      self.num_layers,
            "dim_feedforward": self.dim_feedforward,
            "dropout":         self.dropout,
            "threshold":       threshold,
            "scaler_min":      meta["scaler_min"],
            "scaler_max":      meta["scaler_max"],
            "subsystem":       meta.get("subsystem", "unknown"),
        }, model_path)

        with open(log_path, "w") as f:
            json.dump({
                "model":            "DQMTransformer",
                "epochs":           self.epochs,
                "final_train_loss": train_log[-1]["train_loss"],
                "final_val_loss":   train_log[-1]["val_loss"],
                "anomaly_threshold": threshold,
                "threshold_sigma":  self.threshold_sigma,
                "architecture": {
                    "d_model":         self.d_model,
                    "nhead":           self.nhead,
                    "num_layers":      self.num_layers,
                    "dim_feedforward": self.dim_feedforward,
                    "n_params":        sum(p.numel() for p in model.parameters()),
                },
                "per_epoch": train_log,
            }, f, indent=2)

        return {
            "status":           "ok",
            "model_path":       model_path,
            "train_log":        log_path,
            "n_training_runs":  int(len(X_tr)),
            "n_val_runs":       int(len(X_val)),
            "final_train_loss": train_log[-1]["train_loss"],
            "final_val_loss":   train_log[-1]["val_loss"],
            "anomaly_threshold": round(threshold, 6),
            "n_params":         sum(p.numel() for p in model.parameters()),
        }
