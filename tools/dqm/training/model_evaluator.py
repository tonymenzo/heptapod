"""
Tool 4: DQMModelEvaluatorTool
------------------------------
Evaluates a trained DQM autoencoder against a labelled or unlabelled hold-out
set. Computes reconstruction errors per run, compares against the stored
threshold, and reports precision / recall / F1 if labels are available.

Input:   dqm_autoencoder.pt  (from DQMAutoencoderTrainTool)
         dqm_processed.npy + dqm_meta.json  (can be a separate test set)
Output:  dqm_eval_report.json  — per-run scores and aggregate metrics
"""

import json
import os
from typing import Optional, List

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .._compat import BaseTool, RuntimeField, StateField


if TORCH_AVAILABLE:
    class _DQMAutoencoder(torch.nn.Module):
        def __init__(self, n_bins, hidden_dim, latent_dim):
            super().__init__()
            self.encoder = torch.nn.Sequential(
                torch.nn.Linear(n_bins, hidden_dim), torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, latent_dim),
            )
            self.decoder = torch.nn.Sequential(
                torch.nn.Linear(latent_dim, hidden_dim), torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, n_bins), torch.nn.Sigmoid(),
            )
        def forward(self, x):
            return self.decoder(self.encoder(x))



if TORCH_AVAILABLE:
    class _DQMTransformer(torch.nn.Module):
        """Minimal Transformer for checkpoint loading in evaluator."""
        import math as _math
        def __init__(self, n_bins, d_model, nhead, num_layers, dim_feedforward, dropout=0.1):
            super().__init__()
            import math
            self.n_bins = n_bins
            self.d_model = d_model
            self.bin_embed = torch.nn.Linear(1, d_model)
            self.cls_token = torch.nn.Parameter(torch.randn(1, 1, d_model))
            encoder_layer = torch.nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                dropout=dropout, batch_first=True)
            self.transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.reconstruction_head = torch.nn.Sequential(
                torch.nn.Linear(d_model, dim_feedforward), torch.nn.GELU(),
                torch.nn.Linear(dim_feedforward, n_bins), torch.nn.Sigmoid())

        def forward(self, x):
            B = x.size(0)
            tokens = self.bin_embed(x.unsqueeze(-1))
            cls = self.cls_token.expand(B, -1, -1)
            tokens = torch.cat([cls, tokens], dim=1)
            encoded = self.transformer(tokens)
            return self.reconstruction_head(encoded[:, 0])

class DQMModelEvaluatorTool(BaseTool):
    """
    Evaluates a trained DQM autoencoder on a (possibly new) dataset.
    Reports per-run reconstruction errors and flags anomalous runs.
    If ground-truth bad-run labels are provided, also computes
    precision, recall, and F1 for anomaly detection.
    """

    # --- RuntimeFields ---
    model_path: str = RuntimeField(
        description="Path to dqm_autoencoder.pt (relative to sandbox_dir)")
    eval_npy: str = RuntimeField(
        description="Path to evaluation dqm_processed.npy (relative to sandbox_dir)")
    eval_meta: str = RuntimeField(
        description="Path to evaluation dqm_meta.json (relative to sandbox_dir)")
    output_dir: str = RuntimeField(
        description="Subdirectory for evaluation report")
    known_bad_runs: Optional[List[int]] = RuntimeField(
        default=None,
        description="Optional list of known-bad run IDs for supervised metrics")

    # --- StateField ---
    sandbox_dir: str = StateField(
        description="Root sandbox directory for this HEPTAPOD session")

    # ------------------------------------------------------------------ #
    def run(self) -> dict:
        if not (NUMPY_AVAILABLE and TORCH_AVAILABLE):
            return {"status": "error",
                    "message": "pip install numpy torch"}

        model_path = os.path.join(self.sandbox_dir, self.model_path)
        npy_path   = os.path.join(self.sandbox_dir, self.eval_npy)
        meta_path  = os.path.join(self.sandbox_dir, self.eval_meta)
        out_dir    = os.path.join(self.sandbox_dir, self.output_dir)
        os.makedirs(out_dir, exist_ok=True)

        for p in [model_path, npy_path, meta_path]:
            if not os.path.exists(p):
                return {"status": "error", "message": f"File not found: {p}"}

        # ── load model (supports autoencoder + transformer checkpoints) ──
        ckpt       = torch.load(model_path, map_location="cpu")
        n_bins     = ckpt["n_bins"]
        threshold  = ckpt["threshold"]
        model_type = ckpt.get("model_type", "autoencoder")

        if model_type == "transformer":
            model = _DQMTransformer(
                n_bins=n_bins,
                d_model=ckpt.get("d_model", 32),
                nhead=ckpt.get("nhead", 4),
                num_layers=ckpt.get("num_layers", 2),
                dim_feedforward=ckpt.get("dim_feedforward", 64),
            )
        else:
            hidden_dim = ckpt.get("hidden_dim", 32)
            latent_dim = ckpt.get("latent_dim", 8)
            model = _DQMAutoencoder(n_bins, hidden_dim, latent_dim)

        model.load_state_dict(ckpt["state_dict"])
        model.eval()

        # ── load data ──
        X = np.load(npy_path).astype(np.float32)
        with open(meta_path) as f:
            meta = json.load(f)
        run_ids = meta.get("run_ids", list(range(len(X))))

        X_t    = torch.tensor(X)
        with torch.no_grad():
            x_hat   = model(X_t)
            errors  = ((X_t - x_hat) ** 2).mean(dim=1).numpy()

        # ── per-run results ──
        per_run = []
        flagged = []
        for i, (rid, err) in enumerate(zip(run_ids, errors)):
            is_anomaly = bool(err > threshold)
            per_run.append({
                "run_id":           rid,
                "recon_error":      round(float(err), 6),
                "flagged_anomaly":  is_anomaly,
            })
            if is_anomaly:
                flagged.append(rid)

        # ── supervised metrics (if labels provided) ──
        sup_metrics = {}
        if self.known_bad_runs:
            bad_set = set(self.known_bad_runs)
            tp = sum(1 for r in per_run
                     if r["flagged_anomaly"] and r["run_id"] in bad_set)
            fp = sum(1 for r in per_run
                     if r["flagged_anomaly"] and r["run_id"] not in bad_set)
            fn = sum(1 for r in per_run
                     if not r["flagged_anomaly"] and r["run_id"] in bad_set)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1        = (2 * precision * recall / (precision + recall)
                         if (precision + recall) > 0 else 0.0)
            sup_metrics = {
                "tp": tp, "fp": fp, "fn": fn,
                "precision": round(precision, 4),
                "recall":    round(recall, 4),
                "f1":        round(f1, 4),
            }

        report = {
            "subsystem":         ckpt.get("subsystem", "unknown"),
            "n_runs_evaluated":  len(run_ids),
            "n_flagged":         len(flagged),
            "anomaly_threshold": round(threshold, 6),
            "flagged_run_ids":   flagged,
            "supervised_metrics": sup_metrics,
            "per_run":           per_run,
        }

        report_path = os.path.join(out_dir, "dqm_eval_report.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        return {
            "status":            "ok",
            "report_path":       report_path,
            "n_evaluated":       len(run_ids),
            "n_flagged":         len(flagged),
            "flagged_run_ids":   flagged,
            "supervised_metrics": sup_metrics,
        }
