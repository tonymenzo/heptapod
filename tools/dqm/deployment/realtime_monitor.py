"""
Tool 5: DQMRealtimeMonitorTool
-------------------------------
Simulates real-time CMS DQM monitoring by reading a stream of incoming
histogram records (one per line from a JSONL file), running the trained
autoencoder on each, and emitting structured alert JSON for any run whose
reconstruction error exceeds the stored threshold.

In production this tool would hook into the CMS online DQM stream
(e.g. via CMSSW DQMNet or the DQM GUI WebSocket API).

Input:   dqm_autoencoder.pt  — trained model + threshold
         stream.jsonl         — live / simulated histogram stream
Output:  dqm_alerts.jsonl     — one alert JSON object per flagged run
         dqm_monitor_log.json — summary of the monitoring session
"""

import json
import os
import time
from datetime import datetime, timezone
from typing import Optional

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
    class _DQMAutoencoder(nn.Module):
        def __init__(self, n_bins, hidden_dim, latent_dim):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(n_bins, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, latent_dim),
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, n_bins), nn.Sigmoid(),
            )
        def forward(self, x):
            return self.decoder(self.encoder(x))



if TORCH_AVAILABLE:
    class _DQMTransformer(nn.Module):
        """Lightweight Transformer for monitor inference."""
        def __init__(self, n_bins, d_model, nhead, num_layers, dim_feedforward, dropout=0.1):
            super().__init__()
            self.bin_embed = nn.Linear(1, d_model)
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                dropout=dropout, batch_first=True)
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.reconstruction_head = nn.Sequential(
                nn.Linear(d_model, dim_feedforward), nn.GELU(),
                nn.Linear(dim_feedforward, n_bins), nn.Sigmoid())

        def forward(self, x):
            B = x.size(0)
            tokens = self.bin_embed(x.unsqueeze(-1))
            cls = self.cls_token.expand(B, -1, -1)
            tokens = torch.cat([cls, tokens], dim=1)
            return self.reconstruction_head(self.transformer(tokens)[:, 0])

class DQMRealtimeMonitorTool(BaseTool):
    """
    Monitors incoming CMS DQM histogram data in near-real-time using a
    trained autoencoder. Flags runs whose reconstruction error exceeds
    the learned anomaly threshold and emits structured alert records
    compatible with the CMS DQM alert notification system.

    Alert severity levels:
      - WARNING  : error between 1x and 2x threshold
      - CRITICAL : error > 2x threshold
    """

    # --- RuntimeFields ---
    model_path: str = RuntimeField(
        description="Path to dqm_autoencoder.pt (relative to sandbox_dir)")
    stream_jsonl: str = RuntimeField(
        description="Path to input histogram stream JSONL (relative to sandbox_dir)")
    output_dir: str = RuntimeField(
        description="Subdirectory for alerts and monitoring log")
    simulate_delay_ms: int = RuntimeField(
        default=0,
        description="Milliseconds to sleep between records (simulates real-time feed; 0=fast)")
    threshold_multiplier: float = RuntimeField(
        default=1.0,
        description="Multiply stored threshold by this factor (>1 = more lenient, <1 = stricter)")

    # --- StateField ---
    sandbox_dir: str = StateField(
        description="Root sandbox directory for this HEPTAPOD session")

    # ------------------------------------------------------------------ #
    def run(self) -> dict:
        if not (NUMPY_AVAILABLE and TORCH_AVAILABLE):
            return {"status": "error",
                    "message": "pip install numpy torch"}

        model_path  = os.path.join(self.sandbox_dir, self.model_path)
        stream_path = os.path.join(self.sandbox_dir, self.stream_jsonl)
        out_dir     = os.path.join(self.sandbox_dir, self.output_dir)
        os.makedirs(out_dir, exist_ok=True)

        for p in [model_path, stream_path]:
            if not os.path.exists(p):
                return {"status": "error", "message": f"File not found: {p}"}

        # ── load model (supports autoencoder + transformer checkpoints) ──
        ckpt        = torch.load(model_path, map_location="cpu")
        n_bins      = ckpt["n_bins"]
        threshold   = ckpt["threshold"] * self.threshold_multiplier
        scaler_min  = ckpt.get("scaler_min", 0.0)
        scaler_max  = ckpt.get("scaler_max", 1.0)
        subsystem   = ckpt.get("subsystem", "unknown")
        model_type  = ckpt.get("model_type", "autoencoder")

        if model_type == "transformer":
            model = _DQMTransformer(
                n_bins=n_bins,
                d_model=ckpt.get("d_model", 32),
                nhead=ckpt.get("nhead", 4),
                num_layers=ckpt.get("num_layers", 2),
                dim_feedforward=ckpt.get("dim_feedforward", 64),
            )
        else:
            model = _DQMAutoencoder(n_bins, ckpt.get("hidden_dim", 32), ckpt.get("latent_dim", 8))

        model.load_state_dict(ckpt["state_dict"])
        model.eval()

        alerts_path = os.path.join(out_dir, "dqm_alerts.jsonl")
        log_path    = os.path.join(out_dir, "dqm_monitor_log.json")

        session_start = datetime.now(timezone.utc).isoformat()
        n_processed = n_flagged = n_warning = n_critical = 0
        flagged_runs = []

        with open(alerts_path, "w") as alert_f, \
             open(stream_path) as stream_f:

            for raw_line in stream_f:
                raw_line = raw_line.strip()
                if not raw_line:
                    continue

                rec     = json.loads(raw_line)
                run_id  = rec.get("run_id")
                hist    = rec.get("histogram", [])

                if not hist:
                    continue

                # ── normalise with training scaler ──
                hist = hist[:n_bins] + [0.0] * max(0, n_bins - len(hist))
                denom = (scaler_max - scaler_min) or 1.0
                hist_norm = [(v - scaler_min) / denom for v in hist]
                x = torch.tensor([hist_norm], dtype=torch.float32)

                # ── inference ──
                with torch.no_grad():
                    x_hat = model(x)
                    error = float(((x - x_hat) ** 2).mean().item())

                n_processed += 1

                if error > threshold:
                    if error > 2.0 * threshold:
                        severity = "CRITICAL"
                        n_critical += 1
                    else:
                        severity = "WARNING"
                        n_warning += 1
                    n_flagged += 1

                    alert = {
                        "timestamp":       datetime.now(timezone.utc).isoformat(),
                        "run_id":          run_id,
                        "subsystem":       subsystem,
                        "monitor_element": rec.get("monitor_element", ""),
                        "recon_error":     round(error, 6),
                        "threshold":       round(threshold, 6),
                        "severity":        severity,
                        "action":          (
                            "Immediately notify shift crew and flag run for exclusion"
                            if severity == "CRITICAL"
                            else "Flag run for expert review before certification"
                        ),
                    }
                    alert_f.write(json.dumps(alert) + "\n")
                    flagged_runs.append(run_id)

                if self.simulate_delay_ms > 0:
                    time.sleep(self.simulate_delay_ms / 1000.0)

        session_end = datetime.now(timezone.utc).isoformat()

        log = {
            "session_start":    session_start,
            "session_end":      session_end,
            "subsystem":        subsystem,
            "n_processed":      n_processed,
            "n_flagged":        n_flagged,
            "n_warning":        n_warning,
            "n_critical":       n_critical,
            "anomaly_rate_pct": round(100.0 * n_flagged / max(n_processed, 1), 2),
            "threshold_used":   round(threshold, 6),
            "flagged_run_ids":  flagged_runs,
        }
        with open(log_path, "w") as f:
            json.dump(log, f, indent=2)

        return {
            "status":           "ok",
            "n_processed":      n_processed,
            "n_flagged":        n_flagged,
            "n_warning":        n_warning,
            "n_critical":       n_critical,
            "anomaly_rate_pct": log["anomaly_rate_pct"],
            "alerts_jsonl":     alerts_path,
            "monitor_log":      log_path,
            "flagged_run_ids":  flagged_runs,
        }
