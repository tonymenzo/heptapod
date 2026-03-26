"""
Tool: DQMAdaptiveThresholdTool
--------------------------------
Updates the anomaly detection threshold using a rolling window of recent
run reconstruction errors. This is critical for real CMS DQM operation:
detector conditions evolve fill-by-fill, so a threshold computed once at
training time gradually becomes stale.

Strategy: Every `update_every` runs, recompute threshold as the
`percentile`th percentile of errors in the last `window_size` runs.
The new threshold is saved back into the model checkpoint so downstream
monitoring tools pick it up automatically on next load.

Input:   dqm_autoencoder.pt  — model checkpoint (modified in-place)
         rolling_errors.jsonl — log of recent (run_id, recon_error) pairs
Output:  rolling_errors.jsonl updated
         model checkpoint threshold field updated
         dqm_threshold_log.json — history of threshold updates
"""

import json
import os
from datetime import datetime, timezone
from typing import Optional

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .._compat import BaseTool, RuntimeField, StateField


class DQMAdaptiveThresholdTool(BaseTool):
    """
    Maintains a rolling-window anomaly threshold for CMS DQM monitoring.

    CMS detector conditions change fill-by-fill. A threshold computed once
    at training time becomes stale within hours of operation. This tool
    recomputes the anomaly threshold every `update_every` runs using the
    most recent `window_size` reconstruction errors — keeping the false
    positive rate stable as conditions evolve.

    Usage: call after DQMRealtimeMonitorTool on each batch of runs.
    The updated threshold is written back to the model checkpoint and
    picked up automatically on the next monitoring call.

    Inputs:
      model_path:         path to dqm_autoencoder.pt (relative to sandbox)
      new_errors_jsonl:   JSONL of {"run_id": int, "recon_error": float}
                          records from the current monitoring session
      window_size:        number of recent errors to keep (default 200)
      update_every:       update threshold after this many new records (default 50)
      percentile:         threshold percentile (default 95.0)
      min_threshold:      floor to prevent threshold collapsing to 0 (default 0.001)

    Output JSON:
      {"status": "ok", "threshold_updated": bool, "new_threshold": float,
       "n_errors_in_window": int, "update_count": int}
    """

    # --- RuntimeFields ---
    model_path: str = RuntimeField(
        description="Path to model checkpoint .pt file (relative to sandbox_dir)")
    new_errors_jsonl: str = RuntimeField(
        description="Path to JSONL of {run_id, recon_error} from current session "
                    "(relative to sandbox_dir). Will be created if absent.")
    window_size: int = RuntimeField(
        default=200,
        description="Rolling window: keep last N reconstruction errors for threshold update")
    update_every: int = RuntimeField(
        default=50,
        description="Recompute threshold after accumulating this many new errors")
    percentile: float = RuntimeField(
        default=95.0,
        description="Threshold = Nth percentile of rolling error window (default 95.0)")
    min_threshold: float = RuntimeField(
        default=0.001,
        description="Minimum threshold floor — prevents collapsing to near-zero")
    output_dir: str = RuntimeField(
        default="monitoring",
        description="Directory for threshold update log")

    # --- StateField ---
    sandbox_dir: str = StateField(
        description="Root sandbox directory for this HEPTAPOD session")

    # ------------------------------------------------------------------ #
    def run(self) -> dict:
        if not (NUMPY_AVAILABLE and TORCH_AVAILABLE):
            return {"status": "error", "message": "pip install numpy torch"}

        model_path  = os.path.join(self.sandbox_dir, self.model_path)
        errors_path = os.path.join(self.sandbox_dir, self.new_errors_jsonl)
        out_dir     = os.path.join(self.sandbox_dir, self.output_dir)
        log_path    = os.path.join(out_dir, "dqm_threshold_log.json")
        os.makedirs(out_dir, exist_ok=True)

        if not os.path.exists(model_path):
            return {"status": "error", "message": f"Model not found: {model_path}"}

        # ── load existing rolling error history ──
        history: list[dict] = []
        if os.path.exists(errors_path):
            with open(errors_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        history.append(json.loads(line))

        n_before = len(history)

        # ── check if we have enough new errors to trigger an update ──
        if len(history) - (self._last_update_count(log_path)) < self.update_every:
            return {
                "status":           "ok",
                "threshold_updated": False,
                "reason":           f"Need {self.update_every} new errors; "
                                    f"have {len(history) - self._last_update_count(log_path)}",
                "n_errors_in_window": len(history),
            }

        # ── keep only the last window_size errors ──
        window = history[-self.window_size:]
        errors = np.array([r["recon_error"] for r in window], dtype=np.float32)

        new_threshold = float(max(
            np.percentile(errors, self.percentile),
            self.min_threshold,
        ))

        # ── update checkpoint threshold in-place ──
        ckpt = torch.load(model_path, map_location="cpu")
        old_threshold = float(ckpt.get("threshold", -1.0))
        ckpt["threshold"] = new_threshold
        torch.save(ckpt, model_path)

        # ── trim history file to window_size ──
        with open(errors_path, "w") as f:
            for r in window:
                f.write(json.dumps(r) + "\n")

        # ── append to threshold log ──
        log_entry = {
            "timestamp":       datetime.now(timezone.utc).isoformat(),
            "old_threshold":   round(old_threshold, 6),
            "new_threshold":   round(new_threshold, 6),
            "n_window_errors": len(errors),
            "percentile":      self.percentile,
            "error_mean":      round(float(errors.mean()), 6),
            "error_std":       round(float(errors.std()), 6),
        }
        existing_log = []
        if os.path.exists(log_path):
            with open(log_path) as f:
                try:
                    existing_log = json.load(f)
                except json.JSONDecodeError:
                    existing_log = []
        existing_log.append(log_entry)
        with open(log_path, "w") as f:
            json.dump(existing_log, f, indent=2)

        return {
            "status":            "ok",
            "threshold_updated": True,
            "old_threshold":     round(old_threshold, 6),
            "new_threshold":     round(new_threshold, 6),
            "n_errors_in_window": len(errors),
            "percentile_used":   self.percentile,
            "update_count":      len(existing_log),
            "threshold_log":     log_path,
        }

    def _last_update_count(self, log_path: str) -> int:
        """Return total errors processed at last update (0 if no log exists)."""
        if not os.path.exists(log_path):
            return 0
        try:
            with open(log_path) as f:
                log = json.load(f)
            return len(log) * self.update_every
        except Exception:
            return 0
