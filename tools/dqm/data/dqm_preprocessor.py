"""
Tool 2: DQMPreprocessorTool
----------------------------
Reads the raw JSONL produced by CMSDQMFetchTool, normalises histograms
(zero-pad / truncate to fixed length, min-max scale), removes bad runs,
and writes a clean numpy .npy array + metadata JSON ready for model training.

Input:  dqm_raw.jsonl
Output: dqm_processed.npy   — float32 array of shape (N_runs, N_bins)
        dqm_meta.json        — run IDs, subsystem, scaler params, bad-run list
"""

import json
import os
from typing import Optional

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

from .._compat import BaseTool, RuntimeField, StateField


class DQMPreprocessorTool(BaseTool):
    """
    Preprocesses raw CMS DQM histogram data for ML training.
    Steps:
      1. Load JSONL, extract histogram vectors.
      2. Pad / truncate to uniform bin length.
      3. Drop runs whose histograms are all-zero (bad readout).
      4. Min-max normalise across the training set.
      5. Save as float32 numpy array + JSON metadata.
    """

    # --- RuntimeFields ---
    input_jsonl: str = RuntimeField(
        description="Path to dqm_raw.jsonl produced by CMSDQMFetchTool "
                    "(relative to sandbox_dir)")
    output_dir: str = RuntimeField(
        description="Subdirectory (inside sandbox) for processed outputs")
    n_bins: int = RuntimeField(
        default=64,
        description="Target number of histogram bins (pad / truncate to this)")
    drop_zero_fraction: float = RuntimeField(
        default=0.95,
        description="Drop a histogram if more than this fraction of bins are zero")

    # --- StateField ---
    sandbox_dir: str = StateField(
        description="Root sandbox directory for this HEPTAPOD session")

    # ------------------------------------------------------------------ #
    def run(self) -> dict:
        if not NUMPY_AVAILABLE:
            return {"status": "error",
                    "message": "numpy is required: pip install numpy"}

        in_path  = os.path.join(self.sandbox_dir, self.input_jsonl)
        out_dir  = os.path.join(self.sandbox_dir, self.output_dir)
        os.makedirs(out_dir, exist_ok=True)

        if not os.path.exists(in_path):
            return {"status": "error",
                    "message": f"Input file not found: {in_path}"}

        # ---- load ----
        raw_records = []
        with open(in_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    raw_records.append(json.loads(line))

        if not raw_records:
            return {"status": "error", "message": "Input JSONL is empty"}

        # ---- extract & pad ----
        run_ids, histograms = [], []
        bad_runs = []
        for rec in raw_records:
            hist = rec.get("histogram", [])
            # Skip known-anomaly records from training set so model learns normal only
            if rec.get("is_synthetic_anomaly", False):
                bad_runs.append(rec.get("run_id"))
                continue
            if not hist:
                bad_runs.append(rec.get("run_id"))
                continue
            # truncate or pad
            hist = hist[: self.n_bins]
            hist = hist + [0.0] * (self.n_bins - len(hist))
            arr = np.array(hist, dtype=np.float32)
            zero_frac = float((arr == 0).sum()) / self.n_bins
            if zero_frac > self.drop_zero_fraction:
                bad_runs.append(rec.get("run_id"))
                continue
            run_ids.append(rec.get("run_id"))
            histograms.append(arr)

        if not histograms:
            return {"status": "error",
                    "message": "All runs were filtered out as bad/empty"}

        X = np.stack(histograms)   # shape (N, n_bins)

        # ---- min-max normalise ----
        x_min = float(X.min())
        x_max = float(X.max())
        denom = x_max - x_min if x_max != x_min else 1.0
        X_norm = (X - x_min) / denom

        # ---- save ----
        npy_path  = os.path.join(out_dir, "dqm_processed.npy")
        meta_path = os.path.join(out_dir, "dqm_meta.json")

        np.save(npy_path, X_norm)

        meta = {
            "run_ids":        run_ids,
            "bad_runs":       bad_runs,
            "n_runs":         len(run_ids),
            "n_bins":         self.n_bins,
            "scaler_min":     x_min,
            "scaler_max":     x_max,
            "subsystem":      raw_records[0].get("subsystem", "unknown"),
            "monitor_element": raw_records[0].get("monitor_element", ""),
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        return {
            "status":          "ok",
            "n_runs_kept":     len(run_ids),
            "n_runs_dropped":  len(bad_runs),
            "bad_runs":        bad_runs,
            "array_shape":     list(X_norm.shape),
            "processed_npy":   npy_path,
            "metadata_json":   meta_path,
        }
