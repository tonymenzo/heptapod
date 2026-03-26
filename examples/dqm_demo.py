"""
DQM Pipeline Demo
-----------------
Demonstrates the full ML4DQM workflow end-to-end without needing
the orchestral_ai framework or a real CMS connection:

  Fetch (synthetic) → Preprocess → Train → Evaluate → Monitor → Dispatch

Run from the repo root:
    python examples/dqm_demo.py
"""

import json
import os
import sys
import tempfile

# ── allow running from repo root ──────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tools.dqm.data.cms_dqm_fetch       import CMSDQMFetchTool
from tools.dqm.data.dqm_preprocessor    import DQMPreprocessorTool
from tools.dqm.training.autoencoder_train import DQMAutoencoderTrainTool
from tools.dqm.training.model_evaluator  import DQMModelEvaluatorTool
from tools.dqm.deployment.realtime_monitor import DQMRealtimeMonitorTool
from tools.dqm.deployment.alert_dispatcher import DQMAlertDispatcherTool


def run_tool(tool, label: str) -> dict:
    print(f"\n{'─'*55}")
    print(f"  Running: {label}")
    print(f"{'─'*55}")
    result = tool.run()
    for k, v in result.items():
        if k not in ("per_run",):   # skip long lists
            print(f"  {k:30s}: {v}")
    if result.get("status") == "error":
        raise RuntimeError(f"{label} failed: {result.get('message')}")
    return result


def main():
    sandbox = tempfile.mkdtemp(prefix="heptapod_dqm_demo_")
    print(f"\n  Sandbox: {sandbox}")

    # ── shared StateField value ──────────────────────────────────────────────
    S = sandbox   # convenience alias

    # ── 1. Fetch ─────────────────────────────────────────────────────────────
    fetch_result = run_tool(
        CMSDQMFetchTool(
            sandbox_dir=S,
            run_start=360000, run_end=360099,
            subsystem="Pixel",
            output_dir="data/raw",
            max_runs=100,
        ),
        "CMSDQMFetchTool (synthetic histograms)",
    )

    # ── 2. Preprocess ─────────────────────────────────────────────────────────
    prep_result = run_tool(
        DQMPreprocessorTool(
            sandbox_dir=S,
            input_jsonl="data/raw/dqm_raw.jsonl",
            output_dir="data/processed",
            n_bins=64,
        ),
        "DQMPreprocessorTool",
    )

    # ── 3. Train ──────────────────────────────────────────────────────────────
    train_result = run_tool(
        DQMAutoencoderTrainTool(
            sandbox_dir=S,
            processed_npy="data/processed/dqm_processed.npy",
            meta_json="data/processed/dqm_meta.json",
            output_dir="models",
            hidden_dim=32, latent_dim=8,
            epochs=30, batch_size=16,
            learning_rate=1e-3,
            threshold_sigma=3.0,
        ),
        "DQMAutoencoderTrainTool",
    )

    # ── 4. Evaluate ───────────────────────────────────────────────────────────
    run_tool(
        DQMModelEvaluatorTool(
            sandbox_dir=S,
            model_path="models/dqm_autoencoder.pt",
            eval_npy="data/processed/dqm_processed.npy",
            eval_meta="data/processed/dqm_meta.json",
            output_dir="eval",
            # runs 360000, 360010, 360020 ... are synthetic anomalies
            known_bad_runs=[360000 + i*10 for i in range(10)],
        ),
        "DQMModelEvaluatorTool",
    )

    # ── 5. Simulate real-time stream ─────────────────────────────────────────
    # Re-use the raw fetch data as the "live" stream
    stream_path = os.path.join(S, "data/raw/dqm_raw.jsonl")
    run_tool(
        DQMRealtimeMonitorTool(
            sandbox_dir=S,
            model_path="models/dqm_autoencoder.pt",
            stream_jsonl="data/raw/dqm_raw.jsonl",
            output_dir="monitoring",
            simulate_delay_ms=0,
        ),
        "DQMRealtimeMonitorTool",
    )

    # ── 6. Dispatch alerts ───────────────────────────────────────────────────
    run_tool(
        DQMAlertDispatcherTool(
            sandbox_dir=S,
            alerts_jsonl="monitoring/dqm_alerts.jsonl",
            output_dir="monitoring",
            channels=["console", "file"],
            min_severity="WARNING",
        ),
        "DQMAlertDispatcherTool",
    )

    print(f"\n{'='*55}")
    print("  ML4DQM pipeline demo completed successfully!")
    print(f"  All outputs written to: {sandbox}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
