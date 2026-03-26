"""
tools/dqm/dqm_mcp_tools.py
----------------------------
Registers all ML4DQM tools with the HEPTAPOD MCP server so they are
callable from Claude Code, Claude Desktop, and any other MCP client.

Usage — add to heptapod_tools.py or call directly from the MCP server:

    from tools.dqm.dqm_mcp_tools import register_dqm_tools
    register_dqm_tools(server, sandbox_dir="./sandbox")

Each tool is exposed as an MCP tool function with a JSON-schema description
that lets an LLM agent understand its inputs and outputs without reading
the source code.
"""

import json
import os
import tempfile
from typing import Any, Dict, Optional

# ── lazy imports so this module loads even without torch/numpy installed ──────

def _import_tools():
    from tools.dqm.data.cms_dqm_fetch       import CMSDQMFetchTool
    from tools.dqm.data.dqm_preprocessor    import DQMPreprocessorTool
    from tools.dqm.training.autoencoder_train import DQMAutoencoderTrainTool
    from tools.dqm.training.transformer_train import DQMTransformerTrainTool
    from tools.dqm.training.model_evaluator  import DQMModelEvaluatorTool
    from tools.dqm.deployment.realtime_monitor import DQMRealtimeMonitorTool
    from tools.dqm.deployment.alert_dispatcher import DQMAlertDispatcherTool
    return (CMSDQMFetchTool, DQMPreprocessorTool, DQMAutoencoderTrainTool,
            DQMTransformerTrainTool, DQMModelEvaluatorTool,
            DQMRealtimeMonitorTool, DQMAlertDispatcherTool)


def register_dqm_tools(server, sandbox_dir: Optional[str] = None):
    """
    Register all ML4DQM tools as MCP tool functions on `server`.

    Parameters
    ----------
    server      : An object with a `.tool()` decorator (FastMCP or compatible).
    sandbox_dir : Root directory for all tool outputs.
                  Defaults to a temporary directory if not provided.
    """
    _sandbox = sandbox_dir or tempfile.mkdtemp(prefix="heptapod_dqm_")

    (CMSDQMFetchTool, DQMPreprocessorTool, DQMAutoencoderTrainTool,
     DQMTransformerTrainTool, DQMModelEvaluatorTool,
     DQMRealtimeMonitorTool, DQMAlertDispatcherTool) = _import_tools()

    # ── 1. Fetch ──────────────────────────────────────────────────────────────
    @server.tool()
    def dqm_fetch(
        run_start: int,
        run_end: int,
        subsystem: str = "Pixel",
        output_dir: str = "data/raw",
        max_runs: int = 100,
        dataset: str = "/Global/Online/ALL",
    ) -> Dict[str, Any]:
        """
        Fetch CMS Data Quality Monitoring histograms from the public CMS DQM
        REST API for a specified run range and detector subsystem.

        Falls back to deterministic synthetic histograms when the CMS network
        is not reachable (e.g. local development).

        subsystem choices: Pixel | ECAL | Tracker | Muon | Hcal
        """
        return CMSDQMFetchTool(
            sandbox_dir=_sandbox,
            run_start=run_start,
            run_end=run_end,
            subsystem=subsystem,
            output_dir=output_dir,
            max_runs=max_runs,
            dataset=dataset,
        ).run()

    # ── 2. Preprocess ─────────────────────────────────────────────────────────
    @server.tool()
    def dqm_preprocess(
        input_jsonl: str = "data/raw/dqm_raw.jsonl",
        output_dir: str = "data/processed",
        n_bins: int = 64,
        drop_zero_fraction: float = 0.95,
    ) -> Dict[str, Any]:
        """
        Normalise and clean raw DQM histogram data for ML training.

        Reads dqm_raw.jsonl produced by dqm_fetch, pads/truncates histograms
        to n_bins, removes empty/bad runs, and min-max normalises the data.

        Outputs dqm_processed.npy (float32, shape N×n_bins) and dqm_meta.json.
        """
        return DQMPreprocessorTool(
            sandbox_dir=_sandbox,
            input_jsonl=input_jsonl,
            output_dir=output_dir,
            n_bins=n_bins,
            drop_zero_fraction=drop_zero_fraction,
        ).run()

    # ── 3a. Train autoencoder ─────────────────────────────────────────────────
    @server.tool()
    def dqm_train_autoencoder(
        processed_npy: str = "data/processed/dqm_processed.npy",
        meta_json: str = "data/processed/dqm_meta.json",
        output_dir: str = "models",
        hidden_dim: int = 32,
        latent_dim: int = 8,
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        threshold_sigma: float = 3.0,
    ) -> Dict[str, Any]:
        """
        Train a fully-connected autoencoder for unsupervised DQM anomaly detection.

        The anomaly threshold is set at mean + sigma × std of per-run
        reconstruction errors on the training set. Runs with error above
        the threshold are flagged as anomalous.

        Outputs dqm_autoencoder.pt (PyTorch state dict) and dqm_train_log.json.
        """
        return DQMAutoencoderTrainTool(
            sandbox_dir=_sandbox,
            processed_npy=processed_npy,
            meta_json=meta_json,
            output_dir=output_dir,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            threshold_sigma=threshold_sigma,
        ).run()

    # ── 3b. Train Transformer ─────────────────────────────────────────────────
    @server.tool()
    def dqm_train_transformer(
        processed_npy: str = "data/processed/dqm_processed.npy",
        meta_json: str = "data/processed/dqm_meta.json",
        output_dir: str = "models",
        d_model: int = 32,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 64,
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        threshold_sigma: float = 3.0,
    ) -> Dict[str, Any]:
        """
        Train a Transformer-based autoencoder for DQM anomaly detection.

        The Transformer uses multi-head self-attention over histogram bins,
        capturing long-range bin correlations that fully-connected models miss.
        Particularly effective for detecting structured anomalies such as
        dead η–φ regions or partial high-voltage trips.

        Outputs dqm_transformer.pt and dqm_transformer_log.json.
        """
        return DQMTransformerTrainTool(
            sandbox_dir=_sandbox,
            processed_npy=processed_npy,
            meta_json=meta_json,
            output_dir=output_dir,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            threshold_sigma=threshold_sigma,
        ).run()

    # ── 4. Evaluate ───────────────────────────────────────────────────────────
    @server.tool()
    def dqm_evaluate(
        model_path: str = "models/dqm_autoencoder.pt",
        eval_npy: str = "data/processed/dqm_processed.npy",
        eval_meta: str = "data/processed/dqm_meta.json",
        output_dir: str = "eval",
        known_bad_runs: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate a trained DQM model on a dataset.

        Computes reconstruction errors per run and flags anomalies.
        If known_bad_runs is provided as a JSON list (e.g. '[360000, 360010]'),
        also computes precision, recall, and F1 score.

        Works with both autoencoder and Transformer checkpoints.
        """
        bad_runs = None
        if known_bad_runs:
            try:
                bad_runs = json.loads(known_bad_runs)
            except json.JSONDecodeError:
                return {"status": "error",
                        "message": "known_bad_runs must be a JSON list, e.g. '[360000, 360010]'"}

        return DQMModelEvaluatorTool(
            sandbox_dir=_sandbox,
            model_path=model_path,
            eval_npy=eval_npy,
            eval_meta=eval_meta,
            output_dir=output_dir,
            known_bad_runs=bad_runs,
        ).run()

    # ── 5. Monitor ────────────────────────────────────────────────────────────
    @server.tool()
    def dqm_monitor(
        model_path: str = "models/dqm_autoencoder.pt",
        stream_jsonl: str = "data/raw/dqm_raw.jsonl",
        output_dir: str = "monitoring",
        simulate_delay_ms: int = 0,
        threshold_multiplier: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Run near-real-time DQM anomaly monitoring on a histogram stream.

        Reads records from stream_jsonl one at a time (simulating a live feed),
        scores each with the trained model, and emits structured alert JSON
        for runs above the anomaly threshold.

        Alert severities:
          WARNING  — reconstruction error between 1× and 2× threshold
          CRITICAL — reconstruction error above 2× threshold

        In production this would be connected to the CMS online DQM stream.
        """
        return DQMRealtimeMonitorTool(
            sandbox_dir=_sandbox,
            model_path=model_path,
            stream_jsonl=stream_jsonl,
            output_dir=output_dir,
            simulate_delay_ms=simulate_delay_ms,
            threshold_multiplier=threshold_multiplier,
        ).run()

    # ── 6. Dispatch ───────────────────────────────────────────────────────────
    @server.tool()
    def dqm_dispatch_alerts(
        alerts_jsonl: str = "monitoring/dqm_alerts.jsonl",
        output_dir: str = "monitoring",
        channels: str = "console,file",
        min_severity: str = "WARNING",
        webhook_url: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Dispatch DQM anomaly alerts to shift crew.

        Reads dqm_alerts.jsonl and sends formatted summaries via one or more
        channels. Supported channels: console, file, webhook.

        min_severity: WARNING (default) or CRITICAL
        webhook_url:  Mattermost or Slack webhook endpoint (optional)
        """
        channel_list = [c.strip() for c in channels.split(",") if c.strip()]
        return DQMAlertDispatcherTool(
            sandbox_dir=_sandbox,
            alerts_jsonl=alerts_jsonl,
            output_dir=output_dir,
            channels=channel_list,
            min_severity=min_severity,
            webhook_url=webhook_url,
        ).run()

    # ── 7. Full pipeline shortcut ─────────────────────────────────────────────
    @server.tool()
    def dqm_run_full_pipeline(
        subsystem: str = "Pixel",
        run_start: int = 360000,
        run_end: int = 360099,
        max_runs: int = 100,
        model_type: str = "autoencoder",
        epochs: int = 50,
    ) -> Dict[str, Any]:
        """
        Execute the complete ML4DQM pipeline in a single call:
          Fetch → Preprocess → Train → Evaluate → Monitor → Dispatch

        model_type: 'autoencoder' (default) or 'transformer'
        subsystem:  Pixel | ECAL | Tracker | Muon | Hcal

        Returns a summary dict with the results of each stage.
        """
        results = {}

        # 1. Fetch
        r = CMSDQMFetchTool(
            sandbox_dir=_sandbox, run_start=run_start, run_end=run_end,
            subsystem=subsystem, output_dir="data/raw", max_runs=max_runs,
        ).run()
        results["fetch"] = r
        if r.get("status") != "ok":
            return {"status": "error", "stage": "fetch", "detail": r}

        # 2. Preprocess
        r = DQMPreprocessorTool(
            sandbox_dir=_sandbox, input_jsonl="data/raw/dqm_raw.jsonl",
            output_dir="data/processed", n_bins=64,
        ).run()
        results["preprocess"] = r
        if r.get("status") != "ok":
            return {"status": "error", "stage": "preprocess", "detail": r}

        # 3. Train
        if model_type == "transformer":
            r = DQMTransformerTrainTool(
                sandbox_dir=_sandbox,
                processed_npy="data/processed/dqm_processed.npy",
                meta_json="data/processed/dqm_meta.json",
                output_dir="models", epochs=epochs,
            ).run()
            model_pt = "models/dqm_transformer.pt"
        else:
            r = DQMAutoencoderTrainTool(
                sandbox_dir=_sandbox,
                processed_npy="data/processed/dqm_processed.npy",
                meta_json="data/processed/dqm_meta.json",
                output_dir="models", epochs=epochs,
            ).run()
            model_pt = "models/dqm_autoencoder.pt"
        results["train"] = r
        if r.get("status") != "ok":
            return {"status": "error", "stage": "train", "detail": r}

        # 4. Monitor
        r = DQMRealtimeMonitorTool(
            sandbox_dir=_sandbox, model_path=model_pt,
            stream_jsonl="data/raw/dqm_raw.jsonl",
            output_dir="monitoring", simulate_delay_ms=0,
        ).run()
        results["monitor"] = r
        if r.get("status") != "ok":
            return {"status": "error", "stage": "monitor", "detail": r}

        # 5. Dispatch
        r = DQMAlertDispatcherTool(
            sandbox_dir=_sandbox,
            alerts_jsonl="monitoring/dqm_alerts.jsonl",
            output_dir="monitoring", channels=["file"],
        ).run()
        results["dispatch"] = r

        return {
            "status":       "ok",
            "sandbox_dir":  _sandbox,
            "subsystem":    subsystem,
            "model_type":   model_type,
            "n_runs_fetched":   results["fetch"].get("records_written", 0),
            "n_runs_trained":   results["train"].get("n_training_runs", 0),
            "n_anomalies":      results["monitor"].get("n_flagged", 0),
            "anomaly_rate_pct": results["monitor"].get("anomaly_rate_pct", 0.0),
            "stage_results":    results,
        }
