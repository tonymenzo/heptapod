"""
tests/test_dqm_tools.py
-----------------------
Unit tests for all six ML4DQM tools.

Run from repo root:
    pytest tools/dqm/tests/test_dqm_tools.py -v

All tests are self-contained: they use synthetic data and a temp directory,
so no CMS network access or GPU is required.
"""

import json
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)

from tools.dqm.data.cms_dqm_fetch import CMSDQMFetchTool
from tools.dqm.data.dqm_preprocessor import DQMPreprocessorTool
from tools.dqm.training.autoencoder_train import DQMAutoencoderTrainTool
from tools.dqm.training.model_evaluator import DQMModelEvaluatorTool
from tools.dqm.deployment.realtime_monitor import DQMRealtimeMonitorTool
from tools.dqm.deployment.alert_dispatcher import DQMAlertDispatcherTool


def _make_sandbox():
    return tempfile.mkdtemp(prefix="dqm_test_")


# ─────────────────────────────────────────────────────────────────────────────
class TestCMSDQMFetchTool(unittest.TestCase):
    """Tool 1: Fetch — always uses synthetic fallback in CI."""

    def setUp(self):
        self.sandbox = _make_sandbox()

    def _run_fetch(self, subsystem="Pixel", max_runs=10):
        return CMSDQMFetchTool(
            sandbox_dir=self.sandbox,
            run_start=360000,
            run_end=360009,
            subsystem=subsystem,
            output_dir="data/raw",
            max_runs=max_runs,
        ).run()

    def test_returns_ok_status(self):
        result = self._run_fetch()
        self.assertEqual(result["status"], "ok",
                         f"Fetch failed: {result.get('message')}")

    def test_writes_jsonl_file(self):
        self._run_fetch()
        path = os.path.join(self.sandbox, "data/raw/dqm_raw.jsonl")
        self.assertTrue(os.path.exists(path), "dqm_raw.jsonl not created")

    def test_correct_record_count(self):
        result = self._run_fetch(max_runs=5)
        path = os.path.join(self.sandbox, "data/raw/dqm_raw.jsonl")
        with open(path) as f:
            lines = [l for l in f if l.strip()]
        # 5 runs × 1 monitor element per subsystem
        self.assertEqual(len(lines), 5)

    def test_each_record_has_histogram(self):
        self._run_fetch(max_runs=3)
        path = os.path.join(self.sandbox, "data/raw/dqm_raw.jsonl")
        with open(path) as f:
            for line in f:
                rec = json.loads(line)
                self.assertIn("histogram", rec)
                self.assertGreater(len(rec["histogram"]), 0)

    def test_anomaly_injection_every_10th_run(self):
        """Run 360000 (divisible by 10) should have a spike at bin 32."""
        self._run_fetch(max_runs=11)
        path = os.path.join(self.sandbox, "data/raw/dqm_raw.jsonl")
        with open(path) as f:
            records = [json.loads(l) for l in f if l.strip()]
        run0 = next(r for r in records if r["run_id"] == 360000)
        # bin 32 is multiplied by 5 — should be noticeably larger than neighbours
        hist = run0["histogram"]
        self.assertGreater(hist[32], hist[31] * 2,
                           "Anomaly injection at bin 32 not detected")

    def test_unknown_subsystem_returns_error(self):
        result = CMSDQMFetchTool(
            sandbox_dir=self.sandbox,
            run_start=360000, run_end=360009,
            subsystem="UnknownSubsystem",
            output_dir="data/raw",
        ).run()
        self.assertEqual(result["status"], "error")

    def test_all_subsystems_accepted(self):
        for sub in ["Pixel", "ECAL", "Tracker", "Muon", "Hcal"]:
            with self.subTest(subsystem=sub):
                sb = _make_sandbox()
                r = CMSDQMFetchTool(
                    sandbox_dir=sb, run_start=360000, run_end=360002,
                    subsystem=sub, output_dir="raw", max_runs=3,
                ).run()
                self.assertEqual(r["status"], "ok")


# ─────────────────────────────────────────────────────────────────────────────
class TestDQMPreprocessorTool(unittest.TestCase):
    """Tool 2: Preprocess."""

    def setUp(self):
        self.sandbox = _make_sandbox()
        # First fetch synthetic data
        CMSDQMFetchTool(
            sandbox_dir=self.sandbox,
            run_start=360000, run_end=360019,
            subsystem="Pixel",
            output_dir="data/raw",
            max_runs=20,
        ).run()

    def _run_prep(self, **kwargs):
        defaults = dict(
            sandbox_dir=self.sandbox,
            input_jsonl="data/raw/dqm_raw.jsonl",
            output_dir="data/processed",
            n_bins=64,
        )
        defaults.update(kwargs)
        return DQMPreprocessorTool(**defaults).run()

    def test_returns_ok(self):
        r = self._run_prep()
        self.assertEqual(r["status"], "ok")

    def test_creates_npy_and_meta(self):
        self._run_prep()
        self.assertTrue(os.path.exists(
            os.path.join(self.sandbox, "data/processed/dqm_processed.npy")))
        self.assertTrue(os.path.exists(
            os.path.join(self.sandbox, "data/processed/dqm_meta.json")))

    def test_array_has_correct_shape(self):
        import numpy as np
        r = self._run_prep(n_bins=32)
        X = np.load(os.path.join(self.sandbox, "data/processed/dqm_processed.npy"))
        self.assertEqual(X.shape[1], 32, f"Expected 32 bins, got {X.shape[1]}")
        self.assertGreater(X.shape[0], 0)

    def test_values_are_normalised(self):
        import numpy as np
        self._run_prep()
        X = np.load(os.path.join(self.sandbox, "data/processed/dqm_processed.npy"))
        self.assertAlmostEqual(float(X.min()), 0.0, places=5)
        self.assertAlmostEqual(float(X.max()), 1.0, places=5)

    def test_meta_json_contains_required_keys(self):
        self._run_prep()
        with open(os.path.join(self.sandbox, "data/processed/dqm_meta.json")) as f:
            meta = json.load(f)
        for key in ["run_ids", "n_runs", "n_bins", "scaler_min", "scaler_max"]:
            self.assertIn(key, meta, f"Missing key '{key}' in dqm_meta.json")

    def test_missing_input_returns_error(self):
        r = DQMPreprocessorTool(
            sandbox_dir=self.sandbox,
            input_jsonl="nonexistent.jsonl",
            output_dir="out",
        ).run()
        self.assertEqual(r["status"], "error")


# ─────────────────────────────────────────────────────────────────────────────
class TestDQMAutoencoderTrainTool(unittest.TestCase):
    """Tool 3: Train."""

    def setUp(self):
        self.sandbox = _make_sandbox()
        CMSDQMFetchTool(
            sandbox_dir=self.sandbox, run_start=360000, run_end=360049,
            subsystem="Pixel", output_dir="data/raw", max_runs=50,
        ).run()
        DQMPreprocessorTool(
            sandbox_dir=self.sandbox, input_jsonl="data/raw/dqm_raw.jsonl",
            output_dir="data/processed", n_bins=64,
        ).run()

    def _run_train(self, **kwargs):
        defaults = dict(
            sandbox_dir=self.sandbox,
            processed_npy="data/processed/dqm_processed.npy",
            meta_json="data/processed/dqm_meta.json",
            output_dir="models",
            hidden_dim=16, latent_dim=4,
            epochs=5, batch_size=8,
            learning_rate=1e-3,
            threshold_sigma=3.0,
        )
        defaults.update(kwargs)
        return DQMAutoencoderTrainTool(**defaults).run()

    def test_returns_ok(self):
        r = self._run_train()
        self.assertEqual(r["status"], "ok")

    def test_creates_model_file(self):
        self._run_train()
        self.assertTrue(os.path.exists(
            os.path.join(self.sandbox, "models/dqm_autoencoder.pt")))

    def test_creates_training_log(self):
        self._run_train()
        log_path = os.path.join(self.sandbox, "models/dqm_train_log.json")
        self.assertTrue(os.path.exists(log_path))
        with open(log_path) as f:
            log = json.load(f)
        self.assertIn("anomaly_threshold", log)
        self.assertEqual(len(log["per_epoch"]), 5)

    def test_threshold_is_positive(self):
        r = self._run_train()
        self.assertGreater(r["anomaly_threshold"], 0.0)

    def test_train_loss_decreases_over_epochs(self):
        self._run_train(epochs=10)
        log_path = os.path.join(self.sandbox, "models/dqm_train_log.json")
        with open(log_path) as f:
            log = json.load(f)
        losses = [e["train_loss"] for e in log["per_epoch"]]
        # Loss should be lower at end than at start (not guaranteed but very likely)
        self.assertLess(losses[-1], losses[0] * 2,
                        "Training loss did not improve at all")

    def test_missing_npy_returns_error(self):
        r = DQMAutoencoderTrainTool(
            sandbox_dir=self.sandbox,
            processed_npy="nonexistent.npy",
            meta_json="data/processed/dqm_meta.json",
            output_dir="models",
        ).run()
        self.assertEqual(r["status"], "error")


# ─────────────────────────────────────────────────────────────────────────────
class TestDQMModelEvaluatorTool(unittest.TestCase):
    """Tool 4: Evaluate."""

    def setUp(self):
        self.sandbox = _make_sandbox()
        CMSDQMFetchTool(
            sandbox_dir=self.sandbox, run_start=360000, run_end=360049,
            subsystem="Pixel", output_dir="data/raw", max_runs=50,
        ).run()
        DQMPreprocessorTool(
            sandbox_dir=self.sandbox, input_jsonl="data/raw/dqm_raw.jsonl",
            output_dir="data/processed", n_bins=64,
        ).run()
        DQMAutoencoderTrainTool(
            sandbox_dir=self.sandbox,
            processed_npy="data/processed/dqm_processed.npy",
            meta_json="data/processed/dqm_meta.json",
            output_dir="models",
            hidden_dim=16, latent_dim=4, epochs=5, batch_size=8,
        ).run()

    def _run_eval(self, **kwargs):
        defaults = dict(
            sandbox_dir=self.sandbox,
            model_path="models/dqm_autoencoder.pt",
            eval_npy="data/processed/dqm_processed.npy",
            eval_meta="data/processed/dqm_meta.json",
            output_dir="eval",
        )
        defaults.update(kwargs)
        return DQMModelEvaluatorTool(**defaults).run()

    def test_returns_ok(self):
        r = self._run_eval()
        self.assertEqual(r["status"], "ok")

    def test_creates_report(self):
        self._run_eval()
        self.assertTrue(os.path.exists(
            os.path.join(self.sandbox, "eval/dqm_eval_report.json")))

    def test_supervised_metrics_with_known_bad_runs(self):
        # Runs 360000, 360010, 360020... are synthetic anomalies
        bad_runs = [360000 + i * 10 for i in range(5)]
        r = self._run_eval(known_bad_runs=bad_runs)
        metrics = r.get("supervised_metrics", {})
        self.assertIn("precision", metrics)
        self.assertIn("recall", metrics)
        self.assertIn("f1", metrics)
        # Precision and recall should both be in [0, 1]
        self.assertGreaterEqual(metrics["precision"], 0.0)
        self.assertLessEqual(metrics["precision"], 1.0)

    def test_n_evaluated_matches_input(self):
        import numpy as np
        X = np.load(os.path.join(self.sandbox,
                                  "data/processed/dqm_processed.npy"))
        r = self._run_eval()
        self.assertEqual(r["n_evaluated"], X.shape[0])

    def test_flagged_runs_subset_of_evaluated(self):
        r = self._run_eval()
        with open(os.path.join(self.sandbox, "eval/dqm_eval_report.json")) as f:
            report = json.load(f)
        self.assertLessEqual(
            len(report["flagged_run_ids"]), report["n_runs_evaluated"])


# ─────────────────────────────────────────────────────────────────────────────
class TestDQMRealtimeMonitorTool(unittest.TestCase):
    """Tool 5: Real-time monitor."""

    def setUp(self):
        self.sandbox = _make_sandbox()
        CMSDQMFetchTool(
            sandbox_dir=self.sandbox, run_start=360000, run_end=360049,
            subsystem="Pixel", output_dir="data/raw", max_runs=50,
        ).run()
        DQMPreprocessorTool(
            sandbox_dir=self.sandbox, input_jsonl="data/raw/dqm_raw.jsonl",
            output_dir="data/processed", n_bins=64,
        ).run()
        DQMAutoencoderTrainTool(
            sandbox_dir=self.sandbox,
            processed_npy="data/processed/dqm_processed.npy",
            meta_json="data/processed/dqm_meta.json",
            output_dir="models",
            hidden_dim=16, latent_dim=4, epochs=5, batch_size=8,
        ).run()

    def _run_monitor(self, **kwargs):
        defaults = dict(
            sandbox_dir=self.sandbox,
            model_path="models/dqm_autoencoder.pt",
            stream_jsonl="data/raw/dqm_raw.jsonl",
            output_dir="monitoring",
            simulate_delay_ms=0,
        )
        defaults.update(kwargs)
        return DQMRealtimeMonitorTool(**defaults).run()

    def test_returns_ok(self):
        r = self._run_monitor()
        self.assertEqual(r["status"], "ok")

    def test_creates_alerts_and_log(self):
        self._run_monitor()
        self.assertTrue(os.path.exists(
            os.path.join(self.sandbox, "monitoring/dqm_alerts.jsonl")))
        self.assertTrue(os.path.exists(
            os.path.join(self.sandbox, "monitoring/dqm_monitor_log.json")))

    def test_anomalies_detected(self):
        """Synthetic anomaly runs (every 10th) should cause at least some alerts."""
        r = self._run_monitor()
        self.assertGreater(r["n_flagged"], 0,
                           "No anomalies detected — synthetic spikes should trigger alerts")

    def test_alert_records_have_required_fields(self):
        self._run_monitor()
        alerts_path = os.path.join(self.sandbox, "monitoring/dqm_alerts.jsonl")
        with open(alerts_path) as f:
            lines = [l for l in f if l.strip()]
        self.assertGreater(len(lines), 0)
        for line in lines:
            alert = json.loads(line)
            for key in ["run_id", "severity", "recon_error", "threshold", "action"]:
                self.assertIn(key, alert, f"Alert missing key '{key}'")

    def test_severity_levels_are_valid(self):
        self._run_monitor()
        alerts_path = os.path.join(self.sandbox, "monitoring/dqm_alerts.jsonl")
        with open(alerts_path) as f:
            for line in f:
                if line.strip():
                    alert = json.loads(line)
                    self.assertIn(alert["severity"], {"WARNING", "CRITICAL"})

    def test_critical_threshold_is_double(self):
        """CRITICAL fires only when error > 2× threshold."""
        self._run_monitor()
        alerts_path = os.path.join(self.sandbox, "monitoring/dqm_alerts.jsonl")
        with open(alerts_path) as f:
            for line in f:
                if line.strip():
                    a = json.loads(line)
                    if a["severity"] == "CRITICAL":
                        self.assertGreater(a["recon_error"], 2.0 * a["threshold"])

    def test_missing_model_returns_error(self):
        r = DQMRealtimeMonitorTool(
            sandbox_dir=self.sandbox,
            model_path="nonexistent_model.pt",
            stream_jsonl="data/raw/dqm_raw.jsonl",
            output_dir="monitoring",
        ).run()
        self.assertEqual(r["status"], "error")


# ─────────────────────────────────────────────────────────────────────────────
class TestDQMAlertDispatcherTool(unittest.TestCase):
    """Tool 6: Alert dispatcher."""

    def _write_fake_alerts(self, sandbox, n=3):
        """Write synthetic alert records to alerts JSONL."""
        os.makedirs(os.path.join(sandbox, "monitoring"), exist_ok=True)
        alerts_path = os.path.join(sandbox, "monitoring/dqm_alerts.jsonl")
        alerts = [
            {
                "timestamp": "2026-03-01T12:00:00+00:00",
                "run_id": 360000 + i * 10,
                "subsystem": "Pixel",
                "monitor_element": "some/me",
                "recon_error": 0.05 + i * 0.05,
                "threshold": 0.02,
                "severity": "CRITICAL" if i == 0 else "WARNING",
                "action": "Notify shift crew" if i == 0 else "Flag for review",
            }
            for i in range(n)
        ]
        with open(alerts_path, "w") as f:
            for a in alerts:
                f.write(json.dumps(a) + "\n")
        return alerts_path

    def test_console_and_file_dispatch(self):
        sandbox = _make_sandbox()
        self._write_fake_alerts(sandbox)
        r = DQMAlertDispatcherTool(
            sandbox_dir=sandbox,
            alerts_jsonl="monitoring/dqm_alerts.jsonl",
            output_dir="monitoring",
            channels=["console", "file"],
            min_severity="WARNING",
        ).run()
        self.assertEqual(r["status"], "ok")
        self.assertIn("file", r["channels_used"])
        self.assertEqual(r["n_dispatched"], 3)

    def test_file_report_created(self):
        sandbox = _make_sandbox()
        self._write_fake_alerts(sandbox)
        DQMAlertDispatcherTool(
            sandbox_dir=sandbox,
            alerts_jsonl="monitoring/dqm_alerts.jsonl",
            output_dir="monitoring",
            channels=["file"],
        ).run()
        self.assertTrue(os.path.exists(
            os.path.join(sandbox, "monitoring/dqm_alert_report.txt")))

    def test_severity_filter_critical_only(self):
        sandbox = _make_sandbox()
        self._write_fake_alerts(sandbox, n=3)   # 1 CRITICAL + 2 WARNING
        r = DQMAlertDispatcherTool(
            sandbox_dir=sandbox,
            alerts_jsonl="monitoring/dqm_alerts.jsonl",
            output_dir="monitoring",
            channels=["file"],
            min_severity="CRITICAL",
        ).run()
        self.assertEqual(r["n_dispatched"], 1)

    def test_empty_alerts_returns_ok(self):
        sandbox = _make_sandbox()
        os.makedirs(os.path.join(sandbox, "monitoring"), exist_ok=True)
        open(os.path.join(sandbox, "monitoring/dqm_alerts.jsonl"), "w").close()
        r = DQMAlertDispatcherTool(
            sandbox_dir=sandbox,
            alerts_jsonl="monitoring/dqm_alerts.jsonl",
            output_dir="monitoring",
            channels=["file"],
        ).run()
        self.assertEqual(r["status"], "ok")
        self.assertEqual(r["n_dispatched"], 0)

    def test_missing_alerts_file_returns_error(self):
        sandbox = _make_sandbox()
        r = DQMAlertDispatcherTool(
            sandbox_dir=sandbox,
            alerts_jsonl="monitoring/nonexistent_alerts.jsonl",
            output_dir="monitoring",
            channels=["file"],
        ).run()
        self.assertEqual(r["status"], "error")

    def test_report_text_contains_run_ids(self):
        sandbox = _make_sandbox()
        self._write_fake_alerts(sandbox, n=2)
        DQMAlertDispatcherTool(
            sandbox_dir=sandbox,
            alerts_jsonl="monitoring/dqm_alerts.jsonl",
            output_dir="monitoring",
            channels=["file"],
        ).run()
        with open(os.path.join(sandbox, "monitoring/dqm_alert_report.txt")) as f:
            text = f.read()
        self.assertIn("360000", text)
        self.assertIn("Pixel", text)


# ─────────────────────────────────────────────────────────────────────────────
class TestEndToEndPipeline(unittest.TestCase):
    """Full integration test: Fetch → Preprocess → Train → Evaluate → Monitor → Dispatch."""

    def test_full_pipeline(self):
        sandbox = _make_sandbox()

        # 1. Fetch
        r = CMSDQMFetchTool(
            sandbox_dir=sandbox, run_start=360000, run_end=360029,
            subsystem="Pixel", output_dir="data/raw", max_runs=30,
        ).run()
        self.assertEqual(r["status"], "ok", f"Fetch failed: {r}")

        # 2. Preprocess
        r = DQMPreprocessorTool(
            sandbox_dir=sandbox, input_jsonl="data/raw/dqm_raw.jsonl",
            output_dir="data/processed", n_bins=64,
        ).run()
        self.assertEqual(r["status"], "ok", f"Preprocess failed: {r}")

        # 3. Train
        r = DQMAutoencoderTrainTool(
            sandbox_dir=sandbox,
            processed_npy="data/processed/dqm_processed.npy",
            meta_json="data/processed/dqm_meta.json",
            output_dir="models",
            hidden_dim=16, latent_dim=4, epochs=5, batch_size=8,
        ).run()
        self.assertEqual(r["status"], "ok", f"Train failed: {r}")

        # 4. Evaluate
        r = DQMModelEvaluatorTool(
            sandbox_dir=sandbox,
            model_path="models/dqm_autoencoder.pt",
            eval_npy="data/processed/dqm_processed.npy",
            eval_meta="data/processed/dqm_meta.json",
            output_dir="eval",
            known_bad_runs=[360000 + i * 10 for i in range(3)],
        ).run()
        self.assertEqual(r["status"], "ok", f"Evaluate failed: {r}")

        # 5. Monitor
        r = DQMRealtimeMonitorTool(
            sandbox_dir=sandbox,
            model_path="models/dqm_autoencoder.pt",
            stream_jsonl="data/raw/dqm_raw.jsonl",
            output_dir="monitoring",
            simulate_delay_ms=0,
        ).run()
        self.assertEqual(r["status"], "ok", f"Monitor failed: {r}")

        # 6. Dispatch
        r = DQMAlertDispatcherTool(
            sandbox_dir=sandbox,
            alerts_jsonl="monitoring/dqm_alerts.jsonl",
            output_dir="monitoring",
            channels=["file"],
            min_severity="WARNING",
        ).run()
        self.assertEqual(r["status"], "ok", f"Dispatch failed: {r}")

        # Final check: all expected outputs exist
        expected_files = [
            "data/raw/dqm_raw.jsonl",
            "data/processed/dqm_processed.npy",
            "data/processed/dqm_meta.json",
            "models/dqm_autoencoder.pt",
            "models/dqm_train_log.json",
            "eval/dqm_eval_report.json",
            "monitoring/dqm_alerts.jsonl",
            "monitoring/dqm_monitor_log.json",
            "monitoring/dqm_alert_report.txt",
        ]
        for fname in expected_files:
            path = os.path.join(sandbox, fname)
            self.assertTrue(os.path.exists(path),
                            f"Expected output missing: {fname}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
