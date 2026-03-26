"""
Tool 6: DQMAlertDispatcherTool
-------------------------------
Reads the alert JSONL produced by DQMRealtimeMonitorTool and dispatches
human-readable summaries to one or more channels:

  - console  : prints formatted table to stdout
  - file     : writes a plain-text report to dqm_alert_report.txt
  - webhook  : HTTP POST to a configurable URL (e.g. Mattermost / Slack)

In a production CMS environment the webhook would target the DQM Mattermost
channel or the CMS on-call pager endpoint.
"""

import json
import os
from datetime import datetime, timezone
from typing import ClassVar, Dict, Optional, List

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

from .._compat import BaseTool, RuntimeField, StateField


class DQMAlertDispatcherTool(BaseTool):
    """
    Dispatches DQM anomaly alerts to shift crew and DQM experts.
    Reads dqm_alerts.jsonl, formats summaries, and sends them via
    one or more output channels (console, file, webhook).

    Severity filtering: only alerts at or above min_severity are dispatched.
    """

    # --- RuntimeFields ---
    alerts_jsonl: str = RuntimeField(
        description="Path to dqm_alerts.jsonl from DQMRealtimeMonitorTool "
                    "(relative to sandbox_dir)")
    output_dir: str = RuntimeField(
        description="Subdirectory for text report output")
    channels: List[str] = RuntimeField(
        default=["console", "file"],
        description="Dispatch channels: any of ['console', 'file', 'webhook']")
    min_severity: str = RuntimeField(
        default="WARNING",
        description="Minimum severity to dispatch: 'WARNING' or 'CRITICAL'")
    webhook_url: Optional[str] = RuntimeField(
        default=None,
        description="HTTP POST endpoint for webhook channel (Mattermost / Slack)")
    max_alerts_per_dispatch: int = RuntimeField(
        default=20,
        description="Cap dispatched alerts to avoid flooding channels")

    # --- StateField ---
    sandbox_dir: str = StateField(
        description="Root sandbox directory for this HEPTAPOD session")

    SEVERITY_RANK: ClassVar[Dict[str, int]] = {"WARNING": 1, "CRITICAL": 2}

    # ------------------------------------------------------------------ #
    def run(self) -> dict:
        alerts_path = os.path.join(self.sandbox_dir, self.alerts_jsonl)
        out_dir     = os.path.join(self.sandbox_dir, self.output_dir)
        os.makedirs(out_dir, exist_ok=True)

        if not os.path.exists(alerts_path):
            return {"status": "error",
                    "message": f"Alerts file not found: {alerts_path}"}

        # ── load + filter alerts ──
        min_rank = self.SEVERITY_RANK.get(self.min_severity, 1)
        alerts   = []
        with open(alerts_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                a = json.loads(line)
                if self.SEVERITY_RANK.get(a.get("severity", ""), 0) >= min_rank:
                    alerts.append(a)

        alerts = alerts[:self.max_alerts_per_dispatch]

        if not alerts:
            # Always create the report file even when empty (required by test suite)
            report_path = os.path.join(out_dir, "dqm_alert_report.txt")
            with open(report_path, "w") as f:
                f.write("No alerts met the severity threshold.\n")
            return {
                "status":         "ok",
                "n_dispatched":   0,
                "message":        "No alerts met the severity threshold",
                "channels_used":  [],
                "report_path":    report_path,
            }

        report_text = self._format_report(alerts)
        channels_used = []

        if "console" in self.channels:
            print(report_text)
            channels_used.append("console")

        if "file" in self.channels:
            report_path = os.path.join(out_dir, "dqm_alert_report.txt")
            with open(report_path, "w") as f:
                f.write(report_text)
            channels_used.append("file")

        if "webhook" in self.channels and self.webhook_url:
            if REQUESTS_AVAILABLE:
                try:
                    payload = {"text": f"```\n{report_text[:3000]}\n```"}
                    resp    = requests.post(self.webhook_url, json=payload, timeout=5)
                    channels_used.append(
                        f"webhook({resp.status_code})")
                except Exception as exc:
                    channels_used.append(f"webhook(FAILED: {exc})")
            else:
                channels_used.append("webhook(SKIPPED: requests not installed)")

        return {
            "status":        "ok",
            "n_dispatched":  len(alerts),
            "channels_used": channels_used,
            "report_preview": report_text[:500],
        }

    # ------------------------------------------------------------------ #
    def _format_report(self, alerts: list) -> str:
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        header = (
            f"{'='*60}\n"
            f"  CMS DQM ANOMALY ALERT REPORT — {now}\n"
            f"{'='*60}\n"
            f"  {len(alerts)} alert(s) dispatched "
            f"(min severity: {self.min_severity})\n"
            f"{'='*60}\n\n"
        )
        rows = []
        for a in alerts:
            rows.append(
                f"  [{a.get('severity','?'):8s}] Run {a.get('run_id','?'):>8} | "
                f"Subsystem: {a.get('subsystem','?'):8s} | "
                f"Error: {a.get('recon_error', 0):.4f} "
                f"(threshold: {a.get('threshold', 0):.4f})\n"
                f"           Action: {a.get('action','')}\n"
            )
        footer = f"\n{'='*60}\n"
        return header + "\n".join(rows) + footer
