"""
# audit_tool.py is a part of the HEPTAPOD package.
# Copyright (C) 2026 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

Audit-trail tool — record / render / read a structured provenance ledger.

Exposes the audit.py ledger to the agent as an MCP tool so an end-to-end run
records its own provenance (paper id, extracted model, generated .fr/UFO,
validation gate outcomes) into a machine-readable audit.json the mentors and the
eval harness can read back. See tools/logging/audit.py for the ledger format.
"""

from __future__ import annotations

import json
import os
from typing import Optional

from orchestral.tools.base.tool import BaseTool
from orchestral.tools.base.field_utils import RuntimeField, StateField

from tools.logging.audit import (
    AUDIT_FILENAME,
    AUDIT_SCHEMA,
    append_event,
    read_audit,
    render_audit_md,
)

SCHEMA_VERSION = AUDIT_SCHEMA


def _safe_join(base_directory: str, rel: str) -> Optional[str]:
    """Resolve rel against base_directory, rejecting traversal outside it."""
    if not rel:
        return None
    base = os.path.realpath(base_directory)
    full = os.path.realpath(os.path.join(base, rel))
    if full != base and not full.startswith(base + os.sep):
        return None
    return full


class AuditTrailTool(BaseTool):
    """
    Record, render, or read a structured provenance ledger (audit.json).

    Actions:
      - record: append one event {stage, status, summary, data} to the ledger.
      - render: return the ledger as a markdown timeline (also written as a
        sibling <ledger>.md file).
      - read:   return the full ledger JSON.

    Input:
        action: "record" (default) | "render" | "read".
        stage: Stage name for record (e.g. "extract", "validate", "width_gate").
        status: Outcome for record ("ok" | "fail" | "info" | "error" | "skip").
        summary: One-line human-readable summary for record.
        data: Optional JSON string of provenance (ids, hashes, checks) for record.
        ledger_name: Ledger filename relative to base_directory (default audit.json)
            — set a per-case name to keep separate run trails.

    State:
        base_directory.

    Returns JSON with status "ok" and the action's payload, or a formatted error.
    """

    # ======================== Runtime fields ======================== #
    action: Optional[str] = RuntimeField(
        default="record", description="record | render | read"
    )
    stage: Optional[str] = RuntimeField(
        default=None, description="Pipeline stage name (required for record)"
    )
    status: Optional[str] = RuntimeField(
        default="info", description="Outcome: ok | fail | info | error | skip"
    )
    summary: Optional[str] = RuntimeField(
        default=None, description="One-line human-readable summary (record)"
    )
    data: Optional[str] = RuntimeField(
        default=None,
        description="Optional JSON string of provenance data to store on the event",
    )
    ledger_name: Optional[str] = RuntimeField(
        default=AUDIT_FILENAME,
        description="Ledger filename relative to base_directory (default audit.json)",
    )
    # ================================================================ #

    # ========================= State fields ========================= #
    base_directory: str = StateField(description="Base sandbox directory")
    # ================================================================ #

    def _run(self) -> str:
        ledger_name = self.ledger_name or AUDIT_FILENAME
        # Containment check: ledger_name must resolve inside base_directory.
        if _safe_join(self.base_directory, ledger_name) is None:
            return self.format_error(
                error="Access Denied",
                reason="ledger_name escapes base_directory",
                context=ledger_name,
            )

        action = (self.action or "record").lower()

        if action == "record":
            return self._record(ledger_name)
        if action == "read":
            return self._read(ledger_name)
        if action == "render":
            return self._render(ledger_name)
        return self.format_error(
            error="Invalid Input",
            reason="action must be one of: record, render, read",
            context=action,
        )

    def _record(self, ledger_name: str) -> str:
        if not self.stage:
            return self.format_error(
                error="Missing Parameter",
                reason="stage is required when action=record",
            )
        data = None
        if self.data:
            try:
                data = json.loads(self.data)
            except json.JSONDecodeError as e:
                return self.format_error(
                    error="Invalid Input",
                    reason=f"data is not valid JSON: {e}",
                )
            if not isinstance(data, dict):
                return self.format_error(
                    error="Invalid Input",
                    reason="data must be a JSON object",
                )
        try:
            event = append_event(
                self.base_directory,
                stage=self.stage,
                status=self.status or "info",
                summary=self.summary,
                data=data,
                ledger_name=ledger_name,
            )
            ledger = read_audit(self.base_directory, ledger_name)
        except (ValueError, OSError) as e:
            return self.format_error(error="Audit Write Failed", reason=str(e))

        return json.dumps(
            {
                "status": "ok",
                "schema": SCHEMA_VERSION,
                "action": "record",
                "event": event,
                "n_events": len(ledger["events"]),
                "audit_path": ledger_name,
            },
            indent=2,
        )

    def _read(self, ledger_name: str) -> str:
        try:
            ledger = read_audit(self.base_directory, ledger_name)
        except (ValueError, OSError) as e:
            return self.format_error(error="Audit Read Failed", reason=str(e))
        return json.dumps(
            {"status": "ok", "schema": SCHEMA_VERSION, "action": "read", "ledger": ledger},
            indent=2,
        )

    def _render(self, ledger_name: str) -> str:
        try:
            ledger = read_audit(self.base_directory, ledger_name)
        except (ValueError, OSError) as e:
            return self.format_error(error="Audit Read Failed", reason=str(e))
        md = render_audit_md(ledger)
        md_rel = None
        md_full = _safe_join(
            self.base_directory, os.path.splitext(ledger_name)[0] + ".md"
        )
        if md_full is not None:
            try:
                with open(md_full, "w", encoding="utf-8") as fh:
                    fh.write(md)
                md_rel = os.path.relpath(md_full, os.path.realpath(self.base_directory))
            except OSError:
                md_rel = None
        return json.dumps(
            {
                "status": "ok",
                "schema": SCHEMA_VERSION,
                "action": "render",
                "markdown": md,
                "audit_md_path": md_rel,
                "n_events": len(ledger["events"]),
            },
            indent=2,
        )
