"""
# audit.py is a part of the HEPTAPOD package.
# Copyright (C) 2026 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

Provenance ledger — a structured, machine-readable record of pipeline stages.

Where findings.py is a free-form markdown scratchpad for the agent, this is a
structured JSON ledger (audit.json) recording one event per pipeline stage:
which paper was ingested, which model was extracted, which .fr/UFO was produced,
and the pass/fail of each validation gate. It gives an end-to-end run a
reproducible audit trail (task 6 of the HEPSIM5 spec) that downstream tooling
(the eval harness, the committed example run) can read back and render.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

AUDIT_FILENAME = "audit.json"
AUDIT_SCHEMA = "audit-1.0"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _empty_ledger() -> Dict[str, Any]:
    return {"schema": AUDIT_SCHEMA, "created": _now_iso(), "events": []}


def _ledger_path(base_directory: str, ledger_name: str = AUDIT_FILENAME) -> str:
    return os.path.join(base_directory, ledger_name)


def read_audit(base_directory: str, ledger_name: str = AUDIT_FILENAME) -> Dict[str, Any]:
    """Return the ledger dict, or a fresh empty ledger if none exists yet.

    Raises ValueError if an existing ledger file is not valid JSON (so a corrupt
    file surfaces as an error rather than being silently overwritten).
    """
    path = _ledger_path(base_directory, ledger_name)
    if not os.path.exists(path):
        return _empty_ledger()
    with open(path, "r", encoding="utf-8") as fh:
        raw = fh.read()
    if not raw.strip():
        return _empty_ledger()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"existing {ledger_name} is not valid JSON: {e}") from e
    if not isinstance(data, dict) or not isinstance(data.get("events"), list):
        raise ValueError(f"existing {ledger_name} is not a valid audit ledger")
    return data


def append_event(
    base_directory: str,
    stage: str,
    status: str = "info",
    summary: Optional[str] = None,
    data: Optional[Dict[str, Any]] = None,
    ledger_name: str = AUDIT_FILENAME,
    ts: Optional[str] = None,
) -> Dict[str, Any]:
    """Append one event to the ledger and persist it, returning the event.

    Args:
        base_directory: Sandbox root directory.
        stage: Pipeline stage name (e.g. "search", "extract", "generate_fr",
            "compile_ufo", "validate", "width_gate").
        status: Outcome marker ("ok" | "fail" | "info" | "error" | "skip").
        summary: One-line human-readable summary.
        data: Arbitrary JSON-serializable provenance (ids, hashes, checks).
        ledger_name: Ledger filename (default audit.json), to namespace per case.
        ts: Optional ISO timestamp override (defaults to now, UTC).
    """
    if not stage or not isinstance(stage, str):
        raise ValueError("stage must be a non-empty string")
    if not status or not isinstance(status, str):
        raise ValueError("status must be a non-empty string")
    if data is not None:
        # Fail loudly here rather than at write time on a non-serializable payload.
        json.dumps(data)

    ledger = read_audit(base_directory, ledger_name)
    seq = len(ledger["events"]) + 1
    event: Dict[str, Any] = {
        "seq": seq,
        "ts": ts or _now_iso(),
        "stage": stage,
        "status": status,
    }
    if summary is not None:
        event["summary"] = summary
    if data is not None:
        event["data"] = data
    ledger["events"].append(event)

    path = _ledger_path(base_directory, ledger_name)
    tmp = f"{path}.part"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(ledger, fh, indent=2)
    os.replace(tmp, path)
    return event


_STATUS_MARK = {"ok": "✓", "pass": "✓", "fail": "✗", "error": "✗", "skip": "⊘", "info": "·"}


def render_audit_md(ledger: Dict[str, Any]) -> str:
    """Render a ledger dict as a readable markdown timeline + per-event details."""
    events: List[Dict[str, Any]] = ledger.get("events", [])
    lines = [
        "# Pipeline Audit Trail",
        "",
        f"_schema `{ledger.get('schema', AUDIT_SCHEMA)}` — "
        f"{len(events)} event(s) — created {ledger.get('created', '?')}._",
        "",
        "| # | time (UTC) | stage | status | summary |",
        "|---|---|---|---|---|",
    ]
    for e in events:
        mark = _STATUS_MARK.get(str(e.get("status", "")).lower(), "")
        status = f"{mark} {e.get('status', '')}".strip()
        summary = (e.get("summary") or "").replace("|", "\\|")
        lines.append(
            f"| {e.get('seq', '')} | {e.get('ts', '')} | "
            f"{e.get('stage', '')} | {status} | {summary} |"
        )

    detailed = [e for e in events if e.get("data")]
    if detailed:
        lines += ["", "## Details", ""]
        for e in detailed:
            lines.append(f"### {e.get('seq')}. {e.get('stage')}")
            lines.append("```json")
            lines.append(json.dumps(e["data"], indent=2))
            lines.append("```")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"
