#!/usr/bin/env python3
"""
# test_audit.py is a part of the HEPTAPOD package.
# Copyright (C) 2026 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

Tests for the audit provenance ledger + AuditTrailTool.

Run with:
    python test_audit.py
"""

import json
import os
import sys
import tempfile
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tools.logging.audit import (
    AUDIT_FILENAME,
    AUDIT_SCHEMA,
    append_event,
    read_audit,
    render_audit_md,
)
from tools.logging.audit_tool import AuditTrailTool


def test_append_creates_and_sequences() -> bool:
    print(">> Testing append_event creates ledger and sequences events...")
    tmp = tempfile.mkdtemp()
    e1 = append_event(tmp, "search", "ok", summary="found 2103.02708",
                      data={"arxiv_id": "2103.02708"})
    e2 = append_event(tmp, "extract", "ok", summary="1 particle")
    assert e1["seq"] == 1 and e2["seq"] == 2, (e1, e2)
    path = os.path.join(tmp, AUDIT_FILENAME)
    assert os.path.exists(path), "audit.json not written"
    ledger = json.loads(open(path).read())
    assert ledger["schema"] == AUDIT_SCHEMA, ledger
    assert len(ledger["events"]) == 2, ledger
    assert ledger["events"][0]["data"]["arxiv_id"] == "2103.02708", ledger
    assert "ts" in ledger["events"][0], ledger
    print("[✓] append/sequence test passed\n")
    return True


def test_read_empty_and_roundtrip() -> bool:
    print(">> Testing read_audit on empty + roundtrip...")
    tmp = tempfile.mkdtemp()
    empty = read_audit(tmp)
    assert empty["events"] == [] and empty["schema"] == AUDIT_SCHEMA, empty
    append_event(tmp, "validate", "fail", summary="width gate off")
    again = read_audit(tmp)
    assert len(again["events"]) == 1 and again["events"][0]["status"] == "fail", again
    print("[✓] read/roundtrip test passed\n")
    return True


def test_corrupt_ledger_raises() -> bool:
    print(">> Testing corrupt ledger surfaces as error (no silent overwrite)...")
    tmp = tempfile.mkdtemp()
    with open(os.path.join(tmp, AUDIT_FILENAME), "w") as fh:
        fh.write("{not json")
    try:
        read_audit(tmp)
        raise AssertionError("should have raised on corrupt ledger")
    except ValueError:
        pass
    print("[✓] corrupt-ledger test passed\n")
    return True


def test_render_markdown() -> bool:
    print(">> Testing render_audit_md...")
    tmp = tempfile.mkdtemp()
    append_event(tmp, "search", "ok", summary="found paper", data={"id": "x"})
    append_event(tmp, "validate", "fail", summary="a|b pipe")
    md = render_audit_md(read_audit(tmp))
    assert "# Pipeline Audit Trail" in md, md
    assert "| search |" in md and "validate" in md, md
    assert "a\\|b pipe" in md, "pipe not escaped"
    assert "## Details" in md and '"id": "x"' in md, md
    print("[✓] render test passed\n")
    return True


def test_tool_record_read_render() -> bool:
    print(">> Testing AuditTrailTool record/read/render actions...")
    tmp = tempfile.mkdtemp()
    rec = json.loads(
        AuditTrailTool(
            action="record", stage="extract", status="ok",
            summary="S1 model", data=json.dumps({"n_particles": 1}),
            base_directory=tmp,
        )._run()
    )
    assert rec["status"] == "ok" and rec["event"]["seq"] == 1, rec
    assert rec["n_events"] == 1, rec

    read = json.loads(AuditTrailTool(action="read", base_directory=tmp)._run())
    assert read["ledger"]["events"][0]["data"]["n_particles"] == 1, read

    rendered = json.loads(AuditTrailTool(action="render", base_directory=tmp)._run())
    assert "Pipeline Audit Trail" in rendered["markdown"], rendered
    assert rendered["audit_md_path"] == "audit.md", rendered
    assert os.path.exists(os.path.join(tmp, "audit.md")), "audit.md not written"
    print("[✓] tool action test passed\n")
    return True


def test_tool_guards() -> bool:
    print(">> Testing AuditTrailTool guards (bad action, traversal, missing stage)...")
    tmp = tempfile.mkdtemp()
    bad_action = AuditTrailTool(action="frobnicate", base_directory=tmp)._run()
    assert "action must be one of" in bad_action, bad_action

    no_stage = AuditTrailTool(action="record", base_directory=tmp)._run()
    assert "stage is required" in no_stage, no_stage

    escape = AuditTrailTool(
        action="record", stage="x", ledger_name="../evil.json", base_directory=tmp
    )._run()
    assert "escapes base_directory" in escape, escape

    bad_data = AuditTrailTool(
        action="record", stage="x", data="{not json", base_directory=tmp
    )._run()
    assert "not valid JSON" in bad_data, bad_data
    print("[✓] tool guards test passed\n")
    return True


TESTS = [
    test_append_creates_and_sequences,
    test_read_empty_and_roundtrip,
    test_corrupt_ledger_raises,
    test_render_markdown,
    test_tool_record_read_render,
    test_tool_guards,
]


if __name__ == "__main__":
    all_passed = True
    for test in TESTS:
        try:
            if not test():
                all_passed = False
        except Exception as e:  # noqa: BLE001
            print(f"[✗] {test.__name__} failed: {e}\n")
            all_passed = False
    if all_passed:
        print("[✓] All tests passed!\n")
        sys.exit(0)
    print("[✗] Some tests failed!\n")
    sys.exit(1)
