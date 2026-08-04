"""
# job_manager.py is a part of the HEPTAPOD package.
# Copyright (C) 2026 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

Filesystem-backed background jobs for long-running heptapod tools.

The MCP server handles one tool call at a time, so a 30-minute MadGraph run
would block the whole channel (and most clients time out long before). This
module lets an agent SUBMIT such a run as a detached subprocess and get a
job_id back in milliseconds, then poll status / fetch the result and keep
working in between — the same workflow as a coding harness waiting on a
background task.

Layout (all under ``<base_directory>/.jobs/<job_id>/``):
    spec.json    what to run (tool, args, state config)     [written once]
    status.json  queued -> running -> done | failed          [atomic updates]
    log.txt      runner stdout/stderr (tool prints, tracebacks)
    result.json  the tool's verbatim output string, once finished

Security: only tools in the in-code WHITELIST can be run, and the runner
re-resolves the module/class from that constant — a tampered spec.json under
the user-writable base_directory cannot import arbitrary code. ``tool_args``
may only contain the target tool's RuntimeFields, so state (paths to
executables) always comes from the server-side config, never the caller.
"""

from __future__ import annotations

import importlib
import json
import os
import re
import secrets
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

SPEC_SCHEMA = "job-spec-1.0"
STATUS_SCHEMA = "job-status-1.0"
RESULT_SCHEMA = "job-result-1.0"

JOBS_DIRNAME = ".jobs"

# tool_name -> (module, class). A security boundary: keep it in code, never
# derive it from a config file or the spec.json on disk.
WHITELIST: Dict[str, Tuple[str, str]] = {
    "feynrulestoufo": ("tools.feynrules.feynrules", "FeynRulesToUFOTool"),
    "validatemodel": ("tools.validate.validate_tool", "ValidateModelTool"),
    "madgraphfromruncard": ("tools.mg5.mg5", "MadGraphFromRunCardTool"),
    "reverselagrangian": ("tools.reverse.reverse_tool", "ReverseLagrangianTool"),
    # Fast and dependency-free — used by the offline test suite to exercise
    # the full submit -> run -> result path without Wolfram/MG5/codex.
    "audittrail": ("tools.logging.audit_tool", "AuditTrailTool"),
}

JOB_ID_RE = re.compile(r"^job_\d{8}_\d{6}_[0-9a-f]{6}$")

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def new_job_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"job_{ts}_{secrets.token_hex(3)}"


def job_dir(base_directory: str, job_id: str) -> Optional[str]:
    """Resolve a job dir, refusing malformed ids (also blocks traversal)."""
    if not JOB_ID_RE.match(job_id or ""):
        return None
    return os.path.join(os.path.realpath(base_directory), JOBS_DIRNAME, job_id)


def write_json_atomic(path: str, payload: Dict[str, Any]) -> None:
    part = path + ".part"
    with open(part, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    os.replace(part, path)


def read_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None


def resolve_tool(tool_name: str):
    """Import and return the tool class for a whitelisted name."""
    module, cls_name = WHITELIST[tool_name]
    return getattr(importlib.import_module(module), cls_name)


def validate_args(cls, tool_args: Dict[str, Any]) -> Optional[str]:
    """Return an error string if args don't fit the tool's runtime fields."""
    runtime = set(cls._get_runtime_fields())
    unknown = sorted(set(tool_args) - runtime)
    if unknown:
        return (
            f"unknown args {unknown}; allowed runtime args: {sorted(runtime)} "
            "(state fields like base_directory come from server config)"
        )
    # No missing-required check here: orchestral's RuntimeField gives every
    # runtime field a default (None when unspecified), so "required" cannot be
    # told apart from "optional with None default" by introspection. A truly
    # missing arg surfaces as the tool's own structured error in result.json
    # (tool_error=true), which is still actionable feedback for the agent.
    return None


def pid_alive(pid: Optional[int]) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def submit(
    base_directory: str,
    tool_name: str,
    tool_args: Dict[str, Any],
    state: Dict[str, Any],
    ledger_name: Optional[str] = None,
    max_seconds: Optional[int] = None,
) -> Dict[str, Any]:
    """Validate, persist the spec, and detach a runner. Returns submit info.

    Raises ValueError on validation problems (caller formats the error).
    """
    if tool_name not in WHITELIST:
        raise ValueError(
            f"tool '{tool_name}' is not submittable; whitelist: {sorted(WHITELIST)}"
        )
    cls = resolve_tool(tool_name)
    err = validate_args(cls, tool_args)
    if err:
        raise ValueError(err)

    job_id = new_job_id()
    jdir = job_dir(base_directory, job_id)
    os.makedirs(jdir, exist_ok=False)

    spec = {
        "schema": SPEC_SCHEMA,
        "job_id": job_id,
        "tool": tool_name,
        "args": tool_args,
        "state": state,
        "ledger_name": ledger_name,
        "max_seconds": max_seconds,
        "submitted": _utcnow_iso(),
    }
    write_json_atomic(os.path.join(jdir, "spec.json"), spec)
    write_json_atomic(
        os.path.join(jdir, "status.json"),
        {
            "schema": STATUS_SCHEMA,
            "job_id": job_id,
            "tool": tool_name,
            "state": "queued",
            "pid": None,
            "exit_code": None,
            "submitted": spec["submitted"],
            "started": None,
            "finished": None,
            "error": None,
        },
    )

    log_fh = open(os.path.join(jdir, "log.txt"), "ab")
    try:
        subprocess.Popen(
            [sys.executable, "-m", "tools.jobs.runner", jdir],
            cwd=str(_REPO_ROOT),
            start_new_session=True,
            stdin=subprocess.DEVNULL,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
    finally:
        log_fh.close()

    return {"job_id": job_id, "job_dir": os.path.join(JOBS_DIRNAME, job_id)}


def list_jobs(base_directory: str) -> list:
    """Newest-first summaries of every job under base_directory."""
    root = os.path.join(os.path.realpath(base_directory), JOBS_DIRNAME)
    out = []
    if not os.path.isdir(root):
        return out
    for name in sorted(os.listdir(root), reverse=True):
        if not JOB_ID_RE.match(name):
            continue
        st = read_json(os.path.join(root, name, "status.json")) or {}
        out.append(
            {
                "job_id": name,
                "tool": st.get("tool"),
                "state": st.get("state", "unknown"),
                "submitted": st.get("submitted"),
            }
        )
    return out
