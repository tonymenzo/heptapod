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
    status.json  queued -> running -> done | failed | cancelled   [atomic]
                 (queued with a pid: the runner is waiting for a job slot)
    log.txt      runner stdout/stderr (tool prints, tracebacks)
    result.json  the tool's verbatim output string, once finished

What the WHITELIST does and does not do: only the tools named in it can be
submitted, and both submit() and the runner resolve the module/class from that
constant rather than from spec.json, so a spec on disk cannot name arbitrary
code to import. ``tool_args`` is restricted to the target tool's RuntimeFields,
checked at submit and again when the runner reads the spec, so executable
paths cannot be supplied as job arguments.

It is not a privilege boundary against the agent. spec.json holds the state a
detached runner needs -- base_directory and the configured paths -- and lives
under base_directory, so anything that can write there can change what a queued
job runs. That is by design rather than an oversight: in every deployment the
agent already has code execution, from the harness over MCP or from
RunCommandTool in the orchestral demos, so the whitelist is a guard against a
tool being invoked unintentionally, not a sandbox. Treat base_directory's
filesystem permissions as the boundary.
"""

from __future__ import annotations

import importlib
import json
import os
import re
import secrets
import signal
import subprocess
import sys
import threading
import time
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
    "validatelagrangian": ("tools.feynrules.feynrules", "ValidateLagrangianTool"),
    "validateufomodel": ("tools.feynrules.validate_tool", "ValidateUFOModelTool"),
    "madgraphfromruncard": ("tools.mg5.mg5", "MadGraphFromRunCardTool"),
    "reverselagrangian": ("tools.reverse.reverse_tool", "ReverseLagrangianTool"),
    "pythiafromruncard": ("tools.pythia", "PythiaFromRunCardTool"),
    "delphesfromhepmc": ("tools.delphes.delphes", "DelphesFromHepMCTool"),
    "enumerateprocesses": ("tools.feyngraph.catalogue_tools", "EnumerateProcessesTool"),
    "validateprocesscatalogue": ("tools.feyngraph.catalogue_tools", "ValidateProcessCatalogueTool"),
    # Fast and dependency-free — used by the offline test suite to exercise
    # the full submit -> run -> result path without Wolfram/MG5.
    "audittrail": ("tools.logging.audit_tool", "AuditTrailTool"),
}

JOB_ID_RE = re.compile(r"^job_\d{8}_\d{6}_[0-9a-f]{6}$")

TERMINAL_STATES = ("done", "failed", "cancelled")

# A runner marks its job running within a second or two of being started. A
# job still queued with no pid after this long never got a runner.
QUEUED_GRACE_SECONDS = 60

_REPO_ROOT = Path(__file__).resolve().parents[2]

# Jobs that may run at once on this machine, across every server and
# base_directory. Each runner holds an flock on one slot file while its tool
# runs; the kernel drops the lock when the runner dies, however it dies, so a
# crashed job never leaks a slot. Without a cap, a burst of submissions starts
# every job at once and can exhaust the machine's memory.
DEFAULT_MAX_JOBS = 4
SLOT_POLL_SECONDS = 2.0


def max_jobs() -> int:
    try:
        return max(1, int(os.environ.get("HEPTAPOD_MAX_JOBS", DEFAULT_MAX_JOBS)))
    except ValueError:
        return DEFAULT_MAX_JOBS


def slots_dir() -> str:
    return os.environ.get("HEPTAPOD_JOB_SLOTS_DIR") or os.path.join(
        os.path.expanduser("~"), ".cache", "heptapod", "job-slots")


def try_acquire_slot() -> Optional[Any]:
    """An open, flocked slot file, or None when all slots are taken.

    Keep the returned file object alive for as long as the job runs.
    """
    import fcntl

    d = slots_dir()
    os.makedirs(d, exist_ok=True)
    for i in range(max_jobs()):
        fh = open(os.path.join(d, f"slot-{i}.lock"), "a")
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return fh
        except OSError:
            fh.close()
    return None


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
            f"tool '{tool_name}' is not submittable; call it directly as an "
            f"ordinary tool. Only these long-running tools take jobs: {sorted(WHITELIST)}"
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
        proc = subprocess.Popen(
            [sys.executable, "-m", "tools.jobs.runner", jdir],
            cwd=str(_REPO_ROOT),
            start_new_session=True,
            stdin=subprocess.DEVNULL,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
    except OSError as e:
        set_status(jdir, state="failed", finished=_utcnow_iso(),
                   error=f"runner failed to start: {type(e).__name__}: {e}")
        raise
    finally:
        log_fh.close()

    # Reap the runner when it exits. Left unreaped it stays a zombie until the
    # next Popen, and kill(pid, 0) succeeds on a zombie, so a runner killed
    # outright (OOM, SIGKILL) would read as alive and its job as running.
    # Not SIGCHLD=SIG_IGN: that is process-wide and would break the exit
    # codes of every other tool's subprocess.run in this server.
    threading.Thread(target=proc.wait, name=f"reap-{job_id}", daemon=True).start()

    return {"job_id": job_id, "job_dir": os.path.join(JOBS_DIRNAME, job_id)}


def set_status(jdir: str, **updates) -> None:
    """Read-modify-write status.json. Callers serialise their own writers."""
    path = os.path.join(jdir, "status.json")
    st = read_json(path) or {"schema": STATUS_SCHEMA}
    st.update(updates)
    write_json_atomic(path, st)


def _age_seconds(iso: Optional[str]) -> Optional[float]:
    try:
        return time.time() - datetime.fromisoformat(iso).timestamp()
    except (TypeError, ValueError):
        return None


def effective_status(jdir: str) -> Dict[str, Any]:
    """status.json, corrected for runners that died without saying so."""
    st = read_json(os.path.join(jdir, "status.json")) or {}
    state = st.get("state")
    if state == "running" and not pid_alive(st.get("pid")):
        st["state"] = "failed"
        st["error"] = "runner process died before finishing (stale job)"
    elif state == "queued" and st.get("pid") and not pid_alive(st.get("pid")):
        st["state"] = "failed"
        st["error"] = "runner died while waiting for a job slot (stale job)"
    elif state == "queued" and not st.get("pid"):
        age = _age_seconds(st.get("submitted"))
        if age is not None and age > QUEUED_GRACE_SECONDS:
            st["state"] = "failed"
            st["error"] = (f"runner never started ({int(age)}s after submit); "
                           "see log_tail")
    return st


def _fmt_seconds(sec: float) -> str:
    sec = int(round(sec))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m" if h else (f"{m}m{s:02d}s" if m else f"{s}s")


def render_progress(progress: Dict[str, Any], started: Optional[str],
                    width: int = 24) -> Dict[str, Any]:
    """Add fraction, ETA and a text bar to a status.json progress record.

    The ETA assumes a steady rate since the job started, so it is rough for
    runs whose stages differ in speed (MG5's survey vs refine).
    """
    out = dict(progress)
    done, total = progress.get("done"), progress.get("total")
    unit = progress.get("unit") or ""
    msg = progress.get("message") or ""
    if done is not None and total:
        frac = max(0.0, min(1.0, float(done) / float(total)))
        out["fraction"] = round(frac, 4)
        elapsed = _age_seconds(started)
        eta = None
        if elapsed and 0 < frac < 1:
            eta = elapsed * (1 - frac) / frac
            out["eta_seconds"] = int(eta)
        filled = int(round(frac * width))
        bar = f"[{'#' * filled}{'-' * (width - filled)}] {frac:4.0%}"
        bar += f"  {done:g}/{total:g} {unit}".rstrip()
        if eta is not None:
            bar += f"  ETA {_fmt_seconds(eta)}"
        out["bar"] = f"{bar}  {msg}".rstrip() if msg else bar
    elif msg or done is not None:
        count = f"{done:g} {unit}".strip() if done is not None else ""
        out["bar"] = " · ".join(p for p in (msg, count) if p)
    return out


def _signal_group(pid: int, sig: int) -> bool:
    """Signal the runner's process group; False if it is already gone.

    The runner is started in its own session, so its group id is its pid. The
    check guards against a reused pid that now leads some unrelated group.
    """
    try:
        if os.getpgid(pid) != pid:
            return False
        os.killpg(pid, sig)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def cancel(base_directory: str, job_id: str, grace_seconds: float = 5.0) -> Dict[str, Any]:
    """Stop a queued or running job and everything its tool started.

    SIGTERM goes to the runner's whole process group (MG5 and its madevent
    workers, a wolframscript kernel), and the runner records the job as
    cancelled. A runner that does not finish within ``grace_seconds`` is
    SIGKILLed along with its group and marked cancelled here instead.

    Children that start their own session (the blank-slate agent CLI) are out
    of the group's reach; the tool that started them has to stop them itself.
    """
    jdir = job_dir(base_directory, job_id)
    if jdir is None or not os.path.isdir(jdir):
        raise FileNotFoundError(job_id)
    st = effective_status(jdir)
    state, pid = st.get("state"), st.get("pid")
    if state in TERMINAL_STATES:
        return {"job_id": job_id, "state": state, "already_finished": True}

    if state == "queued" and not pid:
        # The runner checks for this before it starts the tool.
        set_status(jdir, state="cancelled", finished=_utcnow_iso(),
                   error="cancelled before the runner started")
        return {"job_id": job_id, "state": "cancelled", "already_finished": False}

    _signal_group(pid, signal.SIGTERM)
    deadline = time.time() + grace_seconds
    while time.time() < deadline:
        if (read_json(os.path.join(jdir, "status.json")) or {}).get("state") in TERMINAL_STATES:
            break
        time.sleep(0.2)
    else:
        _signal_group(pid, signal.SIGKILL)
        set_status(jdir, state="cancelled", exit_code=-signal.SIGKILL,
                   finished=_utcnow_iso(),
                   error=f"cancelled; runner killed after a {grace_seconds:g}s grace period")
    final = (read_json(os.path.join(jdir, "status.json")) or {}).get("state")
    return {"job_id": job_id, "state": final, "already_finished": False}


def list_jobs(base_directory: str) -> list:
    """Newest-first summaries of every job under base_directory."""
    root = os.path.join(os.path.realpath(base_directory), JOBS_DIRNAME)
    out = []
    if not os.path.isdir(root):
        return out
    for name in sorted(os.listdir(root), reverse=True):
        if not JOB_ID_RE.match(name):
            continue
        st = effective_status(os.path.join(root, name))
        row = {
            "job_id": name,
            "tool": st.get("tool"),
            "state": st.get("state", "unknown"),
            "submitted": st.get("submitted"),
        }
        if st.get("state") == "running" and st.get("progress"):
            bar = render_progress(st["progress"], st.get("started")).get("bar")
            if bar:
                row["progress"] = bar
        out.append(row)
    return out
