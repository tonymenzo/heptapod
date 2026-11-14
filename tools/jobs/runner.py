"""
# runner.py is a part of the HEPTAPOD package.
# Copyright (C) 2026 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

Detached job runner: ``python -m tools.jobs.runner <job_dir>``.

Reads spec.json, runs the whitelisted tool, writes result.json and drives
status.json through running -> done | failed | cancelled. Never trusts
module/class strings from the (user-writable) spec — the tool class is
re-resolved from the in-code WHITELIST. stdout/stderr already point at log.txt
(the submitter wired them), so plain prints and tracebacks land there.

Stopping a job. The submitter starts the runner in its own session, so the
runner leads a process group holding everything the tool starts: MG5 and its
madevent workers, a wolframscript kernel, Delphes. Both ways a job is stopped
send SIGTERM to that whole group:

  * the ``max_seconds`` watchdog, from a thread in this process;
  * ``canceljob``, from the MCP server.

The runner's own SIGTERM handler raises ``JobTerminated`` in the main thread.
It derives from BaseException so the tools' ``except Exception`` blocks let it
through, and subprocess.run / communicate kill their child on the way out.
The runner then records the outcome and SIGKILLs whatever is left of the group.

Job slots. Before the tool starts, the runner takes one of HEPTAPOD_MAX_JOBS
machine-wide slots (default 4; see job_manager.try_acquire_slot) and waits,
still queued, until one is free. The slot is an flock, released when the
runner exits or dies.

Progress. The runner installs itself as the tool's ``_stream_callback``, so a
tool's ``report_progress(...)`` calls (tools/jobs/progress.py) land in
status.json, throttled, where jobstatus renders them as a progress bar.
"""

from __future__ import annotations

import os
import signal
import sys
import threading
import time
import traceback
from datetime import datetime, timezone

from tools.jobs.job_manager import (
    RESULT_SCHEMA,
    SLOT_POLL_SECONDS,
    WHITELIST,
    max_jobs,
    read_json,
    resolve_tool,
    set_status,
    try_acquire_slot,
    write_json_atomic,
)

# Seconds the watchdog waits after SIGTERM before it SIGKILLs the group and
# records the timeout itself (a tool stuck in C code never sees the signal).
_KILL_GRACE_SECONDS = 10.0

# Minimum seconds between progress writes to status.json.
_PROGRESS_INTERVAL = 1.0


class JobTerminated(BaseException):
    """Raised in the main thread when the job is told to stop."""


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class _Finish:
    """Decides, once, how the job ended.

    The main thread (tool returned or raised) and the watchdog thread can both
    try to record an outcome; whichever claims it first wins and the other
    stands down. The same lock serialises this process's status.json writes,
    progress updates included.
    """

    def __init__(self, jdir: str):
        self.jdir = jdir
        self.lock = threading.RLock()
        self.claimed = False
        self.timeout_reason = None  # set by the watchdog before it signals

    def status(self, **updates) -> None:
        with self.lock:
            set_status(self.jdir, **updates)

    def claim(self) -> bool:
        with self.lock:
            if self.claimed:
                return False
            self.claimed = True
            return True


class _Progress:
    """The ``_stream_callback`` the runner installs on the tool."""

    def __init__(self, finish: _Finish):
        self.finish = finish
        self.last_write = 0.0
        self.pending = None

    def progress(self, done=None, total=None, unit="", message="") -> None:
        record = {"done": done, "total": total, "unit": unit,
                  "message": message, "updated": _now()}
        now = time.monotonic()
        # Throttled, but a completed count is always written.
        if now - self.last_write < _PROGRESS_INTERVAL and not (total and done == total):
            self.pending = record
            return
        self._write(record, now)

    def __call__(self, text: str) -> None:
        self.progress(message=str(text))

    def flush(self) -> None:
        if self.pending:
            self._write(self.pending, time.monotonic())

    def _write(self, record, now) -> None:
        self.pending, self.last_write = None, now
        with self.finish.lock:
            if not self.finish.claimed:
                set_status(self.finish.jdir, progress=record)


def _leads_group() -> bool:
    return os.getpgid(0) == os.getpid()


def _stop_group(sig: int) -> None:
    """Signal the runner's group; outside its own group, just the runner."""
    try:
        if _leads_group():
            os.killpg(0, sig)
        else:
            os.kill(os.getpid(), sig)
    except ProcessLookupError:
        pass


def _audit(spec: dict, status: str, summary: str, data: dict) -> None:
    """Best-effort audit event; never fails the job."""
    try:
        from tools.logging.audit import append_event

        append_event(
            spec["state"]["base_directory"],
            stage="job",
            status=status,
            summary=summary,
            data=data,
            ledger_name=spec.get("ledger_name") or "audit.json",
        )
    except Exception:  # noqa: BLE001
        pass


def _start_watchdog(finish: _Finish, max_seconds: float) -> None:
    def _expire() -> None:
        with finish.lock:
            if finish.claimed:
                return
            finish.timeout_reason = f"watchdog timeout after {max_seconds:g}s"
        _stop_group(signal.SIGTERM)
        time.sleep(_KILL_GRACE_SECONDS)
        # Still alive: the main thread never acted on the signal.
        if finish.claim():
            finish.status(state="failed", exit_code=124, finished=_now(),
                          error=finish.timeout_reason + " (killed)")
        _stop_group(signal.SIGKILL)

    t = threading.Timer(max_seconds, _expire)
    t.daemon = True
    t.start()


def _cancelled(jdir: str) -> bool:
    return (read_json(os.path.join(jdir, "status.json")) or {}).get("state") == "cancelled"


def main(jdir: str) -> int:
    spec = read_json(os.path.join(jdir, "spec.json"))
    if not spec or spec.get("tool") not in WHITELIST:
        set_status(jdir, state="failed", finished=_now(),
                   error="invalid spec or non-whitelisted tool")
        return 2

    # A job cancelled while it was still queued never starts.
    if _cancelled(jdir):
        return 0

    job_id, tool_name = spec["job_id"], spec["tool"]
    finish = _Finish(jdir)

    def _on_sigterm(signum, frame):
        # Once an outcome is being written, let it finish.
        if not finish.claimed:
            raise JobTerminated()

    signal.signal(signal.SIGTERM, _on_sigterm)
    progress = _Progress(finish)
    slot = None
    try:
        # Wait for a machine-wide job slot. The pid goes in first, so the job
        # reads as alive and canceljob can signal it while it waits.
        slot = try_acquire_slot()
        if slot is None:
            finish.status(pid=os.getpid(), progress={
                "message": f"waiting for a job slot ({max_jobs()} run at once)",
                "updated": _now()})
            while slot is None:
                if _cancelled(jdir):
                    return 0
                time.sleep(SLOT_POLL_SECONDS)
                slot = try_acquire_slot()
        if _cancelled(jdir):
            return 0
        finish.status(state="running", pid=os.getpid(), started=_now(), progress=None)

        cls = resolve_tool(tool_name)
        state = spec.get("state") or {}
        kwargs = {k: v for k, v in state.items() if k in cls.model_fields}
        # Instantiate with state + runtime args merged and call _run()
        # directly (the pattern the repo's own harnesses use): orchestral's
        # execute() re-checks "required" fields with default=None-means-
        # required semantics and would reject omitted optional args.
        # spec.json lives under the user-writable base_directory, so the
        # restriction submit() applied has to be applied again here rather than
        # trusted. Without it a spec carrying a StateField in "args" -- for
        # FeynRulesToUFOTool that includes wolframscript_path, which becomes
        # argv[0] of a subprocess -- would override the server-side config and
        # step around the whitelist this runner exists to enforce.
        spec_args = spec.get("args") or {}
        runtime = set(cls._get_runtime_fields())
        rejected = sorted(set(spec_args) - runtime)
        if rejected:
            finish.claim()
            finish.status(state="failed", finished=_now(),
                          error=f"spec args are not runtime fields: {', '.join(rejected)}")
            return 2
        kwargs.update(spec_args)
        tool = cls(**kwargs)
        tool._stream_callback = progress

        if spec.get("max_seconds"):
            _start_watchdog(finish, float(spec["max_seconds"]))

        output = tool._run()
        progress.flush()
        if not finish.claim():
            return 1
        tool_error = isinstance(output, str) and output.startswith("Error:")
        write_json_atomic(
            os.path.join(jdir, "result.json"),
            {
                "schema": RESULT_SCHEMA,
                "job_id": job_id,
                "tool": tool_name,
                "tool_error": tool_error,
                "output": output,
            },
        )
        finish.status(state="done", exit_code=0, finished=_now())
        _audit(spec, "fail" if tool_error else "ok",
               f"{tool_name} job {job_id} finished",
               {"job_id": job_id, "tool": tool_name, "tool_error": tool_error})
        return 0
    except JobTerminated:
        if not finish.claim():
            return 1
        reason = finish.timeout_reason
        if reason:
            finish.status(state="failed", exit_code=124, finished=_now(), error=reason)
            _audit(spec, "error", f"{tool_name} job {job_id} timed out",
                   {"job_id": job_id, "tool": tool_name, "error": reason})
        else:
            finish.status(state="cancelled", exit_code=130, finished=_now(),
                          error="cancelled by request")
            _audit(spec, "error", f"{tool_name} job {job_id} cancelled",
                   {"job_id": job_id, "tool": tool_name})
        # Anything the tool started that ignored SIGTERM goes now, runner too.
        # Only as a detached runner: called in-process it would kill its caller.
        if _leads_group():
            _stop_group(signal.SIGKILL)
        return 124 if reason else 130
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        if not finish.claim():
            return 1
        finish.status(state="failed", exit_code=1, finished=_now(),
                      error=f"{type(e).__name__}: {e}")
        _audit(spec, "error", f"{tool_name} job {job_id} crashed",
               {"job_id": job_id, "tool": tool_name, "error": str(e)[:300]})
        return 1
    finally:
        if slot is not None:
            slot.close()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python -m tools.jobs.runner <job_dir>", file=sys.stderr)
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1]))
