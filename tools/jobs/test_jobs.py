#!/usr/bin/env python3
"""
# test_jobs.py is a part of the HEPTAPOD package.
# Copyright (C) 2026 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.

Tests for the background-job layer.

All tests are offline (no Wolfram / MG5 / network): validation failures are
exercised against the real ValidateUFOModelTool class (import only), and the full
submit -> detached runner -> result round-trip uses the dependency-free
AuditTrailTool. The detached subprocess is real.

Cancellation, the watchdog and progress need a tool that runs for a while and
starts a child process. ``_SleepTool`` below is one; it is whitelisted only
inside the detached test runner (``_HARNESS``), never in the real WHITELIST.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import tools.jobs.job_manager as jm  # noqa: E402
from tools.jobs.jobs_tool import (  # noqa: E402
    CancelJobTool,
    JobResultTool,
    JobStatusTool,
    SubmitJobTool,
)
from tools.jobs.progress import report_progress  # noqa: E402
from orchestral.tools.base.tool import BaseTool  # noqa: E402
from orchestral.tools.base.field_utils import RuntimeField, StateField  # noqa: E402

_TMPDIRS = []

# Tests take job slots from a private directory, never the machine's.
os.environ["HEPTAPOD_JOB_SLOTS_DIR"] = tempfile.mkdtemp(prefix="jobs_slots_")
_TMPDIRS.append(os.environ["HEPTAPOD_JOB_SLOTS_DIR"])


def _base() -> str:
    d = tempfile.mkdtemp(prefix="jobs_test_")
    _TMPDIRS.append(d)
    return d


def cleanup_test_files() -> None:
    for d in _TMPDIRS:
        shutil.rmtree(d, ignore_errors=True)


def _wait(base: str, jid: str, seconds: float = 15.0) -> dict:
    deadline = time.time() + seconds
    while time.time() < deadline:
        st = json.loads(JobStatusTool(base_directory=base, job_id=jid)._run())
        if st.get("state") in jm.TERMINAL_STATES:
            return st
        time.sleep(0.2)
    return st


class _SleepTool(BaseTool):
    """Test-only: runs ``sleep`` as a child, reporting progress as it waits."""

    seconds: int = RuntimeField(description="How long the child sleeps")
    pid_file: str = RuntimeField(description="Where to write the child's pid")
    base_directory: str = StateField(description="Base sandbox directory")

    def _run(self) -> str:
        proc = subprocess.Popen(["sleep", str(self.seconds)])
        Path(self.pid_file).write_text(str(proc.pid))
        steps = self.seconds * 5
        for i in range(steps):
            try:
                proc.wait(timeout=0.2)
                break
            except subprocess.TimeoutExpired:
                report_progress(self, done=i + 1, total=steps, unit="steps",
                                message="sleeping")
        proc.wait()
        return '{"status": "ok"}'


_HARNESS = (
    "import sys\n"
    "import tools.jobs.job_manager as jm\n"
    "jm.WHITELIST['sleeptest'] = ('tools.jobs.test_jobs', '_SleepTool')\n"
    "from tools.jobs import runner\n"
    "raise SystemExit(runner.main(sys.argv[1]))\n"
)


def _launch_sleep_job(base: str, seconds: int, max_seconds=None):
    """Start _SleepTool through the real runner, detached like submit() does."""
    jid = jm.new_job_id()
    jdir = jm.job_dir(base, jid)
    os.makedirs(jdir)
    pid_file = os.path.join(base, f"{jid}.child")
    jm.write_json_atomic(os.path.join(jdir, "spec.json"), {
        "schema": jm.SPEC_SCHEMA, "job_id": jid, "tool": "sleeptest",
        "args": {"seconds": seconds, "pid_file": pid_file},
        "state": {"base_directory": base}, "max_seconds": max_seconds,
        "submitted": datetime.now(timezone.utc).isoformat(),
    })
    jm.write_json_atomic(os.path.join(jdir, "status.json"), {
        "schema": jm.STATUS_SCHEMA, "job_id": jid, "tool": "sleeptest",
        "state": "queued", "pid": None, "exit_code": None,
        "submitted": datetime.now(timezone.utc).isoformat(),
        "started": None, "finished": None, "error": None,
    })
    with open(os.path.join(jdir, "log.txt"), "ab") as log:
        proc = subprocess.Popen(
            [sys.executable, "-c", _HARNESS, jdir], cwd=str(REPO_ROOT),
            start_new_session=True, stdout=log, stderr=subprocess.STDOUT)
    return jid, pid_file, proc


def _child_pid(pid_file: str, seconds: float = 10.0):
    deadline = time.time() + seconds
    while time.time() < deadline:
        try:
            return int(Path(pid_file).read_text())
        except (OSError, ValueError):
            time.sleep(0.1)
    return None


def _gone(pid, seconds: float = 5.0) -> bool:
    deadline = time.time() + seconds
    while time.time() < deadline:
        if not jm.pid_alive(pid):
            return True
        time.sleep(0.1)
    return False


def test_job_id_format() -> bool:
    ids = {jm.new_job_id() for _ in range(50)}
    ok = len(ids) == 50 and all(jm.JOB_ID_RE.match(i) for i in ids)
    print(f"[{'✓' if ok else '✗'}] test_job_id_format")
    return ok


def test_validate_args_against_real_tool() -> bool:
    from tools.feynrules.validate_tool import ValidateUFOModelTool

    ok = True
    # unknown key rejected (state-field smuggling)
    err = jm.validate_args(ValidateUFOModelTool, {"model_path": "x", "base_directory": "/x"})
    ok &= err is not None and "unknown args" in err
    err = jm.validate_args(ValidateUFOModelTool, {"model_path": "x", "wolframscript_path": "/x"})
    ok &= err is not None and "unknown args" in err
    # valid passes (missing-with-default args are the tool's concern at run time)
    err = jm.validate_args(ValidateUFOModelTool, {"model_path": "m.fr", "madgraph_check": True,
                                                  "lagrangian": "LSM + LBSM"})
    ok &= err is None
    print(f"[{'✓' if ok else '✗'}] test_validate_args_against_real_tool")
    return ok


def test_whitelist_names_match_tools() -> bool:
    """Each whitelist key is the name the agent sees for that tool.

    An agent submits the name it was shown, which orchestral derives from the
    class name. A key that drifts from it makes the tool unreachable as a job.
    """
    import importlib

    ok = True
    for key, (module, cls_name) in jm.WHITELIST.items():
        cls = getattr(importlib.import_module(module), cls_name)
        if cls.get_name() != key:
            print(f"    whitelist key {key!r} != tool name {cls.get_name()!r}")
            ok = False
    print(f"[{'✓' if ok else '✗'}] test_whitelist_names_match_tools")
    return ok


def test_submit_rejections() -> bool:
    base = _base()
    ok = True
    r = SubmitJobTool(base_directory=base, tool_name="nope", tool_args="{}")._run()
    ok &= r.startswith("Error:") and "not submittable" in r and "call it directly" in r
    r = SubmitJobTool(base_directory=base, tool_name="audittrail", tool_args="[1,2]")._run()
    ok &= r.startswith("Error:")
    r = SubmitJobTool(base_directory=base, tool_name="audittrail", tool_args="not json")._run()
    ok &= r.startswith("Error:")
    print(f"[{'✓' if ok else '✗'}] test_submit_rejections")
    return ok


def test_detached_round_trip() -> bool:
    base = _base()
    args = {"action": "record", "stage": "t", "status": "ok",
            "summary": "round trip", "data": "{}"}
    out = json.loads(SubmitJobTool(
        base_directory=base, tool_name="audittrail",
        tool_args=json.dumps(args))._run())
    st = _wait(base, out["job_id"])
    res_raw = JobResultTool(base_directory=base, job_id=out["job_id"])._run()
    ok = st.get("state") == "done" and not res_raw.startswith("Error:")
    if ok:
        res = json.loads(res_raw)
        ok &= res["tool_error"] is False and '"status": "ok"' in res["output"]
        # audit events (job_submit + job): the runner appends its event just
        # AFTER flipping status to done, so retry briefly.
        stages: list = []
        deadline = time.time() + 3.0
        while time.time() < deadline:
            ledger = jm.read_json(os.path.join(base, "audit.json")) or {"events": []}
            stages = [e["stage"] for e in ledger["events"]]
            if "job" in stages:
                break
            time.sleep(0.1)
        ok &= "job_submit" in stages and "job" in stages
    print(f"[{'✓' if ok else '✗'}] test_detached_round_trip")
    return ok


def test_result_before_done_and_bad_ids() -> bool:
    base = _base()
    ok = True
    r = JobResultTool(base_directory=base, job_id="job_00000000_000000_abcdef")._run()
    ok &= "Job Not Found" in r
    r = JobStatusTool(base_directory=base, job_id="../../etc")._run()
    ok &= "Job Not Found" in r
    r = JobResultTool(base_directory=base, job_id="../../etc")._run()
    ok &= "Job Not Found" in r
    print(f"[{'✓' if ok else '✗'}] test_result_before_done_and_bad_ids")
    return ok


def test_stale_running_detection() -> bool:
    base = _base()
    out = json.loads(SubmitJobTool(
        base_directory=base, tool_name="audittrail",
        tool_args=json.dumps({"action": "record", "stage": "t",
                              "summary": "s", "data": "{}"}))._run())
    jid = out["job_id"]
    _wait(base, jid)
    jd = jm.job_dir(base, jid)
    st = jm.read_json(os.path.join(jd, "status.json"))
    st.update(state="running", pid=99999999)
    jm.write_json_atomic(os.path.join(jd, "status.json"), st)
    got = json.loads(JobStatusTool(base_directory=base, job_id=jid)._run())
    ok = got["state"] == "failed" and "stale" in (got.get("error") or "")
    print(f"[{'✓' if ok else '✗'}] test_stale_running_detection")
    return ok


def test_stale_queued_detection() -> bool:
    """A job still queued with no runner long after submit reads as failed."""
    base = _base()
    jid = jm.new_job_id()
    jdir = jm.job_dir(base, jid)
    os.makedirs(jdir)
    old = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()
    jm.write_json_atomic(os.path.join(jdir, "status.json"), {
        "schema": jm.STATUS_SCHEMA, "job_id": jid, "tool": "audittrail",
        "state": "queued", "pid": None, "submitted": old,
    })
    got = json.loads(JobStatusTool(base_directory=base, job_id=jid)._run())
    ok = got["state"] == "failed" and "never started" in (got.get("error") or "")
    print(f"[{'✓' if ok else '✗'}] test_stale_queued_detection")
    return ok


def test_finished_runner_is_reaped() -> bool:
    """A finished runner must not linger as a zombie that reads as alive.

    kill(pid, 0) succeeds on a zombie, so without reaping, a runner killed
    outright would leave its job looking like it is still running.
    """
    base = _base()
    out = json.loads(SubmitJobTool(
        base_directory=base, tool_name="audittrail",
        tool_args=json.dumps({"action": "record", "stage": "t",
                              "summary": "s", "data": "{}"}))._run())
    st = _wait(base, out["job_id"])
    ok = st.get("state") == "done" and _gone(st.get("pid"))
    print(f"[{'✓' if ok else '✗'}] test_finished_runner_is_reaped")
    return ok


def test_progress_reported() -> bool:
    """report_progress() in a tool shows up in jobstatus as a progress bar."""
    base = _base()
    jid, _, proc = _launch_sleep_job(base, seconds=3)
    bar = None
    deadline = time.time() + 10
    while time.time() < deadline and not bar:
        st = json.loads(JobStatusTool(base_directory=base, job_id=jid)._run())
        bar = (st.get("progress") or {}).get("bar")
        time.sleep(0.2)
    st = _wait(base, jid)
    proc.wait()
    ok = bool(bar) and "%" in bar and "steps" in bar and st["state"] == "done"
    listed = jm.list_jobs(base)
    ok &= listed and listed[0]["state"] == "done"
    print(f"[{'✓' if ok else '✗'}] test_progress_reported  ({bar})")
    return ok


def test_cancel_stops_job_and_children() -> bool:
    base = _base()
    jid, pid_file, proc = _launch_sleep_job(base, seconds=60)
    child = _child_pid(pid_file)
    out = json.loads(CancelJobTool(base_directory=base, job_id=jid)._run())
    proc.wait(timeout=10)
    st = json.loads(JobStatusTool(base_directory=base, job_id=jid)._run())
    res = JobResultTool(base_directory=base, job_id=jid)._run()
    again = json.loads(CancelJobTool(base_directory=base, job_id=jid)._run())
    ok = (child is not None and out["state"] == "cancelled"
          and st["state"] == "cancelled" and _gone(child)
          and "Job Cancelled" in res and again["already_finished"])
    print(f"[{'✓' if ok else '✗'}] test_cancel_stops_job_and_children")
    return ok


def test_cancel_queued_job_never_runs() -> bool:
    base = _base()
    jid = jm.new_job_id()
    jdir = jm.job_dir(base, jid)
    os.makedirs(jdir)
    jm.write_json_atomic(os.path.join(jdir, "spec.json"), {
        "schema": jm.SPEC_SCHEMA, "job_id": jid, "tool": "audittrail",
        "args": {"action": "record", "stage": "t", "summary": "s", "data": "{}"},
        "state": {"base_directory": base},
    })
    jm.write_json_atomic(os.path.join(jdir, "status.json"), {
        "schema": jm.STATUS_SCHEMA, "job_id": jid, "tool": "audittrail",
        "state": "queued", "pid": None,
        "submitted": datetime.now(timezone.utc).isoformat(),
    })
    out = json.loads(CancelJobTool(base_directory=base, job_id=jid)._run())
    from tools.jobs.runner import main as runner_main
    rc = runner_main(jdir)
    st = jm.read_json(os.path.join(jdir, "status.json"))
    ok = (out["state"] == "cancelled" and rc == 0 and st["state"] == "cancelled"
          and not os.path.exists(os.path.join(jdir, "result.json")))
    print(f"[{'✓' if ok else '✗'}] test_cancel_queued_job_never_runs")
    return ok


def test_watchdog_stops_job_and_children() -> bool:
    """max_seconds stops the tool's children too, not just the runner."""
    base = _base()
    jid, pid_file, proc = _launch_sleep_job(base, seconds=60, max_seconds=1)
    child = _child_pid(pid_file)
    proc.wait(timeout=20)
    st = json.loads(JobStatusTool(base_directory=base, job_id=jid)._run())
    ok = (child is not None and st["state"] == "failed"
          and "watchdog" in (st.get("error") or "") and _gone(child))
    print(f"[{'✓' if ok else '✗'}] test_watchdog_stops_job_and_children")
    return ok


def test_render_progress() -> bool:
    started = (datetime.now(timezone.utc) - timedelta(seconds=30)).isoformat()
    r = jm.render_progress({"done": 250, "total": 1000, "unit": "events",
                            "message": "generating"}, started)
    ok = (r["fraction"] == 0.25 and 80 <= r["eta_seconds"] <= 100
          and r["bar"].startswith("[######------------------]  25%")
          and "250/1000 events" in r["bar"] and r["bar"].endswith("generating"))
    r = jm.render_progress({"done": None, "total": None, "unit": "",
                            "message": "survey"}, started)
    ok &= r["bar"] == "survey" and "fraction" not in r
    print(f"[{'✓' if ok else '✗'}] test_render_progress")
    return ok


def test_listing() -> bool:
    base = _base()
    out = json.loads(SubmitJobTool(
        base_directory=base, tool_name="audittrail",
        tool_args=json.dumps({"action": "record", "stage": "t",
                              "summary": "s", "data": "{}"}))._run())
    _wait(base, out["job_id"])
    lst = json.loads(JobStatusTool(base_directory=base)._run())
    ok = lst["jobs"] and lst["jobs"][0]["job_id"] == out["job_id"]
    print(f"[{'✓' if ok else '✗'}] test_listing")
    return ok


def test_runner_rejects_state_fields_in_args() -> bool:
    """spec args must be runtime fields, re-checked when the runner reads them.

    submit() validates this, but spec.json sits under the user-writable
    base_directory. A StateField smuggled into "args" would override the
    server-side config: for FeynRulesToUFOTool, wolframscript_path becomes
    argv[0] of a subprocess, stepping around the whitelist entirely.
    """
    base = _base()
    jid = jm.new_job_id()
    jdir = jm.job_dir(base, jid)
    os.makedirs(jdir)
    jm.write_json_atomic(os.path.join(jdir, "spec.json"), {
        "schema": jm.SPEC_SCHEMA, "job_id": jid, "tool": "audittrail",
        "args": {"stage": "x", "base_directory": "/somewhere/else"},
        "state": {"base_directory": base},
    })
    jm.write_json_atomic(os.path.join(jdir, "status.json"), {
        "schema": jm.STATUS_SCHEMA, "job_id": jid, "tool": "audittrail",
        "state": "queued", "pid": None, "exit_code": None,
        "submitted": None, "started": None, "finished": None, "error": None,
    })
    from tools.jobs.runner import main as runner_main
    rc = runner_main(jdir)
    st = jm.read_json(os.path.join(jdir, "status.json"))
    ok = rc == 2 and st["state"] == "failed" and "runtime fields" in (st["error"] or "")
    print(f"[{'✓' if ok else 'x'}] test_runner_rejects_state_fields_in_args")
    return ok


def test_runner_rejects_tampered_spec() -> bool:
    """A spec whose tool key isn't whitelisted must fail, not import."""
    base = _base()
    jid = jm.new_job_id()
    jdir = jm.job_dir(base, jid)
    os.makedirs(jdir)
    jm.write_json_atomic(os.path.join(jdir, "spec.json"), {
        "schema": jm.SPEC_SCHEMA, "job_id": jid, "tool": "osystem",
        "args": {}, "state": {"base_directory": base},
    })
    jm.write_json_atomic(os.path.join(jdir, "status.json"), {
        "schema": jm.STATUS_SCHEMA, "job_id": jid, "tool": "osystem",
        "state": "queued", "pid": None, "exit_code": None,
        "submitted": None, "started": None, "finished": None, "error": None,
    })
    from tools.jobs.runner import main as runner_main
    rc = runner_main(jdir)
    st = jm.read_json(os.path.join(jdir, "status.json"))
    ok = rc == 2 and st["state"] == "failed" and "whitelisted" in st["error"]
    print(f"[{'✓' if ok else '✗'}] test_runner_rejects_tampered_spec")
    return ok


def _state(base: str, jid: str) -> str:
    return jm.effective_status(jm.job_dir(base, jid)).get("state")


def test_slot_cap_queues_extra_jobs() -> bool:
    """With one slot, a second job waits queued until the first finishes."""
    base = _base()
    old = os.environ.get("HEPTAPOD_MAX_JOBS")
    os.environ["HEPTAPOD_MAX_JOBS"] = "1"
    try:
        a, _, _ = _launch_sleep_job(base, seconds=3)
        time.sleep(1.5)
        b, _, _ = _launch_sleep_job(base, seconds=1)
        time.sleep(1.5)
        a_running, b_queued = _state(base, a) == "running", _state(base, b) == "queued"
        b_final = _wait(base, b, seconds=20).get("state")
        a_final = _state(base, a)
    finally:
        if old is None:
            os.environ.pop("HEPTAPOD_MAX_JOBS", None)
        else:
            os.environ["HEPTAPOD_MAX_JOBS"] = old
    ok = a_running and b_queued and a_final == "done" and b_final == "done"
    print(f"[{'✓' if ok else '✗'}] test_slot_cap_queues_extra_jobs "
          f"(a_running={a_running}, b_queued={b_queued}, a={a_final}, b={b_final})")
    return ok


def test_cancel_job_waiting_for_slot() -> bool:
    """A job waiting for a slot can be cancelled and never runs its tool."""
    base = _base()
    old = os.environ.get("HEPTAPOD_MAX_JOBS")
    os.environ["HEPTAPOD_MAX_JOBS"] = "1"
    try:
        a, _, _ = _launch_sleep_job(base, seconds=4)
        time.sleep(1.5)
        b, b_pid_file, _ = _launch_sleep_job(base, seconds=30)
        time.sleep(1.5)
        res = jm.cancel(base, b)
        _wait(base, a, seconds=20)
    finally:
        if old is None:
            os.environ.pop("HEPTAPOD_MAX_JOBS", None)
        else:
            os.environ["HEPTAPOD_MAX_JOBS"] = old
    ok = res.get("state") == "cancelled" and not os.path.exists(b_pid_file)
    print(f"[{'✓' if ok else '✗'}] test_cancel_job_waiting_for_slot ({res})")
    return ok


def test_mg5_card_core_cap() -> bool:
    """An MG5 card without nb_core or run_mode gets one; others are left alone."""
    from tools.mg5.mg5 import _cap_cores
    plain = "import model sm\ngenerate p p > z\noutput x\nlaunch\n"
    capped = _cap_cores(plain)
    own = "set nb_core 2\n" + plain
    serial = "set run_mode 0\n" + plain
    ok = (capped.startswith("set nb_core 4\n") and capped.endswith(plain)
          and _cap_cores(own) == own and _cap_cores(serial) == serial)
    print(f"[{'✓' if ok else '✗'}] test_mg5_card_core_cap")
    return ok


TESTS = [
    test_job_id_format,
    test_validate_args_against_real_tool,
    test_whitelist_names_match_tools,
    test_submit_rejections,
    test_detached_round_trip,
    test_result_before_done_and_bad_ids,
    test_stale_running_detection,
    test_listing,
    test_runner_rejects_state_fields_in_args,
    test_runner_rejects_tampered_spec,
    test_stale_queued_detection,
    test_finished_runner_is_reaped,
    test_render_progress,
    test_progress_reported,
    test_cancel_stops_job_and_children,
    test_cancel_queued_job_never_runs,
    test_watchdog_stops_job_and_children,
    test_slot_cap_queues_extra_jobs,
    test_cancel_job_waiting_for_slot,
    test_mg5_card_core_cap,
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run tests for the jobs toolkit")
    parser.add_argument("--keep-files", action="store_true")
    args = parser.parse_args()

    all_passed = True
    for test in TESTS:
        try:
            if not test():
                all_passed = False
        except Exception as e:  # noqa: BLE001
            print(f"[✗] {test.__name__} failed: {e}\n")
            all_passed = False

    if not args.keep_files:
        cleanup_test_files()

    if all_passed:
        print("[✓] All tests passed!\n")
        sys.exit(0)
    print("[✗] Some tests failed!\n")
    sys.exit(1)
