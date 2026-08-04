"""
# jobs_tool.py is a part of the HEPTAPOD package.
# Copyright (C) 2026 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

Background-job tools: submitjob / jobstatus / jobresult.

Give any MCP agent the "background task" workflow for long-running physics
tools (Wolfram compiles, MadGraph runs, reverse checks): submit and get a
job_id back immediately, keep working, poll status, fetch the result, and
continue — instead of blocking the single MCP channel for minutes and
tripping client timeouts.
"""

from __future__ import annotations

import json
import os
from typing import Optional

from orchestral.tools.base.tool import BaseTool
from orchestral.tools.base.field_utils import RuntimeField, StateField

from tools.jobs import job_manager as jm
from tools.logging.audit import append_event

_WHITELIST_DOC = ", ".join(sorted(jm.WHITELIST))


class SubmitJobTool(BaseTool):
    """
    Submit a long-running heptapod tool as a detached background job.

    Use this instead of calling the tool directly whenever the run may take
    more than ~2 minutes (UFO compiles, validatemodel with madgraph_check,
    MadGraph event generation, reverselagrangian). Returns a job_id
    immediately; poll with jobstatus and fetch the output with jobresult,
    continuing other work in between.

    Input:
        tool_name: The tool to run (see the field description for the list).
        tool_args: JSON object of the target tool's runtime arguments only
            (paths to executables come from server config, not from here).
        ledger_name: Optional audit ledger to record job events in.
        max_seconds: Optional hard watchdog kill for the job.

    Returns JSON: {"status","schema","job_id","job_dir","tool","state","hint"}.
    """

    # ======================== Runtime fields ======================== #
    tool_name: str = RuntimeField(
        description=f"Tool to run in the background; one of: {_WHITELIST_DOC}"
    )
    tool_args: str = RuntimeField(
        description="JSON object of the target tool's runtime arguments"
    )
    ledger_name: Optional[str] = RuntimeField(
        default=None, description="Audit ledger name for job events"
    )
    max_seconds: Optional[int] = RuntimeField(
        default=None, description="Optional watchdog timeout for the job (seconds)"
    )
    # ================================================================ #

    # ========================= State fields ========================= #
    base_directory: str = StateField(description="Base sandbox directory")
    feynrules_path: Optional[str] = StateField(
        default="", description="FeynRules install root (forwarded to the job)"
    )
    wolframscript_path: Optional[str] = StateField(
        default="", description="wolframscript command/path (forwarded to the job)"
    )
    mg5_path: Optional[str] = StateField(
        default="", description="MG5_aMC install dir (forwarded to the job)"
    )
    blank_agent_cmd: Optional[str] = StateField(
        default="", description="Blank-slate agent command template (forwarded)"
    )
    # ================================================================ #

    def _run(self) -> str:
        try:
            args = json.loads(self.tool_args) if self.tool_args else {}
            if not isinstance(args, dict):
                raise ValueError("tool_args must be a JSON object")
        except (json.JSONDecodeError, ValueError) as e:
            return self.format_error(
                error="Invalid Input", reason=f"tool_args is not a JSON object: {e}"
            )

        state = {
            "base_directory": self.base_directory,
            "feynrules_path": self.feynrules_path or "",
            "wolframscript_path": self.wolframscript_path or "",
            "mg5_path": self.mg5_path or "",
            "blank_agent_cmd": self.blank_agent_cmd or "",
        }
        try:
            info = jm.submit(
                self.base_directory,
                self.tool_name,
                args,
                state,
                ledger_name=self.ledger_name,
                max_seconds=self.max_seconds,
            )
        except ValueError as e:
            return self.format_error(error="Invalid Input", reason=str(e))
        except OSError as e:
            return self.format_error(
                error="Job Submit Failed", reason=f"{type(e).__name__}: {e}"
            )

        try:
            append_event(
                self.base_directory,
                stage="job_submit",
                status="ok",
                summary=f"submitted {self.tool_name} as {info['job_id']}",
                data={"job_id": info["job_id"], "tool": self.tool_name},
                ledger_name=self.ledger_name or "audit.json",
            )
        except Exception:  # noqa: BLE001 — audit is best-effort
            pass

        return json.dumps(
            {
                "status": "ok",
                "schema": "job-submit-1.0",
                "job_id": info["job_id"],
                "job_dir": info["job_dir"],
                "tool": self.tool_name,
                "state": "queued",
                "hint": (
                    "poll with jobstatus(job_id=...); when state=done fetch "
                    "the output with jobresult(job_id=...). Keep working on "
                    "other steps while it runs."
                ),
            },
            indent=2,
        )


class JobStatusTool(BaseTool):
    """
    Check on background jobs.

    Input:
        job_id: The job to inspect. Omit to list all jobs, newest first.
        log_tail_lines: How many trailing log lines to include (default 25).

    Returns JSON: the job's status.json fields plus a log tail, or
    {"jobs": [...]} when listing. A job whose runner process died without
    finishing is reported as state="failed" with a stale-runner note.
    """

    # ======================== Runtime fields ======================== #
    job_id: Optional[str] = RuntimeField(
        default=None, description="Job id; omit to list all jobs"
    )
    log_tail_lines: Optional[int] = RuntimeField(
        default=25, description="Trailing log lines to include"
    )
    # ================================================================ #

    # ========================= State fields ========================= #
    base_directory: str = StateField(description="Base sandbox directory")
    # ================================================================ #

    def _run(self) -> str:
        if not self.job_id:
            return json.dumps(
                {"status": "ok", "schema": jm.STATUS_SCHEMA,
                 "jobs": jm.list_jobs(self.base_directory)},
                indent=2,
            )
        jdir = jm.job_dir(self.base_directory, self.job_id)
        if jdir is None or not os.path.isdir(jdir):
            return self.format_error(
                error="Job Not Found",
                reason=f"no job '{self.job_id}' under {jm.JOBS_DIRNAME}/",
            )
        st = jm.read_json(os.path.join(jdir, "status.json")) or {}
        if st.get("state") == "running" and not jm.pid_alive(st.get("pid")):
            st["state"] = "failed"
            st["error"] = "runner process died before finishing (stale job)"
        tail = ""
        try:
            with open(os.path.join(jdir, "log.txt"), "r", encoding="utf-8",
                      errors="replace") as fh:
                lines = fh.readlines()
            n = max(0, int(self.log_tail_lines or 25))
            tail = "".join(lines[-n:]) if n else ""
        except OSError:
            pass
        st["log_tail"] = tail
        st.setdefault("schema", jm.STATUS_SCHEMA)
        return json.dumps(st, indent=2)


class JobResultTool(BaseTool):
    """
    Fetch a finished background job's output.

    Input:
        job_id: The job whose result to fetch.

    Returns the job's result.json — {"schema","job_id","tool","tool_error",
    "output"} where output is the verbatim string the tool returned (usually
    its own JSON). Errors if the job is still queued/running.
    """

    # ======================== Runtime fields ======================== #
    job_id: str = RuntimeField(description="Job id to fetch the result for")
    # ================================================================ #

    # ========================= State fields ========================= #
    base_directory: str = StateField(description="Base sandbox directory")
    # ================================================================ #

    def _run(self) -> str:
        jdir = jm.job_dir(self.base_directory, self.job_id)
        if jdir is None or not os.path.isdir(jdir):
            return self.format_error(
                error="Job Not Found",
                reason=f"no job '{self.job_id}' under {jm.JOBS_DIRNAME}/",
            )
        st = jm.read_json(os.path.join(jdir, "status.json")) or {}
        state = st.get("state")
        if state == "failed" and not os.path.isfile(os.path.join(jdir, "result.json")):
            return self.format_error(
                error="Job Failed",
                reason=st.get("error") or "job failed before producing a result",
                suggestion="Inspect jobstatus(job_id=...) log_tail for the traceback.",
            )
        if state != "done":
            return self.format_error(
                error="Job Not Finished",
                reason=f"state={state or 'unknown'}",
                suggestion="Poll jobstatus(job_id=...) until state=done.",
            )
        res = jm.read_json(os.path.join(jdir, "result.json"))
        if res is None:
            return self.format_error(
                error="Result Missing",
                reason="job is done but result.json is unreadable",
            )
        return json.dumps(res, indent=2)
