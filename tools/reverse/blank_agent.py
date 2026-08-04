"""
# blank_agent.py is a part of the HEPTAPOD package.
# Copyright (C) 2026 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

Run a blank-slate agent: a fresh, sandboxed CLI-agent session in an isolated
working directory, with no MCP tools and no conversation history — only the
files placed in that directory and the prompt.

Engine-agnostic via a command template (``blank_agent_cmd`` config):
    codex exec --sandbox read-only --skip-git-repo-check --model gpt-5.5 \
        --output-last-message {output}
Tokens: ``{output}`` -> path the engine should write its final message to;
``{prompt}`` -> the prompt text. If no ``{prompt}`` token is present the
prompt is appended as the last argument (codex-exec style). A read-only
sandbox cannot write files itself, so the final message must flow out through
``{output}`` (written by the CLI process, outside the sandbox) or stdout —
never instruct the blank agent to write files.
"""

from __future__ import annotations

import os
import shlex
import signal
import subprocess
import time
from typing import Dict, Optional

DEFAULT_BLANK_AGENT_CMD = (
    "codex exec --sandbox read-only --skip-git-repo-check --model gpt-5.5 "
    "-c model_reasoning_effort=medium --output-last-message {output}"
)


def _build_argv(cmd_template: str, prompt: str, output_path: str) -> list:
    argv = []
    saw_prompt = False
    for token in shlex.split(cmd_template):
        if "{output}" in token:
            token = token.replace("{output}", output_path)
        if "{prompt}" in token:
            token = token.replace("{prompt}", prompt)
            saw_prompt = True
        argv.append(token)
    if not saw_prompt:
        argv.append(prompt)
    return argv


def run_blank_agent(
    cmd_template: str,
    workdir: str,
    prompt: str,
    output_path: str,
    timeout_sec: int = 900,
) -> Dict[str, object]:
    """Run the engine once; never raises. Returns a structured result:

    {"ok", "exit_code", "seconds", "output_text", "stderr_tail", "error"}
    ``output_text`` comes from ``output_path`` if the engine wrote it,
    else from captured stdout (final block).
    """
    argv = _build_argv(cmd_template, prompt, output_path)
    t0 = time.time()
    error: Optional[str] = None
    stdout, stderr, exit_code = "", "", -1
    try:
        proc = subprocess.Popen(
            argv,
            cwd=workdir,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        try:
            stdout, stderr = proc.communicate(timeout=timeout_sec)
            exit_code = proc.returncode
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            stdout, stderr = proc.communicate()
            exit_code = 124
            error = f"timed out after {timeout_sec}s"
    except FileNotFoundError as e:
        error = f"engine not found: {e}"
    except OSError as e:
        error = f"engine failed to start: {type(e).__name__}: {e}"

    seconds = round(time.time() - t0, 1)

    # persist raw streams next to the expected output for debugging
    out_dir = os.path.dirname(output_path) or "."
    try:
        os.makedirs(out_dir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(output_path))[0]
        with open(os.path.join(out_dir, f"{stem}_stdout.txt"), "w",
                  encoding="utf-8", errors="replace") as fh:
            fh.write(stdout or "")
        with open(os.path.join(out_dir, f"{stem}_stderr.txt"), "w",
                  encoding="utf-8", errors="replace") as fh:
            fh.write(stderr or "")
    except OSError:
        pass

    output_text = ""
    if os.path.isfile(output_path):
        try:
            with open(output_path, "r", encoding="utf-8", errors="replace") as fh:
                output_text = fh.read().strip()
        except OSError:
            output_text = ""
    if not output_text and stdout:
        output_text = stdout.strip()

    ok = exit_code == 0 and bool(output_text) and error is None
    if not ok and error is None:
        error = (
            f"engine exit {exit_code}"
            + ("" if output_text else ", no output produced")
        )
    return {
        "ok": ok,
        "exit_code": exit_code,
        "seconds": seconds,
        "output_text": output_text,
        "stderr_tail": (stderr or "")[-800:],
        "error": error,
    }
