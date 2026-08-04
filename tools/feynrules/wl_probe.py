"""
# wl_probe.py is a part of the HEPTAPOD package.
# Copyright (C) 2026 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

Probe whether ``wolframscript`` can actually execute — i.e. it is both installed
AND the Wolfram product is activated. A configured-but-unactivated Mathematica
(no license) exits 0 while printing a license notice instead of the result, so a
returncode check is not enough. Used by the gated live tests so they SKIP rather
than FAIL on machines that have paths configured but no usable Wolfram license.
"""

from __future__ import annotations

import shutil
import subprocess


def wolframscript_activated(ws_path: str, timeout: int = 30) -> bool:
    """Return True iff ``ws_path`` runs a trivial computation and returns 2."""
    if not ws_path or not shutil.which(ws_path):
        return False
    try:
        proc = subprocess.run(
            [ws_path, "-code", "1+1"],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    combined = f"{proc.stdout or ''}\n{proc.stderr or ''}".lower()
    if "not activated" in combined or "license" in combined:
        return False
    return "2" in [line.strip() for line in (proc.stdout or "").splitlines()]
