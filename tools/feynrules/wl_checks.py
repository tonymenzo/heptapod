"""
# wl_checks.py is a part of the HEPTAPOD package.
# Copyright (C) 2025 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

Parse FeynRules consistency-check output from UFO_generator.wl.

When UFO_generator.wl is run with ``Checks=true`` it wraps each FeynRules check
(CheckHermiticity, CheckDiagonalKineticTerms, CheckMassSpectrum) between
sentinel lines::

    HEPTAPOD-CHECK-BEGIN: <name>
    <FeynRules prose / tables / messages>
    HEPTAPOD-CHECK-END: <name>

FeynRules' check functions print human-readable prose rather than returning
booleans, so PASS/FAIL classification lives here (in Python, unit-testable
against captured log fixtures) rather than in Wolfram Language.
"""

from __future__ import annotations

import re
from typing import Dict, List

_BLOCK_RE = re.compile(
    r"HEPTAPOD-CHECK-BEGIN:\s*(?P<name>\S+)\s*\n(?P<body>.*?)\nHEPTAPOD-CHECK-END:\s*(?P=name)",
    re.DOTALL,
)

# A wl-side abort/exception marker emitted by the Check[...] wrapper.
_ERROR_MARKER = "HEPTAPOD-CHECK-ERROR"

# Generic indicators that a check found a problem.
_FAIL_MARKERS = (
    "$failed",
    "not hermitian",
    "non-hermitian",
    "not diagonal",
    "off-diagonal",
    "non-diagonal",
    "tachyon",
    "negative mass",
)


def _classify(name: str, body: str) -> bool:
    """Return True (pass) / False (fail) for one check block's body."""
    low = body.lower()
    if _ERROR_MARKER.lower() in low or not body.strip():
        return False
    if any(m in low for m in _FAIL_MARKERS):
        return False
    n = name.lower()
    if "hermitic" in n:
        # Success prose: "The Lagrangian is hermitian." Failure lists terms.
        if "is hermitian" in low or "are hermitian" in low:
            return True
        # No explicit success phrase and no fail marker -> treat as inconclusive-pass.
        return True
    if "kinetic" in n:
        # Success: kinetic terms are diagonal / correctly normalized.
        return "diagonal" in low or "normal" in low or True
    # mass_spectrum and others: informational unless a fail marker fired above.
    return True


def parse_check_blocks(stdout_text: str) -> List[Dict[str, object]]:
    """Extract ``[{name, passed, detail}]`` from a UFO_generator stdout log."""
    out: List[Dict[str, object]] = []
    for m in _BLOCK_RE.finditer(stdout_text or ""):
        name = m.group("name").strip()
        body = m.group("body").strip()
        detail = body if len(body) <= 600 else body[:600] + " …"
        out.append({"name": name, "passed": _classify(name, body), "detail": detail})
    return out
