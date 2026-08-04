"""
# sanitize.py is a part of the HEPTAPOD package.
# Copyright (C) 2026 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

Sanitize a FeynRules .fr file for a blank-slate reconstruction.

Strips everything that could identify the (public) model to a trained LLM —
comments, the M$Information block, the model name, and human-prose labels —
while leaving the physics content untouched, so the reconstruction has to come
from the declarations and Lagrangian alone. Field/parameter *symbol names*
(S1, yRR11) are deliberately kept: the physicist reviewer needs them to map
the reconstruction back onto the file. State that residual-hint scope in any
report built on top of this.
"""

from __future__ import annotations

import hashlib
import re
from typing import Dict, Tuple

from tools.frgen.fr_parser import _matching_brace, strip_fr_comments

_MODEL_NAME_RE = re.compile(
    r'(M\$ModelName\s*=\s*)"((?:[^"\\]|\\.)*)"(\s*;)'
)
# Prose-bearing options whose values are human labels, not physics.
_PROSE_OPT_RE = re.compile(
    r'\b(Description|FullName)(\s*->\s*)"(?:[^"\\]|\\.)*"'
)

ANON_MODEL_NAME = "ANON-MODEL"


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def _remove_information_blocks(text: str) -> Tuple[str, int]:
    """Delete every ``M$Information = { ... };`` statement (brace-aware)."""
    removed = 0
    while True:
        idx = text.find("M$Information")
        if idx < 0:
            break
        brace = text.find("{", idx)
        if brace < 0:
            # malformed; drop just the token to avoid an infinite loop
            text = text[:idx] + text[idx + len("M$Information"):]
            removed += 1
            continue
        end = _matching_brace(text, brace)
        if end < 0:
            text = text[:idx]
            removed += 1
            break
        stop = end + 1
        while stop < len(text) and text[stop] in " \t\r\n":
            stop += 1
        if stop < len(text) and text[stop] == ";":
            stop += 1
        text = text[:idx] + text[stop:]
        removed += 1
    return text, removed


def sanitize_fr(text: str) -> Tuple[str, Dict[str, object]]:
    """Return (sanitized_text, report).

    Applied transformations, in order:
      1. strip all ``(* ... *)`` comments (nesting-aware);
      2. remove every ``M$Information = {...};`` block;
      3. anonymize ``M$ModelName`` to "ANON-MODEL";
      4. blank Description/FullName prose strings.

    The physics (fields, quantum numbers, parameters, Lagrangian) is left
    byte-identical. ``parse_fr`` on the result yields the same structure.
    """
    original = text
    text = strip_fr_comments(text)
    text, n_info = _remove_information_blocks(text)

    original_name = None
    m = _MODEL_NAME_RE.search(text)
    if m:
        original_name = m.group(2)
        text = _MODEL_NAME_RE.sub(rf'\g<1>"{ANON_MODEL_NAME}"\g<3>', text, count=1)

    n_prose = len(_PROSE_OPT_RE.findall(text))
    text = _PROSE_OPT_RE.sub(r'\1\2""', text)

    # collapse the whitespace holes left by removals
    text = re.sub(r"\n{4,}", "\n\n\n", text)

    report: Dict[str, object] = {
        "original_model_name": original_name,
        "anon_model_name": ANON_MODEL_NAME,
        "information_blocks_removed": n_info,
        "prose_values_scrubbed": n_prose,
        "sha256_original": _sha256(original),
        "sha256_sanitized": _sha256(text),
        "blindness_scope": (
            "blank slate w.r.t. paper, authors, model name, comments and "
            "prose labels — NOT w.r.t. field/parameter symbol names, which "
            "are kept so the reviewer can map the reconstruction back"
        ),
    }
    return text, report
