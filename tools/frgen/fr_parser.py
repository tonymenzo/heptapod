"""
# fr_parser.py is a part of the HEPTAPOD package.
# Copyright (C) 2025 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

Lightweight parser for FeynRules ``.fr`` model files (the inverse of render.py).

A hand-rolled balanced-brace scanner — no Mathematica, no new dependencies.
It splits the ``M$ClassesDescription`` / ``M$Parameters`` / ``M$GaugeGroups``
blocks into ``Sym == {...}`` entries and each association into ``key -> value``
pairs at brace-depth 0, keeping values verbatim. Used to (a) build the
convention catalog from reference model files and (b) derive expected content
for the benchmark from reference implementations.

Regex alone cannot handle nested associations like ``{MS1, 1500.}`` or
``Value -> {yl[1,1] -> ye}``; this ~scanner does. pyparsing was rejected to
avoid re-adding a dependency the repo trimmed.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple


def strip_fr_comments(text: str) -> str:
    """Remove Mathematica ``(* ... *)`` comments, honoring nesting."""
    out: List[str] = []
    depth = 0
    i, n = 0, len(text)
    while i < n:
        two = text[i : i + 2]
        if two == "(*":
            depth += 1
            i += 2
            continue
        if two == "*)" and depth > 0:
            depth -= 1
            i += 2
            continue
        if depth == 0:
            out.append(text[i])
        i += 1
    return "".join(out)


def _matching_brace(s: str, open_idx: int) -> int:
    """Index of the ``}`` matching the ``{`` at ``open_idx`` (string-aware)."""
    depth = 0
    i = open_idx
    n = len(s)
    while i < n:
        c = s[i]
        if c == '"':
            i += 1
            while i < n and not (s[i] == '"' and s[i - 1] != "\\"):
                i += 1
            i += 1
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def find_block(text: str, name: str) -> Optional[str]:
    """Return the content inside the ``{...}`` following ``name = {`` (or ``==``)."""
    idx = text.find(name)
    if idx < 0:
        return None
    br = text.find("{", idx)
    if br < 0:
        return None
    end = _matching_brace(text, br)
    if end < 0:
        return None
    return text[br + 1 : end]


def split_top_level(s: str, sep: str = ",") -> List[str]:
    """Split ``s`` on ``sep`` at bracket-depth 0, skipping quoted strings."""
    parts: List[str] = []
    cur: List[str] = []
    depth = 0
    i, n = 0, len(s)
    while i < n:
        c = s[i]
        if c == '"':
            cur.append(c)
            i += 1
            while i < n:
                cur.append(s[i])
                if s[i] == '"' and s[i - 1] != "\\":
                    i += 1
                    break
                i += 1
            continue
        if c in "{[(":
            depth += 1
        elif c in "}])":
            depth -= 1
        elif c == sep and depth == 0:
            parts.append("".join(cur))
            cur = []
            i += 1
            continue
        cur.append(c)
        i += 1
    tail = "".join(cur)
    if tail.strip():
        parts.append(tail)
    return parts


def _split_on_arrow(pair: str) -> Optional[Tuple[str, str]]:
    """Split ``key -> value`` (or ``:>``) on the first top-level arrow."""
    depth = 0
    i, n = 0, len(pair)
    while i < n:
        c = pair[i]
        if c == '"':
            i += 1
            while i < n and not (pair[i] == '"' and pair[i - 1] != "\\"):
                i += 1
            i += 1
            continue
        if c in "{[(":
            depth += 1
        elif c in "}])":
            depth -= 1
        elif depth == 0 and c == "-" and i + 1 < n and pair[i + 1] == ">":
            return pair[:i], pair[i + 2 :]
        elif depth == 0 and c == ":" and i + 1 < n and pair[i + 1] == ">":
            return pair[:i], pair[i + 2 :]
        i += 1
    return None


def parse_association(assoc: str) -> Dict[str, str]:
    """Parse ``{ Key -> value, ... }`` inner text into a dict of verbatim values."""
    opts: Dict[str, str] = {}
    for pair in split_top_level(assoc, ","):
        kv = _split_on_arrow(pair)
        if kv is None:
            continue
        opts[kv[0].strip()] = kv[1].strip()
    return opts


_ENTRY_RE = re.compile(r"\s*([A-Za-z0-9$]+(?:\[[0-9]+\])?)\s*==\s*")


def _parse_entries(block: str) -> List[Tuple[str, Dict[str, str]]]:
    entries: List[Tuple[str, Dict[str, str]]] = []
    for raw in split_top_level(block, ","):
        m = _ENTRY_RE.match(raw)
        if not m:
            continue
        lhs = m.group(1)
        rest = raw[m.end() :]
        br = rest.find("{")
        if br < 0:
            continue
        end = _matching_brace(rest, br)
        if end < 0:
            continue
        entries.append((lhs, parse_association(rest[br + 1 : end])))
    return entries


def _unquote(v: Optional[str]) -> Optional[str]:
    if v is None:
        return None
    v = v.strip()
    if len(v) >= 2 and v[0] == '"' and v[-1] == '"':
        return v[1:-1]
    return v


def parse_quantum_numbers(value: Optional[str]) -> Dict[str, str]:
    """``{Q -> -1/3, LeptonNumber -> 1}`` -> ``{'Q': '-1/3', ...}``."""
    if not value:
        return {}
    inner = value.strip()
    if inner.startswith("{") and inner.endswith("}"):
        inner = inner[1:-1]
    return {k.strip(): v.strip() for k, v in (parse_association(inner)).items()}


_CLASS_LABEL_RE = re.compile(r"^([A-Za-z]+)\[([0-9]+)\]$")


def parse_fr(text: str) -> Dict[str, Any]:
    """Parse a ``.fr`` file's text into a structured dict.

    Returns ``{model_name, classes: [...], parameters: [...], gauge_groups: [...]}``.
    Each class: ``{label, spin_type, class_index, class_name, self_conjugate,
    indices, quantum_numbers, mass, pdg, particle_name, options}``. Each
    parameter: ``{name, parameter_type, block_name, value, interaction_order,
    options}``.
    """
    text = strip_fr_comments(text)

    name_m = re.search(r'M\$ModelName\s*=\s*"([^"]*)"', text)
    model_name = name_m.group(1) if name_m else None

    classes: List[Dict[str, Any]] = []
    cblock = find_block(text, "M$ClassesDescription")
    if cblock is not None:
        for label, opts in _parse_entries(cblock):
            m = _CLASS_LABEL_RE.match(label)
            spin_type = m.group(1) if m else None
            class_index = int(m.group(2)) if m else None
            classes.append(
                {
                    "label": label,
                    "spin_type": spin_type,
                    "class_index": class_index,
                    "class_name": _unquote(opts.get("ClassName")),
                    "self_conjugate": opts.get("SelfConjugate"),
                    "indices": opts.get("Indices"),
                    "quantum_numbers": parse_quantum_numbers(opts.get("QuantumNumbers")),
                    "mass": opts.get("Mass"),
                    "pdg": opts.get("PDG"),
                    "particle_name": _unquote(opts.get("ParticleName")),
                    "options": opts,
                }
            )

    parameters: List[Dict[str, Any]] = []
    pblock = find_block(text, "M$Parameters")
    if pblock is not None:
        for name, opts in _parse_entries(pblock):
            parameters.append(
                {
                    "name": name,
                    "parameter_type": opts.get("ParameterType"),
                    "block_name": opts.get("BlockName") or opts.get("Blockname"),
                    "value": opts.get("Value"),
                    "interaction_order": opts.get("InteractionOrder"),
                    "options": opts,
                }
            )

    gauge_groups: List[Dict[str, Any]] = []
    gblock = find_block(text, "M$GaugeGroups")
    if gblock is not None:
        for gname, opts in _parse_entries(gblock):
            gauge_groups.append({"name": gname, "options": opts})

    return {
        "model_name": model_name,
        "classes": classes,
        "parameters": parameters,
        "gauge_groups": gauge_groups,
    }


def parse_fr_file(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        return parse_fr(fh.read())


_LAGRANGIAN_STMT_RE = re.compile(
    r"^\s*([A-Za-z][A-Za-z0-9]*)\s*(:=|=)\s*(.+)$", re.DOTALL
)

# Top-level assignments that are FeynRules declarations, not Lagrangian terms.
# M$... blocks self-exclude via the name regex ($ not allowed); these are the
# bare-name declarations that would otherwise slip through.
_NON_LAGRANGIAN_NAMES = {"FeynmanGauge", "IndexRange", "IndexStyle", "GaugeXi"}


def parse_lagrangian_terms(text: str) -> List[Dict[str, str]]:
    """Extract top-level Lagrangian assignments from a ``.fr`` file.

    Returns ``[{"name": "LkinS1", "op": "=", "expression": "<verbatim RHS>"}]``
    for every top-level ``Name := rhs;`` / ``Name = rhs;`` statement.
    ``M$...`` blocks are excluded by the name pattern; ``;`` inside
    ``Block[{...}, ...]`` bodies never splits because ``split_top_level``
    tracks bracket depth. Complements :func:`parse_fr`, which deliberately
    only reads the declaration blocks.
    """
    terms: List[Dict[str, str]] = []
    for stmt in split_top_level(strip_fr_comments(text), ";"):
        m = _LAGRANGIAN_STMT_RE.match(stmt)
        if not m:
            continue
        name, op, rhs = m.group(1), m.group(2), m.group(3).strip()
        if name in _NON_LAGRANGIAN_NAMES or not rhs:
            continue
        terms.append({"name": name, "op": op, "expression": rhs})
    return terms
