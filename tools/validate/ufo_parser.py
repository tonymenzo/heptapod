"""
# ufo_parser.py is a part of the HEPTAPOD package.
# Copyright (C) 2025 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

Static, import-free parsing of UFO model files.

UFO ``particles.py`` executes arbitrary code on import (it instantiates
``Particle(...)`` objects and imports the model package), so we NEVER import it.
Instead we parse it with ``ast`` — which executes nothing and is robust to
formatting (multi-line kwargs, ``charge = -1/3`` BinOps, ``X = Y.anti()``). Used
by ValidateModelTool to check that the generated model's declared particles
actually reached the UFO with the right quantum numbers.
"""

from __future__ import annotations

import ast
import os
from fractions import Fraction
from typing import Any, Dict, List, Optional


def _attr_to_str(node: ast.AST) -> Optional[str]:
    """Render ``Param.MS1`` / a dotted attribute chain to a string."""
    parts: List[str] = []
    cur = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
        return ".".join(reversed(parts))
    return None


def fold_number(node: ast.AST) -> Optional[Fraction]:
    """Fold a numeric AST expression to an exact Fraction (or None if not numeric).

    Handles int/float constants and +,-,*,/ over them, plus unary +/- — enough
    for UFO charges like ``-1/3``, ``2/3``, ``0``. Floats are rationalized with a
    bounded denominator so ``0.6666...`` compares to ``2/3``.
    """
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool):
            return None
        if isinstance(node.value, int):
            return Fraction(node.value)
        if isinstance(node.value, float):
            return Fraction(node.value).limit_denominator(100000)
        return None
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        val = fold_number(node.operand)
        if val is None:
            return None
        return -val if isinstance(node.op, ast.USub) else val
    if isinstance(node, ast.BinOp) and isinstance(
        node.op, (ast.Div, ast.Mult, ast.Add, ast.Sub)
    ):
        left, right = fold_number(node.left), fold_number(node.right)
        if left is None or right is None:
            return None
        try:
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Add):
                return left + right
            return left - right
        except ZeroDivisionError:
            return None
    return None


def _kw_value(node: ast.AST) -> Any:
    """Interpret a Particle(...) kwarg value: number->Fraction, name/attr->str."""
    num = fold_number(node)
    if num is not None:
        return num
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Attribute):
        return _attr_to_str(node)
    if isinstance(node, ast.Name):
        return node.id
    return None


def parse_particles(particles_py_path: str) -> List[Dict[str, Any]]:
    """Parse a UFO ``particles.py`` into a list of particle dicts (no import).

    Each dict carries the kwargs found on the ``Particle(...)`` call (pdg_code,
    name, antiname, spin, color, charge as Fraction, mass/width as strings,
    ...) plus ``var``. ``X = Y.anti()`` assignments are recorded as
    antiparticles referencing ``Y``.
    """
    with open(particles_py_path, "r", encoding="utf-8", errors="replace") as fh:
        tree = ast.parse(fh.read())

    particles: List[Dict[str, Any]] = []
    anti: List[Dict[str, str]] = []
    for stmt in tree.body:
        if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
            continue
        target = stmt.targets[0]
        if not isinstance(target, ast.Name):
            continue
        var = target.id
        val = stmt.value
        if (
            isinstance(val, ast.Call)
            and isinstance(val.func, ast.Name)
            and val.func.id == "Particle"
        ):
            rec: Dict[str, Any] = {"var": var}
            for kw in val.keywords:
                if kw.arg is None:
                    continue
                rec[kw.arg] = _kw_value(kw.value)
            particles.append(rec)
        elif (
            isinstance(val, ast.Call)
            and isinstance(val.func, ast.Attribute)
            and val.func.attr == "anti"
            and isinstance(val.func.value, ast.Name)
        ):
            anti.append({"var": var, "of": val.func.value.id})
    return particles


# FeynRules spin_type -> UFO spin (2s+1) convention.
_SPIN_2SP1 = {"S": 1, "F": 2, "V": 3, "R": 4, "T": 5, "W": 2, "RW": 4, "U": -1}


def _model_expected_color(indices: List[str]) -> Optional[int]:
    idx = {str(i) for i in (indices or [])}
    if "Colour" in idx or "Color" in idx:
        return 3
    if "Gluon" in idx or "Sextet" in idx:
        return None  # out of scope; don't assert
    return 1


def check_particle_properties(ufo_dir: str, model_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Check each declared physical particle appears in the UFO with matching
    spin / color / charge. Returns a list of ``{name, passed, detail}`` checks.
    """
    particles_py = os.path.join(ufo_dir, "particles.py")
    if not os.path.isfile(particles_py):
        return [{"name": "ufo_particles_parse", "passed": False, "detail": "particles.py not found"}]
    try:
        ufo = parse_particles(particles_py)
    except SyntaxError as e:
        return [{"name": "ufo_particles_parse", "passed": False, "detail": f"ast parse failed: {e}"}]

    by_name = {str(p.get("name")): p for p in ufo if p.get("name") is not None}
    by_pdg = {p.get("pdg_code"): p for p in ufo if isinstance(p.get("pdg_code"), int)}

    checks: List[Dict[str, Any]] = []
    for p in model_dict.get("particles", []):
        if p.get("unphysical"):
            continue
        want_name = p.get("particle_name") or p.get("class_name")
        if isinstance(want_name, list):
            want_name = want_name[0] if want_name else None
        want_pdg = p.get("pdg")
        if isinstance(want_pdg, list):
            want_pdg = want_pdg[0] if want_pdg else None

        found = by_name.get(str(want_name))
        if found is None and isinstance(want_pdg, int):
            found = by_pdg.get(want_pdg)
        label = f"particle_in_ufo:{want_name}"
        if found is None:
            checks.append({"name": label, "passed": False, "detail": f"'{want_name}' not found in UFO"})
            continue

        mism: List[str] = []
        # spin
        want_spin = _SPIN_2SP1.get(str(p.get("spin_type")))
        if want_spin is not None and found.get("spin") is not None and int(found["spin"]) != want_spin:
            mism.append(f"spin {found['spin']}!={want_spin}")
        # color
        want_color = _model_expected_color(p.get("indices", []))
        if want_color is not None and found.get("color") is not None and int(found["color"]) != want_color:
            mism.append(f"color {found['color']}!={want_color}")
        # charge
        q = (p.get("quantum_numbers") or {}).get("Q")
        if q is not None and found.get("charge") is not None:
            try:
                want_q = Fraction(str(q))
                got_q = found["charge"] if isinstance(found["charge"], Fraction) else Fraction(str(found["charge"]))
                if want_q != got_q:
                    mism.append(f"charge {got_q}!={want_q}")
            except (ValueError, ZeroDivisionError):
                pass

        checks.append(
            {
                "name": f"particle_props:{want_name}",
                "passed": not mism,
                "detail": "matches (spin/color/charge)" if not mism else "; ".join(mism),
            }
        )
    return checks
