"""
# width_gate.py is a part of the HEPTAPOD package.
# Copyright (C) 2025 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

Analytic decay-width gate for validating a generated UFO model.

Reads external parameter values and the FeynRules-computed partial-width
expressions from a UFO directory (statically, via ``ast`` — never importing the
model), evaluates a chosen partial width in a locked, whitelisted namespace, and
compares it to a textbook analytic formula. This is a license-cheap physics
check: it confirms the generated model reproduces a known closed-form width,
catching sign/normalization errors that structural checks miss.
"""

from __future__ import annotations

import ast
import cmath
import math
import os
from typing import Any, Dict, List, Optional, Tuple


def _num(node: ast.AST) -> Optional[float]:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return float(node.value)
        if isinstance(node.value, str):
            try:
                return float(node.value)
            except ValueError:
                return None
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        v = _num(node.operand)
        if v is None:
            return None
        return -v if isinstance(node.op, ast.USub) else v
    return None


def parse_external_params(ufo_dir: str) -> Dict[str, float]:
    """Return ``{name: value}`` for external (numeric) parameters in parameters.py."""
    path = os.path.join(ufo_dir, "parameters.py")
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        tree = ast.parse(fh.read())
    params: Dict[str, float] = {}
    for stmt in tree.body:
        if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
            continue
        if not isinstance(stmt.targets[0], ast.Name):
            continue
        val = stmt.value
        if not (isinstance(val, ast.Call) and isinstance(val.func, ast.Name) and val.func.id == "Parameter"):
            continue
        kw = {k.arg: k.value for k in val.keywords if k.arg}
        name_node = kw.get("name")
        name = name_node.value if isinstance(name_node, ast.Constant) else stmt.targets[0].id
        value = _num(kw["value"]) if "value" in kw else None
        if value is not None:
            params[str(name)] = value
    return params


def _final_state(key_node: ast.AST) -> Tuple[str, ...]:
    """Turn a partial-width dict key ``(P.e__minus__, P.u)`` into ``('e__minus__','u')``."""
    parts: List[str] = []
    elts = key_node.elts if isinstance(key_node, ast.Tuple) else [key_node]
    for el in elts:
        if isinstance(el, ast.Attribute):
            parts.append(el.attr)
        elif isinstance(el, ast.Name):
            parts.append(el.id)
    return tuple(parts)


def parse_decays(ufo_dir: str) -> List[Dict[str, Any]]:
    """Return decay records: ``{name, particle, partial_widths: {finals: expr}}``."""
    path = os.path.join(ufo_dir, "decays.py")
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        tree = ast.parse(fh.read())
    decays: List[Dict[str, Any]] = []
    for stmt in tree.body:
        if not isinstance(stmt, ast.Assign) or not isinstance(stmt.value, ast.Call):
            continue
        call = stmt.value
        if not (isinstance(call.func, ast.Name) and call.func.id == "Decay"):
            continue
        kw = {k.arg: k.value for k in call.keywords if k.arg}
        particle = None
        pnode = kw.get("particle")
        if isinstance(pnode, ast.Attribute):
            particle = pnode.attr
        elif isinstance(pnode, ast.Name):
            particle = pnode.id
        pw: Dict[Tuple[str, ...], str] = {}
        pwn = kw.get("partial_widths")
        if isinstance(pwn, ast.Dict):
            for k, v in zip(pwn.keys, pwn.values):
                if isinstance(v, ast.Constant) and isinstance(v.value, str):
                    pw[_final_state(k)] = v.value
        decays.append({"name": stmt.targets[0].id if isinstance(stmt.targets[0], ast.Name) else None,
                       "particle": particle, "partial_widths": pw})
    return decays


def find_partial_width(
    decays: List[Dict[str, Any]], particle: str, finals: Tuple[str, ...]
) -> Optional[str]:
    """Find the partial-width expression for ``particle`` decaying to ``finals``
    (order-insensitive on the final state)."""
    want = frozenset(finals)
    for d in decays:
        if d.get("particle") != particle:
            continue
        for key, expr in d["partial_widths"].items():
            if frozenset(key) == want:
                return expr
    return None


# Whitelist for evaluating UFO width expressions: arithmetic + cmath.sqrt/pi +
# abs + complexconjugate. No attribute access other than cmath.*, no calls
# other than those three, no names other than the supplied parameters.
_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant, ast.Name, ast.Load,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.USub, ast.UAdd,
    ast.Call, ast.Attribute,
)


def _verify_expr(node: ast.AST) -> None:
    for n in ast.walk(node):
        if not isinstance(n, _ALLOWED_NODES):
            raise ValueError(f"disallowed syntax in width expr: {type(n).__name__}")
        if isinstance(n, ast.Attribute):
            if not (isinstance(n.value, ast.Name) and n.value.id == "cmath"):
                raise ValueError("only cmath.* attributes allowed")
        if isinstance(n, ast.Call):
            f = n.func
            ok = (isinstance(f, ast.Name) and f.id in {"abs", "complexconjugate"}) or (
                isinstance(f, ast.Attribute)
                and isinstance(f.value, ast.Name)
                and f.value.id == "cmath"
                and f.attr in {"sqrt"}
            )
            if not ok:
                raise ValueError("disallowed function call in width expr")


def safe_eval_width(expr: str, params: Dict[str, float]) -> float:
    """Evaluate a UFO partial-width expression to a real number.

    Locked namespace (parameters + cmath + abs + a real ``complexconjugate``);
    the AST is whitelist-verified first. Returns the real part (widths are real).
    """
    tree = ast.parse(expr, mode="eval")
    _verify_expr(tree)
    ns: Dict[str, Any] = dict(params)
    ns["cmath"] = cmath
    ns["abs"] = abs
    ns["complexconjugate"] = lambda z: z.conjugate() if isinstance(z, complex) else z
    result = eval(compile(tree, "<width>", "eval"), {"__builtins__": {}}, ns)  # noqa: S307
    return float(getattr(result, "real", result))


def analytic_scalar_lq_width(m_lq: float, y: float) -> float:
    """Textbook Γ(S1 → q ℓ) for a scalar leptoquark, massless final state:
    Γ = |y|² · m_LQ / (16π)."""
    return (y ** 2) * m_lq / (16.0 * math.pi)


def compare_width(analytic: float, reference: float, rel_tol: float = 0.02) -> Dict[str, Any]:
    """Compare two widths within a relative tolerance."""
    denom = abs(reference) if reference else 1.0
    rel_err = abs(analytic - reference) / denom
    return {
        "passed": rel_err <= rel_tol,
        "analytic": analytic,
        "reference": reference,
        "rel_err": rel_err,
        "rel_tol": rel_tol,
    }
