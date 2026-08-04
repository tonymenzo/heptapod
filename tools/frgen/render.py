"""
# render.py is a part of the HEPTAPOD package.
# Copyright (C) 2025 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

Render a validated FeynRulesModel (frmodel.py) to a FeynRules ``.fr`` file.

The fiddly Mathematica syntax (association comma placement, ``->`` vs ``:>``,
mass tuples, ``Index[...]`` wrapping, rationals) is built here in tested Python;
a thin Jinja2 template (templates/model.fr.j2) supplies the file skeleton.
"""

from __future__ import annotations

import os
from typing import List, Optional, Sequence, Tuple

from jinja2 import Environment, FileSystemLoader, select_autoescape

from .frmodel import (
    FeynRulesModel,
    GaugeGroup,
    IndexDecl,
    MassSpec,
    Parameter,
    ParticleClass,
)

_TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "templates")


# --------------------------- scalar helpers --------------------------- #


def _bool(b: bool) -> str:
    return "True" if b else "False"


def _quote(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _strlist(items: Sequence[str]) -> str:
    """M$Information-style: single -> "x"; many -> {"x", "y"}."""
    if len(items) == 1:
        return _quote(items[0])
    return "{" + ", ".join(_quote(x) for x in items) + "}"


def _name_field(v) -> str:
    """ParticleName/FullName/PropagatorLabel: str -> quoted; list -> {...}."""
    if isinstance(v, list):
        return "{" + ", ".join(_quote(x) for x in v) + "}"
    return _quote(v)


def _pdg(v) -> str:
    if isinstance(v, list):
        return "{" + ", ".join(str(x) for x in v) + "}"
    return str(v)


def _indices(names: Sequence[str]) -> str:
    return "{" + ", ".join(f"Index[{n}]" for n in names) + "}"


# FeynRules QuantumNumbers holds only additive (U(1)-type) charges. Colour/SU2/
# etc. are REPRESENTATION labels fixed by a field's Indices, not quantum numbers;
# if left in QuantumNumbers, FeynRules emits an unevaluated `Colour[field]` into
# the UFO's particles.py, which MadGraph rejects with "name 'Colour' is not
# defined". Drop these reserved index/representation names.
_REP_LABELS = {
    "Colour", "Color", "Sextet", "Gluon", "Generation", "Spin", "Lorentz",
    "SU2", "SU2W", "SU2L", "SU2D", "SU2R",
}


def _additive_qns(d: dict) -> dict:
    return {k: v for k, v in d.items() if k not in _REP_LABELS}


def _reserved_mass_width_names(particles) -> set:
    """Mass/width parameter names FeynRules auto-declares from particle Mass/Width
    specs. Declaring them again in M$Parameters double-defines the name in the UFO
    (MadGraph InvalidModel), so the renderer drops those duplicates."""
    names: set = set()
    for p in particles or []:
        for spec in (getattr(p, "mass", None), getattr(p, "width", None)):
            if spec is None:
                continue
            if getattr(spec, "sym", None):
                names.add(spec.sym)
            for sub, _val in (getattr(spec, "members", None) or []):
                names.add(sub)
    return names


def _is_internal_param(p) -> bool:
    """Internal/derived parameters carry a formula (Definitions) that FeynRules
    cannot auto-create from a particle Mass/Width spec, so they must stay in
    M$Parameters even when named like a mass/width (e.g. a computed decay width
    Width -> {WV1, Internal}). Only plain External duplicates are dropped."""
    if getattr(p, "definitions", None):
        return True
    return "internal" in str(getattr(p, "parameter_type", "")).lower()


def _qn(d: dict) -> str:
    return "{" + ", ".join(f"{k} -> {v}" for k, v in _additive_qns(d).items()) + "}"


def _interaction_order(t: Tuple[str, int]) -> str:
    return f"{{{t[0]}, {t[1]}}}"


def _mw_value(v):
    """Normalize a mass/width value. FeynRules' ``Automatic`` width is computed
    only when AddDecays is on — which is off for us (broken on Wolfram >=15), so
    it leaks the literal ``Automatic`` into the UFO and MadGraph fails with
    ``name 'Automatic' is not defined``. Emit a numeric placeholder instead;
    MadGraph's compute_widths recovers the real width from the model."""
    return "1." if str(v).strip().lower() == "automatic" else v


def _mass(m: MassSpec) -> str:
    if m.massless:
        return "0"
    if m.members:
        parts = [m.sym]
        for sub, val in m.members:
            val = _mw_value(val)
            parts.append(f"{{{sub}, {val}}}" if val is not None else f"{{{sub}}}")
        return "{" + ", ".join(parts) + "}"
    if m.value is not None:
        return f"{{{m.sym}, {_mw_value(m.value)}}}"
    return f"{{{m.sym}}}"


def _reps(reps: Sequence[Tuple[str, str]]) -> str:
    if len(reps) == 1:
        r, s = reps[0]
        return f"{{{r}, {s}}}"
    return "{" + ", ".join(f"{{{r}, {s}}}" for r, s in reps) + "}"


def _assoc(pairs: List[Tuple[str, Optional[str]]], indent: str = "    ") -> str:
    """Render ``{ Key -> val, ... }``, dropping entries whose value is None."""
    kept = [(k, v) for k, v in pairs if v is not None]
    if not kept:
        return "{}"
    inner = ",\n".join(f"{indent}{k} -> {v}" for k, v in kept)
    close_indent = indent[:-2] if len(indent) >= 2 else ""
    return "{\n" + inner + "\n" + close_indent + "}"


# ------------------------------ blocks ------------------------------- #


def render_information(info) -> str:
    pairs = [
        ("Authors", _strlist(info.authors)),
        ("Version", _quote(info.version)),
        ("Date", _quote(info.date)),
        ("Institutions", _strlist(info.institutions) if info.institutions else None),
        ("Emails", _strlist(info.emails) if info.emails else None),
    ]
    return "M$Information = " + _assoc(pairs, indent="  ") + ";"


def _render_gauge_group(g: GaugeGroup) -> str:
    pairs: List[Tuple[str, Optional[str]]] = [
        ("Abelian", _bool(g.abelian)),
        ("CouplingConstant", g.coupling_constant),
        ("GaugeBoson", g.gauge_boson),
        ("Charge", g.charge),
        ("StructureConstant", g.structure_constant),
        ("Representations", _reps(g.representations) if g.representations else None),
        ("SymmetricTensor", g.symmetric_tensor),
        (
            "Definitions",
            ("{" + ", ".join(g.definitions) + "}") if g.definitions else None,
        ),
    ]
    return f"  {g.name} == " + _assoc(pairs, indent="    ")


def render_gauge_groups(groups: List[GaugeGroup]) -> str:
    entries = ",\n".join(_render_gauge_group(g) for g in groups)
    return "M$GaugeGroups = {\n" + entries + "\n};"


def render_index_decls(decls: List[IndexDecl]) -> str:
    lines: List[str] = []
    for d in decls:
        if d.range_kind.value == "Range":
            rng = f"Range[{d.size}]"
        else:
            rng = f"{d.range_kind.value}[Range[{d.size}]]"
        lines.append(f"IndexRange[Index[{d.name}]] = {rng};")
    for d in decls:
        if d.style_symbol:
            lines.append(f"IndexStyle[{d.name}, {d.style_symbol}];")
    return "\n".join(lines)


def _render_parameter(p: Parameter) -> str:
    pairs: List[Tuple[str, Optional[str]]] = [
        ("ParameterType", p.parameter_type.value if p.parameter_type else None),
    ]
    if p.is_tensor:
        pairs.append(
            (
                "Value",
                ("{" + ", ".join(r.render() for r in p.value_rules) + "}")
                if p.value_rules
                else None,
            )
        )
        pairs.append(("Indices", _indices(p.indices)))
    else:
        pairs.append(("Value", p.value))
    pairs += [
        ("ComplexParameter", _bool(p.complex) if p.complex is not None else None),
        ("BlockName", p.block_name),
        ("OrderBlock", str(p.order_block) if p.order_block is not None else None),
        (
            "InteractionOrder",
            _interaction_order(p.interaction_order) if p.interaction_order else None,
        ),
        (
            "Definitions",
            ("{" + ", ".join(r.render() for r in p.definitions) + "}")
            if p.definitions
            else None,
        ),
        ("ParameterName", p.parameter_name),
        ("TensorClass", p.tensor_class),
        ("Unitary", _bool(p.unitary) if p.unitary is not None else None),
        ("Hermitian", _bool(p.hermitian) if p.hermitian is not None else None),
        ("Orthogonal", _bool(p.orthogonal) if p.orthogonal is not None else None),
        (
            "AllowSummation",
            _bool(p.allow_summation) if p.allow_summation is not None else None,
        ),
        # TeX is a LaTeX label; render it as a quoted Mathematica string with
        # backslashes escaped ("\\Gamma_U"), else a raw value like \Gamma_U is a
        # Mathematica syntax error that aborts the whole M$Parameters block.
        ("TeX", _quote(p.tex) if p.tex else None),
        ("Description", _quote(p.description) if p.description else None),
    ]
    return f"  {p.name} == " + _assoc(pairs, indent="    ")


def render_parameters(params: List[Parameter]) -> str:
    entries = ",\n".join(_render_parameter(p) for p in params)
    return "M$Parameters = {\n" + entries + "\n};"


def _render_particle(p: ParticleClass) -> str:
    pairs: List[Tuple[str, Optional[str]]] = [
        ("ClassName", p.class_name),
        ("SelfConjugate", _bool(p.self_conjugate)),
        (
            "ClassMembers",
            ("{" + ", ".join(p.class_members) + "}") if p.class_members else None,
        ),
        ("Indices", _indices(p.indices) if p.indices else None),
        ("FlavorIndex", p.flavor_index),
        ("Mass", _mass(p.mass) if p.mass else None),
        ("Width", _mass(p.width) if p.width else None),
        ("QuantumNumbers",
         _qn(p.quantum_numbers) if _additive_qns(p.quantum_numbers or {}) else None),
        ("PDG", _pdg(p.pdg) if p.pdg is not None else None),
        ("ParticleName", _name_field(p.particle_name) if p.particle_name else None),
        (
            "AntiParticleName",
            _name_field(p.antiparticle_name) if p.antiparticle_name else None,
        ),
        ("FullName", _name_field(p.full_name) if p.full_name else None),
        (
            "PropagatorLabel",
            _name_field(p.propagator_label) if p.propagator_label else None,
        ),
        ("PropagatorType", p.propagator_type.value if p.propagator_type else None),
        ("PropagatorArrow", p.propagator_arrow.value if p.propagator_arrow else None),
        ("Ghost", p.ghost),
        ("Goldstone", p.goldstone),
        ("Chirality", p.chirality),
        ("MajoranaPhase", p.majorana_phase),
        (
            "WeylComponents",
            ("{" + ", ".join(p.weyl_components) + "}") if p.weyl_components else None,
        ),
        ("Unphysical", _bool(True) if p.unphysical else None),
        (
            "Definitions",
            ("{" + ", ".join(p.definitions) + "}") if p.definitions else None,
        ),
    ]
    return f"  {p.class_label} == " + _assoc(pairs, indent="    ")


def render_classes(particles: List[ParticleClass]) -> str:
    entries = ",\n".join(_render_particle(p) for p in particles)
    return "M$ClassesDescription = {\n" + entries + "\n};"


def render_model(model: FeynRulesModel) -> str:
    """Render a validated FeynRulesModel to a complete ``.fr`` string."""
    sections: List[str] = [render_information(model.info)]

    if model.interaction_order_hierarchy:
        ih = ", ".join(_interaction_order(t) for t in model.interaction_order_hierarchy)
        sections.append("M$InteractionOrderHierarchy = {" + ih + "};")
    if model.interaction_order_limit:
        il = ", ".join(_interaction_order(t) for t in model.interaction_order_limit)
        sections.append("M$InteractionOrderLimit = {" + il + "};")
    if model.feynman_gauge is not None:
        sections.append(f"FeynmanGauge = {_bool(model.feynman_gauge)};")
    if model.vevs:
        vv = ", ".join(f"{{{f}, {v}}}" for f, v in model.vevs)
        sections.append("M$vevs = {" + vv + "};")

    for raw in model.raw_preamble:
        sections.append(raw)

    if model.index_decls:
        sections.append(render_index_decls(model.index_decls))
    if model.gauge_groups:
        sections.append(render_gauge_groups(model.gauge_groups))
    if model.parameters:
        # FeynRules auto-declares the mass/width parameters named in each
        # particle's Mass/Width spec. Re-declaring them in M$Parameters makes the
        # UFO define the name twice, which MadGraph rejects ("name X define
        # multiple time"). Drop those duplicates; keep genuine couplings.
        reserved = _reserved_mass_width_names(model.particles)
        params = [p for p in model.parameters
                  if p.name not in reserved or _is_internal_param(p)]
        if params:
            sections.append(render_parameters(params))
    if model.particles:
        sections.append(render_classes(model.particles))
    for field, expr in model.gauge_xi:
        sections.append(f"GaugeXi[{field}] = {expr};")

    for raw in model.raw_blocks:
        sections.append(raw)

    for t in model.lagrangian_terms:
        op = ":=" if t.delayed else "="
        # Defensive: some (LLM-produced) expressions already carry a leading
        # assignment operator ("= ..." or ":= ..."); strip it so we don't emit a
        # duplicated operator (e.g. "L := := Block[...]").
        rhs = t.expression.lstrip()
        for lead in (":=", "="):
            if rhs.startswith(lead):
                rhs = rhs[len(lead):].lstrip()
                break
        sections.append(f"{t.name} {op} {rhs};")

    env = Environment(
        loader=FileSystemLoader(_TEMPLATE_DIR),
        autoescape=select_autoescape(enabled_extensions=(), default=False),
        keep_trailing_newline=True,
    )
    template = env.get_template("model.fr.j2")
    return template.render(model_name=model.model_name, body="\n\n".join(sections))
