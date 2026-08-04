"""
# frmodel.py is a part of the HEPTAPOD package.
# Copyright (C) 2025 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

Structured schema for a FeynRules ``.fr`` model file.

This is the shared representation produced by Lagrangian extraction and consumed
by the ``.fr`` generator (render.py). Design principle: STRUCTURE the regular,
error-prone blocks (M$Parameters, M$ClassesDescription, M$GaugeGroups,
M$Information, IndexRange) so an LLM never hand-writes their Mathematica syntax,
and CARRY the irregular parts (Lagrangian algebra, mixing blocks, NLO
bookkeeping) as verbatim Mathematica strings.

All numeric fields are strings so exact FeynRules syntax survives untouched:
rationals like ``-1/3``, decimals like ``1500.``, scientific ``2.5*^-3``.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, model_validator

# A verbatim numeric string: "-1/3", "1500.", "2.5*^-3", "Internal", "Automatic".
NumberStr = str


class Rule(BaseModel):
    """One Mathematica replacement rule, e.g. ``CKM[1,2] -> Sin[cabi]``."""

    lhs: str
    rhs: str
    delayed: bool = False  # False -> "->", True -> ":>"

    def render(self) -> str:
        return f"{self.lhs} {':>' if self.delayed else '->'} {self.rhs}"


class ParamType(str, Enum):
    External = "External"
    Internal = "Internal"


class SpinType(str, Enum):
    S = "S"  # scalar
    F = "F"  # fermion
    V = "V"  # vector
    U = "U"  # ghost
    T = "T"  # tensor
    W = "W"  # Weyl fermion
    R = "R"  # spin-3/2
    RW = "RW"


class RangeKind(str, Enum):
    Unfold = "Unfold"
    NoUnfold = "NoUnfold"
    Range = "Range"


class PropType(str, Enum):
    Sine = "Sine"
    Straight = "Straight"
    ScalarDash = "ScalarDash"
    GhostDash = "GhostDash"
    Curly = "Curly"
    W = "W"
    C = "C"
    S = "S"
    D = "D"


class PropArrow(str, Enum):
    true = "True"
    false = "False"
    none = "None"
    forward = "Forward"


class ModelInfo(BaseModel):
    authors: List[str]
    version: str
    date: str
    institutions: List[str] = []
    emails: List[str] = []


class GaugeGroup(BaseModel):
    """A M$GaugeGroups entry. Omitted entirely by BSM add-on models."""

    name: str  # LHS symbol: U1Y, SU2L, SU3C
    abelian: bool
    gauge_boson: str
    coupling_constant: Optional[str] = None
    charge: Optional[str] = None  # required iff abelian
    structure_constant: Optional[str] = None  # required iff non-abelian
    symmetric_tensor: Optional[str] = None
    representations: List[Tuple[str, str]] = []  # [("T","Colour")] or many
    definitions: List[str] = []  # VERBATIM rule strings

    @model_validator(mode="after")
    def _check(self) -> "GaugeGroup":
        if self.abelian and not self.charge:
            raise ValueError(f"gauge group {self.name}: Abelian group needs charge")
        if not self.abelian and not self.structure_constant:
            raise ValueError(
                f"gauge group {self.name}: non-Abelian group needs structure_constant"
            )
        return self


class IndexDecl(BaseModel):
    name: str  # Colour, Generation, SU2W ...
    range_kind: RangeKind
    size: int  # Range[size]
    style_symbol: Optional[str] = None  # IndexStyle[name, style_symbol]


class Parameter(BaseModel):
    """A M$Parameters entry (LHS symbol of ``==``)."""

    name: str
    parameter_type: Optional[ParamType] = None
    value: Optional[NumberStr] = None  # scalar; tensors use value_rules
    value_rules: List[Rule] = []  # tensor Value -> { ... }
    complex: Optional[bool] = None  # ComplexParameter
    block_name: Optional[str] = None  # External only
    order_block: Optional[int] = None  # External SCALAR only
    interaction_order: Optional[Tuple[str, int]] = None  # -> {NP, 1}
    indices: List[str] = []  # non-empty => tensor
    definitions: List[Rule] = []
    parameter_name: Optional[str] = None
    tex: Optional[str] = None
    description: Optional[str] = None
    tensor_class: Optional[str] = None
    # tensor-only flags
    unitary: Optional[bool] = None
    hermitian: Optional[bool] = None
    orthogonal: Optional[bool] = None
    allow_summation: Optional[bool] = None

    @property
    def is_tensor(self) -> bool:
        return bool(self.indices)

    @model_validator(mode="after")
    def _check(self) -> "Parameter":
        if self.is_tensor:
            if self.value is not None:
                raise ValueError(
                    f"parameter {self.name}: tensor uses value_rules, not scalar value"
                )
            if self.order_block is not None:
                raise ValueError(
                    f"parameter {self.name}: order_block forbidden on tensors"
                )
        exclusive = [f for f in (self.unitary, self.hermitian, self.orthogonal) if f]
        if len(exclusive) > 1:
            raise ValueError(
                f"parameter {self.name}: at most one of unitary/hermitian/orthogonal"
            )
        if (self.unitary or self.hermitian) and self.complex is False:
            raise ValueError(
                f"parameter {self.name}: unitary/hermitian require complex=True"
            )
        if self.orthogonal and self.complex is True:
            raise ValueError(f"parameter {self.name}: orthogonal must be real")
        if self.parameter_type == ParamType.External and self.complex is True:
            raise ValueError(
                f"parameter {self.name}: External params must be real "
                "(split into real/imag or make Internal)"
            )
        return self


class MassSpec(BaseModel):
    """Renders 0 | {sym} | {sym,val} | {sym,Internal} | {gen,{m1,v1},...}."""

    massless: bool = False  # -> literal 0
    sym: Optional[str] = None
    value: Optional[NumberStr] = None  # number | "Internal" | "Automatic" | None
    members: List[Tuple[str, Optional[NumberStr]]] = []  # multi-gen

    @model_validator(mode="after")
    def _check(self) -> "MassSpec":
        if not self.massless and not self.sym:
            raise ValueError("MassSpec needs massless=True or a sym")
        return self


class ParticleClass(BaseModel):
    """A M$ClassesDescription entry, e.g. ``S[100] == {...}``."""

    spin_type: SpinType
    class_index: int  # the n in S[100]
    class_name: str
    self_conjugate: bool
    indices: List[str] = []  # -> Index[...] wrapped
    flavor_index: Optional[str] = None  # BARE name
    class_members: List[str] = []
    mass: Optional[MassSpec] = None
    width: Optional[MassSpec] = None
    quantum_numbers: Dict[str, NumberStr] = {}  # values are strings -> preserve -1/3
    pdg: Optional[Union[int, List[int]]] = None
    particle_name: Optional[Union[str, List[str]]] = None
    antiparticle_name: Optional[Union[str, List[str]]] = None
    full_name: Optional[Union[str, List[str]]] = None
    propagator_label: Optional[Union[str, List[str]]] = None
    propagator_type: Optional[PropType] = None
    propagator_arrow: Optional[PropArrow] = None
    unphysical: bool = False
    definitions: List[str] = []  # VERBATIM Mathematica rule strings
    ghost: Optional[str] = None  # U[n] only
    goldstone: Optional[str] = None  # S[n] only
    weyl_components: List[str] = []
    majorana_phase: Optional[str] = None
    chirality: Optional[str] = None

    @property
    def class_label(self) -> str:
        return f"{self.spin_type.value}[{self.class_index}]"

    @model_validator(mode="after")
    def _check(self) -> "ParticleClass":
        if self.unphysical:
            if self.mass or self.pdg:
                raise ValueError(
                    f"particle {self.class_name}: unphysical fields omit mass/pdg"
                )
            if not self.definitions:
                raise ValueError(
                    f"particle {self.class_name}: unphysical fields require definitions"
                )
        if len(self.class_members) > 1:
            if not self.flavor_index:
                raise ValueError(
                    f"particle {self.class_name}: multi-member class needs flavor_index"
                )
            n = len(self.class_members)
            for lst in (
                self.pdg,
                self.particle_name,
                self.antiparticle_name,
                self.full_name,
            ):
                if isinstance(lst, list) and len(lst) != n:
                    raise ValueError(
                        f"particle {self.class_name}: list length must equal "
                        f"member count ({n})"
                    )
        if self.spin_type == SpinType.U:
            if self.self_conjugate:
                raise ValueError(
                    f"particle {self.class_name}: ghosts must be self_conjugate=False"
                )
            if not self.ghost:
                raise ValueError(f"particle {self.class_name}: ghost needs ghost=<boson>")
        return self


class LagrangianTerm(BaseModel):
    name: str  # LHS symbol
    expression: str  # VERBATIM Mathematica RHS
    delayed: bool = False  # True -> ":=", False -> "="


def _references(text: str, name: str) -> bool:
    """True if ``name`` appears in ``text`` as a whole Mathematica token."""
    if not text:
        return False
    return re.search(rf"(?<![A-Za-z0-9$]){re.escape(name)}(?![A-Za-z0-9$])", text) is not None


def _param_refs(p: Parameter, names: set) -> set:
    """Names of other parameters referenced by ``p``'s value/rules/definitions."""
    texts: List[str] = []
    if p.value:
        texts.append(p.value)
    for r in p.value_rules:
        texts.append(r.rhs)
    for r in p.definitions:
        texts.append(r.rhs)
    blob = " ".join(texts)
    return {n for n in names if n != p.name and _references(blob, n)}


def topo_sort_parameters(params: List[Parameter]) -> List[Parameter]:
    """Order parameters so each is defined before others that reference it.

    Stable (preserves input order among independent params). Raises on a
    dependency cycle.
    """
    names = {p.name for p in params}
    deps = {p.name: _param_refs(p, names) for p in params}
    by_name = {p.name: p for p in params}

    ordered: List[Parameter] = []
    emitted: set = set()
    remaining = [p.name for p in params]  # preserve input order

    while remaining:
        progressed = False
        still: List[str] = []
        for n in remaining:
            if deps[n] <= emitted:
                ordered.append(by_name[n])
                emitted.add(n)
                progressed = True
            else:
                still.append(n)
        remaining = still
        if not progressed:
            raise ValueError(
                f"cyclic parameter dependency among: {sorted(remaining)}"
            )
    return ordered


class FeynRulesModel(BaseModel):
    """A complete FeynRules ``.fr`` model, ready to render."""

    model_name: str
    info: ModelInfo
    interaction_order_hierarchy: List[Tuple[str, int]] = []
    interaction_order_limit: List[Tuple[str, int]] = []
    feynman_gauge: Optional[bool] = None
    vevs: List[Tuple[str, str]] = []  # {{Phi[..], vev}}
    gauge_groups: List[GaugeGroup] = []
    index_decls: List[IndexDecl] = []
    parameters: List[Parameter] = []
    particles: List[ParticleClass] = []
    gauge_xi: List[Tuple[str, str]] = []  # (V[3], "GaugeXi[W]")
    lagrangian_terms: List[LagrangianTerm] = []
    # Verbatim escape hatches for irregular constructs (FR$LoopSwitches,
    # Mix[...] mixing blocks, SUSY superfields, ...).
    raw_preamble: List[str] = []  # emitted after M$Information, before IndexRange
    raw_blocks: List[str] = []  # emitted just before Lagrangian terms

    @model_validator(mode="after")
    def _check(self) -> "FeynRulesModel":
        labels = [p.class_label for p in self.particles]
        if len(labels) != len(set(labels)):
            dupes = sorted({x for x in labels if labels.count(x) > 1})
            raise ValueError(f"duplicate ParticleClass labels: {dupes}")

        # Lorentz/Spin* are FeynRules built-ins. Colour/Gluon/Generation/SU2D/
        # SU2W are declared by SM.fr, which every BSM add-on is loaded on top of,
        # so they need no IndexRange in the add-on itself (matches the reference
        # S1_LQ_RR.fr, which uses Index[Colour] without redeclaring it). Only
        # genuinely new indices must be declared in index_decls.
        builtin = {
            "Lorentz", "Spin", "Spin1", "Spin2",
            "Colour", "Gluon", "Generation", "SU2D", "SU2W",
        }
        declared = {d.name for d in self.index_decls} | builtin
        used = {i for p in self.particles for i in p.indices} | {
            i for prm in self.parameters for i in prm.indices
        }
        missing = used - declared
        if missing:
            raise ValueError(
                f"indices used without an IndexRange declaration: {sorted(missing)}"
            )

        self.parameters = topo_sort_parameters(self.parameters)
        return self
