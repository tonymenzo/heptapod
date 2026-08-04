#!/usr/bin/env python3
"""
# test_frgen.py is a part of the HEPTAPOD package.
# Copyright (C) 2025 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.

Tests for the FeynRules .fr generator.

Offline tests exercise the schema, validators, renderer, and the tool's JSON
interface (including error paths). An optional integration test compiles a
generated .fr into a UFO via FeynRulesToUFOTool when wolframscript + FeynRules
are configured (skipped gracefully otherwise).
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()
TOOL_DIR = SCRIPT_PATH.parent
REPO_ROOT = TOOL_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tools.frgen.frmodel import (
    FeynRulesModel,
    IndexDecl,
    LagrangianTerm,
    MassSpec,
    ModelInfo,
    Parameter,
    ParticleClass,
    topo_sort_parameters,
)
from tools.frgen.render import render_model
from tools.frgen.frgen_tool import GenerateFeynRulesModelTool

TEST_DIR = TOOL_DIR / "test_files"


def _s1_model() -> FeynRulesModel:
    """A scalar-leptoquark S1 add-on, mirroring S1_LQ_RR.fr."""
    return FeynRulesModel(
        model_name="S1_LQ_RR",
        info=ModelInfo(
            authors=["Tony Menzo"],
            version="1.0.0",
            date="11.11.2025",
            institutions=["University of Alabama", "Fermilab"],
            emails=["amenzo@ua.edu"],
        ),
        interaction_order_hierarchy=[("NP", 2)],
        index_decls=[IndexDecl(name="Colour", range_kind="NoUnfold", size=3)],
        parameters=[
            Parameter(
                name="yRR11",
                parameter_type="External",
                block_name="BSMINPUTS",
                complex=False,
                interaction_order=("NP", 1),
                value="0.5",
                description="S1-e-u Yukawa",
            )
        ],
        particles=[
            ParticleClass(
                spin_type="S",
                class_index=100,
                class_name="S1",
                self_conjugate=False,
                indices=["Colour"],
                mass=MassSpec(sym="MS1", value="1500."),
                width=MassSpec(sym="W1", value="Automatic"),
                quantum_numbers={"Q": "-1/3"},
                particle_name="S1",
                antiparticle_name="S1~",
                full_name="Scalar leptoquark S1",
                propagator_label="S1",
                propagator_type="ScalarDash",
                propagator_arrow="None",
            )
        ],
        lagrangian_terms=[
            LagrangianTerm(
                name="LkinS1",
                expression="Block[{mu,aa}, ExpandIndices[DC[S1bar[aa],mu] DC[S1[aa],mu] - MS1^2 * HC[S1].S1]]",
                delayed=False,
            ),
            LagrangianTerm(
                name="L1YukRRNonHC",
                expression="Block[{sp, aa}, yRR11 * anti[CC[uR]][sp, 1, aa].lR[sp, 1] * HC[S1][aa]]",
                delayed=True,
            ),
            LagrangianTerm(
                name="L1YukRR",
                expression="L1YukRRNonHC + HC[L1YukRRNonHC]",
                delayed=True,
            ),
            LagrangianTerm(name="LBSM", expression="LkinS1 + L1YukRR", delayed=False),
        ],
    )


def test_render_roundtrip() -> bool:
    print(">> Testing S1 render round-trip...\n")
    fr = render_model(_s1_model())
    required = [
        'M$ModelName = "S1_LQ_RR";',
        "M$InteractionOrderHierarchy = {{NP, 2}};",
        "IndexRange[Index[Colour]] = NoUnfold[Range[3]];",
        "yRR11 == {",
        "ParameterType -> External",
        "BlockName -> BSMINPUTS",
        "ComplexParameter -> False",
        "InteractionOrder -> {NP, 1}",
        "Value -> 0.5",
        "S[100] == {",
        "ClassName -> S1",
        "SelfConjugate -> False",
        "Mass -> {MS1, 1500.}",
        # Automatic widths are rendered as a numeric placeholder: FeynRules
        # only resolves Automatic when AddDecays is on (broken on Wolfram
        # >=15), otherwise the literal leaks into the UFO and MadGraph fails
        # with "name 'Automatic' is not defined". See render._mw_value.
        "Width -> {W1, 1.}",
        "QuantumNumbers -> {Q -> -1/3}",
        "Indices -> {Index[Colour]}",
        'ParticleName -> "S1"',
        'AntiParticleName -> "S1~"',
        "LkinS1 = Block[",
        "L1YukRRNonHC := Block[",
        "L1YukRR := L1YukRRNonHC + HC[L1YukRRNonHC];",
        "LBSM = LkinS1 + L1YukRR;",
    ]
    missing = [r for r in required if r not in fr]
    assert not missing, f"missing constructs:\n" + "\n".join(missing) + f"\n---\n{fr}"
    # Rational charge preserved, not floated.
    assert "-0.33" not in fr, "rational -1/3 was floated"
    print("[✓] Render round-trip test passed\n")
    return True


def test_tool_json_interface() -> bool:
    print(">> Testing tool JSON interface...\n")
    model_json = _s1_model().model_dump_json()
    tool = GenerateFeynRulesModelTool(model_json=model_json, base_directory=str(TEST_DIR))
    result = json.loads(tool._run())
    assert result.get("status") == "ok", result
    assert result["model_name"] == "S1_LQ_RR", result
    assert result["n_particles"] == 1 and result["n_parameters"] == 1, result
    assert (TEST_DIR / result["fr_path"]).exists(), result
    print("[✓] Tool JSON interface test passed\n")
    return True


def test_validation_error_surfaced() -> bool:
    print(">> Testing schema validation error path...\n")
    # Abelian gauge group without a charge -> validator error.
    bad = {
        "model_name": "Bad",
        "info": {"authors": ["X"], "version": "1", "date": "2026"},
        "gauge_groups": [{"name": "U1X", "abelian": True, "gauge_boson": "X"}],
    }
    tool = GenerateFeynRulesModelTool(
        model_json=json.dumps(bad), base_directory=str(TEST_DIR)
    )
    result = tool._run()
    assert "error" in result.lower() and "charge" in result.lower(), result
    print("[✓] Validation error test passed\n")
    return True


def test_number_must_be_string() -> bool:
    print(">> Testing numeric-as-string enforcement...\n")
    # quantum_numbers Q passed as a JSON number must be rejected (would float -1/3).
    bad = {
        "model_name": "Bad2",
        "info": {"authors": ["X"], "version": "1", "date": "2026"},
        "particles": [
            {
                "spin_type": "S",
                "class_index": 1,
                "class_name": "P",
                "self_conjugate": True,
                "quantum_numbers": {"Q": -0.3333},
            }
        ],
    }
    tool = GenerateFeynRulesModelTool(
        model_json=json.dumps(bad), base_directory=str(TEST_DIR)
    )
    result = tool._run()
    assert "error" in result.lower() or "valid" in result.lower(), result
    print("[✓] Numeric-as-string test passed\n")
    return True


def test_topo_sort() -> bool:
    print(">> Testing parameter topological sort...\n")
    # gs depends on aS -> aS must render first even though listed second.
    params = [
        Parameter(name="gs", parameter_type="Internal", value="Sqrt[4 Pi aS]"),
        Parameter(
            name="aS", parameter_type="External", block_name="SMINPUTS", value="0.118"
        ),
    ]
    ordered = topo_sort_parameters(params)
    names = [p.name for p in ordered]
    assert names.index("aS") < names.index("gs"), names
    # Substring false-positive guard: 'sw' must not be seen inside 'sw2'.
    p2 = [
        Parameter(name="sw2", parameter_type="Internal", value="1-(MW/MZ)^2"),
        Parameter(name="sw", parameter_type="Internal", value="Sqrt[sw2]"),
    ]
    o2 = [p.name for p in topo_sort_parameters(p2)]
    assert o2.index("sw2") < o2.index("sw"), o2
    print("[✓] Topological sort test passed\n")
    return True


def test_cyclic_dependency_raises() -> bool:
    print(">> Testing cyclic parameter dependency detection...\n")
    params = [
        Parameter(name="a", parameter_type="Internal", value="b + 1"),
        Parameter(name="b", parameter_type="Internal", value="a + 1"),
    ]
    try:
        topo_sort_parameters(params)
    except ValueError as e:
        assert "cyclic" in str(e).lower(), e
        print("[✓] Cyclic dependency test passed\n")
        return True
    raise AssertionError("expected a ValueError for the cycle")


def test_ufo_generation() -> bool:
    """Compile the generated .fr into a UFO (skipped unless the stack is configured)."""
    print(">> Testing .fr -> UFO via FeynRulesToUFOTool...\n")
    try:
        from config import feynrules_path, wolframscript_path
    except Exception:  # noqa: BLE001
        print("[⊘] Skipping: config.py not set up\n")
        return True
    if not shutil.which(wolframscript_path) or feynrules_path in (
        None,
        "/path/to/FeynRules",
    ):
        print("[⊘] Skipping: wolframscript/FeynRules not configured\n")
        return True
    from tools.feynrules.wl_probe import wolframscript_activated

    if not wolframscript_activated(wolframscript_path):
        print("[⊘] Skipping: wolframscript not activated (no Wolfram license)\n")
        return True

    from tools.feynrules import FeynRulesToUFOTool

    gen = GenerateFeynRulesModelTool(
        model_json=_s1_model().model_dump_json(),
        output_path="models/gen_S1.fr",
        base_directory=str(TEST_DIR),
    )
    gres = json.loads(gen._run())
    assert gres["status"] == "ok", gres

    ufo = FeynRulesToUFOTool(
        base_directory=str(TEST_DIR),
        feynrules_path=feynrules_path,
        wolframscript_path=wolframscript_path,
        model_path=gres["fr_path"],
        output_dir="data/gen_S1_UFO",
        timeout_sec=900,
    )
    ures = json.loads(ufo._run())
    assert ures.get("ok"), ures
    print("[✓] UFO generation from generated .fr succeeded\n")
    return True


_MODELS_DIR = REPO_ROOT / "tools" / "feynrules" / "test_files" / "models"


def test_fr_parser_s1() -> bool:
    print(">> Testing .fr parser on S1_LQ_RR.fr...\n")
    from tools.frgen.fr_parser import parse_fr_file

    m = parse_fr_file(str(_MODELS_DIR / "S1_LQ_RR.fr"))
    assert m["model_name"] == "S1_LQ_RR", m["model_name"]
    s1 = [c for c in m["classes"] if c["class_name"] == "S1"]
    assert len(s1) == 1, m["classes"]
    assert s1[0]["spin_type"] == "S" and s1[0]["class_index"] == 100, s1[0]
    assert s1[0]["quantum_numbers"].get("Q") == "-1/3", s1[0]["quantum_numbers"]
    # Commented-out MS1 parameter block must NOT be parsed; only yRR11 remains.
    names = [p["name"] for p in m["parameters"]]
    assert names == ["yRR11"], names
    assert m["parameters"][0]["parameter_type"] == "External", m["parameters"][0]
    print("[✓] .fr parser (S1) test passed\n")
    return True


def test_fr_parser_sm() -> bool:
    print(">> Testing .fr parser on SM.fr (full model)...\n")
    from tools.frgen.fr_parser import parse_fr_file

    m = parse_fr_file(str(_MODELS_DIR / "SM.fr"))
    assert m["model_name"] == "Standard Model", m["model_name"]
    assert len(m["classes"]) >= 15, len(m["classes"])
    assert len(m["parameters"]) >= 20, len(m["parameters"])
    gg = {g["name"] for g in m["gauge_groups"]}
    assert {"U1Y", "SU2L", "SU3C"} <= gg, gg
    pnames = {p["name"] for p in m["parameters"]}
    assert {"aEWM1", "Gf", "aS", "CKM"} <= pnames, sorted(pnames)[:10]
    print("[✓] .fr parser (SM) test passed\n")
    return True


def cleanup_test_files() -> None:
    print("\n>> Cleaning up test files...\n")
    if TEST_DIR.exists():
        shutil.rmtree(TEST_DIR)
        print(f"[✓] Removed: {TEST_DIR.name}\n")
    else:
        print("[i] No test files to clean up\n")


TESTS = [
    test_render_roundtrip,
    test_tool_json_interface,
    test_validation_error_surfaced,
    test_number_must_be_string,
    test_topo_sort,
    test_cyclic_dependency_raises,
    test_fr_parser_s1,
    test_fr_parser_sm,
    test_ufo_generation,
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run tests for the frgen toolkit")
    parser.add_argument("--keep-files", action="store_true", help="Keep test-generated files")
    args = parser.parse_args()

    all_passed = True
    for test in TESTS:
        try:
            if not test():
                all_passed = False
        except Exception as e:  # noqa: BLE001
            print(f"[✗] {test.__name__} failed: {e}\n")
            all_passed = False

    if not args.keep_files:
        cleanup_test_files()
    else:
        print("\n[i] Keeping test files (--keep-files set)\n")

    if all_passed:
        print("[✓] All tests passed!\n")
        sys.exit(0)
    print("[✗] Some tests failed!\n")
    sys.exit(1)
