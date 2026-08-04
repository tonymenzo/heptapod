#!/usr/bin/env python3
"""
# test_validate.py is a part of the HEPTAPOD package.
# Copyright (C) 2025 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.

Tests for the model validation tool.

Offline tests cover the deterministic checks (UFO file presence, particle
presence, expected-name extraction) and the tool's control flow with a mocked
FeynRulesToUFOTool (success and failure). A gated live test compiles a generated
.fr to a UFO and validates it when FeynRules + wolframscript are configured.
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from unittest import mock

SCRIPT_PATH = Path(__file__).resolve()
TOOL_DIR = SCRIPT_PATH.parent
REPO_ROOT = TOOL_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import tools.validate.validate_tool as VT
from tools.validate.validate_tool import (
    ValidateModelTool,
    check_particles_present,
    check_ufo_files,
    _expected_particle_names,
)

TEST_DIR = TOOL_DIR / "test_files"


def _make_fake_ufo(ufo_dir: Path, particle_names=("S1",)) -> None:
    ufo_dir.mkdir(parents=True, exist_ok=True)
    for fname in ("__init__.py", "particles.py", "parameters.py", "vertices.py", "couplings.py"):
        (ufo_dir / fname).write_text("# ufo\n")
    body = "\n".join(f"{n} = Particle(pdg_code=100, name='{n}')" for n in particle_names)
    (ufo_dir / "particles.py").write_text(body + "\n")


def _write_dummy_fr(rel="models/x.fr") -> str:
    p = TEST_DIR / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text('M$ModelName = "X";\n')
    return rel


def test_check_ufo_files() -> bool:
    print(">> Testing UFO file presence checks...\n")
    ufo = TEST_DIR / "ufo_a"
    ufo.mkdir(parents=True, exist_ok=True)
    (ufo / "particles.py").write_text("x")  # only one present
    checks = check_ufo_files(str(ufo))
    by = {c["name"]: c["passed"] for c in checks}
    assert by["ufo_file:particles.py"] is True, by
    assert by["ufo_file:vertices.py"] is False, by
    print("[✓] UFO file-check test passed\n")
    return True


def test_check_particles_present() -> bool:
    print(">> Testing particle-presence checks...\n")
    ufo = TEST_DIR / "ufo_b"
    _make_fake_ufo(ufo, particle_names=("S1",))
    checks = check_particles_present(str(ufo), ["S1", "Zprime"])
    by = {c["name"]: c["passed"] for c in checks}
    assert by["particle_in_ufo:S1"] is True, by
    assert by["particle_in_ufo:Zprime"] is False, by
    print("[✓] Particle-presence test passed\n")
    return True


def test_expected_particle_names() -> bool:
    print(">> Testing expected-name extraction (skips unphysical)...\n")
    model = {
        "particles": [
            {"class_name": "S1", "particle_name": "S1", "unphysical": False},
            {"class_name": "Phi", "unphysical": True, "definitions": ["Phi[1] -> 0"]},
        ]
    }
    names = _expected_particle_names(json.dumps(model))
    assert names == ["S1"], names
    assert _expected_particle_names(None) == []
    print("[✓] Expected-name extraction test passed\n")
    return True


def test_validate_missing_model() -> bool:
    print(">> Testing missing model handling...\n")
    tool = ValidateModelTool(
        model_path="models/nope.fr",
        base_directory=str(TEST_DIR),
        feynrules_path="/fr",
        wolframscript_path="wolframscript",
    )
    result = tool._run()
    assert "not found" in result.lower() or "error" in result.lower(), result
    print("[✓] Missing-model test passed\n")
    return True


def test_validate_success_mocked() -> bool:
    print(">> Testing validation success path (mocked UFO tool)...\n")
    rel = _write_dummy_fr()
    ufo_dir = TEST_DIR / "UFO_ok"
    _make_fake_ufo(ufo_dir, particle_names=("S1",))
    model_json = json.dumps({"particles": [{"class_name": "S1", "unphysical": False}]})

    inst = mock.Mock()
    inst._run.return_value = json.dumps({"ok": True, "output_dir": str(ufo_dir)})
    with mock.patch.object(VT, "FeynRulesToUFOTool", return_value=inst):
        tool = ValidateModelTool(
            model_path=rel,
            feynrules_model_json=model_json,
            base_directory=str(TEST_DIR),
            feynrules_path="/fr",
            wolframscript_path="wolframscript",
        )
        result = json.loads(tool._run())

    assert result["passed"] is True, result
    names = {c["name"]: c["passed"] for c in result["checks"]}
    assert names["ufo_generation"] is True, result
    assert names["particle_in_ufo:S1"] is True, result
    print("[✓] Validation success (mocked) test passed\n")
    return True


def test_validate_failure_mocked() -> bool:
    print(">> Testing validation failure path (mocked UFO tool error)...\n")
    rel = _write_dummy_fr()
    inst = mock.Mock()
    # FeynRulesToUFOTool returns a plain-string format_error on failure.
    inst._run.return_value = "Error: UFO Generation Failed\n- Reason: LoadModel::NoClasses"
    with mock.patch.object(VT, "FeynRulesToUFOTool", return_value=inst):
        tool = ValidateModelTool(
            model_path=rel,
            base_directory=str(TEST_DIR),
            feynrules_path="/fr",
            wolframscript_path="wolframscript",
        )
        result = json.loads(tool._run())

    assert result["passed"] is False, result
    gen = [c for c in result["checks"] if c["name"] == "ufo_generation"][0]
    assert gen["passed"] is False, result
    assert "feynrules_log" in result, result
    print("[✓] Validation failure (mocked) test passed\n")
    return True


def test_validate_live() -> bool:
    print(">> Testing live .fr -> UFO validation (needs FeynRules + wolframscript)...\n")
    try:
        from config import feynrules_path, wolframscript_path
    except Exception:  # noqa: BLE001
        print("[⊘] Skipping: config.py not set up\n")
        return True
    if not shutil.which(wolframscript_path) or feynrules_path in (None, "/path/to/FeynRules"):
        print("[⊘] Skipping: wolframscript/FeynRules not configured\n")
        return True
    from tools.feynrules.wl_probe import wolframscript_activated

    if not wolframscript_activated(wolframscript_path):
        print("[⊘] Skipping: wolframscript not activated (no Wolfram license)\n")
        return True

    from tools.frgen.frgen_tool import GenerateFeynRulesModelTool
    from tools.frgen.test_frgen import _s1_model

    s1 = _s1_model()
    gen = GenerateFeynRulesModelTool(
        model_json=s1.model_dump_json(),
        output_path="models/val_S1.fr",
        base_directory=str(TEST_DIR),
    )
    gres = json.loads(gen._run())
    assert gres["status"] == "ok", gres

    tool = ValidateModelTool(
        model_path=gres["fr_path"],
        feynrules_model_json=s1.model_dump_json(),
        output_dir="UFO_val_S1",
        base_directory=str(TEST_DIR),
        feynrules_path=feynrules_path,
        wolframscript_path=wolframscript_path,
        timeout_sec=900,
    )
    vres = json.loads(tool._run())
    assert vres["passed"], vres
    print("[✓] Live validation passed\n")
    return True


_UFO_FIXTURE = REPO_ROOT / "tools" / "feynrules" / "test_files" / "models" / "S1_LQ_RR_UFO"
_CHECKS_LOG = REPO_ROOT / "tools" / "feynrules" / "test_files" / "logs" / "checks_stdout_S1.log"

_S1_MODEL = {
    "particles": [
        {
            "spin_type": "S",
            "class_index": 100,
            "class_name": "S1",
            "particle_name": "S1",
            "self_conjugate": False,
            "indices": ["Colour"],
            "pdg": 9000005,
            "quantum_numbers": {"Q": "-1/3"},
        }
    ]
}


def test_ufo_parser_particles() -> bool:
    print(">> Testing UFO particles.py AST parsing...\n")
    from fractions import Fraction

    from tools.validate.ufo_parser import parse_particles

    parts = parse_particles(str(_UFO_FIXTURE / "particles.py"))
    by_name = {p.get("name"): p for p in parts}
    assert len(parts) >= 20, len(parts)  # full SM + S1
    s1 = by_name["S1"]
    assert s1["pdg_code"] == 9000005, s1
    assert s1["spin"] == 1 and s1["color"] == 3, s1
    assert s1["charge"] == Fraction(-1, 3), s1["charge"]
    # photon: spin 3 (2s+1), charge 0.
    assert by_name["a"]["charge"] == Fraction(0), by_name["a"]
    print("[✓] UFO particles parse test passed\n")
    return True


def test_check_particle_properties() -> bool:
    print(">> Testing particle-property checks (spin/color/charge)...\n")
    from tools.validate.ufo_parser import check_particle_properties

    good = {c["name"]: c["passed"] for c in check_particle_properties(str(_UFO_FIXTURE), _S1_MODEL)}
    assert good.get("particle_props:S1") is True, good

    bad_model = json.loads(json.dumps(_S1_MODEL))
    bad_model["particles"][0]["quantum_numbers"]["Q"] = "2/3"  # wrong charge
    bad = {c["name"]: (c["passed"], c["detail"]) for c in check_particle_properties(str(_UFO_FIXTURE), bad_model)}
    assert bad["particle_props:S1"][0] is False and "charge" in bad["particle_props:S1"][1], bad
    print("[✓] Particle-property check test passed\n")
    return True


def test_wl_checks_parser() -> bool:
    print(">> Testing FeynRules check-log parsing...\n")
    from tools.feynrules.wl_checks import parse_check_blocks

    log = _CHECKS_LOG.read_text()
    checks = {c["name"]: c["passed"] for c in parse_check_blocks(log)}
    assert checks == {"hermiticity": True, "kinetic_terms": True, "mass_spectrum": True}, checks

    fail = parse_check_blocks(
        "HEPTAPOD-CHECK-BEGIN: hermiticity\nThe Lagrangian is not hermitian.\n"
        "HEPTAPOD-CHECK-END: hermiticity"
    )
    assert fail[0]["passed"] is False, fail
    err = parse_check_blocks(
        "HEPTAPOD-CHECK-BEGIN: mass_spectrum\nHEPTAPOD-CHECK-ERROR\nHEPTAPOD-CHECK-END: mass_spectrum"
    )
    assert err[0]["passed"] is False, err
    assert parse_check_blocks("no sentinels here") == []
    print("[✓] Check-log parser test passed\n")
    return True


def test_width_gate_s1() -> bool:
    print(">> Testing analytic width gate on the S1 fixture...\n")
    from tools.validate.width_gate import (
        analytic_scalar_lq_width,
        compare_width,
        find_partial_width,
        parse_decays,
        parse_external_params,
        safe_eval_width,
    )

    params = parse_external_params(str(_UFO_FIXTURE))
    assert abs(params["MS1"] - 1500.0) < 1e-9 and abs(params["yRR11"] - 0.5) < 1e-9, params

    decays = parse_decays(str(_UFO_FIXTURE))
    expr = find_partial_width(decays, "S1", ("e__minus__", "u"))
    assert expr is not None, "S1 -> e u partial width not found"

    ufo_width = safe_eval_width(expr, params)
    analytic = analytic_scalar_lq_width(params["MS1"], params["yRR11"])
    cmp = compare_width(analytic, ufo_width, rel_tol=0.02)
    assert cmp["passed"], cmp
    assert 7.0 < analytic < 8.0, analytic  # |y|^2 m/16pi = 0.25*1500/16pi ~ 7.46

    # Safe-eval rejects anything outside the whitelist.
    for bad in ("__import__('os').system('x')", "MS1.__class__", "open('x')"):
        try:
            safe_eval_width(bad, params)
            raise AssertionError(f"should have rejected: {bad}")
        except ValueError:
            pass
    print(f"[✓] Width gate passed (analytic {analytic:.4f} vs UFO {ufo_width:.4f} GeV)\n")
    return True


def test_validate_width_gate_wired() -> bool:
    print(">> Testing ValidateModelTool width-gate wiring (S1 fixture, mocked UFO)...\n")
    rel = _write_dummy_fr()
    gate = json.dumps(
        {
            "particle": "S1",
            "finals": ["e__minus__", "u"],
            "formula": "scalar_lq",
            "mass_param": "MS1",
            "coupling_param": "yRR11",
            "rel_tol": 0.02,
        }
    )
    inst = mock.Mock()
    inst._run.return_value = json.dumps({"ok": True, "output_dir": str(_UFO_FIXTURE)})
    with mock.patch.object(VT, "FeynRulesToUFOTool", return_value=inst):
        tool = ValidateModelTool(
            model_path=rel,
            feynrules_model_json=json.dumps(_S1_MODEL),
            width_gate=gate,
            base_directory=str(TEST_DIR),
            feynrules_path="/fr",
            wolframscript_path="wolframscript",
        )
        result = json.loads(tool._run())

    gate_checks = [c for c in result["checks"] if c["name"].startswith("width_gate")]
    assert gate_checks, result
    assert gate_checks[0]["passed"] is True, gate_checks
    assert "GeV" in gate_checks[0]["detail"], gate_checks

    # A channel that isn't in the UFO must fail (not error out the whole run).
    bad_gate = json.loads(gate)
    bad_gate["finals"] = ["mu__minus__", "c"]
    with mock.patch.object(VT, "FeynRulesToUFOTool", return_value=inst):
        tool = ValidateModelTool(
            model_path=rel,
            feynrules_model_json=json.dumps(_S1_MODEL),
            width_gate=json.dumps(bad_gate),
            base_directory=str(TEST_DIR),
            feynrules_path="/fr",
            wolframscript_path="wolframscript",
        )
        bad_result = json.loads(tool._run())
    bad_check = [c for c in bad_result["checks"] if c["name"].startswith("width_gate")][0]
    assert bad_check["passed"] is False, bad_check
    assert bad_result["passed"] is False, bad_result
    print("[✓] Width-gate wiring test passed\n")
    return True


def test_validate_physics_checks_merged_mocked() -> bool:
    print(">> Testing ValidateModelTool merges wl:* + particle_props (mocked UFO)...\n")
    rel = _write_dummy_fr()
    inst = mock.Mock()
    inst._run.return_value = json.dumps(
        {
            "ok": True,
            "output_dir": str(_UFO_FIXTURE),
            "checks": [
                {"name": "hermiticity", "passed": True, "detail": "hermitian"},
                {"name": "mass_spectrum", "passed": True, "detail": "ok"},
            ],
        }
    )
    with mock.patch.object(VT, "FeynRulesToUFOTool", return_value=inst):
        tool = ValidateModelTool(
            model_path=rel,
            feynrules_model_json=json.dumps(_S1_MODEL),
            base_directory=str(TEST_DIR),
            feynrules_path="/fr",
            wolframscript_path="wolframscript",
        )
        result = json.loads(tool._run())
    names = {c["name"]: c["passed"] for c in result["checks"]}
    assert names.get("wl:hermiticity") is True, names
    assert names.get("particle_props:S1") is True, names
    assert result["passed"] is True, result
    print("[✓] Physics-checks merge test passed\n")
    return True


def cleanup_test_files() -> None:
    print("\n>> Cleaning up test files...\n")
    if TEST_DIR.exists():
        shutil.rmtree(TEST_DIR)
        print(f"[✓] Removed: {TEST_DIR.name}\n")
    else:
        print("[i] No test files to clean up\n")


TESTS = [
    test_check_ufo_files,
    test_check_particles_present,
    test_expected_particle_names,
    test_validate_missing_model,
    test_validate_success_mocked,
    test_validate_failure_mocked,
    test_ufo_parser_particles,
    test_check_particle_properties,
    test_wl_checks_parser,
    test_width_gate_s1,
    test_validate_width_gate_wired,
    test_validate_physics_checks_merged_mocked,
    test_validate_live,
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run tests for the validate toolkit")
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
