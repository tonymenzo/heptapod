#!/usr/bin/env python3
"""
# test_extract.py is a part of the HEPTAPOD package.
# Copyright (C) 2025 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.

Tests for the Lagrangian extraction tool.

Offline tests cover the deterministic surface (message/prompt building,
parameter validation, sandbox path safety, and graceful handling when no LLM is
available). A gated live test performs a real schema-constrained extraction when
an LLM provider is configured and reachable (skipped otherwise).
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

from tools.extract import extract_tool as ET
from tools.extract.extract_tool import (
    EXTRACTION_SYSTEM_PROMPT,
    ExtractLagrangianTool,
    build_extraction_message,
)

TEST_DIR = TOOL_DIR / "test_files"

_S1_SNIPPET = """\
We introduce a scalar leptoquark S1 transforming as (3, 1, -1/3) under the SM
gauge group. It couples to a right-handed up quark and a right-handed charged
lepton via a Yukawa coupling y_RR. We take the S1 mass to be 1500 GeV and study
pair production pp -> S1 S1~ followed by S1 -> e- u.
"""


def test_build_message() -> bool:
    print(">> Testing extraction message builder...\n")
    msg = build_extraction_message("PAPER BODY", "scalar leptoquark S1")
    assert "FeynRulesModel" in msg
    assert "scalar leptoquark S1" in msg
    assert "PAPER BODY" in msg
    assert "-----BEGIN PAPER-----" in msg
    print("[✓] Message builder test passed\n")
    return True


def test_system_prompt_rules() -> bool:
    print(">> Testing system prompt encodes key rules...\n")
    p = EXTRACTION_SYSTEM_PROMPT
    for needle in ("STRINGS", "-1/3", "VERBATIM", "add-on", "delayed"):
        assert needle in p, needle
    print("[✓] System prompt test passed\n")
    return True


def test_invalid_provider() -> bool:
    print(">> Testing invalid provider rejection...\n")
    tool = ExtractLagrangianTool(
        paper_text="x", llm_provider="bogus", base_directory=str(TEST_DIR)
    )
    result = tool._run()
    assert "error" in result.lower() and "provider" in result.lower(), result
    print("[✓] Invalid provider test passed\n")
    return True


def test_missing_text() -> bool:
    print(">> Testing missing text handling...\n")
    tool = ExtractLagrangianTool(llm_provider="ollama", base_directory=str(TEST_DIR))
    result = tool._run()
    assert "error" in result.lower() or "missing" in result.lower(), result
    print("[✓] Missing text test passed\n")
    return True


def test_text_path_traversal() -> bool:
    print(">> Testing text_path traversal safety...\n")
    tool = ExtractLagrangianTool(
        text_path="../../../../etc/passwd", base_directory=str(TEST_DIR)
    )
    result = tool._run()
    assert "denied" in result.lower() or "error" in result.lower(), result
    print("[✓] Path traversal test passed\n")
    return True


def test_llm_unavailable_graceful() -> bool:
    print(">> Testing graceful handling when LLM is unavailable...\n")
    # Force _get_llm to raise (simulating missing config / unreachable provider).
    # format_error returns a plain (non-JSON) string, so assert on the raw text.
    with mock.patch.object(ET, "_get_llm", side_effect=RuntimeError("no config")):
        tool = ExtractLagrangianTool(paper_text=_S1_SNIPPET, base_directory=str(TEST_DIR))
        result = tool._run()
    assert "error" in result.lower(), result
    assert "unavailable" in result.lower() or "no config" in result.lower(), result
    print("[✓] LLM-unavailable graceful test passed\n")
    return True


def test_extraction_with_mocked_agent() -> bool:
    print(">> Testing extraction success path (mocked Agent)...\n")
    from tools.frgen.frmodel import (
        FeynRulesModel,
        IndexDecl,
        MassSpec,
        ModelInfo,
        ParticleClass,
    )

    fake_model = FeynRulesModel(
        model_name="S1_extracted",
        info=ModelInfo(authors=["auto"], version="1.0", date="2026"),
        index_decls=[IndexDecl(name="Colour", range_kind="NoUnfold", size=3)],
        particles=[
            ParticleClass(
                spin_type="S", class_index=100, class_name="S1",
                self_conjugate=False, indices=["Colour"],
                mass=MassSpec(sym="MS1", value="1500."),
                quantum_numbers={"Q": "-1/3"}, particle_name="S1",
            )
        ],
    )

    class _FakeAgent:
        def __init__(self, *a, **k):
            pass

        def structured(self, message, output_type, **kw):
            assert output_type is FeynRulesModel
            return fake_model

    # Patch the LLM acquisition and the Agent used inside orchestral.
    import orchestral

    with mock.patch.object(ET, "_get_llm", return_value=object()), \
         mock.patch.object(orchestral, "Agent", _FakeAgent):
        tool = ExtractLagrangianTool(
            paper_text=_S1_SNIPPET,
            scenario="scalar leptoquark S1",
            output_path="models/extracted.json",
            base_directory=str(TEST_DIR),
        )
        result = json.loads(tool._run())

    assert result["status"] == "ok", result
    assert result["model_name"] == "S1_extracted", result
    assert result["n_particles"] == 1, result
    assert result["model"]["particles"][0]["quantum_numbers"]["Q"] == "-1/3", result
    assert (TEST_DIR / result["model_json_path"]).exists(), result
    print("[✓] Mocked-Agent extraction test passed\n")
    return True


def test_live_extraction() -> bool:
    print(">> Testing live extraction (needs a configured, reachable LLM)...\n")
    try:
        tool = ExtractLagrangianTool(
            paper_text=_S1_SNIPPET,
            scenario="scalar leptoquark S1, first generation",
            base_directory=str(TEST_DIR),
            max_retries=2,
        )
        raw = tool._run()
    except Exception as e:  # noqa: BLE001
        print(f"[⊘] Skipping live extraction (error: {e})\n")
        return True
    # Errors are plain strings (no LLM configured/reachable) -> skip.
    if not raw.lstrip().startswith("{"):
        print(f"[⊘] Skipping live extraction: {raw.splitlines()[0] if raw else 'no output'}\n")
        return True
    result = json.loads(raw)
    assert result.get("status") == "ok", result
    # A live LLM is stochastic — assert the pipeline returns a SCHEMA-VALID
    # FeynRulesModel (structural), not a specific particle count. Semantic
    # extraction quality is measured (and scored) by the eval harness, not
    # asserted here, so a weaker local model can't make this a flaky failure.
    from tools.frgen.frmodel import FeynRulesModel

    model = result["model"] if isinstance(result["model"], dict) else json.loads(result["model"])
    FeynRulesModel(**model)  # re-validates the returned model; raises if malformed
    print(f"[✓] Live extraction produced schema-valid model '{result['model_name']}' "
          f"with {result['n_particles']} particle(s), {result['n_parameters']} parameter(s)\n")
    return True


def cleanup_test_files() -> None:
    print("\n>> Cleaning up test files...\n")
    if TEST_DIR.exists():
        shutil.rmtree(TEST_DIR)
        print(f"[✓] Removed: {TEST_DIR.name}\n")
    else:
        print("[i] No test files to clean up\n")


TESTS = [
    test_build_message,
    test_system_prompt_rules,
    test_invalid_provider,
    test_missing_text,
    test_text_path_traversal,
    test_llm_unavailable_graceful,
    test_extraction_with_mocked_agent,
    test_live_extraction,
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run tests for the extract toolkit")
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
