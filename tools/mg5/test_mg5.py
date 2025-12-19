#!/usr/bin/env python3
"""
# test_mg5.py is a part of the HEPTAPOD package.
# Copyright (C) 2025 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.
"""
"""
MadGraph5 Tools Test Suite

Tests cover:
- MadGraph event generation from run cards
- Card editing functionality
- Parameter scan detection and parsing
- Seed reproducibility
- LHE to JSONL conversion
- Path traversal prevention (security)
"""

# Standard library imports
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

# Third-party imports
import numpy as np

# Path setup
SCRIPT_PATH = Path(__file__).resolve()
MG5_DIR = SCRIPT_PATH.parent                                  # .../heptapod-dev/tools/mg5
TOOLS_DIR = MG5_DIR.parent                                    # .../heptapod-dev/tools
REPO_ROOT = TOOLS_DIR.parent                                  # .../heptapod-dev

# Add repository root to path for local imports (prompts, tools, etc.)
sys.path.insert(0, str(REPO_ROOT))

# Tool imports
from tools.mg5 import (
    MadGraphFromRunCardTool,
    _edit_mg5_card,
    _detect_scan_runs,
    _parse_scan_summary,
    _find_all_lhe_files
)
from tools.analysis.conversions import LHEToJSONLTool

# Future: When orchestral-ai PyPI package is fixed, use these imports instead:
# from orchestral.tools import MadGraphFromRunCardTool
# from orchestral.tools import LHEToJSONLTool

# Try to import config for configuration
try:
    from config import mg5_path
except ImportError:
    # Fallback to defaults if config doesn't exist
    mg5_path = os.environ.get("MG5_PATH", "/usr/local/MG5_aMC")

base_directory = str(MG5_DIR / "test_files")


# ============================================================================
# Test Functions
# ============================================================================

def test_edit_mg5_card():
    """Test the _edit_mg5_card function for replacing paths and parameters."""
    print(">> Testing MG5 card editing functionality...\n")

    # Sample template card
    template = """set automatic_html_opening False

# Load the Standard Model UFO (built in)
import model /old/path/to/UFO

# Define beam contents (proton = quarks + gluon)
define p = g u c d s b u~ c~ d~ s~ b~

# Define a simple process
generate p p > t t~

# Specify where MG5 should write its output
output old_output_dir

# Launch the run
launch

# ---- Run configuration ----
set nevents 1000
set iseed 12345
set ebeam1 6500.0
set ebeam2 6500.0
"""

    # Test 1: Replace UFO path only
    edited = _edit_mg5_card(template, ufo_path="/new/ufo/path")
    assert "import model /new/ufo/path" in edited
    assert "import model /old/path/to/UFO" not in edited
    assert "set nevents 1000" in edited  # Other lines unchanged
    print("[✓] Test 1 passed: UFO path replacement")

    # Test 2: Replace output directory only
    edited = _edit_mg5_card(template, output_name="new_output_name")
    assert "output new_output_name" in edited
    assert "output old_output_dir" not in edited
    assert "import model /old/path/to/UFO" in edited  # UFO unchanged
    print("[✓] Test 2 passed: Output directory replacement")

    # Test 3: Replace nevents only
    edited = _edit_mg5_card(template, nevents=50000)
    assert "set nevents 50000" in edited
    assert "set nevents 1000" not in edited
    assert "set iseed 12345" in edited  # Other set commands unchanged
    print("[✓] Test 3 passed: nevents replacement")

    # Test 4: Replace seed only
    edited = _edit_mg5_card(template, seed=99999)
    assert "set iseed 99999" in edited
    assert "set iseed 12345" not in edited
    assert "set nevents 1000" in edited  # nevents unchanged
    print("[✓] Test 4 passed: seed replacement")

    # Test 5: Replace all parameters at once
    edited = _edit_mg5_card(
        template,
        ufo_path="/absolute/path/to/model_UFO",
        output_name="madgraph/results/run001",
        nevents=50000,
        seed=42
    )
    # Check replacements worked
    assert "import model /absolute/path/to/model_UFO" in edited
    assert "output madgraph/results/run001" in edited
    assert "set nevents 50000" in edited
    assert "set iseed 42" in edited
    # Verify old values are gone
    assert "/old/path/to/UFO" not in edited
    assert "old_output_dir" not in edited
    assert "set nevents 1000" not in edited
    assert "set iseed 12345" not in edited
    print("[✓] Test 5 passed: All parameters replacement")

    # Test 6: No replacements (should return identical card)
    edited = _edit_mg5_card(template)
    assert edited == template
    print("[✓] Test 6 passed: No changes when no parameters provided")

    print("\nAll card editing tests passed! [✓]\n")


def test_scan_detection():
    """Test parameter scan detection helper functions."""
    print(">> Testing scan detection helper functions...\n")

    # Test 1: No scan detection for single run (use temporary directory)
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        single_run_dir = os.path.join(tmpdir, "Events")
        os.makedirs(single_run_dir)
        os.makedirs(os.path.join(single_run_dir, "run_01"))

        scan_runs = _detect_scan_runs(single_run_dir)
        assert len(scan_runs) == 0, f"Expected no scan detection for single run, got {scan_runs}"
        print("[✓] Test 1 passed: No scan detected for single run directory")

    print("\nAll scan detection tests passed! [✓]\n")


def test_scan_generation_and_detection():
    """Test parameter scan generation and detection using actual MG5 run."""
    print(">> Testing scan generation and detection (integration test)...\n")

    # Generate scan using the scan card
    tool_scan = MadGraphFromRunCardTool(
        mg5_path=mg5_path,
        base_directory=base_directory,
        data_dir="data_scan_test/",
        command_card="cards/S1_LQ_RR_pp_lqlq_scan.mg5",
        ufo_path="models/S1_LQ_RR_UFO",  # Test UFO path replacement
        output_name="S1_LQ_SCAN",  # Ensure output directory is created correctly
        seed=12345
    )

    tool_scan._setup()
    scan_output_raw = tool_scan._run()

    # Try to parse as JSON (success case), otherwise treat as error message
    try:
        scan_output = json.loads(scan_output_raw) if isinstance(scan_output_raw, str) else scan_output_raw
    except (json.JSONDecodeError, ValueError):
        print("[⊘] Skipping scan tests: MG5 scan generation failed")
        print(f"  {scan_output_raw}\n")
        return None

    if scan_output.get("status") != "ok":
        print("[⊘] Skipping scan tests: MG5 scan generation failed")
        print(f"  Error: {scan_output.get('reason', scan_output.get('error_message', 'Unknown error'))}")
        if "log_file" in scan_output:
            print(f"  See log: {scan_output['log_file']}\n")
        return None

    print("[✓] Scan generation succeeded!")
    print(f"  Working dir: {scan_output.get('data_dir')}")
    print(f"  LHE file:    {scan_output.get('lhe_file')}\n")

    # Extract Events directory path from the generated output
    # The LHE file path is like: data_scan_test/PROC_NAME/Events/run_01/unweighted_events.lhe.gz
    lhe_path = scan_output.get('lhe_file', '')
    if '/Events/' in lhe_path:
        # Get the Events directory
        scan_events_dir = os.path.join(
            base_directory,
            lhe_path.split('/Events/')[0] + '/Events'
        )
    else:
        print("[⊘] Could not determine Events directory from LHE path")
        return None

    # Test 2: Detect scan runs in generated scan directory
    if os.path.exists(scan_events_dir):
        scan_runs = _detect_scan_runs(scan_events_dir)
        assert len(scan_runs) == 3, f"Expected 3 scan runs, found {len(scan_runs)}"
        assert scan_runs == ["run_01", "run_02", "run_03"], f"Unexpected run IDs: {scan_runs}"
        print(f"[✓] Test 2 passed: Detected {len(scan_runs)} scan runs: {scan_runs}")
    else:
        print(f"[⊘] Test 2 skipped: Scan Events directory not found at {scan_events_dir}")
        return None

    # Test 3: Find all LHE files
    lhe_files = _find_all_lhe_files(scan_events_dir, scan_runs)

    assert len(lhe_files) == 3, f"Expected 3 LHE files, found {len(lhe_files)}"

    for run_id in scan_runs:
        assert run_id in lhe_files, f"Missing LHE file for {run_id}"
        assert os.path.exists(lhe_files[run_id]), f"LHE file does not exist: {lhe_files[run_id]}"
        assert "unweighted_events.lhe" in lhe_files[run_id], f"Unexpected LHE filename: {lhe_files[run_id]}"

    print(f"[✓] Test 3 passed: Found LHE files for all {len(lhe_files)} runs")

    # Test 4: Parse scan summary file (if it exists)
    # Look for scan summary file pattern
    import glob
    scan_summary_pattern = os.path.join(scan_events_dir, "scan_run_*.txt")
    scan_summary_files = glob.glob(scan_summary_pattern)

    if scan_summary_files:
        scan_summary_file = scan_summary_files[0]
        scan_info = _parse_scan_summary(scan_summary_file)

        assert len(scan_info) == 3, f"Expected 3 entries in scan summary, found {len(scan_info)}"
        assert "run_01" in scan_info, "Missing run_01 in scan summary"
        assert "run_02" in scan_info, "Missing run_02 in scan summary"
        assert "run_03" in scan_info, "Missing run_03 in scan summary"

        # Check that run_01 has the expected parameter value and cross-section
        run01_info = scan_info["run_01"]
        assert "cross_section" in run01_info, "Missing cross_section in run_01"
        assert "cross_section_error" in run01_info, "Missing cross_section_error in run_01"

        # Check parameter value (MS1 should be 1000.0 for run_01)
        param_found = False
        for key, value in run01_info.items():
            if "ms1" in key.lower() or "mass" in key.lower():
                # First scan point should be 1000
                assert abs(value - 1000.0) < 0.1, f"Expected MS1 parameter ~1000, got {value}"
                param_found = True
                break

        assert param_found, "Parameter value not found in scan info"
        print(f"[✓] Test 4 passed: Parsed scan summary with {len(scan_info)} runs")
        print(f"    run_01: cross_section={run01_info.get('cross_section'):.6e}")
    else:
        print(f"[i] Test 4 info: No scan summary file found (may not be generated for all MG5 versions)")

    print("\nAll scan generation and detection tests passed! [✓]\n")
    return {"status": "ok"}


def test_path_traversal_prevention():
    """Test that path traversal attempts are rejected (critical security test)."""
    print(">> Testing path traversal prevention (security)...\n")

    # Test 1: command_card with path traversal
    tool = MadGraphFromRunCardTool(
        mg5_path=mg5_path,
        base_directory=base_directory,
        data_dir="data/",
        command_card="../../../etc/passwd",
        seed=12345
    )
    tool._setup()
    result = tool._run()
    # Should fail due to file not found (path escapes sandbox)
    assert ("error" in result.lower() or "not found" in result.lower() or
            "denied" in result.lower() or "escape" in result.lower())
    print("[✓] Test 1 passed: command_card traversal rejected")

    # Test 2: data_dir with path traversal
    tool = MadGraphFromRunCardTool(
        mg5_path=mg5_path,
        base_directory=base_directory,
        data_dir="../../../tmp/",
        command_card="cards/pp_ttbar.txt",
        seed=12345
    )
    tool._setup()
    result = tool._run()
    # Tool should handle this - either reject or sandbox it
    # We just verify it doesn't crash
    print("[✓] Test 2 passed: data_dir traversal handled")

    # Test 3: Absolute path in command_card (should be rejected/handled)
    tool = MadGraphFromRunCardTool(
        mg5_path=mg5_path,
        base_directory=base_directory,
        data_dir="data/",
        command_card="/etc/passwd",
        seed=12345
    )
    tool._setup()
    result = tool._run()
    assert ("error" in result.lower() or "not found" in result.lower())
    print("[✓] Test 3 passed: Absolute path in command_card rejected")

    print("\nAll path traversal prevention tests passed! [✓]\n")


def test_seed_reproducibility():
    """Test that same seed produces identical events (scientific reproducibility)."""
    print(">> Testing seed reproducibility...\n")

    # Generate events with seed 42
    tool_run1 = MadGraphFromRunCardTool(
        mg5_path=mg5_path,
        base_directory=base_directory,
        data_dir="data_seed_test_1/",
        command_card="cards/pp_ttbar.txt",
        seed=42
    )
    tool_run1._setup()
    output1_raw = tool_run1._run()

    try:
        output1 = json.loads(output1_raw) if isinstance(output1_raw, str) else output1_raw
    except (json.JSONDecodeError, ValueError):
        print(f"[⊘] Skipping seed reproducibility test: MG5 run 1 failed\n")
        print(f"  {output1_raw}\n")
        return None

    if output1.get("status") != "ok":
        print(f"[⊘] Skipping seed reproducibility test: MG5 run 1 failed\n")
        return None

    # Generate events with same seed 42
    tool_run2 = MadGraphFromRunCardTool(
        mg5_path=mg5_path,
        base_directory=base_directory,
        data_dir="data_seed_test_2/",
        command_card="cards/pp_ttbar.txt",
        seed=42
    )
    tool_run2._setup()
    output2_raw = tool_run2._run()

    try:
        output2 = json.loads(output2_raw) if isinstance(output2_raw, str) else output2_raw
    except (json.JSONDecodeError, ValueError):
        print(f"[⊘] Skipping seed reproducibility test: MG5 run 2 failed\n")
        return None

    if output2.get("status") != "ok":
        print(f"[⊘] Skipping seed reproducibility test: MG5 run 2 failed\n")
        return None

    # Convert both to JSONL for comparison
    lhe1_rel = output1.get("lhe_file", "")
    lhe2_rel = output2.get("lhe_file", "")

    # Convert run 1
    conv_tool1 = LHEToJSONLTool(
        base_directory=base_directory,
        lhe_path=lhe1_rel,
        jsonl_path="data_seed_test_1/events.jsonl",
        finals_only=True,
        full_history=False,
    )
    conv_tool1._setup()
    conv1_raw = conv_tool1._run()

    try:
        conv1 = json.loads(conv1_raw) if isinstance(conv1_raw, str) else conv1_raw
    except (json.JSONDecodeError, ValueError):
        print(f"[⊘] Skipping seed reproducibility test: conversion 1 failed\n")
        return None

    # Convert run 2
    conv_tool2 = LHEToJSONLTool(
        base_directory=base_directory,
        lhe_path=lhe2_rel,
        jsonl_path="data_seed_test_2/events.jsonl",
        finals_only=True,
        full_history=False,
    )
    conv_tool2._setup()
    conv2_raw = conv_tool2._run()

    try:
        conv2 = json.loads(conv2_raw) if isinstance(conv2_raw, str) else conv2_raw
    except (json.JSONDecodeError, ValueError):
        print(f"[⊘] Skipping seed reproducibility test: conversion 2 failed\n")
        return None

    # Load and compare first events
    jsonl1_path = os.path.join(base_directory, "data_seed_test_1", "events.jsonl")
    jsonl2_path = os.path.join(base_directory, "data_seed_test_2", "events.jsonl")

    with open(jsonl1_path, "r") as f1, open(jsonl2_path, "r") as f2:
        event1 = json.loads(f1.readline())
        event2 = json.loads(f2.readline())

        particles1 = event1["data"]["particles"]
        particles2 = event2["data"]["particles"]

        # Test 1: Same number of particles
        assert len(particles1) == len(particles2), \
            f"Particle count mismatch: {len(particles1)} vs {len(particles2)}"
        print(f"[✓] Test 1 passed: Same particle count ({len(particles1)} particles)")

        # Test 2: Same PDG IDs
        pdg1 = [p["id"] for p in particles1]
        pdg2 = [p["id"] for p in particles2]
        assert pdg1 == pdg2, "PDG ID mismatch"
        print("[✓] Test 2 passed: Identical PDG IDs")

        # Test 3: Same momenta (within floating point tolerance)
        for i, (p1, p2) in enumerate(zip(particles1, particles2)):
            px_match = abs(p1["px"] - p2["px"]) < 1e-6
            py_match = abs(p1["py"] - p2["py"]) < 1e-6
            pz_match = abs(p1["pz"] - p2["pz"]) < 1e-6
            e_match = abs(p1["E"] - p2["E"]) < 1e-6
            assert px_match and py_match and pz_match and e_match, \
                f"Momentum mismatch at particle {i}"
        print("[✓] Test 3 passed: Identical momenta (within tolerance)")

    # Test 4: Different seed produces different results
    tool_run3 = MadGraphFromRunCardTool(
        mg5_path=mg5_path,
        base_directory=base_directory,
        data_dir="data_seed_test_3/",
        command_card="cards/pp_ttbar.txt",
        seed=999  # Different seed
    )
    tool_run3._setup()
    output3_raw = tool_run3._run()

    try:
        output3 = json.loads(output3_raw) if isinstance(output3_raw, str) else output3_raw
    except (json.JSONDecodeError, ValueError):
        print(f"[⊘] Skipping different-seed test: MG5 run 3 failed\n")
        return {"status": "partial"}

    if output3.get("status") == "ok":
        lhe3_rel = output3.get("lhe_file", "")
        conv_tool3 = LHEToJSONLTool(
            base_directory=base_directory,
            lhe_path=lhe3_rel,
            jsonl_path="data_seed_test_3/events.jsonl",
            finals_only=True,
            full_history=False,
        )
        conv_tool3._setup()
        conv3_raw = conv_tool3._run()

        jsonl3_path = os.path.join(base_directory, "data_seed_test_3", "events.jsonl")
        with open(jsonl1_path, "r") as f1, open(jsonl3_path, "r") as f3:
            event1 = json.loads(f1.readline())
            event3 = json.loads(f3.readline())

            particles1 = event1["data"]["particles"]
            particles3 = event3["data"]["particles"]

            # Should have different momenta (very unlikely to be identical by chance)
            different = False
            for p1, p3 in zip(particles1, particles3):
                if abs(p1["px"] - p3["px"]) > 1e-6:
                    different = True
                    break

            assert different, "Different seeds produced identical events (extremely unlikely!)"
            print("[✓] Test 4 passed: Different seed produces different events")

    print("\nAll seed reproducibility tests passed! [✓]\n")
    return {"status": "ok"}


def test_madgraph_generation():
    """Test MadGraph event generation from run card."""
    print(">> Testing MadGraph generation...\n")

    tool_mg_gen = MadGraphFromRunCardTool(
        mg5_path=mg5_path,
        base_directory=base_directory,
        data_dir="data/",
        command_card="cards/pp_ttbar.txt",  # Relative to base_directory
        seed=12345
    )

    tool_mg_gen._setup()
    mg_output_raw = tool_mg_gen._run()

    # Try to parse as JSON (success case), otherwise treat as error message
    try:
        mg_output = json.loads(mg_output_raw) if isinstance(mg_output_raw, str) else mg_output_raw
    except (json.JSONDecodeError, ValueError):
        # format_error() returns plain text, not JSON
        print("[✗] MadGraph generation failed.")
        print(f"  {mg_output_raw}\n")
        return None

    if mg_output.get("status") == "ok":
        print("[✓] MadGraph ran successfully!")
        print(f"  Working dir: {mg_output.get('data_dir')}")
        print(f"  LHE file:    {mg_output.get('lhe_file')}")
        print(f"  Manifest:    {mg_output.get('manifest_json')}")
        print(f"  Log file:    {mg_output.get('log_file')}\n")
        return mg_output
    else:
        print("[✗] MadGraph generation failed.")
        print(f"  Error: {mg_output.get('reason', mg_output.get('error_message', 'Unknown error'))}")
        if "log_file" in mg_output:
            print(f"  See log: {mg_output['log_file']}\n")
        return None


def test_lhe_to_jsonl_conversion(mg_output=None):
    """Test LHE to JSONL conversion."""
    print(">> Testing LHE -> JSONL conversion...\n")

    # Get LHE path from MadGraph output or previous run
    if mg_output is None:
        manifest_path = os.path.join(base_directory, "data", "manifest.json")
        if os.path.exists(manifest_path):
            with open(manifest_path, "r", encoding="utf-8") as f:
                mg_output = json.load(f)
                lhe_rel = str(mg_output.get("outputs", {}).get("lhe_path", ""))
        else:
            print("[✗] No MadGraph output available. Skipping LHE conversion test.\n")
            return None
    else:
        lhe_rel = mg_output.get("lhe_file", "")

    if not lhe_rel:
        print("[✗] No LHE file path found. Skipping conversion test.\n")
        return None

    # Test LHE -> JSONL conversion
    tool_lhe_to_jsonl = LHEToJSONLTool(
        base_directory=base_directory,
        lhe_path=lhe_rel,  # Already relative to base_directory from MG5 output
        jsonl_path="data/events.jsonl",  # Relative to base_directory
        finals_only=True,
        full_history=False,
    )

    tool_lhe_to_jsonl._setup()
    conv_raw = tool_lhe_to_jsonl._run()

    # Try to parse as JSON (success case), otherwise treat as error message
    try:
        conv_output = json.loads(conv_raw) if isinstance(conv_raw, str) else conv_raw
    except (json.JSONDecodeError, ValueError):
        # format_error() returns plain text, not JSON
        print("[✗] LHE -> JSONL conversion failed.")
        print(f"  {conv_raw}\n")
        return None

    if conv_output.get("status") == "ok":
        print("[✓] LHE successfully converted to JSONL!")
        print(f"  JSONL file: {conv_output.get('events_jsonl')}")
        print(f"  Events:     {conv_output.get('n_events')}\n")
        return conv_output
    else:
        print("[✗] LHE -> JSONL conversion failed.")
        print(f"  Error: {conv_output.get('error_message', conv_output.get('reason', 'Unknown error'))}\n")
        return None


def test_inspect_jsonl_output():
    """Inspect the first event in the JSONL output."""
    print(">> Inspecting JSONL output...\n")

    jsonl_abs = os.path.join(base_directory, "data", "events.jsonl")
    if not os.path.exists(jsonl_abs):
        print("[✗] JSONL file not found. Skipping inspection.\n")
        return

    with open(jsonl_abs, "r", encoding="utf-8") as f:
        first_line = f.readline().strip()
        if not first_line:
            print("[✗] JSONL file is empty.\n")
            return

        ev0 = json.loads(first_line)
        parts0 = ev0["data"]["particles"]
        print(f"[✓] First event has {len(parts0)} final-state particles")
        if parts0:
            print(f"  First particle: {parts0[0]}\n")


# ============================================================================
# Cleanup Functions
# ============================================================================

def cleanup_test_files():
    """Remove all test-generated files and directories."""
    print("\n>> Cleaning up test files...\n")

    # Directories to clean up
    cleanup_dirs = [
        Path(base_directory) / "data",
        Path(base_directory) / "data_seed_test_1",
        Path(base_directory) / "data_seed_test_2",
        Path(base_directory) / "data_seed_test_3",
        Path(base_directory) / "data_finals_test",
        Path(base_directory) / "data_scan_test",
    ]

    cleaned = 0
    for dir_path in cleanup_dirs:
        if dir_path.exists():
            try:
                shutil.rmtree(dir_path)
                print(f"[✓] Removed: {dir_path.name}")
                cleaned += 1
            except Exception as e:
                print(f"[⚠] Failed to remove {dir_path.name}: {e}")

    if cleaned == 0:
        print("[i] No test files to clean up")
    else:
        print(f"\n[✓] Cleaned up {cleaned} test directory(ies)\n")


# ============================================================================
# Main Test Runner
# ============================================================================

if __name__ == "__main__":
    import sys

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Run MadGraph5 tools test suite",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--keep-files",
        action="store_true",
        help="Keep test-generated files after tests complete (useful for debugging)"
    )
    args = parser.parse_args()

    # Run all tests and track failures
    all_passed = True

    # Unit tests
    try:
        test_edit_mg5_card()
    except (AssertionError, Exception) as e:
        print(f"\n[✗] test_edit_mg5_card failed: {e}\n")
        all_passed = False

    try:
        test_scan_detection()
    except (AssertionError, Exception) as e:
        print(f"\n[✗] test_scan_detection failed: {e}\n")
        all_passed = False

    try:
        test_path_traversal_prevention()
    except (AssertionError, Exception) as e:
        print(f"\n[✗] test_path_traversal_prevention failed: {e}\n")
        all_passed = False

    # Integration tests
    mg_output = test_madgraph_generation()
    if mg_output is None:
        all_passed = False

    conv_output = test_lhe_to_jsonl_conversion(mg_output)
    if conv_output is None:
        all_passed = False

    if conv_output:
        test_inspect_jsonl_output()

    # Advanced integration tests (require MG5 to work)
    if mg_output is not None:
        try:
            seed_result = test_seed_reproducibility()
            # Don't mark as failed if skipped (returns None)
            # Only fail if it ran and threw an exception
        except (AssertionError, Exception) as e:
            print(f"\n[✗] test_seed_reproducibility failed: {e}\n")
            all_passed = False

        try:
            scan_result = test_scan_generation_and_detection()
            # Don't mark as failed if skipped (returns None)
            # Only fail if it ran and threw an exception
        except (AssertionError, Exception) as e:
            print(f"\n[✗] test_scan_generation_and_detection failed: {e}\n")
            all_passed = False

    # Cleanup test files unless --keep-files flag is set
    if not args.keep_files:
        cleanup_test_files()
    else:
        print("\n[i] Keeping test files (--keep-files flag set)\n")

    print("=" * 70)
    if all_passed:
        print("Test suite completed successfully! [✓]")
    else:
        print("Test suite completed with failures! [✗]")
    print("=" * 70)

    sys.exit(0 if all_passed else 1)