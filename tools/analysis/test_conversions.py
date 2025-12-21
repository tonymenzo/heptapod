#!/usr/bin/env python3
"""
# test_conversions.py is a part of the HEPTAPOD package.
# Copyright (C) 2025 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.
"""
"""
Data Conversion Tools Test Suite

Tests cover:
- Path traversal prevention (security)
- JSONL to NumPy conversion with padding
- LHE to JSONL conversion with finals_only and full_history flags
- Jets JSONL to NumPy conversion with multiple extraction modes
- Schema validation for evtjsonl-1.0 format
"""

# Standard library imports
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

# Path setup
SCRIPT_PATH = Path(__file__).resolve()
ANALYSIS_DIR = SCRIPT_PATH.parent                             # .../heptapod/tools/analysis
TOOLS_DIR = ANALYSIS_DIR.parent                               # .../heptapod/tools
REPO_ROOT = TOOLS_DIR.parent                                  # .../heptapod

# Add repository root to path for local imports (prompts, tools, etc.)
sys.path.insert(0, str(REPO_ROOT))

# Tool imports
from tools.analysis.conversions import EventJSONLToNumpyTool, LHEToJSONLTool, JetsJSONLToNumpyTool

# Future: When orchestral-ai PyPI package is fixed, use these imports instead:
# from orchestral.tools import EventJSONLToNumpyTool, LHEToJSONLTool, JetsJSONLToNumpyTool

# Initialize base directory
base_directory = str(ANALYSIS_DIR / "test_files_conversions")

# Global flag for keeping test files (set by main)
_keep_files = False


# ============================================================================
# Test Functions
# ============================================================================

def test_path_traversal_prevention():
    """Test that path traversal attempts are rejected (critical security test)."""
    print(">> Testing path traversal prevention (security)...\n")

    # Create base directory for security tests
    Path(base_directory).mkdir(parents=True, exist_ok=True)

    # Test 1: EventJSONLToNumpyTool - jsonl_path traversal
    tool = EventJSONLToNumpyTool(
        base_directory=base_directory,
        jsonl_path="../../../etc/passwd",
        output_path="output/events.npy"
    )
    tool._setup()
    result = tool._run()
    assert ("error" in result.lower() or "denied" in result.lower() or
            "escape" in result.lower())
    print("[✓] Test 1 passed: EventJSONLToNumpyTool jsonl_path traversal rejected")

    # Test 2: EventJSONLToNumpyTool - output_path traversal
    tool = EventJSONLToNumpyTool(
        base_directory=base_directory,
        jsonl_path="data/events.jsonl",
        output_path="../../../tmp/evil.npy"
    )
    tool._setup()
    result = tool._run()
    assert ("error" in result.lower() or "denied" in result.lower() or
            "escape" in result.lower())
    print("[✓] Test 2 passed: EventJSONLToNumpyTool output_path traversal rejected")

    # Test 3: LHEToJSONLTool - lhe_path traversal
    tool = LHEToJSONLTool(
        base_directory=base_directory,
        lhe_path="../../../etc/passwd",
        jsonl_path="output/events.jsonl",
        finals_only=True,
        full_history=False
    )
    result = tool._run()
    assert ("error" in result.lower() or "denied" in result.lower() or
            "escape" in result.lower())
    print("[✓] Test 3 passed: LHEToJSONLTool lhe_path traversal rejected")

    # Test 4: LHEToJSONLTool - jsonl_path traversal
    tool = LHEToJSONLTool(
        base_directory=base_directory,
        lhe_path="data/events.lhe",
        jsonl_path="../../../tmp/evil.jsonl",
        finals_only=True,
        full_history=False
    )
    result = tool._run()
    assert ("error" in result.lower() or "denied" in result.lower() or
            "escape" in result.lower())
    print("[✓] Test 4 passed: LHEToJSONLTool jsonl_path traversal rejected")

    print("\nAll path traversal prevention tests passed! [✓]\n")


def test_jsonl_to_numpy_conversion():
    """Test JSONL to NumPy conversion with various edge cases."""
    print(">> Testing JSONL to NumPy conversion...\n")

    # Create test directory
    test_dir = Path(base_directory) / "test_jsonl_numpy"
    test_dir.mkdir(parents=True, exist_ok=True)

    # Test 1: Basic conversion with uniform events
    jsonl_path = test_dir / "uniform_events.jsonl"
    with open(jsonl_path, "w") as f:
        for i in range(3):
            event = {
                "schema": "evtjsonl-1.0",
                "event_id": i,
                "finals_only": True,
                "full_history": False,
                "data": {
                    "n_particles": 2,
                    "particles": [
                        {"i": 0, "id": 11, "px": 1.0, "py": 2.0, "pz": 3.0, "E": 4.0, "m": 0.0},
                        {"i": 1, "id": -11, "px": -1.0, "py": -2.0, "pz": -3.0, "E": 4.0, "m": 0.0}
                    ]
                }
            }
            f.write(json.dumps(event) + "\n")

    tool = EventJSONLToNumpyTool(
        base_directory=str(test_dir),
        jsonl_path="uniform_events.jsonl",
        output_path="uniform_events.npy"
    )
    tool._setup()
    result = tool._run()

    try:
        output = json.loads(result)
        assert output.get("status") == "ok", f"Expected ok status, got: {result}"
        assert output.get("input_events") == 3
        assert output.get("max_particles") == 2
        assert output.get("output_shape") == [3, 2, 5]  # 3 events, 2 particles, 5 features
        print("[✓] Test 1 passed: Uniform events conversion")
    except json.JSONDecodeError:
        print(f"[✗] Test 1 failed: Could not parse output as JSON")
        print(f"  Raw output: {result}")
        return None

    # Test 2: Conversion with variable-length events (padding test)
    jsonl_path = test_dir / "variable_events.jsonl"
    with open(jsonl_path, "w") as f:
        # Event with 1 particle
        event1 = {
            "schema": "evtjsonl-1.0",
            "event_id": 0,
            "finals_only": True,
            "full_history": False,
            "data": {
                "n_particles": 1,
                "particles": [
                    {"i": 0, "id": 22, "px": 1.0, "py": 0.0, "pz": 0.0, "E": 1.0, "m": 0.0}
                ]
            }
        }
        f.write(json.dumps(event1) + "\n")

        # Event with 3 particles
        event2 = {
            "schema": "evtjsonl-1.0",
            "event_id": 1,
            "finals_only": True,
            "full_history": False,
            "data": {
                "n_particles": 3,
                "particles": [
                    {"i": 0, "id": 11, "px": 1.0, "py": 2.0, "pz": 3.0, "E": 4.0, "m": 0.0},
                    {"i": 1, "id": -11, "px": -1.0, "py": -2.0, "pz": -3.0, "E": 4.0, "m": 0.0},
                    {"i": 2, "id": 22, "px": 0.0, "py": 0.0, "pz": 1.0, "E": 1.0, "m": 0.0}
                ]
            }
        }
        f.write(json.dumps(event2) + "\n")

    tool = EventJSONLToNumpyTool(
        base_directory=str(test_dir),
        jsonl_path="variable_events.jsonl",
        output_path="variable_events.npy"
    )
    tool._setup()
    result = tool._run()

    try:
        output = json.loads(result)
        assert output.get("status") == "ok"
        assert output.get("input_events") == 2
        assert output.get("max_particles") == 3  # Padded to longest event
        assert output.get("output_shape") == [2, 3, 5]  # 2 events, 3 particles (padded), 5 features
        print("[✓] Test 2 passed: Variable-length events with padding")

        # Verify padding is zero
        import numpy as np
        arr = np.load(test_dir / "variable_events.npy")
        # First event should have zeros in positions [0, 1:3, :] (particles 1 and 2)
        assert np.all(arr[0, 1:3, :] == 0), "Padding should be all zeros"
        print("[✓] Test 2b passed: Zero-padding verified")
    except json.JSONDecodeError:
        print(f"[✗] Test 2 failed: Could not parse output as JSON")
        return None

    # Test 3: Empty events file
    jsonl_path = test_dir / "empty_events.jsonl"
    jsonl_path.touch()  # Create empty file

    tool = EventJSONLToNumpyTool(
        base_directory=str(test_dir),
        jsonl_path="empty_events.jsonl",
        output_path="empty_events.npy"
    )
    tool._setup()
    result = tool._run()

    # Should fail gracefully
    assert ("error" in result.lower() or "status" in result.lower())
    print("[✓] Test 3 passed: Empty file handled gracefully")

    # Cleanup
    if not _keep_files:
        shutil.rmtree(test_dir)

    print("\nAll JSONL to NumPy conversion tests passed! [✓]\n")
    return {"status": "ok"}


def test_all_empty_events():
    """Test that all-empty events are handled gracefully."""
    print(">> Testing JSONL to NumPy with all-empty events...\n")

    # Create test directory
    test_dir = Path(base_directory) / "test_all_empty"
    test_dir.mkdir(parents=True, exist_ok=True)

    # Create JSONL with all empty events (edge case)
    jsonl_path = test_dir / "all_empty.jsonl"
    with open(jsonl_path, "w") as f:
        for i in range(5):
            event = {
                "schema": "evtjsonl-1.0",
                "event_id": i,
                "finals_only": True,
                "full_history": False,
                "data": {
                    "n_particles": 0,
                    "particles": []
                }
            }
            f.write(json.dumps(event) + "\n")

    tool = EventJSONLToNumpyTool(
        base_directory=str(test_dir),
        jsonl_path="all_empty.jsonl",
        output_path="all_empty.npy"
    )
    tool._setup()
    result = tool._run()

    # Should fail gracefully with Empty Dataset error
    assert ("Empty Dataset" in result or "error" in result.lower())
    assert "zero particles" in result.lower()
    print("[✓] Test passed: All-empty events handled with clear error")

    # Cleanup
    if not _keep_files:
        shutil.rmtree(test_dir)

    print("\nAll-empty events test passed! [✓]\n")
    return {"status": "ok"}


def test_lhe_to_jsonl_finals_only():
    """Test that finals_only parameter correctly filters LHE particles."""
    print(">> Testing LHE finals_only filtering...\n")

    # Create test directory and minimal LHE file
    test_dir = Path(base_directory) / "test_lhe_finals"
    test_dir.mkdir(parents=True, exist_ok=True)

    # Create a minimal valid LHE file with both intermediate and final particles
    lhe_content = """<LesHouchesEvents version="1.0">
<header>
</header>
<init>
2212 2212 6.500000e+03 6.500000e+03 0 0 247000 247000 -4 1
1.000000e+00 0.000000e+00 1.000000e+00 1
</init>
<event>
4 1 1.000000e+00 1.000000e+02 7.546772e-03 1.180000e-01
21 -1 0 0 501 502 0.000000e+00 0.000000e+00 1.000000e+02 1.000000e+02 0.000000e+00 0.0000e+00 1.0000e+00
21 -1 0 0 502 501 0.000000e+00 0.000000e+00 -1.000000e+02 1.000000e+02 0.000000e+00 0.0000e+00 -1.0000e+00
11 1 1 2 0 0 5.000000e+01 0.000000e+00 0.000000e+00 5.000000e+01 0.000511e+00 0.0000e+00 1.0000e+00
-11 1 1 2 0 0 -5.000000e+01 0.000000e+00 0.000000e+00 5.000000e+01 0.000511e+00 0.0000e+00 -1.0000e+00
</event>
</LesHouchesEvents>
"""
    lhe_path = test_dir / "test_events.lhe"
    with open(lhe_path, "w") as f:
        f.write(lhe_content)

    # Test 1: Convert with finals_only=True
    tool_finals = LHEToJSONLTool(
        base_directory=str(test_dir),
        lhe_path="test_events.lhe",
        jsonl_path="events_finals_only.jsonl",
        finals_only=True,
        full_history=False
    )
    result = tool_finals._run()

    try:
        output = json.loads(result)
        assert output.get("status") == "ok", f"Expected ok status, got: {result}"
        assert output.get("n_events") == 1
        print("[✓] Test 1 passed: LHE conversion with finals_only=True")

        # Verify only final-state particles (status=1) are kept
        with open(test_dir / "events_finals_only.jsonl", "r") as f:
            event = json.loads(f.readline())
            particles = event["data"]["particles"]
            # Should have 2 final particles (e+, e-) not 4 total
            assert len(particles) == 2, f"Expected 2 final particles, got {len(particles)}"
            # Check PDG IDs are electrons
            pdg_ids = [p["id"] for p in particles]
            assert set(pdg_ids) == {11, -11}, f"Expected electrons, got PDG IDs: {pdg_ids}"
        print("[✓] Test 1b passed: Only final-state particles retained")

    except json.JSONDecodeError:
        print(f"[✗] Test 1 failed: Could not parse output as JSON")
        print(f"  Raw output: {result}")
        shutil.rmtree(test_dir)
        return None

    # Test 2: Convert with finals_only=False
    tool_all = LHEToJSONLTool(
        base_directory=str(test_dir),
        lhe_path="test_events.lhe",
        jsonl_path="events_all.jsonl",
        finals_only=False,
        full_history=False
    )
    result = tool_all._run()

    try:
        output = json.loads(result)
        assert output.get("status") == "ok"
        assert output.get("n_events") == 1
        print("[✓] Test 2 passed: LHE conversion with finals_only=False")

        # Verify all particles are kept
        with open(test_dir / "events_all.jsonl", "r") as f:
            event = json.loads(f.readline())
            particles = event["data"]["particles"]
            # Should have all 4 particles (2 gluons + 2 electrons)
            assert len(particles) == 4, f"Expected 4 particles, got {len(particles)}"
        print("[✓] Test 2b passed: All particles retained when finals_only=False")

    except json.JSONDecodeError:
        print(f"[✗] Test 2 failed: Could not parse output as JSON")
        shutil.rmtree(test_dir)
        return None

    # Test 3: Verify finals_only filtering difference
    with open(test_dir / "events_finals_only.jsonl", "r") as f:
        event_finals = json.loads(f.readline())
        n_finals = event_finals["data"]["n_particles"]

    with open(test_dir / "events_all.jsonl", "r") as f:
        event_all = json.loads(f.readline())
        n_all = event_all["data"]["n_particles"]

    assert n_finals < n_all, f"finals_only should have fewer particles: {n_finals} vs {n_all}"
    assert n_all - n_finals == 2, f"Expected 2 intermediate particles filtered, got {n_all - n_finals}"
    print(f"[✓] Test 3 passed: Filtering removed {n_all - n_finals} intermediate particles")

    # Cleanup
    if not _keep_files:
        shutil.rmtree(test_dir)

    print("\nAll LHE finals_only filtering tests passed! [✓]\n")
    return {"status": "ok"}


def test_lhe_full_history():
    """Test that full_history parameter includes mother/status information."""
    print(">> Testing LHE full_history flag...\n")

    # Create test directory and minimal LHE file
    test_dir = Path(base_directory) / "test_lhe_history"
    test_dir.mkdir(parents=True, exist_ok=True)

    # Create LHE file with mother relationships
    lhe_content = """<LesHouchesEvents version="1.0">
<header>
</header>
<init>
2212 2212 6.500000e+03 6.500000e+03 0 0 247000 247000 -4 1
1.000000e+00 0.000000e+00 1.000000e+00 1
</init>
<event>
4 1 1.000000e+00 1.000000e+02 7.546772e-03 1.180000e-01
21 -1 0 0 501 502 0.000000e+00 0.000000e+00 1.000000e+02 1.000000e+02 0.000000e+00 0.0000e+00 1.0000e+00
21 -1 0 0 502 501 0.000000e+00 0.000000e+00 -1.000000e+02 1.000000e+02 0.000000e+00 0.0000e+00 -1.0000e+00
11 1 1 2 0 0 5.000000e+01 0.000000e+00 0.000000e+00 5.000000e+01 0.000511e+00 0.0000e+00 1.0000e+00
-11 1 1 2 0 0 -5.000000e+01 0.000000e+00 0.000000e+00 5.000000e+01 0.000511e+00 0.0000e+00 -1.0000e+00
</event>
</LesHouchesEvents>
"""
    lhe_path = test_dir / "test_events.lhe"
    with open(lhe_path, "w") as f:
        f.write(lhe_content)

    # Test 1: Convert with full_history=False (default)
    tool_no_history = LHEToJSONLTool(
        base_directory=str(test_dir),
        lhe_path="test_events.lhe",
        jsonl_path="events_no_history.jsonl",
        finals_only=True,
        full_history=False
    )
    result = tool_no_history._run()

    try:
        output = json.loads(result)
        assert output.get("status") == "ok"

        with open(test_dir / "events_no_history.jsonl", "r") as f:
            event = json.loads(f.readline())
            particles = event["data"]["particles"]
            # Should NOT have status, mother1, mother2 fields
            assert "status" not in particles[0], "status should not be present when full_history=False"
            assert "mother1" not in particles[0], "mother1 should not be present when full_history=False"
            assert "mother2" not in particles[0], "mother2 should not be present when full_history=False"
        print("[✓] Test 1 passed: full_history=False excludes mother/status info")

    except json.JSONDecodeError:
        print(f"[✗] Test 1 failed: Could not parse output as JSON")
        shutil.rmtree(test_dir)
        return None

    # Test 2: Convert with full_history=True
    tool_history = LHEToJSONLTool(
        base_directory=str(test_dir),
        lhe_path="test_events.lhe",
        jsonl_path="events_with_history.jsonl",
        finals_only=True,
        full_history=True
    )
    result = tool_history._run()

    try:
        output = json.loads(result)
        assert output.get("status") == "ok"

        with open(test_dir / "events_with_history.jsonl", "r") as f:
            event = json.loads(f.readline())
            particles = event["data"]["particles"]
            # Should have status, mother1, mother2 fields
            assert "status" in particles[0], "status should be present when full_history=True"
            assert "mother1" in particles[0], "mother1 should be present when full_history=True"
            assert "mother2" in particles[0], "mother2 should be present when full_history=True"
            # Final state particles should have status=1
            assert particles[0]["status"] == 1, "Final state particles should have status=1"
        print("[✓] Test 2 passed: full_history=True includes mother/status info")

    except json.JSONDecodeError:
        print(f"[✗] Test 2 failed: Could not parse output as JSON")
        shutil.rmtree(test_dir)
        return None

    # Cleanup
    if not _keep_files:
        shutil.rmtree(test_dir)

    print("\nAll LHE full_history tests passed! [✓]\n")
    return {"status": "ok"}


def test_jsonl_schema_validation():
    """Test that JSONL output conforms to evtjsonl-1.0 schema."""
    print(">> Testing JSONL schema validation...\n")

    # Create test directory and minimal LHE file
    test_dir = Path(base_directory) / "test_schema"
    test_dir.mkdir(parents=True, exist_ok=True)

    # Create minimal LHE file
    lhe_content = """<LesHouchesEvents version="1.0">
<header>
</header>
<init>
2212 2212 6.500000e+03 6.500000e+03 0 0 247000 247000 -4 1
1.000000e+00 0.000000e+00 1.000000e+00 1
</init>
<event>
2 1 1.000000e+00 1.000000e+02 7.546772e-03 1.180000e-01
11 1 0 0 0 0 5.000000e+01 0.000000e+00 0.000000e+00 5.000000e+01 0.000511e+00 0.0000e+00 1.0000e+00
-11 1 0 0 0 0 -5.000000e+01 0.000000e+00 0.000000e+00 5.000000e+01 0.000511e+00 0.0000e+00 -1.0000e+00
</event>
</LesHouchesEvents>
"""
    lhe_path = test_dir / "test_events.lhe"
    with open(lhe_path, "w") as f:
        f.write(lhe_content)

    tool = LHEToJSONLTool(
        base_directory=str(test_dir),
        lhe_path="test_events.lhe",
        jsonl_path="events.jsonl",
        finals_only=True,
        full_history=False
    )
    result = tool._run()

    try:
        output = json.loads(result)
        assert output.get("status") == "ok"

        # Validate schema compliance
        with open(test_dir / "events.jsonl", "r") as f:
            event = json.loads(f.readline())

            # Check top-level fields
            assert event["schema"] == "evtjsonl-1.0", "Schema version mismatch"
            assert "event_id" in event
            assert "finals_only" in event
            assert "full_history" in event
            assert "data" in event
            print("[✓] Test 1 passed: Top-level schema fields present")

            # Check data structure
            data = event["data"]
            assert "n_particles" in data
            assert "particles" in data
            assert data["n_particles"] == len(data["particles"])
            print("[✓] Test 2 passed: Data structure valid")

            # Check particle fields
            if data["particles"]:
                p = data["particles"][0]
                required_fields = ["i", "id", "px", "py", "pz", "E", "m"]
                for field in required_fields:
                    assert field in p, f"Missing required field: {field}"
                print("[✓] Test 3 passed: Particle fields complete")

    except json.JSONDecodeError:
        print(f"[✗] Schema validation failed: Could not parse output as JSON")
        shutil.rmtree(test_dir)
        return None

    # Cleanup
    if not _keep_files:
        shutil.rmtree(test_dir)

    print("\nAll JSONL schema validation tests passed! [✓]\n")
    return {"status": "ok"}


def test_jets_jsonl_to_numpy_conversion():
    """Test Jets JSONL to NumPy conversion with all extraction modes."""
    print(">> Testing Jets JSONL to NumPy conversion...\n")

    # Create test directory
    test_dir = Path(base_directory) / "test_jets_numpy"
    test_dir.mkdir(parents=True, exist_ok=True)

    # Create test jets JSONL file with realistic structure
    jsonl_path = test_dir / "test_jets.jsonl"
    with open(jsonl_path, "w") as f:
        # Event 1: 2 jets with constituents (unified evtjsonl-1.0 schema)
        event1 = {
            "schema": "evtjsonl-1.0",
            "event_id": 0,
            "data": {
                "algorithm": "antikt",
                "R": 0.4,
                "ptmin": 20.0,
                "etamax": 5.0,
                "mass_option": 1,
                "n_jets": 2,
                "jets": [
                    {
                        "index": 0,
                        "px": 100.0, "py": 50.0, "pz": 200.0, "E": 230.0, "m": 10.5,
                        "pT": 111.8, "eta": 1.2, "phi": 0.46,
                        "n_const": 2,
                        "constituents": [
                            {"event_index": 0, "px": 60.0, "py": 30.0, "pz": 120.0, "E": 140.0, "m": 5.0},
                            {"event_index": 1, "px": 40.0, "py": 20.0, "pz": 80.0, "E": 90.0, "m": 5.5}
                        ]
                    },
                    {
                        "index": 1,
                        "px": -50.0, "py": -25.0, "pz": -100.0, "E": 115.0, "m": 5.2,
                        "pT": 55.9, "eta": -1.2, "phi": -2.68,
                        "n_const": 3,
                        "constituents": [
                            {"event_index": 2, "px": -20.0, "py": -10.0, "pz": -40.0, "E": 46.0, "m": 2.0},
                            {"event_index": 3, "px": -15.0, "py": -7.5, "pz": -30.0, "E": 34.5, "m": 1.5},
                            {"event_index": 4, "px": -15.0, "py": -7.5, "pz": -30.0, "E": 34.5, "m": 1.7}
                        ]
                    }
                ]
            }
        }
        f.write(json.dumps(event1) + "\n")

        # Event 2: 1 jet with constituents (test padding)
        event2 = {
            "schema": "evtjsonl-1.0",
            "event_id": 1,
            "data": {
                "algorithm": "antikt",
                "R": 0.4,
                "ptmin": 20.0,
                "etamax": 5.0,
                "mass_option": 1,
                "n_jets": 1,
                "jets": [
                    {
                        "index": 0,
                        "px": 75.0, "py": 0.0, "pz": 150.0, "E": 170.0, "m": 8.0,
                        "pT": 75.0, "eta": 1.4, "phi": 0.0,
                        "n_const": 1,
                        "constituents": [
                            {"event_index": 0, "px": 75.0, "py": 0.0, "pz": 150.0, "E": 170.0, "m": 8.0}
                        ]
                    }
                ]
            }
        }
        f.write(json.dumps(event2) + "\n")

    # Test 1: Extract jets only
    tool_jets = JetsJSONLToNumpyTool(
        base_directory=str(test_dir),
        jsonl_path="test_jets.jsonl",
        output_path="jets_only.npy",
        extraction_mode="jets"
    )
    tool_jets._setup()
    result = tool_jets._run()

    try:
        output = json.loads(result)
        assert output.get("status") == "ok", f"Expected ok status, got: {result}"
        assert output.get("extraction_mode") == "jets"
        assert output.get("input_events") == 2
        assert output.get("max_objects_per_event") == 2  # Event 1 has 2 jets (max)
        assert output.get("output_shape") == [2, 2, 5]  # 2 events, 2 jets (padded), 5 features
        assert output.get("feature_columns") == ["px", "py", "pz", "E", "m"]
        print("[✓] Test 1 passed: Jets extraction mode")

        # Verify data integrity
        import numpy as np
        arr = np.load(test_dir / "jets_only.npy")
        # Check first jet of first event
        assert arr[0, 0, 0] == 100.0  # px
        assert arr[0, 0, 1] == 50.0   # py
        assert arr[0, 0, 2] == 200.0  # pz
        assert arr[0, 0, 3] == 230.0  # E
        assert arr[0, 0, 4] == 10.5   # m
        # Check padding on second event (should have zeros in second jet position)
        assert np.all(arr[1, 1, :] == 0), "Second jet of event 2 should be zero-padded"
        print("[✓] Test 1b passed: Jets data integrity verified")

    except json.JSONDecodeError:
        print(f"[✗] Test 1 failed: Could not parse output as JSON")
        print(f"  Raw output: {result}")
        shutil.rmtree(test_dir)
        return None

    # Test 2: Extract constituents
    tool_const = JetsJSONLToNumpyTool(
        base_directory=str(test_dir),
        jsonl_path="test_jets.jsonl",
        output_path="constituents.npy",
        extraction_mode="constituents"
    )
    tool_const._setup()
    result = tool_const._run()

    try:
        output = json.loads(result)
        assert output.get("status") == "ok"
        assert output.get("extraction_mode") == "constituents"
        assert output.get("input_events") == 2
        # Event 1 has 2+3=5 constituents (max), Event 2 has 1
        assert output.get("max_objects_per_event") == 5
        assert output.get("output_shape") == [2, 5, 5]  # 2 events, 5 constituents (padded), 5 features
        assert output.get("feature_columns") == ["px", "py", "pz", "E", "m"]
        print("[✓] Test 2 passed: Constituents extraction mode")

        # Verify constituent data
        arr = np.load(test_dir / "constituents.npy")
        # Check first constituent of first event
        assert arr[0, 0, 0] == 60.0  # px of first constituent
        assert arr[0, 0, 4] == 5.0   # m of first constituent
        # Event 2 should have padding in positions [1:5, :]
        assert np.all(arr[1, 1:5, :] == 0), "Event 2 should be padded from constituent 1 onward"
        print("[✓] Test 2b passed: Constituent data integrity verified")

    except json.JSONDecodeError:
        print(f"[✗] Test 2 failed: Could not parse output as JSON")
        shutil.rmtree(test_dir)
        return None

    # Test 3: Extract jets with metadata
    tool_meta = JetsJSONLToNumpyTool(
        base_directory=str(test_dir),
        jsonl_path="test_jets.jsonl",
        output_path="jets_with_metadata.npy",
        extraction_mode="jets_with_metadata"
    )
    tool_meta._setup()
    result = tool_meta._run()

    try:
        output = json.loads(result)
        assert output.get("status") == "ok"
        assert output.get("extraction_mode") == "jets_with_metadata"
        assert output.get("input_events") == 2
        assert output.get("max_objects_per_event") == 2
        assert output.get("output_shape") == [2, 2, 8]  # 2 events, 2 jets, 8 features
        assert output.get("feature_columns") == ["px", "py", "pz", "E", "m", "pT", "eta", "phi"]
        print("[✓] Test 3 passed: Jets with metadata extraction mode")

        # Verify metadata is included
        arr = np.load(test_dir / "jets_with_metadata.npy")
        # Check kinematic variables for first jet
        assert arr[0, 0, 5] == 111.8  # pT
        assert arr[0, 0, 6] == 1.2    # eta
        assert arr[0, 0, 7] == 0.46   # phi
        print("[✓] Test 3b passed: Jet metadata (pT, eta, phi) verified")

    except json.JSONDecodeError:
        print(f"[✗] Test 3 failed: Could not parse output as JSON")
        shutil.rmtree(test_dir)
        return None

    # Cleanup
    if not _keep_files:
        shutil.rmtree(test_dir)

    print("\nAll Jets JSONL to NumPy conversion tests passed! [✓]\n")
    return {"status": "ok"}


def test_jets_path_traversal_prevention():
    """Test that Jets tool prevents path traversal attacks."""
    print(">> Testing Jets tool path traversal prevention...\n")

    # Test 1: JetsJSONLToNumpyTool - jsonl_path traversal
    tool = JetsJSONLToNumpyTool(
        base_directory=base_directory,
        jsonl_path="../../../etc/passwd",
        output_path="output/jets.npy",
        extraction_mode="jets"
    )
    tool._setup()
    result = tool._run()
    assert ("error" in result.lower() or "denied" in result.lower() or
            "escape" in result.lower())
    print("[✓] Test 1 passed: JetsJSONLToNumpyTool jsonl_path traversal rejected")

    # Test 2: JetsJSONLToNumpyTool - output_path traversal
    tool = JetsJSONLToNumpyTool(
        base_directory=base_directory,
        jsonl_path="data/jets.jsonl",
        output_path="../../../tmp/evil.npy",
        extraction_mode="jets"
    )
    tool._setup()
    result = tool._run()
    assert ("error" in result.lower() or "denied" in result.lower() or
            "escape" in result.lower())
    print("[✓] Test 2 passed: JetsJSONLToNumpyTool output_path traversal rejected")

    print("\nAll Jets path traversal prevention tests passed! [✓]\n")


def test_jets_invalid_extraction_mode():
    """Test that invalid extraction modes are rejected."""
    print(">> Testing Jets tool invalid extraction mode handling...\n")

    # Create test directory and minimal jets file
    test_dir = Path(base_directory) / "test_jets_invalid"
    test_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = test_dir / "test_jets.jsonl"
    with open(jsonl_path, "w") as f:
        event = {
            "algorithm": "antikt",
            "R": 0.4,
            "ptmin": 20.0,
            "etamax": 5.0,
            "mass_option": 1,
            "n_jets": 1,
            "jets": [
                {
                    "index": 0,
                    "px": 100.0, "py": 50.0, "pz": 200.0, "E": 230.0, "m": 10.5,
                    "pT": 111.8, "eta": 1.2, "phi": 0.46,
                    "n_const": 1,
                    "constituents": [
                        {"event_index": 0, "px": 100.0, "py": 50.0, "pz": 200.0, "E": 230.0, "m": 10.5}
                    ]
                }
            ],
            "event_index": 0
        }
        f.write(json.dumps(event) + "\n")

    # Test invalid extraction mode
    tool = JetsJSONLToNumpyTool(
        base_directory=str(test_dir),
        jsonl_path="test_jets.jsonl",
        output_path="output.npy",
        extraction_mode="invalid_mode"
    )
    tool._setup()
    result = tool._run()

    try:
        output = json.loads(result)
        # Should return error for invalid mode
        assert output.get("error") is not None or "error" in result.lower()
        print("[✓] Test passed: Invalid extraction mode rejected with error")

    except json.JSONDecodeError:
        # If result is not JSON, check it contains error message
        assert "error" in result.lower() or "invalid" in result.lower()
        print("[✓] Test passed: Invalid extraction mode rejected")

    # Cleanup
    if not _keep_files:
        shutil.rmtree(test_dir)

    print("\nAll Jets invalid mode tests passed! [✓]\n")
    return {"status": "ok"}


def test_jets_schema_validation():
    """Test that Jets tool validates schema correctly."""
    print(">> Testing Jets tool schema validation...\n")

    # Create test directory
    test_dir = Path(base_directory) / "test_jets_schema"
    test_dir.mkdir(parents=True, exist_ok=True)

    # Test 1: Invalid schema (missing 'jets' key)
    jsonl_path = test_dir / "invalid_schema.jsonl"
    with open(jsonl_path, "w") as f:
        # Write events schema instead of jets schema
        event = {
            "schema": "evtjsonl-1.0",
            "event_id": 0,
            "data": {"n_particles": 0, "particles": []}
        }
        f.write(json.dumps(event) + "\n")

    tool = JetsJSONLToNumpyTool(
        base_directory=str(test_dir),
        jsonl_path="invalid_schema.jsonl",
        output_path="output.npy",
        extraction_mode="jets"
    )
    tool._setup()
    result = tool._run()

    # Should fail with schema error
    assert "error" in result.lower() and "jets" in result.lower()
    print("[✓] Test 1 passed: Invalid schema (missing 'jets' key) rejected")

    # Test 2: Empty file
    jsonl_path = test_dir / "empty.jsonl"
    jsonl_path.touch()

    tool = JetsJSONLToNumpyTool(
        base_directory=str(test_dir),
        jsonl_path="empty.jsonl",
        output_path="output.npy",
        extraction_mode="jets"
    )
    tool._setup()
    result = tool._run()

    assert "error" in result.lower() or "empty" in result.lower()
    print("[✓] Test 2 passed: Empty file handled gracefully")

    # Cleanup
    if not _keep_files:
        shutil.rmtree(test_dir)

    print("\nAll Jets schema validation tests passed! [✓]\n")
    return {"status": "ok"}


# ============================================================================
# Main Test Runner
# ============================================================================

if __name__ == "__main__":
    import sys
    import argparse

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Test suite for data conversion tools")
    parser.add_argument("--keep-files", action="store_true",
                        help="Keep test-generated files after tests complete")
    args = parser.parse_args()

    # Set module-level flag (no 'global' needed - we're at module level)
    _keep_files = args.keep_files

    print(f"Base directory: {base_directory}\n")

    # Run all tests and track failures
    all_passed = True

    # Unit tests
    try:
        test_path_traversal_prevention()
    except (AssertionError, Exception) as e:
        print(f"\n[✗] test_path_traversal_prevention failed: {e}\n")
        all_passed = False

    # Integration tests
    try:
        result = test_jsonl_to_numpy_conversion()
        if result is None:
            all_passed = False
    except (AssertionError, Exception) as e:
        print(f"\n[✗] test_jsonl_to_numpy_conversion failed: {e}\n")
        all_passed = False

    try:
        result = test_all_empty_events()
        if result is None:
            all_passed = False
    except (AssertionError, Exception) as e:
        print(f"\n[✗] test_all_empty_events failed: {e}\n")
        all_passed = False

    try:
        result = test_lhe_to_jsonl_finals_only()
        if result is None:
            all_passed = False
    except (AssertionError, Exception) as e:
        print(f"\n[✗] test_lhe_to_jsonl_finals_only failed: {e}\n")
        all_passed = False

    try:
        result = test_lhe_full_history()
        if result is None:
            all_passed = False
    except (AssertionError, Exception) as e:
        print(f"\n[✗] test_lhe_full_history failed: {e}\n")
        all_passed = False

    try:
        result = test_jsonl_schema_validation()
        if result is None:
            all_passed = False
    except (AssertionError, Exception) as e:
        print(f"\n[✗] test_jsonl_schema_validation failed: {e}\n")
        all_passed = False

    # Jets JSONL to NumPy tests
    try:
        result = test_jets_jsonl_to_numpy_conversion()
        if result is None:
            all_passed = False
    except (AssertionError, Exception) as e:
        print(f"\n[✗] test_jets_jsonl_to_numpy_conversion failed: {e}\n")
        all_passed = False

    try:
        test_jets_path_traversal_prevention()
    except (AssertionError, Exception) as e:
        print(f"\n[✗] test_jets_path_traversal_prevention failed: {e}\n")
        all_passed = False

    try:
        result = test_jets_invalid_extraction_mode()
        if result is None:
            all_passed = False
    except (AssertionError, Exception) as e:
        print(f"\n[✗] test_jets_invalid_extraction_mode failed: {e}\n")
        all_passed = False

    try:
        result = test_jets_schema_validation()
        if result is None:
            all_passed = False
    except (AssertionError, Exception) as e:
        print(f"\n[✗] test_jets_schema_validation failed: {e}\n")
        all_passed = False

    if all_passed:
        print()
        print("=" * 70)
        print("Test suite completed successfully! [✓]")
        print("=" * 70)

        # Cleanup base test directory
        if not _keep_files:
            import shutil
            shutil.rmtree(base_directory, ignore_errors=True)
    else:
        print()
        print("=" * 70)
        print("Test suite completed with failures! [✗]")
        print("=" * 70)

    sys.exit(0 if all_passed else 1)
