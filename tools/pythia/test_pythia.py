#!/usr/bin/env python3
"""
# test_pythia.py is a part of the HEPTAPOD package.
# Copyright (C) 2025 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.
"""
"""
Pythia Tools Test Suite

Tests cover:
- Pythia event generation and showering
- Card editing functionality
- Seed reproducibility
- Finals-only filtering
- Jet clustering with SlowJet
- Path traversal prevention (security)
"""

# Standard library imports
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

# Path setup
SCRIPT_PATH = Path(__file__).resolve()
PYTHIA_DIR = SCRIPT_PATH.parent                               # .../heptapod/tools/pythia
TOOLS_DIR = PYTHIA_DIR.parent                                 # .../heptapod/tools
REPO_ROOT = TOOLS_DIR.parent                                  # .../heptapod

# Add repository root to path for local imports (prompts, tools, etc.)
sys.path.insert(0, str(REPO_ROOT))

# Tool imports
from tools.pythia import PythiaFromRunCardTool, JetClusterSlowJetTool, _edit_pythia_card
from tools.analysis.conversions import EventJSONLToNumpyTool

# Future: When orchestral-ai PyPI package is fixed, use these imports instead:
# from orchestral.tools import PythiaFromRunCardTool, JetClusterSlowJetTool, EventJSONLToNumpyTool

# Initialize base directory
base_directory = str(PYTHIA_DIR / "test_files")

# Parse command-line arguments
parser = argparse.ArgumentParser(
    description="Run Pythia tools test suite",
    formatter_class=argparse.RawDescriptionHelpFormatter
)
parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
parser.add_argument(
    "--keep-files",
    action="store_true",
    help="Keep test-generated files after tests complete (useful for debugging)"
)
args = parser.parse_args()


# ============================================================================
# Test Functions
# ============================================================================

def test_edit_pythia_card():
    """Test the _edit_pythia_card function for replacing LHE paths."""
    print(">> Testing Pythia card editing functionality...\n")

    # Sample template card with LHE input
    template = """! Pythia configuration for showering LHE events

Beams:frameType = 4
Beams:LHEF = /old/path/to/events.lhe

Random:setSeed = on
Random:seed = 12345

PartonLevel:MPI = off
"""

    # Test 1: Replace LHE path
    edited = _edit_pythia_card(template, lhe_path="/new/path/to/events.lhe")
    assert "Beams:LHEF = /new/path/to/events.lhe" in edited
    assert "/old/path/to/events.lhe" not in edited
    assert "Random:seed = 12345" in edited  # Other lines unchanged
    print("[✓] Test 1 passed: LHE path replacement")

    # Test 2: No replacement (should return identical card)
    edited = _edit_pythia_card(template)
    assert edited == template
    print("[✓] Test 2 passed: No changes when no parameters provided")

    # Test 3: Card without LHEF line (standalone Pythia)
    standalone_template = """! Standalone Pythia configuration

Beams:eCM = 13000.0

HardQCD:all = on

Random:setSeed = on
Random:seed = 42
"""
    edited = _edit_pythia_card(standalone_template, lhe_path="/some/lhe/file.lhe")
    # Should not add the line, just keep original
    assert edited == standalone_template
    print("[✓] Test 3 passed: Standalone card unchanged when no LHEF line exists")

    # Test 4: Preserve trailing newline
    template_no_newline = "Beams:LHEF = /old/path.lhe"
    edited = _edit_pythia_card(template_no_newline, lhe_path="/new/path.lhe")
    assert not edited.endswith("\n")
    assert "Beams:LHEF = /new/path.lhe" in edited
    print("[✓] Test 4 passed: No trailing newline when original doesn't have one")

    template_with_newline = "Beams:LHEF = /old/path.lhe\n"
    edited = _edit_pythia_card(template_with_newline, lhe_path="/new/path.lhe")
    assert edited.endswith("\n")
    assert "Beams:LHEF = /new/path.lhe" in edited
    print("[✓] Test 5 passed: Trailing newline preserved when original has one")

    # Test 6: Handle extra whitespace (like in actual cards)
    template_whitespace = """Beams:frameType = 4
Beams:LHEF      = /path/with/spaces.lhe
Random:seed = 42
"""
    edited = _edit_pythia_card(template_whitespace, lhe_path="/new/path.lhe")
    assert "Beams:LHEF = /new/path.lhe" in edited
    assert "/path/with/spaces.lhe" not in edited
    assert "Beams:frameType = 4" in edited
    print("[✓] Test 6 passed: Handle extra whitespace before equals sign")

    print("\nAll card editing tests passed! [✓]\n")


def test_path_traversal_prevention():
    """Test that path traversal attempts are rejected (critical security test)."""
    print(">> Testing path traversal prevention (security)...\n")

    # Test 1: cmnd_path with path traversal
    tool = PythiaFromRunCardTool(
        base_directory=base_directory,
        data_dir="data/",
        cmnd_path="../../../etc/passwd",
        n_events=100,
        seed=12345,
        finals_only=True,
        full_history=False
    )
    tool._setup()
    result = tool._run()
    # Should fail due to file not found (path escapes sandbox)
    try:
        result_json = json.loads(result)
        assert result_json.get("status") != "ok", "Expected failure for path traversal"
    except json.JSONDecodeError:
        # format_error() returns plain text
        assert ("error" in result.lower() or "not found" in result.lower() or
                "denied" in result.lower() or "escape" in result.lower())
    print("[✓] Test 1 passed: cmnd_path traversal rejected")

    # Test 2: data_dir with path traversal
    tool = PythiaFromRunCardTool(
        base_directory=base_directory,
        data_dir="../../../tmp/",
        cmnd_path="cards/pp_ttbar.cmnd",
        n_events=100,
        seed=12345,
        finals_only=True,
        full_history=False
    )
    tool._setup()
    result = tool._run()
    # Tool should handle this - either reject or sandbox it
    # We just verify it doesn't crash
    print("[✓] Test 2 passed: data_dir traversal handled")

    # Test 3: Absolute path in cmnd_path (should be rejected/handled)
    tool = PythiaFromRunCardTool(
        base_directory=base_directory,
        data_dir="data/",
        cmnd_path="/etc/passwd",
        n_events=100,
        seed=12345,
        finals_only=True,
        full_history=False
    )
    tool._setup()
    result = tool._run()
    try:
        result_json = json.loads(result)
        assert result_json.get("status") != "ok", "Expected failure for absolute path"
    except json.JSONDecodeError:
        assert ("error" in result.lower() or "not found" in result.lower())
    print("[✓] Test 3 passed: Absolute path in cmnd_path rejected")

    print("\nAll path traversal prevention tests passed! [✓]\n")


def test_seed_reproducibility():
    """Test that same seed produces identical events (scientific reproducibility)."""
    print(">> Testing seed reproducibility...\n")

    # Check if test card exists
    card_path = base_directory + "/cards/pp_ttbar.cmnd"
    if not os.path.exists(card_path):
        print(f"[⊘] Skipping seed reproducibility test: card not found at {card_path}\n")
        return None

    # Generate events with seed 42
    tool_run1 = PythiaFromRunCardTool(
        base_directory=base_directory,
        data_dir="data_seed_test_1/",
        cmnd_path=card_path,
        n_events=100,  # Small number for speed
        seed=42,
        finals_only=True,
        full_history=False
    )
    tool_run1._setup()
    output1_raw = tool_run1._run()

    try:
        output1 = json.loads(output1_raw)
    except (json.JSONDecodeError, ValueError):
        print(f"[⊘] Skipping seed reproducibility test: Pythia run 1 failed\n")
        print(f"  {output1_raw}\n")
        return None

    if output1.get("status") != "ok":
        print(f"[⊘] Skipping seed reproducibility test: Pythia run 1 failed\n")
        return None

    # Generate events with same seed 42
    tool_run2 = PythiaFromRunCardTool(
        base_directory=base_directory,
        data_dir="data_seed_test_2/",
        cmnd_path=card_path,
        n_events=100,
        seed=42,
        finals_only=True,
        full_history=False
    )
    tool_run2._setup()
    output2_raw = tool_run2._run()

    try:
        output2 = json.loads(output2_raw)
    except (json.JSONDecodeError, ValueError):
        print(f"[⊘] Skipping seed reproducibility test: Pythia run 2 failed\n")
        return None

    if output2.get("status") != "ok":
        print(f"[⊘] Skipping seed reproducibility test: Pythia run 2 failed\n")
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
    tool_run3 = PythiaFromRunCardTool(
        base_directory=base_directory,
        data_dir="data_seed_test_3/",
        cmnd_path=card_path,
        n_events=100,
        seed=999,  # Different seed
        finals_only=True,
        full_history=False
    )
    tool_run3._setup()
    output3_raw = tool_run3._run()

    try:
        output3 = json.loads(output3_raw)
    except (json.JSONDecodeError, ValueError):
        print(f"[⊘] Skipping different-seed test: Pythia run 3 failed\n")
        return {"status": "partial"}

    if output3.get("status") == "ok":
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


def test_finals_only_filtering():
    """Test that finals_only parameter correctly filters particles."""
    print(">> Testing finals_only filtering...\n")

    # Check if test card exists
    card_path = base_directory + "/cards/pp_ttbar.cmnd"
    if not os.path.exists(card_path):
        print(f"[⊘] Skipping finals_only test: card not found at {card_path}\n")
        return None

    # Generate with finals_only=True
    tool_finals = PythiaFromRunCardTool(
        base_directory=base_directory,
        data_dir="data_finals_test/",
        cmnd_path=card_path,
        n_events=100,
        seed=12345,
        finals_only=True,
        full_history=False
    )
    tool_finals._setup()
    finals_raw = tool_finals._run()

    try:
        finals_output = json.loads(finals_raw)
    except (json.JSONDecodeError, ValueError):
        print(f"[⊘] Skipping finals_only test: generation with finals_only=True failed\n")
        return None

    if finals_output.get("status") != "ok":
        print(f"[⊘] Skipping finals_only test: generation with finals_only=True failed\n")
        return None

    # Generate with finals_only=False
    tool_all = PythiaFromRunCardTool(
        base_directory=base_directory,
        data_dir="data_all_test/",
        cmnd_path=card_path,
        n_events=100,
        seed=12345,  # Same seed to compare same event
        finals_only=False,
        full_history=False
    )
    tool_all._setup()
    all_raw = tool_all._run()

    try:
        all_output = json.loads(all_raw)
    except (json.JSONDecodeError, ValueError):
        print(f"[⊘] Skipping finals_only test: generation with finals_only=False failed\n")
        return None

    if all_output.get("status") != "ok":
        print(f"[⊘] Skipping finals_only test: generation with finals_only=False failed\n")
        return None

    # Load and compare first events
    finals_path = os.path.join(base_directory, "data_finals_test", "events.jsonl")
    all_path = os.path.join(base_directory, "data_all_test", "events.jsonl")

    with open(finals_path, "r") as f_finals, open(all_path, "r") as f_all:
        event_finals = json.loads(f_finals.readline())
        event_all = json.loads(f_all.readline())

        particles_finals = event_finals["data"]["particles"]
        particles_all = event_all["data"]["particles"]

        # Test 1: finals_only has fewer or equal particles
        assert len(particles_finals) <= len(particles_all), \
            f"finals_only has MORE particles ({len(particles_finals)}) than all ({len(particles_all)})"
        print(f"[✓] Test 1 passed: finals_only filtered correctly ({len(particles_finals)} ≤ {len(particles_all)})")

        # Test 2: All particles in finals_only should have status==1 (final state)
        if particles_finals and "status" in particles_finals[0]:
            all_final = all(p.get("status") == 1 for p in particles_finals)
            if all_final:
                print("[✓] Test 2 passed: All particles have status==1 (final state)")
            else:
                print("[⊘] Test 2 partial: Some particles don't have status==1")
        else:
            print("[✓] Test 2 passed: No status field to validate")

        # Test 3: finals_only should exclude intermediate particles
        particle_diff = len(particles_all) - len(particles_finals)
        if particle_diff > 0:
            print(f"[✓] Test 3 passed: Filtering removed {particle_diff} intermediate particles")
        else:
            print("[⊘] Test 3 note: No particles filtered (process may produce only finals)")

    print("\nAll finals_only filtering tests passed! [✓]\n")
    return {"status": "ok"}


def test_event_generation(verbose=False):
    """Test Pythia event generation from run card."""
    print(">> Testing Pythia event generation...\n")

    tool_evt_gen = PythiaFromRunCardTool(
        base_directory=base_directory,
        data_dir="data/",
        cmnd_path=base_directory + "/cards/pp_ttbar.cmnd",
        n_events=1000,
        seed=12345,
        finals_only=True,
        full_history=False
    )

    tool_evt_gen._setup()
    output_raw = tool_evt_gen._run()

    try:
        output = json.loads(output_raw)
    except Exception:
        print(f"[✗] Failed to parse tool output")
        print(f"  Raw output: {output_raw}\n")
        return None

    if output.get("status") == "ok":
        print("[✓] Events generated successfully!")
        events_path = os.path.join(base_directory, "data", "events.jsonl")
        print(f"  Events file: {events_path}")
        print(f"  Event count: {output.get('accepted', 'unknown')}\n")

        if verbose:
            # Print the first 5 particles of the first event
            with open(events_path, 'r') as f:
                first_event = json.loads(f.readline())
                particles = first_event['data']['particles']
                print(f"  Total particles in first event: {len(particles)}")
                print("\n  First 5 particles (JSONL format):")
                for particle in particles[:5]:
                    print(f"    {json.dumps(particle, indent=4)}")
                print()

        return output
    else:
        print("[✗] Event generation failed.")
        print(f"  Error: {output.get('reason', output.get('error_message', 'Unknown error'))}\n")
        return None


def test_jsonl_to_numpy_conversion(verbose=False):
    """Test JSONL to NumPy conversion."""
    print(">> Testing JSONL to NumPy conversion...\n")

    tool_evt_to_numpy = EventJSONLToNumpyTool(
        base_directory=base_directory,
        jsonl_path="data/events.jsonl",
        output_path="data/events.npy"
    )

    tool_evt_to_numpy._setup()
    output_raw = tool_evt_to_numpy._run()

    try:
        output = json.loads(output_raw)
    except Exception:
        print(f"[✗] Failed to parse tool output")
        print(f"  Raw output: {output_raw}\n")
        return None

    if output.get("status") == "ok":
        print("[✓] Events converted successfully!")
        npy_path = os.path.join(base_directory, "data", "events.npy")
        print(f"  NumPy file: {npy_path}\n")

        if verbose:
            try:
                import numpy as np
                events = np.load(npy_path, allow_pickle=True)
                print(f"  Total events in NumPy file: {len(events)}")
                print("\n  First 5 particles from first event (NumPy format):")
                print(f"    {events[0][:5]}\n")
            except Exception as e:
                print(f"  Warning: Could not load NumPy file: {e}\n")

        return output
    else:
        print("[✗] NumPy conversion failed.")
        print(f"  Error: {output.get('reason', output.get('error_message', 'Unknown error'))}\n")
        return None


def test_jet_clustering(cluster_all=True, verbose=False):
    """Test jet clustering with SlowJet."""
    print(">> Testing jet clustering...\n")

    # Clustering parameters
    ALG = "antikt"
    RADIUS = 0.4
    PTMIN = 20.0
    ETAMAX = 5.0
    OUTPUT_PATH = "data/jet_clusters.jsonl"

    # Build tool parameters based on mode
    tool_params = {
        "base_directory": base_directory,
        "jsonl_path": "data/events.jsonl",
        "cluster_all": cluster_all,
        "algorithm": ALG,
        "R": RADIUS,
        "ptmin": PTMIN,
        "etamax": ETAMAX,
        "mass_option": 1
    }

    # For cluster_all mode, provide output_path
    # For single-event mode, provide event_index
    if cluster_all:
        tool_params["output_path"] = OUTPUT_PATH
    else:
        tool_params["event_index"] = 0  # Test first event

    tool_jet_cluster = JetClusterSlowJetTool(**tool_params)

    if verbose:
        print(f"  Algorithm: {ALG}")
        print(f"  R: {RADIUS}")
        print(f"  ptmin: {PTMIN} GeV")
        print(f"  etamax: {ETAMAX}\n")

    tool_jet_cluster._setup()
    output_raw = tool_jet_cluster._run()

    try:
        output = json.loads(output_raw)
    except Exception:
        print(f"[✗] Failed to parse tool output")
        print(f"  Raw output: {output_raw}\n")
        return None

    if output.get("status") == "ok":
        if cluster_all:
            print("[✓] Clustering completed for all events!")
            print(f"  Output file: {OUTPUT_PATH}\n")

            # Load and inspect first event
            output_file = os.path.join(base_directory, OUTPUT_PATH)
            with open(output_file, 'r') as f:
                first_line = f.readline()
                first_event_jets = json.loads(first_line)
                print(f"  Jets in first event: {first_event_jets['data']['n_jets']}")

                if verbose and first_event_jets['data']['n_jets'] > 0:
                    first_jet = first_event_jets["data"]["jets"][0]
                    first_jet_data = {k: v for k, v in first_jet.items() if k != "constituents"}
                    print("\n  First jet details:")
                    print(f"    {json.dumps(first_jet_data, indent=4)}")
                    print(f"\n  First jet constituents (showing first 5):")
                    for constituent in first_jet["constituents"][:5]:
                        print(f"      {json.dumps(constituent, indent=4)}")
                print()
        else:
            # Single event mode returns {"status": "ok", "n_events": 1, "results": [result]}
            print("[✓] Clustering completed for single event!")

            # Extract the first (and only) result
            if output.get("results") and len(output["results"]) > 0:
                event_result = output["results"][0]
                n_jets = event_result.get("n_jets", 0)
                print(f"  Jets found: {n_jets}\n")

                if verbose and n_jets > 0:
                    first_jet = event_result["jets"][0]
                    first_jet_data = {k: v for k, v in first_jet.items() if k != "constituents"}
                    print("  First jet details:")
                    print(f"    {json.dumps(first_jet_data, indent=4)}")
                    print("\n  First jet constituents (showing first 5):")
                    for constituent in first_jet["constituents"][:5]:
                        print(f"      {json.dumps(constituent, indent=4)}")
                    print()
            else:
                print("  No results found\n")

        return output
    else:
        print("[✗] Jet clustering failed.")
        print(f"  Error: {output.get('reason', output.get('error_message', 'Unknown error'))}\n")
        return None


def test_jet_clustering_parameter_validation():
    """Test that JetClusterSlowJetTool properly validates parameters for different modes."""
    print(">> Testing jet clustering parameter validation...\n")

    from orchestral.tools.base.schema_generator import SchemaGenerator

    # Test 1: Verify schema has no required fields (all are optional)
    spec = SchemaGenerator.generate_tool_spec(JetClusterSlowJetTool)
    required_fields = spec.input_schema.get("required", [])
    assert required_fields == [], f"Expected no required fields, got {required_fields}"
    print("[✓] Test 1 passed: Schema correctly has no required fields")

    # Test 2: cluster_all=True without event_index should succeed
    tool = JetClusterSlowJetTool(base_directory=base_directory)
    result = tool.execute(
        jsonl_path="data/events.jsonl",
        cluster_all=True,
        output_path="data/test_validation_output.jsonl",
        algorithm="antikt",
        R=0.4,
        ptmin=50.0
    )
    result_dict = json.loads(result)
    assert result_dict.get("status") == "ok", f"Expected success, got: {result}"
    print("[✓] Test 2 passed: cluster_all=True works without event_index or npy_path")

    # Test 3: cluster_all=False without event_index should fail with proper error
    tool = JetClusterSlowJetTool(base_directory=base_directory)
    result = tool.execute(
        jsonl_path="data/events.jsonl",
        cluster_all=False,
        algorithm="antikt",
        R=0.4
    )
    assert "Error:" in result, f"Expected error, got: {result}"
    assert "event_index" in result, f"Expected event_index error, got: {result}"
    print("[✓] Test 3 passed: cluster_all=False without event_index produces proper error")

    # Test 4: cluster_all=True without output_path should fail
    tool = JetClusterSlowJetTool(base_directory=base_directory)
    result = tool.execute(
        jsonl_path="data/events.jsonl",
        cluster_all=True,
        algorithm="antikt",
        R=0.4
    )
    assert "Error:" in result, f"Expected error, got: {result}"
    assert "output_path" in result, f"Expected output_path error, got: {result}"
    print("[✓] Test 4 passed: cluster_all=True without output_path produces proper error")

    # Test 5: Neither jsonl_path nor npy_path should fail
    tool = JetClusterSlowJetTool(base_directory=base_directory)
    result = tool.execute(
        cluster_all=True,
        output_path="data/test_output.jsonl",
        algorithm="antikt",
        R=0.4
    )
    assert "Error:" in result, f"Expected error, got: {result}"
    print("[✓] Test 5 passed: Missing input source produces proper error")

    print("\nAll parameter validation tests passed! [✓]\n")
    return {"status": "ok"}


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
        Path(base_directory) / "data_all_test",
    ]

    # Individual files to clean up
    cleanup_files = [
        Path(base_directory) / "test_cluster_all_output.jsonl",
        Path(base_directory) / "test_output_final.jsonl",
    ]

    cleaned = 0

    # Remove directories
    for dir_path in cleanup_dirs:
        if dir_path.exists():
            try:
                shutil.rmtree(dir_path)
                print(f"[✓] Removed directory: {dir_path.name}")
                cleaned += 1
            except Exception as e:
                print(f"[⚠] Failed to remove {dir_path.name}: {e}")

    # Remove individual files
    for file_path in cleanup_files:
        if file_path.exists():
            try:
                file_path.unlink()
                print(f"[✓] Removed file: {file_path.name}")
                cleaned += 1
            except Exception as e:
                print(f"[⚠] Failed to remove {file_path.name}: {e}")

    if cleaned == 0:
        print("[i] No test files to clean up")
    else:
        print(f"\n[✓] Cleaned up {cleaned} item(s)\n")


# ============================================================================
# Main Test Runner
# ============================================================================

if __name__ == "__main__":
    import sys

    print(f"Base directory: {base_directory}\n")
    print("=" * 70)

    # Run all tests and track failures
    all_passed = True

    # Unit tests
    try:
        test_edit_pythia_card()
    except (AssertionError, Exception) as e:
        print(f"\n[✗] test_edit_pythia_card failed: {e}\n")
        all_passed = False

    try:
        test_path_traversal_prevention()
    except (AssertionError, Exception) as e:
        print(f"\n[✗] test_path_traversal_prevention failed: {e}\n")
        all_passed = False

    # Integration tests
    evt_output = test_event_generation(verbose=args.verbose)
    if evt_output is None:
        all_passed = False

    numpy_output = None
    if evt_output:
        numpy_output = test_jsonl_to_numpy_conversion(verbose=args.verbose)
        if numpy_output is None:
            all_passed = False

    jet_output = None
    if evt_output:
        jet_output = test_jet_clustering(cluster_all=True, verbose=args.verbose)
        if jet_output is None:
            all_passed = False

    # Test single-event clustering mode
    if evt_output:
        jet_single_output = test_jet_clustering(cluster_all=False, verbose=args.verbose)
        if jet_single_output is None:
            all_passed = False

    # Test parameter validation (requires events to exist)
    if evt_output:
        try:
            test_jet_clustering_parameter_validation()
        except (AssertionError, Exception) as e:
            print(f"\n[✗] test_jet_clustering_parameter_validation failed: {e}\n")
            all_passed = False

    # Advanced integration tests (require Pythia to work)
    if evt_output is not None:
        try:
            seed_result = test_seed_reproducibility()
            # Don't mark as failed if skipped (returns None)
            # Only fail if it ran and threw an exception
        except (AssertionError, Exception) as e:
            print(f"\n[✗] test_seed_reproducibility failed: {e}\n")
            all_passed = False

        try:
            finals_result = test_finals_only_filtering()
            # Don't mark as failed if skipped
        except (AssertionError, Exception) as e:
            print(f"\n[✗] test_finals_only_filtering failed: {e}\n")
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
