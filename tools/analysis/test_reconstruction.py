#!/usr/bin/env python3
"""
# test_reconstruction.py is a part of the HEPTAPOD package.
# Copyright (C) 2025 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.
"""
"""
Comprehensive unit tests for ResonanceReconstructionTool.

Tests cover:
- Core invariant mass calculation
- Array loading and merging (both particles and jets keys)
- two_body_symmetric template (leptoquark analysis)
- n_body_all_pairs template
- Histogramming
- File I/O (JSONL)
- Error handling
- Integration with GetHardestNTool workflow
"""

import os
import sys
from pathlib import Path
import json
import numpy as np
import tempfile
import shutil

SCRIPT_PATH = Path(__file__).resolve()
ANALYSIS_DIR = SCRIPT_PATH.parent                             # .../heptapod/tools/analysis
TOOLS_DIR = ANALYSIS_DIR.parent                               # .../heptapod/tools
REPO_ROOT = TOOLS_DIR.parent                                  # .../heptapod

# Add repository root to path for local imports (prompts, tools, etc.)
sys.path.insert(0, str(REPO_ROOT))

from tools.analysis.reconstruction import ResonanceReconstructionTool
from tools.analysis.kinematics import GetHardestNTool

# Initialize test directory
test_base_dir = str(ANALYSIS_DIR / "test_files_reconstruction")

# Global flag for keeping test files (set by main)
_keep_files = False


# ====================================================================== #
# ====================== Core Function Tests =========================== #
# ====================================================================== #

def test_invariant_mass_calculation():
    """Test core invariant mass calculation."""
    print(">> Testing invariant mass calculation...\n")

    tool = ResonanceReconstructionTool(
        base_directory=test_base_dir,
        particle_arrays=["dummy.jsonl"],
        template="two_body_symmetric"
    )
    tool._setup()

    # Test 1: Two particles back-to-back (should give 100 GeV)
    four_vecs = np.array([
        [50.0, 0.0, 0.0, 50.0],
        [-50.0, 0.0, 0.0, 50.0]
    ])
    mass = tool._calculate_invariant_mass(four_vecs)
    assert abs(mass - 100.0) < 1e-6, f"Expected 100, got {mass}"
    print(f"[✓] Test 1 passed: Two back-to-back particles = {mass:.2f} GeV")

    # Test 2: Single particle at rest (should give rest mass)
    four_vecs = np.array([[0.0, 0.0, 0.0, 125.0]])
    mass = tool._calculate_invariant_mass(four_vecs)
    assert abs(mass - 125.0) < 1e-6
    print(f"[✓] Test 2 passed: Particle at rest = {mass:.2f} GeV")

    # Test 3: Empty array (should give 0)
    four_vecs = np.array([]).reshape(0, 4)
    mass = tool._calculate_invariant_mass(four_vecs)
    assert mass == 0.0
    print("[✓] Test 3 passed: Empty array = 0.0 GeV")

    # Test 4: Four particles (leptoquark-like)
    four_vecs = np.array([
        [100.0, 0.0, 0.0, 100.0],
        [50.0, 0.0, 0.0, 50.0],
        [-100.0, 0.0, 0.0, 100.0],
        [-50.0, 0.0, 0.0, 50.0]
    ])
    mass = tool._calculate_invariant_mass(four_vecs)
    # Total 4-momentum: (0, 0, 0, 300) -> M = 300 GeV
    assert abs(mass - 300.0) < 1e-6
    print(f"[✓] Test 4 passed: Four particles = {mass:.2f} GeV")

    print("\nAll invariant mass calculation tests passed! [✓]\n")


# ====================================================================== #
# =================== Array Loading Tests ============================== #
# ====================================================================== #

def test_load_and_merge_arrays():
    """Test loading and merging multiple particle arrays."""
    print(">> Testing array loading and merging...\n")

    # Create test directory
    test_dir = Path(test_base_dir) / "test_merge_arrays"
    test_dir.mkdir(parents=True, exist_ok=True)

    # Create first array (leptons with 'particles' key)
    leptons_path = test_dir / "leptons.jsonl"
    leptons = [
        {
            "schema": "evtjsonl-1.0",
            "event_id": 0,
            "data": {
                "n_particles": 2,
                "particles": [
                    {"i": 0, "id": 11, "px": 100.0, "py": 0.0, "pz": 10.0, "E": 101.0, "m": 0.0},
                    {"i": 1, "id": -11, "px": -100.0, "py": 0.0, "pz": -10.0, "E": 101.0, "m": 0.0}
                ]
            }
        }
    ]
    with open(leptons_path, 'w') as f:
        for ev in leptons:
            f.write(json.dumps(ev) + "\n")

    # Create second array (jets with 'jets' key)
    jets_path = test_dir / "jets.jsonl"
    jets = [
        {
            "schema": "evtjsonl-1.0",
            "event_id": 0,
            "data": {
                "n_jets": 2,
                "jets": [
                    {"index": 0, "px": 50.0, "py": 0.0, "pz": 5.0, "E": 51.0, "m": 0.0},
                    {"index": 1, "px": -50.0, "py": 0.0, "pz": -5.0, "E": 51.0, "m": 0.0}
                ]
            }
        }
    ]
    with open(jets_path, 'w') as f:
        for ev in jets:
            f.write(json.dumps(ev) + "\n")

    # Test: Load and merge both arrays
    tool = ResonanceReconstructionTool(
        base_directory=str(test_base_dir),
        particle_arrays=[
            "test_merge_arrays/leptons.jsonl",
            "test_merge_arrays/jets.jsonl"
        ],
        template="two_body_symmetric"
    )
    tool._setup()

    events = tool._load_and_merge_arrays([str(leptons_path), str(jets_path)])

    assert len(events) == 1, f"Expected 1 event, got {len(events)}"
    assert len(events[0]["objects"]) == 4, f"Expected 4 objects, got {len(events[0]['objects'])}"
    print("[✓] Test 1 passed: Successfully merged 2 leptons + 2 jets = 4 objects")

    # Verify 4-vectors are correct
    objects = events[0]["objects"]
    assert objects.shape == (4, 4), f"Expected shape (4,4), got {objects.shape}"
    print("[✓] Test 2 passed: Object array has correct shape")

    print("\nAll array loading tests passed! [✓]\n")


def test_particles_and_jets_keys():
    """Test that tool handles both 'particles' and 'jets' keys."""
    print(">> Testing 'particles' and 'jets' key handling...\n")

    test_dir = Path(test_base_dir) / "test_keys"
    test_dir.mkdir(parents=True, exist_ok=True)

    # Array 1: Uses 'particles' key
    array1_path = test_dir / "array1.jsonl"
    with open(array1_path, 'w') as f:
        f.write(json.dumps({
            "schema": "evtjsonl-1.0",
            "event_id": 0,
            "data": {
                "n_particles": 2,
                "particles": [
                    {"i": 0, "id": 11, "px": 50.0, "py": 0.0, "pz": 0.0, "E": 50.0, "m": 0.0},
                    {"i": 1, "id": -11, "px": -50.0, "py": 0.0, "pz": 0.0, "E": 50.0, "m": 0.0}
                ]
            }
        }) + "\n")

    # Array 2: Uses 'jets' key
    array2_path = test_dir / "array2.jsonl"
    with open(array2_path, 'w') as f:
        f.write(json.dumps({
            "schema": "evtjsonl-1.0",
            "event_id": 0,
            "data": {
                "n_jets": 2,
                "jets": [
                    {"index": 0, "px": 30.0, "py": 0.0, "pz": 0.0, "E": 30.0, "m": 0.0},
                    {"index": 1, "px": -30.0, "py": 0.0, "pz": 0.0, "E": 30.0, "m": 0.0}
                ]
            }
        }) + "\n")

    # Test: Merge both
    tool = ResonanceReconstructionTool(
        base_directory=str(test_base_dir),
        particle_arrays=[
            "test_keys/array1.jsonl",
            "test_keys/array2.jsonl"
        ],
        template="two_body_symmetric"
    )
    tool._setup()

    events = tool._load_and_merge_arrays([str(array1_path), str(array2_path)])

    assert len(events) == 1
    assert len(events[0]["objects"]) == 4
    print("[✓] Test passed: Tool correctly handles mixed 'particles' and 'jets' keys")

    print("\nKey handling test passed! [✓]\n")


# ====================================================================== #
# ==================== Template Tests ================================== #
# ====================================================================== #

def test_two_body_symmetric_template():
    """Test two_body_symmetric template (leptoquark analysis)."""
    print(">> Testing two_body_symmetric template...\n")

    # Create test directory
    test_dir = Path(test_base_dir) / "test_two_body_symmetric"
    test_dir.mkdir(parents=True, exist_ok=True)

    # Create two pre-selected arrays
    # Array 1: 2 leptons
    leptons_path = test_dir / "leptons.jsonl"
    with open(leptons_path, 'w') as f:
        f.write(json.dumps({
            "schema": "evtjsonl-1.0",
            "event_id": 0,
            "data": {
                "n_particles": 2,
                "particles": [
                    {"i": 0, "id": -11, "px": 300.0, "py": 0.0, "pz": 0.0, "E": 300.0, "m": 0.0},
                    {"i": 1, "id": 11, "px": -300.0, "py": 0.0, "pz": 0.0, "E": 300.0, "m": 0.0}
                ]
            }
        }) + "\n")

    # Array 2: 2 jets
    jets_path = test_dir / "jets.jsonl"
    with open(jets_path, 'w') as f:
        f.write(json.dumps({
            "schema": "evtjsonl-1.0",
            "event_id": 0,
            "data": {
                "n_jets": 2,
                "jets": [
                    {"index": 0, "px": 300.0, "py": 0.0, "pz": 0.0, "E": 300.0, "m": 0.0},
                    {"index": 1, "px": -300.0, "py": 0.0, "pz": 0.0, "E": 300.0, "m": 0.0}
                ]
            }
        }) + "\n")

    # Run analysis
    tool = ResonanceReconstructionTool(
        base_directory=str(test_base_dir),
        particle_arrays=[
            "test_two_body_symmetric/leptons.jsonl",
            "test_two_body_symmetric/jets.jsonl"
        ],
        template="two_body_symmetric",
        output_prefix="test_two_body_symmetric/output"
    )
    tool._setup()
    result_str = tool._run()
    result = json.loads(result_str)

    assert result["status"] == "ok"
    assert result["template"] == "two_body_symmetric"
    assert result["n_arrays"] == 2
    print(f"[✓] Test 1 passed: Analysis completed with {result['n_arrays']} arrays")

    # Check observables
    observable_names = [obs["name"] for obs in result["observables"]]
    assert "m1" in observable_names
    assert "m2" in observable_names
    assert "m_min" in observable_names
    assert "m_max" in observable_names
    print(f"[✓] Test 2 passed: Observables = {observable_names}")

    # Check histograms created
    assert len(result["histograms"]) == 4
    print(f"[✓] Test 3 passed: {len(result['histograms'])} histograms created")

    # Check data files saved
    assert "m1" in result["data_paths"]
    m1_path = Path(test_base_dir) / result["data_paths"]["m1"]
    assert m1_path.exists()
    print("[✓] Test 4 passed: Data files saved")

    # Load and check masses
    m1_data = np.load(m1_path)
    m2_data = np.load(Path(test_base_dir) / result["data_paths"]["m2"])

    # The pairing algorithm should find (e+, e-) + (j, j) → m1=600, m2=600
    assert len(m1_data) == 1
    assert abs(m1_data[0] - m2_data[0]) < 1.0  # Should be symmetric
    assert abs(m1_data[0] - 600.0) < 1.0  # Should reconstruct 600 GeV
    print(f"[✓] Test 5 passed: Leptoquark masses m1={m1_data[0]:.1f}, m2={m2_data[0]:.1f} GeV")

    print("\nAll two_body_symmetric tests passed! [✓]\n")


def test_n_body_all_pairs_template():
    """Test n_body_all_pairs template."""
    print(">> Testing n_body_all_pairs template...\n")

    # Create test directory
    test_dir = Path(test_base_dir) / "test_n_body_all_pairs"
    test_dir.mkdir(parents=True, exist_ok=True)

    # Create single array with 4 particles
    array_path = test_dir / "particles.jsonl"
    with open(array_path, 'w') as f:
        f.write(json.dumps({
            "schema": "evtjsonl-1.0",
            "event_id": 0,
            "data": {
                "n_particles": 4,
                "particles": [
                    {"i": 0, "id": 11, "px": 50.0, "py": 0.0, "pz": 0.0, "E": 50.0, "m": 0.0},
                    {"i": 1, "id": -11, "px": -50.0, "py": 0.0, "pz": 0.0, "E": 50.0, "m": 0.0},
                    {"i": 2, "id": 13, "px": 30.0, "py": 0.0, "pz": 0.0, "E": 30.0, "m": 0.0},
                    {"i": 3, "id": -13, "px": -30.0, "py": 0.0, "pz": 0.0, "E": 30.0, "m": 0.0}
                ]
            }
        }) + "\n")

    # Run analysis with max_k=3
    tool = ResonanceReconstructionTool(
        base_directory=str(test_base_dir),
        particle_arrays=["test_n_body_all_pairs/particles.jsonl"],
        template="n_body_all_pairs",
        max_k=3,
        output_prefix="test_n_body_all_pairs/output"
    )
    tool._setup()
    result_str = tool._run()
    result = json.loads(result_str)

    assert result["status"] == "ok"

    # Check observables: m_k2, m_k3
    observable_names = [obs["name"] for obs in result["observables"]]
    assert "m_k2" in observable_names
    assert "m_k3" in observable_names
    print(f"[✓] Test 1 passed: Observables = {observable_names}")

    # Check number of combinations
    # For k=2: C(4,2) = 6 combinations per event
    # For k=3: C(4,3) = 4 combinations per event
    m_k2_data = np.load(Path(test_base_dir) / result["data_paths"]["m_k2"])
    m_k3_data = np.load(Path(test_base_dir) / result["data_paths"]["m_k3"])

    assert len(m_k2_data) == 6
    assert len(m_k3_data) == 4
    print(f"[✓] Test 2 passed: k=2 gives {len(m_k2_data)} combinations, k=3 gives {len(m_k3_data)}")

    print("\nAll n_body_all_pairs tests passed! [✓]\n")


# ====================================================================== #
# ====================== Integration Tests ============================= #
# ====================================================================== #

def test_integration_with_gethardestntool():
    """Integration test: GetHardestNTool → ResonanceReconstructionTool workflow."""
    print(">> Testing GetHardestNTool → ResonanceReconstructionTool integration...\n")

    # Create temp directory for this test
    tmpdir = tempfile.mkdtemp()

    try:
        # ============ Step 1: Create sample event data ============

        events_path = os.path.join(tmpdir, "events.jsonl")
        with open(events_path, 'w') as f:
            # Event with leptons and quarks
            for i in range(3):
                ev = {
                    "schema": "evtjsonl-1.0",
                    "event_id": i,
                    "data": {
                        "n_particles": 6,
                        "particles": [
                            {"i": 0, "id": 11, "px": 100.0 + i*10, "py": 0.0, "pz": 0.0, "E": 100.0 + i*10, "m": 0.0},
                            {"i": 1, "id": -11, "px": -90.0 - i*10, "py": 0.0, "pz": 0.0, "E": 90.0 + i*10, "m": 0.0},
                            {"i": 2, "id": 13, "px": 50.0, "py": 0.0, "pz": 0.0, "E": 50.0, "m": 0.0},  # Lower pT muon
                            {"i": 3, "id": 2, "px": 80.0 + i*5, "py": 0.0, "pz": 0.0, "E": 80.0 + i*5, "m": 0.0},
                            {"i": 4, "id": -2, "px": -70.0 - i*5, "py": 0.0, "pz": 0.0, "E": 70.0 + i*5, "m": 0.0},
                            {"i": 5, "id": 1, "px": 40.0, "py": 0.0, "pz": 0.0, "E": 40.0, "m": 0.0}  # Lower pT quark
                        ]
                    }
                }
                f.write(json.dumps(ev) + "\n")

        # ============ Step 2: Get hardest 2 leptons ============

        leptons_path = os.path.join(tmpdir, "hardest_2_leptons.jsonl")

        lepton_tool = GetHardestNTool(
            input_path=events_path,
            pdgids=[11, -11, 13, -13],
            n_hardest=2,
            output_path=leptons_path
        )
        lepton_tool.base_directory = tmpdir

        lepton_result = json.loads(lepton_tool._run())
        print(f"Lepton selection: {lepton_result['status']}")
        assert lepton_result['status'] == 'ok'
        assert os.path.exists(leptons_path)

        # Debug: check what's in the lepton file
        with open(leptons_path, 'r') as f:
            lep_events = [json.loads(line) for line in f]
        print(f"  Lepton file has {len(lep_events)} events")
        if len(lep_events) > 0:
            print(f"  Event 0 has {lep_events[0]['data']['n_particles']} particles")
        print("  ✓ Extracted hardest 2 leptons\n")

        # ============ Step 3: Get hardest 2 quarks ============

        quarks_path = os.path.join(tmpdir, "hardest_2_quarks.jsonl")

        quark_tool = GetHardestNTool(
            input_path=events_path,
            pdgids=[1, 2, 3, 4, 5, -1, -2, -3, -4, -5],
            n_hardest=2,
            output_path=quarks_path
        )
        quark_tool.base_directory = tmpdir

        quark_result = json.loads(quark_tool._run())
        print(f"Quark selection: {quark_result['status']}")
        assert quark_result['status'] == 'ok'
        assert os.path.exists(quarks_path)

        # Debug: check what's in the quark file
        with open(quarks_path, 'r') as f:
            quark_events = [json.loads(line) for line in f]
        print(f"  Quark file has {len(quark_events)} events")
        if len(quark_events) > 0:
            print(f"  Event 0 has {quark_events[0]['data']['n_particles']} particles")
        print("  ✓ Extracted hardest 2 quarks\n")

        # ============ Step 4: Reconstruct leptoquarks ============

        output_prefix = os.path.join(tmpdir, "leptoquark_reco")

        # Debug: manually check what ResonanceReconstructionTool will load
        print(f"About to reconstruct with:")
        print(f"  Leptons: {leptons_path}")
        print(f"  Quarks: {quarks_path}")

        reco_tool = ResonanceReconstructionTool(
            particle_arrays=[leptons_path, quarks_path],
            template="two_body_symmetric",
            output_prefix=output_prefix
        )
        reco_tool.base_directory = tmpdir

        # Debug: try loading arrays manually
        print("  Loading arrays manually...")
        events = reco_tool._load_and_merge_arrays([leptons_path, quarks_path])
        print(f"  Event 0: {len(events[0]['objects'])} objects, array_indices: {events[0]['array_indices']}")

        reco_result = json.loads(reco_tool._run())
        print(f"Reconstruction: {reco_result['status']}")
        print(f"  Arrays: {reco_result['n_arrays']}")
        print(f"  Events: {reco_result['n_events_analyzed']}")
        print(f"  Observables: {[obs['name'] for obs in reco_result['observables']]}")

        assert reco_result['status'] == 'ok'
        assert reco_result['n_arrays'] == 2
        assert reco_result['n_events_analyzed'] == 3

        # Check output files
        for obs in reco_result['observables']:
            obs_path = os.path.join(tmpdir, reco_result['data_paths'][obs['name']])
            assert os.path.exists(obs_path), f"Output file for {obs['name']} not created"

        print("  ✓ Reconstruction successful\n")

        # ============ Step 5: Verify physics results ============

        m1_path = os.path.join(tmpdir, reco_result['data_paths']['m1'])
        m2_path = os.path.join(tmpdir, reco_result['data_paths']['m2'])

        m1_values = np.load(m1_path)
        m2_values = np.load(m2_path)

        print(f"  Reconstructed masses (first 3 events):")
        for i in range(3):
            print(f"    Event {i}: m1={m1_values[i]:.2f} GeV, m2={m2_values[i]:.2f} GeV")

        # Check that masses are non-negative (0 is valid for back-to-back massless particles)
        assert np.all(m1_values >= 0), "All m1 values should be non-negative"
        assert np.all(m2_values >= 0), "All m2 values should be non-negative"

        # Check that reconstruction ran successfully
        assert reco_result['n_events_analyzed'] == 3, "Should have analyzed 3 events"

        print("  ✓ Physics results validated\n")
        print("Integration test passed! [✓]\n")

    finally:
        # Cleanup
        if not _keep_files:
            shutil.rmtree(tmpdir)


# ====================================================================== #
# ====================== Error Handling Tests ========================== #
# ====================================================================== #

def test_error_handling():
    """Test error handling."""
    print(">> Testing error handling...\n")

    test_dir = Path(test_base_dir) / "test_errors"
    test_dir.mkdir(parents=True, exist_ok=True)

    # Test 1: Invalid template
    tool = ResonanceReconstructionTool(
        base_directory=str(test_base_dir),
        particle_arrays=["test_errors/dummy.jsonl"],
        template="invalid_template"
    )
    tool._setup()
    result = tool._run()
    assert "error" in result.lower()
    print("[✓] Test 1 passed: Invalid template rejected")

    # Test 2: Empty particle_arrays list
    tool = ResonanceReconstructionTool(
        base_directory=str(test_base_dir),
        particle_arrays=[],
        template="two_body_symmetric"
    )
    tool._setup()
    result = tool._run()
    assert "error" in result.lower()
    print("[✓] Test 2 passed: Empty particle_arrays rejected")

    # Test 3: File not found
    tool = ResonanceReconstructionTool(
        base_directory=str(test_base_dir),
        particle_arrays=["test_errors/nonexistent.jsonl"],
        template="two_body_symmetric"
    )
    tool._setup()
    result = tool._run()
    assert "error" in result.lower()
    print("[✓] Test 3 passed: File not found error")

    # Test 4: Path traversal prevention
    tool = ResonanceReconstructionTool(
        base_directory=str(test_base_dir),
        particle_arrays=["../../../etc/passwd"],
        template="two_body_symmetric"
    )
    tool._setup()
    result = tool._run()
    assert "error" in result.lower() or "denied" in result.lower()
    print("[✓] Test 4 passed: Path traversal prevented")

    # Test 5: Mismatched event counts
    # Create two arrays with different numbers of events
    array1_path = test_dir / "array1_mismatch.jsonl"
    with open(array1_path, 'w') as f:
        f.write(json.dumps({
            "schema": "evtjsonl-1.0",
            "event_id": 0,
            "data": {"n_particles": 1, "particles": [{"i": 0, "id": 11, "px": 1.0, "py": 0.0, "pz": 0.0, "E": 1.0, "m": 0.0}]}
        }) + "\n")
        f.write(json.dumps({
            "schema": "evtjsonl-1.0",
            "event_id": 1,
            "data": {"n_particles": 1, "particles": [{"i": 0, "id": 11, "px": 1.0, "py": 0.0, "pz": 0.0, "E": 1.0, "m": 0.0}]}
        }) + "\n")

    array2_path = test_dir / "array2_mismatch.jsonl"
    with open(array2_path, 'w') as f:
        f.write(json.dumps({
            "schema": "evtjsonl-1.0",
            "event_id": 0,
            "data": {"n_particles": 1, "particles": [{"i": 0, "id": 11, "px": 1.0, "py": 0.0, "pz": 0.0, "E": 1.0, "m": 0.0}]}
        }) + "\n")

    tool = ResonanceReconstructionTool(
        base_directory=str(test_base_dir),
        particle_arrays=[
            "test_errors/array1_mismatch.jsonl",
            "test_errors/array2_mismatch.jsonl"
        ],
        template="two_body_symmetric"
    )
    tool._setup()
    result = tool._run()
    assert "error" in result.lower()
    print("[✓] Test 5 passed: Mismatched event counts rejected")

    print("\nAll error handling tests passed! [✓]\n")


# ====================================================================== #
# ============================= Main Runner ============================ #
# ====================================================================== #

def main():
    """Run all tests."""
    import argparse

    # Declare global before any use
    global _keep_files

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Test suite for resonance reconstruction tool")
    parser.add_argument("--keep-files", action="store_true",
                        help="Keep test-generated files after tests complete")
    args = parser.parse_args()

    # Set global flag
    _keep_files = args.keep_files

    # Create base test directory
    os.makedirs(test_base_dir, exist_ok=True)

    try:
        # Core function tests
        test_invariant_mass_calculation()

        # Array loading tests
        test_load_and_merge_arrays()
        test_particles_and_jets_keys()

        # Template tests
        test_two_body_symmetric_template()
        test_n_body_all_pairs_template()

        # Integration tests
        test_integration_with_gethardestntool()

        # Error handling tests
        test_error_handling()

        print()
        print("=" * 60)
        print("ResonanceReconstructionTool test suite completed successfully! [✓]")
        print("=" * 60)

        # Cleanup test directory
        if not _keep_files:
            shutil.rmtree(test_base_dir, ignore_errors=True)

    except AssertionError as e:
        print("\n" + "=" * 60)
        print("Test suite completed with failures! [✗]")
        print("=" * 60)
        print(f"Assertion error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    except Exception as e:
        print("\n" + "=" * 60)
        print("Test suite completed with failures! [✗]")
        print("=" * 60)
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
    # If we reach here, all tests passed (main() exits with 1 on failure)
    sys.exit(0)
