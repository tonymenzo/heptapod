#!/usr/bin/env python3
"""
# test_kinematics.py is a part of the HEPTAPOD package.
# Copyright (C) 2025 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.
"""
"""
Analysis Tools Test Suite

Tests cover:
- Kinematic calculations (invariant mass, pT, Delta R)
- Event selection tools (hardest-N, PDG ID filtering, sorting)
- Cuts application (pT, eta, PDG ID filters)
- Path traversal prevention (security)
- Multiple input formats (particles list, NumPy, JSONL)
"""

# Standard library imports
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

# Third-party imports
import numpy as np

# Path setup
SCRIPT_PATH = Path(__file__).resolve()
ANALYSIS_DIR = SCRIPT_PATH.parent                             # .../heptapod-dev/tools/analysis
TOOLS_DIR = ANALYSIS_DIR.parent                               # .../heptapod-dev/tools
REPO_ROOT = TOOLS_DIR.parent                                  # .../heptapod-dev

# Add repository root to path for local imports (prompts, tools, etc.)
sys.path.insert(0, str(REPO_ROOT))

# Tool imports
from tools.analysis.kinematics import (
    CalculateInvariantMassTool,
    CalculateTransverseMomentumTool,
    CalculateDeltaRTool,
    ApplyCutsTool,
    GetHardestNTool,
    GetHardestNJetsTool,
    FilterByPDGIDTool,
    SortByPtTool
)

# Future: When orchestral-ai PyPI package is fixed, use these imports instead:
# from orchestral.tools import (
#     CalculateInvariantMassTool,
#     CalculateTransverseMomentumTool,
#     CalculateDeltaRTool,
#     ApplyCutsTool,
#     GetHardestNTool,
#     GetHardestNJetsTool,
#     FilterByPDGIDTool,
#     SortByPtTool
# )

# Initialize test directory
test_base_dir = str(ANALYSIS_DIR / "test_files_kinematics")

# Global flag for keeping test files (set by main)
_keep_files = False


# ============================================================================
# Test Functions
# ============================================================================

# ====================================================================== #
# =================== Kinematics Tools Tests =========================== #
# ====================================================================== #

def test_invariant_mass_from_particles():
    """Test invariant mass calculation from direct particle list."""
    print(">> Testing invariant mass calculation from particle list...\n")

    # Test 1: Z boson decay to two leptons (Z -> l+ l-)
    # Use known Z mass ~ 91.2 GeV
    # Simple symmetric case: two particles back-to-back
    tool = CalculateInvariantMassTool(
        base_directory=test_base_dir,
        particles=[
            [50.0, 0.0, 0.0, 50.0],   # particle 1: px=50, py=0, pz=0, E=50
            [-50.0, 0.0, 0.0, 50.0],  # particle 2: px=-50, py=0, pz=0, E=50
        ]
    )
    tool._setup()
    result = tool._run()
    result_dict = json.loads(result)

    assert result_dict["status"] == "ok"
    # Total 4-momentum: (0, 0, 0, 100) -> M = 100 GeV
    expected_mass = 100.0
    assert abs(result_dict["invariant_mass"] - expected_mass) < 1e-6
    assert result_dict["n_particles"] == 2
    print(f"[✓] Test 1 passed: Two-particle invariant mass = {result_dict['invariant_mass']:.2f} GeV")

    # Test 2: Single particle (should give particle's rest mass)
    # particle with px=0, py=0, pz=0, E=125 -> M = 125 (Higgs-like)
    tool = CalculateInvariantMassTool(
        base_directory=test_base_dir,
        particles=[[0.0, 0.0, 0.0, 125.0]]
    )
    tool._setup()
    result = tool._run()
    result_dict = json.loads(result)

    assert result_dict["status"] == "ok"
    assert abs(result_dict["invariant_mass"] - 125.0) < 1e-6
    print(f"[✓] Test 2 passed: Single particle rest mass = {result_dict['invariant_mass']:.2f} GeV")

    # Test 3: Three particles (top decay: t -> W b, W -> l nu)
    tool = CalculateInvariantMassTool(
        base_directory=test_base_dir,
        particles=[
            [40.0, 30.0, 20.0, 60.0],
            [30.0, -20.0, 10.0, 50.0],
            [-10.0, 5.0, -5.0, 20.0],
        ]
    )
    tool._setup()
    result = tool._run()
    result_dict = json.loads(result)

    assert result_dict["status"] == "ok"
    assert result_dict["n_particles"] == 3
    # Just check it's positive and reasonable
    assert result_dict["invariant_mass"] > 0
    print(f"[✓] Test 3 passed: Three-particle invariant mass = {result_dict['invariant_mass']:.2f} GeV")

    # Test 4: Empty particle list
    tool = CalculateInvariantMassTool(
        base_directory=test_base_dir,
        particles=[]
    )
    tool._setup()
    result = tool._run()
    result_dict = json.loads(result)

    assert result_dict["status"] == "ok"
    assert result_dict["invariant_mass"] == 0.0
    print("[✓] Test 4 passed: Empty particle list returns mass = 0")

    print("\nAll invariant mass tests passed! [✓]\n")


def test_invariant_mass_from_numpy():
    """Test invariant mass calculation from NumPy array."""
    print(">> Testing invariant mass calculation from NumPy array...\n")

    # Create test directory
    test_dir = Path(test_base_dir) / "test_invariant_mass_numpy"
    test_dir.mkdir(parents=True, exist_ok=True)

    # Test 1: Single event (2D array)
    npy_path = test_dir / "single_event.npy"
    particles = np.array([
        [50.0, 0.0, 0.0, 50.0, 0.0],   # [px, py, pz, E, pdgid]
        [-50.0, 0.0, 0.0, 50.0, 0.0],
    ])
    np.save(str(npy_path), particles)

    tool = CalculateInvariantMassTool(
        base_directory=str(test_base_dir),
        npy_path="test_invariant_mass_numpy/single_event.npy"
    )
    tool._setup()
    result = tool._run()
    result_dict = json.loads(result)

    assert result_dict["status"] == "ok"
    assert abs(result_dict["invariant_mass"] - 100.0) < 1e-6
    print(f"[✓] Test 1 passed: Single event mass = {result_dict['invariant_mass']:.2f} GeV")

    # Test 2: Multiple events (3D array)
    npy_path = test_dir / "multiple_events.npy"
    events = np.array([
        # Event 0
        [[50.0, 0.0, 0.0, 50.0, 0.0],
         [-50.0, 0.0, 0.0, 50.0, 0.0],
         [0.0, 0.0, 0.0, 0.0, 0.0]],  # padding
        # Event 1
        [[30.0, 40.0, 0.0, 50.0, 0.0],
         [-30.0, -40.0, 0.0, 50.0, 0.0],
         [0.0, 0.0, 0.0, 0.0, 0.0]],
    ])
    np.save(str(npy_path), events)

    tool = CalculateInvariantMassTool(
        base_directory=str(test_base_dir),
        npy_path="test_invariant_mass_numpy/multiple_events.npy"
    )
    tool._setup()
    result = tool._run()
    result_dict = json.loads(result)

    assert result_dict["status"] == "ok"
    assert result_dict["n_events"] == 2
    assert len(result_dict["invariant_masses"]) == 2
    assert all(m == 100.0 for m in result_dict["invariant_masses"])
    print(f"[✓] Test 2 passed: Multiple events, masses = {result_dict['invariant_masses']}")

    # Test 3: Specific event index
    tool = CalculateInvariantMassTool(
        base_directory=str(test_base_dir),
        npy_path="test_invariant_mass_numpy/multiple_events.npy",
        event_index=0
    )
    tool._setup()
    result = tool._run()
    result_dict = json.loads(result)

    assert result_dict["status"] == "ok"
    assert result_dict["event_index"] == 0
    assert abs(result_dict["invariant_mass"] - 100.0) < 1e-6
    print(f"[✓] Test 3 passed: Event index 0 mass = {result_dict['invariant_mass']:.2f} GeV")

    print("\nAll NumPy invariant mass tests passed! [✓]\n")


def test_invariant_mass_from_jsonl():
    """Test invariant mass calculation from JSONL."""
    print(">> Testing invariant mass calculation from JSONL...\n")

    # Create test directory
    test_dir = Path(test_base_dir) / "test_invariant_mass_jsonl"
    test_dir.mkdir(parents=True, exist_ok=True)

    # Create test JSONL file
    jsonl_path = test_dir / "events.jsonl"
    events = [
        {
            "schema": "evtjsonl-1.0",
            "event_id": 0,
            "finals_only": True,
            "full_history": False,
            "data": {
                "n": 2,
                "particles": [
                    {"i": 0, "id": 11, "px": 50.0, "py": 0.0, "pz": 0.0, "E": 50.0, "m": 0.0},
                    {"i": 1, "id": -11, "px": -50.0, "py": 0.0, "pz": 0.0, "E": 50.0, "m": 0.0},
                ]
            }
        },
        {
            "schema": "evtjsonl-1.0",
            "event_id": 1,
            "finals_only": True,
            "full_history": False,
            "data": {
                "n": 4,
                "particles": [
                    {"i": 0, "id": 1, "px": 25.0, "py": 0.0, "pz": 0.0, "E": 25.0, "m": 0.0},
                    {"i": 1, "id": -1, "px": -25.0, "py": 0.0, "pz": 0.0, "E": 25.0, "m": 0.0},
                    {"i": 2, "id": 21, "px": 10.0, "py": 0.0, "pz": 0.0, "E": 10.0, "m": 0.0},
                    {"i": 3, "id": 21, "px": -10.0, "py": 0.0, "pz": 0.0, "E": 10.0, "m": 0.0},
                ]
            }
        }
    ]

    with open(jsonl_path, 'w') as f:
        for ev in events:
            f.write(json.dumps(ev) + "\n")

    # Test 1: All events
    tool = CalculateInvariantMassTool(
        base_directory=str(test_base_dir),
        jsonl_path="test_invariant_mass_jsonl/events.jsonl"
    )
    tool._setup()
    result = tool._run()
    result_dict = json.loads(result)

    assert result_dict["status"] == "ok"
    assert result_dict["n_events"] == 2
    assert abs(result_dict["invariant_masses"][0] - 100.0) < 1e-6
    assert abs(result_dict["invariant_masses"][1] - 70.0) < 1e-6
    print(f"[✓] Test 1 passed: All events, masses = {result_dict['invariant_masses']}")

    # Test 2: Filter by PDG ID (electrons only)
    tool = CalculateInvariantMassTool(
        base_directory=str(test_base_dir),
        jsonl_path="test_invariant_mass_jsonl/events.jsonl",
        pdgids=[11, -11],
        event_index=0
    )
    tool._setup()
    result = tool._run()
    result_dict = json.loads(result)

    assert result_dict["status"] == "ok"
    assert abs(result_dict["invariant_mass"] - 100.0) < 1e-6
    assert result_dict["n_particles"] == 2
    print(f"[✓] Test 2 passed: PDG ID filter, mass = {result_dict['invariant_mass']:.2f} GeV")

    # Test 3: Filter by PDG ID (quarks + gluons in event 1)
    tool = CalculateInvariantMassTool(
        base_directory=str(test_base_dir),
        jsonl_path="test_invariant_mass_jsonl/events.jsonl",
        pdgids=[1, -1, 21],
        event_index=1
    )
    tool._setup()
    result = tool._run()
    result_dict = json.loads(result)

    assert result_dict["status"] == "ok"
    assert abs(result_dict["invariant_mass"] - 70.0) < 1e-6
    assert result_dict["n_particles"] == 4
    print(f"[✓] Test 3 passed: Quarks + gluons, mass = {result_dict['invariant_mass']:.2f} GeV")

    print("\nAll JSONL invariant mass tests passed! [✓]\n")


def test_transverse_momentum():
    """Test pT calculation."""
    print(">> Testing transverse momentum calculation...\n")

    # Test 1: Single particle
    tool = CalculateTransverseMomentumTool(
        base_directory=test_base_dir,
        particles=[3.0, 4.0, 5.0, 10.0]  # pT should be sqrt(9+16) = 5
    )
    tool._setup()
    result = tool._run()
    result_dict = json.loads(result)

    assert result_dict["status"] == "ok"
    assert abs(result_dict["pt"] - 5.0) < 1e-6
    print(f"[✓] Test 1 passed: Single particle pT = {result_dict['pt']:.2f} GeV")

    # Test 2: Multiple particles
    tool = CalculateTransverseMomentumTool(
        base_directory=test_base_dir,
        particles=[
            [3.0, 4.0, 5.0, 10.0],   # pT = 5
            [6.0, 8.0, 0.0, 15.0],   # pT = 10
            [0.0, 0.0, 100.0, 100.0], # pT = 0
        ]
    )
    tool._setup()
    result = tool._run()
    result_dict = json.loads(result)

    assert result_dict["status"] == "ok"
    assert result_dict["n_particles"] == 3
    assert abs(result_dict["pt_values"][0] - 5.0) < 1e-6
    assert abs(result_dict["pt_values"][1] - 10.0) < 1e-6
    assert abs(result_dict["pt_values"][2] - 0.0) < 1e-6
    print(f"[✓] Test 2 passed: Multiple particles pT = {result_dict['pt_values']}")

    print("\nAll pT tests passed! [✓]\n")


def test_delta_r():
    """Test Delta R calculation."""
    print(">> Testing Delta R calculation...\n")

    # Test 1: Same particle (Delta R = 0)
    tool = CalculateDeltaRTool(
        base_directory=test_base_dir,
        particle1=[10.0, 20.0, 30.0, 40.0],
        particle2=[10.0, 20.0, 30.0, 40.0]
    )
    result = tool._run()
    result_dict = json.loads(result)

    assert result_dict["status"] == "ok"
    assert abs(result_dict["delta_r"]) < 1e-6
    print(f"[✓] Test 1 passed: Same particle Delta R = {result_dict['delta_r']:.6f}")

    # Test 2: Different particles
    tool = CalculateDeltaRTool(
        base_directory=test_base_dir,
        particle1=[10.0, 0.0, 5.0, 15.0],
        particle2=[0.0, 10.0, 5.0, 15.0]
    )
    result = tool._run()
    result_dict = json.loads(result)

    assert result_dict["status"] == "ok"
    assert result_dict["delta_r"] > 0
    print(f"[✓] Test 2 passed: Different particles Delta R = {result_dict['delta_r']:.4f}")

    # Test 3: Back-to-back in transverse plane (Delta phi ~ pi)
    tool = CalculateDeltaRTool(
        base_directory=test_base_dir,
        particle1=[10.0, 0.0, 0.0, 15.0],
        particle2=[-10.0, 0.0, 0.0, 15.0]
    )
    result = tool._run()
    result_dict = json.loads(result)

    assert result_dict["status"] == "ok"
    # Delta eta = 0, Delta phi = pi, so Delta R = pi
    assert abs(result_dict["delta_r"] - np.pi) < 1e-2
    print(f"[✓] Test 3 passed: Back-to-back Delta R = {result_dict['delta_r']:.4f} (~ pi)")

    print("\nAll Delta R tests passed! [✓]\n")


def test_apply_cuts():
    """Test kinematic cuts application."""
    print(">> Testing kinematic cuts...\n")

    # Create test directory
    test_dir = Path(test_base_dir) / "test_apply_cuts"
    test_dir.mkdir(parents=True, exist_ok=True)

    # Create test JSONL file with various particles
    jsonl_path = test_dir / "input.jsonl"
    events = [
        {
            "schema": "evtjsonl-1.0",
            "event_id": 0,
            "finals_only": True,
            "full_history": False,
            "data": {
                "n": 5,
                "particles": [
                    # High pT electron
                    {"i": 0, "id": 11, "px": 50.0, "py": 0.0, "pz": 10.0, "E": 51.0, "m": 0.0},
                    # Low pT muon
                    {"i": 1, "id": 13, "px": 5.0, "py": 0.0, "pz": 2.0, "E": 5.4, "m": 0.1},
                    # Medium pT photon
                    {"i": 2, "id": 22, "px": 20.0, "py": 20.0, "pz": 5.0, "E": 30.0, "m": 0.0},
                    # High eta jet
                    {"i": 3, "id": 21, "px": 10.0, "py": 0.0, "pz": 100.0, "E": 101.0, "m": 0.0},
                    # Good jet
                    {"i": 4, "id": 21, "px": 30.0, "py": 0.0, "pz": 20.0, "E": 36.0, "m": 0.0},
                ]
            }
        }
    ]

    with open(jsonl_path, 'w') as f:
        for ev in events:
            f.write(json.dumps(ev) + "\n")

    # Test 1: pT cut (pT > 20 GeV)
    output_path = test_dir / "output_pt_cut.jsonl"
    tool = ApplyCutsTool(
        base_directory=str(test_base_dir),
        input_path="test_apply_cuts/input.jsonl",
        output_path="test_apply_cuts/output_pt_cut.jsonl",
        pt_min=20.0
    )
    tool._setup()
    result = tool._run()
    result_dict = json.loads(result)

    assert result_dict["status"] == "ok"
    # Should keep: particles 0 (pT=50), 2 (pT~28.3), 4 (pT=30)
    # Should drop: particles 1 (pT=5), 3 (pT=10)
    # Actually particle 3 has pT=10, so we expect 3 particles
    assert result_dict["particles_before"] == 5
    assert result_dict["particles_after"] == 3
    print(f"[✓] Test 1 passed: pT cut, {result_dict['particles_after']}/{result_dict['particles_before']} particles kept")

    # Test 2: eta cut (|eta| < 2.5)
    output_path = test_dir / "output_eta_cut.jsonl"
    tool = ApplyCutsTool(
        base_directory=str(test_base_dir),
        input_path="test_apply_cuts/input.jsonl",
        output_path="test_apply_cuts/output_eta_cut.jsonl",
        eta_max=2.5
    )
    tool._setup()
    result = tool._run()
    result_dict = json.loads(result)

    assert result_dict["status"] == "ok"
    # Particle 3 has very high eta (pz=100, pT=10 -> eta ~ 3.0), should be cut
    assert result_dict["particles_after"] == 4
    print(f"[✓] Test 2 passed: eta cut, {result_dict['particles_after']}/{result_dict['particles_before']} particles kept")

    # Test 3: PDG ID filter (leptons only: 11, -11, 13, -13)
    output_path = test_dir / "output_pdgid.jsonl"
    tool = ApplyCutsTool(
        base_directory=str(test_base_dir),
        input_path="test_apply_cuts/input.jsonl",
        output_path="test_apply_cuts/output_pdgid.jsonl",
        pdgids=[11, -11, 13, -13]
    )
    tool._setup()
    result = tool._run()
    result_dict = json.loads(result)

    assert result_dict["status"] == "ok"
    assert result_dict["particles_after"] == 2  # electron + muon
    print(f"[✓] Test 3 passed: PDG ID filter, {result_dict['particles_after']} leptons kept")

    # Test 4: Combined cuts (pT > 25 and |eta| < 2.5)
    output_path = test_dir / "output_combined.jsonl"
    tool = ApplyCutsTool(
        base_directory=str(test_base_dir),
        input_path="test_apply_cuts/input.jsonl",
        output_path="test_apply_cuts/output_combined.jsonl",
        pt_min=25.0,
        eta_max=2.5
    )
    tool._setup()
    result = tool._run()
    result_dict = json.loads(result)

    assert result_dict["status"] == "ok"
    # Should keep particles 0 (pT=50), 2 (pT~28.3), 4 (pT=30)
    # All have reasonable eta
    assert result_dict["particles_after"] == 3
    print(f"[✓] Test 4 passed: Combined cuts, {result_dict['particles_after']} particles kept")

    print("\nAll cuts tests passed! [✓]\n")


# ====================================================================== #
# ================ Event Selection Tools Tests ========================= #
# ====================================================================== #

def test_get_hardest_n():
    """Test hardest-N particle selection."""
    print(">> Testing hardest-N selection...\n")

    # Create test directory
    test_dir = Path(test_base_dir) / "test_hardest_n"
    test_dir.mkdir(parents=True, exist_ok=True)

    # Create test JSONL with particles of varying pT
    jsonl_path = test_dir / "input.jsonl"
    events = [
        {
            "schema": "evtjsonl-1.0",
            "event_id": 0,
            "finals_only": True,
            "full_history": False,
            "data": {
                "n": 5,
                "particles": [
                    {"i": 0, "id": 11, "px": 10.0, "py": 0.0, "pz": 0.0, "E": 10.0, "m": 0.0},   # pT = 10
                    {"i": 1, "id": 13, "px": 30.0, "py": 0.0, "pz": 0.0, "E": 30.0, "m": 0.0},   # pT = 30
                    {"i": 2, "id": 22, "px": 20.0, "py": 0.0, "pz": 0.0, "E": 20.0, "m": 0.0},   # pT = 20
                    {"i": 3, "id": 21, "px": 50.0, "py": 0.0, "pz": 0.0, "E": 50.0, "m": 0.0},   # pT = 50
                    {"i": 4, "id": 21, "px": 40.0, "py": 0.0, "pz": 0.0, "E": 40.0, "m": 0.0},   # pT = 40
                ]
            }
        }
    ]

    with open(jsonl_path, 'w') as f:
        for ev in events:
            f.write(json.dumps(ev) + "\n")

    # Test 1: Get hardest 3 particles
    output_path = test_dir / "output_top3.jsonl"
    tool = GetHardestNTool(
        base_directory=str(test_base_dir),
        input_path="test_hardest_n/input.jsonl",
        output_path="test_hardest_n/output_top3.jsonl",
        n_hardest=3
    )
    tool._setup()
    result = tool._run()
    result_dict = json.loads(result)

    assert result_dict["status"] == "ok"
    assert result_dict["n_hardest"] == 3

    # Verify output contains top 3 by pT
    with open(output_path, 'r') as f:
        output_events = [json.loads(line) for line in f]

    assert len(output_events[0]["data"]["particles"]) == 3
    # Should be in order: 50, 40, 30
    pts = [p["px"] for p in output_events[0]["data"]["particles"]]
    assert pts == [50.0, 40.0, 30.0]
    print(f"[✓] Test 1 passed: Top 3 particles by pT = {pts}")

    # Test 2: Get hardest 2 with PDG filter (leptons only)
    output_path = test_dir / "output_top2_leptons.jsonl"
    tool = GetHardestNTool(
        base_directory=str(test_base_dir),
        input_path="test_hardest_n/input.jsonl",
        output_path="test_hardest_n/output_top2_leptons.jsonl",
        n_hardest=2,
        pdgids=[11, -11, 13, -13]
    )
    tool._setup()
    result = tool._run()
    result_dict = json.loads(result)

    assert result_dict["status"] == "ok"

    with open(output_path, 'r') as f:
        output_events = [json.loads(line) for line in f]

    # Should have electron (pT=10) and muon (pT=30), ordered by pT
    assert len(output_events[0]["data"]["particles"]) == 2
    pts = [p["px"] for p in output_events[0]["data"]["particles"]]
    assert pts == [30.0, 10.0]
    print(f"[✓] Test 2 passed: Top 2 leptons by pT = {pts}")

    print("\nAll hardest-N tests passed! [✓]\n")


def test_get_hardest_n_jets():
    """Test hardest-N jets selection from jets JSONL."""
    print(">> Testing hardest-N jets selection...\n")

    # Create test directory
    test_dir = Path(test_base_dir) / "test_hardest_n_jets"
    test_dir.mkdir(parents=True, exist_ok=True)

    # Create test jets JSONL with jets of varying pT (NEW structure with "data" wrapper)
    jsonl_path = test_dir / "input_jets.jsonl"
    events = [
        {
            "algorithm": "antikt",
            "R": 0.4,
            "ptmin": 20.0,
            "etamax": 5.0,
            "mass_option": 1,
            "event_index": 0,
            "data": {
                "n": 5,
                "jets": [
                    {"index": 0, "px": 10.0, "py": 0.0, "pz": 0.0, "E": 10.0, "m": 0.0, "pT": 10.0, "eta": 0.0, "phi": 0.0, "n_const": 2, "constituents": []},
                    {"index": 1, "px": 50.0, "py": 0.0, "pz": 0.0, "E": 50.0, "m": 0.0, "pT": 50.0, "eta": 0.0, "phi": 0.0, "n_const": 3, "constituents": []},
                    {"index": 2, "px": 30.0, "py": 0.0, "pz": 0.0, "E": 30.0, "m": 0.0, "pT": 30.0, "eta": 0.0, "phi": 0.0, "n_const": 4, "constituents": []},
                    {"index": 3, "px": 40.0, "py": 0.0, "pz": 0.0, "E": 40.0, "m": 0.0, "pT": 40.0, "eta": 0.0, "phi": 0.0, "n_const": 2, "constituents": []},
                    {"index": 4, "px": 20.0, "py": 0.0, "pz": 0.0, "E": 20.0, "m": 0.0, "pT": 20.0, "eta": 0.0, "phi": 0.0, "n_const": 5, "constituents": []},
                ]
            }
        }
    ]

    with open(jsonl_path, 'w') as f:
        for ev in events:
            f.write(json.dumps(ev) + "\n")

    # Test 1: Get hardest 3 jets
    output_path = test_dir / "output_top3_jets.jsonl"
    tool = GetHardestNJetsTool(
        base_directory=str(test_base_dir),
        input_path="test_hardest_n_jets/input_jets.jsonl",
        output_path="test_hardest_n_jets/output_top3_jets.jsonl",
        n_hardest=3
    )
    tool._setup()
    result = tool._run()
    result_dict = json.loads(result)

    assert result_dict["status"] == "ok"
    assert result_dict["n_hardest"] == 3

    # Verify output contains top 3 by pT
    with open(output_path, 'r') as f:
        output_events = [json.loads(line) for line in f]

    assert len(output_events[0]["data"]["jets"]) == 3
    # Should be in order: 50, 40, 30
    pts = [j["pT"] for j in output_events[0]["data"]["jets"]]
    assert pts == [50.0, 40.0, 30.0]
    print(f"[✓] Test 1 passed: Top 3 jets by pT = {pts}")

    # Test 2: Get hardest 2 jets
    output_path = test_dir / "output_top2_jets.jsonl"
    tool = GetHardestNJetsTool(
        base_directory=str(test_base_dir),
        input_path="test_hardest_n_jets/input_jets.jsonl",
        output_path="test_hardest_n_jets/output_top2_jets.jsonl",
        n_hardest=2
    )
    tool._setup()
    result = tool._run()
    result_dict = json.loads(result)

    assert result_dict["status"] == "ok"

    with open(output_path, 'r') as f:
        output_events = [json.loads(line) for line in f]

    assert len(output_events[0]["data"]["jets"]) == 2
    pts = [j["pT"] for j in output_events[0]["data"]["jets"]]
    assert pts == [50.0, 40.0]
    print(f"[✓] Test 2 passed: Top 2 jets by pT = {pts}")

    # Test 3: Verify error on particle events (wrong structure)
    particle_jsonl_path = test_dir / "particle_events.jsonl"
    particle_events = [
        {
            "schema": "evtjsonl-1.0",
            "event_id": 0,
            "data": {
                "n": 2,
                "particles": [
                    {"i": 0, "id": 11, "px": 50.0, "py": 0.0, "pz": 0.0, "E": 50.0, "m": 0.0},
                ]
            }
        }
    ]
    with open(particle_jsonl_path, 'w') as f:
        for ev in particle_events:
            f.write(json.dumps(ev) + "\n")

    tool = GetHardestNJetsTool(
        base_directory=str(test_base_dir),
        input_path="test_hardest_n_jets/particle_events.jsonl",
        output_path="test_hardest_n_jets/output_error.jsonl",
        n_hardest=2
    )
    tool._setup()
    result = tool._run()

    # format_error returns plain text, not JSON
    assert "error" in result.lower()
    assert "jets" in result.lower()
    print(f"[✓] Test 3 passed: Correctly rejected particle events structure")

    print("\nAll hardest-N jets tests passed! [✓]\n")


def test_filter_by_pdgid():
    """Test PDG ID filtering."""
    print(">> Testing PDG ID filtering...\n")

    # Create test directory
    test_dir = Path(test_base_dir) / "test_filter_pdgid"
    test_dir.mkdir(parents=True, exist_ok=True)

    # Create test JSONL
    jsonl_path = test_dir / "input.jsonl"
    events = [
        {
            "schema": "evtjsonl-1.0",
            "event_id": 0,
            "finals_only": True,
            "full_history": False,
            "data": {
                "n": 6,
                "particles": [
                    {"i": 0, "id": 11, "px": 10.0, "py": 0.0, "pz": 0.0, "E": 10.0, "m": 0.0},
                    {"i": 1, "id": -11, "px": 10.0, "py": 0.0, "pz": 0.0, "E": 10.0, "m": 0.0},
                    {"i": 2, "id": 13, "px": 10.0, "py": 0.0, "pz": 0.0, "E": 10.0, "m": 0.0},
                    {"i": 3, "id": 22, "px": 10.0, "py": 0.0, "pz": 0.0, "E": 10.0, "m": 0.0},
                    {"i": 4, "id": 21, "px": 10.0, "py": 0.0, "pz": 0.0, "E": 10.0, "m": 0.0},
                    {"i": 5, "id": 2, "px": 10.0, "py": 0.0, "pz": 0.0, "E": 10.0, "m": 0.0},
                ]
            }
        }
    ]

    with open(jsonl_path, 'w') as f:
        for ev in events:
            f.write(json.dumps(ev) + "\n")

    # Test: Keep electrons only
    output_path = test_dir / "output_electrons.jsonl"
    tool = FilterByPDGIDTool(
        base_directory=str(test_base_dir),
        input_path="test_filter_pdgid/input.jsonl",
        output_path="test_filter_pdgid/output_electrons.jsonl",
        pdgids=[11, -11]
    )
    tool._setup()
    result = tool._run()
    result_dict = json.loads(result)

    assert result_dict["status"] == "ok"
    assert result_dict["particles_before"] == 6
    assert result_dict["particles_after"] == 2
    print(f"[✓] Test passed: Filtered to {result_dict['particles_after']} electrons")

    print("\nAll PDG ID filter tests passed! [✓]\n")


def test_sort_by_pt():
    """Test pT sorting."""
    print(">> Testing pT sorting...\n")

    # Create test directory
    test_dir = Path(test_base_dir) / "test_sort_pt"
    test_dir.mkdir(parents=True, exist_ok=True)

    # Create test JSONL with unsorted particles
    jsonl_path = test_dir / "input.jsonl"
    events = [
        {
            "schema": "evtjsonl-1.0",
            "event_id": 0,
            "finals_only": True,
            "full_history": False,
            "data": {
                "n": 4,
                "particles": [
                    {"i": 0, "id": 11, "px": 20.0, "py": 0.0, "pz": 0.0, "E": 20.0, "m": 0.0},   # pT = 20
                    {"i": 1, "id": 13, "px": 50.0, "py": 0.0, "pz": 0.0, "E": 50.0, "m": 0.0},   # pT = 50
                    {"i": 2, "id": 22, "px": 10.0, "py": 0.0, "pz": 0.0, "E": 10.0, "m": 0.0},   # pT = 10
                    {"i": 3, "id": 21, "px": 30.0, "py": 0.0, "pz": 0.0, "E": 30.0, "m": 0.0},   # pT = 30
                ]
            }
        }
    ]

    with open(jsonl_path, 'w') as f:
        for ev in events:
            f.write(json.dumps(ev) + "\n")

    # Test 1: Sort descending (default)
    output_path = test_dir / "output_desc.jsonl"
    tool = SortByPtTool(
        base_directory=str(test_base_dir),
        input_path="test_sort_pt/input.jsonl",
        output_path="test_sort_pt/output_desc.jsonl",
        ascending=False
    )
    tool._setup()
    result = tool._run()
    result_dict = json.loads(result)

    assert result_dict["status"] == "ok"

    with open(output_path, 'r') as f:
        output_events = [json.loads(line) for line in f]

    pts = [p["px"] for p in output_events[0]["data"]["particles"]]
    assert pts == [50.0, 30.0, 20.0, 10.0]
    print(f"[✓] Test 1 passed: Descending pT order = {pts}")

    # Test 2: Sort ascending
    output_path = test_dir / "output_asc.jsonl"
    tool = SortByPtTool(
        base_directory=str(test_base_dir),
        input_path="test_sort_pt/input.jsonl",
        output_path="test_sort_pt/output_asc.jsonl",
        ascending=True
    )
    tool._setup()
    result = tool._run()
    result_dict = json.loads(result)

    assert result_dict["status"] == "ok"

    with open(output_path, 'r') as f:
        output_events = [json.loads(line) for line in f]

    pts = [p["px"] for p in output_events[0]["data"]["particles"]]
    assert pts == [10.0, 20.0, 30.0, 50.0]
    print(f"[✓] Test 2 passed: Ascending pT order = {pts}")

    print("\nAll pT sorting tests passed! [✓]\n")




# ====================================================================== #
# ========================= Path Security Tests ======================== #
# ====================================================================== #

def test_path_traversal_prevention():
    """Test that path traversal attempts are rejected."""
    print(">> Testing path traversal prevention (security)...\n")

    # Test 1: CalculateInvariantMassTool - npy_path traversal
    tool = CalculateInvariantMassTool(
        base_directory=test_base_dir,
        npy_path="../../../etc/passwd"
    )
    tool._setup()
    result = tool._run()
    assert "error" in result.lower() or "denied" in result.lower()
    print("[✓] Test 1 passed: npy_path traversal rejected")

    # Test 2: ApplyCutsTool - input_path traversal
    tool = ApplyCutsTool(
        base_directory=test_base_dir,
        input_path="../../../etc/passwd",
        output_path="output.jsonl"
    )
    tool._setup()
    result = tool._run()
    assert "error" in result.lower() or "denied" in result.lower()
    print("[✓] Test 2 passed: input_path traversal rejected")

    # Test 3: GetHardestNTool - output_path traversal
    tool = GetHardestNTool(
        base_directory=test_base_dir,
        input_path="test.jsonl",
        output_path="../../../tmp/evil.jsonl",
        n_hardest=2
    )
    tool._setup()
    result = tool._run()
    assert "error" in result.lower() or "denied" in result.lower()
    print("[✓] Test 3 passed: output_path traversal rejected")

    print("\nAll path traversal prevention tests passed! [✓]\n")


# ============================================================================
# Main Test Runner
# ============================================================================

def main():
    """Run all tests."""
    import argparse

    # Declare global before any use
    global _keep_files

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Test suite for analysis/kinematics tools")
    parser.add_argument("--keep-files", action="store_true",
                        help="Keep test-generated files after tests complete")
    args = parser.parse_args()

    # Set global flag
    _keep_files = args.keep_files

    # Create base test directory
    os.makedirs(test_base_dir, exist_ok=True)

    try:
        # Kinematics tools tests
        test_invariant_mass_from_particles()
        test_invariant_mass_from_numpy()
        test_invariant_mass_from_jsonl()
        test_transverse_momentum()
        test_delta_r()
        test_apply_cuts()

        # Event selection tools tests
        test_get_hardest_n()
        test_get_hardest_n_jets()
        test_filter_by_pdgid()
        test_sort_by_pt()

        # Security tests
        test_path_traversal_prevention()

        print()
        print("=" * 70)
        print("Test suite completed successfully! [✓]")
        print("=" * 70)

        # Cleanup test directory
        if not _keep_files:
            import shutil
            shutil.rmtree(test_base_dir, ignore_errors=True)

    except AssertionError as e:
        print("\n" + "=" * 70)
        print("Test suite completed with failures! [✗]")
        print("=" * 70)
        print(f"Assertion error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    except Exception as e:
        print("\n" + "=" * 70)
        print("Test suite completed with failures! [✗]")
        print("=" * 70)
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
    # If we reach here, all tests passed (main() exits with 1 on failure)
    sys.exit(0)
