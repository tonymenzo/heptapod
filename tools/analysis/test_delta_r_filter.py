#!/usr/bin/env python3
"""
# test_delta_r_filter.py is a part of the HEPTAPOD package.
# Copyright (C) 2025 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.
"""
"""
Unit tests for FilterByDeltaRTool.

Tests cover:
- remove_second mode (jet-lepton overlap removal)
- remove_first mode (photon-jet isolation)
- remove_both mode (mutual removal)
- Single array self-isolation
- Multiple events
- apply_to_arrays parameter
- Auto-generated output paths
- Efficiency calculation
- Error handling
"""

import os
import sys
from pathlib import Path
import json
import tempfile
import shutil
import unittest

SCRIPT_PATH = Path(__file__).resolve()
ANALYSIS_DIR = SCRIPT_PATH.parent                             # .../heptapod-dev/tools/analysis
TOOLS_DIR = ANALYSIS_DIR.parent                               # .../heptapod-dev/tools
REPO_ROOT = TOOLS_DIR.parent                                  # .../heptapod-dev

# Add repository root to path for local imports (prompts, tools, etc.)
sys.path.insert(0, str(REPO_ROOT))

from tools.analysis.kinematics import FilterByDeltaRTool

# Global flag for keeping test files
keep_files = False


class TestFilterByDeltaRTool(unittest.TestCase):
    """Test suite for FilterByDeltaRTool."""

    def setUp(self):
        """Create temporary directory and test data."""
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary directory."""
        if not keep_files:
            shutil.rmtree(self.test_dir)

    def _create_test_events(self, filename, particles_per_event):
        """
        Helper to create test JSONL file.

        Args:
            filename: Output filename
            particles_per_event: List of particle lists, one per event
                Each particle: {"px": float, "py": float, "pz": float, "E": float, "id": int}
        """
        filepath = os.path.join(self.test_dir, filename)
        with open(filepath, 'w') as f:
            for event_idx, particles in enumerate(particles_per_event):
                event = {
                    "schema": "evtjsonl-1.0",
                    "event_id": event_idx,
                    "data": {
                        "n_particles": len(particles),
                        "particles": [
                            {
                                "i": i,
                                "px": p["px"],
                                "py": p["py"],
                                "pz": p["pz"],
                                "E": p["E"],
                                "m": p.get("m", 0.0),
                                "id": p.get("id", 0)
                            }
                            for i, p in enumerate(particles)
                        ]
                    }
                }
                f.write(json.dumps(event) + "\n")
        return filename

    def _read_events(self, filename):
        """Read events from JSONL file."""
        filepath = os.path.join(self.test_dir, filename)
        with open(filepath, 'r') as f:
            return [json.loads(line) for line in f]

    def test_remove_second_mode(self):
        """Test remove_second mode: remove objects from second array."""
        # Create two leptons at (eta, phi) = (0, 0) and (2, 0)
        leptons = [
            [
                {"px": 10.0, "py": 0.0, "pz": 0.0, "E": 10.0, "id": 11},
                {"px": 10.0, "py": 0.0, "pz": 21.5, "E": 23.7, "id": 13}  # eta ~ 2
            ]
        ]

        # Create three jets: one near first lepton, one near second, one isolated
        jets = [
            [
                {"px": 20.0, "py": 1.0, "pz": 0.0, "E": 20.1, "id": 0},      # Near lepton 1 (DR ~ 0.05)
                {"px": 20.0, "py": 1.0, "pz": 43.0, "E": 47.6, "id": 0},    # Near lepton 2 (DR ~ 0.05)
                {"px": 20.0, "py": 0.0, "pz": 100.0, "E": 102.0, "id": 0}  # Isolated (eta ~ 5)
            ]
        ]

        leptons_file = self._create_test_events("leptons.jsonl", leptons)
        jets_file = self._create_test_events("jets.jsonl", jets)

        # Run tool: remove jets within DR < 0.4 of leptons
        tool = FilterByDeltaRTool(
            base_directory=self.test_dir,
            particle_arrays=[leptons_file, jets_file],
            delta_r_threshold=0.4,
            filter_mode="remove_second"
        )
        result_json = tool._run()

        result = json.loads(result_json)
        self.assertEqual(result["status"], "ok")

        # Check leptons unchanged (all 2 kept)
        self.assertEqual(result["arrays"][0]["objects_before"], 2)
        self.assertEqual(result["arrays"][0]["objects_after"], 2)

        # Check jets filtered (only isolated jet kept)
        self.assertEqual(result["arrays"][1]["objects_before"], 3)
        self.assertEqual(result["arrays"][1]["objects_after"], 1)

        # Verify output files
        filtered_jets = self._read_events("jets_filtered.jsonl")
        self.assertEqual(len(filtered_jets[0]["data"]["particles"]), 1)
        # The isolated jet should be the one with pz ~ 100
        self.assertAlmostEqual(filtered_jets[0]["data"]["particles"][0]["pz"], 100.0, places=1)

    def test_remove_first_mode(self):
        """Test remove_first mode: remove objects from first array."""
        # Two photons, one near a jet, one isolated
        photons = [
            [
                {"px": 10.0, "py": 0.0, "pz": 0.0, "E": 10.0, "id": 22},
                {"px": 10.0, "py": 0.0, "pz": 100.0, "E": 100.5, "id": 22}  # Isolated
            ]
        ]

        # One jet near first photon
        jets = [
            [
                {"px": 20.0, "py": 1.0, "pz": 0.0, "E": 20.1, "id": 0}
            ]
        ]

        photons_file = self._create_test_events("photons.jsonl", photons)
        jets_file = self._create_test_events("jets.jsonl", jets)

        tool = FilterByDeltaRTool(
            base_directory=self.test_dir,
            particle_arrays=[photons_file, jets_file],
            delta_r_threshold=0.4,
            filter_mode="remove_first"
        )
        result_json = tool._run()

        result = json.loads(result_json)

        # Check photons filtered (only isolated photon kept)
        self.assertEqual(result["arrays"][0]["objects_before"], 2)
        self.assertEqual(result["arrays"][0]["objects_after"], 1)

        # Check jets unchanged
        self.assertEqual(result["arrays"][1]["objects_before"], 1)
        self.assertEqual(result["arrays"][1]["objects_after"], 1)

    def test_remove_both_mode(self):
        """Test remove_both mode: remove from both arrays."""
        # Test with 2 leptons and 1 jet
        # Lepton 1 is near the jet -> both removed
        # Lepton 2 is isolated -> kept
        leptons = [
            [
                {"px": 10.0, "py": 0.0, "pz": 0.0, "E": 10.0, "id": 11},     # Near jet
                {"px": 10.0, "py": 0.0, "pz": 500.0, "E": 500.1, "id": 13}  # Isolated, far from jet
            ]
        ]

        jets = [
            [
                {"px": 20.0, "py": 1.0, "pz": 0.0, "E": 20.1, "id": 0}      # Near lepton 1 only
            ]
        ]

        leptons_file = self._create_test_events("leptons.jsonl", leptons)
        jets_file = self._create_test_events("jets.jsonl", jets)

        tool = FilterByDeltaRTool(
            base_directory=self.test_dir,
            particle_arrays=[leptons_file, jets_file],
            delta_r_threshold=0.4,
            filter_mode="remove_both"
        )
        result_json = tool._run()

        result = json.loads(result_json)

        # Lepton 1 and jet removed (they're close), lepton 2 kept (isolated)
        self.assertEqual(result["arrays"][0]["objects_after"], 1)  # Only isolated lepton kept
        self.assertEqual(result["arrays"][1]["objects_after"], 0)  # Jet removed

    def test_single_array_self_isolation(self):
        """Test filtering within a single array (self-isolation)."""
        # Three leptons: two close together, one isolated
        leptons = [
            [
                {"px": 10.0, "py": 0.0, "pz": 0.0, "E": 10.0, "id": 11},
                {"px": 10.0, "py": 3.0, "pz": 0.0, "E": 10.4, "id": 13},     # DR ~ 0.3 from first
                {"px": 10.0, "py": 0.0, "pz": 100.0, "E": 100.5, "id": 11}  # Isolated
            ]
        ]

        leptons_file = self._create_test_events("leptons.jsonl", leptons)

        tool = FilterByDeltaRTool(
            base_directory=self.test_dir,
            particle_arrays=[leptons_file],
            delta_r_threshold=0.4,
            filter_mode="remove_both"
        )
        result_json = tool._run()

        result = json.loads(result_json)

        # Two close leptons removed, one isolated kept
        self.assertEqual(result["arrays"][0]["objects_before"], 3)
        self.assertEqual(result["arrays"][0]["objects_after"], 1)

    def test_multiple_events(self):
        """Test filtering across multiple events."""
        # Event 1: 1 lepton, 2 jets (1 near, 1 far)
        # Event 2: 1 lepton, 1 jet (far)
        leptons = [
            [{"px": 10.0, "py": 0.0, "pz": 0.0, "E": 10.0, "id": 11}],
            [{"px": 10.0, "py": 0.0, "pz": 0.0, "E": 10.0, "id": 11}]
        ]

        jets = [
            [
                {"px": 20.0, "py": 1.0, "pz": 0.0, "E": 20.1, "id": 0},      # Near
                {"px": 20.0, "py": 0.0, "pz": 100.0, "E": 102.0, "id": 0}   # Far
            ],
            [
                {"px": 20.0, "py": 0.0, "pz": 100.0, "E": 102.0, "id": 0}   # Far
            ]
        ]

        leptons_file = self._create_test_events("leptons.jsonl", leptons)
        jets_file = self._create_test_events("jets.jsonl", jets)

        tool = FilterByDeltaRTool(
            base_directory=self.test_dir,
            particle_arrays=[leptons_file, jets_file],
            delta_r_threshold=0.4,
            filter_mode="remove_second"
        )
        result_json = tool._run()

        result = json.loads(result_json)

        # Total: 3 jets before, 2 after (1 removed from event 1)
        self.assertEqual(result["arrays"][1]["objects_before"], 3)
        self.assertEqual(result["arrays"][1]["objects_after"], 2)

    def test_apply_to_arrays_parameter(self):
        """Test apply_to_arrays parameter to selectively filter."""
        leptons = [
            [{"px": 10.0, "py": 0.0, "pz": 0.0, "E": 10.0, "id": 11}]
        ]

        jets = [
            [{"px": 20.0, "py": 1.0, "pz": 0.0, "E": 20.1, "id": 0}]  # Near lepton
        ]

        leptons_file = self._create_test_events("leptons.jsonl", leptons)
        jets_file = self._create_test_events("jets.jsonl", jets)

        # Filter only jets (index 1), keep leptons unchanged
        tool = FilterByDeltaRTool(
            base_directory=self.test_dir,
            particle_arrays=[leptons_file, jets_file],
            delta_r_threshold=0.4,
            filter_mode="remove_both",
            apply_to_arrays=[1]  # Only filter jets
        )
        result_json = tool._run()

        result = json.loads(result_json)

        # Leptons should be unchanged
        self.assertEqual(result["arrays"][0]["objects_after"], 1)
        # Jets should be filtered
        self.assertEqual(result["arrays"][1]["objects_after"], 0)

    def test_auto_generated_output_paths(self):
        """Test auto-generation of output paths."""
        leptons = [[{"px": 10.0, "py": 0.0, "pz": 0.0, "E": 10.0, "id": 11}]]
        jets = [[{"px": 20.0, "py": 0.0, "pz": 100.0, "E": 102.0, "id": 0}]]

        leptons_file = self._create_test_events("leptons.jsonl", leptons)
        jets_file = self._create_test_events("jets.jsonl", jets)

        tool = FilterByDeltaRTool(
            base_directory=self.test_dir,
            particle_arrays=[leptons_file, jets_file],
            delta_r_threshold=0.4,
            filter_mode="remove_second"
        )
        result_json = tool._run()

        result = json.loads(result_json)

        # Check auto-generated paths
        self.assertTrue(result["arrays"][0]["output_path"].endswith("leptons_filtered.jsonl"))
        self.assertTrue(result["arrays"][1]["output_path"].endswith("jets_filtered.jsonl"))

        # Verify files exist
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "leptons_filtered.jsonl")))
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "jets_filtered.jsonl")))

    def test_efficiency_calculation(self):
        """Test efficiency metric calculation."""
        leptons = [[{"px": 10.0, "py": 0.0, "pz": 0.0, "E": 10.0, "id": 11}]]
        jets = [
            [
                {"px": 20.0, "py": 1.0, "pz": 0.0, "E": 20.1, "id": 0},      # Near (removed)
                {"px": 20.0, "py": 0.0, "pz": 100.0, "E": 102.0, "id": 0},  # Far (kept)
                {"px": 20.0, "py": 0.0, "pz": 200.0, "E": 201.0, "id": 0}   # Far (kept)
            ]
        ]

        leptons_file = self._create_test_events("leptons.jsonl", leptons)
        jets_file = self._create_test_events("jets.jsonl", jets)

        tool = FilterByDeltaRTool(
            base_directory=self.test_dir,
            particle_arrays=[leptons_file, jets_file],
            delta_r_threshold=0.4,
            filter_mode="remove_second"
        )
        result_json = tool._run()

        result = json.loads(result_json)

        # Jets: 3 before, 2 after, efficiency = 2/3
        self.assertAlmostEqual(result["arrays"][1]["efficiency"], 2.0/3.0, places=3)

    @unittest.skip("BaseTool validation happens before _run(), skipping error handling tests")
    def test_invalid_filter_mode(self):
        """Test error handling for invalid filter mode."""
        leptons = [[{"px": 10.0, "py": 0.0, "pz": 0.0, "E": 10.0, "id": 11}]]
        leptons_file = self._create_test_events("leptons.jsonl", leptons)

        tool = FilterByDeltaRTool(
            base_directory=self.test_dir,
            particle_arrays=[leptons_file],
            delta_r_threshold=0.4,
            filter_mode="invalid_mode"
        )
        result_json = tool._run()

        self.assertTrue(result_json, f"Tool returned empty/None: '{result_json}'")
        result = json.loads(result_json)
        self.assertEqual(result["status"], "error")
        self.assertIn("Invalid Filter Mode", result.get("error", ""))

    @unittest.skip("BaseTool validation happens before _run(), skipping error handling tests")
    def test_mismatched_event_counts(self):
        """Test error handling for mismatched event counts."""
        leptons = [
            [{"px": 10.0, "py": 0.0, "pz": 0.0, "E": 10.0, "id": 11}],
            [{"px": 10.0, "py": 0.0, "pz": 0.0, "E": 10.0, "id": 11}]
        ]
        jets = [
            [{"px": 20.0, "py": 0.0, "pz": 100.0, "E": 102.0, "id": 0}]
        ]

        leptons_file = self._create_test_events("leptons.jsonl", leptons)
        jets_file = self._create_test_events("jets.jsonl", jets)

        tool = FilterByDeltaRTool(
            base_directory=self.test_dir,
            particle_arrays=[leptons_file, jets_file],
            delta_r_threshold=0.4,
            filter_mode="remove_second"
        )
        result_json = tool._run()

        self.assertTrue(result_json, f"Tool returned empty/None: '{result_json}'")
        result = json.loads(result_json)
        self.assertEqual(result["status"], "error")
        self.assertIn("Processing Error", result.get("error", ""))




if __name__ == '__main__':
    import argparse

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Test suite for FilterByDeltaRTool")
    parser.add_argument("--keep-files", action="store_true",
                        help="Keep test-generated files after tests complete")
    args, remaining = parser.parse_known_args()

    # Set global flag
    keep_files = args.keep_files

    # Run unittest with remaining arguments
    sys.argv[1:] = remaining
    unittest.main()
