#!/usr/bin/env python3
"""
# kinematics_example.py is a part of the HEPTAPOD package.
# Copyright (C) 2025 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.
"""
"""
Example demonstrating the new file-based interface for analysis tools.

This shows the recommended way to use analysis tools with large datasets:
- Input: .npy or .jsonl files
- Output: .npy files for efficient storage and chaining
- Minimal token usage for agents
"""

import os, sys
from pathlib import Path
import json
import numpy as np

SCRIPT_PATH = Path(__file__).resolve()
ANALYSIS_DIR = SCRIPT_PATH.parent
TOOLS_DIR = ANALYSIS_DIR.parent
REPO_ROOT = TOOLS_DIR.parent
DEP_ROOT = REPO_ROOT / "external" / "dep"

sys.path.insert(0, str(DEP_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from tools.analysis.kinematics import (
    CalculateInvariantMassTool,
    CalculateTransverseMomentumTool,
    CalculateDeltaRTool
)

# Setup example directory
example_dir = UTILS_DIR / "file_based_example"
example_dir.mkdir(exist_ok=True)

print("=" * 70)
print("File-Based Analysis Tools Example")
print("=" * 70)
print()

# ============================================================================
# Step 1: Create example data (simulating output from data conversion tools)
# ============================================================================
print("Step 1: Creating example particle data...")
print()

# Create example event data: Z boson decay to two leptons
# Events with 2 particles each
n_events = 100
events_data = []

for i in range(n_events):
    # Z mass ~ 91.2 GeV, create leptons back-to-back with some variation
    px1 = 40.0 + np.random.normal(0, 5)
    py1 = 30.0 + np.random.normal(0, 5)
    pz1 = 10.0 + np.random.normal(0, 5)
    E1 = np.sqrt(px1**2 + py1**2 + pz1**2)

    # Second particle roughly back-to-back
    px2 = -px1 + np.random.normal(0, 2)
    py2 = -py1 + np.random.normal(0, 2)
    pz2 = -pz1 + np.random.normal(0, 2)
    E2 = np.sqrt(px2**2 + py2**2 + pz2**2)

    event = np.array([
        [px1, py1, pz1, E1, 11],   # electron
        [px2, py2, pz2, E2, -11],  # positron
        [0, 0, 0, 0, 0],           # padding
        [0, 0, 0, 0, 0],           # padding
    ])
    events_data.append(event)

particles_array = np.array(events_data)  # Shape: (100, 4, 5)

input_file = example_dir / "particles.npy"
np.save(input_file, particles_array)

print(f"✓ Created {input_file}")
print(f"  Shape: {particles_array.shape}")
print(f"  Events: {n_events}")
print()

# ============================================================================
# Step 2: Calculate invariant mass (file-based interface)
# ============================================================================
print("Step 2: Calculate invariant mass using file-based interface...")
print()

tool = CalculateInvariantMassTool(
    base_directory=str(UTILS_DIR),
    input_file=str(input_file.relative_to(UTILS_DIR)),
    # output_file is optional - will auto-generate as particles_invariant_masses.npy
)
tool._setup()
result = tool._run()
result_dict = json.loads(result)

print(f"✓ Invariant mass calculation complete")
print(f"  Output file: {result_dict['output_file']}")
print(f"  Number of events: {result_dict['n_events']}")
print(f"  Mean mass: {result_dict['mean_mass']:.2f} GeV")
print(f"  Std mass: {result_dict['std_mass']:.2f} GeV")
print(f"  Min mass: {result_dict['min_mass']:.2f} GeV")
print(f"  Max mass: {result_dict['max_mass']:.2f} GeV")
print()

# Load and inspect the output
masses_file = UTILS_DIR / result_dict['output_file']
masses = np.load(masses_file)
print(f"  Loaded output array shape: {masses.shape}")
print(f"  First 5 masses: {masses[:5]}")
print()

# ============================================================================
# Step 3: Calculate pT for all particles (file-based interface)
# ============================================================================
print("Step 3: Calculate transverse momentum using file-based interface...")
print()

tool = CalculateTransverseMomentumTool(
    base_directory=str(UTILS_DIR),
    input_file=str(input_file.relative_to(UTILS_DIR)),
    output_file="file_based_example/pt_values.npy"
)
tool._setup()
result = tool._run()
result_dict = json.loads(result)

print(f"✓ pT calculation complete")
print(f"  Output file: {result_dict['output_file']}")
print(f"  Number of events: {result_dict['n_events']}")
print(f"  Output shape: {result_dict['output_shape']}")
print()

# Load and inspect the output
pt_file = UTILS_DIR / result_dict['output_file']
pt_values = np.load(pt_file)
print(f"  Loaded output array shape: {pt_values.shape}")
print(f"  First event pT values: {pt_values[0]}")
print()

# ============================================================================
# Step 4: Calculate Delta R between particles (file-based interface)
# ============================================================================
print("Step 4: Calculate Delta R between particle pairs...")
print()

# Calculate Delta R between particles 0 and 1 in each event
tool = CalculateDeltaRTool(
    base_directory=str(UTILS_DIR),
    input_file=str(input_file.relative_to(UTILS_DIR)),
    particle_pairs=[[0, 1]],  # Only calculate Delta R between first two particles
    output_file="file_based_example/deltaR_values.npy"
)
tool._setup()
result = tool._run()
result_dict = json.loads(result)

print(f"✓ Delta R calculation complete")
print(f"  Output file: {result_dict['output_file']}")
print(f"  Number of events: {result_dict['n_events']}")
print(f"  Output shape: {result_dict['output_shape']}")
print()

# Load and inspect the output
deltaR_file = UTILS_DIR / result_dict['output_file']
deltaR_values = np.load(deltaR_file)
print(f"  Loaded output array shape: {deltaR_values.shape}")
print(f"  First 5 Delta R values: {deltaR_values[:5]}")
print(f"  Mean Delta R: {np.mean(deltaR_values[deltaR_values > 0]):.3f}")
print()

# ============================================================================
# Step 5: Calculate invariant mass for specific particle combinations
# ============================================================================
print("Step 5: Calculate invariant mass for first particle only...")
print()

tool = CalculateInvariantMassTool(
    base_directory=str(UTILS_DIR),
    input_file=str(input_file.relative_to(UTILS_DIR)),
    particle_indices=[0],  # Only use first particle
    output_file="file_based_example/particle0_masses.npy"
)
tool._setup()
result = tool._run()
result_dict = json.loads(result)

print(f"✓ Single particle mass calculation complete")
print(f"  Output file: {result_dict['output_file']}")
print(f"  Mean mass (should be ~0 for massless particles): {result_dict['mean_mass']:.6f} GeV")
print()

# ============================================================================
# Summary
# ============================================================================
print("=" * 70)
print("Summary: File-Based Interface Benefits")
print("=" * 70)
print()
print("1. Token Efficiency:")
print("   - Old way: Agents pass entire arrays as lists (huge token cost)")
print("   - New way: Agents only pass file paths (minimal tokens)")
print()
print("2. Scalability:")
print("   - Can process arbitrarily large datasets")
print("   - No token limits on data size")
print()
print("3. Composability:")
print("   - Output files can be directly used as input to other tools")
print("   - Easy to chain operations in a pipeline")
print()
print("4. Persistence:")
print("   - Results saved for later analysis or inspection")
print("   - No need to recompute intermediate results")
print()
print("5. Backward Compatibility:")
print("   - Legacy interface (particles=, npy_path=, jsonl_path=) still works")
print("   - But file-based interface (input_file=, output_file=) is recommended")
print()
print("=" * 70)
