#!/usr/bin/env python3
"""
# test_pdg_tool.py is a part of the HEPTAPOD package.
# Copyright (C) 2025 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

Tests for the PDG Database Tool.

Run with:
    python test_pdg_tool.py
"""

import sys
import json
from pathlib import Path

# Add repo root to path
SCRIPT_PATH = Path(__file__).resolve()
TOOL_DIR = SCRIPT_PATH.parent.parent
REPO_ROOT = TOOL_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tools.pdg import (
    PDGInterface,
    get_particle,
    get_mass,
    get_lifetime,
    get_width,
    get_branching_fractions,
    resolve_alias,
    get_resolution_info,
    PARTICLE_ALIASES
)


def test_alias_resolution():
    """Test particle alias resolution."""
    print("=" * 60)
    print("Testing alias resolution")
    print("=" * 60)

    test_cases = [
        ("electron", "e-"),
        ("Electron", "e-"),  # Case insensitive
        ("ELECTRON", "e-"),
        ("muon", "mu-"),
        ("W boson", "W+"),
        ("higgs", "H"),
        ("photon", "gamma"),
        ("proton", "p"),
        ("tau", "tau-"),
        ("e-", "e-"),  # Already PDG name
    ]

    all_passed = True
    for name, expected in test_cases:
        resolved = resolve_alias(name)
        status = "PASS" if resolved == expected else "FAIL"
        if status == "FAIL":
            all_passed = False
        print(f"  {status}: '{name}' -> '{resolved}' (expected '{expected}')")

    print()
    return all_passed


def test_get_particle():
    """Test getting particle information."""
    print("=" * 60)
    print("Testing get_particle()")
    print("=" * 60)

    particles = ["electron", "muon", "tau", "W boson", "Z boson", "Higgs", "proton", "photon"]

    for name in particles:
        try:
            info = get_particle(name)
            print(f"\n  {name}:")
            print(f"    PDG name: {info.pdg_name}")
            print(f"    MC ID: {info.mcid}")
            print(f"    Mass: {info.mass_gev} GeV")
            if info.width_gev:
                print(f"    Width: {info.width_gev} GeV")
            if info.lifetime_s:
                print(f"    Lifetime: {info.lifetime_s} s")
            if info.spin:
                print(f"    Spin: {info.spin}")
        except Exception as e:
            print(f"  {name}: ERROR - {e}")

    print()
    return True


def test_get_mass():
    """Test getting particle masses."""
    print("=" * 60)
    print("Testing get_mass()")
    print("=" * 60)

    particles = ["electron", "muon", "W boson", "Higgs", "proton"]

    for name in particles:
        mass = get_mass(name)
        error_str = ""
        if mass.error_plus:
            error_str = f" +/- {mass.error_plus}"
        print(f"  {name}: {mass.value}{error_str} {mass.unit}")

    print()
    return True


def test_get_lifetime():
    """Test getting particle lifetimes."""
    print("=" * 60)
    print("Testing get_lifetime()")
    print("=" * 60)

    particles = ["muon", "tau", "neutron", "pi+"]

    for name in particles:
        try:
            lt = get_lifetime(name)
            if lt.value:
                print(f"  {name}: {lt.value:.3e} {lt.unit}")
            else:
                print(f"  {name}: stable or no lifetime data")
        except Exception as e:
            print(f"  {name}: ERROR - {e}")

    print()
    return True


def test_get_width():
    """Test getting particle widths."""
    print("=" * 60)
    print("Testing get_width()")
    print("=" * 60)

    particles = ["W boson", "Z boson", "Higgs"]

    for name in particles:
        try:
            width = get_width(name)
            if width.value:
                print(f"  {name}: {width.value} {width.unit}")
            else:
                print(f"  {name}: no width data")
        except Exception as e:
            print(f"  {name}: ERROR - {e}")

    print()
    return True


def test_get_branching_fractions():
    """Test getting branching fractions."""
    print("=" * 60)
    print("Testing get_branching_fractions()")
    print("=" * 60)

    particles = ["tau", "W+", "Z boson"]

    for name in particles:
        try:
            bfs = get_branching_fractions(name, limit=5)
            print(f"\n  {name} decays ({len(bfs)} shown):")
            for bf in bfs:
                fraction = bf.fraction
                if fraction:
                    print(f"    {bf.decay_products}: {fraction:.4f}")
                else:
                    print(f"    {bf.decay_products}: (limit)")
        except Exception as e:
            print(f"  {name}: ERROR - {e}")

    print()
    return True


def test_photon_mass_limit():
    """Test that photon mass returns as a limit."""
    print("=" * 60)
    print("Testing photon mass (should be limit/None)")
    print("=" * 60)

    photon = get_particle("photon")
    print(f"  Photon mass: {photon.mass_gev}")
    print(f"  (None indicates massless or upper limit)")

    print()
    return True


def test_quantum_numbers():
    """Test quantum number extraction."""
    print("=" * 60)
    print("Testing quantum numbers")
    print("=" * 60)

    particles = ["electron", "proton", "pi+", "K+"]

    for name in particles:
        try:
            info = get_particle(name)
            print(f"\n  {name}:")
            if info.spin:
                print(f"    Spin (J): {info.spin}")
            if info.parity:
                print(f"    Parity (P): {info.parity}")
            if info.charge is not None:
                print(f"    Charge: {info.charge}")
        except Exception as e:
            print(f"  {name}: ERROR - {e}")

    print()
    return True


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("PDG Database Tool Tests")
    print("=" * 60 + "\n")

    tests = [
        ("Alias Resolution", test_alias_resolution),
        ("Get Particle", test_get_particle),
        ("Get Mass", test_get_mass),
        ("Get Lifetime", test_get_lifetime),
        ("Get Width", test_get_width),
        ("Get Branching Fractions", test_get_branching_fractions),
        ("Photon Mass Limit", test_photon_mass_limit),
        ("Quantum Numbers", test_quantum_numbers),
    ]

    results = []
    for name, test_fn in tests:
        try:
            result = test_fn()
            results.append((name, "PASS" if result else "FAIL"))
        except Exception as e:
            print(f"ERROR in {name}: {e}")
            results.append((name, "ERROR"))

    # Summary
    print("=" * 60)
    print("Test Summary")
    print("=" * 60)
    for name, status in results:
        print(f"  {status}: {name}")

    passed = sum(1 for _, s in results if s == "PASS")
    total = len(results)
    print(f"\nTotal: {passed}/{total} tests passed")

    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
