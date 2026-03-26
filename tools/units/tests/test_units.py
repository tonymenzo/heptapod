#!/usr/bin/env python3
"""
# test_units.py is a part of the HEPTAPOD package.
# Copyright (C) 2025 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

Tests for the Unit Conversion Tools.

Run with:
    python test_units.py
"""

import sys
from pathlib import Path

# Add repo root to path
SCRIPT_PATH = Path(__file__).resolve()
TOOL_DIR = SCRIPT_PATH.parent.parent
REPO_ROOT = TOOL_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tools.units import NaturalUnitsConverter, MetricPrefixConverter


def _run_conversion(converter, request):
    """Helper to run a conversion request through a tool."""
    converter.conversion_request = request
    return converter._run()


def _check_result(result, expected_value, tolerance=1e-3):
    """
    Check that a conversion result string contains a value
    within tolerance of expected_value.

    Returns True if the extracted numeric value is within relative tolerance.
    """
    # Extract the numeric result after '='
    try:
        rhs = result.split("=")[1].strip()
        # Extract the number (handles scientific notation)
        num_str = rhs.split()[0]
        actual = float(num_str)
        if expected_value == 0:
            return abs(actual) < tolerance
        return abs(actual - expected_value) / abs(expected_value) < tolerance
    except (IndexError, ValueError):
        return False


# ==================== NaturalUnitsConverter Tests ==================== #

def test_natural_energy_mass():
    """Test energy ↔ mass conversions (E = mc²)."""
    print("=" * 60)
    print("Testing NaturalUnitsConverter: energy ↔ mass")
    print("=" * 60)

    conv = NaturalUnitsConverter(base_directory="/tmp")
    all_passed = True

    # 100 GeV to kg (proton-scale mass)
    result = _run_conversion(conv, "100 GeV to kg")
    ok = _check_result(result, 1.782663e-25)
    if not ok:
        all_passed = False
    print(f"  {'PASS' if ok else 'FAIL'}: 100 GeV to kg -> {result}")

    # 1 kg to GeV (reverse)
    result = _run_conversion(conv, "1 kg to GeV")
    ok = _check_result(result, 5.609e26, tolerance=1e-2)
    if not ok:
        all_passed = False
    print(f"  {'PASS' if ok else 'FAIL'}: 1 kg to GeV -> {result}")

    # Electron mass: 0.511 MeV = 9.109e-31 kg
    result = _run_conversion(conv, "0.511 MeV to kg")
    ok = _check_result(result, 9.109e-31, tolerance=1e-2)
    if not ok:
        all_passed = False
    print(f"  {'PASS' if ok else 'FAIL'}: 0.511 MeV to kg -> {result}")

    print()
    return all_passed


def test_natural_energy_joules():
    """Test energy unit conversions (eV ↔ J)."""
    print("=" * 60)
    print("Testing NaturalUnitsConverter: energy ↔ joules")
    print("=" * 60)

    conv = NaturalUnitsConverter(base_directory="/tmp")
    all_passed = True

    # 1 eV to J
    result = _run_conversion(conv, "1 eV to J")
    ok = _check_result(result, 1.602177e-19)
    if not ok:
        all_passed = False
    print(f"  {'PASS' if ok else 'FAIL'}: 1 eV to J -> {result}")

    # 1 J to eV (reverse)
    result = _run_conversion(conv, "1 J to eV")
    ok = _check_result(result, 6.242e18, tolerance=1e-2)
    if not ok:
        all_passed = False
    print(f"  {'PASS' if ok else 'FAIL'}: 1 J to eV -> {result}")

    print()
    return all_passed


def test_natural_length():
    """Test length conversions (m ↔ eV^-1)."""
    print("=" * 60)
    print("Testing NaturalUnitsConverter: length")
    print("=" * 60)

    conv = NaturalUnitsConverter(base_directory="/tmp")
    all_passed = True

    # 1 fm to GeV^-1  (1 fm = 1e-15 m, ℏc ≈ 0.197 GeV·fm → 1 fm ≈ 5.068 GeV^-1)
    result = _run_conversion(conv, "1 fm to GeV^-1")
    ok = _check_result(result, 5.068, tolerance=1e-2)
    if not ok:
        all_passed = False
    print(f"  {'PASS' if ok else 'FAIL'}: 1 fm to GeV^-1 -> {result}")

    # 1 eV^-1 to m
    result = _run_conversion(conv, "1 eV^-1 to m")
    ok = _check_result(result, 1.9733e-7, tolerance=1e-2)
    if not ok:
        all_passed = False
    print(f"  {'PASS' if ok else 'FAIL'}: 1 eV^-1 to m -> {result}")

    print()
    return all_passed


def test_natural_time():
    """Test time conversions (s ↔ eV^-1)."""
    print("=" * 60)
    print("Testing NaturalUnitsConverter: time")
    print("=" * 60)

    conv = NaturalUnitsConverter(base_directory="/tmp")
    all_passed = True

    # 1 s to eV^-1
    result = _run_conversion(conv, "1 s to eV^-1")
    ok = _check_result(result, 1.519e15, tolerance=1e-2)
    if not ok:
        all_passed = False
    print(f"  {'PASS' if ok else 'FAIL'}: 1 s to eV^-1 -> {result}")

    # 1 eV^-1 to s
    result = _run_conversion(conv, "1 eV^-1 to s")
    ok = _check_result(result, 6.582e-16, tolerance=1e-2)
    if not ok:
        all_passed = False
    print(f"  {'PASS' if ok else 'FAIL'}: 1 eV^-1 to s -> {result}")

    print()
    return all_passed


def test_natural_frequency():
    """Test frequency conversions (eV ↔ Hz)."""
    print("=" * 60)
    print("Testing NaturalUnitsConverter: frequency")
    print("=" * 60)

    conv = NaturalUnitsConverter(base_directory="/tmp")
    all_passed = True

    # 1 eV to Hz
    result = _run_conversion(conv, "1 eV to Hz")
    ok = _check_result(result, 1.519e15, tolerance=1e-2)
    if not ok:
        all_passed = False
    print(f"  {'PASS' if ok else 'FAIL'}: 1 eV to Hz -> {result}")

    print()
    return all_passed


def test_natural_parse_and_format():
    """Test input parsing and error handling."""
    print("=" * 60)
    print("Testing NaturalUnitsConverter: parsing & errors")
    print("=" * 60)

    conv = NaturalUnitsConverter(base_directory="/tmp")
    all_passed = True

    # Missing request
    result = _run_conversion(conv, "")
    ok = "Format" in result
    if not ok:
        all_passed = False
    print(f"  {'PASS' if ok else 'FAIL'}: empty request shows format hint")

    # Bad format (no 'to')
    conv.conversion_request = "100 GeV"
    result = conv._run()
    ok = "Format" in result
    if not ok:
        all_passed = False
    print(f"  {'PASS' if ok else 'FAIL'}: missing 'to' shows format hint")

    # Unsupported conversion
    result = _run_conversion(conv, "1 barn to eV")
    ok = "not yet implemented" in result.lower() or "not recognized" in result.lower() or "Could not" in result
    if not ok:
        all_passed = False
    print(f"  {'PASS' if ok else 'FAIL'}: unsupported conversion -> {result[:60]}")

    print()
    return all_passed


# ==================== MetricPrefixConverter Tests ==================== #

def test_metric_basic():
    """Test basic metric prefix conversions."""
    print("=" * 60)
    print("Testing MetricPrefixConverter: basic conversions")
    print("=" * 60)

    conv = MetricPrefixConverter(base_directory="/tmp")
    all_passed = True

    # 1 m to nm (1e9 nm)
    result = _run_conversion(conv, "1 m to nm")
    ok = _check_result(result, 1e9)
    if not ok:
        all_passed = False
    print(f"  {'PASS' if ok else 'FAIL'}: 1 m to nm -> {result}")

    # 500 mg to kg (5e-4 kg)
    result = _run_conversion(conv, "500 mg to kg")
    ok = _check_result(result, 5e-4)
    if not ok:
        all_passed = False
    print(f"  {'PASS' if ok else 'FAIL'}: 500 mg to kg -> {result}")

    # 1 MHz to Hz (1e6 Hz)
    result = _run_conversion(conv, "1 MHz to Hz")
    ok = _check_result(result, 1e6)
    if not ok:
        all_passed = False
    print(f"  {'PASS' if ok else 'FAIL'}: 1 MHz to Hz -> {result}")

    # 1 GeV to MeV (1e3 MeV)
    result = _run_conversion(conv, "1 GeV to MeV")
    ok = _check_result(result, 1e3)
    if not ok:
        all_passed = False
    print(f"  {'PASS' if ok else 'FAIL'}: 1 GeV to MeV -> {result}")

    # 1 TeV to GeV (1e3 GeV)
    result = _run_conversion(conv, "1 TeV to GeV")
    ok = _check_result(result, 1e3)
    if not ok:
        all_passed = False
    print(f"  {'PASS' if ok else 'FAIL'}: 1 TeV to GeV -> {result}")

    print()
    return all_passed


def test_metric_mismatched_units():
    """Test that mismatched base units are rejected."""
    print("=" * 60)
    print("Testing MetricPrefixConverter: error handling")
    print("=" * 60)

    conv = MetricPrefixConverter(base_directory="/tmp")
    all_passed = True

    # Different base units
    result = _run_conversion(conv, "1 m to kg")
    ok = "cannot convert" in result.lower()
    if not ok:
        all_passed = False
    print(f"  {'PASS' if ok else 'FAIL'}: m to kg rejected -> {result[:60]}")

    # Missing 'to'
    conv.conversion_request = "1 nm"
    result = conv._run()
    ok = "Format" in result
    if not ok:
        all_passed = False
    print(f"  {'PASS' if ok else 'FAIL'}: missing 'to' shows format hint")

    print()
    return all_passed


# ==================== Runner ==================== #

def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("Unit Conversion Tool Tests")
    print("=" * 60 + "\n")

    tests = [
        ("Natural: Energy-Mass", test_natural_energy_mass),
        ("Natural: Energy-Joules", test_natural_energy_joules),
        ("Natural: Length", test_natural_length),
        ("Natural: Time", test_natural_time),
        ("Natural: Frequency", test_natural_frequency),
        ("Natural: Parsing & Errors", test_natural_parse_and_format),
        ("Metric: Basic Conversions", test_metric_basic),
        ("Metric: Error Handling", test_metric_mismatched_units),
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
