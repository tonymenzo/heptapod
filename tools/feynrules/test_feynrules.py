#!/usr/bin/env python3
"""
# test_feynrules.py is a part of the HEPTAPOD package.
# Copyright (C) 2025 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.
"""
"""
FeynRules Tools Test Suite

Tests cover:
- UFO model generation from FeynRules
- Helper functions (_abs_path, _ensure_dir)
- Parameter validation
- Path resolution
- Integration with Mathematica/wolframscript
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
FEYNRULES_DIR = SCRIPT_PATH.parent                            # .../heptapod/tools/feynrules
TOOLS_DIR = FEYNRULES_DIR.parent                              # .../heptapod/tools
REPO_ROOT = TOOLS_DIR.parent                                  # .../heptapod

# Add repository root to path for local imports (prompts, tools, etc.)
sys.path.insert(0, str(REPO_ROOT))

# Tool imports
from tools.feynrules import FeynRulesToUFOTool

# Future: When orchestral-ai PyPI package is fixed, use these imports instead:
# from orchestral.tools import FeynRulesToUFOTool

# Try to import config for configuration
try:
    from config import feynrules_path, wolframscript_path
    # Check if paths are still at default placeholder values
    config_is_incomplete = (
        feynrules_path == "/path/to/FeynRules" or
        wolframscript_path == "/path/to/wolframscript"
    )
except ImportError:
    # Fallback to environment variables if config doesn't exist
    feynrules_path = os.environ.get("FEYNRULES_PATH", "/usr/local/FeynRules")
    wolframscript_path = os.environ.get("WOLFRAMSCRIPT_PATH", "wolframscript")
    config_is_incomplete = False  # Using env vars or defaults is acceptable

# Initialize base directory
base_directory = str(FEYNRULES_DIR / "test_files")

# Parse command-line arguments
parser = argparse.ArgumentParser(
    description="Run FeynRules tools test suite",
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

def test_helper_functions():
    """Test helper functions (_abs_path, _ensure_dir)."""
    print(">> Testing FeynRules helper functions...\n")

    tool = FeynRulesToUFOTool(
        base_directory=base_directory,
        feynrules_path=feynrules_path,
        wolframscript_path=wolframscript_path
    )

    # Test _abs_path with relative path
    rel_path = "models/test.fr"
    abs_path = tool._abs_path(rel_path)
    assert abs_path is not None
    assert Path(abs_path).is_absolute()
    assert "models/test.fr" in abs_path or "models\\test.fr" in abs_path
    print("[✓] Test 1 passed: _abs_path converts relative to absolute")

    # Test _abs_path with absolute path
    abs_input = "/absolute/path/test.fr"
    abs_output = tool._abs_path(abs_input)
    assert abs_output is not None
    assert Path(abs_output).is_absolute()
    print("[✓] Test 2 passed: _abs_path handles absolute paths")

    # Test _abs_path with None
    none_output = tool._abs_path(None)
    assert none_output is None
    print("[✓] Test 3 passed: _abs_path handles None input")

    # Test _ensure_dir creates directories
    test_dir = Path(base_directory) / "test_temp_dir"
    if test_dir.exists():
        import shutil
        shutil.rmtree(test_dir)

    tool._ensure_dir(str(test_dir))
    assert test_dir.exists()
    assert test_dir.is_dir()
    print("[✓] Test 4 passed: _ensure_dir creates directories")

    # Cleanup
    import shutil
    shutil.rmtree(test_dir)

    print("\nAll helper function tests passed! [✓]\n")


def test_missing_parameters():
    """Test that missing required parameters are handled correctly."""
    print(">> Testing parameter validation...\n")

    # Test 1: Model path that doesn't exist
    tool = FeynRulesToUFOTool(
        base_directory=base_directory,
        feynrules_path=feynrules_path,
        wolframscript_path=wolframscript_path,
        model_path="nonexistent/model.fr",
        output_dir="test_output"
    )

    result = tool._run()
    # Should fail because model file doesn't exist or wolframscript will fail
    assert "error" in result.lower() or "not found" in result.lower() or "failed" in result.lower()
    print("[✓] Test 1 passed: Nonexistent model file is handled")

    # Test 2: Invalid wolframscript path
    tool = FeynRulesToUFOTool(
        base_directory=base_directory,
        feynrules_path=feynrules_path,
        wolframscript_path="/nonexistent/wolframscript",
        model_path="models/test.fr",
        output_dir="test_output"
    )

    result = tool._run()
    # Should fail because wolframscript doesn't exist
    assert "error" in result.lower() or "not found" in result.lower() or "executable" in result.lower()
    print("[✓] Test 2 passed: Invalid wolframscript path is handled")

    print("\nAll parameter validation tests passed! [✓]\n")


def test_path_resolution():
    """Test that paths are correctly resolved."""
    print(">> Testing path resolution...\n")

    # Create a tool instance with relative paths
    model_rel = "models/S1_LQ_RR.fr"
    output_rel = "output/UFO_test"

    tool = FeynRulesToUFOTool(
        base_directory=base_directory,
        feynrules_path=feynrules_path,
        wolframscript_path=wolframscript_path,
        model_path=model_rel,
        output_dir=output_rel
    )

    # Check that _abs_path resolves correctly
    abs_model = tool._abs_path(model_rel)
    abs_output = tool._abs_path(output_rel)

    assert Path(abs_model).is_absolute()
    assert Path(abs_output).is_absolute()
    assert str(base_directory) in abs_model
    assert str(base_directory) in abs_output

    print("[✓] Test 1 passed: Relative paths resolved to absolute within base_directory")

    print("\nAll path resolution tests passed! [✓]\n")


def test_ufo_generation(verbose=False):
    """Test UFO generation from FeynRules model (if Mathematica is available).

    Returns:
        True if test passed, False if test failed, None if test was skipped
    """
    print(">> Testing UFO generation...\n")

    # Check if test model exists
    model_path = "models/S1_LQ_RR.fr"
    model_abs = Path(base_directory) / model_path

    if not model_abs.exists():
        print(f"[⊘] Skipping UFO generation test: model file not found at {model_abs}\n")
        return None

    # Check if wolframscript is available
    import shutil
    if not shutil.which(wolframscript_path):
        print(f"[⊘] Skipping UFO generation test: wolframscript not found at {wolframscript_path}\n")
        return None

    output_dir = "data/UFO_test_output"

    tool = FeynRulesToUFOTool(
        base_directory=base_directory,
        feynrules_path=feynrules_path,
        wolframscript_path=wolframscript_path,
        model_path=model_path,
        output_dir=output_dir,
        timeout_sec=600  # 10 minutes
    )

    if verbose:
        print(f"  Base directory: {base_directory}")
        print(f"  Model path: {model_path}")
        print(f"  Output directory: {output_dir}")
        print(f"  FeynRules path: {feynrules_path}")
        print(f"  wolframscript: {wolframscript_path}\n")

    output_raw = tool._run()

    try:
        output = json.loads(output_raw)
    except Exception:
        print(f"[✗] Failed to parse tool output")
        print(f"  Raw output: {output_raw}\n")
        return False

    if output.get("ok"):
        print("[✓] UFO generation succeeded!")
        print(f"  Output directory: {output.get('output_dir')}")
        print(f"  Files created: {len(output.get('files_created', []))}")

        if verbose and output.get('files_created'):
            print(f"\n  Generated files:")
            for fname in output['files_created'][:10]:  # Show first 10
                print(f"    - {fname}")
            if len(output['files_created']) > 10:
                print(f"    ... and {len(output['files_created']) - 10} more")

        print(f"  Logs: {output.get('logs', {}).get('stdout', 'N/A')}\n")

        # Verify some expected UFO files exist
        output_path = Path(output.get('output_dir'))
        expected_files = ['__init__.py', 'particles.py', 'vertices.py', 'couplings.py']
        found_expected = []
        for fname in expected_files:
            if (output_path / fname).exists():
                found_expected.append(fname)

        if found_expected:
            print(f"  Verified UFO files: {', '.join(found_expected)}")

        return True
    else:
        print("[✗] UFO generation failed.")
        error_info = output.get('error', output.get('reason', 'Unknown error'))
        print(f"  Error: {error_info}")

        if verbose:
            logs = output.get('logs', {})
            if logs.get('stderr'):
                print(f"  Stderr log: {logs['stderr']}")

        print()
        return False


# ============================================================================
# Cleanup Functions
# ============================================================================

def cleanup_test_files():
    """Remove all test-generated files and directories."""
    print("\n>> Cleaning up test files...\n")

    # Directories to clean up
    cleanup_dirs = [
        Path(base_directory) / "data",
        Path(base_directory) / "test_output",
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

    if cleaned == 0:
        print("[i] No test files to clean up")
    else:
        print(f"\n[✓] Cleaned up {cleaned} directory(ies)\n")


# ============================================================================
# Main Test Runner
# ============================================================================

if __name__ == "__main__":
    import sys

    print(f"Base directory: {base_directory}\n")
    print("=" * 70)

    # Run all tests and track failures
    all_passed = True

    try:
        test_helper_functions()
    except (AssertionError, Exception) as e:
        print(f"\n[✗] test_helper_functions failed: {e}\n")
        all_passed = False

    try:
        test_missing_parameters()
    except (AssertionError, Exception) as e:
        print(f"\n[✗] test_missing_parameters failed: {e}\n")
        all_passed = False

    try:
        test_path_resolution()
    except (AssertionError, Exception) as e:
        print(f"\n[✗] test_path_resolution failed: {e}\n")
        all_passed = False

    ufo_result = test_ufo_generation(verbose=args.verbose)
    # Note: UFO generation test can be skipped if dependencies not available
    # ufo_result: True = passed, False = failed, None = skipped
    if ufo_result is False:
        all_passed = False
    elif ufo_result is None and config_is_incomplete:
        # If test was skipped due to incomplete config, fail the overall suite
        missing_paths = []
        if feynrules_path == "/path/to/FeynRules":
            missing_paths.append("feynrules_path")
        if wolframscript_path == "/path/to/wolframscript":
            missing_paths.append("wolframscript_path")
        print(f"[✗] Test suite failed: {' and '.join(missing_paths)} not configured in config.py")
        print(f"    Please set the correct path(s) in config.py\n")
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
