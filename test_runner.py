#!/usr/bin/env python3
"""
Comprehensive test runner for all HEPTAPOD tools.

This script runs unit tests for:
- LLM utilities (Ollama integration, local and remote connections)
- Data conversion tools (LHE to JSONL, JSONL to NumPy)
- Analysis tools (kinematics, event selection, plotting)
- FeynRules tools (UFO generation)
- MadGraph tools (event generation, LHE conversion, card editing)
- Pythia tools (event generation, jet clustering, card editing)

Usage:
    python test_runner.py                # Run all tests
    python test_runner.py --verbose      # Run with verbose output
    python test_runner.py --skip-slow    # Skip slow integration tests
    python test_runner.py --only llm     # Run only LLM tests
"""

import sys
import os
import argparse
import subprocess
from pathlib import Path

# Get repository root
REPO_ROOT = Path(__file__).resolve().parent

def print_section(title):
    """Print a formatted section header."""
    # Special formatting for the main test suite banner
    if title == "HEPTAPOD Tools Test Suite":
        print()
        print("╔" + "═" * 68 + "╗")
        print("║" + " " * 68 + "║")
        print("║" + title.center(68) + "║")
        print("║" + " " * 68 + "║")
        print("╚" + "═" * 68 + "╝")
        print()
    else:
        # Standard formatting for other sections
        width = 70
        print("\n" + "=" * width)
        print(f"  {title}")
        print("=" * width + "\n")

def check_prerequisites():
    """
    Check that all prerequisites are met before running tests.

    Returns:
        Tuple of (all_ok: bool, has_ollama: bool)
        - all_ok: True if all prerequisites met, False otherwise
        - has_ollama: True if Ollama is available and running
    """
    print_section("Checking Prerequisites")

    all_ok = True
    has_ollama = False

    # Check Python version (3.12 or 3.13)
    version_info = sys.version_info
    major, minor = version_info.major, version_info.minor

    print(f">> Python version: {major}.{minor}.{version_info.micro}")

    if major != 3:
        print(f"   [✗] Python 3.x required, found Python {major}.{minor}")
        all_ok = False
    elif minor < 12:
        print(f"   [✗] Python 3.12+ required, found Python {major}.{minor}")
        all_ok = False
    elif minor >= 14:
        print(f"   [✗] Python 3.14+ not supported (pythia8mc), found Python {major}.{minor}")
        all_ok = False
    else:
        print(f"   [✓] Python {major}.{minor} is supported")

    # Check orchestral-ai installation
    print("\n>> Checking orchestral-ai installation...")
    try:
        import orchestral
        from orchestral import Agent
        from orchestral.llm import GPT, Claude
        from orchestral.tools import RunCommandTool
        print(f"   [✓] orchestral package found")
    except ImportError as e:
        print(f"   [✗] orchestral package not found!")
        print(f"   Please install: pip install orchestral-ai")
        all_ok = False

    # Check LLM availability (need at least one: API keys OR Ollama)
    print("\n>> Checking LLM configuration...")

    # First, check for API keys in .env
    has_api_keys = False
    env_file = REPO_ROOT / ".env"
    if env_file.exists():
        print(f"   [✓] .env file found")

        with open(env_file, 'r') as f:
            env_lines = f.readlines()

        # Check for all possible API keys
        api_keys = {
            'OPENAI_API_KEY': ('OpenAI (GPT models)', 'sk-'),
            'ANTHROPIC_API_KEY': ('Anthropic (Claude models)', 'sk-ant-'),
            'GOOGLE_API_KEY': ('Google (Gemini models)', 'AIza'),
            'GROQ_API_KEY': ('Groq', 'gsk_'),
            'DEEPSEEK_API_KEY': ('DeepSeek', 'sk-'),
        }

        # Check for local LLM endpoints (no API key needed)
        local_endpoints = {
            'OLLAMA_ENDPOINT': 'Ollama (local models)',
        }

        found_keys = []
        found_endpoints = []

        for line in env_lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            # Check for API keys
            for key, (description, prefix) in api_keys.items():
                if line.startswith(f'{key}='):
                    # Extract the value after the =
                    value = line.split('=', 1)[1].strip()
                    # Remove quotes if present
                    value = value.strip('"\'')

                    # Check if it's a placeholder (contains "your_", "placeholder", etc.)
                    is_placeholder = any(placeholder in value.lower() for placeholder in
                                       ['your_', 'placeholder', 'xxx', 'yyy', 'zzz', 'insert_', 'add_'])

                    # Check if it starts with the expected prefix or looks like a real key (long alphanumeric)
                    looks_real = value.startswith(prefix) or (len(value) > 20 and value.replace('-', '').replace('_', '').isalnum())

                    if is_placeholder or not looks_real:
                        print(f"   [⚠] {key} found but appears to be a placeholder")
                    else:
                        print(f"   [✓] {key} found ({description})")
                        found_keys.append(key)
                    break

            # Check for local endpoints
            for endpoint, description in local_endpoints.items():
                if line.startswith(f'{endpoint}='):
                    value = line.split('=', 1)[1].strip().strip('"\'')
                    # Check if it looks like a valid URL
                    if value.startswith('http://') or value.startswith('https://'):
                        print(f"   [✓] {endpoint} found ({description})")
                        found_endpoints.append(endpoint)
                    break

        has_api_keys = len(found_keys) > 0 or len(found_endpoints) > 0
        if has_api_keys and not found_keys:
            print(f"   [i] Using local LLM endpoints only (no cloud API keys found)")
    else:
        print(f"   [i] .env file not found (checking for Ollama as alternative)")

    # Check config.py
    print("\n>> Checking config.py...")
    config_file = REPO_ROOT / "config.py"
    if config_file.exists():
        print(f"   [✓] config.py found")
    else:
        print(f"   [✗] config.py not found!")
        print(f"   Please create config.py with paths to FeynRules, MG5, etc.")
        all_ok = False

    # Check Ollama availability (can substitute for API keys)
    print("\n>> Checking Ollama availability...")
    ollama_available = False
    try:
        result = subprocess.run(
            ['ollama', '--version'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            version = result.stdout.strip()
            print(f"   [✓] Ollama found: {version}")

            # Check if Ollama server is running
            try:
                result = subprocess.run(
                    ['curl', '-s', 'http://localhost:11434/api/tags'],
                    capture_output=True,
                    text=True,
                    timeout=2
                )
                if result.returncode == 0:
                    print(f"   [✓] Ollama server is running")
                    ollama_available = True
                else:
                    print(f"   [i] Ollama installed but server not running (start with: ollama serve)")
            except Exception:
                print(f"   [i] Ollama installed but server not running (start with: ollama serve)")
        else:
            print(f"   [i] Ollama not found")
    except FileNotFoundError:
        print(f"   [i] Ollama not found")
    except Exception as e:
        print(f"   [i] Could not check Ollama: {e}")

    has_ollama = ollama_available

    # Verify at least one LLM option is available
    print("\n>> Verifying LLM availability...")
    if has_api_keys or has_ollama:
        if has_api_keys and has_ollama:
            print(f"   [✓] Multiple LLM options available (API keys + Ollama)")
        elif has_api_keys:
            print(f"   [✓] LLM access via API keys")
        else:
            print(f"   [✓] LLM access via Ollama")
    else:
        print(f"   [✗] No LLM available!")
        print(f"   You need at least one of:")
        print(f"     Option 1: API keys in .env (OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.)")
        print(f"     Option 2: Ollama installed and running (ollama serve)")
        all_ok = False

    # Check essential directories
    print("\n>> Checking project structure...")
    essential_dirs = ['prompts', 'tools', 'examples', 'llm']
    for dir_name in essential_dirs:
        dir_path = REPO_ROOT / dir_name
        if dir_path.exists():
            print(f"   [✓] {dir_name}/ directory found")
        else:
            print(f"   [✗] {dir_name}/ directory not found!")
            all_ok = False

    print()
    if all_ok:
        print("[✓] All prerequisites met\n")
    else:
        print("[✗] Some prerequisites missing - please address issues above\n")

    return all_ok, has_ollama

def run_test_script(script_path, verbose=False, keep_files=False, description=None):
    """
    Run a test script and return success status.

    Args:
        script_path: Path to the test script
        verbose: If True, pass -v flag to the script
        keep_files: If True, pass --keep-files flag to the script
        description: Optional description of what's being tested

    Returns:
        True if tests passed, False otherwise
    """
    if description:
        print(f">> {description}")

    # Build command
    cmd = [sys.executable, str(script_path)]
    if verbose:
        cmd.append("-v")
    if keep_files:
        cmd.append("--keep-files")

    print(f"   Running: {' '.join(cmd)}\n")

    try:
        result = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            capture_output=False,  # Show output in real-time
            text=True
        )

        if result.returncode == 0:
            print(f"\n[✓] {script_path.name} passed\n")
            return True
        else:
            print(f"\n[✗] {script_path.name} failed with exit code {result.returncode}\n")
            return False

    except Exception as e:
        print(f"\n[✗] Error running {script_path.name}: {e}\n")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Run all HEPTAPOD tool tests",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose test output"
    )
    parser.add_argument(
        "--skip-slow",
        action="store_true",
        help="Skip slow integration tests (MG5, Pythia generation)"
    )
    parser.add_argument(
        "--skip-prereq",
        action="store_true",
        help="Skip prerequisites check"
    )
    parser.add_argument(
        "--only",
        choices=["prereqs", "conversions", "kinematics", "reconstruction", "delta_r_filter", "feynrules", "mg5", "pythia", "sherpa", "llm"],
        help="Run only tests for specified component (prereqs = prerequisites check only)"
    )
    parser.add_argument(
        "--keep-files",
        action="store_true",
        help="Keep test-generated files after tests complete (useful for debugging)"
    )

    args = parser.parse_args()

    print_section("HEPTAPOD Tools Test Suite")

    # Check prerequisites first (unless skipped or not in --only)
    run_prereqs = not args.skip_prereq and (not args.only or args.only == "prereqs")
    ollama_was_available = False

    if run_prereqs:
        prereqs_ok, ollama_was_available = check_prerequisites()
        if not prereqs_ok:
            print("[✗] Prerequisites check failed. Fix issues above or use --skip-prereq to continue anyway.\n")
            sys.exit(1)

        # If --only prereqs, exit after prerequisites check
        if args.only == "prereqs":
            sys.exit(0)

    # Define test scripts
    # NOTE: Order matters! conversions must run before mg5/pythia (which depend on data conversion tools)
    test_suites = {
        "llm": {
            "script": REPO_ROOT / "llm" / "test_ollama_basic.py",
            "description": "LLM utilities (Ollama integration, local and remote connections)"
        },
        "conversions": {
            "script": REPO_ROOT / "tools" / "analysis" / "test_conversions.py",
            "description": "Data conversion tools (LHE to JSONL, JSONL to NumPy, schema validation)"
        },
        "kinematics": {
            "script": REPO_ROOT / "tools" / "analysis" / "test_kinematics.py",
            "description": "Analysis tools (invariant mass, pT, Delta R, cuts, hardest-N, plotting)"
        },
        "reconstruction": {
            "script": REPO_ROOT / "tools" / "analysis" / "test_reconstruction.py",
            "description": "Resonance reconstruction tool (template-based, multi-array merging, two-body symmetric)"
        },
        "delta_r_filter": {
            "script": REPO_ROOT / "tools" / "analysis" / "test_delta_r_filter.py",
            "description": "Delta R filter tool (overlap removal, isolation, multi-array filtering)"
        },
        "feynrules": {
            "script": REPO_ROOT / "tools" / "feynrules" / "test_feynrules.py",
            "description": "FeynRules tools (UFO generation, path resolution, parameter validation)"
        },
        "mg5": {
            "script": REPO_ROOT / "tools" / "mg5" / "test_mg5.py",
            "description": "MadGraph tools (card editing, event generation, LHE conversion)"
        },
        "pythia": {
            "script": REPO_ROOT / "tools" / "pythia" / "test_pythia.py",
            "description": "Pythia tools (card editing, event generation, jet clustering)"
        },
        "sherpa": {
            "script": REPO_ROOT / "tools" / "sherpa" / "test_sherpa.py",
            "description": "Sherpa tools (card editing, event generation)"
        },
    }

    # Track results
    results = {}

    # Run selected tests
    for component, config in test_suites.items():
        # Skip if --only specified and this isn't it
        if args.only and args.only != component:
            continue
        # Skip slow tests if requested (utils is fast, so don't skip it)
        if args.skip_slow and component in ["feynrules", "mg5", "pythia", "sherpa"]:
            print(f"⊘ Skipping {component} tests (--skip-slow)\n")
            continue

        # Handle LLM tests based on what was detected in prerequisites
        if component == "llm" and not args.only:
            # If Ollama was available during prereqs, run the test (it should pass)
            # If Ollama was NOT available during prereqs, skip the test gracefully
            if not ollama_was_available:
                print(f"⊘ Skipping {component} tests (no Ollama detected in prerequisites)\n")
                print(f"   To enable: Install Ollama and run 'ollama serve'\n")
                continue
            # Otherwise, run the test - it should pass since Ollama was available

        script_path = config["script"]
        description = config["description"]

        if not script_path.exists():
            print(f"[⚠] Warning: Test script not found: {script_path}\n")
            results[component] = False
            continue

        print_section(f"Testing: {component.upper()}")
        results[component] = run_test_script(
            script_path,
            verbose=args.verbose,
            keep_files=args.keep_files,
            description=description
        )

    # Print summary
    print_section("Test Summary")

    total = len(results)
    passed = sum(1 for v in results.values() if v)
    failed = total - passed

    for component, success in results.items():
        status = "[✓] PASS" if success else "[✗] FAIL"
        print(f"  {component:15} {status}")

    print(f"\n  Total: {passed}/{total} passed")

    if failed > 0:
        print(f"\n[⚠] {failed} test suite(s) failed\n")
        sys.exit(1)
    else:
        print("\n[✓] All tests passed!\n")
        sys.exit(0)

if __name__ == "__main__":
    main()
