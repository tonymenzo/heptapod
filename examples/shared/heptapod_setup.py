"""
# heptapod_setup.py is a part of the HEPTAPOD package.
# Copyright (C) 2025 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.
"""
"""
Shared HEPTAPOD configuration and setup utilities.

This module provides a single function to configure HEPTAPOD for notebook use,
including repository paths, environment variables, and external tool verification.

Usage:
    from examples.shared.heptapod_setup import setup_heptapod
    config = setup_heptapod()  # Prints configuration and returns config dict

    # Or silently:
    config = setup_heptapod(verbose=False)
"""

import sys
import os
from pathlib import Path
from typing import Optional


def setup_heptapod(
    verbose: bool = True,
    verify_tools: bool = True,
    notebook_depth: int = 2
) -> dict:
    """
    Configure HEPTAPOD for notebook use.

    This function:
    1. Adds the repository root to sys.path
    2. Loads environment variables from .env
    3. Verifies external tool paths from config.py
    4. Returns configuration for inspection

    Args:
        verbose: If True, print configuration details (default: True)
        verify_tools: If True, check if external tools exist (default: True)
        notebook_depth: Directory levels from notebook to repo root (default: 2)
                       e.g., examples/orchestral/ -> 2 levels up

    Returns:
        dict: Configuration containing:
            - repo_root: Path to repository root
            - feynrules_path: Path to FeynRules
            - wolframscript_path: Path to WolframScript
            - mg5_path: Path to MadGraph5
            - ollama_host: Ollama server host
            - ollama_model: Default Ollama model
            - api_keys: Dict of configured API key names (not values)

    Example:
        >>> config = setup_heptapod()
        Repository root: /path/to/heptapod
        ...
        >>> print(config['feynrules_path'])
        /path/to/FeynRules_v2.3.49
    """
    result = {}

    # -------------------------------------------------------------------------
    # 1. Repository Path Setup
    # -------------------------------------------------------------------------
    repo_root = Path.cwd()
    for _ in range(notebook_depth):
        repo_root = repo_root.parent

    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    result['repo_root'] = repo_root

    # Verify repository structure
    config_exists = (repo_root / "config.py").exists()
    requirements_exists = (repo_root / "requirements.txt").exists()

    if not config_exists:
        raise FileNotFoundError(f"config.py not found at {repo_root}")

    if verbose:
        print(f"Repository root: {repo_root}")
        print(f"  config.py: {'Found' if config_exists else 'Not found'}")
        print(f"  requirements.txt: {'Found' if requirements_exists else 'Not found'}")

    # -------------------------------------------------------------------------
    # 2. Load Configuration from config.py
    # -------------------------------------------------------------------------
    from config import (
        feynrules_path,
        wolframscript_path,
        mg5_path,
        ollama_host,
        ollama_model
    )

    result['feynrules_path'] = feynrules_path
    result['wolframscript_path'] = wolframscript_path
    result['mg5_path'] = mg5_path
    result['ollama_host'] = ollama_host
    result['ollama_model'] = ollama_model

    if verbose:
        print()
        print("External tool paths (from config.py):")
        print(f"  feynrules_path:     {feynrules_path}")
        print(f"  wolframscript_path: {wolframscript_path}")
        print(f"  mg5_path:           {mg5_path}")
        print(f"  ollama_host:        {ollama_host or 'localhost:11434 (default)'}")
        print(f"  ollama_model:       {ollama_model}")

    # -------------------------------------------------------------------------
    # 3. Verify External Tools (optional)
    # -------------------------------------------------------------------------
    if verify_tools:
        tool_status = {
            'FeynRules': Path(feynrules_path).exists() if feynrules_path else False,
            'WolframScript': Path(wolframscript_path).exists() if wolframscript_path else False,
            'MadGraph5': Path(mg5_path).exists() if mg5_path else False,
        }
        result['tool_status'] = tool_status

        if verbose:
            print()
            print("Tool availability:")
            for tool, available in tool_status.items():
                status = "Found" if available else "Not found"
                symbol = "+" if available else "-"
                print(f"  [{symbol}] {tool}: {status}")

    # -------------------------------------------------------------------------
    # 4. Load Environment Variables
    # -------------------------------------------------------------------------
    env_path = repo_root / ".env"
    api_keys_status = {}

    if env_path.exists():
        from dotenv import load_dotenv
        load_dotenv(env_path)

        # Check which API keys are configured (without revealing values)
        api_key_names = {
            "OPENAI_API_KEY": "OpenAI (GPT)",
            "ANTHROPIC_API_KEY": "Anthropic (Claude)",
            "GOOGLE_API_KEY": "Google (Gemini)",
            "GROQ_API_KEY": "Groq",
            "MISTRAL_API_KEY": "Mistral",
        }

        for key, name in api_key_names.items():
            api_keys_status[key] = bool(os.getenv(key))

        result['api_keys'] = api_keys_status

        if verbose:
            print()
            print(f"Environment (.env): Loaded from {env_path}")
            print("  API keys configured:")
            for key, name in api_key_names.items():
                status = "+" if api_keys_status.get(key) else "-"
                print(f"    [{status}] {name}")
    else:
        result['api_keys'] = {}
        if verbose:
            print()
            print(f"Environment (.env): Not found at {env_path}")
            print("  Create .env with API keys to use cloud LLM providers")

    if verbose:
        print()
        print("=" * 60)
        print("HEPTAPOD setup complete!")
        print("=" * 60)

    return result


def get_config_template() -> str:
    """
    Return a template for config.py with documentation.

    Useful for users who need to create or modify their config.py.
    """
    return '''"""
HEPTAPOD Configuration File

Edit the paths below to match your system.
"""

# Ollama LLM Configuration
ollama_host = None  # None = localhost:11434, or "http://SERVER_IP:11434"
ollama_model = "llama3:70b"  # Your preferred local model

# External Dependencies
feynrules_path = "/path/to/FeynRules_v2.3.49"
wolframscript_path = "/Applications/Mathematica.app/Contents/MacOS/wolframscript"  # macOS
# wolframscript_path = "/usr/local/bin/wolframscript"  # Linux
mg5_path = "/path/to/MG5_aMC_v3.x.x"
'''


def get_env_template() -> str:
    """
    Return a template for .env file with documentation.

    Useful for users who need to create their .env file.
    """
    return '''# HEPTAPOD Environment Variables
# Add API keys for the LLM providers you want to use

OPENAI_API_KEY=sk-...           # For GPT models
ANTHROPIC_API_KEY=sk-ant-...    # For Claude models
GOOGLE_API_KEY=...              # For Gemini models
GROQ_API_KEY=gsk_...            # For Groq (fast inference)
MISTRAL_API_KEY=...             # For Mistral models
'''


def update_config(
    new_values: dict,
    dry_run: bool = True,
    repo_root: Optional[Path] = None
) -> bool:
    """
    Update config.py with new values.

    Args:
        new_values: Dict of config variable names to new values.
            Supported keys: feynrules_path, wolframscript_path, mg5_path,
                          ollama_host, ollama_model
        dry_run: If True, only preview changes without writing (default: True)
        repo_root: Path to repository root. If None, auto-detected.

    Returns:
        bool: True if changes were made (or would be made in dry_run mode)

    Example:
        >>> update_config({
        ...     "feynrules_path": "/path/to/FeynRules",
        ...     "mg5_path": "/path/to/MG5",
        ... }, dry_run=True)
        Changes to apply:
        ...
    """
    import re

    if repo_root is None:
        repo_root = Path.cwd()
        # Try to find config.py by walking up
        while not (repo_root / "config.py").exists() and repo_root.parent != repo_root:
            repo_root = repo_root.parent

    config_path = repo_root / "config.py"
    if not config_path.exists():
        print(f"Error: config.py not found at {config_path}")
        return False

    content = config_path.read_text()

    changes = []
    for key, new_value in new_values.items():
        # Format the value as Python literal
        if new_value is None:
            value_str = "None"
        elif isinstance(new_value, str):
            value_str = f'"{new_value}"'
        else:
            value_str = repr(new_value)

        # Match the variable assignment (handles comments after the value)
        pattern = rf'^({key}\s*=\s*)([^\n#]+)(.*?)$'
        match = re.search(pattern, content, re.MULTILINE)

        if match:
            old_value = match.group(2).strip()
            if old_value != value_str:
                changes.append((key, old_value, value_str))
                content = re.sub(
                    pattern, rf'\g<1>{value_str}\g<3>', content, flags=re.MULTILINE
                )

    if not changes:
        print("No changes to make - config.py is already up to date.")
        return False

    print("Changes to apply:")
    print("-" * 70)
    for key, old, new in changes:
        print(f"  {key}:")
        print(f"    old: {old}")
        print(f"    new: {new}")
    print()

    if dry_run:
        print("DRY RUN - No changes written.")
        print("Set dry_run=False to apply changes.")
    else:
        config_path.write_text(content)
        print(f"Updated {config_path}")
        print("\nRestart kernel to reload config values.")

    return True


def update_env(
    new_values: dict,
    dry_run: bool = True,
    repo_root: Optional[Path] = None
) -> bool:
    """
    Update .env file with new API keys.

    Args:
        new_values: Dict of environment variable names to values.
            Supported keys: OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY,
                          GROQ_API_KEY, MISTRAL_API_KEY
        dry_run: If True, only preview changes without writing (default: True)
        repo_root: Path to repository root. If None, auto-detected.

    Returns:
        bool: True if changes were made (or would be made in dry_run mode)

    Example:
        >>> update_env({
        ...     "OPENAI_API_KEY": "sk-proj-...",
        ...     "ANTHROPIC_API_KEY": "sk-ant-...",
        ... }, dry_run=True)
        Changes to apply:
        ...
    """
    import re

    if repo_root is None:
        repo_root = Path.cwd()
        # Try to find .env or config.py by walking up
        while not (repo_root / "config.py").exists() and repo_root.parent != repo_root:
            repo_root = repo_root.parent

    env_path = repo_root / ".env"

    # Read existing content or start fresh
    if env_path.exists():
        content = env_path.read_text()
        existing_keys = set(re.findall(r'^([A-Z_]+)=', content, re.MULTILINE))
    else:
        content = "# HEPTAPOD Environment Variables\n\n"
        existing_keys = set()

    changes = []
    additions = []

    for key, new_value in new_values.items():
        if not new_value:  # Skip empty values
            continue

        # Mask the value for display (show first 8 and last 4 chars)
        if len(new_value) > 16:
            masked = f"{new_value[:8]}...{new_value[-4:]}"
        else:
            masked = "***"

        if key in existing_keys:
            # Update existing key
            pattern = rf'^({key}=)(.*)$'
            match = re.search(pattern, content, re.MULTILINE)
            if match:
                old_value = match.group(2)
                old_masked = f"{old_value[:8]}...{old_value[-4:]}" if len(old_value) > 16 else "***"
                if old_value != new_value:
                    changes.append((key, old_masked, masked))
                    content = re.sub(pattern, rf'\g<1>{new_value}', content, flags=re.MULTILINE)
        else:
            # Add new key
            additions.append((key, masked))
            content += f"{key}={new_value}\n"

    if not changes and not additions:
        print("No changes to make - .env is already up to date.")
        return False

    if changes:
        print("Changes to apply:")
        print("-" * 70)
        for key, old, new in changes:
            print(f"  {key}:")
            print(f"    old: {old}")
            print(f"    new: {new}")

    if additions:
        print("New keys to add:")
        print("-" * 70)
        for key, masked in additions:
            print(f"  {key}: {masked}")

    print()

    if dry_run:
        print("DRY RUN - No changes written.")
        print("Set dry_run=False to apply changes.")
    else:
        env_path.write_text(content)
        print(f"Updated {env_path}")
        print("\nRestart kernel to reload environment variables.")

    return True
