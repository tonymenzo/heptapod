"""
# sandbox_utils.py is a part of the HEPTAPOD package.
# Copyright (C) 2025 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.
"""
"""
Utilities for managing demo sandbox directories.
"""

import shutil
import re
from pathlib import Path


def create_new_sandbox(demo_files_dir: Path, mode: str = "todo") -> tuple[str, str]:
    """
    Create a new sandbox directory with the next available number.
    Copies appropriate template files based on the mode.

    Args:
        demo_files_dir: Path to hep_bsm_demo_files directory
        mode: Operating mode - "todo", "plan", or "explorer"
              - "todo": Copies template files AND todos.md, uses TODO prompt
              - "plan": Copies template files only (agent makes own plan), uses PLAN prompt
              - "explorer": Copies template files only, uses EXPLORER prompt (interactive, waits for user)

    Returns:
        tuple: (relative_path_to_sandbox, system_prompt)
    """
    # Import prompts here to avoid circular imports
    from prompts import HEP_BSM_EVT_GEN_TODO_PROMPT, HEP_BSM_EVT_GEN_PLAN_PROMPT, HEP_BSM_EVT_GEN_EXPLORER_PROMPT

    # Mode configuration
    MODE_CONFIG = {
        "todo": {
            "copy_todos": True,
            "prompt": HEP_BSM_EVT_GEN_TODO_PROMPT,
            "description": "Todo list mode"
        },
        "plan": {
            "copy_todos": False,
            "prompt": HEP_BSM_EVT_GEN_PLAN_PROMPT,
            "description": "Plan mode"
        },
        "explorer": {
            "copy_todos": False,
            "prompt": HEP_BSM_EVT_GEN_EXPLORER_PROMPT,
            "description": "Interactive explorer mode"
        }
    }

    # Validate mode
    if mode not in MODE_CONFIG:
        raise ValueError(f"Invalid mode '{mode}'. Must be one of: {', '.join(MODE_CONFIG.keys())}")

    config = MODE_CONFIG[mode]

    # Find existing sandbox directories
    existing_sandboxes = []
    for item in demo_files_dir.iterdir():
        if item.is_dir() and item.name.startswith('sandbox'):
            match = re.match(r'sandbox(\d+)', item.name)
            if match:
                existing_sandboxes.append(int(match.group(1)))

    # Determine next sandbox number
    next_num = max(existing_sandboxes) + 1 if existing_sandboxes else 1
    new_sandbox_name = f'sandbox{next_num:03d}'
    new_sandbox_path = demo_files_dir / new_sandbox_name

    # Create new sandbox directory
    new_sandbox_path.mkdir(exist_ok=True)
    print(f"Created new sandbox: {new_sandbox_name} ({config['description']})")

    # Copy template files (excluding .DS_Store files)
    template_dir = demo_files_dir / 'template'
    if template_dir.exists():
        def ignore_ds_store(directory, contents):
            """Ignore function to exclude .DS_Store files during copy."""
            return ['.DS_Store']

        for item in template_dir.iterdir():
            if item.name == '.DS_Store':
                continue
            src = item
            dst = new_sandbox_path / item.name
            if src.is_dir():
                shutil.copytree(src, dst, dirs_exist_ok=True, ignore=ignore_ds_store)
                print(f"Copied template directory: {item.name}")
            else:
                shutil.copy2(src, dst)
                print(f"Copied template file: {item.name}")

    # Copy todos.md only in "todo" mode
    if config["copy_todos"]:
        todos_src = Path(__file__).resolve().parent.parent / 'prompts' / 'demos' / 'S1_LQ' / 'todos.md'
        if todos_src.exists():
            todos_dst = new_sandbox_path / 'todos.md'
            shutil.copy2(todos_src, todos_dst)
            print(f"Copied todos.md from prompts/demos/S1_LQ to {new_sandbox_name}")
        else:
            print(f"Warning: todos.md not found at {todos_src}")

    # Return absolute path to the sandbox and the system prompt
    sandbox_path = str(new_sandbox_path)
    return sandbox_path, config["prompt"]
