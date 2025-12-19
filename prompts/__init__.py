"""
# __init__.py is a part of the HEPTAPOD package.
# Copyright (C) 2025 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.
"""
"""
System Prompts for Orchestral

Centralized location for system prompts used across demos and applications.
"""

import os

# Get the directory where this file is located
PROMPTS_DIR = os.path.dirname(os.path.abspath(__file__))


def load_prompt(filename: str) -> str:
    """
    Load a system prompt from the prompts directory.

    Args:
        filename: Name of the prompt file (e.g., 'hep_bsm_evt_gen_prompt.md')

    Returns:
        str: The prompt content
    """
    filepath = os.path.join(PROMPTS_DIR, filename)
    with open(filepath, 'r') as f:
        return f.read()


# Convenient pre-loaded prompts
HEP_BSM_EVT_GEN_TODO_PROMPT = load_prompt('demos/hep_bsm/system/hep_bsm_evt_gen_todo_prompt.md')
HEP_BSM_EVT_GEN_PLAN_PROMPT = load_prompt('demos/hep_bsm/system/hep_bsm_evt_gen_plan_prompt.md')
HEP_BSM_EVT_GEN_EXPLORER_PROMPT = load_prompt('demos/hep_bsm/system/hep_bsm_evt_gen_explorer_prompt.md')


__all__ = ['load_prompt', 'HEP_BSM_EVT_GEN_TODO_PROMPT', 'HEP_BSM_EVT_GEN_PLAN_PROMPT', 'HEP_BSM_EVT_GEN_EXPLORER_PROMPT']