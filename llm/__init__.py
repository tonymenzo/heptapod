"""
# __init__.py is a part of the HEPTAPOD package.
# Copyright (C) 2025 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.
"""

"""
LLM configuration and utilities for HEP workflows.

Provides easy access to configured Ollama instances using settings from config.py.
"""

from .utils import (
    get_ollama,
    get_reasoning_ollama,
    list_available_models,
    print_config_info
)

__all__ = [
    'get_ollama',
    'get_reasoning_ollama',
    'list_available_models',
    'print_config_info'
]
