"""
# __init__.py is a part of the HEPTAPOD package.
# Copyright (C) 2025 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

Model validation toolkit: compile a generated .fr to a UFO and run structural
pass/fail checks.
"""

from .validate_tool import ValidateModelTool

__all__ = ["ValidateModelTool"]
