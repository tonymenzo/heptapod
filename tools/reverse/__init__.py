"""
# __init__.py is a part of the HEPTAPOD package.
# Copyright (C) 2026 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

Reverse Lagrangian check: a blank-slate agent reconstructs the physics from a
sanitized .fr, optionally cross-checked against the source paper, packaged for
physicist sign-off.
"""

from tools.reverse.reverse_tool import ReverseLagrangianTool

__all__ = ["ReverseLagrangianTool"]
