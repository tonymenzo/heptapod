"""
# __init__.py is a part of the HEPTAPOD package.
# Copyright (C) 2025 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.
"""
"""MadGraph5 tools for event generation."""
from .mg5 import (
    MadGraphFromRunCardTool,
    _edit_mg5_card,
    _detect_scan_runs,
    _parse_scan_summary,
    _find_all_lhe_files,
)
