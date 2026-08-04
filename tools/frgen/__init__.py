"""
# __init__.py is a part of the HEPTAPOD package.
# Copyright (C) 2025 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

FeynRules .fr model-file generation toolkit: a structured schema
(FeynRulesModel) + renderer, exposed via GenerateFeynRulesModelTool.
"""

from .frgen_tool import GenerateFeynRulesModelTool
from .frmodel import FeynRulesModel

__all__ = ["GenerateFeynRulesModelTool", "FeynRulesModel"]
