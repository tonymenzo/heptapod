"""
# heptapod_tools.py is a part of the HEPTAPOD package.
# Copyright (C) 2025 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.
"""
"""
HEPTAPOD Tool Registry for MCP Servers

Organizes all HEPTAPOD tools into named groups for selective MCP export.
Each group is lazy-loaded so missing optional dependencies (e.g., Pythia,
Sherpa) only cause that group to be skipped rather than crashing the server.

Usage:
    from heptapod_tools import get_tools, get_available_groups

    # Get all tools that can be loaded
    tools = get_tools()

    # Get specific groups
    tools = get_tools("pdg", "units")

    # See what's available
    print(get_available_groups())

Groups:
    pdg           - Particle Data Group database queries
    inspire       - INSPIRE HEP literature search
    units         - Natural units & metric prefix conversions
    analysis      - Kinematics, reconstruction, data conversion
    event_gen     - MadGraph, Pythia, Sherpa event generation (requires external software)
    feynrules     - FeynRules-to-UFO conversion (requires Mathematica)
"""

import sys
import os
from pathlib import Path
from typing import Callable

# Repository root for imports
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Default workspace directory for tool output
DEFAULT_WORKSPACE = os.path.expanduser("~/.heptapod/mcp_workspace")


# ================================================================== #
# ======================== Group Factories ========================= #
# ================================================================== #
# Each factory takes a base_directory string and returns a list of
# instantiated BaseTool objects.  Imports happen inside the factory
# so a missing dependency only affects its own group.
# ================================================================== #


def _named(tool, display_name: str):
    """Set a custom MCP display name on a tool instance and return it."""
    tool._mcp_display_name = display_name
    return tool


def _make_pdg_tools(base_dir: str) -> list:
    """PDG database tools — particle properties, masses, widths, branching fractions."""
    from tools.pdg import PDGDatabaseTool, PDGSearchTool
    return [
        _named(PDGDatabaseTool(base_directory=base_dir),  "PDGDatabase"),
        _named(PDGSearchTool(base_directory=base_dir),     "PDGSearch"),
    ]


def _make_inspire_tools(base_dir: str) -> list:
    """INSPIRE HEP tools — literature search, citations, BibTeX, reading list."""
    from tools.inspire import (
        InspireSearchTool, InspirePaperTool, InspireCitationTool,
        InspireBibTeXTool, InspireAuthorTool, InspireReadingListTool,
        InspireNotesTool,
    )
    return [
        _named(InspireSearchTool(base_directory=base_dir),      "INSPIRESearch"),
        _named(InspirePaperTool(base_directory=base_dir),       "INSPIREPaper"),
        _named(InspireCitationTool(base_directory=base_dir),    "INSPIRECitation"),
        _named(InspireBibTeXTool(base_directory=base_dir),      "INSPIREBibTeX"),
        _named(InspireAuthorTool(base_directory=base_dir),      "INSPIREAuthor"),
        _named(InspireReadingListTool(base_directory=base_dir), "INSPIREReadingList"),
        _named(InspireNotesTool(base_directory=base_dir),       "INSPIRENotes"),
    ]


def _make_units_tools(base_dir: str) -> list:
    """Unit conversion tools — natural units, metric prefixes."""
    from tools.units import NaturalUnitsConverter, MetricPrefixConverter
    return [
        _named(NaturalUnitsConverter(base_directory=base_dir), "NaturalUnitsConverter"),
        _named(MetricPrefixConverter(base_directory=base_dir), "MetricPrefixConverter"),
    ]


def _make_analysis_tools(base_dir: str) -> list:
    """Analysis tools — kinematics, reconstruction, format conversion."""
    from tools.analysis.kinematics import (
        CalculateInvariantMassTool, CalculateTransverseMomentumTool,
        CalculateDeltaRTool, ApplyCutsTool, GetHardestNTool,
        FilterByPDGIDTool, SortByPtTool, FilterByDeltaRTool,
    )
    from tools.analysis.reconstruction import ResonanceReconstructionTool
    from tools.analysis.conversions import (
        LHEToJSONLTool, EventJSONLToNumpyTool,
    )
    return [
        _named(CalculateInvariantMassTool(base_directory=base_dir),       "CalculateInvariantMass"),
        _named(CalculateTransverseMomentumTool(base_directory=base_dir),  "CalculateTransverseMomentum"),
        _named(CalculateDeltaRTool(base_directory=base_dir),              "CalculateDeltaR"),
        _named(ApplyCutsTool(base_directory=base_dir),                    "ApplyCuts"),
        _named(GetHardestNTool(base_directory=base_dir),                  "GetHardestN"),
        _named(FilterByPDGIDTool(base_directory=base_dir),                "FilterByPDGID"),
        _named(SortByPtTool(base_directory=base_dir),                     "SortByPT"),
        _named(FilterByDeltaRTool(base_directory=base_dir),               "FilterByDeltaR"),
        _named(ResonanceReconstructionTool(base_directory=base_dir),      "ResonanceReconstruction"),
        _named(LHEToJSONLTool(base_directory=base_dir),                   "LHEToJSONL"),
        _named(EventJSONLToNumpyTool(base_directory=base_dir),            "EventJSONLToNumpy"),
    ]


def _make_event_gen_tools(base_dir: str) -> list:
    """Event generation tools — MadGraph, Pythia, Sherpa (require external software)."""
    import config
    from tools.mg5 import MadGraphFromRunCardTool
    from tools.pythia import PythiaFromRunCardTool, JetClusterSlowJetTool
    from tools.sherpa import SherpaFromRunCardTool
    return [
        _named(MadGraphFromRunCardTool(base_directory=base_dir, mg5_path=config.mg5_path), "MadGraphFromRunCard"),
        _named(PythiaFromRunCardTool(base_directory=base_dir),  "PythiaFromRunCard"),
        _named(JetClusterSlowJetTool(base_directory=base_dir),  "JetClusterSlowJet"),
        _named(SherpaFromRunCardTool(base_directory=base_dir),  "SherpaFromRunCard"),
    ]


def _make_feynrules_tools(base_dir: str) -> list:
    """FeynRules-to-UFO tool — convert .fr model files to UFO format (requires Mathematica)."""
    import config
    from tools.feynrules import FeynRulesToUFOTool
    return [
        _named(FeynRulesToUFOTool(
            base_directory=base_dir,
            feynrules_path=config.feynrules_path,
            wolframscript_path=config.wolframscript_path,
        ), "FeynRulesToUFO"),
    ]


# ================================================================== #
# ======================== Group Registry ========================== #
# ================================================================== #

TOOL_GROUPS: dict[str, Callable[[str], list]] = {
    "pdg":           _make_pdg_tools,
    "inspire":       _make_inspire_tools,
    "units":         _make_units_tools,
    "analysis":      _make_analysis_tools,
    "event_gen":     _make_event_gen_tools,
    "feynrules":     _make_feynrules_tools,
}

# Groups that work out of the box (no external software)
LIGHTWEIGHT_GROUPS = ["pdg", "inspire", "units"]


def get_available_groups(base_dir: str | None = None) -> list[str]:
    """Return names of tool groups that can be successfully imported.

    Attempts to instantiate each group and returns only those that succeed.
    """
    base_dir = _ensure_workspace(base_dir)
    available = []
    for name, factory in TOOL_GROUPS.items():
        try:
            factory(base_dir)
            available.append(name)
        except Exception:
            pass
    return available


def get_tools(*group_names: str, base_dir: str | None = None) -> list:
    """Instantiate and return tools for the requested groups.

    Args:
        *group_names: Names of tool groups to load. If none specified,
                      loads all groups that can be successfully imported.
        base_dir: Override the default workspace directory.

    Returns:
        Flat list of BaseTool instances.
    """
    base_dir = _ensure_workspace(base_dir)

    if not group_names:
        group_names = tuple(TOOL_GROUPS.keys())

    tools = []
    for name in group_names:
        if name not in TOOL_GROUPS:
            print(f"Warning: Unknown tool group '{name}'. "
                  f"Available: {', '.join(TOOL_GROUPS.keys())}", file=sys.stderr)
            continue
        try:
            tools.extend(TOOL_GROUPS[name](base_dir))
        except Exception as e:
            print(f"Warning: Could not load tool group '{name}': {e}", file=sys.stderr)

    return tools


def _ensure_workspace(base_dir: str | None) -> str:
    """Resolve and create the workspace directory."""
    base_dir = base_dir or DEFAULT_WORKSPACE
    Path(base_dir).mkdir(parents=True, exist_ok=True)
    return base_dir
