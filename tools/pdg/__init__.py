"""
# __init__.py is a part of the HEPTAPOD package.
# Copyright (C) 2025 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

PDG Database Tool - Query the Particle Data Group database.

This module provides agent-friendly access to official PDG particle physics data
including masses, widths, lifetimes, branching fractions, and experimental limits.

Quick Start:
    from tools.pdg import get_particle, get_mass, get_branching_fractions

    # Get electron mass
    mass = get_mass("electron")
    print(f"Electron mass: {mass.value} {mass.unit}")

    # Get all W boson properties
    w = get_particle("W boson")
    print(f"W mass: {w.mass_gev} GeV, width: {w.width_gev} GeV")

    # Get tau branching fractions
    bfs = get_branching_fractions("tau")
    for bf in bfs[:5]:
        print(f"  {bf.decay_products}: {bf.fraction}")

Tool Usage (for agents):
    tool = PDGDatabaseTool(base_directory="/tmp")
    tool.particle = "Higgs"
    tool.property = "mass"
    result = tool._run()  # Returns JSON
"""

# Main tools
from .pdg_tool import (
    PDGDatabaseTool,
    PDGSearchTool,
    PDGPropertyTool,
    SCHEMA_VERSION
)

# Interface and data classes
from .pdg_interface import (
    PDGInterface,
    ParticleInfo,
    PropertyValue,
    BranchingFractionInfo,
    get_interface,
    get_particle,
    get_mass,
    get_width,
    get_lifetime,
    get_branching_fractions
)

# Alias utilities
from .particle_aliases import (
    PARTICLE_ALIASES,
    resolve_alias,
    get_all_aliases_for,
    get_resolution_info
)

__all__ = [
    # Tools
    "PDGDatabaseTool",
    "PDGSearchTool",
    "PDGPropertyTool",
    "SCHEMA_VERSION",

    # Interface
    "PDGInterface",
    "ParticleInfo",
    "PropertyValue",
    "BranchingFractionInfo",
    "get_interface",
    "get_particle",
    "get_mass",
    "get_width",
    "get_lifetime",
    "get_branching_fractions",

    # Aliases
    "PARTICLE_ALIASES",
    "resolve_alias",
    "get_all_aliases_for",
    "get_resolution_info",
]
