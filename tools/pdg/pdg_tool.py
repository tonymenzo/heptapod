"""
# pdg_tool.py is a part of the HEPTAPOD package.
# Copyright (C) 2025 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

PDG Database Tool for querying particle physics data.

Provides agent-friendly access to the Particle Data Group database
for masses, widths, lifetimes, branching fractions, and limits.
"""

import json
import os
from typing import Optional, Dict, Any

from orchestral.tools.base.tool import BaseTool
from orchestral.tools.base.field_utils import RuntimeField, StateField

from .pdg_interface import PDGInterface
from .particle_aliases import resolve_alias, get_resolution_info

SCHEMA_VERSION = "pdg-database-1.0"


class PDGDatabaseTool(BaseTool):
    """
    Query the Particle Data Group database for particle properties.

    This tool provides natural language access to official PDG data including:
    - Particle masses with uncertainties
    - Decay widths
    - Lifetimes
    - Branching fractions
    - Experimental limits
    - Quantum numbers (spin, parity, charge)

    Input:
        particle: Particle name (e.g., "electron", "W boson", "Higgs", "tau")
        property: What to retrieve - "mass", "width", "lifetime",
                  "branching_fractions", "quantum_numbers", or "all" (default)

    Example queries:
        - particle="electron", property="mass"
        - particle="tau", property="branching_fractions"
        - particle="W boson", property="all"
        - particle="photon", property="mass"  # Returns limit info

    Returns JSON with:
        {
            "status": "ok",
            "schema": "pdg-database-1.0",
            "particle": { ... particle properties ... },
            "query_info": { ... resolution details ... }
        }
    """

    # ======================== Runtime fields ======================== #
    particle: str = RuntimeField(
        description="Particle name (e.g., 'electron', 'W boson', 'Higgs', 'tau', 'pi+')"
    )
    property: Optional[str] = RuntimeField(
        description="Property to fetch: 'mass', 'width', 'lifetime', 'branching_fractions', 'quantum_numbers', or 'all' (default)",
        default="all"
    )
    branching_limit: Optional[int] = RuntimeField(
        description="Maximum number of branching fractions to return (default 20)",
        default=20
    )
    # ================================================================ #

    # ========================= State fields ========================= #
    base_directory: str = StateField(
        description="Base sandbox directory for file operations"
    )
    # ================================================================ #

    def _setup(self):
        """Validate and initialize."""
        self.base_directory = os.path.abspath(self.base_directory)
        self._interface = PDGInterface()

    def _run(self) -> str:
        """
        Query PDG database and return JSON result.

        Returns:
            JSON string with particle properties
        """
        try:
            result = self._query_particle()
            return json.dumps(result, indent=2)
        except Exception as e:
            return json.dumps({
                "status": "error",
                "schema": SCHEMA_VERSION,
                "reason": str(e),
                "particle_requested": self.particle,
                "property_requested": self.property
            }, indent=2)

    def _query_particle(self) -> Dict[str, Any]:
        """Execute the particle query."""
        prop = self.property or "all"
        prop_lower = prop.lower().strip()

        # Resolve alias
        pdg_name, was_aliased, alias_info = get_resolution_info(self.particle)

        # Get particle info
        particle_info = self._interface.get_particle(self.particle)

        result = {
            "status": "ok",
            "schema": SCHEMA_VERSION,
            "query_info": {
                "particle_requested": self.particle,
                "property_requested": prop,
                "pdg_name": particle_info.pdg_name,
            }
        }

        if was_aliased:
            result["query_info"]["alias_resolved"] = alias_info

        # Build response based on requested property
        if prop_lower == "all":
            result["particle"] = particle_info.to_dict()
            # Also get branching fractions for unstable particles
            if particle_info.lifetime_s is not None or particle_info.width_gev is not None:
                try:
                    bfs = self._interface.get_branching_fractions(
                        self.particle, limit=self.branching_limit or 20
                    )
                    if bfs:
                        result["branching_fractions"] = [bf.to_dict() for bf in bfs]
                except Exception:
                    pass  # Some particles may not have branching fractions

        elif prop_lower == "mass":
            mass = self._interface.get_mass(self.particle)
            result["mass"] = mass.to_dict()

        elif prop_lower == "width":
            width = self._interface.get_width(self.particle)
            result["width"] = width.to_dict()

        elif prop_lower == "lifetime":
            lifetime = self._interface.get_lifetime(self.particle)
            result["lifetime"] = lifetime.to_dict()

        elif prop_lower in ("branching_fractions", "branching", "decays"):
            bfs = self._interface.get_branching_fractions(
                self.particle, limit=self.branching_limit or 50
            )
            result["branching_fractions"] = [bf.to_dict() for bf in bfs]
            result["count"] = len(bfs)

        elif prop_lower in ("quantum_numbers", "quantum", "spin"):
            result["quantum_numbers"] = {
                "spin": particle_info.spin,
                "parity": particle_info.parity,
                "charge_conjugation": particle_info.charge_conjugation,
                "charge": particle_info.charge,
                "isospin": particle_info.isospin,
                "gparity": particle_info.gparity
            }
            # Remove None values
            result["quantum_numbers"] = {
                k: v for k, v in result["quantum_numbers"].items() if v is not None
            }

        else:
            # Unknown property
            result["status"] = "error"
            result["reason"] = f"Unknown property: '{prop}'. Valid options: mass, width, lifetime, branching_fractions, quantum_numbers, all"

        return result


class PDGSearchTool(BaseTool):
    """
    Search the PDG database for particles matching a query.

    Use this tool when you don't know the exact particle name.

    Input:
        query: Search string to match against particle names

    Returns JSON with list of matching particles.
    """

    # ======================== Runtime fields ======================== #
    query: str = RuntimeField(
        description="Search string to match against particle names"
    )
    limit: Optional[int] = RuntimeField(
        description="Maximum number of results to return (default 20)",
        default=20
    )
    # ================================================================ #

    # ========================= State fields ========================= #
    base_directory: str = StateField(
        description="Base sandbox directory for file operations"
    )
    # ================================================================ #

    def _setup(self):
        """Validate and initialize."""
        self.base_directory = os.path.abspath(self.base_directory)
        self._interface = PDGInterface()

    def _run(self) -> str:
        """Search PDG database."""
        try:
            results = self._interface.search_particles(
                self.query, limit=self.limit or 20
            )
            return json.dumps({
                "status": "ok",
                "schema": SCHEMA_VERSION,
                "query": self.query,
                "count": len(results),
                "particles": results
            }, indent=2)
        except Exception as e:
            return json.dumps({
                "status": "error",
                "schema": SCHEMA_VERSION,
                "reason": str(e),
                "query": self.query
            }, indent=2)


class PDGPropertyTool(BaseTool):
    """
    Query a specific PDG property by its PDGID.

    Use this tool when you know the exact PDG identifier for a property
    (e.g., "S126M" for Higgs mass, "S008.1" for specific measurements).

    Input:
        pdgid: PDG identifier string

    Returns JSON with the property value and metadata.
    """

    # ======================== Runtime fields ======================== #
    pdgid: str = RuntimeField(
        description="PDG identifier (e.g., 'S126M' for Higgs mass)"
    )
    # ================================================================ #

    # ========================= State fields ========================= #
    base_directory: str = StateField(
        description="Base sandbox directory for file operations"
    )
    # ================================================================ #

    def _setup(self):
        """Validate and initialize."""
        self.base_directory = os.path.abspath(self.base_directory)
        self._interface = PDGInterface()

    def _run(self) -> str:
        """Query PDG property by ID."""
        try:
            result = self._interface.get_property_by_pdgid(self.pdgid)
            return json.dumps({
                "status": "ok",
                "schema": SCHEMA_VERSION,
                "property": result
            }, indent=2)
        except Exception as e:
            return json.dumps({
                "status": "error",
                "schema": SCHEMA_VERSION,
                "reason": str(e),
                "pdgid": self.pdgid
            }, indent=2)
