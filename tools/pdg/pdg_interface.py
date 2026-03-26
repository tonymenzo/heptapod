"""
# pdg_interface.py is a part of the HEPTAPOD package.
# Copyright (C) 2025 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

PDG database interface wrapper.

Provides a unified, error-handled interface to the PDG Python package
with alias resolution and structured data output.
"""

import pdg
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any, Union
from .particle_aliases import resolve_alias, get_resolution_info


@dataclass
class PropertyValue:
    """A single property value with errors and limit information."""
    value: Optional[float]
    unit: str
    error_plus: Optional[float] = None
    error_minus: Optional[float] = None
    is_limit: bool = False
    limit_type: Optional[str] = None  # "upper" or "lower"
    confidence_level: Optional[float] = None
    description: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, omitting None values."""
        result = {}
        for k, v in asdict(self).items():
            if v is not None:
                result[k] = v
        return result


@dataclass
class BranchingFractionInfo:
    """A branching fraction for a decay mode."""
    decay_products: str
    fraction: Optional[float]
    error_plus: Optional[float] = None
    error_minus: Optional[float] = None
    is_limit: bool = False
    limit_type: Optional[str] = None
    pdgid: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, omitting None values."""
        result = {}
        for k, v in asdict(self).items():
            if v is not None:
                result[k] = v
        return result


@dataclass
class ParticleInfo:
    """Complete particle information from PDG."""
    name: str
    pdg_name: str
    mcid: Optional[int] = None  # Monte Carlo ID (PDG code)
    mass_gev: Optional[float] = None
    mass_error_plus: Optional[float] = None
    mass_error_minus: Optional[float] = None
    mass_is_limit: bool = False
    mass_limit_type: Optional[str] = None
    width_gev: Optional[float] = None
    width_error_plus: Optional[float] = None
    width_error_minus: Optional[float] = None
    lifetime_s: Optional[float] = None
    lifetime_error_plus: Optional[float] = None
    lifetime_error_minus: Optional[float] = None
    spin: Optional[str] = None  # J value
    parity: Optional[str] = None  # P value
    charge_conjugation: Optional[str] = None  # C value
    charge: Optional[float] = None
    isospin: Optional[str] = None
    gparity: Optional[str] = None
    aliases_used: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, omitting None values and empty lists."""
        result = {}
        for k, v in asdict(self).items():
            if v is not None and v != []:
                result[k] = v
        return result


class PDGInterface:
    """
    Interface to the PDG database with alias resolution.

    Provides methods to query particle properties, branching fractions,
    and experimental limits from the PDG database.

    Example:
        interface = PDGInterface()
        particle = interface.get_particle("electron")
        print(particle.mass_gev)  # 0.000511 GeV
    """

    def __init__(self):
        """Initialize connection to PDG database."""
        self._api = None

    @property
    def api(self):
        """Lazy initialization of PDG API connection."""
        if self._api is None:
            self._api = pdg.connect()
        return self._api

    def get_particle(self, name: str) -> ParticleInfo:
        """
        Get comprehensive particle information.

        Args:
            name: Particle name (can be alias like "electron" or PDG name like "e-")

        Returns:
            ParticleInfo with all available properties

        Raises:
            ValueError: If particle not found
        """
        pdg_name, was_aliased, alias_info = get_resolution_info(name)
        aliases_used = [alias_info] if was_aliased else []

        try:
            particle = self.api.get_particle_by_name(pdg_name)
        except Exception as e:
            # Try by Monte Carlo ID if name fails and input looks like a number
            if name.lstrip('-').isdigit():
                try:
                    particle = self.api.get_particle_by_mcid(int(name))
                except Exception:
                    raise ValueError(f"Particle not found: '{name}' (tried as '{pdg_name}'). Error: {e}")
            else:
                raise ValueError(f"Particle not found: '{name}' (tried as '{pdg_name}'). Error: {e}")

        # Extract properties
        info = ParticleInfo(
            name=name,
            pdg_name=particle.name,
            mcid=particle.mcid,
            aliases_used=aliases_used
        )

        # Mass - wrap in try/except as some properties can throw errors
        try:
            if particle.mass is not None:
                info.mass_gev = particle.mass
        except Exception:
            pass

        try:
            mass_err = getattr(particle, 'mass_error', None)
            if mass_err is not None:
                info.mass_error_plus = mass_err
                info.mass_error_minus = mass_err
        except Exception:
            pass

        # Width
        try:
            width = getattr(particle, 'width', None)
            if width is not None:
                info.width_gev = width
        except Exception:
            pass

        try:
            width_err = getattr(particle, 'width_error', None)
            if width_err is not None:
                info.width_error_plus = width_err
                info.width_error_minus = width_err
        except Exception:
            pass

        # Lifetime
        try:
            lifetime = getattr(particle, 'lifetime', None)
            if lifetime is not None:
                info.lifetime_s = lifetime
        except Exception:
            pass

        try:
            lifetime_err = getattr(particle, 'lifetime_error', None)
            if lifetime_err is not None:
                info.lifetime_error_plus = lifetime_err
                info.lifetime_error_minus = lifetime_err
        except Exception:
            pass

        # Quantum numbers
        if hasattr(particle, 'quantum_J'):
            info.spin = str(particle.quantum_J) if particle.quantum_J is not None else None
        if hasattr(particle, 'quantum_P'):
            info.parity = str(particle.quantum_P) if particle.quantum_P is not None else None
        if hasattr(particle, 'quantum_C'):
            info.charge_conjugation = str(particle.quantum_C) if particle.quantum_C is not None else None
        if hasattr(particle, 'charge'):
            info.charge = particle.charge
        if hasattr(particle, 'quantum_I'):
            info.isospin = str(particle.quantum_I) if particle.quantum_I is not None else None
        if hasattr(particle, 'quantum_G'):
            info.gparity = str(particle.quantum_G) if particle.quantum_G is not None else None

        return info

    def get_mass(self, name: str) -> PropertyValue:
        """
        Get particle mass with errors.

        Args:
            name: Particle name

        Returns:
            PropertyValue with mass in GeV
        """
        particle = self.get_particle(name)
        return PropertyValue(
            value=particle.mass_gev,
            unit="GeV",
            error_plus=particle.mass_error_plus,
            error_minus=particle.mass_error_minus,
            is_limit=particle.mass_is_limit,
            limit_type=particle.mass_limit_type,
            description=f"Mass of {particle.pdg_name}"
        )

    def get_width(self, name: str) -> PropertyValue:
        """
        Get particle decay width with errors.

        Args:
            name: Particle name

        Returns:
            PropertyValue with width in GeV
        """
        particle = self.get_particle(name)
        return PropertyValue(
            value=particle.width_gev,
            unit="GeV",
            error_plus=particle.width_error_plus,
            error_minus=particle.width_error_minus,
            description=f"Decay width of {particle.pdg_name}"
        )

    def get_lifetime(self, name: str) -> PropertyValue:
        """
        Get particle lifetime with errors.

        Args:
            name: Particle name

        Returns:
            PropertyValue with lifetime in seconds
        """
        particle = self.get_particle(name)
        return PropertyValue(
            value=particle.lifetime_s,
            unit="s",
            error_plus=particle.lifetime_error_plus,
            error_minus=particle.lifetime_error_minus,
            description=f"Lifetime of {particle.pdg_name}"
        )

    def get_branching_fractions(
        self, name: str, limit: int = 50, include_inclusive: bool = False
    ) -> List[BranchingFractionInfo]:
        """
        Get branching fractions for particle decays.

        Args:
            name: Particle name
            limit: Maximum number of branching fractions to return
            include_inclusive: Include inclusive branching fractions

        Returns:
            List of BranchingFractionInfo objects
        """
        pdg_name = resolve_alias(name)

        try:
            particle = self.api.get_particle_by_name(pdg_name)
        except Exception as e:
            raise ValueError(f"Particle not found: '{name}' (tried as '{pdg_name}'). Error: {e}")

        results = []

        # Exclusive branching fractions
        try:
            for bf in particle.exclusive_branching_fractions():
                if len(results) >= limit:
                    break

                bf_info = BranchingFractionInfo(
                    decay_products=bf.description if hasattr(bf, 'description') else str(bf),
                    fraction=bf.value if hasattr(bf, 'value') else None,
                    pdgid=bf.pdgid if hasattr(bf, 'pdgid') else None
                )

                # Check for errors
                if hasattr(bf, 'error_positive') and bf.error_positive is not None:
                    bf_info.error_plus = bf.error_positive
                if hasattr(bf, 'error_negative') and bf.error_negative is not None:
                    bf_info.error_minus = bf.error_negative

                # Check for limits
                if hasattr(bf, 'is_limit') and bf.is_limit:
                    bf_info.is_limit = True
                    if hasattr(bf, 'is_upper_limit') and bf.is_upper_limit:
                        bf_info.limit_type = "upper"
                    else:
                        bf_info.limit_type = "lower"

                results.append(bf_info)
        except Exception:
            pass  # Some particles may not have branching fractions

        # Inclusive branching fractions
        if include_inclusive:
            try:
                for bf in particle.inclusive_branching_fractions():
                    if len(results) >= limit:
                        break

                    bf_info = BranchingFractionInfo(
                        decay_products=bf.description if hasattr(bf, 'description') else str(bf),
                        fraction=bf.value if hasattr(bf, 'value') else None,
                        pdgid=bf.pdgid if hasattr(bf, 'pdgid') else None
                    )
                    results.append(bf_info)
            except Exception:
                pass

        return results

    def search_particles(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Search for particles matching a query.

        This performs a simple substring search on particle names.

        Args:
            query: Search string
            limit: Maximum results to return

        Returns:
            List of matching particle info dicts
        """
        results = []
        query_lower = query.lower()

        try:
            for particle_list in self.api.get_particles():
                if len(results) >= limit:
                    break

                # get_particles returns PdgParticleList, need to iterate
                try:
                    for p in particle_list.particles:
                        if query_lower in p.name.lower():
                            results.append({
                                "name": p.name,
                                "mcid": p.mcid,
                                "mass_gev": p.mass
                            })
                            if len(results) >= limit:
                                break
                except Exception:
                    # Try accessing as single particle
                    try:
                        p = particle_list.particle
                        if query_lower in p.name.lower():
                            results.append({
                                "name": p.name,
                                "mcid": p.mcid,
                                "mass_gev": p.mass
                            })
                    except Exception:
                        pass
        except Exception:
            pass

        return results

    def get_property_by_pdgid(self, pdgid: str) -> Dict[str, Any]:
        """
        Get a specific property by its PDG identifier.

        This uses the PDG API's get() method for direct PDGID queries.

        Args:
            pdgid: PDG identifier (e.g., "S126M" for Higgs mass)

        Returns:
            Dictionary with property information
        """
        try:
            items = self.api.get(pdgid)
            if not items:
                raise ValueError(f"No data found for PDGID: {pdgid}")

            item = items[0]
            result = {
                "pdgid": pdgid,
                "description": getattr(item, 'description', None),
                "value": getattr(item, 'value', None),
                "unit": getattr(item, 'unit', None)
            }

            if hasattr(item, 'error_positive'):
                result["error_plus"] = item.error_positive
            if hasattr(item, 'error_negative'):
                result["error_minus"] = item.error_negative
            if hasattr(item, 'is_limit') and item.is_limit:
                result["is_limit"] = True
                result["limit_type"] = "upper" if getattr(item, 'is_upper_limit', False) else "lower"

            return result
        except Exception as e:
            raise ValueError(f"Failed to get PDGID '{pdgid}': {e}")


# Module-level convenience functions
_interface = None


def get_interface() -> PDGInterface:
    """Get or create the singleton PDG interface."""
    global _interface
    if _interface is None:
        _interface = PDGInterface()
    return _interface


def get_particle(name: str) -> ParticleInfo:
    """Get particle info using the singleton interface."""
    return get_interface().get_particle(name)


def get_mass(name: str) -> PropertyValue:
    """Get particle mass using the singleton interface."""
    return get_interface().get_mass(name)


def get_lifetime(name: str) -> PropertyValue:
    """Get particle lifetime using the singleton interface."""
    return get_interface().get_lifetime(name)


def get_width(name: str) -> PropertyValue:
    """Get particle width using the singleton interface."""
    return get_interface().get_width(name)


def get_branching_fractions(name: str, limit: int = 50) -> List[BranchingFractionInfo]:
    """Get branching fractions using the singleton interface."""
    return get_interface().get_branching_fractions(name, limit=limit)
