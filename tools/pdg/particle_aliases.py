"""
# particle_aliases.py is a part of the HEPTAPOD package.
# Copyright (C) 2025 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

Particle name aliases for natural language queries.

Maps common particle names and variations to the official PDG names
used by the pdg Python package.
"""

from typing import Optional, List, Tuple

# Maps natural/common names to PDG package names
PARTICLE_ALIASES = {
    # ========================================================================
    # Leptons - charged
    # ========================================================================
    "electron": "e-",
    "e": "e-",
    "e-": "e-",
    "positron": "e+",
    "e+": "e+",

    "muon": "mu-",
    "mu": "mu-",
    "mu-": "mu-",
    "antimuon": "mu+",
    "mu+": "mu+",

    "tau": "tau-",
    "tau-": "tau-",
    "tau lepton": "tau-",
    "antitau": "tau+",
    "tau+": "tau+",

    # ========================================================================
    # Leptons - neutrinos
    # ========================================================================
    "electron neutrino": "nu(e)",
    "nu_e": "nu(e)",
    "nue": "nu(e)",
    "nu(e)": "nu(e)",
    "electron antineutrino": "nu(e)bar",
    "nuebar": "nu(e)bar",

    "muon neutrino": "nu(mu)",
    "nu_mu": "nu(mu)",
    "numu": "nu(mu)",
    "nu(mu)": "nu(mu)",
    "muon antineutrino": "nu(mu)bar",
    "numubar": "nu(mu)bar",

    "tau neutrino": "nu(tau)",
    "nu_tau": "nu(tau)",
    "nutau": "nu(tau)",
    "nu(tau)": "nu(tau)",
    "tau antineutrino": "nu(tau)bar",
    "nutaubar": "nu(tau)bar",

    # ========================================================================
    # Gauge bosons
    # ========================================================================
    "photon": "gamma",
    "gamma": "gamma",
    "g": "gamma",

    "gluon": "g",  # Note: gluon may need special handling

    "w boson": "W+",
    "w+": "W+",
    "w-": "W-",
    "w": "W+",  # Default to W+
    "W": "W+",
    "W+": "W+",
    "W-": "W-",

    "z boson": "Z0",
    "z": "Z0",
    "z0": "Z0",
    "Z": "Z0",
    "Z0": "Z0",

    # ========================================================================
    # Higgs boson
    # ========================================================================
    "higgs": "H",
    "higgs boson": "H",
    "h": "H",
    "H": "H",
    "h0": "H",
    "H0": "H",

    # ========================================================================
    # Quarks
    # ========================================================================
    "up quark": "u",
    "up": "u",
    "u": "u",
    "anti-up": "ubar",
    "ubar": "ubar",

    "down quark": "d",
    "down": "d",
    "d": "d",
    "anti-down": "dbar",
    "dbar": "dbar",

    "strange quark": "s",
    "strange": "s",
    "s": "s",
    "anti-strange": "sbar",
    "sbar": "sbar",

    "charm quark": "c",
    "charm": "c",
    "c": "c",
    "anti-charm": "cbar",
    "cbar": "cbar",

    "bottom quark": "b",
    "bottom": "b",
    "b quark": "b",
    "b": "b",
    "anti-bottom": "bbar",
    "bbar": "bbar",

    "top quark": "t",
    "top": "t",
    "t quark": "t",
    "t": "t",
    "anti-top": "tbar",
    "tbar": "tbar",

    # ========================================================================
    # Light mesons
    # ========================================================================
    "pion": "pi+",
    "pi+": "pi+",
    "pi-": "pi-",
    "pi0": "pi0",
    "pi": "pi+",
    "charged pion": "pi+",
    "neutral pion": "pi0",
    "pion+": "pi+",
    "pion-": "pi-",
    "pion0": "pi0",

    "kaon": "K+",
    "k+": "K+",
    "k-": "K-",
    "k0": "K0",
    "K": "K+",
    "K+": "K+",
    "K-": "K-",
    "K0": "K0",
    "charged kaon": "K+",
    "neutral kaon": "K0",

    "eta": "eta",
    "eta'": "eta'",
    "eta prime": "eta'",

    # ========================================================================
    # Heavy mesons
    # ========================================================================
    "d meson": "D+",
    "D+": "D+",
    "D-": "D-",
    "D0": "D0",

    "b meson": "B+",
    "B+": "B+",
    "B-": "B-",
    "B0": "B0",

    "j/psi": "J/psi",
    "jpsi": "J/psi",
    "J/psi": "J/psi",

    "upsilon": "Upsilon",
    "Upsilon": "Upsilon",

    # ========================================================================
    # Baryons
    # ========================================================================
    "proton": "p",
    "p": "p",
    "antiproton": "pbar",
    "pbar": "pbar",

    "neutron": "n",
    "n": "n",
    "antineutron": "nbar",
    "nbar": "nbar",

    "lambda": "Lambda",
    "Lambda": "Lambda",

    "sigma+": "Sigma+",
    "sigma-": "Sigma-",
    "sigma0": "Sigma0",
    "Sigma+": "Sigma+",
    "Sigma-": "Sigma-",
    "Sigma0": "Sigma0",

    "xi": "Xi0",
    "xi-": "Xi-",
    "xi0": "Xi0",
    "Xi-": "Xi-",
    "Xi0": "Xi0",

    "omega-": "Omega-",
    "Omega-": "Omega-",
}


def resolve_alias(name: str) -> str:
    """
    Resolve a particle name to its PDG package name.

    First tries exact match, then case-insensitive match,
    then returns the original name if no alias found.

    Args:
        name: Particle name (may be alias or PDG name)

    Returns:
        PDG package particle name
    """
    # Exact match
    if name in PARTICLE_ALIASES:
        return PARTICLE_ALIASES[name]

    # Case-insensitive match
    name_lower = name.lower().strip()
    for alias, pdg_name in PARTICLE_ALIASES.items():
        if alias.lower() == name_lower:
            return pdg_name

    # No alias found, return original
    return name


def get_all_aliases_for(pdg_name: str) -> List[str]:
    """
    Get all aliases that map to a given PDG name.

    Args:
        pdg_name: Official PDG package particle name

    Returns:
        List of all aliases (including the PDG name itself)
    """
    aliases = [pdg_name]
    for alias, target in PARTICLE_ALIASES.items():
        if target == pdg_name and alias != pdg_name:
            aliases.append(alias)
    return aliases


def get_resolution_info(name: str) -> Tuple[str, bool, str]:
    """
    Get detailed resolution information for a particle name.

    Args:
        name: Input particle name

    Returns:
        Tuple of (resolved_name, was_aliased, alias_used)
    """
    resolved = resolve_alias(name)
    was_aliased = resolved != name
    alias_used = f"{name} -> {resolved}" if was_aliased else ""
    return resolved, was_aliased, alias_used
