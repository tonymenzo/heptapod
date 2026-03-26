# PDG Database Tool

Query the Particle Data Group (PDG) database for particle physics data including masses, widths, lifetimes, branching fractions, experimental limits, and quantum numbers.

## Overview

This tool provides agent-friendly access to official PDG data via the `pdg` Python package, which uses a bundled SQLite database (no rate limiting, works offline).

**Key Features:**
- Natural language particle names ("electron", "W boson", "Higgs")
- Automatic alias resolution to PDG names
- Masses, widths, lifetimes with uncertainties
- Branching fractions for decays
- Quantum numbers (spin, parity, charge)
- Experimental limits (e.g., photon mass limit)

## Quick Start

### Direct Python Usage

```python
from tools.pdg import get_particle, get_mass, get_branching_fractions

# Get electron mass
mass = get_mass("electron")
print(f"Electron mass: {mass.value} {mass.unit}")
# Output: Electron mass: 0.000510999 GeV

# Get all W boson properties
w = get_particle("W boson")
print(f"W mass: {w.mass_gev:.3f} GeV")
print(f"W width: {w.width_gev:.3f} GeV")
# Output: W mass: 80.377 GeV
#         W width: 2.137 GeV

# Get tau branching fractions
bfs = get_branching_fractions("tau", limit=5)
for bf in bfs:
    print(f"  {bf.decay_products}: {bf.fraction:.4f}")
```

### Agent/Tool Usage

```python
from tools.pdg import PDGDatabaseTool

tool = PDGDatabaseTool(base_directory="/tmp")
tool.particle = "Higgs"
tool.property = "mass"
result = tool._run()
print(result)
```

Output:
```json
{
  "status": "ok",
  "schema": "pdg-database-1.0",
  "query_info": {
    "particle_requested": "Higgs",
    "property_requested": "mass",
    "pdg_name": "H",
    "alias_resolved": "Higgs -> H"
  },
  "mass": {
    "value": 125.20,
    "unit": "GeV",
    "error_plus": 0.11,
    "error_minus": 0.11,
    "description": "Mass of H"
  }
}
```

## Available Tools

### PDGDatabaseTool

Main tool for querying particle properties.

**RuntimeFields:**
- `particle` (str, required): Particle name (e.g., "electron", "W boson", "tau")
- `property` (str, optional): What to retrieve:
  - `"all"` (default): All available properties
  - `"mass"`: Mass with uncertainties
  - `"width"`: Decay width
  - `"lifetime"`: Lifetime
  - `"branching_fractions"`: Decay modes
  - `"quantum_numbers"`: Spin, parity, charge, etc.
- `branching_limit` (int, optional): Max branching fractions to return (default 20)

**StateFields:**
- `base_directory` (str): Sandbox directory for file operations

### PDGSearchTool

Search for particles by name.

**RuntimeFields:**
- `query` (str, required): Search string
- `limit` (int, optional): Max results (default 20)

### PDGPropertyTool

Query specific PDG properties by PDGID.

**RuntimeFields:**
- `pdgid` (str, required): PDG identifier (e.g., "S126M" for Higgs mass)

## Particle Name Aliases

The tool supports natural language names that are automatically resolved:

| Common Name | PDG Name |
|-------------|----------|
| electron | e- |
| muon | mu- |
| tau | tau- |
| W boson | W+ |
| Z boson | Z0 |
| Higgs | H |
| photon | gamma |
| proton | p |
| neutron | n |
| pion | pi+ |
| kaon | K+ |

See `particle_aliases.py` for the complete mapping.

## Example Queries

### Get particle mass

```python
from tools.pdg import get_mass

# All of these work:
get_mass("electron")
get_mass("e-")
get_mass("ELECTRON")  # Case insensitive
```

### Get lifetime

```python
from tools.pdg import get_lifetime

muon_lt = get_lifetime("muon")
print(f"Muon lifetime: {muon_lt.value:.3e} s")
# Output: Muon lifetime: 2.197e-06 s
```

### Get branching fractions

```python
from tools.pdg import get_branching_fractions

# Top 10 tau decay modes
for bf in get_branching_fractions("tau", limit=10):
    if bf.fraction:
        print(f"{bf.decay_products}: {bf.fraction*100:.2f}%")
```

### Check experimental limits

```python
from tools.pdg import get_particle

photon = get_particle("photon")
print(f"Photon mass: {photon.mass_gev}")
# Output: Photon mass: None  (indicates massless/limit)
```

## Data Source

Data comes from the Particle Data Group (PDG) via the `pdg` Python package, which bundles the official SQLite database. The 2025 edition is currently available.

**Citation:** Review of Particle Physics, Particle Data Group

**License:** CC BY 4.0 (from 2024 onwards)

## Dependencies

- `pdg>=0.2.0` - PDG Python API package
- `SQLAlchemy>=1.4` - Database access (installed with pdg)

Install with:
```bash
pip install pdg
```

## Testing

```bash
cd tools/pdg/tests
python test_pdg_tool.py
```

## File Structure

```
tools/pdg/
├── __init__.py           # Module exports
├── pdg_tool.py           # BaseTool implementations
├── pdg_interface.py      # PDG package wrapper
├── particle_aliases.py   # Name mappings
├── README.md             # This file
└── tests/
    └── test_pdg_tool.py  # Test suite
```
