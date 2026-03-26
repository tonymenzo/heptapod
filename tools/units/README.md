# Unit Conversion Tools

Utilities for converting between different unit systems used in particle physics.

## NaturalUnitsConverter

Convert between natural units (ℏ=c=1) and SI (MKSA) units.

### Usage

```python
from tools.units import NaturalUnitsConverter

converter = NaturalUnitsConverter(base_directory="/path/to/dir")

# Energy to mass (via E=mc²)
converter("125 GeV to kg")  # Higgs mass in kg
# Returns: "125 GeV = 2.228329e-25 kg"

# Mass to energy
converter("9.109e-31 kg to MeV")  # Electron mass in MeV
# Returns: "9.109e-31 kg = 5.109991e-01 MeV"

# Length conversions
converter("1 fm to GeV^-1")  # Proton radius scale
# Returns: "1 fm = 5.067731e+00 GeV^-1"

# Time conversions
converter("1 ns to eV^-1")
# Returns: "1 ns = 1.519267e+06 eV^-1"
```

### With LLM Agents

```python
from orchestral import Agent
from orchestral.llm import GPT
from tools.units import NaturalUnitsConverter

tools = [
    NaturalUnitsConverter(base_directory="/path/to/dir"),
    # ... other tools
]

agent = Agent(llm=GPT(), tools=tools, ...)

# Agent can now convert units naturally
response = agent.run("What is the Higgs mass in kilograms?")
# Agent will use the converter tool: "125 GeV to kg"
```

### Supported Conversions

All conversions are bidirectional:

| Quantity | Natural Units | SI Units | Example |
|----------|---------------|----------|---------|
| Energy | eV | J (Joules) | `1 GeV to J` |
| Mass | eV | kg | `100 GeV to kg` |
| Length | eV^-1 | m, fm, nm | `1 fm to GeV^-1` |
| Time | eV^-1 | s, ns, fs | `1 ns to eV^-1` |
| Momentum | eV | kg·m/s | `10 GeV to kg*m/s` |
| Force | eV^2 | N | `1 GeV^2 to N` |
| Power | eV^2 | W | `1 GeV^2 to W` |
| Frequency | eV | Hz | `1 eV to Hz` |

### Unit Prefixes

Supports standard SI prefixes:
- Large: T (tera), G (giga), M (mega), k (kilo)
- Small: m (milli), μ (micro), n (nano), p (pico), f (femto)

Examples: `TeV`, `GeV`, `MeV`, `keV`, `fm`, `nm`, `ns`, `fs`

## Future Tools

Planned additions to this module:
- Lorentz transformation utilities
- Cross-section unit conversions (pb, fb, etc.)
- Temperature/energy conversions
