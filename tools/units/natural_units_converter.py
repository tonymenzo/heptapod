"""
# natural_units_converter.py is a part of the HEPTAPOD package.
# Copyright (C) 2025 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

Natural Units Converter Tool

Converts between natural units (ℏ=c=1) and SI (MKSA) units.
Based on the standard conversion table where energy (eV) is the base unit.
"""

from orchestral.tools.base.tool import BaseTool
from orchestral.tools.base.field_utils import RuntimeField, StateField
import re


class NaturalUnitsConverter(BaseTool):
    """
    Convert between natural units (ℏ=c=1) and SI units.

    Natural units are commonly used in particle physics where ℏ=c=1,
    reducing all dimensions to powers of energy (eV).

    Supported conversions (bidirectional):
        - Energy: eV ↔ J (Joules)
        - Mass: eV ↔ kg (via E=mc²)
        - Length: eV^-1 ↔ m, fm, nm
        - Time: eV^-1 ↔ s, ns, fs
        - Momentum: eV ↔ kg·m/s
        - Force: eV^2 ↔ N
        - Power: eV^2 ↔ W
        - Frequency: eV ↔ Hz

    Usage:
        converter = NaturalUnitsConverter(base_directory="/path/to/dir")
        result = converter("100 GeV to kg")
        # Returns: "100 GeV = 1.782663e-25 kg"
    """

    # Runtime field: the conversion request string
    conversion_request: str = RuntimeField(
        default=None,
        description=(
            "Conversion request in format '<value> <unit> to <target_unit>'. "
            "Examples: '100 GeV to kg', '1e-25 kg to GeV', '1 fm to GeV^-1'."
        )
    )

    # State field: base directory
    base_directory: str = StateField(description="Base working directory")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "NaturalUnitsConverter"
        self.description = (
            "Convert between natural units (ℏ=c=1) and SI (MKSA) units. "
            "Natural units use energy (eV) as the base unit. "
            "Input format: '<value> <unit> to <target_unit>' "
            "Examples: '100 GeV to kg', '1e-25 kg to GeV', '1 fm to GeV^-1'. "
            "Supports bidirectional: length, mass, time, frequency, momentum, energy, force, power."
        )

        # Conversion factors from natural to SI (from standard table)
        # Format: {quantity: (natural_unit_str, si_per_natural_unit, si_unit)}
        self.conversions = {
            'length': ('1/eV', 1.9732705e-7, 'm'),        # ℏc/eV
            'mass': ('eV', 1.7826627e-36, 'kg'),          # eV/c²
            'time': ('1/eV', 6.5821220e-16, 's'),         # ℏ/eV
            'frequency': ('eV', 1.5192669e15, 'Hz'),      # eV/ℏ
            'speed': ('1', 2.99792458e8, 'm/s'),          # c
            'momentum': ('eV', 5.3442883e-28, 'kg*m/s'),  # eV/c
            'force': ('eV^2', 8.1194003e-13, 'N'),        # eV²/ℏc
            'power': ('eV^2', 2.4341350e-4, 'W'),         # eV²/ℏ
            'energy': ('eV', 1.6021773e-19, 'J'),         # eV
        }

        # SI unit prefixes (case-sensitive to distinguish M=mega from m=milli)
        self.prefixes = {
            'T': 1e12, 'G': 1e9, 'M': 1e6, 'k': 1e3,
            'm': 1e-3, 'μ': 1e-6, 'u': 1e-6, 'n': 1e-9, 'p': 1e-12, 'f': 1e-15,
        }

        # Map common units to physical quantities
        self.unit_map = {
            'eV': 'energy', 'J': 'energy',
            'm': 'length', 'fm': 'length', 'nm': 'length',
            'kg': 'mass', 'g': 'mass',
            's': 'time', 'ns': 'time', 'fs': 'time',
            'Hz': 'frequency',
            'N': 'force',
            'W': 'power',
        }

    def parse_value_unit(self, text):
        """
        Parse a value-unit string into components.

        Args:
            text: String like "100 GeV" or "1e-25 kg"

        Returns:
            Tuple of (value, unit) or (None, None) if parsing fails
        """
        match = re.match(
            r'([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\s*(\S+)',
            text.strip()
        )
        if match:
            return float(match.group(1)), match.group(2)
        return None, None

    def apply_prefix(self, value, unit):
        """
        Handle SI prefixes (G, M, k, m, μ, n, p, f).

        Args:
            value: Numerical value
            unit: Unit string (e.g., "GeV", "fm")

        Returns:
            Tuple of (scaled_value, base_unit)
        """
        # Special cases: units that shouldn't be split into prefix + base
        # kg: SI base unit for mass
        # m, s, g, J, W, N, etc.: base units that match prefix letters
        no_prefix_units = {'kg', 'm', 's', 'g', 'j', 'w', 'n', 'hz'}
        if unit.lower() in no_prefix_units:
            return value, unit

        # Check if this is an inverse unit (e.g., GeV^-1)
        is_inverse = '^-1' in unit or '**-1' in unit

        for prefix, factor in self.prefixes.items():
            if unit.startswith(prefix):
                base_unit = unit[len(prefix):]
                # Only apply prefix if there's a base unit left
                if base_unit:
                    # For inverse units, invert the factor
                    # e.g., 1 GeV^-1 = 1e-9 eV^-1 (not 1e9)
                    if is_inverse:
                        return value / factor, base_unit
                    else:
                        return value * factor, base_unit
        return value, unit

    def _run(self) -> str:
        """
        Convert between units.

        Returns:
            String with conversion result or error message
        """
        try:
            if not self.conversion_request:
                return (
                    "Format: '<value> <unit> to <target_unit>'. "
                    "Example: '100 GeV to kg'"
                )

            # Parse input: "100 GeV to kg"
            # Split by ' to ' case-insensitively, but preserve original case for prefix matching
            parts = self.conversion_request.split(' to ')
            if len(parts) != 2:
                parts = self.conversion_request.split(' TO ')
            if len(parts) != 2:
                return (
                    "Format: '<value> <unit> to <target_unit>'. "
                    "Example: '100 GeV to kg'"
                )

            from_str, to_unit_orig = parts[0].strip(), parts[1].strip()
            value, from_unit_orig = self.parse_value_unit(from_str)

            if value is None:
                return f"Could not parse value from '{from_str}'"

            # Apply prefixes with original case (GeV → eV with scaling, etc.)
            value, from_unit = self.apply_prefix(value, from_unit_orig)
            # Lowercase base unit for comparison
            from_unit = from_unit.lower()

            # Special case: eV^-1 and eV^2 notation (keep lowercase for comparison)
            if 'ev^-1' in from_unit or 'ev**-1' in from_unit:
                from_unit = 'ev^-1'
            if 'ev^2' in from_unit or 'ev**2' in from_unit:
                from_unit = 'ev^2'

            # Get target prefix scaling (preserve case for prefix, lowercase base unit)
            to_val, to_unit_clean = self.apply_prefix(1.0, to_unit_orig)
            to_unit_clean = to_unit_clean.lower()

            # ============================================================
            # Energy ↔ Mass conversions (E = mc² → E in natural units)
            # ============================================================
            if from_unit == 'ev' and to_unit_clean == 'kg':
                # eV → kg
                result = value * 1.7826627e-36 / to_val
                return f"{value} eV = {result:.6e} {to_unit_orig}"

            elif from_unit == 'kg' and 'ev' in to_unit_clean:
                # kg → eV (reverse)
                result = value / 1.7826627e-36 / to_val
                return f"{value} kg = {result:.6e} {to_unit_orig}"

            # ============================================================
            # Energy unit conversions (eV ↔ J)
            # ============================================================
            elif (from_unit == 'ev' or 'ev' in from_unit) and to_unit_clean == 'j':
                # eV → J
                result = value * 1.6021773e-19 / to_val
                return f"{value} {from_unit} = {result:.6e} {to_unit_orig}"

            elif from_unit == 'j' and 'ev' in to_unit_clean:
                # J → eV (reverse)
                result = value / 1.6021773e-19 / to_val
                return f"{value} J = {result:.6e} {to_unit_orig}"

            # ============================================================
            # Length conversions (m ↔ eV^-1)
            # ============================================================
            elif from_unit in ['m', 'fm', 'nm'] and 'ev' in to_unit_clean:
                # m → eV^-1
                length_m = value * (
                    1e-15 if from_unit == 'fm'
                    else 1e-9 if from_unit == 'nm'
                    else 1
                )
                result = length_m / 1.9732705e-7 / to_val
                return f"{value} {from_unit} = {result:.6e} {to_unit_orig}"

            elif 'ev^-1' in from_unit and to_unit_clean in ['m', 'fm', 'nm']:
                # eV^-1 → m (reverse)
                result = value * 1.9732705e-7 / to_val
                return f"{value} eV^-1 = {result:.6e} {to_unit_orig}"

            # ============================================================
            # Time conversions (s ↔ eV^-1)
            # ============================================================
            elif from_unit in ['s', 'ns', 'fs'] and 'ev' in to_unit_clean:
                # s → eV^-1
                time_s = value * (
                    1e-9 if from_unit == 'ns'
                    else 1e-15 if from_unit == 'fs'
                    else 1
                )
                result = time_s / 6.5821220e-16 / to_val
                return f"{value} {from_unit} = {result:.6e} {to_unit_orig}"

            elif 'ev^-1' in from_unit and to_unit_clean in ['s', 'ns', 'fs']:
                # eV^-1 → s (reverse)
                result = value * 6.5821220e-16 / to_val
                return f"{value} eV^-1 = {result:.6e} {to_unit_orig}"

            # ============================================================
            # Momentum conversions (eV ↔ kg·m/s)
            # ============================================================
            elif from_unit == 'ev' and to_unit_clean in ['kg*m/s', 'kgm/s']:
                # eV → kg·m/s
                result = value * 5.3442883e-28 / to_val
                return f"{value} eV = {result:.6e} kg·m/s"

            elif from_unit in ['kg*m/s', 'kgm/s'] and 'ev' in to_unit_clean:
                # kg·m/s → eV (reverse)
                result = value / 5.3442883e-28 / to_val
                return f"{value} kg·m/s = {result:.6e} {to_unit_orig}"

            # ============================================================
            # Force conversions (eV^2 ↔ N)
            # ============================================================
            elif 'ev^2' in from_unit and to_unit_clean == 'n':
                # eV^2 → N
                result = value * 8.1194003e-13 / to_val
                return f"{value} eV^2 = {result:.6e} {to_unit_orig}"

            elif from_unit == 'n' and 'ev^2' in to_unit_clean:
                # N → eV^2 (reverse)
                result = value / 8.1194003e-13 / to_val
                return f"{value} N = {result:.6e} {to_unit_orig}"

            # ============================================================
            # Power conversions (eV^2 ↔ W)
            # ============================================================
            elif 'ev^2' in from_unit and to_unit_clean == 'w':
                # eV^2 → W
                result = value * 2.4341350e-4 / to_val
                return f"{value} eV^2 = {result:.6e} {to_unit_orig}"

            elif from_unit == 'w' and 'ev^2' in to_unit_clean:
                # W → eV^2 (reverse)
                result = value / 2.4341350e-4 / to_val
                return f"{value} W = {result:.6e} {to_unit_orig}"

            # ============================================================
            # Frequency conversions (eV ↔ Hz)
            # ============================================================
            elif from_unit == 'ev' and to_unit_clean == 'hz':
                # eV → Hz
                result = value * 1.5192669e15 / to_val
                return f"{value} eV = {result:.6e} {to_unit_orig}"

            elif from_unit == 'hz' and 'ev' in to_unit_clean:
                # Hz → eV (reverse)
                result = value / 1.5192669e15 / to_val
                return f"{value} Hz = {result:.6e} {to_unit_orig}"

            else:
                return (
                    f"Conversion from {from_unit} to {to_unit_orig} not yet implemented. "
                    f"Supported (bidirectional): "
                    f"energy (GeV↔kg, eV↔J), length (fm/m↔eV^-1), "
                    f"time (ns/fs↔eV^-1), momentum, force, power, frequency"
                )

        except Exception as e:
            return f"Error during conversion: {str(e)}"
