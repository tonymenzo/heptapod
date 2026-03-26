"""
# metric_prefix_converter.py is a part of the HEPTAPOD package.
# Copyright (C) 2025 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

Metric Prefix Converter Tool

Converts between SI metric prefixes for the same base unit.
For example: meters ↔ nanometers, grams ↔ kilograms, seconds ↔ milliseconds.
"""

from orchestral.tools.base.tool import BaseTool
from orchestral.tools.base.field_utils import RuntimeField, StateField
import re


class MetricPrefixConverter(BaseTool):
    """
    Convert between SI metric prefixes for the same base unit.

    This tool handles conversions within the same unit type using
    standard SI prefixes (yotta to yocto).

    Supported base units:
        - Length: m (meter)
        - Mass: g (gram)
        - Time: s (second)
        - Electric current: A (ampere)
        - Amount: mol (mole)
        - Luminous intensity: cd (candela)
        - Frequency: Hz (hertz)
        - Force: N (newton)
        - Energy: J (joule)
        - Power: W (watt)
        - Pressure: Pa (pascal)
        - Electric charge: C (coulomb)
        - Voltage: V (volt)
        - Capacitance: F (farad)
        - Resistance: Ω, ohm (ohm)
        - Magnetic flux: Wb (weber)
        - Magnetic field: T (tesla)
        - Inductance: H (henry)
        - Byte: B (byte)
        - Electronvolt: eV (electronvolt)

    Usage:
        converter = MetricPrefixConverter(base_directory="/path/to/dir")
        result = converter("1 m to nm")
        # Returns: "1 m = 1.000000e+09 nm"
    """

    # Runtime field: the conversion request string
    conversion_request: str = RuntimeField(
        default=None,
        description=(
            "Conversion request in format '<value> <unit> to <target_unit>'. "
            "Examples: '1 m to nm', '500 mg to kg', '1 MHz to Hz'."
        )
    )

    # State field: base directory (inherited pattern)
    base_directory: str = StateField(description="Base working directory")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "MetricPrefixConverter"
        self.description = (
            "Convert between SI metric prefixes for the same base unit. "
            "Input format: '<value> <unit> to <target_unit>'. "
            "Examples: '1 m to nm', '500 mg to kg', '1 MHz to Hz', '1 GB to MB'. "
            "Supports all SI prefixes from yotta (10^24) to yocto (10^-24)."
        )

        # SI prefixes: symbol -> (name, power of 10)
        self._prefixes = {
            'Y': ('yotta', 24),
            'Z': ('zetta', 21),
            'E': ('exa', 18),
            'P': ('peta', 15),
            'T': ('tera', 12),
            'G': ('giga', 9),
            'M': ('mega', 6),
            'k': ('kilo', 3),
            'h': ('hecto', 2),
            'da': ('deca', 1),
            '': ('', 0),  # base unit (no prefix)
            'd': ('deci', -1),
            'c': ('centi', -2),
            'm': ('milli', -3),
            'μ': ('micro', -6),
            'u': ('micro', -6),  # ASCII alternative for micro
            'n': ('nano', -9),
            'p': ('pico', -12),
            'f': ('femto', -15),
            'a': ('atto', -18),
            'z': ('zepto', -21),
            'y': ('yocto', -24),
        }

        # Recognized base units (case-sensitive)
        self._base_units = {
            # SI base units
            'm', 'g', 's', 'A', 'K', 'mol', 'cd',
            # SI derived units
            'Hz', 'N', 'Pa', 'J', 'W', 'C', 'V', 'F', 'S',
            'Wb', 'T', 'H', 'lm', 'lx', 'Bq', 'Gy', 'Sv', 'kat',
            # Common alternatives
            'ohm', 'Ω',
            # Computing
            'B', 'b',  # byte, bit
            # Physics
            'eV',  # electronvolt
            # Common derived
            'L', 'l',  # liter
        }

        # Build prefix lookup by length (longer prefixes first for correct matching)
        self._sorted_prefixes = sorted(
            self._prefixes.keys(),
            key=lambda x: -len(x)
        )

    def _parse_value_unit(self, text):
        """
        Parse a value-unit string into components.

        Args:
            text: String like "100 nm" or "1e-9 m"

        Returns:
            Tuple of (value, unit_string) or (None, None) if parsing fails
        """
        match = re.match(
            r'([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\s*(\S+)',
            text.strip()
        )
        if match:
            return float(match.group(1)), match.group(2)
        return None, None

    def _split_prefix_unit(self, unit_str):
        """
        Split a unit string into prefix and base unit.

        Args:
            unit_str: Unit string like "nm", "kg", "MHz"

        Returns:
            Tuple of (prefix, base_unit) or (None, None) if not recognized
        """
        # Try to match known base units first (some are multi-character)
        for base in sorted(self._base_units, key=lambda x: -len(x)):
            if unit_str.endswith(base):
                prefix = unit_str[:-len(base)] if len(base) < len(unit_str) else ''
                if prefix in self._prefixes:
                    return prefix, base
                # If prefix not recognized but unit_str == base, it's just the base
                if unit_str == base:
                    return '', base

        # Try matching with each prefix
        for prefix in self._sorted_prefixes:
            if prefix and unit_str.startswith(prefix):
                base = unit_str[len(prefix):]
                if base in self._base_units:
                    return prefix, base

        # Check if the whole string is a base unit
        if unit_str in self._base_units:
            return '', unit_str

        return None, None

    def _run(self) -> str:
        """
        Convert between metric prefixes.

        Returns:
            String with conversion result or error message
        """
        try:
            if not self.conversion_request:
                return (
                    "Format: '<value> <unit> to <target_unit>'. "
                    "Example: '1 m to nm' or '500 mg to kg'"
                )

            # Parse input: "1 m to nm"
            parts = self.conversion_request.split(' to ')
            if len(parts) != 2:
                return (
                    "Format: '<value> <unit> to <target_unit>'. "
                    "Example: '1 m to nm' or '500 mg to kg'"
                )

            from_str, to_unit = parts[0].strip(), parts[1].strip()
            value, from_unit = self._parse_value_unit(from_str)

            if value is None:
                return f"Could not parse value from '{from_str}'"

            # Split into prefix and base unit
            from_prefix, from_base = self._split_prefix_unit(from_unit)
            to_prefix, to_base = self._split_prefix_unit(to_unit)

            if from_base is None:
                return (
                    f"Unrecognized unit: '{from_unit}'. "
                    f"Supported base units include: m, g, s, Hz, J, W, eV, etc."
                )

            if to_base is None:
                return (
                    f"Unrecognized unit: '{to_unit}'. "
                    f"Supported base units include: m, g, s, Hz, J, W, eV, etc."
                )

            # Check that base units match
            if from_base != to_base:
                return (
                    f"Cannot convert between different unit types: "
                    f"'{from_base}' and '{to_base}'. "
                    f"This tool converts between prefixes of the same unit. "
                    f"For cross-unit conversions (e.g., eV to kg), "
                    f"use NaturalUnitsConverter."
                )

            # Get prefix powers
            from_power = self._prefixes[from_prefix][1]
            to_power = self._prefixes[to_prefix][1]

            # Calculate conversion factor
            power_diff = from_power - to_power
            result = value * (10 ** power_diff)

            # Format output
            return f"{value} {from_unit} = {result:.6e} {to_unit}"

        except Exception as e:
            return f"Error during conversion: {str(e)}"
