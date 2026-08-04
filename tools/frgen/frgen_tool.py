"""
# frgen_tool.py is a part of the HEPTAPOD package.
# Copyright (C) 2025 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

FeynRules .fr model-file generator tool.

Turns a STRUCTURED model spec (the FeynRulesModel schema in frmodel.py, supplied
as JSON) into a syntactically valid FeynRules .fr file. Pairs with the existing
FeynRulesToUFOTool, which compiles the .fr into a UFO for MadGraph — this tool
produces the .fr that tool consumes. Structuring the spec keeps the LLM out of
error-prone Mathematica syntax; Pydantic validation returns actionable errors an
agent can repair.
"""

import json
import os
import re
from typing import Optional

from orchestral.tools.base.tool import BaseTool
from orchestral.tools.base.field_utils import RuntimeField, StateField

from .frmodel import FeynRulesModel
from .render import render_model

SCHEMA_VERSION = "feynrules-model-1.0"


def _safe_join(base_directory: str, rel_or_abs: str) -> Optional[str]:
    """Resolve ``rel_or_abs`` under ``base_directory`` (realpath); None if it escapes."""
    if not rel_or_abs:
        return None
    base = os.path.realpath(base_directory)
    full = os.path.realpath(os.path.join(base, rel_or_abs))
    if full != base and not full.startswith(base + os.sep):
        return None
    return full


def _slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "model"


class GenerateFeynRulesModelTool(BaseTool):
    """
    Generate a FeynRules .fr model file from a structured JSON spec.

    Provide `model_json`: a JSON object matching the FeynRulesModel schema
    (see tools/frgen/frmodel.py). Key rules the JSON must follow:
      - ALL numeric values are strings, to preserve exact FeynRules syntax:
        charges/orders as rationals ("-1/3", "2/3"), decimals ("1500."),
        scientific ("2.5*^-3"). Never pass raw JSON numbers for these.
      - Enum/symbol/boolean values render bare; text (Description, TeX,
        ParticleName, FullName) is quoted automatically.
      - Lagrangian terms are carried verbatim as Mathematica strings with a
        `delayed` flag (":=" vs "=").
      - BSM add-on models (loaded on top of SM.fr) omit `gauge_groups`; a
        standalone model includes them plus `index_decls`.

    Minimal example (a scalar-leptoquark-like add-on):
      {"model_name": "MyLQ",
       "info": {"authors": ["A. Physicist"], "version": "1.0.0", "date": "2026-07-01"},
       "index_decls": [{"name": "Colour", "range_kind": "NoUnfold", "size": 3}],
       "parameters": [{"name": "yLQ", "parameter_type": "External",
                       "block_name": "BSMINPUTS", "value": "0.5",
                       "description": "LQ Yukawa"}],
       "particles": [{"spin_type": "S", "class_index": 100, "class_name": "S1",
                      "self_conjugate": false, "indices": ["Colour"],
                      "mass": {"sym": "MS1", "value": "1500."},
                      "width": {"sym": "WS1", "value": "Automatic"},
                      "quantum_numbers": {"Q": "-1/3"},
                      "particle_name": "S1", "antiparticle_name": "S1~"}],
       "lagrangian_terms": [{"name": "LBSM", "expression": "...", "delayed": false}]}

    Input:
        model_json: JSON string for the FeynRulesModel spec (see above).
        output_path: Optional .fr path relative to base_directory
                     (default "models/<model_name>.fr").

    Returns JSON:
        {"status": "ok", "schema": "feynrules-model-1.0", "model_name": "...",
         "fr_path": "models/MyLQ.fr", "n_particles": N, "n_parameters": M,
         "preview": "<first lines of the .fr>"}

    On a malformed spec, returns a formatted error listing the Pydantic
    validation problems so the caller can fix the JSON and retry.
    """

    # ======================== Runtime fields ======================== #
    model_json: str = RuntimeField(
        description="JSON object matching the FeynRulesModel schema (numbers as strings)"
    )
    output_path: Optional[str] = RuntimeField(
        default=None,
        description="Optional .fr output path relative to base_directory",
    )
    # ================================================================ #

    # ========================= State fields ========================= #
    base_directory: str = StateField(
        description="Base sandbox directory for file operations"
    )
    # ================================================================ #

    def _run(self) -> str:
        # 1. Parse JSON.
        try:
            data = json.loads(self.model_json)
        except json.JSONDecodeError as e:
            return self.format_error(
                error="Invalid JSON",
                reason=str(e),
                suggestion="model_json must be a valid JSON object for FeynRulesModel",
            )
        if not isinstance(data, dict):
            return self.format_error(
                error="Invalid Input",
                reason="model_json must be a JSON object (the FeynRulesModel spec)",
            )

        # 2. Validate against the schema.
        try:
            model = FeynRulesModel(**data)
        except Exception as e:  # pydantic ValidationError or ValueError from validators
            return self.format_error(
                error="Schema Validation Failed",
                reason=str(e),
                suggestion=(
                    "Fix the fields named above. Remember: numeric values are strings "
                    "(e.g. \"-1/3\", \"1500.\"), and see the FeynRulesModel schema."
                ),
            )

        # 3. Render.
        try:
            fr_text = render_model(model)
        except Exception as e:  # noqa: BLE001
            return self.format_error(
                error="Render Failed",
                reason=str(e),
            )

        # 4. Write into the sandbox.
        rel = self.output_path or f"models/{_slug(model.model_name)}.fr"
        dest = _safe_join(self.base_directory, rel)
        if dest is None:
            return self.format_error(
                error="Access Denied",
                reason="output_path escapes base_directory",
                context=rel,
                suggestion="Use a relative path inside base_directory",
            )
        try:
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with open(dest, "w", encoding="utf-8") as fh:
                fh.write(fr_text)
        except OSError as e:
            return self.format_error(
                error="Filesystem Error",
                reason=str(e),
                context=rel,
            )

        return json.dumps(
            {
                "status": "ok",
                "schema": SCHEMA_VERSION,
                "model_name": model.model_name,
                "fr_path": os.path.relpath(dest, os.path.realpath(self.base_directory)),
                "n_particles": len(model.particles),
                "n_parameters": len(model.parameters),
                "preview": "\n".join(fr_text.splitlines()[:40]),
            },
            indent=2,
        )
