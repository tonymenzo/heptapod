"""
# extract_tool.py is a part of the HEPTAPOD package.
# Copyright (C) 2025 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

Lagrangian extraction tool.

Reads the full text of a BSM paper and extracts a STRUCTURED FeynRulesModel
(frmodel.py) via schema-constrained LLM decoding (Orchestral Agent.structured).
This is the previously-manual "literature -> Lagrangian" bottleneck: the LLM
identifies fields, quantum numbers, parameters, and Lagrangian terms; the schema
guarantees the shape; and the downstream GenerateFeynRulesModelTool turns the
result into a .fr file. Provider-agnostic (Ollama / vLLM / LiteLLM) via
HEPTAPOD's llm utilities, so it runs against a local open model too.
"""

import json
import os
from typing import Optional

from orchestral.tools.base.tool import BaseTool
from orchestral.tools.base.field_utils import RuntimeField, StateField

from tools.frgen.frmodel import FeynRulesModel

SCHEMA_VERSION = "lagrangian-extraction-1.0"

_VALID_PROVIDERS = {"ollama", "vllm", "litellm"}

EXTRACTION_SYSTEM_PROMPT = """\
You are a theoretical high-energy physicist building FeynRules models. Given the
text of a Beyond-Standard-Model (BSM) paper, extract a machine-usable structured
representation of the model's new-physics Lagrangian, conforming to the provided
schema (FeynRulesModel).

Follow these rules exactly:
- Treat the model as a BSM ADD-ON loaded on top of the Standard Model unless the
  paper clearly defines a standalone theory. For an add-on: leave gauge_groups
  empty and reuse SM field/gauge names (e.g. uR, lR, QL, G, colour index Colour).
- ALL numeric values are STRINGS to preserve exact syntax: electric charges and
  hypercharges as rationals ("-1/3", "2/3"), masses/couplings as decimals
  ("1500.", "0.5") or scientific ("2.5*^-3"). NEVER emit a bare number for these.
- Each new field is a ParticleClass with spin_type S/F/V/U, a unique class_index,
  ClassName, self_conjugate, indices (bare names like "Colour"; the generator
  wraps them in Index[...]), quantum_numbers (e.g. {"Q": "-1/3"}), a MassSpec
  mass, and particle_name/antiparticle_name.
- Each new parameter is a Parameter: External (measured/input, with block_name)
  or Internal (derived, with a value expression). Yukawa/coupling constants are
  usually External.
- Carry Lagrangian terms VERBATIM as Mathematica expression strings in
  lagrangian_terms, using FeynRules idioms (Block[...], ExpandIndices, DC, HC,
  CC, Ga, ProjM/ProjP). Use delayed=true (":=") when the RHS uses Block[...] or
  references terms defined later or HC of itself; delayed=false ("=") otherwise.
  Always include a final total term (e.g. LBSM) summing the pieces.
- Declare an IndexDecl for every non-SM index you use (e.g. a colour triplet
  scalar needs Colour with range_kind NoUnfold, size 3).
- Prefer fidelity to the paper's stated conventions; if a value is not given,
  use a symbolic name and omit the numeric value rather than inventing one.
"""


def _safe_join(base_directory: str, rel_or_abs: str) -> Optional[str]:
    if not rel_or_abs:
        return None
    base = os.path.realpath(base_directory)
    full = os.path.realpath(os.path.join(base, rel_or_abs))
    if full != base and not full.startswith(base + os.sep):
        return None
    return full


def build_extraction_message(paper_text: str, scenario: Optional[str]) -> str:
    """Compose the user message for the extraction agent (pure, testable)."""
    head = "Extract the BSM Lagrangian as a FeynRulesModel."
    if scenario:
        head += f"\n\nTarget scenario / focus: {scenario}"
    return (
        f"{head}\n\n"
        "Paper text (may be truncated):\n"
        "-----BEGIN PAPER-----\n"
        f"{paper_text}\n"
        "-----END PAPER-----"
    )


def _get_llm(provider: str, model: Optional[str]):
    """Acquire a provider-agnostic Orchestral LLM via HEPTAPOD's llm utilities.

    Imported lazily so this tool imports without config.py present; only an
    actual extraction call needs a configured provider.
    """
    from llm.utils import get_litellm, get_ollama, get_vllm

    if provider == "ollama":
        return get_ollama(model=model)
    if provider == "vllm":
        return get_vllm(model=model)
    if provider == "litellm":
        return get_litellm(model=model)
    raise ValueError(f"unknown llm_provider: {provider}")


class ExtractLagrangianTool(BaseTool):
    """
    Extract a structured FeynRules model (FeynRulesModel) from BSM paper text.

    Use this after ExtractPaperTextTool to turn a paper's new-physics content into
    a validated FeynRulesModel JSON, which GenerateFeynRulesModelTool then renders
    to a .fr file. Extraction is schema-constrained, so the output is guaranteed
    to fit the schema (or a formatted error is returned).

    Input:
        paper_text: The paper's full text (inline). If omitted, text_path is read.
        text_path: Path to an extracted-text file, relative to base_directory
                   (e.g. the ExtractPaperTextTool output "text/2103.02708.txt").
        scenario: Optional BSM scenario / focus to guide extraction
                  (e.g. "scalar leptoquark S1, first generation").
        llm_provider: "ollama" (default), "vllm", or "litellm".
        model: Optional model-name override for the provider.
        max_chars: Max characters of paper text to send (default 60000).
        max_retries: Structured-decode retries on schema-validation failure (default 2).
        output_path: Optional path (relative to base_directory) to also write the
                     FeynRulesModel JSON to.

    Returns JSON:
        {"status": "ok", "schema": "lagrangian-extraction-1.0", "model_name": "...",
         "n_particles": N, "n_parameters": M, "model": { <FeynRulesModel JSON> },
         "model_json_path": "models/extracted.json"?}
    Pass the "model" object (as a JSON string) to GenerateFeynRulesModelTool.

    On failure returns a formatted error (missing text, provider unavailable, or
    the model could not produce a schema-valid extraction).
    """

    # ======================== Runtime fields ======================== #
    paper_text: Optional[str] = RuntimeField(
        default=None, description="Full paper text (inline); if omitted, text_path is read"
    )
    text_path: Optional[str] = RuntimeField(
        default=None, description="Path to extracted-text file, relative to base_directory"
    )
    scenario: Optional[str] = RuntimeField(
        default=None, description="Optional BSM scenario/focus to guide extraction"
    )
    llm_provider: Optional[str] = RuntimeField(
        default="ollama", description="'ollama' (default), 'vllm', or 'litellm'"
    )
    model: Optional[str] = RuntimeField(
        default=None, description="Optional model-name override for the provider"
    )
    max_chars: Optional[int] = RuntimeField(
        default=60000, description="Max characters of paper text to send (default 60000)"
    )
    max_retries: Optional[int] = RuntimeField(
        default=2, description="Structured-decode retries on validation failure (default 2)"
    )
    output_path: Optional[str] = RuntimeField(
        default=None, description="Optional path to also write the FeynRulesModel JSON"
    )
    # ================================================================ #

    # ========================= State fields ========================= #
    base_directory: str = StateField(
        description="Base sandbox directory for file operations"
    )
    # ================================================================ #

    def _run(self) -> str:
        provider = (self.llm_provider or "ollama").lower()
        if provider not in _VALID_PROVIDERS:
            return self.format_error(
                error="Invalid Parameter",
                reason=f"llm_provider must be one of {sorted(_VALID_PROVIDERS)}",
                context=f"llm_provider={self.llm_provider}",
            )

        # 1. Resolve paper text.
        text = self.paper_text
        if not text:
            if not self.text_path:
                return self.format_error(
                    error="Missing Parameter",
                    reason="Provide either paper_text or text_path",
                )
            src = _safe_join(self.base_directory, self.text_path)
            if src is None:
                return self.format_error(
                    error="Access Denied",
                    reason="text_path escapes base_directory",
                    context=self.text_path,
                )
            if not os.path.exists(src):
                return self.format_error(
                    error="File Not Found",
                    reason="text file does not exist",
                    context=self.text_path,
                )
            try:
                with open(src, "r", encoding="utf-8", errors="replace") as fh:
                    text = fh.read()
            except OSError as e:
                return self.format_error(error="Filesystem Error", reason=str(e))

        if not text.strip():
            return self.format_error(
                error="Empty Input", reason="paper text is empty"
            )
        text = text[: self.max_chars or 60000]

        # 2. Acquire the LLM (lazy; needs config.py + provider reachable).
        try:
            llm = _get_llm(provider, self.model)
        except Exception as e:  # noqa: BLE001 - ImportError (no config), conn errors, etc.
            return self.format_error(
                error="LLM Unavailable",
                reason=str(e),
                suggestion=(
                    "Configure config.py (ollama_model / vllm_host+model / "
                    "litellm_host+model) and ensure the provider is reachable."
                ),
            )

        # 3. Schema-constrained extraction.
        message = build_extraction_message(text, self.scenario)
        try:
            from orchestral import Agent, StructuredOutputError

            agent = Agent(llm=llm, system_prompt=EXTRACTION_SYSTEM_PROMPT)
            extracted: FeynRulesModel = agent.structured(
                message, FeynRulesModel, max_retries=self.max_retries or 2
            )
        except StructuredOutputError as e:
            return self.format_error(
                error="Extraction Failed",
                reason=str(e),
                suggestion=(
                    "The model could not produce a schema-valid FeynRulesModel. "
                    "Try a stronger model, narrow the scenario, or raise max_retries."
                ),
            )
        except Exception as e:  # noqa: BLE001
            return self.format_error(error="Extraction Error", reason=str(e))

        model_dict = json.loads(extracted.model_dump_json())

        result = {
            "status": "ok",
            "schema": SCHEMA_VERSION,
            "model_name": extracted.model_name,
            "n_particles": len(extracted.particles),
            "n_parameters": len(extracted.parameters),
            "model": model_dict,
        }

        # 4. Optionally persist the FeynRulesModel JSON.
        if self.output_path:
            dest = _safe_join(self.base_directory, self.output_path)
            if dest is None:
                return self.format_error(
                    error="Access Denied",
                    reason="output_path escapes base_directory",
                    context=self.output_path,
                )
            try:
                os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
                with open(dest, "w", encoding="utf-8") as fh:
                    fh.write(extracted.model_dump_json(indent=2))
            except OSError as e:
                return self.format_error(error="Filesystem Error", reason=str(e))
            result["model_json_path"] = os.path.relpath(
                dest, os.path.realpath(self.base_directory)
            )

        return json.dumps(result, indent=2)
