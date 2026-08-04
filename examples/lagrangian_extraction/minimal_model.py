#!/usr/bin/env python3
"""
# minimal_model.py is a part of the HEPTAPOD package.
# Copyright (C) 2026 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

Minimal end-to-end example for the BSM model-building bundles.

Builds a scalar leptoquark S1 as a structured spec, hands it to
`GenerateFeynRulesModelTool`, and writes a FeynRules `.fr` file. Pure python:
no Mathematica, no LLM, no network. It runs anywhere the repo installs.

The full pipeline has four stages. This example runs stage 2, because it is
the only one with no external dependency:

    1. literature   arXiv search / LaTeX source     -> paper text
    2. frgen        structured spec                 -> model.fr      <- here
    3. feynrules    model.fr                        -> UFO + checks
    4. reverse      model.fr                        -> REVIEW.pdf

Stages 3 and 4 need FeynRules + wolframscript and an agent CLI respectively;
the README shows the commands.

Usage:
    python examples/lagrangian_extraction/minimal_model.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from tools.frgen.frgen_tool import GenerateFeynRulesModelTool  # noqa: E402
from tools.frgen.frmodel import (  # noqa: E402
    FeynRulesModel,
    IndexDecl,
    LagrangianTerm,
    MassSpec,
    ModelInfo,
    Parameter,
    ParticleClass,
)

SANDBOX = Path(__file__).parent / "sandbox"


def s1_leptoquark() -> FeynRulesModel:
    """A scalar leptoquark S1 coupling to right-handed up quarks and leptons.

    S1 is a colour triplet, weak singlet, charge -1/3 scalar. The one new
    interaction is a Yukawa coupling yRR11 * (u_R)^c l_R S1*, plus its
    Hermitian conjugate. That single term is what makes it a leptoquark: it
    carries both colour and lepton number, so it mediates S1 -> e u.
    """
    return FeynRulesModel(
        model_name="S1_LQ_minimal",
        info=ModelInfo(
            authors=["HEPTAPOD example"],
            version="1.0.0",
            date="04.08.2026",
            institutions=["HEPTAPOD"],
            emails=[],
        ),
        # Every new coupling counts as one power of NP; truncate at NP^2.
        interaction_order_hierarchy=[("NP", 2)],
        index_decls=[IndexDecl(name="Colour", range_kind="NoUnfold", size=3)],
        parameters=[
            Parameter(
                name="yRR11",
                parameter_type="External",
                block_name="BSMINPUTS",
                complex=False,
                interaction_order=("NP", 1),
                value="0.5",
                description="S1-e-u Yukawa coupling",
            )
        ],
        particles=[
            ParticleClass(
                spin_type="S",
                class_index=100,
                class_name="S1",
                self_conjugate=False,
                indices=["Colour"],
                mass=MassSpec(sym="MS1", value="1500."),
                # Automatic asks FeynRules to compute the width. That routine
                # is off by default (see tools/feynrules/UFO_generator.wl), so
                # the renderer emits a numeric placeholder and MadGraph
                # computes the real width from the UFO.
                width=MassSpec(sym="WS1", value="Automatic"),
                quantum_numbers={"Q": "-1/3"},
                particle_name="S1",
                antiparticle_name="S1~",
                full_name="Scalar leptoquark S1",
                propagator_label="S1",
                propagator_type="ScalarDash",
                propagator_arrow="None",
            )
        ],
        lagrangian_terms=[
            # Kinetic term with the covariant derivative, plus the mass term.
            LagrangianTerm(
                name="LkinS1",
                expression=(
                    "Block[{mu,aa}, ExpandIndices["
                    "DC[S1bar[aa],mu] DC[S1[aa],mu] - MS1^2 * HC[S1].S1]]"
                ),
                delayed=False,
            ),
            # The leptoquark Yukawa, written without its conjugate ...
            LagrangianTerm(
                name="L1YukRRNonHC",
                expression=(
                    "Block[{sp, aa}, yRR11 * anti[CC[uR]][sp, 1, aa]"
                    ".lR[sp, 1] * HC[S1][aa]]"
                ),
                delayed=True,
            ),
            # ... then made Hermitian. A non-Hermitian Lagrangian is the most
            # common defect the validation stage catches.
            LagrangianTerm(
                name="L1YukRR",
                expression="L1YukRRNonHC + HC[L1YukRRNonHC]",
                delayed=True,
            ),
            # The total BSM Lagrangian. UFO_generator.wl looks for this symbol
            # by name; pass LagName if you call it something else.
            LagrangianTerm(
                name="LBSM",
                expression="LkinS1 + L1YukRR",
                delayed=False,
            ),
        ],
    )


def main() -> int:
    SANDBOX.mkdir(exist_ok=True)
    model = s1_leptoquark()

    tool = GenerateFeynRulesModelTool(
        model_json=model.model_dump_json(),
        output_path="S1_LQ_minimal.fr",
        base_directory=str(SANDBOX),
    )
    result = json.loads(tool._run())

    if result.get("status") != "ok":
        print("[✗] generation failed:")
        print(json.dumps(result, indent=2))
        return 1

    fr_path = SANDBOX / result["fr_path"]
    print(f"[✓] wrote {fr_path.relative_to(REPO)}")
    print(f"    model      {result['model_name']}")
    print(f"    particles  {result['n_particles']}")
    print(f"    parameters {result['n_parameters']}")
    print()
    print("--- first 25 lines ---")
    for line in fr_path.read_text(encoding="utf-8").splitlines()[:25]:
        print(f"  {line}")
    print("---")
    print()
    print("Next, with FeynRules and wolframscript configured:")
    print(f"  ValidateModelTool(model_path='{result['fr_path']}')")
    print("      compile to a UFO, run the Hermiticity / kinetic-term /")
    print("      mass-spectrum checks, then import into MadGraph.")
    print()
    print("Then, with blank_agent_cmd configured:")
    print(f"  ReverseLagrangianTool(model_path='{result['fr_path']}')")
    print("      an independent agent reads the .fr back and reconstructs")
    print("      the physics. See tools/reverse/README.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
