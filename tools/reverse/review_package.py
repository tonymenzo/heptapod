"""
# review_package.py is a part of the HEPTAPOD package.
# Copyright (C) 2026 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

Assemble the physicist review package (REVIEW.md) for a reverse check:
verbatim Lagrangian terms from the .fr beside the blank-slate reconstruction,
the paper cross-check table when available, a numbered reviewer checklist and
a sign-off block. Modeled on examples/verification_package/README.md.
"""

from __future__ import annotations

from typing import Dict, List, Optional


def build_review_md(
    model_path: str,
    terms: List[Dict[str, str]],
    sanitizer_report: Dict[str, object],
    reconstruction_md: Optional[str],
    crosscheck_md: Optional[str],
    paper_ref: Optional[str] = None,
) -> str:
    name = sanitizer_report.get("original_model_name") or "unknown"
    lines: List[str] = [
        f"# Reverse-check review package — `{name}`",
        "",
        "An independent, **blank-slate** agent instance reconstructed the "
        "physics below from the sanitized `.fr` alone (no paper, no metadata, "
        "no conversation history). "
        + ("A second fresh instance then compared the reconstruction against "
           "the source paper. " if crosscheck_md else "")
        + "The final verdict belongs to the human reviewer — sign off at the "
        "bottom.",
        "",
        "| item | value |",
        "|---|---|",
        f"| model file | `{model_path}` |",
        f"| original model name | `{name}` (hidden from the agent) |",
        f"| paper | {paper_ref or '(not provided — cross-check skipped)'} |",
        f"| blindness scope | {sanitizer_report.get('blindness_scope', '')} |",
        "",
        "## Verbatim Lagrangian terms (from the `.fr`)",
        "",
        "These are the terms the reconstruction must account for, quoted "
        "unmodified. Check each against its reconstructed LaTeX form below.",
        "",
    ]
    if terms:
        for t in terms:
            lines += [
                f"### `{t['name']}` (`{t['op']}`)",
                "",
                "```mathematica",
                t["expression"],
                "```",
                "",
            ]
    else:
        lines += ["*(no top-level Lagrangian assignments found in the file)*", ""]

    lines += ["## Blank-slate reconstruction", ""]
    lines += [reconstruction_md.strip(), ""] if reconstruction_md else [
        "*(reconstruction phase failed — see agent_runs in the tool result)*",
        "",
    ]

    lines += ["## Paper cross-check", ""]
    if crosscheck_md:
        lines += [crosscheck_md.strip(), ""]
    else:
        lines += [
            "*(not run — provide `paper_tex_path` to enable the term-by-term "
            "comparison)*",
            "",
        ]

    lines += [
        "## Suggested checks for the reviewer",
        "",
        "1. Every verbatim `.fr` term above has a reconstructed LaTeX "
        "counterpart with the same field content, chirality and conjugation.",
        "2. Kinetic terms: covariant derivative gauge content matches the "
        "field's representations; normalization is canonical.",
        "3. Non-self-conjugate interaction terms appear together with their "
        "Hermitian conjugates.",
        "4. Quantum numbers in the field table match the `.fr` declarations "
        "(and the paper, where the cross-check table flags disagreements).",
        "5. Numeric masses/couplings are placeholders unless the paper pins "
        "them — treat values as demo inputs, not measurements.",
        f"6. Sanitizer scope: {sanitizer_report.get('prose_values_scrubbed', 0)} "
        "prose labels were scrubbed; field/parameter symbol names were kept "
        "and may hint at the model's identity.",
        "",
        "## Physicist sign-off",
        "",
        "- Reviewed by: ______________________  Date: ____________",
        "- Verdict: [ ] approve   [ ] approve with corrections   [ ] reject",
        "- Notes:",
        "",
        "",
    ]
    return "\n".join(lines)
