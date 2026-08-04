"""
# prompts.py is a part of the HEPTAPOD package.
# Copyright (C) 2026 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

Prompts for the blank-slate reverse check. Both phases run in fresh sessions
with ONLY the files named below in the working directory — the reconstruction
phase never sees the paper; the cross-check phase never sees the .fr.
"""

RECONSTRUCT_PROMPT = """\
You are given a single FeynRules model file `sanitized.fr` in your working \
directory. All metadata (model name, authors, comments, prose labels) has \
been removed; you have NO other context about where this model comes from. \
Do not guess a published model name — work only from the file's content.

Reconstruct the physics it encodes, as a markdown document with:

1. **Lagrangian (LaTeX).** Every Lagrangian term in standard physics \
notation, one display equation per term, each labeled with the `.fr` symbol \
it came from (e.g. `LkinS1`). Expand FeynRules idioms (DC covariant \
derivatives, FS field strengths, Ga gamma matrices, HC hermitian conjugate, \
CC charge conjugation, ProjP/ProjM chiral projectors) into conventional \
notation. State the covariant derivative's gauge content explicitly.

2. **Field table.** For every new particle class: symbol, spin, SU(3) rep, \
SU(2) rep (if declared), U(1) charge or hypercharge, self-conjugate or not, \
mass symbol and value if numeric.

3. **Parameters.** For every new external parameter: symbol, what it \
multiplies in the Lagrangian, and its physical meaning (coupling, mass, \
mixing angle, ...).

4. **Physics summary.** Two or three sentences: what kind of model is this \
(what new states, what interactions), and what processes it would mediate.

Be precise about chirality, conjugation and index contractions — a physicist \
will check your reconstruction term by term. Your final message must be the \
complete markdown document and nothing else.
"""

CROSSCHECK_PROMPT = """\
Your working directory contains exactly two files:
  - `paper.tex` — the LaTeX source (or extracted text) of a physics paper;
  - `reconstruction.md` — an independent reconstruction of a model's \
Lagrangian and field content, produced from an implementation file you do \
NOT have access to.

Task: compare the reconstruction against the model the paper defines.

1. Locate the paper's Lagrangian / field definitions (quote equation numbers \
or section references).
2. Produce a term-by-term comparison table:

| paper term (eq. ref) | reconstruction term | verdict | notes |

where verdict is one of: **agree** / **disagree** / **missing-in-\
reconstruction** / **extra-in-reconstruction**. Compare physics content \
(field reps, charges, chirality, conjugation, coefficient structure), not \
notation.
3. After the table, list every disagreement with a severity (cosmetic / \
convention / substantive) and one sentence on what a human should check.
4. End with a one-paragraph overall assessment. Do NOT give a pass/fail \
verdict — that decision belongs to the human reviewer.

Your final message must be the complete markdown document and nothing else.
"""
