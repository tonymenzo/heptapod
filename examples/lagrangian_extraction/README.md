# Lagrangian extraction: paper to a checked BSM model

Read a BSM paper, write a FeynRules model from it, prove the tool chain
accepts the model, then have an independent agent read the model back and
compare it against the paper.

```
 literature          extract           frgen          feynrules         reverse
 ──────────          ───────           ─────          ─────────         ───────
 arXiv search   ->   paper text   ->   .fr file  ->   UFO + checks  ->  REVIEW.pdf
 LaTeX source        structured        FeynRules      MadGraph          physicist
                     spec              source        import            sign-off
```

Each stage is a separate bundle, so you install only what you need. The
first three are pure python. Only `feynrules` needs Mathematica.

## Run it

[`minimal_model.py`](minimal_model.py) runs the `frgen` stage. It needs no
Mathematica, no LLM and no network:

```bash
python examples/lagrangian_extraction/minimal_model.py
```

It builds a scalar leptoquark S1 — a colour triplet, weak singlet, charge
−1/3 scalar with one new Yukawa coupling — and writes
`sandbox/S1_LQ_minimal.fr`.

The model is small on purpose. It has exactly one new field, one new
parameter and one new interaction, so you can read the generated `.fr`
top to bottom and match every line back to the spec in the script.

## The other stages

**Get the paper.** The `literature` bundle prefers LaTeX source over PDF,
because PDF text extraction mangles equations:

```python
ArxivSearchTool(query="scalar leptoquark")
ArxivSourceTool(arxiv_id="1603.04993")     # LaTeX, not PDF
```

**Check the model.** With `feynrules_path` and `wolframscript_path` set,
`ValidateModelTool` compiles the `.fr` to a UFO, runs the FeynRules
Hermiticity, kinetic-term and mass-spectrum checks, and imports the result
into MadGraph:

```python
ValidateModelTool(model_path="S1_LQ_minimal.fr")
```

A non-Hermitian Lagrangian is the defect this catches most often. That is
why the example writes the Yukawa term twice — once bare, then added to its
own `HC[...]`.

**Check the physics.** Validation proves the tool chain accepts the model.
It does not prove the model matches the paper. `ReverseLagrangianTool` tests
that separately: an agent that never saw the paper reconstructs the physics
from the `.fr` alone, and a second agent grades the reconstruction against
the paper term by term. See [`tools/reverse/README.md`](../../tools/reverse/README.md).

```python
ReverseLagrangianTool(model_path="S1_LQ_minimal.fr", action="full")
```

**Run the slow stages in the background.** UFO compiles and reverse checks
take minutes, which is longer than an agent wants to block for. The `jobs`
bundle detaches them:

```python
SubmitJobTool(
    tool_name="ValidateModelTool",
    tool_args='{"model_path": "S1_LQ_minimal.fr"}',   # JSON string
)
JobStatusTool(job_id=...)
JobResultTool(job_id=...)
```

## What to install

| bundle | needs | gives you |
|---|---|---|
| `literature` | pymupdf | arXiv search, LaTeX source, PDF text |
| `extract` | an LLM provider | paper text to structured spec |
| `frgen` | jinja2 | spec to `.fr` |
| `feynrules` | FeynRules, wolframscript | `.fr` to UFO, checks, MadGraph import |
| `reverse` | an agent CLI (`blank_agent_cmd`) | blank-slate review package |
| `jobs` | nothing | background execution |
| `logging` | nothing | `audit.json` provenance ledger |

```bash
tb install heptapod --bundle frgen --bundle literature
```

## A caution

Passing validation means the tool chain accepts the model. It does not mean
the physics is right, and the tooling never claims otherwise — the reverse
check ends in a sign-off block for a physicist, not a verdict.
