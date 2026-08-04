# Reverse Lagrangian check

The forward chain — paper → extraction → `.fr` → UFO → MadGraph — can be
self-consistent and still wrong. Every stage after the extraction inherits
the extraction's reading of the paper, so no later stage can catch a
misreading. A model that compiles, passes the FeynRules checks and imports
into MadGraph has proved only that **the tool chain accepts it**.

This bundle breaks that circularity by running the chain backwards with an
agent that never saw the paper.

## The loop

```
model.fr
   │
   ├─ 1. sanitize ──────────► sanitized.fr
   │      strip comments, M$Information, the model name and prose labels.
   │      Keep the physics. Symbol names (S1, yRR11) stay on purpose.
   │
   ├─ 2. reconstruct ───────► reconstruction.md
   │      A blank-slate agent reads ONLY sanitized.fr — no paper, no model
   │      name, no history, no MCP tools, empty directory — and writes the
   │      Lagrangian back out in LaTeX.
   │
   ├─ 3. crosscheck ────────► crosscheck.md
   │      A second fresh agent compares that reconstruction against the
   │      paper, term by term, and grades every difference:
   │      convention / substantive / cosmetic.
   │
   └─ 4. package ───────────► REVIEW.md, REVIEW.pdf
          Verbatim .fr terms, the reconstruction, the graded comparison,
          a reviewer checklist, and a sign-off block.
```

If the `.fr` really encodes the paper's model, an independent reader should
recover the paper's Lagrangian from it. Where the reconstruction and the
paper disagree, either the model is wrong or the convention differs — and
the report says which it believes, with its reasoning.

**The tool never declares the physics correct.** The last page of
`REVIEW.pdf` is a sign-off block, because that verdict belongs to a
physicist.

## Layout

| file | what it does |
|---|---|
| `reverse_tool.py` | `ReverseLagrangianTool` — the orchestral entry point |
| `sanitize.py` | strips identifying content, keeps the physics |
| `blank_agent.py` | runs one isolated CLI-agent session |
| `prompts.py` | the reconstruct and cross-check prompts |
| `review_package.py` | assembles `REVIEW.md` |
| `pdf_build.py` | compiles `REVIEW.md` to `REVIEW.pdf` |

## Actions

| action | needs the paper | produces |
|---|---|---|
| `reconstruct` | no | `reconstruction.md` |
| `crosscheck` | yes | `crosscheck.md` |
| `full` | yes | both, plus `REVIEW.md` / `REVIEW.pdf` |

## Configuring the agent

The bundle spawns whatever CLI agent `blank_agent_cmd` names. It is a command
template with two tokens:

- `{output}` — the file the CLI writes its final message to.
- `{prompt}` — the prompt text. If the template has no `{prompt}`, the prompt
  is appended as the last argument instead.

```bash
tb config set heptapod blank_agent_cmd \
  "codex exec --sandbox read-only --skip-git-repo-check --output-last-message {output}"
```

Any agent CLI works, so the check is not tied to one vendor. Use a **different
model** from the one that generated the `.fr` where you can: an independent
reader is the point, and the same model repeating its own reasoning is a
weaker test.

Two constraints on whatever you configure:

1. **Run it read-only.** The agent must not write files. Its answer leaves
   through `{output}` or stdout, which the CLI process writes from outside
   the sandbox.
2. **Give it no network and no tools.** The value of the check is that the
   agent could not have looked the model up.

## What it cannot tell you

- The agent is isolated, but the underlying model may have seen public
  model files in training. Sanitizing removes the labels, not that risk.
- Symbol names survive sanitizing on purpose, and they hint at the model.
- A clean report means one independent reader agreed. It is evidence, not
  proof, and it is why the sign-off block exists.
