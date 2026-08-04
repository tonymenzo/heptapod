# **HEPTAPOD**

---
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python](https://img.shields.io/badge/Python-3.12%20|%203.13-blue.svg)](https://www.python.org/downloads/)
[![Framework](https://img.shields.io/badge/Framework-Orchestral--AI-green.svg)](https://orchestral-ai.com)
[![Served via](https://img.shields.io/badge/Served%20via-toolbase-orange.svg)](https://github.com/alexr314/toolbase)

## Overview

**HEPTAPOD** (High-Energy Physics Toolkit for Agentic Programming/Planning, Orchestration, and Deployment) is an open toolkit for **integrating LLM agents into high-energy physics workflows**, spanning symbolic amplitude calculations, Monte Carlo event generation, and data analysis.

HEPTAPOD provides structured tool interfaces that LLM agents can call directly, allowing researchers to express physics intent in natural language while the agent handles tool selection, execution, and error recovery. The tools are packaged as a **[toolbase](https://github.com/alexr314/toolbase) toolkit**: toolbase installs the toolkit into an isolated environment and **serves the tools over the [Model Context Protocol (MCP)](https://modelcontextprotocol.io)** to coding agents such as Claude Code and OpenAI Codex. The same tools also run directly under [Orchestral](https://orchestral-ai.com), a provider-agnostic agent framework (Claude, GPT, Gemini, Groq, Ollama) well-suited for tool development and API-based agent interaction.

For a detailed discussion of the framework design and its application to Monte Carlo event generation, see [arXiv:2512.15867](https://arxiv.org/abs/2512.15867). The extension to agentic symbolic computation (Diagrammatica) and a general treatment of tool-constrained agentic programming are presented in [arXiv:2603.26990](https://arxiv.org/abs/2603.26990).

---

## Installation

HEPTAPOD is distributed as a toolkit via **[toolbase](https://github.com/alexr314/toolbase)**, a registry and CLI for building, sharing, installing, and serving agent toolkits over MCP. For HEPTAPOD, toolbase provisions an isolated environment for the toolkit, resolves each bundle's dependencies and configuration, and serves the tools to your agent. Install it once in your preferred Python environment

```bash
pip install toolbase        # provides the `tb` command
```

### Quick start

```bash
git clone https://github.com/tonymenzo/heptapod.git && cd heptapod
tb install .
```

2. **`git clone … && cd heptapod`**: grab the toolkit
3. **`tb install .`**: build the environment and install the tools straight from the clone

Useful variations:

```bash
tb install . --bundle analysis --bundle mg5 --bundle nda   # only these bundles
tb install -e .                                            # editable: live-link the source for development
```

Requires **Python 3.12 or 3.13**; toolbase auto-detects venv/conda and installs each selected bundle's dependencies. See [bundles](#bundles) for what each pulls in and [external Dependencies](#external-dependencies) for software (Mathematica, MadGraph, …) that some bundles expect on the system.

### Portable / versioned installs (no clone)

For air-gapped machines, servers, or pinning a specific version, install directly from a packaged tarball on the [Releases](https://github.com/tonymenzo/heptapod/releases) page:

```bash
tb install heptapod-<version>.tar.gz
```

These are self-contained snapshots produced with `tb export`; hand one to a collaborator or `scp` it to a server and `tb install` it as-is.

---

## Getting Started

toolbase serves the toolkit's tools to any MCP-compatible client. The typical flow is **install → activate → connect**.

### Serve to a coding agent (Claude Code / Codex / OpenCode)

Begin with the `pdg` bundle:

```bash
mkdir my_session && cd my_session

tb activate heptapod/pdg    # Particle Data Group lookups; `tb activate heptapod` for everything
tb connect claude-code      # writes this project's .mcp.json (-g for user-level)
claude                      # or codex / opencode
```

Confirm the server is connected with the `/mcp` slash command (`/mcps` in open code), `toolbase` should appear there with three tools. From toolbase's side, `tb connect --list` shows where it is currently wired and `tb serve --dry-run` prints what the active profile would serve.

Then ask something the tools can answer:

> What is the measured width of the $Z$ boson?

> What is the branching ratio of $K^+ → 3\pi^0 e^+ ν_e$?

`tb connect` writes the MCP server entry and your harness starts `tb serve` on launch. Substitute `codex` or `opencode` for `claude-code`; `tb connect --harnesses` lists the supported harnesses and `tb connect --list` reports where toolbase is currently wired. `tb list` shows what is installed and active.

To supply a system prompt, copy one into the working directory as `CLAUDE.md` (Claude Code) or `AGENTS.md` (Codex, OpenCode); each example keeps its own under `examples/<name>/prompts/`.

### Worked examples

New to writing tools? Start with [examples/primer/](examples/primer/) — a
tutorial notebook that builds two tools from scratch (a database query and a
wrapper around external analysis code) and packages them as a toolbase toolkit.
No API key needed for most of it.

Building a BSM model from a paper? See
[examples/lagrangian_extraction/](examples/lagrangian_extraction/). Its
`minimal_model.py` writes a scalar-leptoquark `.fr` file and runs with no
Mathematica, no LLM and no network.

Each example ships a launcher that runs the whole setup above (sandbox, system prompt, bundles, external-software paths, wiring) and then starts the agent in it:

```bash
python examples/nda/launch.py --harness claude-code            # or codex, opencode
python examples/eda/launch.py --harness codex
python examples/sim/s1_lq_rr/launch.py --harness claude-code   # brings the MC run cards
```

Each sandbox is its own toolbase project, so the bundles it activates don't touch your global config. See the per-example READMEs for what each covers.

### Orchestral

All tools inherit from Orchestral's `BaseTool` class and instances are natively supported by `toolbase`. Wire it up or run a demo (after configuring an API key, see [Configuration](#configuration)):

```bash
tb connect orchestral                        # writes a runnable agent script

python examples/eda/eda_demo.py              # symbolic calculations (EDA)
python examples/nda/nda_demo.py              # diagram enumeration + NDA estimation
python examples/sim/s1_lq_rr/s1_lq_rr_demo.py  # S1 leptoquark simulation pipeline
```

Each demo creates a sandboxed workspace, loads the relevant tools and system prompt, and launches a web UI at `http://127.0.0.1:8000`. Configure the LLM provider by editing the demo script (Claude, GPT, Gemini, Groq, Ollama).

### Worked benchmarks

Complete conversation transcripts with agent outputs are available for the EDA and NDA bundles in [examples/eda/](examples/eda/) and [examples/nda/](examples/nda/).

---

## Bundles

HEPTAPOD's capabilities span:

- **Exact tree-level calculations** via automatic FeynCalc code generation (`eda`)
- **Order-of-magnitude rate estimates** via Naive Dimensional Analysis (`nda`)
- **Automatic Feynman diagram enumeration** and ranking (`nda` / FeynGraph)
- **Particle data**, **literature search**, and **unit conversions** (`pdg`, `inspire`, `units`)
- **Monte Carlo event generation** with MadGraph, Pythia, and Sherpa (`mg5`, `event_gen`)
- **Event analysis**: cutflows, kinematics, reconstruction, yield normalization (`analysis`)
- **BSM spectrum setup**: benchmark-point parsing and decay-table construction (`bsm`)
- **BSM model building from papers**: arXiv source retrieval, `.fr` generation, UFO validation, and an independent blank-slate re-reading of the model (`literature`, `extract`, `frgen`, `feynrules`, `reverse`)
- **Reproducible, auditable execution traces** via run cards and structured outputs

Tools are grouped into **bundles**, so you install only what a workflow needs. Bundles with a `requires:` key are gated on config (see [Configuration](#configuration)); their tools stay hidden until it's set.

| Bundle | Provides | Needs |
|--------|----------|-------|
| `units` | Unit / natural-unit conversions | none |
| `inspire` | INSPIRE-HEP literature search | none |
| `pdg` | Particle Data Group lookups | `pdg` pip pkg |
| `analysis` | Cutflows, kinematics, reconstruction, JSONL/NumPy conversions, yield normalization, recast linting | numpy, tqdm, pylhe |
| `bsm` | SUSY benchmark-point spectrum parsing; Pythia decay-table construction | none (pure python) |
| `event_gen` | MadGraph → Pythia/Sherpa event-generation pipeline | pythia8mc, sherpa-mc, pylhe |
| `mg5` | MadGraph5 run-card generation + fast process validation | `mg5_path` |
| `nda` | Feynman-diagram enumeration + Naive Dimensional Analysis | feyngraph |
| `eda` | Exact symbolic amplitudes via Mathematica/FeynCalc | `wolframscript_path` |
| `feynrules` | BSM UFO model generation + full-chain model validation via FeynRules | `feynrules_path`, `wolframscript_path` |
| `literature` | arXiv search, LaTeX e-print retrieval, PDF full-text extraction | pymupdf |
| `extract` | Paper text → structured FeynRules model spec (LLM) | an LLM provider |
| `frgen` | Structured spec → FeynRules `.fr` source | jinja2 |
| `reverse` | Blank-slate re-reading of a `.fr` into a physicist review package | `blank_agent_cmd` |
| `jobs` | Detached background execution for slow tools, with polling | none (pure python) |
| `logging` | Structured `audit.json` provenance ledger for end-to-end runs | none (pure python) |

Additional domain bundles (e.g. `hepmc`/`delphes` detector simulation, `pythia`-only showering, `llp` long-lived-particle reach studies) ship on their respective feature branches. Inspect [`toolkit.yaml`](toolkit.yaml) for the authoritative, up-to-date list.

---

## Skills

Some bundles ship a **skill** — a written guide to using their tools well, covering the failure modes that are predictable but not obvious. They live in [`skills/`](skills/):

| Skill | Covers | Bundle |
|-------|--------|--------|
| `feynrules` | Declaring mass and width once, tagging BSM couplings with an interaction order, checking the UFO before MG5 sees it | `feynrules` |
| `mg5` | Shallow comma decay chains and the `NP=N` alternative, `compute_widths` ordering, reading past MG5's auto-conversion error mask | `mg5` |

Each skill names a bundle, and `tb connect` surfaces it only when that bundle's tools are actually being served — so the `mg5` guide stays hidden until `mg5_path` is set, since without it the tools it describes aren't there either. `tb deactivate heptapod__mg5` turns one off; `tb activate` turns it back on.

How a skill reaches the agent depends on the harness:

- **Claude Code, Antigravity** load it automatically when the conversation looks relevant, and also expose it as a `/heptapod__mg5` slash command.
- **Codex, OpenCode** have no model-facing skill concept, so it arrives as a `/heptapod__mg5` slash command you invoke yourself.

Skills are surfaced by `tb connect` alongside the MCP server, so nothing extra is needed — connect a harness and its guides come with it. Requires toolbase ≥ 0.11.

---

## Configuration

### Toolkit configuration (external tool paths)

Bundles that call external software read their paths from the toolkit's config (`~/.toolbase/config/heptapod.yaml`). Set them with `tb config`:

```bash
tb config set heptapod base_directory /path/to/workspace          # sandbox root for tool I/O
tb config set heptapod mg5_path /path/to/MG5_aMC_v3.6.6           # event generation (mg5 bundle)
tb config set heptapod wolframscript_path /path/to/wolframscript  # eda / feynrules
tb config set heptapod feynrules_path /path/to/FeynRules_v2.3.49  # feynrules
tb config set heptapod delphes_path /path/to/Delphes             # delphes bundle (if installed)

tb config show heptapod                                          # view effective config
tb config validate heptapod                                     # check required fields
```

`base_directory` defaults to the directory the agent launched `tb serve` from; pin it per-project when you want a fixed sandbox.

The `reverse` bundle spawns a CLI agent instead of an external binary, so it
takes a command template rather than a path:

```bash
tb config set heptapod blank_agent_cmd \
  "codex exec --sandbox read-only --skip-git-repo-check --output-last-message {output}"
```

`{output}` is the file the CLI writes its final message to, and `{prompt}` is
the prompt text (appended as the last argument if the token is absent). Any
agent CLI works. Run it read-only and without network: the check is only
worth anything if the agent could not look the model up. See
[`tools/reverse/README.md`](tools/reverse/README.md).

### LLM providers (Orchestral demos only)

When HEPTAPOD is served through an MCP coding agent, **the agent provides its own LLM**, so no API keys are needed. Keys are only required for the standalone Orchestral demos. Copy the templates and edit your local (gitignored) copies:

```bash
cp .env.example .env
cp config.example.py config.py
```

```bash
# .env: cloud providers (only what you use)
ANTHROPIC_API_KEY=your_key_here    # Claude:  https://console.anthropic.com/
OPENAI_API_KEY=your_key_here       # GPT:     https://platform.openai.com/api-keys
GOOGLE_API_KEY=your_key_here       # Gemini:  https://aistudio.google.com/app/apikey
GROQ_API_KEY=your_key_here         # Groq:    https://console.groq.com/
```

For local, key-free inference, install [Ollama](https://ollama.com/download) and set `ollama_host` / `ollama_model` in `config.py`.

---

## External Dependencies

Most bundles (`units`, `inspire`, `pdg`, `nda`, `analysis`, `bsm`) work out of the box; toolbase installs their pip dependencies automatically. The following bundles expect additional software on the system:

#### Mathematica and WolframScript (`eda`, `feynrules`)

1. Install [Mathematica](https://www.wolfram.com/mathematica/) (includes WolframScript)
2. Install [FeynCalc](https://feyncalc.github.io/) (for `eda`)
3. Authenticate: `wolframscript -authenticate`
4. Optionally install [FeynRules](https://feynrules.irmp.ucl.ac.be/) v2.3.49 (for UFO model generation)
5. Point toolbase at it: `tb config set heptapod wolframscript_path <path>`

#### MadGraph5_aMC@NLO (`mg5`, `event_gen`)

```bash
wget https://launchpad.net/mg5amcnlo/3.0/3.6.x/+download/MG5_aMC_v3.6.6.tar.gz
tar -xzf MG5_aMC_v3.6.6.tar.gz
tb config set heptapod mg5_path "$(pwd)/MG5_aMC_v3.6.6"
```

#### Pythia8 and Sherpa3 (`event_gen`)

**Installed automatically** as bundle dependencies when you `tb install . --bundle event_gen`. No separate installation needed.

---

## Testing

```bash
python test_runner.py                # run all tests
python test_runner.py --skip-slow    # skip MG5, Pythia, Sherpa generation
python test_runner.py --only nda     # a single component
python test_runner.py --help         # all options
```

Individual tool suites can also be run directly with `pytest` (e.g. `pytest tools/analysis/`). During development, `tb validate` checks that `toolkit.yaml` and the tool modules are well-formed and servable.

`test_runner.py` runs against your own interpreter, not the isolated environment `tb install` builds, so components whose bundle deps you haven't installed fail on import (`--only pdg` without `pdg`, for instance). Install the ones you want to exercise: `pip install pdg feyngraph pylhe`.

---

## Usage

Once toolbase is connected (or an Orchestral demo is running), interact with the agent in natural language:

**Symbolic calculations (EDA):**
> Compute the tree-level decay width for $H \to b \bar{b}$ with a Yukawa vertex.

**Diagram enumeration and NDA estimation:**
> Enumerate the tree-level diagrams for muon decay to $e^+ \nu_e \bar{\nu}_\mu$ and estimate the branching ratio for each diagram class.

**Particle data and literature:**
> What is the measured width of the $Z$ boson? Find recent papers on Higgs rare decays on INSPIRE.

**Monte Carlo event generation:**
> Generate 10,000 $pp \to tt$ events at 13 TeV using MadGraph, shower with Pythia, and plot the invariant mass distribution.

For detailed tool documentation, see [tools/README.md](tools/README.md).

---

## Contributing

HEPTAPOD is designed to be extended with custom tools. A new tool is a `BaseTool` subclass registered as one entry in `toolkit.yaml`.

**See [CONTRIBUTING.md](CONTRIBUTING.md) for comprehensive guidelines on:**

- Tool architecture and structure (RuntimeFields, StateFields, error handling)
- Registering a tool + bundle in `toolkit.yaml`
- Path safety and sandboxing requirements
- Testing, `tb validate`, and integration
- Best practices and examples

For bug reports, feature requests, or technical discussions, use [GitHub Issues](https://github.com/tonymenzo/heptapod/issues).

---

## Citation

If you use HEPTAPOD in your research, please cite:

```bibtex
@article{Menzo:2025cim,
    author = {Menzo, Tony and Roman, Alexander and Gleyzer, Sergei and Matchev, Konstantin and Fleming, George T. and H{\"o}che, Stefan and Mrenna, Stephen and Shyamsundar, Prasanth},
    title = "{HEPTAPOD: Orchestrating High Energy Physics Workflows Towards Autonomous Agency}",
    eprint = "2512.15867",
    archivePrefix = "arXiv",
    primaryClass = "hep-ph",
    reportNumber = "FERMILAB-PUB-25-0923-CSAID-ETD-T",
    month = "12",
    year = "2025"
}
```

If you use the NDA, FeynGraph, or EDA bundles, please also cite:

```bibtex
@article{Menzo:2026diagrammatica,
    author = {Menzo, Tony and Roman, Alexander and Fleming, George T. and Gleyzer, Sergei and Matchev, Konstantin T. and Mrenna, Stephen},
    title = "{Agentic Diagrammatica: Towards Autonomous Symbolic Computation in High Energy Physics}",
    eprint = "2603.26990",
    archivePrefix = "arXiv",
    primaryClass = "hep-ph",
    reportNumber = "FERMILAB-PUB-26-0208-T",
    month = "3",
    year = "2026"
}
```

If you build on the [Orchestral](https://orchestral-ai.com) framework (for example, via the demos in `examples/`), please also cite:

```bibtex
@misc{roman2026orchestralaiframeworkagent,
    author = {Roman, Alexander and Roman, Jacob},
    title = "{Orchestral AI: A Framework for Agent Orchestration}",
    eprint = "2601.02577",
    archivePrefix = "arXiv",
    primaryClass = "cs.AI",
    year = "2026"
}
```

---

## License

This project is licensed under the GPL-3.0 license - see the [LICENSE](LICENSE.txt) file for details.

---

## Contact

**Maintainers:**

- Tony Menzo - amenzo@ua.edu

**Issues and Support:**

- GitHub Issues: [https://github.com/tonymenzo/heptapod/issues](https://github.com/tonymenzo/heptapod/issues)

**Project Links:**

- Repository: [https://github.com/tonymenzo/heptapod](https://github.com/tonymenzo/heptapod)
- HEPTAPOD paper: [arXiv:2512.15867](https://arxiv.org/abs/2512.15867)
- Diagrammatica paper: [arXiv:2603.26990](https://arxiv.org/abs/2603.26990)

---

**Status**: Active Development
