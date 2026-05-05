# **HEPTAPOD**

---
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python](https://img.shields.io/badge/Python-3.12%20|%203.13-blue.svg)](https://www.python.org/downloads/)
[![Framework](https://img.shields.io/badge/Framework-Orchestral--AI-green.svg)](https://orchestral-ai.com)

## Overview

**HEPTAPOD** (High-Energy Physics Toolkit for Agentic Programming/Planning, Orchestration, and Deployment) is an open toolkit for **integrating LLM agents into high-energy physics workflows** — from symbolic amplitude calculations to Monte Carlo event generation and data analysis.

HEPTAPOD provides structured tool interfaces that LLM agents can call directly, allowing researchers to express physics intent in natural language while the agent handles tool selection, execution, and error recovery. All tools are accessible via the [Model Context Protocol (MCP)](https://modelcontextprotocol.io), making them available to coding agents such as Claude Code and OpenAI Codex, as well as through the [Orchestral AI](https://orchestral-ai.com) framework.

Current capabilities include:

- **Exact tree-level calculations** via automatic FeynCalc code generation (EDA toolkit)
- **Order-of-magnitude rate estimates** via Naive Dimensional Analysis (NDA toolkit)
- **Automatic Feynman diagram enumeration** and ranking (FeynGraph toolkit)
- **Particle data**, **literature search**, and **unit conversions** (PDG, INSPIRE, units)
- **Monte Carlo event generation** pipelines with MadGraph, Pythia, and Sherpa
- **Reproducible, auditable execution traces** via run cards and structured outputs

For a detailed discussion of the framework design and its application to Monte Carlo event generation, see [arXiv:2512.15867](https://arxiv.org/abs/2512.15867). The extension to agentic symbolic computation (Diagrammatica) and a general treatment of tool-constrained agentic programming are presented in [arXiv:2603.26990](https://arxiv.org/abs/2603.26990).

---

## Installation

```bash
git clone https://github.com/tonymenzo/heptapod.git
cd heptapod
pip install -r requirements.txt        # or: conda env create -f environment.yml && conda activate heptapod
python test_runner.py --only prereqs   # verify installation
```

Requires **Python 3.12 or 3.13**. See [External Dependencies](#external-dependencies) below for optional software (Mathematica, MadGraph, etc.) needed by specific toolkits.

---

## Getting Started

HEPTAPOD tools can be used through any **MCP-compatible client** or the **Orchestral AI** framework. Orchestral is provider-agnostic and well-suited for building, testing, and benchmarking new tools. MCP makes those tools available to any compatible client for interactive use — the examples below focus on coding agents (Claude Code, OpenAI Codex) as the most common case.

### MCP Coding Agents

Register the HEPTAPOD MCP server and the agent gains direct access to physics tools.

**Claude Code:**

```bash
# Run from the heptapod repo root
claude mcp add --scope user heptapod -- \
  /path/to/python "$(pwd)/mcp/heptapod_server_stdio.py"
```

**OpenAI Codex:**

```bash
codex mcp add heptapod -- \
  /path/to/python mcp/heptapod_server_stdio.py
```

To verify the server is connected, type `/mcp` in either agent's interactive session. To load a specific toolkit (e.g., `eda_toolkit`, `nda_toolkit`), append `--groups <toolkit>` to the server command.

To load a system prompt, copy it to your working directory as `CLAUDE.md` (Claude Code) or `AGENTS.md` (Codex):

```bash
cp /path/to/system_prompt.md CLAUDE.md
```

Example system prompts for the EDA and NDA toolkits are provided in `prompts/examples/`.

For available toolkits, scope options, creating custom toolkits, and HTTP transport setup, see [mcp/README.md](mcp/README.md).

### Orchestral AI

HEPTAPOD provides demo scripts that launch an Orchestral agent with a web UI:

```bash
python examples/eda/eda_demo.py              # Symbolic calculations (EDA)
python examples/nda/nda_demo.py              # Diagram enumeration + NDA estimation
python examples/workflows/hep_bsm_demo.py   # Monte Carlo event generation pipeline
```

Each demo creates a sandboxed workspace, loads the relevant tools and system prompt, and launches a web server at `http://127.0.0.1:8000`. Configure the LLM provider by editing the demo script (supports Claude, GPT, Gemini, Groq, and Ollama).

### Worked benchmarks

Complete conversation transcripts with agent outputs are available for the EDA and NDA toolkits in [examples/eda/](examples/eda/) and [examples/nda/](examples/nda/).

---

## Configuration

### First-time setup

The repo ships **templates**: `config.example.py` and `.env.example`. Copy each
to its tracked-name-minus-`.example` once, then edit your local copy. The
copies are gitignored so your paths and API keys never enter commit history.

```bash
cp config.example.py config.py
cp .env.example .env
```

### LLM Providers

**Cloud LLMs** (requires API key): Edit `.env` in the repository root (created
above from `.env.example`):

```bash
ANTHROPIC_API_KEY=your_key_here    # Claude — https://console.anthropic.com/
OPENAI_API_KEY=your_key_here       # GPT — https://platform.openai.com/api-keys
GOOGLE_API_KEY=your_key_here       # Gemini — https://aistudio.google.com/app/apikey
GROQ_API_KEY=your_key_here         # Groq — https://console.groq.com/
```

You only need keys for the providers you plan to use. When using HEPTAPOD through an MCP coding agent, the agent provides its own LLM — API keys are only needed for the Orchestral demos.

**Local Ollama** (free, no API key): Install from [ollama.com](https://ollama.com/download), then configure in `config.py` (created above from `config.example.py`):

```python
ollama_host = None              # localhost:11434 (default)
ollama_model = "gpt-oss:20b"    # your preferred model
```

### External Tool Paths

If using toolkits that require external software, edit `config.py` (your local copy):

```python
wolframscript_path = "/path/to/wolframscript"      # EDA toolkit, FeynRules
feynrules_path = "/path/to/FeynRules_v2.3.49"      # FeynRules only
mg5_path = "/path/to/MG5_aMC_v3.6.6"               # Event generation
```

---

## External Dependencies

Most HEPTAPOD tools (NDA, FeynGraph, PDG, INSPIRE, units) work out of the box with no external software. The following are only needed for specific toolkits:

#### Mathematica and WolframScript

**Required for:** EDA toolkit, FeynRules model generation.

1. Install [Mathematica](https://www.wolfram.com/mathematica/) (includes WolframScript)
2. Install [FeynCalc](https://feyncalc.github.io/) (for EDA)
3. Authenticate: `wolframscript -authenticate`
4. Optionally install [FeynRules](https://feynrules.irmp.ucl.ac.be/) v2.3.49 (for UFO model generation)

#### MadGraph5_aMC@NLO

**Required for:** Parton-level event generation, NDA cross-checks.

Download from [MadGraph Launchpad](https://launchpad.net/mg5amcnlo):
```bash
wget https://launchpad.net/mg5amcnlo/3.0/3.6.x/+download/MG5_aMC_v3.6.6.tar.gz
tar -xzf MG5_aMC_v3.6.6.tar.gz
```

#### Pythia8 and Sherpa3

**Installed automatically** via `pip install -r requirements.txt`. No separate installation needed.

---

## Testing

```bash
python test_runner.py                # run all tests
python test_runner.py --skip-slow    # skip MG5, Pythia, Sherpa generation
python test_runner.py --only nda     # NDA tools only
python test_runner.py --only eda     # EDA tools (some require wolframscript)
python test_runner.py --help         # all options
```

Available components: `prereqs`, `nda`, `eda`, `feyngraph`, `logging`, `pdg`, `inspire`, `units`, `llm`, `conversions`, `kinematics`, `reconstruction`, `delta_r_filter`, `feynrules`, `mg5`, `pythia`, `sherpa`.

---

## Usage

Once the MCP server is registered or an Orchestral demo is running, interact with the agent in natural language:

**Symbolic calculations (EDA):**
> Compute the tree-level decay width for H -> b bbar with a Yukawa vertex.

**Diagram enumeration and NDA estimation:**
> Enumerate the tree-level diagrams for muon decay to e+ nu_e nu_mubar and estimate the branching ratio for each diagram class.

**Particle data and literature:**
> What is the measured width of the Z boson? Find recent papers on Higgs rare decays on INSPIRE.

**Monte Carlo event generation:**
> Generate 10,000 pp -> tt events at 13 TeV using MadGraph, shower with Pythia, and plot the invariant mass distribution.

For detailed tool documentation, see [tools/README.md](tools/README.md).

---

## Contributing tools

HEPTAPOD is designed to be extended with custom tools. If you'd like to contribute a new tool for model generation, event simulation, analysis, or any other physics workflow:

**See [CONTRIBUTING.md](CONTRIBUTING.md) for comprehensive guidelines on:**

- Tool architecture and structure
- Required components (RuntimeFields, StateFields, error handling)
- Path safety and sandboxing requirements
- Testing and integration
- Best practices and examples

### Other Contributions

For bug reports, feature requests, or technical discussions:
- **GitHub Issues**: [https://github.com/tonymenzo/heptapod/issues](https://github.com/tonymenzo/heptapod/issues)

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

If you use the NDA, FeynGraph, or EDA toolkits, please also cite:

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

This project is licensed under the GPL-3.0 license - see the [LICENSE](LICENSE) file for details.

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

**Version**: 1.0.0

**Status**: Active Development
