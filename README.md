[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0) [![Python](https://img.shields.io/badge/Python-3.12%20|%203.13-blue.svg)](https://www.python.org/downloads/) [![Served via](https://img.shields.io/badge/Served%20via-toolbase-orange.svg)](https://github.com/alexr314/toolbase)

# **HEPTAPOD**

<p align="center">
  <img src="logo/heptapod.svg" alt="HEPTAPOD" width="150">
</p>

**HEPTAPOD** (High-Energy Physics Toolkit for Agentic Programming/Planning, Orchestration, and Deployment) is an open toolkit for **integrating LLM agents into high-energy physics workflows**, spanning symbolic amplitude calculations, Monte Carlo event generation, data analysis, and more.

The repository includes both **tools** that execute physics tasks and **skills** that guide agents in using them. It runs through [toolbase](https://github.com/alexr314/toolbase), which installs the tools in an isolated environment and serves them over the [Model Context Protocol (MCP)](https://modelcontextprotocol.io).

## Quickstart

You'll need **Python 3.12 or 3.13** and an installed MCP-compatible coding agent, such as Claude Code, Codex, or OpenCode.

Install toolbase:

```bash
pip install toolbase
```

Download the packaged `heptapod-<version>.tar.gz` asset from [Releases](https://github.com/tonymenzo/heptapod/releases) (or [install directly from the cloned repo](#install-from-the-repository)), then install it, replacing the path below with your downloaded file:

```bash
tb install "/path/to/heptapod-<version>.tar.gz"
```

This installs all bundles and their Python dependencies. Bundles that need external software become available once their paths are set under [Configuration](#configuration).

Create a working directory:

```bash
mkdir my_session && cd my_session
```

Activate all of HEPTAPOD with

```bash
tb activate heptapod
```

or select individual bundles (curated sets of tools and skills) using `tb activate heptapod/<bundle>`. For example, to enable particle data lookups:

```bash
tb activate heptapod/pdg
```

To see all available bundles and tools:

```bash
tb list -v
```

Connect and launch your preferred agent using **one** of these options.

**Claude Code:**

```bash
tb connect claude-code
claude
```

**Codex:**

```bash
tb connect codex
codex
```

**OpenCode:**

```bash
tb connect opencode
opencode
```

When prompted, trust the `toolbase` MCP server. Then run `/mcp` in Claude Code or Codex, or `/mcps` in OpenCode, and check that `toolbase` appears in the server list.

You can also use the tools through an MCP-compatible agent extension in **Visual Studio Code**. From your terminal, open the directory where you activated HEPTAPOD and issue

```bash
code .
```

After trusting the directory, the tools will be agent-accessible from the editor's agent interface.

With the `pdg` bundle activated, try prompting:

> What is the measured width of the Z boson?

or

> What is the branching ratio of $K^+ → 3\pi^0 e^+ ν_e$?

## Bundles

Tools are grouped into bundles so you can install only what a workflow needs.

| Workflow | Bundles |
|----------|---------|
| Particle data, literature, unit conversions | `pdg`, `inspire`, `units` |
| Feynman diagrams and dimensional estimates | `nda` |
| Symbolic amplitudes and UFO models | `eda`\*, `feynrules`\* |
| Monte Carlo generation and event analysis | `mg5`\*, `event_gen`, `analysis`, `bsm` |

\* Requires additional [configuration](#configuration).

For a smaller installation, select bundles from the same tarball with repeated `--bundle` flags:

```bash
tb install "/path/to/heptapod-<version>.tar.gz" --bundle pdg --bundle nda
```

You can run this command again with additional bundles to add their dependencies to an existing installation.

See [toolkit.yaml](toolkit.yaml) for the complete bundle definitions and dependencies.

## Configuration

Most bundles work after installation. The `mg5`, `eda`, and `feynrules` bundles require external software; their tools remain inactive until the required paths are configured. Replace the example paths below with your local installations.

**MadGraph** (`mg5`):

```bash
tb config set heptapod mg5_path /path/to/MG5_aMC
```

**Mathematica / WolframScript** (`eda`, `feynrules`); symbolic amplitude calculations also require FeynCalc:

```bash
tb config set heptapod wolframscript_path /path/to/wolframscript
```

**FeynRules** (`feynrules`), in addition to WolframScript:

```bash
tb config set heptapod feynrules_path /path/to/FeynRules
```

Inspect the effective configuration:

```bash
tb config show heptapod
```

Validate required configuration fields:

```bash
tb config validate heptapod
```

See the [external dependency guide](docs/usage.md#external-dependencies) for installation details. Pythia and Sherpa are installed automatically with the `event_gen` bundle.

## Installation options and versions

Installed toolkits and their environments are stored in `~/.toolbase/cache/<name>/<version>/` (for HEPTAPOD, `~/.toolbase/cache/heptapod/<version>/`). Editable installs reference your source checkout.

To install a specific release and select it for your current project, replace `<version>` below with the release version:

```bash
tb install "/path/to/heptapod-<version>.tar.gz"
tb use "heptapod@<version>"
```

Restart your agent session after switching so the new version takes effect.

### Install from the repository

To install from a source checkout:

```bash
git clone https://github.com/tonymenzo/heptapod.git
cd heptapod
tb install .
```

For an editable development installation, use `tb install -e .` instead. From your agent's working directory, select that checkout with:

```bash
tb use heptapod@editable
```

See the [source installation guide](docs/usage.md#installing-from-source) for more details.

## Examples and reference

- **Examples:** [API setup and demo guide](examples/README.md), [symbolic calculations](examples/eda/), [diagram enumeration and NDA](examples/nda/), [leptoquark simulation](examples/sim/s1_lq_rr/).
- **Tools:** [tool reference](tools/README.md) and [advanced setup](docs/usage.md).
- **Skills:** [agent guides](skills/) for FeynRules model building and MadGraph workflows, including common pitfalls and worked templates.

## Contributing

To add or improve tools, start with the [tool-writing tutorial](examples/primer/) and [contributing guide](CONTRIBUTING.md). See [source installation](docs/usage.md#installing-from-source) and [testing](docs/usage.md#testing) for the development workflow.

## Citation

If you use HEPTAPOD in your research, please cite:

- **HEPTAPOD:** [https://arxiv.org/abs/2512.15867](https://arxiv.org/abs/2512.15867).
- For the NDA, FeynGraph, or EDA bundles, also cite **Agentic Diagrammatica:** [https://arxiv.org/abs/2603.26990](https://arxiv.org/abs/2603.26990).
- If you build on the Orchestral framework, also cite **Orchestral AI:** [https://arxiv.org/abs/2601.02577](https://arxiv.org/abs/2601.02577).

BibTeX entries are available in [CITATION.bib](CITATION.bib).

## License

HEPTAPOD is licensed under [GPL-3.0](LICENSE.txt).

## Contact

**Maintainer:** Tony Menzo — [menzo.ynot@gmail.com](mailto:menzo.ynot@gmail.com).

**Issues and support:** [https://github.com/tonymenzo/heptapod/issues](https://github.com/tonymenzo/heptapod/issues).
