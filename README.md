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

Download the packaged release and install it:

```bash
curl -LO https://github.com/tonymenzo/heptapod/releases/download/v3.3.2/heptapod-3.3.2.tar.gz
tb install heptapod-3.3.2.tar.gz
```

Earlier versions are listed under [Releases](https://github.com/tonymenzo/heptapod/releases), and you can also [install directly from the cloned repo](#install-from-the-repository). If a browser download arrives unpacked as `heptapod-<version>.tar`, run `gzip` on it first: `tb install` reads only `.tar.gz` files.

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
| Particle data, literature, published data tables, unit conversions | `pdg`, `inspire`, `arxiv`, `literature`, `hepdata`, `units` |
| Feynman diagrams and dimensional estimates | `nda` |
| Symbolic amplitudes and UFO models | `eda`\*, `feynrules`\* |
| Monte Carlo generation and event analysis | `mg5`\*, `event_gen`, `analysis`, `bsm` |
| Pythia or Sherpa alone, event datasets, detector simulation, comparison with measurements | `pythia`, `sherpa`\*, `events`, `delphes`\*, `rivet`\* |
| Your own Pythia 8 or Sherpa build, driven from C++ | `pythia_cpp`\*, `sherpa_cpp`\* |
| Process catalogues: model inspection, 1→N decay census, independent MG5 graph checks | `processes`, `process_validation`\* |
| Background jobs, provenance ledgers, independent model review | `jobs`, `logging`, `reverse`\* |

\* Requires additional [configuration](#configuration).

For a smaller installation, select bundles from the same tarball with repeated `--bundle` flags:

```bash
tb install heptapod-3.3.2.tar.gz --bundle pdg --bundle nda
```

You can run this command again with additional bundles to add their dependencies to an existing installation.

See [toolkit.yaml](toolkit.yaml) for the complete bundle definitions and dependencies.

## Configuration

Most bundles work after installation. The `mg5`, `eda`, `feynrules`, `delphes`, `rivet`, `sherpa`, `pythia_cpp` and `sherpa_cpp` bundles require external software; their tools remain inactive until the required paths are configured. Replace the example paths below with your local installations.

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

FeynRules 2.3.49 needs a Wolfram kernel older than 14: from 14 on, `Commutator` and `MatrixSymbol` are protected built-ins that FeynRules redefines, and the UFO export comes out malformed. With several kernels installed, point WolframScript at a 13.x one, for example `export WOLFRAMSCRIPT_KERNELPATH=/Applications/Mathematica.app/Contents/MacOS/WolframKernel`.

**Delphes** (`delphes`), the directory holding the `DelphesHepMC3` executable:

```bash
tb config set heptapod delphes_path /path/to/Delphes
```

**Rivet** (`rivet`), the prefix holding `bin/rivet-config`:

```bash
tb config set heptapod rivet_path /path/to/rivet
```

**Your own Pythia 8 or Sherpa build** (`sherpa`, `pythia_cpp`, `sherpa_cpp`), the install prefixes holding `bin/pythia8-config` and `bin/Sherpa-config`; a C++ compiler must be on `PATH`. Note that Pythia is installed into the toolkit environment by default; Sherpa is not, although PyPI has a wheel for a few platforms (see the [Sherpa tools](tools/README.md#sherpafromruncardtool--sherpacppfromruncardtool)):

```bash
tb config set heptapod pythia_path /path/to/pythia8317
tb config set heptapod sherpa_path /path/to/sherpa-3.0.2
```

**Independent model review** (`reverse`) spawns a CLI agent of your choice rather than an external binary; configure it with `blank_agent_cmd` as described in [`tools/reverse/README.md`](tools/reverse/README.md#configuring-the-agent).

Inspect the effective configuration:

```bash
tb config show heptapod
```

Validate required configuration fields:

```bash
tb config validate heptapod
```

See the [external dependency guide](docs/usage.md#external-dependencies) for installation details, including the platforms the Pythia and Sherpa wheels cover.

## Installation options and versions

Installed toolkits and their environments are stored in `~/.toolbase/cache/<name>/<version>/` (for HEPTAPOD, `~/.toolbase/cache/heptapod/<version>/`). Editable installs reference your source checkout.

To install a specific release and select it for your current project, replace `<version>` below with the release version:

```bash
curl -LO https://github.com/tonymenzo/heptapod/releases/download/v<version>/heptapod-<version>.tar.gz
tb install heptapod-<version>.tar.gz
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

- **Examples:** [API setup and demo guide](examples/README.md), [symbolic calculations](examples/eda/), [diagram enumeration and NDA](examples/nda/), [leptoquark simulation](examples/sim/s1_lq_rr/), [Lagrangian extraction from a paper](examples/lagrangian_extraction/).
- **Tools:** [tool reference](tools/README.md) and [advanced setup](docs/usage.md).
- **Process catalogues:** [workflow and portable installation guide](docs/PROCESS_CATALOGUE.md) and [validation record](docs/PROCESS_CATALOGUE_TESTING.md).
- **Skills:** [agent guides](skills/) for FeynRules model building, MadGraph workflows, Pythia run cards and main programs, Sherpa run cards, and Rivet comparisons with measurements, including common pitfalls and worked templates. The [skills guide](docs/usage.md#skills) explains how each harness surfaces them.

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
