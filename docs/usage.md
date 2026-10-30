# HEPTAPOD development and advanced setup

For installation, bundle selection, and connecting a coding agent, use the [main README](../README.md#quickstart). For API-based workflows and provider configuration, see the [examples README](../examples/README.md).

## Installing from source

With the [quickstart prerequisites](../README.md#quickstart) installed, clone the repository:

```bash
git clone https://github.com/tonymenzo/heptapod.git
cd heptapod
```

Install from the checkout:

```bash
tb install .
```

For development, use an editable installation instead so changes to the source are available without reinstalling:

```bash
tb install -e .
```

Both commands accept the [bundle-selection flags](../README.md#bundles). Then follow the README's activation and connection steps. Commands below assume you are working from the repository root.

## Advanced configuration

Tool I/O defaults to the directory from which the agent launches `tb serve`. To pin a fixed workspace for a project, run this from that project:

```bash
tb config set heptapod base_directory /path/to/workspace --project
```

For external-software paths and configuration checks, see [Configuration](../README.md#configuration).

## External Dependencies

Most bundles (`units`, `inspire`, `hepdata`, `pdg`, `arxiv`, `literature`, `nda`, `analysis`, `bsm`, `jobs`, `logging`) work out of the box; toolbase installs their pip dependencies automatically. The following bundles expect additional software on the system:

#### Mathematica and WolframScript (`eda`, `feynrules`)

1. Install [Mathematica](https://www.wolfram.com/mathematica/) (includes WolframScript)
2. Install [FeynCalc](https://feyncalc.github.io/) (for `eda`)
3. Authenticate: `wolframscript -authenticate`
4. Optionally install [FeynRules](https://feynrules.irmp.ucl.ac.be/) (for UFO model generation)
5. Register the installed software using the [configuration commands](../README.md#configuration).

#### MadGraph5_aMC@NLO (`mg5`, `event_gen`)

Download the current release from [Launchpad](https://launchpad.net/mg5amcnlo/+download), unpack it, and register the unpacked directory:

```bash
tar -xzf MG5_aMC_v<version>.tar.gz
tb config set heptapod mg5_path "$(pwd)/MG5_aMC_v<version>"
```

#### Pythia8 and Sherpa3 (`pythia`, `sherpa`, `pythia_cpp`, `sherpa_cpp`)

Each generator has two kinds of tool, named by the language they reach it through. `PythiaFromRunCardTool` and `SherpaFromRunCardTool` run the generator through its Python interface. `PythiaCPPFromRunCardTool` and `SherpaCPPFromRunCardTool` compile a small C++ driver against an install of your own and run that; `PythiaCPPFromProgramTool` and `SherpaCPPFromProgramTool` compile and run an agent-written C++ main program. A run card gives the same dataset, summary and manifest from either run-card tool, and the manifest records which generator ran under `inputs.generator`.

**The Python interface.** `tb install` installs the `pythia8mc` wheel with the `pythia` and `event_gen` bundles, but only where PyPI has a prebuilt wheel (Linux x86_64 and macOS 15+ arm64, Python 3.10 to 3.14). Elsewhere pip would compile Pythia from source, which takes tens of minutes, so heptapod skips the package and the tool says so. `PythiaFromRunCardTool` also accepts conda's `pythia8` module, or the `pythia8` extension of a Pythia 8 build configured `--with-python` when `pythia_path` names that build. The extension loads only under the Python serving heptapod and only one Pythia module can load per process; when it cannot be loaded the tool runs the module on `sys.path` and says so in its warnings. The extension a Pythia 8.317 build makes stores production vertices against the wrong particles (the wheel does not), so the tool refuses to write a store with vertices through it.

Sherpa is not installed by default in heptapod. We recommend building it yourself, following [Sherpa's installation guide](https://sherpa-team.gitlab.io/sherpa/master/getting-started.html), and setting `sherpa_path`, described in the next section. Where PyPI provides a `sherpa-mc` wheel (Linux x86_64 with glibc 2.34 or newer on Python 3.12 or 3.13, macOS 15 arm64 on Python 3.13), uncomment the `sherpa-mc` line under the `sherpa` bundle in `toolkit.yaml` and reinstall from a source checkout. The line is commented out because pip cannot check the glibc version and would otherwise build Sherpa from source. `SherpaFromRunCardTool` runs the wheel in a worker process; it also accepts the Python module of a Sherpa build with Python support when `sherpa_path` names that build. The Python interface exposes only the nominal weight, so a card with `SCALE_VARIATIONS` or `PDF_VARIATIONS` wants the C++ tool.

**C++ against your own install.** Point heptapod at a build:

```bash
tb config set heptapod pythia_path /path/to/pythia8    # holds bin/pythia8-config
tb config set heptapod sherpa_path /path/to/sherpa     # holds bin/Sherpa-config
tb install . --bundle pythia_cpp --bundle sherpa_cpp        # no generator wheels at all
```

With a path set, the `pythia_cpp` and `sherpa_cpp` bundles become available. `PythiaCPPFromRunCardTool` and `SherpaCPPFromRunCardTool` compile a small C++ driver against the install on first use (cached in `~/.cache/heptapod/cpp`, or `$HEPTAPOD_CPP_CACHE`) and generate through it, at whatever version and with whatever plugins the install was built. The driver streams the event record back into the same store spec, run summary and ROOT, HepMC3 and JSONL writers the Python tools use; it needs only the install's own headers and libraries and a C++ compiler on `PATH` (`CXX` overrides), not HepMC3 or Python bindings. The Sherpa driver also builds against the `sherpa-mc` wheel's own libraries, which ship with their headers, so `SherpaCPPFromRunCardTool` is served with the `sherpa` bundle too and needs no `sherpa_path` when the wheel is installed; the Pythia wheel ships no headers, so the Pythia C++ tools always need `pythia_path`.

`PythiaCPPFromProgramTool` and `SherpaCPPFromProgramTool` compile an agent-written C++ main program against the install with its own compiler flags, and run it in the sandbox. Use them for anything a run card cannot express: UserHooks, custom vetoes, reading the event record during generation, or calling the Sherpa API directly.

#### Rivet (`rivet`)

A conda-forge environment is the quickest install; `rivet_path` is its prefix, the directory holding `bin/rivet-config`:

```bash
conda create -n rivet -c conda-forge rivet
tb config set heptapod rivet_path "$(conda info --base)/envs/rivet"
```

#### Delphes (`delphes`)

Build [Delphes](https://github.com/delphes/delphes) with HepMC3 support and point `delphes_path` at the directory holding the `DelphesHepMC3` executable and the `cards/` it ships.

## Skills

Some bundles ship a skill, a written guide to using their tools well that covers the failure modes that are predictable but not obvious. They live in [`skills/`](../skills/):

| Skill | Covers | Bundle |
|-------|--------|--------|
| `feynrules` | Declaring mass and width once, tagging BSM couplings with an interaction order, checking the UFO before MG5 sees it | `feynrules` |
| `mg5` | Shallow comma decay chains and the `NP=N` alternative, `compute_widths` ordering, reading past MG5's auto-conversion error mask | `mg5` |
| `pythia` | Writing `.cmnd` run cards and standalone C++/Python main programs: silent boolean typos, the `Beams:LHEF` injection contract, seed precedence, and recovering the init error Pythia prints but the tool must discard | `event_gen` |
| `sherpa` | Writing Sherpa 3 YAML cards: the keys Sherpa ignores silently, beam energy per beam, cross sections and weights in pb, variations that need the driver, the integration Sherpa does at initialisation and reuses | `sherpa` |
| `rivet` | Comparing a generator with published measurements: choosing the analysis, matching its beams and energy (which Rivet itself does not check), streaming events into Rivet, reading χ² and p-values with the right scepticism, and the C++ route for a local Pythia | `rivet` |

Each skill names a bundle, and `tb connect` surfaces it only when that bundle's tools are being served, so the `mg5` guide stays hidden until `mg5_path` is set. `tb deactivate heptapod__mg5` turns one off; `tb activate` turns it back on.

How a skill reaches the agent depends on the harness:

- **Claude Code, Antigravity** load it automatically when the conversation looks relevant, and also expose it as a `/heptapod__mg5` slash command.
- **Codex, OpenCode** have no model-facing skill concept, so it arrives as a `/heptapod__mg5` slash command you invoke yourself.

Skills are surfaced by `tb connect` alongside the MCP server, so nothing extra is needed. Requires toolbase ≥ 0.11.

## Testing

```bash
python test_runner.py                # run all tests
python test_runner.py --skip-slow    # skip MG5, Pythia, Sherpa generation
python test_runner.py --only nda     # a single component
python test_runner.py --help         # all options
```

Individual tool suites can also be run directly with `pytest` (e.g. `pytest tools/analysis/`). During development, `tb validate` checks that `toolkit.yaml` and the tool modules are well-formed and servable.

`test_runner.py` runs against your own interpreter, not the isolated environment `tb install` builds, so components whose bundle deps you haven't installed fail on import (`--only pdg` without `pdg`, for instance). Install the ones you want to exercise: `pip install pdg feyngraph pylhe`.
