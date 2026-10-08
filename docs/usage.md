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

Most bundles (`units`, `inspire`, `pdg`, `nda`, `analysis`, `bsm`) work out of the box; toolbase installs their pip dependencies automatically. The following bundles expect additional software on the system:

#### Mathematica and WolframScript (`eda`, `feynrules`)

1. Install [Mathematica](https://www.wolfram.com/mathematica/) (includes WolframScript)
2. Install [FeynCalc](https://feyncalc.github.io/) (for `eda`)
3. Authenticate: `wolframscript -authenticate`
4. Optionally install [FeynRules](https://feynrules.irmp.ucl.ac.be/) v2.3.49 (for UFO model generation)
5. Register the installed software using the [configuration commands](../README.md#configuration).

#### MadGraph5_aMC@NLO (`mg5`, `event_gen`)

```bash
wget https://launchpad.net/mg5amcnlo/3.0/3.6.x/+download/MG5_aMC_v3.6.6.tar.gz
tar -xzf MG5_aMC_v3.6.6.tar.gz
tb config set heptapod mg5_path "$(pwd)/MG5_aMC_v3.6.6"
```

#### Pythia8 and Sherpa3 (`event_gen`)

**Installed automatically** as bundle dependencies when you `tb install . --bundle event_gen`. No separate installation needed.

## Testing

```bash
python test_runner.py                # run all tests
python test_runner.py --skip-slow    # skip MG5, Pythia, Sherpa generation
python test_runner.py --only nda     # a single component
python test_runner.py --help         # all options
```

Individual tool suites can also be run directly with `pytest` (e.g. `pytest tools/analysis/`). During development, `tb validate` checks that `toolkit.yaml` and the tool modules are well-formed and servable.

`test_runner.py` runs against your own interpreter, not the isolated environment `tb install` builds, so components whose bundle deps you haven't installed fail on import (`--only pdg` without `pdg`, for instance). Install the ones you want to exercise: `pip install pdg feyngraph pylhe`.
