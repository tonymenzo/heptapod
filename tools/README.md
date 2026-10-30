# HEPTAPOD Tools

This directory contains the physics tools used by HEPTAPOD for event generation, hadronization, data conversion, and analysis.

## Data Format

### evtjsonl-1.0 Schema

All event data uses the `evtjsonl-1.0` schema (JSON Lines format):

**Particle Events:**
```json
{"event_id": 1, "data": {"particles": [
  {"i": 0, "id": 11, "status": 1, "px": 45.2, "py": -12.3, "pz": 156.7, "E": 165.4, "m": 0.000511},
  {"i": 1, "id": -11, "status": 1, "px": -38.1, "py": 52.4, "pz": -89.2, "E": 107.3, "m": 0.000511}
]}}
{"event_id": 2, "data": {"particles": [...]}}
```

**Jet Events:**
```json
{"event_id": 1, "data": {"jets": [
  {"i": 0, "px": 123.4, "py": 56.7, "pz": 234.5, "E": 278.9},
  {"i": 1, "px": 89.2, "py": -45.3, "pz": 167.8, "E": 198.2}
]}}
```

**Field Definitions:**
- `i`: Particle/jet index within event
- `id`: PDG particle ID (particles only)
- `status`: Status code (1 = final state, particles only)
- `px`, `py`, `pz`: Momentum components (GeV)
- `E`: Energy (GeV)
- `m`: Mass (GeV, particles only)

**Schema Validation:**
- All tools verify `"schema_version": "evtjsonl-1.0"`
- Forward/backward compatibility through version checking
- Pydantic models ensure type safety

### LLM-Compatible Data Representation

The JSONL (JSON Lines) format serves dual purposes in HEPTAPOD: efficient storage for large event datasets and native compatibility with Large Language Model contexts. Unlike binary formats (ROOT, HDF5), JSONL provides:

**Key Advantages:**
- **Human-readable**: Direct inspection without specialized tools
- **LLM-native**: Events can be directly included in prompts for analysis and debugging
- **Streaming-friendly**: Line-by-line processing minimizes memory overhead
- **Schema-extensible**: JSON structure supports arbitrary metadata without breaking compatibility
- **Cross-platform**: Pure text format eliminates endianness and architecture dependencies

**Design Rationale:**

The choice of JSONL over traditional HEP formats reflects the integration of AI-assisted workflows. LLMs can directly reason about event structure, identify anomalies, and suggest analyses when event data is represented in JSON. This enables natural language queries like "show me events with high-pT leptons" to be answered by the LLM directly examining JSONL records.

**Format Interoperability:**

While JSONL is the internal working format, HEPTAPOD provides conversion tools for standard HEP formats:

- **LHEToJSONLTool**: Import Les Houches Event files from MadGraph/other generators
- **EventJSONLToNumpyTool**: Export to zero-padded NumPy arrays for ML pipelines
- **JetsJSONLToNumpyTool**: Convert jet collections to vectorized arrays

This architecture allows seamless integration with existing analysis frameworks (ROOT, Pandas, PyTorch/TensorFlow) while maintaining LLM transparency throughout the workflow.

---

## Tool Suite Overview

### Event Generation Tools

#### GenerateFeynRulesModelTool

**Purpose**: Generate a FeynRules `.fr` model file from a structured JSON spec, so the regular and error-prone Mathematica blocks (`M$Parameters`, `M$ClassesDescription`, `M$GaugeGroups`) are rendered rather than hand-written. Irregular parts such as the Lagrangian algebra are carried verbatim.

**Input Parameters:**
- `model_json` (str): JSON matching the `FeynRulesModel` schema in `tools/frgen/frmodel.py`. Numbers are strings so exact FeynRules syntax survives (`-1/3`, `1500.`, `2.5*^-3`).
- `output_path` (str, optional): `.fr` path relative to `base_directory`. Defaults to `models/<model_name>.fr`.

**Output:**
```json
{
  "status": "ok",
  "model_name": "S1_LQ_RR",
  "fr_path": "models/S1_LQ_RR.fr",
  "n_particles": 1,
  "n_parameters": 3,
  "lagrangian_terms": ["LkinS1", "L1YukRR", "LBSM"]
}
```

`lagrangian_terms` names the symbols the file defines, so the caller can hand one to `FeynRulesToUFOTool` instead of guessing a convention that model files do not share. A malformed spec returns the Pydantic validation errors rather than a Mathematica parse failure three steps later.

---

#### FeynRulesToUFOTool

**Purpose**: Compile a FeynRules `.fr` model to a UFO directory via `wolframscript`, running consistency checks first and verifying the UFO is complete before reporting success.

**Input Parameters:**
- `model_path` (str): Path to the `.fr` model file.
- `lagrangian` (str): The total Lagrangian expression, for example `"LSM + LBSM"`, `"LNP"`, or `"LSM + LkinS1 + L1YukRR"`. **Required.** Model files share no naming convention, so this is never inferred; omit it and the tool reports which Lagrangian-like symbols the model actually defines.
- `output_dir` (str): Directory for the generated UFO.
- `load_sm` (bool, optional): Load FeynRules' `SM.fr` alongside the model. Default `true`; set `false` for a model that does not extend the SM.
- `run_checks` (bool, optional): Run the consistency checks before compiling. Default `true`.
- `checks` (str, optional): Which checks to run. Default set is `hermiticity`, `kinetic_diagonal`, `mass_diagonal`, `mass_spectrum`. Pass `"all"` to add `quadratic_diagonal` and `kinetic_norm`, which fail for the Standard Model itself (see `skills/feynrules/references/pitfalls.md` P11).
- `add_decays` (bool, optional): Have FeynRules compute 1→2 partial widths. Defaults on below Wolfram 14 and off above, where the routine is unreliable.
- `feynrules_path`, `wolframscript_path` (str): Installation paths.

**Output:**
```json
{
  "ok": true,
  "output_dir": "feynrules/S1_LQ_RR_UFO",
  "lagrangian": "LSM + LBSM",
  "checks_summary": "4 passed",
  "checks": [{"name": "hermiticity", "verdict": "pass", "message": "..."}],
  "files_created": ["particles.py", "vertices.py", "couplings.py", "..."]
}
```

`ok` means a usable UFO exists, not that the process exited cleanly. `WriteUFO` writes its boilerplate even when vertex extraction produced nothing, so the output is checked for `particles.py`, `vertices.py`, `couplings.py`, `parameters.py` and `coupling_orders.py`.

**Example:**
```python
from tools.feynrules import FeynRulesToUFOTool

tool = FeynRulesToUFOTool(
    base_directory="./workspace",
    feynrules_path=feynrules_path,
    wolframscript_path=wolframscript_path,
)
result = tool.run(
    model_path="models/S1_LQ_RR.fr",
    lagrangian="LSM + LBSM",
    output_dir="ufo_outputs",
)
```

---

#### ValidateLagrangianTool

**Purpose**: Run FeynRules' consistency checks on a model's Lagrangian without writing a UFO. This is the fast loop while a model is being edited, since `WriteUFO` is the expensive half.

**Input Parameters:** `model_path`, `lagrangian`, `load_sm`, `checks`, `baseline_compare` (as above).

**Output:** a verdict per check, three-valued. A check that could not be decided reports `inconclusive`, never `pass`, because several FeynRules check functions return `Null` rather than `False`. Each failure carries a message naming the remedy:

```json
{"name": "hermiticity", "verdict": "fail",
 "message": "The Lagrangian is not Hermitian: FeynRules found residual terms in L - HC[L]. Almost always a term is missing its conjugate -- add `+ HC[...]` around the offending interaction."}
```

---

#### ValidateUFOModelTool

**Purpose**: Compile a `.fr` and check what was produced: the expected UFO files exist, every declared particle reached the UFO with the right spin, colour and charge, and optionally that a partial width matches a closed-form expectation.

**Input Parameters:**
- `model_path`, `lagrangian`, `load_sm`: as above.
- `feynrules_model_json` (str, optional): the same `FeynRulesModel` spec `GenerateFeynRulesModelTool` consumes, used for the particle-property checks.
- `width_gate` (str, optional): a partial width to compare against an analytic formula.

A failed validation is a normal structured result with `passed: false`, not a tool error, so a repair loop can read it and act. UFO files are parsed with `ast` and never imported, since `particles.py` instantiates objects and imports the model package.

---

#### MadGraphFromRunCardTool

**Purpose**: Generate parton-level events using MadGraph5_aMC@NLO

**Input Parameters:**
- `command_card` (str): Path to MadGraph command card (`.mg5` file)
- `data_dir` (str): Output directory for generated events
- `ufo_path` (str, optional): Path to UFO model directory
- `output_name` (str, optional): Name for output process directory
- `nevents` (int, optional): Number of events to generate
- `seed` (int, optional): Random seed for reproducibility

**Output:**
```json
{
  "status": "ok",
  "lhe_path": "path/to/events.lhe",
  "n_events": 10000,
  "process_dir": "path/to/PROC_NAME",
  "scan_runs": ["run_01", "run_02", ...]  // If parameter scan detected
}
```

**Key Features:**
- Automatic template variable substitution in command cards
- Detection of parameter scans (auto-discovers `run_01`, `run_02`, etc.)
- Process-level provenance tracking

**Example Command Card** (`cards/signal.mg5`):
```
import model {UFO_PATH}
generate p p > s1 s1~, (s1 > l+ j, s1~ > l- j)
output {OUTPUT_NAME}
launch {OUTPUT_NAME}
  set nevents {N_EVENTS}
  set iseed {SEED}
  set ebeam1 7000
  set ebeam2 7000
done
```

**Example Usage:**
```python
from mg5.mg5_tools import MadGraphFromRunCardTool

tool = MadGraphFromRunCardTool(
    base_directory="./workspace",
    mg5_path="/path/to/MG5_aMC"
)

result = tool.run(
    command_card="cards/signal.mg5",
    data_dir="mg5_output",
    ufo_path="ufo/S1_LQ_RR_UFO",
    output_name="lq_pair_production",
    nevents=10000,
    seed=42
)
```

---

#### PythiaFromRunCardTool / PythiaCPPFromRunCardTool

**Purpose**: Generate hadron-level events with Pythia 8 from a `.cmnd` run card,
standalone or by showering a Les Houches (LHE) file. The two tools take the
same inputs and write the same dataset; they differ in how they reach Pythia.
`PythiaFromRunCardTool` runs Pythia's Python interface in-process: the
`pythia8mc` wheel, conda's `pythia8`, or the `pythia8` extension of a build
configured `--with-python` when `pythia_path` names it (`pythia_cpp` is not
needed for that). `PythiaCPPFromRunCardTool` compiles a small C++ driver
against the Pythia 8 install named by `pythia_path` and runs it, at whatever
version and with whatever plugins the install was built; it is part of the
`pythia_cpp` bundle and needs a C++ compiler on `PATH`.

**Input Parameters:**
- `cmnd_path` (str): Path to the Pythia run card (`.cmnd`)
- `data_dir` (str): Output directory
- `n_events` (int): Number of events to generate
- `seed` (int): Random seed in [1, 900000000], overriding any in the card;
  -1 (default) leaves the seed to the card
- `output_format` (str): `"root"` (default), `"hepmc3"` (for Delphes or Rivet)
  or `"jsonl"` (human-readable)
- `store` (str): `"finals+kinematics"` (default), `"finals"` or `"full"`
  (every particle, with lineage; as HepMC3 this is the complete vertex
  genealogy, which Rivet's decay-history projections need)
- `precision` (str): `"f32"` (default) or `"f64"` for stored kinematics
- `select_pids` (list[int]): Store only these |PDG id|s (default: all)
- `shower_lhe` (bool) / `lhe_path` (str): Shower an LHE file; the tool sets
  `Beams:frameType = 4` and `Beams:LHEF` itself
- `gzip_output` (bool): Write HepMC3 or JSONL gzipped
- `rivet_analyses` (list[str]): Stream the events into these Rivet analyses
  through a pipe instead of writing a file (needs `store="full"` and a
  configured `rivet_path`); `rivet_output` names the YODA file
- `max_output_gb` (float): Stop early before the dataset exceeds this size
  (default 2.0)
- `pythia_path` (state): A Pythia 8 install prefix (holds
  `bin/pythia8-config`); `PYTHIA_PATH` when unset. Required by the C++ tool;
  the Python tool uses it only for the build's own Python extension

**Example:**
```python
from tools.pythia import PythiaFromRunCardTool

tool = PythiaFromRunCardTool(base_directory="./workspace")
result = tool.execute(
    cmnd_path="cards/pp_ttbar.cmnd",
    data_dir="ttbar",
    n_events=10000,
    seed=42,
)
```

**Output** (abridged):
```json
{
  "status": "ok",
  "events": "ttbar/events.root",
  "format": "root",
  "store": "finals+kinematics",
  "accepted": 10000,
  "failed": 0,
  "backend": "pythia8mc",
  "xsec": {"sigmaGen_mb": 8.1e-07, "sigmaErr_mb": 3.2e-09,
           "sigmaGen_pb": 810.0, "sigmaErr_pb": 3.2, "weightSum": 10000.0},
  "weighted": false,
  "summary": {"particles_per_event": {"mean": 567.3}, "top_species": ["..."]},
  "pythia_log": "ttbar/pythia.log"
}
```

`data_dir` also holds `run.cmnd` (the card exactly as it ran, with includes
inlined and the tool's overrides appended), `pythia.log` and `manifest.json`
(the run's inputs, Pythia version, beams, cross section and weight statistics).
Read `summary` rather than the events to check a run; if `weighted` is true,
every histogram must use each event's `weight`. `backend` and the manifest's
`inputs.generator` record what ran Pythia: `pythia8mc` or `pythia8` for the
Python interface, `cpp` with the install's prefix and version for the driver.

---

#### ValidatePythiaRunCardTool

**Purpose**: Check a run card before generating with it. Every line is checked
against the installed Pythia's settings database and parser, and the mistakes
Pythia accepts silently (a boolean typo read as `off`, `5.5` read as `5`) are
reported too. With `dry_run_init=True` it also initialises Pythia and reports
settings a later line (typically a `Tune:`) silently replaced.

```python
from tools.pythia import ValidatePythiaRunCardTool

report = ValidatePythiaRunCardTool(base_directory="./workspace").execute(
    cmnd_path="cards/pp_ttbar.cmnd", dry_run_init=True,
)
```

---

#### PythiaSettingsSearchTool

**Purpose**: Find a Pythia setting by name or physics phrase (`"beam energy"` →
`Beams:eCM`), with its type, default and allowed range.

---

#### SherpaFromRunCardTool / SherpaCPPFromRunCardTool

**Purpose**: Generate hadron-level events with Sherpa 3 from a YAML run
card, into the same datasets the Pythia tools write. Sherpa cannot be
initialised twice in one process, so both run it in a process of their own.
`SherpaFromRunCardTool` runs Sherpa's Python interface in a worker: the
`sherpa-mc` wheel, or the module of a build with Python support when
`sherpa_path` names it; the interface exposes the nominal weight only.
PyPI has the wheel for Linux x86_64 with glibc 2.34 or newer on Python
3.12 and 3.13, and for macOS 15 arm64 on Python 3.13; elsewhere pip would
build Sherpa from source, so the toolkit ships the dependency commented
out. To install it, uncomment `sherpa-mc` under the `sherpa` bundle in
`toolkit.yaml` and reinstall from a source checkout.
`SherpaCPPFromRunCardTool` compiles a small C++ driver against a Sherpa
install, the one `sherpa_path` names or the wheel's own libraries, and is
the one that stores the card's weight variations.

**Input Parameters:**
- `card_path` (str): Path to the Sherpa `.yaml` run card
- `data_dir` (str): Output directory; Sherpa keeps its `Process/` and
  `Results/` there, so a second run in it reuses the integration
- `n_events` (int): Number of events to generate
- `seed` (int): Random seed, passed to Sherpa as `-R`; -1 (default) leaves
  it to the card or to Sherpa
- `output_format`, `store`, `precision`, `select_pids`, `chunk_size`,
  `gzip_output`, `max_output_gb`: as for `PythiaFromRunCardTool`. Under
  `store="full"` every particle carries the blob it came from and went
  into (`prod_vertex`, `end_vertex`) and Sherpa's production-step flag
  (`info`); the HepMC3 written is the vertex graph
- `ufo_path` (str): A UFO model directory, converted for the Sherpa in use
  on first use
- `rivet_analyses` (list[str]): Stream the events into these Rivet analyses
  instead of writing a file (needs `store="full"` and `rivet_path`)
- `sherpa_path` (state): A Sherpa install prefix (holds `bin/Sherpa-config`);
  `SHERPA_PATH` when unset. The C++ tool falls back to the wheel's libraries
  without it; the Python tool uses it only for the build's own module

**Output** (abridged):
```json
{
  "status": "ok",
  "events": "zee/events.root",
  "format": "root",
  "accepted": 10000,
  "backend": "cpp",
  "xsec": {"sigmaGen_pb": 1568.4, "sigmaErr_pb": 2.1,
           "sigmaGen_mb": 1.5684e-06, "sigmaErr_mb": 2.1e-09, "weightSum": 15684000.0},
  "weighted": false,
  "weight_variations": ["MUR=0.5__MUF=0.5__LHAPDF=-1", "MUR=2__MUF=2__LHAPDF=-1"],
  "summary": {"particles_per_event": {"mean": 221.7}, "top_species": ["..."]},
  "sherpa_log": "zee/sherpa.log"
}
```

Sherpa's event weight is a cross section in pb: in unweighted mode every
event carries the total cross section, and `weighted` reports whether the
weights vary. The cross section is given in pb as Sherpa reports it and in
mb beside it for comparison with Pythia.

---

#### ValidateSherpaRunCardTool

**Purpose**: Validate a Sherpa 3 YAML card before generating. Sherpa
ignores a key it does not know without a word and reports a bad value only
after the process integration has started.

**Input Parameters:**
- `card_path` (str) or `card_text` (str): The card
- `dry_run_init` (bool): Also initialise Sherpa on the card without an
  event: Sherpa's own fatal error if any, the settings it never read, and
  the effective configuration from its Settings_Report
- `work_dir` (str): Where the dry run keeps its files; the planned
  `data_dir` lets the generation reuse the integration
- `timeout_s` (int): Wall-clock limit for the dry run (default 1800)

**Output** (abridged):
```json
{
  "status": "ok",
  "valid": false,
  "counts": {"error": 1, "warning": 0},
  "findings": [{"severity": "error", "line": 2, "key": "BEAM_ENERGIEZ",
                "message": "'BEAM_ENERGIEZ' is not a Sherpa setting; it would be ignored without a word",
                "suggestion": "did you mean BEAM_ENERGIES?"}],
  "variations": false,
  "dry_run": {"ok": true, "fatal": [], "unused_settings": {},
              "effective_settings": [{"setting": "BEAMS", "default": "0, 0", "card": "2212", "final": "2212, 2212"}]},
  "sherpa_version": "3.0.3"
}
```

The static checks compare every key with the settings Sherpa 3.0.3 read in
a set of reference runs (`tools/sherpa/data/settings_snapshot.json`) and
the documented keys of the `PROCESSES` block, check the shape of the beams,
energies, processes and selectors, and flag Pythia settings that strayed
in.

---

#### PythiaCPPFromProgramTool / SherpaCPPFromProgramTool

**Purpose**: Compile a C++ main program against a Pythia 8 or Sherpa
install and run it in the sandbox, for what a run card cannot express:
`UserHooks`, custom vetoes, analysis inside the generation loop, the Sherpa
API. Part of the `pythia_cpp` and `sherpa_cpp` bundles, gated on
`pythia_path` and `sherpa_path`; the Sherpa tool is in the `sherpa` bundle
too, since it also builds against the `sherpa-mc` wheel's own libraries. A
C++ compiler must be on `PATH` (`CXX` overrides it).

**Input Parameters:**
- `source_path` (str): The C++ source holding `main()` (`.cc`, `.cpp`, `.cxx`, `.C`)
- `work_dir` (str): Directory to build into and run in; the program's relative
  output paths land here
- `extra_sources` (list[str]): Further sources to compile and link
- `args` (list[str]): Command-line arguments for the program
- `extra_cxxflags` / `extra_ldflags` (list[str]): Extra compiler and linker
  flags, e.g. for FastJet, HepMC3 or Rivet
- `build_only` (bool): Compile and link without running
- `timeout_s` (int): Wall-clock limit for the run (default 3600)

**Output** (abridged):
```json
{
  "status": "ok",
  "generator": {"generator": "pythia", "prefix": "/opt/pythia8317", "version": "8.317"},
  "exe": "work/main_rivet",
  "build": {"cached": false, "command": "c++ -std=c++17 ..."},
  "run": {"returncode": 0, "elapsed_s": 41.2, "stdout_log": "work/main_rivet.stdout.log",
          "stdout_tail": "...", "changed_files": ["work/out.yoda"]}
}
```

The build is compiled with the install's own flags (from `pythia8-config` or
`Sherpa-config`) and cached by a hash of the sources and flags, so a rerun
of an unchanged program skips the compiler. `changed_files` lists what the
run wrote under `work_dir`.

---

#### JetClusterSlowJetTool

**Purpose**: Cluster jets with Pythia's SlowJet, for one event or a whole
dataset (streamed, so any size)

**Input Parameters:**
- `events_path` (str): Dataset from `PythiaFromRunCardTool` (.root, .hepmc
  or .jsonl), or `npy_path` for a padded NumPy array
- `event_index` (int): One event to cluster, or `cluster_all=True` with an
  `output_path` for every event
- `algorithm` (str): `"antikt"` (default), `"kt"` or `"ca"`; `R` (default 0.4)
- `ptmin` (float): Minimum jet pT in GeV (default 20); `etamax` (default 5)
- `include_invisible` (bool): Also cluster invisible particles (default False)

**Example:**
```python
from tools.pythia import JetClusterSlowJetTool

tool = JetClusterSlowJetTool(base_directory="./workspace")
result = tool.execute(
    events_path="ttbar/events.root",
    cluster_all=True,
    output_path="ttbar/jets.jsonl",
    R=0.4,
    ptmin=20.0,
)
```

Each jet reports `pT`, `eta` (pseudorapidity), `y` (rapidity), `phi`, its
four-momentum and its constituents, indexed into the stored event.

---

### Data Conversion Tools

#### ConvertEventsTool

**Purpose**: Convert an event dataset between ROOT, HepMC3 and JSONL; the
formats come from the file suffixes. HepMC3 output takes the run's beams,
weight names and cross section from the `manifest.json` beside the input, or
the beams from `beam_ids` and `beam_energies_gev`. A complete record
(`store="full"`) becomes the full vertex genealogy; a final-state record the
final state between the two beams (`hepmc_record` overrides the choice).

```python
from tools.events import ConvertEventsTool

ConvertEventsTool(base_directory="./workspace").execute(
    input_path="ttbar/events.root", output_path="ttbar/events.hepmc",
)
```

---

### Rivet Tools (`rivet` bundle)

Generator predictions against published measurements, through a local Rivet
installation (`rivet_path`, the prefix holding `bin/rivet-config`; a
conda-forge environment from `conda create -n rivet -c conda-forge rivet`
works). Everything Rivet-side runs in the installation's own Python, so the
serving interpreter need not match it.

#### RivetFindAnalysesTool

**Purpose**: Search Rivet's ~2000 analyses by measurement, experiment, beams
and √s, or describe one: what to generate (`run_info`), its options, its
histograms, its references, and whether to trust it.

```python
from tools.rivet import RivetFindAnalysesTool

RivetFindAnalysesTool(base_directory="./workspace").execute(
    query="Z transverse momentum", beams="p+ p+", energy_gev=13000)
RivetFindAnalysesTool(base_directory="./workspace").execute(name="ATLAS_2019_I1736531")
```

#### RunRivetTool

**Purpose**: Run analyses on a HepMC3 file (plain or gzipped) and write YODA
histograms. Before running, it checks that each analysis exists, that the
events' beams and energy are what the analysis was measured at (Rivet runs
an analysis at the wrong energy without a word), and that a cross section is
available when one is needed. Reports what Rivet warned about, how many
events were processed, and which histograms stayed empty.

```python
from tools.rivet import RunRivetTool

RunRivetTool(base_directory="./workspace").execute(
    hepmc_path="zpt/events.hepmc.gz",
    analyses=["ATLAS_2019_I1736531", "MC_ZINC:LMODE=MU"],
    output_path="zpt/rivet.yoda",
)
```

The events must be the complete record (`store="full"` in
`PythiaFromRunCardTool`): Rivet's projections walk the decay history.
`PythiaFromRunCardTool(rivet_analyses=[...])` streams the events into Rivet
through a pipe instead, writing no event file.

#### RivetCompareTool

**Purpose**: Compare the histograms with the shipped reference data, bin by
bin: χ², degrees of freedom and p-value, the same for the shape alone (the
prediction scaled to the data's integral), the largest pull, the mean
prediction/data ratio, and the number of bins the prediction left empty.
The full per-bin tables go to a JSON report.

```python
from tools.rivet import RivetCompareTool

RivetCompareTool(base_directory="./workspace").execute(
    yoda_paths=["zpt/rivet.yoda", "zpt_tuneB/rivet.yoda"], labels=["Monash", "tune B"])
```

#### RivetPlotTool

**Purpose**: Draw every histogram in Rivet's layout -- data points with error
bars, each prediction as a stepped line with its statistical band, the
envelope of the weight variations, and a prediction/data ratio panel --
labelled from the analysis's `.plot` file, with an `index.html` gallery.

```python
from tools.rivet import RivetPlotTool

RivetPlotTool(base_directory="./workspace").execute(
    yoda_paths=["zpt/rivet.yoda"], labels=["Pythia 8.317"], output_dir="zpt/plots")
```

#### RivetMergeTool

**Purpose**: Combine the YODA outputs of independent runs (different seeds,
or different processes with their cross sections) with `rivet-merge`, which
reruns each analysis's `finalize` on the merged histograms.

```python
from tools.rivet import RivetMergeTool

RivetMergeTool(base_directory="./workspace").execute(
    yoda_paths=["zpt_s1/rivet.yoda", "zpt_s2/rivet.yoda"], output_path="zpt/merged.yoda")
```

---

#### LHEToJSONLTool

**Purpose**: Convert Les Houches Event (LHE) files to JSONL format

**Input Parameters:**
- `lhe_path` (str): Path to `.lhe` file
- `output_dir` (str): Output directory
- `finals_only` (bool): Keep only final-state particles (default: True)
- `full_history` (bool): Include mother indices (default: False)

**Example:**
```python
from utils.data_conversion_tools import LHEToJSONLTool

tool = LHEToJSONLTool(base_directory="./workspace")
result = tool.run(
    lhe_path="mg5_output/events.lhe",
    output_dir="converted",
    finals_only=True
)
```

---

#### EventJSONLToNumpyTool

**Purpose**: Convert event JSONL to zero-padded NumPy arrays

**Input Parameters:**
- `events_jsonl` (str): Path to JSONL event file
- `output_dir` (str): Output directory

**Output:**
- NumPy array with shape `(N_events, N_max_particles, 5)`
- Columns: `[px, py, pz, E, pid]`
- Zero-padded to match longest event

**Example:**
```python
from utils.data_conversion_tools import EventJSONLToNumpyTool

tool = EventJSONLToNumpyTool(base_directory="./workspace")
result = tool.run(
    events_jsonl="pythia_output/events.jsonl",
    output_dir="numpy_arrays"
)
# Output: workspace/numpy_arrays/events.npy
```

---

#### JetsJSONLToNumpyTool

**Purpose**: Convert jet JSONL to NumPy arrays with flexible extraction modes

**Input Parameters:**
- `jets_jsonl` (str): Path to jets JSONL file
- `output_dir` (str): Output directory
- `extraction_mode` (str): How to extract jets - `"by_pt"`, `"by_index"`, or `"n_hardest"`
- `n_jets` (int): Number of jets to extract
- `jet_indices` (list, optional): Specific jet indices (for `by_index` mode)

**Extraction Modes:**

1. **`by_pt`**: Extract N highest-pT jets per event
   ```python
   result = tool.run(
       jets_jsonl="jets.jsonl",
       output_dir="numpy",
       extraction_mode="by_pt",
       n_jets=4
   )
   ```

2. **`n_hardest`**: Same as `by_pt` (alias)

3. **`by_index`**: Extract specific jet indices
   ```python
   result = tool.run(
       jets_jsonl="jets.jsonl",
       output_dir="numpy",
       extraction_mode="by_index",
       jet_indices=[0, 1]  # First two jets
   )
   ```

**Output:** NumPy array with shape `(N_events, N_jets, 5)`, columns: `[px, py, pz, E, jet_id]`

---

### Analysis Tools

All analysis tools support both **file-based** and **legacy direct-input** modes. File-based mode is recommended for production workflows.

#### CalculateInvariantMassTool

**Purpose**: Calculate invariant mass M² = (ΣE)² - (Σp)²

**File-based Mode:**
```python
from utils.analysis_tools import CalculateInvariantMassTool

tool = CalculateInvariantMassTool(base_directory="./workspace")
result = tool.run(
    particle_array_paths=["leptons.jsonl", "jets.jsonl"],
    output_dir="results"
)
# Output: workspace/results/invariant_masses.npy
```

**Direct Mode (Legacy):**
```python
import numpy as np
particles = np.load("events.npy")  # Shape: (N_events, N_particles, 5)
result = tool.run(particle_arrays=[particles])
masses = result["invariant_masses"]  # Shape: (N_events,)
```

---

#### CalculateTransverseMomentumTool

**Purpose**: Calculate pT = √(px² + py²)

**Example:**
```python
from utils.analysis_tools import CalculateTransverseMomentumTool

tool = CalculateTransverseMomentumTool(base_directory="./workspace")
result = tool.run(
    particle_array_paths=["particles.jsonl"],
    output_dir="results"
)
# Output: workspace/results/pt.npy (shape: N_events × N_particles)
```

---

#### CalculateDeltaRTool

**Purpose**: Calculate ΔR = √(Δη² + Δφ²) between particle pairs

**Parameters:**
- `particle_array_paths`: List of particle files
- `index_pairs`: List of `[i, j]` pairs specifying which particles to compare
- `output_dir`: Output directory

**Example:**
```python
from utils.analysis_tools import CalculateDeltaRTool

tool = CalculateDeltaRTool(base_directory="./workspace")
result = tool.run(
    particle_array_paths=["particles.jsonl"],
    index_pairs=[[0, 1], [0, 2]],  # ΔR(particle0, particle1) and ΔR(particle0, particle2)
    output_dir="results"
)
# Output: workspace/results/delta_r_0_1.npy, delta_r_0_2.npy
```

---

#### ApplyCutsTool

**Purpose**: Apply physics cuts (pT, η, PDG filters)

**Parameters:**
- `particle_array_paths`: Input particles
- `pt_min`, `pt_max`: Transverse momentum range (GeV)
- `eta_min`, `eta_max`: Pseudorapidity range
- `pdg_ids`: List of allowed PDG codes
- `output_dir`: Output directory

**Example:**
```python
from utils.analysis_tools import ApplyCutsTool

tool = ApplyCutsTool(base_directory="./workspace")
result = tool.run(
    particle_array_paths=["events.jsonl"],
    pt_min=20.0,
    eta_max=2.5,
    pdg_ids=[11, -11, 13, -13],  # electrons and muons
    output_dir="selected"
)
```

---

#### GetHardestNTool / GetHardestNJetsTool

**Purpose**: Select N highest-pT particles or jets per event

**Example:**
```python
from utils.analysis_tools import GetHardestNTool

tool = GetHardestNTool(base_directory="./workspace")
result = tool.run(
    particle_array_paths=["leptons.jsonl"],
    n=2,  # Select 2 hardest leptons
    output_dir="hardest"
)
# Output: workspace/hardest/hardest_2_particles.jsonl
```

---

#### FilterByPDGIDTool

**Purpose**: Filter particles by PDG ID

**Example:**
```python
from utils.analysis_tools import FilterByPDGIDTool

tool = FilterByPDGIDTool(base_directory="./workspace")
result = tool.run(
    particle_array_paths=["events.jsonl"],
    pdg_ids=[11, -11],  # Keep only electrons
    output_dir="electrons"
)
```

---

#### SortByPtTool

**Purpose**: Sort particles by transverse momentum (descending)

---

#### FilterByDeltaRTool

**Purpose**: Remove overlapping objects with ΔR below threshold

**Parameters:**
- `particle_array_paths`: Input particles
- `min_delta_r`: Minimum allowed ΔR separation
- `output_dir`: Output directory

**Example:**
```python
from utils.analysis_tools import FilterByDeltaRTool

tool = FilterByDeltaRTool(base_directory="./workspace")
result = tool.run(
    particle_array_paths=["jets.jsonl"],
    min_delta_r=0.4,  # Remove jets closer than ΔR = 0.4
    output_dir="isolated_jets"
)
```

---

### Resonance Reconstruction

#### ResonanceReconstructionTool

**Purpose**: Template-based reconstruction of physics resonances from pre-selected objects

**Templates:**
- `"two_body_symmetric"`: Pair-produced resonances (e.g., leptoquark pairs)
- `"n_body_all_pairs"`: All possible combinations

**Input Parameters:**
- `particle_arrays` (list): Paths to pre-selected object files (e.g., `["hardest_2_leptons.jsonl", "hardest_2_jets.jsonl"]`)
- `template` (str): Analysis template
- `max_k` (int): Maximum k-body multiplicity (default: 2)
- `min_delta_r` (float, optional): Cross-array ΔR constraint

**Physics-Aware Design:**
- ΔR constraints apply **only between different arrays** (e.g., leptons vs jets)
- Objects within same array are **not** constrained (e.g., two leptons can overlap)

**Example: Leptoquark Pair Production Analysis**

```python
from utils.resonance_reconstruction_tool import ResonanceReconstructionTool

tool = ResonanceReconstructionTool(base_directory="./workspace")

# Assume we have pre-selected objects:
# - hardest_2_leptons.jsonl (2 hardest leptons per event)
# - hardest_2_jets.jsonl (2 hardest jets per event)

result = tool.run(
    particle_arrays=[
        "selected/hardest_2_leptons.jsonl",
        "selected/hardest_2_jets.jsonl"
    ],
    template="two_body_symmetric",
    max_k=2,
    min_delta_r=0.4  # Leptons and jets must be separated by ΔR > 0.4
)
```

**Output:**
```json
{
  "status": "ok",
  "template": "two_body_symmetric",
  "n_arrays": 2,
  "n_total_objects": 4,
  "n_events_analyzed": 10000,
  "n_events_successful": 9876,
  "n_events_failed": 124,
  "observables": [
    {"name": "m1", "kind": "per_event"},
    {"name": "m2", "kind": "per_event"},
    {"name": "Delta_m", "kind": "per_event"},
    {"name": "m_avg", "kind": "per_event"}
  ],
  "histograms": [
    {
      "observable": "m1",
      "bins": [0, 100, 200, ...],
      "counts": [45, 123, 234, ...]
    }
  ],
  "data_paths": {
    "m1": "workspace/results/m1.npy",
    "m2": "workspace/results/m2.npy",
    "Delta_m": "workspace/results/Delta_m.npy"
  }
}
```

**Generated Observables:**
- `m1`, `m2`: Individual resonance masses
- `Delta_m`: Mass difference |m1 - m2|
- `m_avg`: Average mass (m1 + m2) / 2

---

### NDA & Diagram Enumeration Tools

#### EstimateDecayWidthNDATool

**Purpose**: Estimate decay widths using Naive Dimensional Analysis (arXiv:1402.1178)

**Input Parameters:**
- `diagram` (dict): Diagram specification with `initial`, `final`, `vertices`, and optionally `propagators`, `couplings`, `color_factor`
- `process_label` (str, optional): Display label (e.g., `"H → bb̄"`)
- `reference_width` (float, optional): Reference width in GeV for comparison
- `include_summary` (bool): Include formatted summary (default: True)

**Output:**
```json
{
  "status": "ok",
  "width_gev": 0.00437,
  "width_gev_nda_raw": 0.01749,
  "method": "improved (arXiv:1402.1178)",
  "formula": "\\Gamma \\sim \\frac{y^2 M}{16\\pi}"
}
```

**Example:**
```python
from tools.nda import EstimateDecayWidthNDATool

tool = EstimateDecayWidthNDATool(
    diagram={
        "initial": [{"label": "H", "spin": 0, "mass": 125.1}],
        "final": [
            {"label": "b", "spin": 0.5, "mass": 4.18},
            {"label": "bbar", "spin": 0.5, "mass": 4.18}
        ],
        "vertices": [{"type": "yukawa", "coupling": "yb"}],
        "couplings": {"yb": 0.0242},
        "color_factor": 3.0
    },
    base_directory="./workspace"
)
```

**Note — back-to-back kinematic correction**: For 2-body decays with
fermionic final states, the improved estimate applies a factor-of-2
correction to account for back-to-back kinematics. In 2-body decays
$\vec{p}_1 = -\vec{p}_2$, so $p_1 \cdot p_2 = 2 E_1 E_2$ (massless)
rather than the $\sim E_1 E_2$ assumed by the NDA polarization factors.
This correction is vertex-type-dependent: it applies to unprojected trace
structures (`yukawa`, `vector`, `va`, etc.) but not to chiral-projected
vertices (`left-handed`, `right-handed`, `chiral`), where the $P_L$
projector halves the trace and exactly cancels the kinematic enhancement.
No correction is applied for $n \geq 3$ body decays.

---

#### EstimateBranchingRatioNDATool

**Purpose**: Estimate branching ratios across multiple diagram classes

**Input Parameters:**
- `diagram_classes` (list): List of `{"diagram": {...}, "n_diagrams": N}` dicts (or `"diagram_path"` by reference)
- `reference_width` (float): Total width in GeV for BR denominator
- `process_label` (str, optional): Display label
- `reference_label` (str): Label for the reference width (default: `"Total width"`)

**Output:**
```json
{
  "status": "ok",
  "n_classes": 2,
  "n_diagrams_total": 126,
  "partial_width_gev": 1.23e-20,
  "branching_ratio": 4.56e-10,
  "class_results": [...]
}
```

---

#### EnumerateDiagramsTool

**Purpose**: Enumerate all Feynman diagrams for a process using FeynGraph

**Input Parameters:**
- `initial` (list): Initial-state particle labels, e.g., `["H"]`
- `final` (list): Final-state particle labels, e.g., `["b", "bbar"]`
- `max_loops` (int): Maximum loop order (default: 0 = tree level)
- `metadata_only` (bool): Return only class counts and representative diagrams (default: False)
- `save_diagrams` (bool): Save JSON + SVG files to disk (default: True)

**Output:**
```json
{
  "status": "ok",
  "n_diagrams": 3,
  "process": "H -> b bbar",
  "diagrams": [...],
  "br_classes": [...]
}
```

The `metadata_only` output includes a `br_classes` array that can be passed
directly to `EstimateBranchingRatioNDA` as `diagram_classes`.

**Example:**
```python
from tools.feyngraph import EnumerateDiagramsTool

tool = EnumerateDiagramsTool(
    initial=["mu-"],
    final=["e-", "nu_ebar", "nu_mu"],
    metadata_only=True,
    base_directory="./workspace"
)
```

### LLP Reach Tools (`llp` bundle)

**Purpose:** Decay-in-volume long-lived-particle (LLP) reach studies — sampling a weighted LLP flux from an analytic parent-flux kernel and computing signal yields over a coupling grid with exact g² reweighting.

The bundle is setting-agnostic: the collider-forward vs beam-dump choice lives entirely in the kernel YAML (parent mass, momentum/angle scales, per-interaction normalization), the geometry YAML, and the `n_int` normalization — never in the tools. Pure Python + numpy + yaml; no external HEP software and no config gating.

Declared conventions (shared by both tools):
- **g²-stripped weights**: `event_weight_g2_stripped = n_per_int * kappa / n_samples`, so `N_sig(g) = N_int * g² * Σᵢ wᵢ * P_dec,i(g) * accᵢ`.
- **Decaying-parents-only kernels**: the kernel parametrizes d²N/(dp dθ) of decaying parents per primary interaction (decay-before-absorption folded into `n_per_int`).
- **Straight-line propagation** of the LLP from the primary vertex at its lab angle.
- **Midpoint two-track convention**: the phi → μ⁺μ⁻ acceptance is evaluated at the midpoint of the in-volume segment, keeping it g-independent so one event set covers the whole coupling grid exactly.

#### LLPFluxFromMesonDecayTool

Samples weighted LLP lab-frame events from an analytic parent-flux kernel (form `pexp_texp`) convolved with the tree-level 3-body production spectrum M → μνφ (scalar radiated off the muon leg). Writes line-delimited JSON records (`E`, `px`, `py`, `pz`, `theta_lab`, `parent_channel`, `event_weight_g2_stripped`) plus a manifest declaring the g²-stripped weight convention. A kinematically closed channel (m_φ ≥ m_M − m_μ) returns status ok with `n_samples = 0` and the cutoff flagged.

**Input Parameters:**
- `kernel_spec` (str): Path to the kernel YAML (carries parent name, `parent_mass_gev`, `n_per_int`, `p0_gev`, `a`, `theta0_rad`)
- `m_phi_gev` (float): LLP mass in GeV
- `m_mu_gev` (float): Muon mass in GeV (default: 0.1056584)
- `kappa` (float): Declared branching coefficient at g = 1, i.e. Br(M → μνφ) = g² κ
- `n_samples` (int): Number of Monte Carlo events
- `seed` (int): RNG seed (deterministic output)
- `output_path` (str): Events file path (line-delimited JSON)

**Example:**
```python
from tools.llp import LLPFluxFromMesonDecayTool

tool = LLPFluxFromMesonDecayTool(
    kernel_spec="kernel/kaon.yaml",
    m_phi_gev=0.25,
    kappa=1.3e-5,
    n_samples=200000,
    seed=42,
    output_path="flux/phi_K.jsonl",
    base_directory="./workspace"
)
# Output: flux/phi_K.jsonl + flux/phi_K.manifest.json
```

#### DecayInVolumeTool

Computes decay-in-volume yields N_sig(g) over a coupling grid from g²-stripped events (multiple channels may be concatenated into one file). Per event: in-volume segment [L1, L2] from the on-axis cylinder intersection, `P_dec = exp(-L1/λ) − exp(-L2/λ)` with `λ = βγ cτ` and `cτ = ħc / (g² width_ref)`; midpoint two-track acceptance. Writes a yields table plus a per-event audit file (`parent_channel`, `beta_gamma`, `L1_m`, `L2_m`, `geom_pass`, `two_track_pass`, `p_decay_at_g_ref`, `weighted_contribution_at_g_ref`).

**Input Parameters:**
- `events_path` (str): Line-delimited JSON events (LLPFluxFromMesonDecayTool output format)
- `geometry_path` (str): Geometry YAML (`z_min_m`, `z_max_m`, `r_volume_m`, `z_det_m`, `r_det_m`)
- `m_phi_gev` / `m_mu_gev` (float): LLP / muon masses in GeV
- `width_ref_gev` (float): Γ_tot at g = 1 (g²-stripped reference width)
- `g_grid` (list[float]): Couplings at which to evaluate N_sig (exact reweighting)
- `n_int` (float): Number of primary interactions (σ_inel·L_int or N_POT)
- `require_two_track` (bool): Apply the two-track acceptance (default: True)
- `seed` (int): RNG seed for the isotropic φ → μμ decay sampling (default: 1)
- `output_path` (str): Yields table JSON path

**Example:**
```python
from tools.llp import DecayInVolumeTool

tool = DecayInVolumeTool(
    events_path="flux/phi_all_channels.jsonl",
    geometry_path="geometry/geometry.yaml",
    m_phi_gev=0.25,
    width_ref_gev=1.52e-3,
    g_grid=[1e-8, 3e-8, 1e-7, 1e-6],
    n_int=2.2e16,
    output_path="yields/m0p25.json",
    base_directory="./workspace"
)
# Output: yields/m0p25.json + yields/m0p25.audit.jsonl
```
---

### Pipeline Plumbing

Two pure-python bundles for long, multi-stage runs.

#### SubmitJobTool / JobStatusTool / JobResultTool / CancelJobTool

**Purpose**: Run slow tools detached. Lagrangian checks on a large model, UFO
compiles, MadGraph and Pythia runs take minutes, which is longer than an agent
should block for. Submit returns a `job_id` at once; poll the state, then fetch
the tool's verbatim output. `canceljob` stops a job that is no longer wanted.

Only tools in the whitelist in `tools/jobs/job_manager.py` can run as jobs:
`feynrulestoufo`, `validatelagrangian`, `validateufomodel`,
`madgraphfromruncard`, `pythiafromruncard`, `delphesfromhepmc`,
`reverselagrangian`, and `audittrail`. `tool_args` takes the target tool's
runtime arguments only; paths to executables come from server config.

A job is `queued`, `running`, `done`, `failed` or `cancelled`. Stopping a job,
by `canceljob` or the `max_seconds` watchdog, stops every process its tool
started, not just the runner.

**Progress.** A tool reports progress with `report_progress(self, done, total,
unit=..., message=...)` from `tools/jobs/progress.py`. As a job, that lands in
`status.json`, and `jobstatus` shows it with a rough ETA:

```
[################--------]  67%  2/3 scan points  ETA 10s  run_03 · survey
```

MadGraph reports the stage, its integration jobs and, for a scan, the points
done. Called directly with no callback, `report_progress` does nothing.

**Example:**
```python
from tools.jobs.jobs_tool import SubmitJobTool

tool = SubmitJobTool(
    tool_name="validateufomodel",
    tool_args='{"model_path": "S1_LQ.fr", "lagrangian": "LSM + LBSM"}',  # JSON string
    base_directory="./workspace",
)
```

#### AuditTrailTool

**Purpose**: Record, render or read a structured `audit.json` provenance
ledger, so a multi-stage run leaves a reviewable trace of what ran, in what
order, and with what outcome.
---

### Reverse Lagrangian Check

#### ReverseLagrangianTool

**Purpose**: Blank-slate reverse check of a `.fr` model. An agent that never
saw the paper reconstructs the physics from the sanitized `.fr` alone. A second
fresh agent grades that reconstruction against the paper, term by term. The
output is a review package that ends in a sign-off block. The tooling never
issues the verdict itself.

Passing the FeynRules checks means the tool chain accepts the model. It does
not mean the model matches the paper. This tool tests that second question.

**Input Parameters:**
- `model_path` (str): `.fr` file relative to `base_directory`
- `action` (str): `reconstruct`, `crosscheck`, or `full`
- `paper_tex_path` (str, optional): paper source, required for the cross-check

Needs `blank_agent_cmd`. The reverse check takes minutes; submit it with
`SubmitJobTool(tool_name="reverselagrangian", ...)`. Full detail in
[`reverse/README.md`](reverse/README.md).
