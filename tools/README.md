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

#### FeynRulesToUFOTool

**Purpose**: Generate Universal FeynRules Output (UFO) from FeynRules model files

**Input Parameters:**
- `model_path` (str): Path to FeynRules `.fr` model file
- `output_dir` (str): Directory for generated UFO output
- `feynrules_path` (str, optional): FeynRules installation path
- `wolframscript_path` (str, optional): WolframScript executable path

**Output:**
```json
{
  "status": "ok",
  "ufo_directory": "path/to/S1_LQ_RR_UFO",
  "files": ["couplings.py", "particles.py", "vertices.py", ...]
}
```

**Example:**
```python
from feynrules.feynrules_tools import FeynRulesToUFOTool

tool = FeynRulesToUFOTool(base_directory="./workspace")
result = tool.run(
    model_path="models/S1_LQ_RR.fr",
    output_dir="ufo_outputs"
)
```

**Requirements**: Mathematica ≤ v13.3, valid license

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
    mg5_path="/path/to/MG5_aMC_v3.6.6"
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

#### PythiaFromRunCardTool

**Purpose**: Hadronize and shower parton-level events using Pythia8

**Input Parameters:**
- `cmnd_path` (str): Path to Pythia command card (`.cmnd` file)
- `data_dir` (str): Output directory
- `n_events` (int): Number of events to generate
- `seed` (int, optional): Random seed
- `finals_only` (bool): Keep only final-state particles (default: True)
- `full_history` (bool): Include mother/daughter indices (default: False)
- `shower_lhe` (bool): Shower pre-generated LHE events (default: False)
- `lhe_path` (str, optional): Path to input LHE file (required if `shower_lhe=True`)

**Modes:**

1. **Standalone Generation** (`shower_lhe=False`):
   ```python
   result = tool.run(
       cmnd_path="cards/pythia_standalone.cmnd",
       data_dir="output",
       n_events=10000,
       seed=42
   )
   ```

2. **LHE Showering** (`shower_lhe=True`):
   ```python
   result = tool.run(
       cmnd_path="cards/pythia_shower.cmnd",
       data_dir="output",
       n_events=10000,
       shower_lhe=True,
       lhe_path="mg5_output/events.lhe"
   )
   ```

**Output:**
```json
{
  "status": "ok",
  "events_jsonl": "path/to/events.jsonl",
  "n_events": 10000,
  "schema_version": "evtjsonl-1.0"
}
```

**Example Command Card** (`pythia_shower.cmnd`):
```
! Beams
Beams:frameType = 4           ! LHEF input
Beams:LHEF = {LHE_FILE_PATH}  ! Will be substituted by tool

! Settings
Random:setSeed = on
Random:seed = {SEED}
PartonLevel:ISR = on
PartonLevel:FSR = on
HadronLevel:all = on

! Output
Next:numberShowEvent = 0
```

---

#### JetClusterSlowJetTool

**Purpose**: Perform anti-kt jet clustering on hadronized events

**Input Parameters:**
- `events_jsonl` (str): Path to hadronized events in JSONL format
- `R` (float): Jet clustering radius (typical: 0.4 or 0.8)
- `pT_min` (float): Minimum jet transverse momentum in GeV

**Output:**
```json
{
  "status": "ok",
  "jets_jsonl": "path/to/jets.jsonl",
  "n_events": 10000,
  "total_jets": 45678
}
```

**Example:**
```python
from pythia.pythia_tools import JetClusterSlowJetTool

tool = JetClusterSlowJetTool(base_directory="./workspace")
result = tool.run(
    events_jsonl="pythia_output/events.jsonl",
    R=0.4,
    pT_min=20.0
)
```

**Note**: Uses Pythia's SlowJet algorithm (exact but slower than FastJet).

---

### Data Conversion Tools

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

---

### BSM Model Building Tools

Turn a BSM paper into a FeynRules model, prove the tool chain accepts it,
then have an independent agent read the model back and compare it to the
paper. Each stage is its own bundle. Only `feynrules` needs Mathematica.

```
literature  ->  extract  ->  frgen  ->  feynrules  ->  reverse
paper text     spec         .fr        UFO+checks     REVIEW.pdf
```

A worked run is in [`examples/lagrangian_extraction/`](../examples/lagrangian_extraction/);
the reverse loop has its own guide in [`reverse/README.md`](reverse/README.md).

#### ArxivSearchTool / ArxivSourceTool / FetchPaperPDFTool / ExtractPaperTextTool

**Purpose**: Get the paper. INSPIRE indexes the metadata but serves neither
the PDF nor the LaTeX source, which is what the extraction actually reads.

Prefer `ArxivSourceTool` over the PDF path: PDF text extraction mangles
equations, and equations are the whole point here.

**Example:**
```python
from tools.literature.literature_tools import ArxivSourceTool

tool = ArxivSourceTool(arxiv_id="1603.04993", base_directory="./workspace")
```

#### ExtractLagrangianTool

**Purpose**: Paper text → a structured `FeynRulesModel` spec, via an LLM
constrained to the schema. Fields, quantum numbers, parameters and
Lagrangian terms come out as data, not prose.

#### GenerateFeynRulesModelTool

**Purpose**: A structured spec → FeynRules `.fr` source. Pure python and
deterministic — no Mathematica, and the same spec always renders the same
file.

**Input Parameters:**
- `model_json` (str): JSON matching the `FeynRulesModel` schema
- `output_path` (str, optional): `.fr` path relative to `base_directory`

**Example:**
```python
from tools.frgen.frgen_tool import GenerateFeynRulesModelTool

tool = GenerateFeynRulesModelTool(
    model_json=model.model_dump_json(),
    output_path="S1_LQ.fr",
    base_directory="./workspace",
)
```

#### ValidateModelTool

**Purpose**: Compile a `.fr` to a UFO with FeynRules, run the Hermiticity,
kinetic-term and mass-spectrum checks, and import the result into MadGraph.

**Input Parameters:**
- `model_path` (str): `.fr` file relative to `base_directory`
- `physics_checks` (bool, optional): run the FeynRules consistency checks
- `madgraph_check` (bool, optional): import the UFO into MadGraph
- `timeout_sec` (int, optional): compile budget; large multiplets need it raised

Passing means **the tool chain accepts the model**. It does not mean the
physics matches the paper — that is what `ReverseLagrangianTool` is for.

#### ReverseLagrangianTool

**Purpose**: Blank-slate reverse check. An agent that never saw the paper
reconstructs the physics from the sanitized `.fr` alone; a second fresh agent
grades that reconstruction against the paper term by term. Output is a
review package ending in a sign-off block — the tooling never issues the
verdict itself.

**Input Parameters:**
- `model_path` (str): `.fr` file relative to `base_directory`
- `action` (str): `reconstruct`, `crosscheck`, or `full`
- `paper_tex_path` (str, optional): paper source, required for the cross-check

Needs `blank_agent_cmd`. Full detail in [`reverse/README.md`](reverse/README.md).

#### SubmitJobTool / JobStatusTool / JobResultTool

**Purpose**: Run slow tools detached. UFO compiles and reverse checks take
minutes, which is longer than an agent should block for.

**Example:**
```python
from tools.jobs.jobs_tool import SubmitJobTool

tool = SubmitJobTool(
    tool_name="ValidateModelTool",
    tool_args='{"model_path": "S1_LQ.fr"}',   # JSON string
    base_directory="./workspace",
)
```

#### AuditTrailTool

**Purpose**: Record, render or read a structured `audit.json` provenance
ledger, so a multi-stage run leaves a reviewable trace of what ran, in what
order, and with what outcome.
