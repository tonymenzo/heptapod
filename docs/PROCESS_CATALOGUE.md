# Process catalogue prototype

This branch provides three agent-facing operations: `InspectProcessModelTool`,
`EnumerateProcessesTool`, and `ValidateProcessCatalogueTool`. They separate model
inspection, bounded enumeration, and independent graph validation. The existing
single-process wrapper now passes the selected model into FeynGraph instead of
silently using its default model. Tau aliases resolve to the installed model's
actual particle names.

## Scope and guarantees

Version `0.1` covers tree-level **1 -> 2 decays only**, with flavors resolved and
charge conjugates separate. Final states are unordered multisets. It enumerates
all physical parent/final-state combinations within the requested parent list,
subject to an explicit candidate budget. UFO ghost and Goldstone flags determine
excluded particles. The output accounts for every candidate as accepted, closed,
without diagrams, or filtered by zero couplings/order constraints.

`model="SM"` uses FeynGraph's locally built-in Standard Model. No download or
FeynRules installation is needed. Its public API exposes structural data, not
numerical parameters or full vertex terms: without a complete explicit mass map,
its catalogue is structural, **not a list of physically open decays**. The built-in
model is not assumed equivalent to an unrestricted SM UFO.

For a parameter-specific census or independent validation, supply an explicit,
trusted UFO directory. Parameters and couplings are evaluated from UFO defaults
or a supplied SLHA card; unresolved parameters and incomplete cards fail. Strict
on-shell kinematics uses parent mass greater than the sum of daughter masses.
Zero coupling terms are removed. Named coupling orders are exact powers;
unspecified orders remain unconstrained. This first version rejects vertices
with mixed coupling orders and UFO/native inventory mismatches rather than
claiming support for every UFO convention. Restriction cards are not implicitly
applied. Unsupported or invalid models never fall back to the SM.

The MG5 validator regenerates the channel inventory from the same unrestricted
UFO, maps MG interactions back to ordered signed UFO particle PDGs, Lorentz
structures and indexed couplings, and compares the retained terms and diagrams.
It uses the same parameter evaluator and kinematic gate as enumeration; those
parts are shared, not independent checks. This is diagram validation, **not
MadGraph integration or a decay-width calculation**. No numerical rate, NDA,
experimental status, or publication certification is implied.

## Portable installation

The source archive is exported with Toolbase. On the destination machine:

```sh
tb install heptapod-2.17.0.dev1.tar.gz --bundle processes --bundle process_validation
tb config set heptapod base_directory /absolute/path/to/workspace
tb config set heptapod mg5_path /absolute/path/to/MG5_aMC
```

For enumeration alone, omit `--bundle process_validation`. The source manifest
retains Python 3.12; this prototype was also exercised under Python 3.13 with
FeynGraph 0.1.0b4 and MG5 3.6.6. Native dependencies must be installed for the
destination architecture. The archive does not bundle Python, MG5, credentials,
virtual environments, or a benchmark UFO.

Alternatively, extract the archive, create a Python environment, install
`requirements.txt`, then `feyngraph==0.1.0b4` and `six`. The JSON worker can be run
without an agent harness:

```sh
python -m tools.feyngraph.catalogue_worker /absolute/path/to/run/request.json
```

Toolbase's tools provide path validation, timeout handling, and the outer artifact
manifest. The worker alone assumes a trusted request and existing output directory.

## Agent workflow

1. Inspect the model and choose the explicit scope and parameter point.
2. Call the enumeration tool once, with a fresh output directory.
3. Give the validator the catalogue path, same UFO/card, and independently recorded
   `expected_scope_json`. Do not derive the expected scope solely from an untrusted
   submission: otherwise a narrowed parent list could appear complete.
4. Pass individual process IDs and associated records to analysis workers.

Example enumeration runtime arguments, with configured workspace state:

```json
{
  "model": "inputs/model",
  "param_card": "inputs/param_card.dat",
  "output_dir": "runs/census",
  "max_candidates": 20000,
  "render": true
}
```

Omitting `parents` selects all physical particles. To constrain them, provide
model particle names such as `["Z"]`. Every output directory must be new; prior
receipts are never overwritten. Inputs and outputs must resolve inside the
configured workspace. MG5's installation path is separately configured.

## Artifact contract

- `request.json`, `response.json`, `stdout.log`, `stderr.log`: execution receipt.
- `model.json`: inspection result, including model source and fingerprint.
- `catalogue.json`: model, scope, coverage and process records.
- `exclusions.json`: candidate IDs and exclusion reasons.
- `diagrams/*.json`, `*.svg`, `*.tex`: graph records and native FeynGraph drawings.
- `validation.json`: independent comparison and per-parent MG generation coverage.
- `manifest.json`: SHA-256 of output files, excluding the manifest itself.

Process IDs combine a model-content digest, loop order, initial PDG and sorted
final PDGs. They are stable under moving the same UFO between machines. Changing
model source intentionally changes IDs; changing only the parameter card does
not. Parameter-point hashes remain separate. Graph IDs are local to the catalogue
revision. The built-in fingerprint is only its exposed structural inventory plus
FeynGraph version; explicit UFO fingerprints cover Python source files.

Vertex records retain ordered signed PDGs, names, color/Lorentz indices, Lorentz
expressions, coupling names and orders. Class representatives and drawings come
from the same enumerated graph. Rendering is native FeynGraph SVG/TikZ, not yet
FeynTable's custom styling. Spin/color fields in the model inventory are native
FeynGraph values, not a promised UFO encoding.

The validator checks process and graph identity, vertex terms, graph-file agreement,
and class partitions. It does not authenticate SVG/TikZ artwork or certify the
exclusion ledger; use the artifact manifest to detect file changes. Hashes are
integrity references, not signatures or evidence of scientific correctness.
Malformed submissions return errors rather than passing. Timeouts leave a failed
receipt; partial artifacts are not complete catalogues.

No FeynTable importer is included yet. This structured record is the proposed
handoff to the single-process analysis benchmark. General multibody topology,
mixed-order vertices, scattering, rate integration, and model-specific publication
adapters remain follow-up work. Legacy NDA conversion/ranking still has SM-specific
assumptions; use these raw catalogue tools for model-wide UFO enumeration.

## Tests

```sh
python -m pip install pytest
python -m pytest tests/process_catalogue -q
python tools/ufo/test_evaluate.py
python -m pytest tools/ufo/test_lorentz_and_cards.py -q
```

The optional full integration test uses `CATALOGUE_SM_UFO`, `CATALOGUE_SM_CARD`, and
`CATALOGUE_MG5`. Its 40-channel expectation applies specifically to the frozen
unrestricted SM/card used in the HEPTAPOD benchmark, not every SM implementation.
A synthetic non-SM scalar model tests model binding and portable IDs without those
external fixtures. See `PROCESS_CATALOGUE_TESTING.md` for the local test record.
