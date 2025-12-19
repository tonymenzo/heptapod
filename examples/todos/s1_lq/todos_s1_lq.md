# HEP BSM Event Generation Pipeline

**CRITICAL:** All card files (.fr, .mg5, .cmnd) are PRE-WRITTEN. Pass paths as tool arguments - tools inject placeholders automatically.

**IMPORTANT:** ALWAYS use RELATIVE paths within the sandbox. Do NOT use absolute paths as they will cause errors. All paths should be relative to the sandbox root directory.

**TODO LIST RULES:**

- DO NOT modify task descriptions or instructions before completing them
- ONLY check off items (change `[ ]` to `[x]`) ONE AT A TIME after full completion
- USE THE TOOLS recommended by each item, if present
- Read the exact task as written, complete it fully, then mark it done
- Before moving to the next task, SUMMARIZE the previous task, PLAN your next step carefully, and inform the user about what you will do next.

---

## Phase 1: FeynRules → UFO

- [ ] **Generate UFO from .fr model**
  - Locate: `feynrules/models/S1_LQ_RR.fr` (ALREADY EXISTS)
  - Call `FeynRulesToUFOTool(model_path="feynrules/models/S1_LQ_RR.fr", output_dir="feynrules/S1_LQ_RR_UFO")`
  - Verify: check output JSON `"ok": true`, confirm UFO files exist

---

## Phase 2: MadGraph Event Generation

- [ ] **Run MadGraph with .mg5 card**
  - Locate: `madgraph/cards/signal/LQ_S1_pp_ljlj.mg5` (ALREADY EXISTS)
  - Get UFO path from Phase 1 (use RELATIVE path, e.g., "feynrules/S1_LQ_RR_UFO")
  - Call `MadGraphFromRunCardTool(command_card="madgraph/cards/S1_LQ_RR_pp_lqlq_scan.mg5", data_dir="data/mg_run001", ufo_path=<relative_ufo_path>, nevents=5000, seed=12345)`
  - **Scan handling:** If `scan_detected: true`, note `runs[i].lhe_file` and `runs[i].scan_params` for each mass point
  - Output: LHE file(s) at `runs[i].lhe_file`

---

## Phase 3: Pythia Showering

- [ ] **Shower LHE with Pythia (FOR EACH scan point if scan)**
  - Locate: `pythia/cards/pp_S1S1d_ljlj.cmnd` (ALREADY EXISTS)
  - Get LHE path(s) from Phase 2: `runs[i].lhe_file`
  - For each LHE file, call `PythiaFromRunCardTool(cmnd_path="pythia/cards/S1_LQ_RR_pp_ljlj.cmnd", data_dir="data/pythia_m<mass>", lhe_path=<lhe_file>, n_events=5000, shower_lhe=True, finals_only=True, full_history=False, seed=<unique_seed>)`
  - Track: which `events_jsonl` corresponds to which scan mass point
  - Output: `events.jsonl` per mass point
  - Use `EventJSONLToNumpyTool` to convert particle data (FOR EACH mass point) for future processing:
    - Input: `data/pythia_m<mass>/events.jsonl` from Phase 3
    - Output: `data/pythia_m<mass>/events.npy`

---

## Phase 4: Jet Clustering

- [ ] **Cluster jets (FOR EACH mass point)**
  - Get `events_jsonl` from Phase 3
  - Call `JetClusterSlowJetTool(jsonl_path=<events_jsonl>, cluster_all=True, output_path="data/jets_m<mass>.jsonl", algorithm="antikt", R=0.4, ptmin=50.0, etamax=2.8)`
  - Cluster **all** jets `cluster_all=True`, **DO NOT** use `event_index` argument.
  - Output: clustered jets JSONL per mass point
  - Use `JetsJSONLToNumpyTool` to convert jet data (FOR EACH mass point) for future processing:
    - Input: `data/jets_m<mass>.jsonl`
    - Output: `data/jets_m<mass>.npy`

---

## Phase 5: Leptoquark Reconstruction

- [ ] **Reconstruct leptoquark masses using InvariantMassTool (FOR EACH mass point)**
  - Get truth particles: `data/pythia_m<mass>/events.jsonl` from Phase 3
  - Call `InvariantMassTool(events_path="data/pythia_m<mass>/events.jsonl", template="two_body_symmetric", object_type="truth_particles", n_objects=4, pdg_filter=[11,-11,1,2,3,4,5], output_prefix="analysis/lq_m<mass>", hist_bins=50)`
  - Track: mass point corresponds to `scan_params` from Phase 2
  - Output: m1, m2, m_min, m_max arrays + histograms per mass point

- [ ] **Create comparison plot across all mass points**
  - Load all `analysis/lq_m<mass>_m1.npy` files from previous step
  - Use matplotlib to create comparison plot overlaying distributions for all mass points
  - Save to `analysis/lq_mass_comparison.pdf` with xlabel="m(lj) [GeV]", ylabel="Events", normalized
  - Verify: peaks appear near expected leptoquark masses

- [ ] **Apply kinematic cuts and reanalyze (FOR EACH mass point)**
  - Apply cuts: `ApplyCutsTool(input_path="data/pythia_m<mass>/events.jsonl", output_path="analysis/events_cuts_m<mass>.jsonl", pt_min=50.0, eta_max=2.5, pdgids=[11,-11,1,2,3,4,5])`
  - Rerun analysis: `InvariantMassTool(events_path="analysis/events_cuts_m<mass>.jsonl", template="two_body_symmetric", object_type="truth_particles", n_objects=4, output_prefix="analysis/lq_cuts_m<mass>", hist_bins=50)`
  - Plot comparison: create overlay of cut vs no-cut distributions → `analysis/lq_mass_w_cuts.pdf`