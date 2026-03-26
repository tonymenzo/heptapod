# GSoC 2026 Proposal: ML4DQM — Machine Learning for CMS Data Quality Monitoring

**Project:** ML4DQM — Integrating ML into CMS Data Quality Monitoring Workflows  
**Organisation:** ML4SCI (Machine Learning for Science)  
**Applicant:** Soumya Das | soumyadastopper2006@gmail.com | [GitHub](https://github.com/Soumya-Das-2006)  
**Repository:** [github.com/tonymenzo/heptapod](https://github.com/tonymenzo/heptapod)  
**Reference Paper:** [arxiv.org/abs/2512.15867](https://arxiv.org/abs/2512.15867)

---

## 1. Synopsis

I have already implemented a working ML4DQM pipeline — **38 unit tests passing**, a Transformer autoencoder added beyond task requirements, MCP server registration, and adaptive threshold management — as my evaluation task submission (PR #6). This proposal is not a plan to begin the work; it is a plan to take what I have built to production quality inside CMS operations.

CMS produces roughly **one billion collision events per second** at the LHC. Ensuring data quality in real time — catching detector faults, readout errors, and hardware degradation before they contaminate the physics dataset — is the mission of the **Data Quality Monitoring (DQM)** system. Today, DQM relies on human shifters who visually inspect thousands of 2-D histograms per run. The certification window is **under 60 seconds per run**. At the High-Luminosity LHC luminosity, this will be impossible to sustain manually.

This proposal describes an end-to-end ML integration layer for CMS DQM, built inside HEPTAPOD. It provides modular, agent-orchestrated tools for data access, model training, **adaptive real-time monitoring**, and alert dispatch — together with a Transformer architecture that captures structured anomalies that fully-connected models miss.

---

## 2. Motivation and Background

### 2.1 The scale of the problem

| CMS DQM metric | Value |
|---|---|
| Monitor elements per subsystem | ~3,000 |
| Subsystems requiring certification | 5 (Pixel, ECAL, Tracker, Muon, HCAL) |
| Total histograms per run | ~15,000 |
| Certification window | < 60 seconds |
| Consequence of missed bad run | Months of downstream analysis wasted |

No human can meaningfully inspect 15,000 histograms in 60 seconds. The current system relies on automated reference comparisons (DQM GUI) and shifter intuition. ML changes both.

### 2.2 Why HEPTAPOD?

HEPTAPOD's design principles — `RuntimeField`/`StateField` separation, schema-validated JSONL outputs, run-card configuration, and MCP compatibility — make it an ideal substrate:

- **Run cards** give shift crew a human-readable, reproducible configuration with zero Python required
- **MCP compatibility** means every DQM tool is callable from Claude Code or Claude Desktop with no boilerplate
- **Auditable execution traces** mean every alert and threshold update is logged and inspectable

### 2.3 Hidden challenges (what I learned building the evaluation task)

1. **Anomaly contamination during training**: If anomaly runs are included in the training set, the autoencoder learns to reconstruct them — defeating the detection. My preprocessor now marks and excludes synthetic anomalies, and the real implementation will use CMS certification labels for the same purpose.

2. **Threshold staleness**: A threshold computed once at training time becomes stale as detector conditions drift fill-by-fill. I built `DQMAdaptiveThresholdTool` (not in the task requirements) to address this with a rolling-window percentile update.

3. **Checkpoint format fragmentation**: The Transformer and autoencoder originally used different checkpoint schemas, causing `DQMRealtimeMonitorTool` to crash on Transformer outputs. I resolved this by adding `model_type` detection in both the monitor and evaluator.

4. **Compat shim shadowing**: The `_compat.py` shim was overriding a real `orchestral` pip install. Fixed with bundled-install detection.

---

## 3. Architecture

### 3.1 Folder Structure

```
heptapod/
├── tools/
│   └── dqm/
│       ├── data/
│       │   ├── cms_dqm_fetch.py        Tool 1: CMS DQM REST API + DQMIO reader
│       │   └── dqm_preprocessor.py    Tool 2: Normalise, clean, export
│       ├── training/
│       │   ├── autoencoder_train.py   Tool 3: FC autoencoder (fast baseline)
│       │   ├── transformer_train.py   Tool 4: Transformer AE (structured anomalies)
│       │   └── model_evaluator.py     Tool 5: Precision / Recall / F1
│       ├── deployment/
│       │   ├── realtime_monitor.py    Tool 6: Stream inference + alerts
│       │   ├── alert_dispatcher.py    Tool 7: Console / file / Mattermost webhook
│       │   └── adaptive_threshold.py  Tool 8: Rolling-window threshold update [NEW]
│       ├── dqm_mcp_tools.py           MCP registration for all 8 tools
│       ├── tests/
│       │   └── test_dqm_tools.py      38 unit tests, full pipeline coverage
│       └── README.md
├── run_cards/
│   └── dqm/
│       ├── dqm_training.yaml
│       └── dqm_monitoring.yaml
└── examples/
    ├── dqm_demo.py
    └── dqm_shifter_tutorial.ipynb     [NEW: zero-code shifter walkthrough]
```

### 3.2 Why two model architectures?

| Property | FC Autoencoder (Tool 3) | Transformer AE (Tool 4) |
|---|---|---|
| Inference latency | < 1 ms / run (CPU) | ~5 ms / run (CPU) |
| Best for | Global occupancy drops, overall gain shifts | Dead η–φ regions, partial HV trips, readout group failures |
| Parameters | ~4,000 (hidden=32, latent=8) | ~45,000 (d_model=32, 2 layers) |
| CMS latency budget | Yes | Yes |
| Checkpoint format | Unified `model_type` field | Unified `model_type` field |

Both share the same checkpoint schema so `DQMRealtimeMonitorTool` works with either without modification.

### 3.3 Adaptive threshold (Tool 8 — my addition beyond task requirements)

CMS detector conditions evolve fill-by-fill. A static threshold causes false positive rates to drift. `DQMAdaptiveThresholdTool` maintains a rolling window of the last N reconstruction errors and recomputes the threshold at the configured percentile every M runs. This is what a production DQM system requires:

```
Session 1:  threshold = 0.097 (training-time, 95th pct of 200 runs)
Fill 2 (+300 runs): threshold updated → 0.089 (beam conditions stable)
Fill 3 (+300 runs): threshold updated → 0.112 (HV trip on 3 channels)
Fill 4 (+300 runs): threshold updated → 0.094 (recovered)
```

### 3.4 Data flow

```
CMS DQM REST API / DQMIO (ROOT)
          │
          ▼
CMSDQMFetchTool ──────────────────────────► dqm_raw.jsonl
          │
          ▼
DQMPreprocessorTool ──────────────────────► dqm_processed.npy + dqm_meta.json
          │
    ┌─────┴──────┐
    ▼            ▼
AE Train    Transformer Train ──────────► dqm_autoencoder.pt / dqm_transformer.pt
    └─────┬──────┘
          ▼
DQMModelEvaluatorTool ────────────────────► dqm_eval_report.json (P/R/F1)
          │
          ▼
DQMRealtimeMonitorTool ───────────────────► dqm_alerts.jsonl
          │                                 dqm_monitor_log.json
          ├──────────────────────────────►  DQMAdaptiveThresholdTool (every 50 runs)
          ▼
DQMAlertDispatcherTool ───────────────────► console / file / Mattermost
```

---

## 4. Technical Design Decisions

### 4.1 Unsupervised anomaly detection strategy

CMS DQM data is **mostly good** — bad runs are rare events (typically < 5% of fills). Reconstruction-based anomaly detection is the natural fit: train exclusively on good runs, flag runs that cannot be reconstructed accurately.

Threshold strategy: percentile-based rather than mean + k×σ. With only a few hundred training runs, σ is poorly estimated and mean + 3σ can be brittle. The 95th percentile of training reconstruction errors is more robust and has a direct operational interpretation: "flags runs worse than 95% of what we trained on."

### 4.2 Graceful offline fallback

The CMS DQM REST API requires CERN network access. `CMSDQMFetchTool` detects connectivity failures and falls back to **deterministic synthetic histograms** (seeded by run number, with injected anomalies every 10th run). This means:
- All 38 tests run in CI without CERN credentials or GPU
- Developers can iterate locally
- Synthetic anomaly injection provides known ground truth for threshold debugging

### 4.3 CMS-compatible alert schema

```json
{
  "timestamp": "2026-03-25T08:00:00Z",
  "run_id": 360020,
  "subsystem": "Pixel",
  "severity": "CRITICAL",
  "recon_error": 0.085,
  "threshold": 0.021,
  "action": "Immediately notify shift crew and flag run for exclusion"
}
```

`DQMAlertDispatcherTool` accepts a `webhook_url` pointing directly to the CMS Mattermost incoming webhook endpoint.

### 4.4 Agent-driven orchestration

Via the HEPTAPOD MCP server, the full pipeline is accessible in natural language:

```
Shifter: "Retrain the Pixel model on runs 360500–361000 and alert me if anomaly rate > 5%"

Agent → dqm_fetch(subsystem="Pixel", run_start=360500, run_end=361000)
      → dqm_preprocess(...)
      → dqm_train_autoencoder(epochs=50)
      → dqm_monitor(...)
      → if anomaly_rate_pct > 5.0: dqm_dispatch_alerts(channels="webhook")
```

Zero Python from the shifter. Full auditability.

---

## 5. Testing Strategy

**38 unit tests covering all tools** — run without CERN credentials or GPU in under 90 seconds:

```bash
pytest tools/dqm/tests/test_dqm_tools.py -v
# 38 passed, 5 subtests passed
```

| Test class | Tests | What is covered |
|---|---|---|
| `TestCMSDQMFetchTool` | 7 | Status, JSONL output, record count, anomaly injection (bin 32 ×50), error handling, all subsystems |
| `TestDQMPreprocessorTool` | 6 | Status, file creation, shape (N×64), min=0 / max=1 normalisation, metadata keys, missing-file error |
| `TestDQMAutoencoderTrainTool` | 6 | Status, model file, training log, threshold > 0, loss improvement, missing-file error |
| `TestDQMModelEvaluatorTool` | 5 | Status, report creation, supervised P/R/F1, evaluated count, flagged ⊆ evaluated |
| `TestDQMRealtimeMonitorTool` | 7 | Status, output files, anomalies detected, alert fields, severity validity, CRITICAL > 2×threshold, missing model error |
| `TestDQMAlertDispatcherTool` | 6 | Console+file dispatch, file created, severity filter, empty alerts, missing file error, report text |
| `TestEndToEndPipeline` | 1 | Full 6-stage pipeline, all 9 expected output files present |

---

## 6. Timeline (May – August 2026, 175 hours)

### Community bonding (May 1–26)
- Obtain CERN account and access to CMS DQM data (apply immediately on acceptance)
- Study DQMIO format and `dasgoclient` + `uproot` for ROOT file reading
- Meet with mentors; agree on Pixel vs ECAL as priority subsystem for benchmarking
- Study CMS Run 3 bad-run certification list for evaluation ground truth

### Week 1–2 (May 27 – June 9): Real data integration
- **Deliverable:** `CMSDQMFetchTool` reads real DQMIO ROOT files via `uproot` (not just REST API)
- **Deliverable:** Latency benchmark: REST API vs DQMIO vs synthetic fallback, documented in README
- **Test:** Run fetch on 1,000 real Pixel runs; verify JSONL schema matches synthetic output

### Week 3–4 (June 10–23): Transformer validation on real data
- **Deliverable:** Transformer trained and evaluated on real CMS Pixel data (runs 360000–361000)
- **Deliverable:** Precision/Recall/F1 report comparing autoencoder vs Transformer on CMS bad-run list
- **Deliverable:** Checkpoint format verified: both models work interchangeably in monitor and evaluator

### Week 5–6 (June 24 – July 7): Adaptive threshold + production hardening
- **Deliverable:** `DQMAdaptiveThresholdTool` integrated into `dqm_run_full_pipeline` MCP shortcut
- **Deliverable:** End-to-end test: 3 simulated fills (300 runs each) with rolling threshold updates
- **Deliverable:** Mattermost webhook integration tested against CMS DQM test channel

### Week 7 (July 8–14): Midterm evaluation
- Submit midterm report
- Code review with mentors on Weeks 1–6 deliverables
- Address feedback

### Week 8–9 (July 15–28): Additional subsystems + few-shot classifier
- **Deliverable:** ECAL and Tracker subsystems added to fetch/preprocess pipeline
- **Deliverable:** Few-shot classifier on frozen Transformer encoder (for labelled bad runs from certification DB)
- **Deliverable:** `DQMModelEvaluatorTool` extended to evaluate few-shot classifier separately from AE

### Week 10–11 (July 29 – August 11): Documentation + shifter notebook
- **Deliverable:** `examples/dqm_shifter_tutorial.ipynb` — zero-Python walkthrough for shift crew
- **Deliverable:** Comprehensive tool documentation with real CMS examples
- **Deliverable:** Integration into main HEPTAPOD orchestral demo

### Week 12 (August 12–18): Final polish
- Address all mentor feedback
- Final code review and cleanup
- PR preparation for merge to main HEPTAPOD repo

### Final evaluation (August 25)
- Submit PR to main HEPTAPOD repo
- Write final project report
- Blog post for ML4SCI community

---

## 7. Risk Mitigation

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| CERN network access delayed | Medium | High | Synthetic fallback already implemented and tested. All 38 tests run without CERN credentials. |
| DQMIO/ROOT format complexity | Medium | Medium | `uproot` reads ROOT files in pure Python. Week 1 latency budget includes ROOT debugging time. |
| Transformer training latency on CPU | Low | Low | Architecture is lightweight (d_model=32, 2 layers, ~45K params). Inference < 5ms/run on CPU. |
| CMS DQM API changes between now and GSoC | Low | Medium | DQMIO format is stable (ROOT-based). REST API is a secondary access path with version pinning. |
| Thesis/exam conflict | Low | High | IMCA 5th semester exams expected May 2026. Community bonding period used for setup; no coding deliverables at risk. |

---

## 8. About Me

I am a 4th-semester IMCA student at Parul University (CGPA 8.54/10) currently interning as an AI & IoT Innovation Intern at Tinkering Hub, Parul University.

**Why I am the right person for this project:**

I deployed quantized CNN models (FP16 + TFLite) on NVIDIA Jetson Orin Nano via CUDA for real-time agricultural diagnosis. The constraint I worked under — detect anomalies in a continuous sensor stream with latency under 200ms — is structurally identical to CMS DQM's 60-second run certification window. The histogram arrays are different (crop images vs particle collision distributions); the ML approach is the same.

Specifically relevant to this project:
- Diagnosed a high time-complexity bottleneck in a production AI system and rewrote core inference logic (800ms → under 200ms, ~40% efficiency gain) — directly analogous to DQM latency requirements
- Led 4 hackathons as team leader: planning, task allocation, milestone tracking, and delivery under tight deadlines — relevant to a 175-hour structured GSoC programme
- Presented research at ICPAT 2025 international conference — I can write technical documentation and explain systems clearly

**What I have already built for this project:**
- Fully-connected autoencoder (Tool 3) with unsupervised anomaly detection
- Transformer autoencoder (Tool 4) with multi-head self-attention — my own addition beyond task requirements
- Adaptive threshold tool (Tool 8) — not required, added because real DQM needs it
- 38 unit tests covering all tools end-to-end
- MCP registration so all tools are immediately usable from Claude Code
- Responded to all 20 GitHub Copilot review comments on PR #6

I can commit 15–20 hours per week. I have no conflicting internship or travel during May–August 2026.

---

## 9. References

1. Menzo et al., "HEPTAPOD: High-Energy Physics Toolkit for Agentic Planning, Orchestration, and Deployment," arXiv:2512.15867 (2024)
2. CMS Collaboration, "CMS Data Quality Monitoring," CMS NOTE-2006/028
3. Vaswani et al., "Attention Is All You Need," NeurIPS 2017
4. Cerminara et al., "CMS DQM — Architecture and Performance," J. Phys.: Conf. Ser. 898 (2017)
5. Pol et al., "Anomaly Detection with Conditional Variational Autoencoders," IEEE SSCI 2019
6. Roman & Roman, "Orchestral AI: A Framework for Agent Orchestration," arXiv:2601.02577 (2026)
