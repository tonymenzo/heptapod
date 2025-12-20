# **HEPTAPOD**

---
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.12%20|%203.13-blue.svg)](https://www.python.org/downloads/)
[![Framework](https://img.shields.io/badge/Framework-Orchestral--AI-green.svg)](https://orchestral-ai.com)

## Overview

**HEPTAPOD** (High-Energy Physics Toolkit for Agentic Planning, Orchestration, and Deployment) is a Python framework for **orchestrating end-to-end HEP simulation and analysis workflows** using large language model (LLM) agents.

Built on the [Orchestral AI](https://orchestral-ai.com) engine, HEPTAPOD treats established HEP tools (**FeynRules, MadGraph, Pythia and Sherpa**) as **schema-validated, auditable operations** that can be coordinated by an LLM under explicit human supervision. Rather than replacing existing workflows, HEPTAPOD provides a structured orchestration layer that automates bookkeeping, parameter propagation, and multi-stage execution while preserving transparency and reproducibility.

In practice, HEPTAPOD currently enables researchers to:

- Define workflows at the level of **physics intent**, not scripts
- Execute **multi-stage BSM pipelines** (model → events → analysis) with consistent metadata
- Automatically handle **parameter scans**, intermediate artifacts, and failure recovery
- Maintain **fully reproducible, auditable execution traces** via run cards and structured outputs

The design and philosophy of HEPTAPOD are described in detail in the accompanying paper [https://arxiv.org/abs/2512.15867](https://arxiv.org/abs/2512.15867).

---

## Key Features

- **General-purpose orchestration for HEP workflows** across theory, simulation, and analysis
- **Agent-driven planning and execution**, with explicit human oversight
- **Schema-validated tool interfaces** that formalize interactions with HEP software
- **Run-card–based configuration** as a stable, auditable orchestration boundary
- **Automatic metadata and state propagation** across multi-stage workflows
- **Structured error handling and recovery** for long-running or branching executions
- **LLM-compatible intermediate data formats** for inspection, validation, and debugging

---

## Directory Structure

```bash
heptapod/
├── tools/                       # Physics tools for event generation and analysis
│   ├── __init__.py
│   ├── README.md                # Tool documentation
│   ├── feynrules/               # FeynRules → UFO model generation
│   │   ├── __init__.py
│   │   ├── feynrules.py
│   │   ├── UFO_generator.wl
│   │   ├── UFO_generator.nb
│   │   └── test_feynrules.py
│   ├── mg5/                     # MadGraph parton-level event generation
│   │   ├── __init__.py
│   │   ├── mg5.py
│   │   └── test_mg5.py
│   ├── pythia/                  # Pythia hadronization and showering
│   │   ├── __init__.py
│   │   ├── pythia.py
│   │   └── test_pythia.py
│   ├── sherpa/                  # Sherpa event generation and ufo conversion
│   │   ├── __init__.py
│   │   ├── sherpa.py
│   │   ├── generate_model.py
│   │   └── test_sherpa.py
│   └── analysis/                # Data conversion and analysis utilities
│       ├── conversions.py
│       ├── kinematics.py
│       ├── kinematics_example.py
│       ├── reconstruction.py
│       ├── test_conversions.py
│       ├── test_kinematics.py
│       ├── test_reconstruction.py
│       └── test_delta_r_filter.py
├── examples/                    # Example workflows and demo applications
│   ├── hep_bsm_demo.py          # Full BSM workflow demo
│   ├── sandbox_utils.py         # Sandbox management utilities
│   ├── hep_bsm_sandbox/         # Working directories (created at runtime)
│   └── todos/                   # Example task lists
│       └── s1_lq/
│           └── todos_s1_lq.md
├── prompts/                     # System prompts for LLM orchestration
│   ├── __init__.py
│   └── demos/
│       └── hep_bsm/
│           └── system/
│               ├── hep_bsm_evt_gen_explorer_prompt.md
│               ├── hep_bsm_evt_gen_plan_prompt.md
│               └── hep_bsm_evt_gen_todo_prompt.md
├── config.py                    # Configuration for external tool paths
├── test_runner.py               # Master test runner
├── requirements.txt             # Python dependencies (pip)
└── environment.yml              # Conda environment specification
```

---

## Installation

### Prerequisites

**Required:**
- **Python 3.12 or 3.13** (3.14+ not supported due to pythia8mc)
- **At least one LLM provider** (Anthropic Claude, OpenAI GPT, Google Gemini, Groq, or local Ollama)

### Quick Start

**1. Clone the Repository**

```bash
git clone https://github.com/tonymenzo/heptapod.git
cd heptapod
```

**2. Install Dependencies**

Choose one of the following methods:

**Using pip**
```bash
pip install -r requirements.txt
```

**Using venv**
```bash
python -m venv heptapod-env
source heptapod-env/bin/activate  # On Windows: heptapod-env\Scripts\activate
pip install -r requirements.txt
```

**Using conda**
```bash
conda env create -f environment.yml
conda activate heptapod
```

**3. Configure API Keys**

A `.env` template file is included in the repository. Edit it to add your API keys:

```bash
# Edit the .env file with your preferred editor
nano .env
# or
code .env
# or 
vim .env
# or 
nvim .env
# or 
emacs .env
```

The template includes placeholders for all supported LLM providers:

```bash
# Anthropic (Claude) - https://console.anthropic.com/
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# OpenAI (GPT) - https://platform.openai.com/api-keys
OPENAI_API_KEY=your_openai_key_here

# Google (Gemini) - https://aistudio.google.com/app/apikey
GOOGLE_API_KEY=your_google_api_key_here

# Groq - https://console.groq.com/
GROQ_API_KEY=your_groq_api_key_here

# Note: You only need to set API keys for the providers you plan to use
```

**Alternatively**, use local models (no API key needed):
```bash
# Add to .env for local Ollama models
OLLAMA_ENDPOINT=http://localhost:11434
```

**4. Verify Installation**

```bash
python test_runner.py --only prereqs
```

This checks:
- Python version (3.12 or 3.13)
- `orchestral-ai` installation
- API keys configuration
- Project structure

### External Dependencies

#### FeynRules and Mathematica

**Required for model generation tools only. Skip if using pre-generated UFO models.**

1. **Mathematica** (version 13.3 or earlier)
   - Download from [Wolfram Research](https://www.wolfram.com/mathematica/)
   - Requires valid license
   - WolframScript included with Mathematica installation

2. **Authenticate WolframScript:**
   ```bash
   wolframscript -authenticate
   # Enter your Wolfram credentials when prompted
   ```

   For details on WolframScript usage, environment variables, and advanced options, see the [WolframScript documentation](https://reference.wolfram.com/language/ref/program/wolframscript.html).

3. **FeynRules** (version 2.3.49 recommended)
   - Download from [FeynRules website](https://feynrules.irmp.ucl.ac.be/)
   - Extract to a permanent location (e.g., `/path/to/FeynRules_v2.3.49`)

#### MadGraph5_aMC@NLO

**Required for parton-level event generation.**

1. Download from [MadGraph Launchpad](https://launchpad.net/mg5amcnlo)
   ```bash
   wget https://launchpad.net/mg5amcnlo/3.0/3.6.x/+download/MG5_aMC_v3.6.6.tar.gz
   tar -xzf MG5_aMC_v3.6.6.tar.gz
   ```

2. ***Optionally*** install additional features (PDF sets, NLO packages)

#### Pythia8

**Required for hadronization and showering.**

The `pythia8mc` Python package (installed via pip above) includes Pythia8 binaries. No separate installation needed.

**Verify installation:**
```bash
python -c "import pythia8; print(pythia8.__version__)"
```

#### Sherpa3

**Required for event generation.**

The `sherpa-mc` Python package (installed via pip above) includes Sherpa3 binaries. No separate installation needed.

**Verify installation:**
```bash
python -c "import Sherpa"
```

### Configuration

**Optional:** If using external physics tools, edit `config.py` to point to your installations:

```python
# FeynRules (for UFO model generation)
feynrules_path = "/path/to/FeynRules_v2.3.49"
wolframscript_path = "/usr/local/bin/wolframscript"

# MadGraph5_aMC (for parton-level event generation)
mg5_path = "/path/to/MG5_aMC_v3.6.6"
```

**Note:** Only needed if using FeynRules or MadGraph. Skip if working with pre-generated events.

---

## Testing

### Test Runner Options

View all available test options:

```bash
python test_runner.py --help
```

### Common Test Commands

```bash
# Run all tests
python test_runner.py

# Run with verbose output
python test_runner.py --verbose

# Skip slow integration tests (MG5, Pythia, Sherpa)
python test_runner.py --skip-slow

# Run only specific components
python test_runner.py --only prereqs
python test_runner.py --only conversions
python test_runner.py --only kinematics
python test_runner.py --only reconstruction
python test_runner.py --only delta_r_filter
python test_runner.py --only feynrules
python test_runner.py --only mg5
python test_runner.py --only pythia
python test_runner.py --only sherpa
```

---

## Usage

### Quick Start

The fastest way to get started is to run the demo with the web UI. This lets you describe physics goals in natural language, watch the agent execute multi-step workflows, and see real-time tool execution.

**Choose an operating mode** by editing `examples/hep_bsm_demo.py`:

- **`explorer`** - Interactive exploration and analysis (recommended for first run)
- **`plan`** - Agent creates its own execution plan
- **`todo`** - Uses predefined task list from `todos.md`

```bash
# Edit examples/hep_bsm_demo.py to set:
# CREATE_NEW_SANDBOX = True
# MODE = "explorer"  # or "plan" or "todo"

python examples/hep_bsm_demo.py
```

The demo will create a numbered sandbox directory (e.g., `sandbox001`), copy template files, launch the web server at `http://127.0.0.1:8000`, and open your browser automatically.

### Getting Started with the Demo

Once the web UI launches, you can interact with the agent in natural language. Here's a suggested workflow to get familiar with the system:

**1. Explore the sandbox environment**

Start by asking the agent to show you what's available:
```
"List the files in the current directory and summarize what's here."
```

The sandbox contains:
- `feynrules/models/` - FeynRules model files (e.g., `S1_LQ_RR.fr` for leptoquark model)
- `mg5/` - MadGraph configuration templates
- `pythia/` - Pythia run card templates
- `sherpa/` - Sherpa run card templates

**2. Check available tools**

Ask the agent about its capabilities:
```
"What tools do you have available for HEP workflows?"
```

The agent has access to:
- **Model generation**: FeynRulesToUFOTool (FeynRules → UFO)
- **Parton-level events**: MadGraphFromRunCardTool
- **Hadronization**: PythiaFromRunCardTool, JetClusterSlowJetTool
- **Parton-level or particle-level events**: SherpaFromRunCardTool
- **Analysis**: Kinematics tools, reconstruction, cuts, filtering
- **Data conversion**: LHE → JSONL → NumPy

**3. Start with a simple task**

Begin with UFO model generation:
```
"Generate the UFO model files from the S1 leptoquark FeynRules model in feynrules/models/S1_LQ_RR.fr"
```

For detailed tool documentation and API reference, see [tools/README.md](tools/README.md).

---

## Citation

If you use HEPTAPOD in your research, please cite:

```bibtex
@misc{heptapod2025,
  title={HEPTAPOD: Orchestrating High-Energy-Workflows Towards Autonomous Agency},
  author={Tony Menzo, Alexander Roman, Sergei Gleyzer, Konstantin Matchev, George T. Fleming, Stefan H\"{o}che, Stephen Mrenna, Prasanth Shyamsundar},
  year={2025},
  eprint="2512.15867"
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Contact

**Maintainers:**
- Tony Menzo - amenzo@ua.edu

**Issues and Support:**
- GitHub Issues: [https://github.com/tonymenzo/heptapod/issues](https://github.com/tonymenzo/issues)

**Project Links:**
- Repository: [https://github.com/tonymenzo/heptapod](https://github.com/tonymenzo/heptapod)
- Research Paper: [arXiv:2512.15867](https://arxiv.org/abs/2512.15867)

---

**Version**: 1.0.0
**Status**: Active Development
