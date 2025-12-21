# **HEPTAPOD**

---
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
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
│   ├── feynrules/               # FeynRules → UFO model generation
│   ├── mg5/                     # MadGraph parton-level event generation
│   ├── pythia/                  # Pythia hadronization and showering
│   ├── sherpa/                  # Sherpa event generation and ufo conversion
│   └── analysis/                # Data conversion and kinematics tools
├── llm/                         # LLM utilities and Ollama integration
│   ├── utils.py                 # Helper functions (get_ollama, etc.)
│   └── test_ollama_*.py         # Ollama integration tests
├── examples/                    # Example workflows and demos
│   ├── hep_bsm_demo.py          # Main demo application
│   └── todos/                   # Example task lists
├── prompts/                     # System prompts for agent orchestration
├── config.py                    # Configuration (Ollama + external tool paths)
├── test_runner.py               # Master test runner
└── requirements.txt             # Python dependencies
```

---

## Installation

### Prerequisites

**Required:**

- **Python 3.12 or 3.13** (3.14+ not supported for some dependencies)
- **At least one LLM provider:**
  - **Cloud LLMs**: Anthropic Claude, OpenAI GPT, Google Gemini, or Groq (requires API key)
  - **Local LLMs**: Ollama (free, runs locally, no API key needed)

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

**3. Configure LLM Provider**

You have two options for LLM access:

**Option A: Cloud LLMs (requires API key)**

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

The template includes placeholders for all supported cloud providers:

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

**Option B: Local Ollama (free, no API key needed)**

If Ollama is not already installed/running:

1. Download from [ollama.com](https://ollama.com/download)
2. Start the server: `ollama serve` (or use the macOS app)
3. Pull a model: `ollama pull gpt-oss:20b`

Configure in `config.py` (at the top of the file):

```python
# Ollama LLM Configuration
ollama_host = None              # Use local Ollama (default port 11434)
ollama_model = "gpt-oss:20b"    # Your preferred model

# For remote Ollama server:
# ollama_host = "http://SERVER_IP:11434"
```

Test Ollama integration:

```bash
python test_runner.py --only llm
```

**4. Verify Installation**

```bash
python test_runner.py --only prereqs
```

This checks:
- Python version (3.12 or 3.13)
- `orchestral-ai` installation
- LLM availability (API keys OR Ollama)
- Project structure

**Note:** You need at least one working LLM (either API keys in `.env` OR Ollama running) to pass prerequisites.

### External Dependencies

#### FeynRules and Mathematica

**Required for model generation tools only. Skip if using pre-generated UFO models.**

1. **Mathematica** (currently FeynRules supports versions 13.3 or earlier)
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
                  ...

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
For a comprehensive test of all supported **HEPTAPOD** functionalities run:
```bash
# Run all tests
python test_runner.py
```

otherwise, subsets of features can be tested with the relevant `--only` flag:

```bash
# Skip slow integration tests (MG5, Pythia, Sherpa generation)
python test_runner.py --skip-slow

# Run only specific components
python test_runner.py --only prereqs
python test_runner.py --only llm
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

**Configure the demo** by editing `examples/hep_bsm_demo.py`:

1. **Select your LLM** (lines 117-147):
   ```python
   # ===== Cloud LLM Providers (requires API key in .env) =====
   # Option 1: OpenAI GPT (default)
   LLM = GPT()

   # Option 2: Anthropic Claude
   #LLM = Claude()

   # Option 3: Google Gemini
   #LLM = Gemini()

   # Option 4: Groq
   #LLM = Groq()

   # ===== Local/Remote Ollama (configured in config.py) =====
   # Option 5: Ollama (uses config.py settings)
   #LLM = get_ollama()

   # Option 8: Ollama with reasoning mode
   #LLM = get_reasoning_ollama()
   ```

2. **Choose an operating mode**:
   - **`explorer`** - Interactive exploration and analysis (recommended for first run)
   - **`plan`** - Agent creates its own execution plan
   - **`todo`** - Uses predefined task list from `todos.md`

   Each mode has a pre-defined default system prompt that can be found/modified in `prompts/`.

3. **Set configuration variables**:
   ```python
   CREATE_NEW_SANDBOX = True
   MODE = "explorer"  # or "plan" or "todo"
   ```

**Run the demo:**

```bash
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

```
What tools are available for HEP workflows?
```

The agent has access to:
- **Model generation**: FeynRulesToUFOTool (FeynRules → UFO)
- **Parton-level events**: MadGraphFromRunCardTool
- **Hadronization**: PythiaFromRunCardTool, JetClusterSlowJetTool
- **Parton-level or particle-level events**: SherpaFromRunCardTool
- **Analysis**: Kinematics tools, reconstruction, cuts, filtering
- **Data conversion**: LHE → JSONL → NumPy

along with default utility tools provided by Orchestral such as `ReadFile`, `WriteFile`, `RunCommand`, `RunPython`, `WebSearch`, etc.

**3. Start with a simple task**

Begin with UFO model generation:
```
"Generate the UFO model files from the S1 leptoquark FeynRules model in feynrules/models/S1_LQ_RR.fr"
```

For detailed tool documentation and API reference, see [tools/README.md](tools/README.md).

---

## Contributing tools

HEPTAPOD is designed to be extended with custom tools. If you'd like to contribute a new tool for model generation, event simulation, analysis, or any other physics workflow:

**See [CONTRIBUTING.md](CONTRIBUTING.md) for comprehensive guidelines on:**

- Tool architecture and structure
- Required components (RuntimeFields, StateFields, error handling)
- Path safety and sandboxing requirements
- Testing and integration
- Best practices and examples

### Other Contributions

For bug reports, feature requests, or technical discussions:
- **GitHub Issues**: [https://github.com/tonymenzo/heptapod/issues](https://github.com/tonymenzo/heptapod/issues)

---

## Citation

If you use HEPTAPOD in your research, please cite:

```bibtex
@article{Menzo:2025cim,
    author = {Menzo, Tony and Roman, Alexander and Gleyzer, Sergei and Matchev, Konstantin and Fleming, George T. and H{\"o}che, Stefan and Mrenna, Stephen and Shyamsundar, Prasanth},
    title = "{HEPTAPOD: Orchestrating High Energy Physics Workflows Towards Autonomous Agency}",
    eprint = "2512.15867",
    archivePrefix = "arXiv",
    primaryClass = "hep-ph",
    reportNumber = "FERMILAB-PUB-25-0923-CSAID-ETD-T",
    month = "12",
    year = "2025"
}
```

---

## License

This project is licensed under the GPL-3.0 license - see the [LICENSE](LICENSE) file for details.

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
