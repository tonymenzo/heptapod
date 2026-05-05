"""
# hep_bsm_demo.py is a part of the HEPTAPOD package.
# Copyright (C) 2025 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.
"""

# Setup repository path for imports
import sys
from pathlib import Path

# Add repository root to path for local imports (prompts, tools, etc.)
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Add shared utilities directory to path
SHARED_DIR = Path(__file__).resolve().parent.parent / 'shared'
sys.path.insert(0, str(SHARED_DIR))

# Load .env so tools that read API keys at construction time (e.g.
# WebSearchTool reading OPENAI_API_KEY) see them.
from dotenv import load_dotenv
load_dotenv(REPO_ROOT / ".env")

# =========================================================== #
# ======================== IMPORTS ========================== #
# =========================================================== #

# Orchestral imports.
from orchestral import Agent
from orchestral.tools import (RunCommandTool, DummyRunCommandTool, WebSearchTool, RunPythonTool,
                              WriteFileTool, ReadFileTool, EditFileTool, FileSearchTool, FindFilesTool,
                              TodoWrite, TodoRead)
from orchestral.tools.hooks import TruncateOutputHook, DangerousCommandHook, SafeguardHook, UserApprovalHook

# LLM imports.
from orchestral.llm import GPT, Claude, Gemini, Groq
from llm import get_ollama, get_reasoning_ollama, get_vllm, get_litellm

# Prompt and tool imports.
from prompts import HEP_BSM_EVT_GEN_EXPLORER_PROMPT
from tools.analysis.conversions import EventJSONLToNumpyTool, JetsJSONLToNumpyTool
from tools.analysis.kinematics import (
    CalculateInvariantMassTool,
    CalculateTransverseMomentumTool,
    CalculateDeltaRTool,
    ApplyCutsTool,
    GetHardestNTool,
    GetHardestNJetsTool,
    FilterByPDGIDTool,
    SortByPtTool,
    FilterByDeltaRTool
)
from tools.analysis.reconstruction import ResonanceReconstructionTool
from tools.feynrules import FeynRulesToUFOTool
from tools.mg5 import MadGraphFromRunCardTool
from tools.pythia import PythiaFromRunCardTool, JetClusterSlowJetTool
from tools.sherpa import SherpaFromRunCardTool

# Configuration imports.
from config import feynrules_path, mg5_path, wolframscript_path

print("Using FeynRules path:", feynrules_path)
print("Using MG5 path:", mg5_path)
print("Using wolframscript path:", wolframscript_path)

# Import the Orchestral app server (qualified path; bare `app` is not
# top-level on the installed orchestral package).
import orchestral.ui.app.server as app_server

# Import sandbox utilities
from sandbox_utils import create_new_sandbox

# Configure workspace - either use existing or create new sandbox.
demo_files_dir = REPO_ROOT / 'examples' / 'hep_bsm_sandbox'

CREATE_NEW_SANDBOX = True  # Set to True to create a new sandbox, False to use existing
MODE = "explorer"          # Options: "todo", "plan", "explorer"

if CREATE_NEW_SANDBOX:
    base_directory, system_prompt = create_new_sandbox(demo_files_dir, mode=MODE)
else:
    # When using existing sandbox, manually specify the prompt
    base_directory = str(demo_files_dir / 'sandbox000')
    system_prompt = HEP_BSM_EVT_GEN_EXPLORER_PROMPT  # Or use TODO/PLAN prompts

# Define tools.
tools = [
    # Core tools.
    RunCommandTool(base_directory=base_directory),
    WriteFileTool(base_directory=base_directory),
    ReadFileTool(base_directory=base_directory, show_line_numbers=True),
    EditFileTool(base_directory=base_directory),
    FindFilesTool(base_directory=base_directory),
    FileSearchTool(base_directory=base_directory),
    RunPythonTool(base_directory=base_directory, timeout=1000),
    WebSearchTool(),
    # Event generation tools.
    FeynRulesToUFOTool(base_directory=base_directory, feynrules_path=feynrules_path, wolframscript_path=wolframscript_path),
    MadGraphFromRunCardTool(base_directory=base_directory, mg5_path=mg5_path),
    PythiaFromRunCardTool(base_directory=base_directory),
    JetClusterSlowJetTool(base_directory=base_directory),
    SherpaFromRunCardTool(base_directory=base_directory),
    # Data conversion tools.
    EventJSONLToNumpyTool(base_directory=base_directory),
    JetsJSONLToNumpyTool(base_directory=base_directory),
    # Analysis tools - Kinematics.
    CalculateInvariantMassTool(base_directory=base_directory),
    CalculateTransverseMomentumTool(base_directory=base_directory),
    CalculateDeltaRTool(base_directory=base_directory),
    ApplyCutsTool(base_directory=base_directory),
    # Analysis tools - Event selection.
    GetHardestNTool(base_directory=base_directory),
    GetHardestNJetsTool(base_directory=base_directory),
    FilterByPDGIDTool(base_directory=base_directory),
    SortByPtTool(base_directory=base_directory),
    #FilterByDeltaRTool(base_directory=base_directory),
    # Analysis tools - Invariant mass.
    ResonanceReconstructionTool(base_directory=base_directory),
    TodoRead(),
    TodoWrite(base_directory=base_directory)
]

# Hooks.
hooks = [
    #SafeguardHook(),
    #UserApprovalHook(),
    #DangerousCommandHook(), 
    TruncateOutputHook(max_length=10000),
]

# Default LLM - Choose one:

# ============================================================ #
# ====== Cloud LLM Providers (requires API key in .env) ====== #
# ============================================================ #

# Option 1: OpenAI GPT
LLM = GPT()

# Option 2: Anthropic Claude
#LLM = Claude()

# Option 3: Google Gemini
#LLM = Gemini()

# Option 4: Groq
#LLM = Groq()

# ============================================================ #
# ======= Local/Remote Ollama (configured in config.py) ====== #
# ============================================================ #

# Option 5: Ollama (uses config.py settings for model and host)
#LLM = get_ollama()

# Option 6: Ollama with model override
#LLM = get_ollama(model='gpt-oss:120b')

# Option 7: Ollama with host override (advanced - prefer config.py)
#LLM = get_ollama(host='http://SERVER_IP:11434')

# Option 8: Ollama with reasoning mode (for models like gpt-oss:20b)
#LLM = get_reasoning_ollama()

# ============================================================ #
# ===== Self-hosted vLLM / OpenAI-compatible server ========== #
# =====       (configured in config.py)              ========= #
# ============================================================ #

# Option 9: vLLM (uses config.vllm_host and config.vllm_model)
#LLM = get_vllm()

# Option 10: vLLM with model override (server may host multiple)
#LLM = get_vllm(model='meta-llama/Meta-Llama-3-8B-Instruct')

# Option 11: vLLM with host override (advanced - prefer config.py)
#LLM = get_vllm(host='http://localhost:8000/v1', model='gpt-oss:120b')

# ============================================================ #
# = LiteLLM Proxy (speaks OpenAI chat-completions on the wire) =#
# =====            (configured in config.py)         ========= #
# ============================================================ #

# Option 12: LiteLLM (uses config.litellm_host and config.litellm_model)
#LLM = get_litellm()

# Option 13: LiteLLM with model override (proxy routes by model name)
#LLM = get_litellm(model='openai/gpt-4o')

# Option 14: LiteLLM with host override (advanced - prefer config.py)
#LLM = get_litellm(host='http://localhost:4000/v1', model='ollama/gpt-oss:20b')

# Create agent.
agent = Agent(llm=LLM,
              tools=tools,
              tool_hooks=hooks,
              system_prompt=system_prompt,
              debug=False)

# Run the app server.
app_server.run_server(agent, host="127.0.0.1", port=8000, open_browser=True, max_tool_iterations=100)
