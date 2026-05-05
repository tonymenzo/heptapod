"""
# eda_demo.py is a part of the HEPTAPOD package.
# Copyright (C) 2026 HEPTAPOD authors (see AUTHORS for details).
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

# =========================================================== #
# ======================== IMPORTS ========================== #
# =========================================================== #

# Orchestral imports.
from orchestral import Agent
from orchestral.tools import (RunCommandTool, WriteFileTool, ReadFileTool,
                              EditFileTool, FileSearchTool, FindFilesTool,
                              RunPythonTool, WebSearchTool, TodoWrite, TodoRead)
from orchestral.tools.hooks import TruncateOutputHook

# LLM imports.
from orchestral.llm import GPT, Claude, Gemini, Groq
from llm import get_ollama, get_reasoning_ollama, get_vllm, get_litellm

# Tool imports — EDA study group: symbolic FeynCalc + NDA cross-checks + PDG.
from tools.eda import (RunWolframScript, RunWolframScriptBatch,
                       ComputeSymbolicAmplitude, ConvertToPython,
                       SimplifyResult, SimplifyResultBatch)
from tools.nda import EstimateDecayWidthNDATool, EstimateDecayWidthFormulaNDATool
from tools.pdg import PDGDatabaseTool

# Configuration imports.
from config import wolframscript_path

# Import sandbox utilities and app server.
from sandbox_utils import create_new_sandbox
import app.server as app_server

print("Using wolframscript path:", wolframscript_path)

# =========================================================== #
# ====================== CONFIGURATION ===================== #
# =========================================================== #

# Configure workspace.
demo_files_dir = Path(__file__).resolve().parent

CREATE_NEW_SANDBOX = True  # Set to True to create a new sandbox, False to use existing

# Load system prompt.
SYSTEM_PROMPT_PATH = REPO_ROOT / 'prompts' / 'examples' / 'eda' / 'system' / 'eda_system_prompt.md'
system_prompt = SYSTEM_PROMPT_PATH.read_text()

if CREATE_NEW_SANDBOX:
    base_directory, _ = create_new_sandbox(demo_files_dir, mode="explorer")
    # Override the default system prompt with the EDA-specific one
else:
    base_directory = str(demo_files_dir / 'sandbox000')

# =========================================================== #
# ========================= TOOLS ========================== #
# =========================================================== #

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
    # EDA tools — exact tree-level calculations via FeynCalc.
    ComputeSymbolicAmplitude(base_directory=base_directory),
    RunWolframScript(base_directory=base_directory, wolframscript_path=wolframscript_path),
    RunWolframScriptBatch(base_directory=base_directory, wolframscript_path=wolframscript_path),
    SimplifyResult(base_directory=base_directory),
    SimplifyResultBatch(base_directory=base_directory),
    ConvertToPython(base_directory=base_directory),
    # NDA tools — order-of-magnitude cross-checks.
    EstimateDecayWidthNDATool(base_directory=base_directory),
    EstimateDecayWidthFormulaNDATool(base_directory=base_directory),
    # PDG tools — reference experimental values.
    PDGDatabaseTool(base_directory=base_directory),
    TodoRead(),
    TodoWrite(base_directory=base_directory),
]

# Hooks.
hooks = [
    TruncateOutputHook(max_length=10000),
]

# =========================================================== #
# =========================== LLM ========================== #
# =========================================================== #

# Default LLM - Choose one:

# Cloud providers (requires API key in .env)
LLM = Claude()
#LLM = GPT()
#LLM = Gemini()
#LLM = Groq()

# Local Ollama (configured in config.py)
#LLM = get_ollama()

# Self-hosted vLLM / OpenAI-compatible server (configured in config.py)
#LLM = get_vllm()

# LiteLLM proxy — speaks OpenAI chat-completions on the wire, routes
# to any registered backend (configured in config.py)
#LLM = get_litellm()

# =========================================================== #
# ========================== RUN =========================== #
# =========================================================== #

# Create agent.
agent = Agent(llm=LLM,
              tools=tools,
              tool_hooks=hooks,
              system_prompt=system_prompt,
              debug=False)

# Run the app server.
app_server.run_server(agent, host="127.0.0.1", port=8000, open_browser=True, max_tool_iterations=100)
