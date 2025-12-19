# Setup repository path for imports
import sys
from pathlib import Path

# Add repository root to path for local imports (prompts, tools, etc.)
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# =========================================================== #
# ======================== IMPORTS ========================== #
# =========================================================== #

from orchestral import Agent
from orchestral.tools import (RunCommandTool, DummyRunCommandTool, WebSearchTool, RunPythonTool,
                              WriteFileTool, ReadFileTool, EditFileTool, FileSearchTool, FindFilesTool,
                              TodoWrite, TodoRead)
from orchestral.tools.hooks import TruncateOutputHook, DangerousCommandHook, SafeguardHook, UserApprovalHook
from orchestral.llm import GPT, Claude


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

from config import feynrules_path, mg5_path, wolframscript_path

print("Using FeynRules path:", feynrules_path)
print("Using MG5 path:", mg5_path)
print("Using wolframscript path:", wolframscript_path)

# Import app server (not the basic web UI server)
import app.server as app_server

# Import sandbox utilities
from sandbox_utils import create_new_sandbox

# Configure workspace - either use existing or create new sandbox
demo_files_dir = Path(__file__).resolve().parent / 'hep_bsm_sandbox'

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

# Default LLM.
#LLM = Claude()
LLM = GPT()

# Create agent.
agent = Agent(llm=LLM,
              tools=tools,
              tool_hooks=hooks,
              system_prompt=system_prompt,
              debug=False)

# Run the app server.
app_server.run_server(agent, host="127.0.0.1", port=8000, open_browser=True, max_tool_iterations=100)