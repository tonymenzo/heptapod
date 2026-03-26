"""
# orchestral_demo.py is a part of the HEPTAPOD package.
# Copyright (C) 2025 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.
"""
"""
Orchestral Demo

Interactive demo showcasing basic Orchestral agent capabilities.
A general-purpose physics assistant with file and command tools.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from orchestral import Agent
from orchestral.tools import (
    RunCommandTool, WriteFileTool, ReadFileTool, EditFileTool,
    FileSearchTool, FindFilesTool, RunPythonTool, TodoWrite, TodoRead
)
from orchestral.tools.hooks import TruncateOutputHook
from orchestral.llm import GPT, Claude

import app.server as app_server

DEMO_DIR = Path(__file__).resolve().parent
base_directory = str(DEMO_DIR / 'sandbox_000')
Path(base_directory).mkdir(exist_ok=True)

print(f"Working directory: {base_directory}")

tools = [
    WriteFileTool(base_directory=base_directory),
    ReadFileTool(base_directory=base_directory, show_line_numbers=True),
    EditFileTool(base_directory=base_directory),
    FindFilesTool(base_directory=base_directory),
    FileSearchTool(base_directory=base_directory),
    RunCommandTool(base_directory=base_directory),
    RunPythonTool(base_directory=base_directory, timeout=120),
    TodoRead(),
    TodoWrite(base_directory=base_directory)
]

hooks = [TruncateOutputHook(max_length=10000)]

system_prompt = """
You are a helpful physics research assistant with expertise in particle physics.

Your capabilities:
- File operations (read, write, edit, search)
- Running Python code and shell commands
- Task management with todos

When solving problems:
1. Break down complex tasks into steps
2. Explain your reasoning clearly
3. Write clean, well-documented code
4. Verify your results

Be concise but thorough in your responses.
""".strip()

LLM = GPT()

agent = Agent(
    llm=LLM,
    tools=tools,
    tool_hooks=hooks,
    system_prompt=system_prompt,
    debug=False
)

if __name__ == "__main__":
    print("Starting Orchestral Demo...")
    app_server.run_server(agent, host="127.0.0.1", port=8000, open_browser=True, max_tool_iterations=50)
