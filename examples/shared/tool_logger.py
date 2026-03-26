"""
# tool_logger.py is a part of the HEPTAPOD package.
# Copyright (C) 2025 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.
"""
"""
Tool call logging utilities for Orchestral agents.

This module provides a ToolCallLogger hook that prints tool calls before and
after execution, useful for debugging and understanding agent behavior.

Usage:
    from examples.shared.tool_logger import ToolCallLogger

    hooks = [ToolCallLogger(verbose=True, show_results=False)]
    agent = Agent(llm=llm, tools=tools, tool_hooks=hooks, ...)
"""

from orchestral.tools.hooks import ToolHook, ToolHookResult
import json


class ToolCallLogger(ToolHook):
    """
    Hook that prints tool calls before and after execution.

    This is useful for:
    - Debugging agent behavior
    - Understanding which tools are being called
    - Seeing the arguments passed to tools
    - Verifying tool composition/chaining

    Example output:
        >>> TOOL CALL: lorentz_gamma
            Arguments: {"velocity": 0.995}
        <<< lorentz_gamma completed

        >>> TOOL CALL: time_dilation
            Arguments: {"proper_time": 2.2, "gamma": 10.01, "velocity": 0.995}
        <<< time_dilation completed
    """

    def __init__(self, verbose: bool = True, show_results: bool = False):
        """
        Initialize the tool call logger.

        Args:
            verbose: If True, show full arguments. If False, just tool names.
            show_results: If True, also print tool results (can be verbose).
        """
        self.verbose = verbose
        self.show_results = show_results

    def before_call(self, tool_name: str, arguments: dict) -> ToolHookResult:
        """Called before each tool execution."""
        if self.verbose:
            # Pretty print arguments (truncate long values)
            args_str = json.dumps(arguments, indent=2, default=str)
            if len(args_str) > 500:
                args_str = args_str[:500] + "\n  ... (truncated)"
            print(f"\n>>> TOOL CALL: {tool_name}")
            print(f"    Arguments: {args_str}")
        else:
            print(f">>> TOOL: {tool_name}")

        return ToolHookResult(approved=True)

    def after_call(self, tool_name: str, result) -> ToolHookResult:
        """Called after each tool execution."""
        if self.show_results:
            result_str = str(result)
            if len(result_str) > 300:
                result_str = result_str[:300] + "... (truncated)"
            print(f"<<< RESULT from {tool_name}: {result_str}")
        else:
            print(f"<<< {tool_name} completed")

        return ToolHookResult(approved=True)
