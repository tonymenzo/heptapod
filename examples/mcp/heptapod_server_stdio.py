#!/usr/bin/env python3
"""
# heptapod_server_stdio.py is a part of the HEPTAPOD package.
# Copyright (C) 2025 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.
"""
"""
HEPTAPOD MCP Server — STDIO Transport

Exposes HEPTAPOD tools via MCP over STDIO (JSON-RPC 2.0).
Designed for direct integration with Claude Code, Claude Desktop,
and other MCP-compatible clients.

Usage:
    # Serve all available tool groups
    python heptapod_server_stdio.py

    # Serve specific groups only
    python heptapod_server_stdio.py --groups pdg,units,inspire

    # Custom workspace directory
    python heptapod_server_stdio.py --workspace /tmp/heptapod

Claude Code integration:
    cd /path/to/heptapod
    claude mcp add heptapod -- python "$(pwd)/examples/mcp/heptapod_server_stdio.py"

    # Or with specific groups
    claude mcp add heptapod -- python "$(pwd)/examples/mcp/heptapod_server_stdio.py" --groups pdg,inspire,units

Available tool groups:
    pdg           - Particle Data Group database (masses, widths, branching fractions)
    inspire       - INSPIRE HEP literature search, citations, BibTeX
    units         - Natural units & metric prefix conversions
    analysis      - Kinematics, reconstruction, data format conversion
    event_gen     - MadGraph, Pythia, Sherpa (requires external software)
    feynrules     - FeynRules-to-UFO conversion (requires Mathematica)
"""

import sys
import os
import argparse

# Repository root for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from heptapod_tools import get_tools, TOOL_GROUPS


def main():
    parser = argparse.ArgumentParser(description="HEPTAPOD MCP Server (STDIO)")
    parser.add_argument(
        "--groups",
        type=str,
        default=None,
        help="Comma-separated tool groups to serve (default: all available). "
             f"Options: {', '.join(TOOL_GROUPS.keys())}",
    )
    parser.add_argument(
        "--workspace",
        type=str,
        default=None,
        help="Workspace directory for tool output (default: ~/.heptapod/mcp_workspace)",
    )
    args = parser.parse_args()

    # Parse group names
    group_names = tuple(g.strip() for g in args.groups.split(",")) if args.groups else ()

    # Load tools
    tools = get_tools(*group_names, base_dir=args.workspace)
    if not tools:
        print("Error: No tools could be loaded. Check group names and dependencies.", file=sys.stderr)
        sys.exit(1)

    # Import MCP server
    try:
        from orchestral.mcp import MCPServer
    except ImportError as e:
        print("Error: MCP SDK not installed. Install with: pip install mcp", file=sys.stderr)
        print(f"Details: {e}", file=sys.stderr)
        sys.exit(1)

    # Log server info to stderr (stdout is reserved for JSON-RPC)
    print("=" * 60, file=sys.stderr)
    print("  HEPTAPOD MCP Server (STDIO Transport)", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    print(file=sys.stderr)
    print(f"Tools ({len(tools)}):", file=sys.stderr)
    for tool in tools:
        spec = tool.get_tool_spec()
        desc = spec.description.split(".")[0] if spec.description else "No description"
        print(f"  - {tool.get_name()}: {desc}", file=sys.stderr)
    print(file=sys.stderr)
    print("Server starting on STDIO...", file=sys.stderr)
    print("Press Ctrl+C to stop", file=sys.stderr)
    print(file=sys.stderr)

    # Create and run
    server = MCPServer(
        tools=tools,
        name="heptapod",
        version="1.0.0",
    )

    try:
        server.run()
    except KeyboardInterrupt:
        print("\nServer stopped.", file=sys.stderr)


if __name__ == "__main__":
    main()
