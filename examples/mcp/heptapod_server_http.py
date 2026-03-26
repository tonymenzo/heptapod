#!/usr/bin/env python3
"""
# heptapod_server_http.py is a part of the HEPTAPOD package.
# Copyright (C) 2025 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.
"""
"""
HEPTAPOD MCP Server — HTTP Transport (Streamable HTTP via FastMCP)

Exposes HEPTAPOD tools via HTTP using the FastMCP adapter.
Suitable for remote access, multi-client scenarios, and web integrations.

Usage:
    # Serve all available tool groups on default port
    python heptapod_server_http.py

    # Custom host, port, and tool groups
    python heptapod_server_http.py --host 0.0.0.0 --port 9000 --groups pdg,inspire

    # Custom workspace
    python heptapod_server_http.py --workspace /tmp/heptapod

Claude Code integration:
    # First, start this server in a terminal
    python heptapod_server_http.py

    # Then register with Claude Code
    claude mcp add --transport http heptapod-http http://127.0.0.1:8765/mcp

    # Or with a custom port
    claude mcp add --transport http heptapod-http http://127.0.0.1:9000/mcp

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
    parser = argparse.ArgumentParser(description="HEPTAPOD MCP Server (HTTP)")
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Port to bind to (default: 8765)",
    )
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

    # Import FastMCP adapter
    try:
        from orchestral.mcp import create_fastmcp_server
    except ImportError as e:
        print("Error: MCP SDK not installed. Install with: pip install mcp", file=sys.stderr)
        print(f"Details: {e}", file=sys.stderr)
        sys.exit(1)

    # Log server info
    print("=" * 60, file=sys.stderr)
    print("  HEPTAPOD MCP Server (HTTP Transport)", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    print(file=sys.stderr)
    print(f"URL: http://{args.host}:{args.port}/mcp", file=sys.stderr)
    print("Transport: Streamable HTTP", file=sys.stderr)
    print(file=sys.stderr)
    print(f"Tools ({len(tools)}):", file=sys.stderr)
    for tool in tools:
        desc = tool.__doc__ or tool.__class__.__doc__ or ""
        first_line = desc.strip().split("\n")[0] if desc else "No description"
        print(f"  - {tool.get_name()}: {first_line}", file=sys.stderr)
    print(file=sys.stderr)
    print("Press Ctrl+C to stop", file=sys.stderr)
    print(file=sys.stderr)

    # Create and run
    mcp = create_fastmcp_server(
        tools=tools,
        name="heptapod",
        host=args.host,
        port=args.port,
        stateless_http=True,
    )

    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
