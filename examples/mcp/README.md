# HEPTAPOD MCP Servers

Expose HEPTAPOD's particle physics tools via the [Model Context Protocol (MCP)](https://modelcontextprotocol.io), making them available to Claude Code, Claude Desktop, OpenAI Codex, and any MCP-compatible client.

## Prerequisites

```bash
pip install mcp
```

The `orchestral-ai` package must also be installed (or on `sys.path` via the repo root).

## Quick Start

### STDIO Server

The client manages the server lifecycle — it spawns the process on startup and stops it when done. Best for single-user local setups where you want zero maintenance after registration.

```bash
# Serve all available tools
python examples/mcp/heptapod_server_stdio.py

# Serve specific groups
python examples/mcp/heptapod_server_stdio.py --groups pdg,units,inspire
```

### HTTP Server

A persistent server that clients connect to over HTTP. Best for multi-client access, remote/shared servers, or when you want the server running independently.

```bash
# Default: http://127.0.0.1:8765/mcp
python examples/mcp/heptapod_server_http.py

# Custom host/port with specific groups
python examples/mcp/heptapod_server_http.py --host 0.0.0.0 --port 9000 --groups pdg,inspire
```

## Tool Groups

Tools are organized into named groups. Use `--groups` to select which ones to serve. If omitted, all groups that can be loaded are served.

| Group | Tools | Requires |
|-------|-------|----------|
| `pdg` | PDG database queries (mass, width, lifetime, branching fractions) | Internet |
| `inspire` | INSPIRE HEP literature search, citations, BibTeX, reading list | Internet |
| `units` | Natural units and metric prefix conversions | — |
| `analysis` | Kinematics, reconstruction, LHE/JSONL/NumPy conversion | — |
| `event_gen` | MadGraph, Pythia, Sherpa event generation | MG5, Pythia, Sherpa |
| `feynrules` | FeynRules `.fr` to UFO conversion | Mathematica |

**Lightweight groups** (`pdg`, `inspire`, `units`) work out of the box with no external software. The remaining groups are skipped automatically if their dependencies are not installed.

## CLI Options

Both `heptapod_server_stdio.py` and `heptapod_server_http.py` accept:

| Flag | Default | Description |
|------|---------|-------------|
| `--groups` | all available | Comma-separated list of tool group names to serve |
| `--workspace` | `~/.heptapod/mcp_workspace` | Working directory where tools write output files |

`heptapod_server_http.py` additionally accepts:

| Flag | Default | Description |
|------|---------|-------------|
| `--host` | `127.0.0.1` | Network interface to bind to (`0.0.0.0` for all interfaces) |
| `--port` | `8765` | Port number for the HTTP endpoint |

### Examples

Serve only the lightweight reference tools:

```bash
python examples/mcp/heptapod_server_stdio.py --groups pdg,inspire,units
```

Serve analysis tools with a custom workspace:

```bash
python examples/mcp/heptapod_server_stdio.py --groups analysis,pdg --workspace /tmp/heptapod_work
```

Expose all tools over HTTP on a non-default port:

```bash
python examples/mcp/heptapod_server_http.py --port 9000
```

## Connecting to Claude Code

### Understanding scope

MCP servers in Claude Code are scoped by **where** they are registered. This determines who can see the tools and in which projects.

| Scope | Config location | Access | CLI flag |
|-------|-----------------|--------|----------|
| **Local** | `~/.claude.json` (per-project entry) | Only you, only in the current project | `--scope local` (default) |
| **Project** | `.mcp.json` at the project root | Everyone with the repo, in this project only | `--scope project` |
| **User** | `~/.claude.json` (global entry) | Only you, across all projects | `--scope user` |

Since HEPTAPOD tools are general-purpose physics tools, **user scope** is usually the best choice — it makes the tools available regardless of which project you're working in. **Project scope** (`.mcp.json`) is useful for sharing the config with collaborators via version control.

> **Note:** MCP servers are configured in `~/.claude.json` or `.mcp.json`, **not** in `.claude/settings.json` or `.claude/settings.local.json` (those are for permissions and general settings).

### Option 1: CLI registration (recommended)

The `claude mcp add` command writes the config file for you. All examples below assume you are in the heptapod repo root and that your Python environment has `orchestral-ai` and `mcp` installed.

#### User scope — available in every project

This is the recommended setup for personal use. The tools will appear in Claude Code no matter what directory you're working in.

```bash
claude mcp add --scope user heptapod -- \
  /path/to/envs/heptapod/bin/python "$(pwd)/examples/mcp/heptapod_server_stdio.py"
```

To serve only specific groups:

```bash
claude mcp add --scope user heptapod -- \
  /path/to/envs/heptapod/bin/python "$(pwd)/examples/mcp/heptapod_server_stdio.py" \
  --groups pdg,inspire,units
```

#### Project scope — shared with collaborators

Writes a `.mcp.json` file at the project root. Anyone who clones the repo gets the same MCP config.

```bash
claude mcp add --scope project heptapod -- \
  /path/to/envs/heptapod/bin/python "$(pwd)/examples/mcp/heptapod_server_stdio.py"
```

#### Local scope — single project, single user

The default if no `--scope` flag is given. Useful for one-off testing.

```bash
claude mcp add heptapod -- \
  /path/to/envs/heptapod/bin/python "$(pwd)/examples/mcp/heptapod_server_stdio.py"
```

#### HTTP transport

If you prefer to run the server independently (e.g., on a remote machine), use HTTP transport. Start the server first, then register.

```bash
# Terminal 1: start the server
python examples/mcp/heptapod_server_http.py

# Terminal 2: register with Claude Code
claude mcp add --scope user --transport http heptapod-http http://127.0.0.1:8765/mcp
```

After registration, Claude Code automatically starts STDIO servers when needed — no separate terminal required. HTTP servers must be started manually before use.

### Option 2: Manual `.mcp.json` (project scope)

Create a `.mcp.json` file at your project root. This file can be checked into version control so collaborators get the same tools.

> **Important:** The `command` field must be the **full path** to the Python executable that has `orchestral` and `mcp` installed. If you use a conda or virtual environment, use that environment's Python (e.g., `/path/to/envs/heptapod/bin/python`), not just `python`.

**STDIO:**

```json
{
  "mcpServers": {
    "heptapod": {
      "command": "/path/to/envs/heptapod/bin/python",
      "args": [
        "/absolute/path/to/heptapod/examples/mcp/heptapod_server_stdio.py",
        "--groups", "pdg,inspire,units"
      ]
    }
  }
}
```

**HTTP:**

```json
{
  "mcpServers": {
    "heptapod-http": {
      "type": "http",
      "url": "http://127.0.0.1:8765/mcp"
    }
  }
}
```

After editing `.mcp.json`, **restart Claude Code** for changes to take effect.

### Verifying the connection

After registering, start (or restart) Claude Code and check that the tools are available:

```bash
claude mcp list
```

You should see `heptapod` (or `heptapod-http`) with the tool count listed. You can also type `/mcp` inside Claude Code to see server status.

Try asking Claude:

- *"What is the mass of the top quark?"* (uses PDG)
- *"Find recent papers on leptoquark searches"* (uses INSPIRE)
- *"Convert 125 GeV to kg"* (uses Units)

### Removing the server

```bash
# Remove from the default scope
claude mcp remove heptapod

# Or specify scope explicitly
claude mcp remove --scope user heptapod
claude mcp remove --scope project heptapod
```

## Connecting to OpenAI Codex

Codex stores MCP configuration in [`config.toml`](https://developers.openai.com/codex/mcp/). The same config is shared between the Codex CLI and the IDE extension.

| Scope    | Config location                              | Access               |
|----------|----------------------------------------------|----------------------|
| **User** | `~/.codex/config.toml`                       | All projects         |
| **Project** | `.codex/config.toml` (trusted projects only) | Current project only |

### CLI registration (recommended)

```bash
cd /path/to/heptapod

# STDIO server
codex mcp add heptapod -- \
  /path/to/envs/heptapod/bin/python "$(pwd)/examples/mcp/heptapod_server_stdio.py"

# STDIO with specific groups
codex mcp add heptapod -- \
  /path/to/envs/heptapod/bin/python "$(pwd)/examples/mcp/heptapod_server_stdio.py" \
  --groups pdg,inspire,units

# HTTP server (start the server first, then register)
codex mcp add heptapod-http --transport http -- http://127.0.0.1:8765/mcp
```

### Manual `config.toml`

Edit `~/.codex/config.toml` (or `.codex/config.toml` at the project root for project scope).

> **Important:** The `command` field must be the full path to the Python executable that has `orchestral` and `mcp` installed.

**STDIO:**

```toml
[mcp_servers.heptapod]
command = "/path/to/envs/heptapod/bin/python"
args = [
  "/absolute/path/to/heptapod/examples/mcp/heptapod_server_stdio.py",
  "--groups", "pdg,inspire,units"
]
```

**HTTP:**

```toml
[mcp_servers.heptapod-http]
url = "http://127.0.0.1:8765/mcp"
```

**Optional settings** (work with either transport):

```toml
[mcp_servers.heptapod]
enabled = true              # disable without deleting (default: true)
startup_timeout_sec = 30    # server initialization timeout (default: 10)
tool_timeout_sec = 120      # per-tool execution timeout (default: 60)
```

### Verifying and removing

```bash
# Check registered servers
codex mcp list

# Remove a server
codex mcp remove heptapod
```

## Programmatic Usage

You can also use the tool registry directly in Python without running a server:

```python
from heptapod_tools import get_tools, get_available_groups

# See which groups can be loaded in the current environment
print(get_available_groups())

# Load specific groups
tools = get_tools("pdg", "units")
```

To build a custom server from loaded tools:

```python
# STDIO server
from orchestral.mcp import MCPServer

server = MCPServer(tools=tools, name="my-heptapod")
server.run()
```

```python
# HTTP server
from orchestral.mcp import create_fastmcp_server

mcp = create_fastmcp_server(tools=tools, name="my-heptapod", port=9000)
mcp.run(transport="streamable-http")
```
