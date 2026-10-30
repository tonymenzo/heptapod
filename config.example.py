"""
# config.py is a part of the HEPTAPOD package.
# Copyright (C) 2025 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.
"""

# ============================================================ #
# ================== Ollama LLM Configuration ================ #
# ============================================================ #

# For local Ollama, use None (default)
# For remote Ollama server, set to "http://SERVER_IP:11434"
ollama_host = None  # None = localhost:11434 (default)

# Default Ollama model to use
ollama_model = "gpt-oss:20b"  # Change to your preferred model

# ============================================================ #
# =================== vLLM LLM Configuration ================= #
# ============================================================ #

# Base URL of an OpenAI-compatible vLLM server (must include the /v1
# suffix). Examples:
#   - Local:  "http://localhost:8000/v1"
#   - Remote: "https://vllm.example.org/v1"
# Set to None to disable.
vllm_host = None

# Default vLLM model identifier, as registered by the server. Query
# `<vllm_host>/models` to list what is available — vLLM only serves
# models the operator launched; there is no on-demand pull.
vllm_model = None  # e.g. "gpt-oss:120b" or "meta-llama/Meta-Llama-3-8B-Instruct"

# Bearer token: set VLLM_API_KEY in `.env`. Any non-empty value works
# for servers launched without `--api-key`.

# ============================================================ #
# ================= LiteLLM Proxy Configuration ============== #
# ============================================================ #

# LiteLLM is an OpenAI-compatible proxy that routes requests to many
# backends (vLLM, Ollama, Bedrock, hosted APIs) through one unified
# endpoint. Same wire protocol as vLLM, so the same client class works.
#
# Examples:
#   - "https://litellm.example.org/v1"
#   - "http://localhost:4000/v1"
litellm_host = None

# Default LiteLLM model identifier. Names depend on how the proxy admin
# registered backends — query `<litellm_host>/models` to list. Often
# qualified, e.g. "openai/gpt-4o" or "ollama/gpt-oss:20b".
litellm_model = None

# Virtual key (sk-...): set LITELLM_API_KEY in `.env`.

# ============================================================ #
# =================== External dependencies ================== #
# ============================================================ #

# FeynRules PATH.
feynrules_path = "/path/to/FeynRules"

# WolframScript executable PATH.
# Example (macOS): "/Applications/Mathematica.app/Contents/MacOS/wolframscript"
# Example (Linux): "/usr/local/bin/wolframscript"
wolframscript_path = "/path/to/wolframscript"

# Local Pythia 8 / Sherpa install prefixes (optional). When set, the Pythia
# and Sherpa run-card tools drive these installs through a C++ driver
# compiled against them instead of the pythia8mc / sherpa-mc wheels.
# pythia_path holds bin/pythia8-config; sherpa_path holds bin/Sherpa-config.
pythia_path = "/path/to/pythia"
sherpa_path = "/path/to/Sherpa"

# MadGraph5_aMC PATH (the unpacked MG5_aMC_v<version> directory).
mg5_path = "/path/to/MG5_aMC"

# Rivet installation prefix (the directory holding bin/rivet-config), for the
# rivet bundle. A conda-forge environment works:
#   conda create -n rivet -c conda-forge rivet
# Example: "/opt/miniconda3/envs/rivet"
rivet_path = "/path/to/rivet"