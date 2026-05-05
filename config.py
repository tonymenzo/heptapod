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
# Example: "/path/to/FeynRules_v2.3.49"
feynrules_path = "/path/to/FeynRules"

# WolframScript executable PATH.
# Example (macOS): "/Applications/Mathematica.app/Contents/MacOS/wolframscript"
# Example (Linux): "/usr/local/bin/wolframscript"
wolframscript_path = "/path/to/wolframscript"

# MadGraph5_aMC PATH.
# Example: "/path/to/MG5_aMC_v3.6.6"
mg5_path = "/path/to/MG5_aMC"