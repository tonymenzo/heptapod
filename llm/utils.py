"""
# utils.py is a part of the HEPTAPOD package.
# Copyright (C) 2025 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.
"""

"""
Utilities for working with LLMs in the HEP workflow.

This module provides helper functions to easily get configured LLM instances
using settings from config.py, abstracting away the connection details from users.
"""

import sys
import os

# Add external/dep to path for orchestral imports
_module_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_module_dir)
sys.path.insert(0, os.path.join(_project_root, 'external', 'dep'))

from orchestral.llm import Ollama

# Add project root for config import
sys.path.insert(0, _project_root)
import config


def get_ollama(model=None, host=None, **kwargs):
    """
    Get a configured Ollama LLM instance using settings from config.py.

    This is the recommended way to get an Ollama instance in your code.
    It automatically uses the configuration from config.py, but allows
    overrides for flexibility.

    Args:
        model: Model name to use. If None, uses config.ollama_model.
               Examples: 'gpt-oss:20b', 'llama3.2', 'mistral'
        host: Ollama server host. If None, uses config.ollama_host.
              Examples: None (local), 'http://192.168.1.100:11434' (remote)
        **kwargs: Additional arguments passed to Ollama constructor
                  (e.g., think=True for reasoning mode)

    Returns:
        Ollama: Configured Ollama LLM instance

    Examples:
        # Basic usage - uses config.py settings
        llm = get_ollama()

        # Override model
        llm = get_ollama(model='llama3.2')

        # Enable reasoning mode
        llm = get_ollama(think=True)

        # Use a different remote server (override config)
        llm = get_ollama(host='http://other-server:11434')
    """
    # Use config defaults if not specified
    if model is None:
        model = config.ollama_model
    if host is None:
        host = config.ollama_host

    # Create Ollama instance
    if host is None:
        # Local instance
        return Ollama(model=model, **kwargs)
    else:
        # Remote instance
        return Ollama(model=model, host=host, **kwargs)


def get_reasoning_ollama(model=None, host=None, **kwargs):
    """
    Get an Ollama instance with reasoning/thinking mode enabled.

    This is a convenience function for models that support chain-of-thought
    reasoning (like gpt-oss:20b). It automatically sets think=True.

    Args:
        model: Model name to use. If None, uses config.ollama_model.
        host: Ollama server host. If None, uses config.ollama_host.
        **kwargs: Additional arguments passed to Ollama constructor

    Returns:
        Ollama: Configured Ollama instance with reasoning enabled

    Example:
        llm = get_reasoning_ollama()
        # Equivalent to: get_ollama(think=True)
    """
    return get_ollama(model=model, host=host, think=True, **kwargs)


def list_available_models(host=None):
    """
    List all models available on the Ollama server.

    Args:
        host: Ollama server host. If None, uses config.ollama_host.

    Returns:
        list: List of available model names

    Example:
        models = list_available_models()
        print(f"Available models: {models}")
    """
    from ollama import Client

    if host is None:
        host = config.ollama_host

    if host is None:
        client = Client()
    else:
        client = Client(host=host)

    models = client.list()
    return [model.model for model in models.models]


def _build_openai_compatible_client(*, host, model, api_key, label):
    """Construct an Orchestral GPT instance pointed at an OpenAI-compatible
    endpoint (vLLM, LiteLLM proxy, or any other server speaking the OpenAI
    wire protocol). Internal helper for get_vllm / get_litellm.
    """
    import openai
    from orchestral.llm import GPT
    from orchestral.llm.base.llm import LLM

    if not host:
        raise ValueError(
            f"No {label} host configured. Set config.{label}_host (e.g. "
            f"'http://localhost:8000/v1') or pass host=... explicitly."
        )
    if not model:
        raise ValueError(
            f"No {label} model specified. Set config.{label}_model or pass "
            f"model=... . Query {host.rstrip('/')}/models to list what the "
            f"server serves."
        )

    gpt = GPT.__new__(GPT)
    LLM.__init__(gpt, tools=None)
    gpt.model = model
    gpt.api_key = api_key
    gpt.client = openai.Client(api_key=api_key, base_url=host, timeout=60.0)
    return gpt


def _list_openai_compatible_models(*, host, api_key, label):
    """Query `<host>/models` (OpenAI listing endpoint) and return ids.
    Internal helper for list_vllm_models / list_litellm_models.
    """
    import json
    from urllib.request import Request, urlopen

    if not host:
        raise ValueError(f"No {label} host configured. Set config.{label}_host.")
    req = Request(f"{host.rstrip('/')}/models",
                  headers={"Authorization": f"Bearer {api_key}"})
    with urlopen(req, timeout=15) as r:
        payload = json.loads(r.read())
    return [m["id"] for m in payload.get("data", [])]


def get_vllm(model=None, host=None, api_key=None, **kwargs):
    """
    Get an Orchestral GPT instance pointed at an OpenAI-compatible vLLM server.

    vLLM speaks the OpenAI `/v1/chat/completions` wire protocol, so the
    same Orchestral `GPT` class works once we override its client's
    `base_url`. Use this for any vLLM deployment — local, cluster-hosted,
    or shared infrastructure.

    Args:
        model: Model identifier as registered by the server (e.g.
            'meta-llama/Meta-Llama-3-8B-Instruct', 'gpt-oss:120b').
            Defaults to config.vllm_model. Query `<host>/models` to list
            what the server actually serves — vLLM does not pull on demand.
        host: Server base URL including the `/v1` suffix (e.g.
            'http://localhost:8000/v1'). Defaults to config.vllm_host.
        api_key: Bearer token. Defaults to env var VLLM_API_KEY, then
            'dummy'. vLLM accepts any non-empty string when launched
            without `--api-key`.
        **kwargs: Reserved for future use; currently ignored.

    Returns:
        Orchestral `GPT` instance with `.client.base_url` pointing at the
        vLLM server. Use exactly like any other LLM in the harness.

    Examples:
        # Uses config.py settings
        llm = get_vllm()

        # Override model (e.g. multi-model server)
        llm = get_vllm(model='Qwen/Qwen2.5-32B-Instruct')

        # Point at a different vLLM server
        llm = get_vllm(host='http://my-vllm:8000/v1', model='gpt-oss:120b')
    """
    import os
    from dotenv import load_dotenv

    if host is None:
        host = config.vllm_host
    if model is None:
        model = config.vllm_model
    if api_key is None:
        load_dotenv()
        api_key = os.getenv("VLLM_API_KEY") or "dummy"
    return _build_openai_compatible_client(
        host=host, model=model, api_key=api_key, label="vllm",
    )


def list_vllm_models(host=None):
    """
    List models served by a vLLM server.

    Unlike Ollama, vLLM does not pull on demand — it only serves models
    the operator launched. This helper queries `<host>/models` (the
    OpenAI-compatible listing endpoint) and returns the model IDs.

    Args:
        host: vLLM server base URL with `/v1` suffix. Defaults to
            config.vllm_host.

    Returns:
        list[str]: Model identifiers as the server reports them.
    """
    import os
    from dotenv import load_dotenv

    if host is None:
        host = config.vllm_host
    load_dotenv()
    api_key = os.getenv("VLLM_API_KEY") or "dummy"
    return _list_openai_compatible_models(
        host=host, api_key=api_key, label="vllm",
    )


def get_litellm(model=None, host=None, api_key=None, **kwargs):
    """
    Get an Orchestral GPT instance pointed at a LiteLLM proxy.

    LiteLLM in proxy mode exposes a `/v1/chat/completions` endpoint that
    speaks OpenAI's chat-completions wire format on the frontend, then
    translates each request to the native API of whatever backend handles
    the requested model (OpenAI, Anthropic, Bedrock, vLLM, Ollama, ...).
    Because the wire protocol is OpenAI's, Orchestral's `GPT` class works
    against any LiteLLM-routed model once we override `client.base_url`.

    Note: routing through LiteLLM means Orchestral uses its OpenAI client
    even for non-OpenAI models, so provider-specific features that live
    in Orchestral's `Claude` / `Gemini` classes (Anthropic prompt caching,
    Gemini grounding metadata, etc.) are not exercised here.

    Args:
        model: Model identifier as the proxy admin registered it.
            Names depend on the proxy config; common examples:
            'openai/gpt-4o', 'anthropic/claude-3-5-sonnet', 'ollama/gpt-oss:20b'.
            Defaults to config.litellm_model. Query `<host>/models` to list.
        host: LiteLLM base URL incl. `/v1` (e.g.
            'https://litellm.example.org/v1'). Defaults to config.litellm_host.
        api_key: Virtual key (typically `sk-...`). Defaults to env var
            LITELLM_API_KEY.
        **kwargs: Reserved for future use; currently ignored.

    Returns:
        Orchestral `GPT` instance with `.client.base_url` pointing at the
        LiteLLM proxy. Use exactly like any other LLM in the harness.

    Examples:
        # Uses config.py settings
        llm = get_litellm()

        # Override the model the proxy routes to
        llm = get_litellm(model='openai/gpt-4o')

        # Different proxy
        llm = get_litellm(host='http://localhost:4000/v1', model='ollama/gpt-oss:20b')
    """
    import os
    from dotenv import load_dotenv

    if host is None:
        host = config.litellm_host
    if model is None:
        model = config.litellm_model
    if api_key is None:
        load_dotenv()
        api_key = os.getenv("LITELLM_API_KEY") or ""
    return _build_openai_compatible_client(
        host=host, model=model, api_key=api_key, label="litellm",
    )


def list_litellm_models(host=None):
    """
    List models routed by a LiteLLM proxy.

    Mirrors `list_vllm_models`. The proxy reports whatever its admin
    registered — names are often qualified, e.g. 'openai/gpt-4o' or
    'ollama/gpt-oss:20b'.

    Args:
        host: Proxy base URL with `/v1` suffix. Defaults to
            config.litellm_host.

    Returns:
        list[str]: Model identifiers as the proxy reports them.
    """
    import os
    from dotenv import load_dotenv

    if host is None:
        host = config.litellm_host
    load_dotenv()
    api_key = os.getenv("LITELLM_API_KEY") or ""
    return _list_openai_compatible_models(
        host=host, api_key=api_key, label="litellm",
    )


def print_config_info():
    """
    Print current Ollama configuration from config.py.

    Useful for debugging or verifying settings.
    """
    print("=" * 60)
    print("Current Ollama Configuration (from config.py)")
    print("=" * 60)
    print(f"Model: {config.ollama_model}")
    print(f"Host:  {config.ollama_host or 'localhost:11434 (default)'}")
    print()

    try:
        models = list_available_models()
        print(f"Available models ({len(models)}):")
        for model in models:
            marker = "✓" if model == config.ollama_model else " "
            print(f"  {marker} {model}")
    except Exception as e:
        print(f"⚠️  Could not connect to Ollama: {e}")

    print("=" * 60)


if __name__ == '__main__':
    # When run directly, print configuration info
    print_config_info()
