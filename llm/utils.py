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
