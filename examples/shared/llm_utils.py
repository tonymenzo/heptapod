"""
# llm_utils.py is a part of the HEPTAPOD package.
# Copyright (C) 2025 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.
"""
"""
LLM utility functions for HEPTAPOD notebooks.

This module provides utilities for discovering and displaying available LLM models
across different providers (OpenAI, Anthropic, Google, Groq, Ollama, etc.).

Usage:
    from examples.shared.llm_utils import list_available_models
    list_available_models()  # Prints formatted model list
"""

import subprocess


def list_available_models(show_ollama: bool = True) -> dict:
    """
    Display all available LLM models from Orchestral-supported providers.

    This function queries the Orchestral framework for available models and
    displays them in a formatted table with context window sizes and output limits.

    Args:
        show_ollama: If True, also check for locally installed Ollama models (default: True)

    Returns:
        dict: The raw models dictionary from get_available_models()

    Example:
        >>> from examples.shared.llm_utils import list_available_models
        >>> models = list_available_models()
        ======================================================================
        AVAILABLE LLM MODELS
        ======================================================================
        ...
    """
    from orchestral.llm import get_available_models

    # Get all available models from Orchestral
    models = get_available_models()

    print("=" * 70)
    print("AVAILABLE LLM MODELS")
    print("=" * 70)
    print()

    # OpenAI
    print("OpenAI GPT Models:")
    print("-" * 70)
    for model in models.get('openai', []):
        ctx = _format_tokens(model['context_window'])
        out = _format_tokens(model['output_limit'])
        print(f"  * {model['model_id']:40s} - {model['friendly_name']:25s} (ctx: {ctx:5s}, out: {out:5s})")
    print(f"  Usage: GPT(model='gpt-4o')")
    print()

    # Anthropic
    print("Anthropic Claude Models:")
    print("-" * 70)
    for model in models.get('anthropic', []):
        ctx = _format_tokens(model['context_window'])
        out = _format_tokens(model['output_limit'])
        print(f"  * {model['model_id']:40s} - {model['friendly_name']:25s} (ctx: {ctx:5s}, out: {out:5s})")
    print(f"  Usage: Claude(model='claude-sonnet-4-5-20250929')")
    print()

    # Google
    print("Google Gemini Models:")
    print("-" * 70)
    for model in models.get('google', [])[:5]:  # Show top 5
        ctx = _format_tokens(model['context_window'])
        out = _format_tokens(model['output_limit'])
        print(f"  * {model['model_id']:40s} - {model['friendly_name']:25s} (ctx: {ctx:5s}, out: {out:5s})")
    print(f"  Usage: Gemini(model='gemini-2.0-flash-exp')")
    print()

    # Groq
    print("Groq Models (Fast Inference):")
    print("-" * 70)
    for model in models.get('groq', []):
        ctx = _format_tokens(model['context_window'])
        out = _format_tokens(model['output_limit'])
        print(f"  * {model['model_id']:45s} - {model['friendly_name']:25s} (ctx: {ctx:5s}, out: {out:5s})")
    print(f"  Usage: Groq(model='llama-3.3-70b-versatile')")
    print()

    # Ollama (check what's installed locally)
    if show_ollama:
        print("Ollama Models (Local):")
        print("-" * 70)
        print("Installed on this system:")
        try:
            result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if len(lines) > 1:
                    for line in lines[1:]:
                        if line.strip():
                            print(f"  * {line.split()[0]}")
                else:
                    print("  (none installed - run 'ollama pull MODEL_NAME' to install)")
            else:
                print("  (Ollama not responding)")
        except FileNotFoundError:
            print("  (Ollama not installed - visit https://ollama.ai)")
        except subprocess.TimeoutExpired:
            print("  (Ollama timed out)")
        except Exception:
            print("  (Ollama not available)")
        print(f"  Usage: get_ollama(model='llama3:70b')")
        print()

    print("=" * 70)
    print(f"Total providers: {len(models)} (OpenAI, Anthropic, Google, Groq, Mistral, Bedrock, Ollama)")
    print("Run get_available_models() to see the full list with Mistral & Bedrock models")

    return models


def _format_tokens(count: int) -> str:
    """Format token count as human-readable string (e.g., 128K, 2M)."""
    if count >= 1_000_000:
        return f"{count // 1_000_000}M"
    elif count >= 1_000:
        return f"{count // 1_000}K"
    else:
        return str(count)
