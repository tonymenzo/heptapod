#!/usr/bin/env python3
"""
# test_ollama_basic.py is a part of the HEPTAPOD package.
# Copyright (C) 2025 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.
"""

"""
Basic test to verify Ollama is working with orchestral.
This bypasses the web UI entirely to isolate the Ollama integration.

This is the main Ollama test used by test_runner.py.
"""

import sys
import os
import subprocess
from pathlib import Path

# Setup paths - we're in llm/, need to go up one level to project root
llm_dir = Path(__file__).resolve().parent
project_root = llm_dir.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "external" / "dep"))

import config
from llm import get_ollama
from orchestral.context import Context
from orchestral.context.message import Message

print("=" * 60)
print("Testing Ollama Integration")
print("=" * 60)


def check_ollama_installed():
    """Check if Ollama is installed."""
    try:
        result = subprocess.run(
            ['ollama', '--version'],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def check_server_running(host_url):
    """Check if Ollama server is running at the specified host."""
    try:
        result = subprocess.run(
            ['curl', '-s', '-f', f'{host_url}/api/tags'],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


# Pre-flight checks
print("\n>> Pre-flight check: Is Ollama installed?")

if not check_ollama_installed():
    print("   ✗ Ollama is not installed")
    print("\n" + "=" * 60)
    print("SKIPPED: Ollama not installed")
    print("=" * 60)
    print("\nTo run this test:")
    print("  1. Install Ollama: https://ollama.com/download")
    print("  2. Start the server: ollama serve (or use the macOS app)")
    print("  3. Pull a model: ollama pull qwen2.5:3b")
    print("=" * 60)
    sys.exit(0)

print("   ✓ Ollama is installed")

# Check configuration
print("\n>> Reading configuration from config.py...")
print(f"   Model: {config.ollama_model}")

# Determine the host to check
if config.ollama_host:
    host_to_check = config.ollama_host
    is_remote = True
    print(f"   Host: {config.ollama_host} (remote server)")
else:
    host_to_check = "http://localhost:11434"
    is_remote = False
    print(f"   Host: localhost:11434 (local Ollama)")

# Check if server is running
print(f"\n>> Checking server availability...")

if not check_server_running(host_to_check):
    print(f"   ✗ Ollama server is not responding at {host_to_check}")
    print("\n" + "=" * 60)

    # If a server is explicitly configured, this is a FAILURE
    # If using default local server, this is a SKIP
    if is_remote:
        print("FAILED: Configured Ollama server not available")
        print("=" * 60)
        print("\nYou have configured a specific Ollama server in config.py:")
        print(f"  ollama_host = '{config.ollama_host}'")
        print("\nThis server is not responding or not reachable.")
        print("\nTroubleshooting:")
        print(f"  1. Verify the server is running and accessible")
        print(f"  2. Check network connectivity: curl {host_to_check}/api/tags")
        print(f"  3. Ensure firewall allows connections")
        print(f"  4. Verify the hostname/IP in config.py is correct")
        print(f"\nTo use local Ollama instead, set in config.py:")
        print(f"  ollama_host = None")
        print("=" * 60)
        sys.exit(1)  # Exit with failure code
    else:
        print("SKIPPED: Ollama not configured in config.py")
        print("=" * 60)
        print("\nNo Ollama server is configured (ollama_host = None in config.py).")
        print("Local Ollama server is also not running.")
        print("\nThis test is optional and will be skipped.")
        print("\nTo run this test, choose one option:")
        print("\n  Option A - Use local Ollama (recommended for development):")
        print("    1. Start Ollama: ollama serve (or use the macOS app)")
        print("    2. Keep config.py as: ollama_host = None")
        print("\n  Option B - Use a remote/external Ollama server:")
        print("    1. Set in config.py: ollama_host = 'http://your-server:11434'")
        print("    2. Ensure that server is running and accessible")
        print("\nFor auto-start testing, use: python llm/test_ollama_remote_auto.py")
        print("=" * 60)
        sys.exit(0)  # Exit with success (skipped, not failed)

print(f"   ✓ Ollama server is running")

try:
    # Get Ollama instance
    print("\n1. Creating Ollama instance...")
    llm = get_ollama()
    print(f"   ✓ Model: {llm.model}")
    print(f"   ✓ Host: {llm.host or 'localhost:11434'}")

    # Create simple context with a test message
    print("\n2. Creating test message...")
    test_prompt = 'Say "Hello, Ollama is working!" and nothing else.'
    context = Context()
    context.add_message(Message(role='user', text=test_prompt))
    print("   ✓ Context created")
    print(f"\n   User prompt:")
    print(f"   > {test_prompt}")

    # Get response
    print("\n3. Getting response from Ollama...")
    import time
    start = time.time()

    response = llm.get_response(context)

    elapsed = time.time() - start
    print(f"   ✓ Response received in {elapsed:.2f}s")

    # Show response
    print("\n4. Ollama response:")
    print(f"   {response.message.text}")

    print("\n" + "=" * 60)
    print("✓ SUCCESS: Ollama is working correctly!")
    print("=" * 60)

    # Now test if it works with the Agent
    print("\n" + "=" * 60)
    print("Testing with Agent (No Tools)")
    print("=" * 60)

    from orchestral import Agent

    print("\n1. Creating Agent with Ollama...")
    agent = Agent(llm=llm, tools=[])
    print("   ✓ Agent created")

    print("\n2. Running agent with test message...")
    agent_test_prompt = "Say 'Agent test successful!' and nothing else."
    print(f"\n   User prompt:")
    print(f"   > {agent_test_prompt}")

    start = time.time()

    agent_response = agent.run(agent_test_prompt)

    elapsed = time.time() - start
    print(f"\n   ✓ Agent response received in {elapsed:.2f}s")

    print("\n3. Agent response:")
    # agent.run() returns a Message, not a Response
    print(f"   {agent_response.text}")

    print("\n" + "=" * 60)
    print("✓ SUCCESS: Agent + Ollama working correctly!")
    print("=" * 60)
    print("\nOllama integration is fully functional.")
    print("If the web UI isn't working, it's a WebSocket/browser issue,")
    print("NOT an Ollama configuration issue.")
    print("=" * 60)

except ConnectionError as e:
    print("\n" + "=" * 60)
    print("✗ CONNECTION ERROR:")
    print(f"   {e}")
    print("=" * 60)
    print("\nTroubleshooting:")
    print("  1. Verify the model is available: ollama list")
    print("  2. Pull the model if needed: ollama pull qwen2.5:3b")
    print(f"  3. Check config.py settings (ollama_model, ollama_host)")
    print("=" * 60)
    sys.exit(1)

except Exception as e:
    print("\n" + "=" * 60)
    print("✗ ERROR:")
    print(f"   {e}")
    print("=" * 60)
    print("\nThis may indicate:")
    print("  - Model not available (run: ollama list)")
    print("  - Configuration issue in config.py")
    print("  - Network connectivity problem")
    print("\nFull traceback:")
    import traceback
    traceback.print_exc()
    print("=" * 60)
    sys.exit(1)
