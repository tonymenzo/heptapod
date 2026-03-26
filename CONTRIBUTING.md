# Contributing to HEPTAPOD

This guide focuses on contributing **custom tools** to extend the framework's capabilities. For general contributions (bug fixes, documentation, etc.), please see our [GitHub Issues](https://github.com/tonymenzo/heptapod/issues).

---

## Contributing Tools

HEPTAPOD is designed to be extensible through custom tools. This guide provides comprehensive instructions for developing and contributing new physics tools to the framework.

### Table of Contents

- [Tool Architecture Overview](#tool-architecture-overview)
- [Tool Structure Template](#tool-structure-template)
- [Critical Requirements](#critical-requirements)
  - [Field Definitions](#1-field-definitions)
  - [Error Handling](#2-error-handling-with-format_error)
  - [Path Safety and Sandboxing](#3-path-safety-and-sandboxing)
  - [Output Format](#4-output-format)
  - [Schema Versioning](#5-schema-versioning)
- [Testing Your Tool](#testing-your-tool)
- [Best Practices](#best-practices)
- [Submitting Your Tool](#submitting-your-tool)

---

## Tool Architecture Overview

All HEPTAPOD tools inherit from Orchestral AI's `BaseTool` class and follow a standardized structure with three key components:

1. **Runtime Fields**: Tool inputs provided by the LLM agent at runtime
2. **State Fields**: Configuration parameters injected from the agent's state
3. **Execution Logic**: The `_run()` method that performs the tool's operation

This architecture enables the LLM agent to:

- Understand when and how to use each tool (via docstrings)
- Provide runtime parameters dynamically
- Maintain consistent configuration across tool calls
- Receive structured feedback for decision-making

---

## Tool Structure Template

Here's the basic structure every tool should follow:

```python
"""
# your_tool.py is a part of the HEPTAPOD package.
# Copyright (C) 2025 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.
"""
import json, os
from typing import Optional
from orchestral.tools.base.tool import BaseTool
from orchestral.tools.base.field_utils import RuntimeField, StateField

SCHEMA_VERSION = "your-schema-1.0"  # Define a schema version for your output format

class YourCustomTool(BaseTool):
    """
    Brief description of what your tool does (1-2 sentences).

    This docstring is CRITICAL - it is ingested by the LLM agent to understand
    when and how to use this tool. Be clear, concise, and specific about:
    - What the tool accomplishes
    - Required inputs and their formats
    - Output format and structure
    - Any important constraints or behaviors

    Inputs (runtime):
      - input_param: description of what this parameter does
      - optional_param: (optional) description and default behavior

    State:
      - config_param: description of configuration parameter
      - base_directory: sandbox root for file operations

    Behavior:
      1. Step-by-step description of what the tool does
      2. Include any important side effects or file operations
      3. Describe error conditions

    Output (JSON):
      {
        "status": "ok",
        "output_field": "<description>",
        "another_field": <type>
      }

    Errors:
      Returns self.format_error() on failures including:
        - error_condition_1
        - error_condition_2
    """

    # ======================== Runtime fields ======================== #
    input_param: str = RuntimeField(
        description="Clear description for LLM - this is ingested by the agent"
    )
    optional_param: Optional[int] = RuntimeField(
        default=None,
        description="Optional parameter with default value"
    )
    # ================================================================ #

    # ========================= State fields ========================= #
    config_param: str = StateField(
        description="Configuration parameter from agent state"
    )
    base_directory: str = StateField(
        description="Base sandbox directory for file operations"
    )
    # ================================================================ #

    def _setup(self):
        """Optional setup method for validation and initialization."""
        self.base_directory = os.path.abspath(self.base_directory)
        if not os.path.isdir(self.base_directory):
            raise ValueError(f"Base directory does not exist: {self.base_directory}")

    def _run(self) -> str:
        """
        Main execution method. Must return a JSON string.

        Returns:
            JSON string with status and results, or error message via format_error()
        """
        try:
            self._setup()
        except Exception as e:
            return self.format_error(
                error="Setup Error",
                reason=str(e)
            )

        # Your tool logic here
        try:
            # Perform your tool's operations
            # ...

            # Return success as JSON
            result = {
                "status": "ok",
                "output_field": "value",
            }
            return json.dumps(result, separators=(",", ":"), ensure_ascii=False)

        except Exception as e:
            return self.format_error(
                error="Execution Error",
                reason=str(e),
                suggestion="Check input parameters and try again"
            )
```

---

## Critical Requirements

### 1. Field Definitions

#### Runtime Fields (provided by LLM at execution time)

- Use `RuntimeField()` for all inputs the agent provides
- Always include clear, descriptive `description` parameters
- **The descriptions are ingested by the LLM** - be explicit and precise
- Support optional parameters with `Optional[Type]` and `default=` values

#### State Fields (injected from agent configuration)

- Use `StateField()` for configuration parameters
- Common state fields: `base_directory`, tool paths (e.g., `mg5_path`, `feynrules_path`), API keys
- These are set once during agent initialization, not per-call

**Example from [mg5.py](tools/mg5/mg5.py#L315-328):**

```python
# Runtime fields
data_dir: str = RuntimeField(
    description="Relative output directory for dataset, e.g. 'data/mg_run001'"
)
command_card: str = RuntimeField(
    description="Relative path (inside base_directory) to MG5 command card template"
)
nevents: Optional[int] = RuntimeField(
    default=None,
    description="Number of events to generate (replaces 'set nevents')"
)

# State fields
mg5_path: str = StateField(
    description="Absolute path to top-level MG5_aMC install dir containing bin/mg5_aMC"
)
base_directory: str = StateField(
    description="Base sandbox root for all file ops"
)
```

---

### 2. Error Handling with `format_error()`

**CRITICAL**: All errors must be reported using `self.format_error()`. This method returns a standardized error message that the LLM can understand and act upon.

#### Standard Signature

```python
self.format_error(
    error="Error Type",           # Brief category (e.g., "File Not Found", "Access Denied")
    reason="Detailed explanation", # What went wrong
    context="Additional info",     # Optional: paths, values, etc.
    suggestion="How to fix it"    # Optional: actionable fix for the user/agent
)
```

#### Common Error Patterns

**File not found:**
```python
return self.format_error(
    error="File Not Found",
    reason="Input file does not exist",
    context=f"path={self.input_file}",
    suggestion="Verify the file path and try again"
)
```

**Access denied / path traversal:**
```python
return self.format_error(
    error="Access Denied",
    reason="Path escapes base_directory",
    context=self.unsafe_path,
    suggestion="Use relative paths inside base_directory"
)
```

**Missing dependency:**
```python
return self.format_error(
    error="Dependency Missing",
    reason="Required package not found",
    suggestion="Install with: pip install package-name",
    context=str(e)
)
```

**External tool failure:**
```python
return self.format_error(
    error="Tool Execution Failed",
    reason="External command returned non-zero exit code",
    context=f"exit_code={result.returncode}",
    suggestion="Check log file for details"
)
```

**Reference examples:**
- [mg5.py:343-467](tools/mg5/mg5.py#L343-467) - Comprehensive error handling
- [conversions.py:66-138](tools/analysis/conversions.py#L66-138) - Data conversion errors
- [feynrules.py:71-145](tools/feynrules/feynrules.py#L71-145) - External dependency errors

---

### 3. Path Safety and Sandboxing

**Security is paramount.** All file operations must be confined to `base_directory` to prevent path traversal attacks.

#### Safe Path Helper

```python
def _safe_path(self, rel_or_abs: str) -> Optional[str]:
    """
    Ensures that the path is within the allowed base directory.

    Args:
        rel_or_abs: Relative or absolute path to validate

    Returns:
        Absolute path if safe, None if path escapes base_directory
    """
    if not rel_or_abs:
        return None
    full = os.path.abspath(os.path.join(self.base_directory, rel_or_abs))
    if not full.startswith(self.base_directory + os.sep) and full != self.base_directory:
        return None
    return full
```

#### Usage

```python
input_path = self._safe_path(self.input_file)
if input_path is None:
    return self.format_error(
        error="Access Denied",
        reason="Path escapes base_directory",
        suggestion="Use relative paths inside base_directory"
    )

if not os.path.exists(input_path):
    return self.format_error(
        error="File Not Found",
        reason="Input file does not exist",
        context=self.input_file
    )
```

**Reference implementation:** [mg5.py:77-91](tools/mg5/mg5.py#L77-91)

---

### 4. Output Format

Tools must return **JSON strings**, not Python dictionaries.

#### Correct Output

```python
# Correct - return JSON string
result = {
    "status": "ok",
    "output": "value"
}
return json.dumps(result, separators=(",", ":"), ensure_ascii=False)
```

```python
# Incorrect - do NOT return dict directly
return {"status": "ok"}  # Wrong! Must be JSON string
```

#### Success Response Format

```python
{
  "status": "ok",
  "output_path": "relative/path/to/output.json",
  "n_events": 1000,
  "metadata": {
    "created_utc": "2025-01-15T10:30:00+00:00",
    "tool": "YourCustomTool",
    "version": "1.0"
  }
}
```

#### Error Response

Error responses are automatically formatted by `format_error()` - you don't need to construct them manually.

---

### 5. Schema Versioning

Define a schema version constant for your output format to enable version tracking and backward compatibility.

```python
SCHEMA_VERSION = "evtjsonl-1.0"  # Or "tool-1.0", "jets-1.0", etc.

# Include in output metadata
result = {
    "schema": SCHEMA_VERSION,
    "status": "ok",
    # ... other fields
}
```

**Existing schemas:**

- `evtjsonl-1.0`: Event data (particles with four-momenta)
- `tool-1.0`: General tool metadata

---

## Testing Your Tool

**Testing is strongly recommended** (though not mandatory). Well-tested tools are more likely to be accepted and easier to maintain.

### Test File Structure

Create a test file in your tool's directory following this pattern:

```python
#!/usr/bin/env python3
"""
# test_your_tool.py is a part of the HEPTAPOD package.
# Copyright (C) 2025 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
"""
import os, sys, json, argparse, shutil
from pathlib import Path

# Path setup
SCRIPT_PATH = Path(__file__).resolve()
TOOL_DIR = SCRIPT_PATH.parent
REPO_ROOT = TOOL_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Import your tool
from tools.your_module import YourCustomTool

# Import config if needed
try:
    from config import external_dependency_path
except ImportError:
    external_dependency_path = None

def test_basic_functionality():
    """Test basic functionality of your tool."""
    print(">> Testing basic functionality...\n")

    tool = YourCustomTool(
        input_param="test_value",
        config_param=external_dependency_path,
        base_directory=str(TOOL_DIR / "test_files")
    )

    tool._setup()
    result_str = tool._run()

    try:
        result = json.loads(result_str)
    except json.JSONDecodeError:
        print(f"[✗] Tool returned invalid JSON: {result_str}\n")
        return None

    if result.get("status") != "ok":
        print(f"[✗] Tool failed: {result.get('reason', result)}\n")
        return None

    print("[✓] Basic functionality test passed\n")
    return result

def test_error_handling():
    """Test that errors are properly formatted."""
    print(">> Testing error handling...\n")

    tool = YourCustomTool(
        input_param="nonexistent_file.dat",
        config_param=external_dependency_path,
        base_directory="/tmp"
    )

    tool._setup()
    result_str = tool._run()

    # Error messages should contain "error" or specific error indicators
    assert "error" in result_str.lower() or "not found" in result_str.lower()
    print("[✓] Error handling test passed\n")

def cleanup_test_files():
    """Remove test-generated files and directories."""
    print("\n>> Cleaning up test files...\n")

    cleanup_dirs = [
        TOOL_DIR / "test_files" / "output",
        # Add other test directories
    ]

    cleaned = 0
    for dir_path in cleanup_dirs:
        if dir_path.exists():
            try:
                shutil.rmtree(dir_path)
                print(f"[✓] Removed: {dir_path.name}")
                cleaned += 1
            except Exception as e:
                print(f"[⚠] Failed to remove {dir_path.name}: {e}")

    if cleaned == 0:
        print("[i] No test files to clean up")
    else:
        print(f"\n[✓] Cleaned up {cleaned} test directory(ies)\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run tests for YourCustomTool")
    parser.add_argument(
        "--keep-files",
        action="store_true",
        help="Keep test-generated files after tests complete (useful for debugging)"
    )
    args = parser.parse_args()

    # Run tests
    all_passed = True

    try:
        result = test_basic_functionality()
        if result is None:
            all_passed = False
    except Exception as e:
        print(f"[✗] test_basic_functionality failed: {e}\n")
        all_passed = False

    try:
        test_error_handling()
    except Exception as e:
        print(f"[✗] test_error_handling failed: {e}\n")
        all_passed = False

    # Cleanup unless --keep-files flag is set
    if not args.keep_files:
        cleanup_test_files()
    else:
        print("\n[i] Keeping test files (--keep-files flag set)\n")

    # Exit with appropriate code
    if all_passed:
        print("[✓] All tests passed!\n")
        sys.exit(0)
    else:
        print("[✗] Some tests failed!\n")
        sys.exit(1)
```

### Integration with Test Runner

To integrate your tests into [test_runner.py](test_runner.py), add an entry to the `test_suites` dictionary (around line 342):

```python
test_suites = {
    # ... existing tests ...
    "your_tool": {
        "script": REPO_ROOT / "tools" / "your_module" / "test_your_tool.py",
        "description": "Your tool description (what it tests)"
    },
}
```

Users can then run your tests with:
```bash
python test_runner.py --only your_tool
```

### Example Test Files

Reference implementations showing different testing patterns:

- [tools/mg5/test_mg5.py](tools/mg5/test_mg5.py) - Comprehensive integration tests with cleanup
- [tools/analysis/test_conversions.py](tools/analysis/test_conversions.py) - Data conversion and schema validation
- [tools/feynrules/test_feynrules.py](tools/feynrules/test_feynrules.py) - External dependency tests (WolframScript)
- [tools/pythia/test_pythia.py](tools/pythia/test_pythia.py) - Event generation and parameter scans

---

## Best Practices

### 1. Docstrings are for the LLM

The class docstring is the primary way the agent learns about your tool. Make it:

- **Clear**: No ambiguous language
- **Concise**: Focus on what the agent needs to know
- **Specific**: Include exact parameter formats, output structure, and behavior
- **Complete**: Document all inputs, outputs, and error conditions

### 2. Use Descriptive Field Names

- Good: `input_lhe_path`, `output_jsonl_path`, `n_events`
- Bad: `input`, `output`, `file`, `n`

### 3. Validate Early

Check for missing/invalid inputs in `_setup()` before expensive operations:

```python
def _setup(self):
    self.base_directory = os.path.abspath(self.base_directory)
    if not os.path.isdir(self.base_directory):
        raise ValueError(f"Base directory does not exist: {self.base_directory}")

    # Validate required external dependencies
    if not self.external_tool_path:
        raise ValueError("external_tool_path is required")
```

### 4. Provide Helpful Errors

Include actionable suggestions in your `format_error()` calls:

```python
# Less helpful
return self.format_error(
    error="Error",
    reason="Something went wrong"
)

# More helpful
return self.format_error(
    error="Dependency Missing",
    reason="wolframscript not found at specified path",
    context=f"path={self.wolframscript_path}",
    suggestion="Install Mathematica or update wolframscript_path in config.py"
)
```

### 5. Preserve Provenance

Log input parameters, versions, and timestamps in your output for reproducibility:

```python
result = {
    "schema": SCHEMA_VERSION,
    "status": "ok",
    "created_utc": datetime.datetime.utcnow().replace(
        tzinfo=datetime.timezone.utc
    ).isoformat(),
    "inputs": {
        "param1": self.param1,
        "param2": self.param2,
    },
    "outputs": {
        "output_path": output_path,
    },
    "tool": "YourCustomTool",
    "version": "1.0"
}
```

### 6. Use Relative Paths

Accept and return paths relative to `base_directory` for portability:

```python
# Input: relative path
input_abs = self._safe_path(self.input_file)  # Convert to absolute

# Output: convert back to relative
output_rel = os.path.relpath(output_abs, self.base_directory)
result = {
    "output_path": output_rel  # Store relative path
}
```

### 7. Handle Edge Cases

Consider and test:
- Empty files
- Zero events
- Missing optional dependencies
- Unusual but valid inputs
- Concurrent access to shared resources

### 8. Progress Indicators

Use `tqdm` for long-running operations to provide user feedback in the terminal:

```python
from tqdm import tqdm
import sys

TQDM_CONFIG = {
    'file': sys.stderr,
    'ncols': 80,
    'leave': True,
    'dynamic_ncols': False
}

for item in tqdm(items, desc="Processing", unit="item", **TQDM_CONFIG):
    # Process item
    pass
```

See [conversions.py:97](tools/analysis/conversions.py#L97) for examples.

### 9. Follow Schema Conventions

- Use `evtjsonl-1.0` for event data (particles with four-momenta)
- Include `schema` and `status` fields in all outputs
- Return JSON strings, not Python dicts
- Use consistent field naming (`n_events` not `numEvents` or `num_events`)

### 10. Clean Up Test Files

Test files should:
- Support `--keep-files` flag for debugging
- Clean up by default to avoid clutter
- Remove ALL generated files (check for hidden/temp files)
- Provide clear feedback about what was cleaned

---

## Tool Categories

HEPTAPOD organizes tools into categories based on their function:

### Model Generation
**Location:** [tools/feynrules/](tools/feynrules/)
**Purpose:** BSM model specification and UFO generation
**Examples:** `FeynRulesToUFOTool`

### Event Generation
**Location:** [tools/mg5/](tools/mg5/), [tools/pythia/](tools/pythia/)
**Purpose:** Parton-level and hadron-level event production
**Examples:** `MadGraphFromRunCardTool`, `PythiaFromRunCardTool`

### Analysis
**Location:** [tools/analysis/](tools/analysis/)
**Purpose:** Data conversion, kinematics, reconstruction, cuts
**Examples:** `LHEToJSONLTool`, `EventJSONLToNumpyTool`, `InvariantMassTool`

### Custom Categories
Feel free to propose new categories as needed. Each category should have:
- Clear purpose and scope
- Dedicated subdirectory under `tools/`
- `__init__.py` exposing the main tools
- Optional `README.md` with category-specific documentation

---

## Submitting Your Tool

### Before Submitting

1. Tool follows the structure template above
2. All critical requirements met (fields, errors, safety, output format)
3. Docstring clearly explains what, how, and when to use the tool
4. Tests written and passing (recommended)
5. Code follows existing style (see [tools/mg5/mg5.py](tools/mg5/mg5.py) for reference)
6. External dependencies documented in tool docstring

### Submission Process

1. **Create a new directory** under `tools/` for your tool category (if needed)

2. **Implement your tool** following the structure above:
   ```
   tools/your_category/
   ├── __init__.py          # Export your tool classes
   ├── your_tool.py         # Tool implementation
   ├── test_your_tool.py    # Tests (recommended)
   └── test_files/          # Test data (if needed)
       ├── input/
       └── expected_output/
   ```

3. **Add tests** (recommended) and integrate with [test_runner.py](test_runner.py)

4. **Document external dependencies**:
   - In tool docstring
   - In [README.md](README.md) (update Installation section if needed)
   - In [config.py](config.py) if configuration paths are needed

5. **Update documentation**:
   - Add entry to [tools/README.md](tools/README.md) with usage examples
   - Update main [README.md](README.md) if adding a major feature

6. **Submit a pull request** with:
   - Clear title describing the tool
   - Description of what the tool does and why it's useful
   - Example usage (input/output)
   - Any external dependencies and installation instructions
   - Test results (`python test_runner.py --only your_tool`)

### PR Checklist

- [ ] Tool implementation complete
- [ ] Tests written and passing (or explanation if skipped)
- [ ] Documentation updated (tool docstring, README, etc.)
- [ ] External dependencies documented
- [ ] Code follows HEPTAPOD style conventions
- [ ] No security issues (path traversal, injection, etc.)
- [ ] Example usage provided in PR description

---

## Questions?

- **Reference implementations**: Check existing tools in [tools/](tools/) for examples
- **Orchestral AI docs**: See [orchestral-ai.com](https://orchestral-ai.com) for `BaseTool` details
- **Get help**: Open an issue for clarification: [GitHub Issues](https://github.com/tonymenzo/heptapod/issues)