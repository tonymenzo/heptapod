---
name: build-tools
bundle: agent_dev
description: Implement complete HEPTAPOD tools from natural-language requests. Use when a user says "let's build a tool," asks to create, add, or write a HEPTAPOD tool, or describes a capability to turn into an executable tool. Do not use merely to run or explain an existing tool.
---

# Build HEPTAPOD tools

1. Translate the request into a concrete contract: purpose, runtime inputs,
   injected configuration, outputs, side effects, dependencies, and acceptance
   cases. Infer routine details from the repository; ask only when ambiguity
   changes scientific meaning, security, or public behavior. Then implement.
2. Read `CONTRIBUTING.md`, the nearest matching tool, its tests and
   `__init__.py`, and related `toolkit.yaml` entries. Prefer current local
   patterns over generic templates.
3. Put the implementation in the closest `tools/<area>/` package, or create the
   package structure prescribed by `CONTRIBUTING.md`. Keep the change scoped to
   the requested capability.
4. Implement the smallest suitable Orchestral `BaseTool` subclass. Define agent
   inputs with `RuntimeField`, injected settings with `StateField`, an
   LLM-facing class docstring, and `_run()` behavior that follows the repository
   contracts for validation, JSON output, schemas, and `format_error()`.
5. For file, network, or subprocess work, enforce `base_directory`, validate
   untrusted inputs, avoid shell interpolation, and return actionable errors.
6. Export the class from the package and register its module, class name,
   description, and bundle in `toolkit.yaml`. Add dependencies, bundle config,
   or new config fields only when the implementation requires them.
7. Mirror the nearest test layout. Test the success path, invalid input, and
   relevant edge or failure cases; mock costly or external services where
   practical.
8. Run the targeted tests and `tb validate`; fix failures before reporting.
   Update broader documentation only when the public capability requires it.
9. Report the implemented contract, files changed, and validation evidence.
   Commit or push only when requested.
