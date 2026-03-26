"""
# __init__.py is a part of the HEPTAPOD package.
# Copyright (C) 2025 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.
"""
"""HEP event generator and analysis tools."""

# Re-export tool modules for convenient imports.
# Wrapped in try/except so missing optional dependencies (tqdm, etc.)
# don't prevent DQM tools from loading in minimal environments.
def _try_import(name):
    try:
        import importlib
        importlib.import_module(f".{name}", package=__name__)
    except Exception:
        pass

for _mod in ["feynrules", "mg5", "pythia", "sherpa", "analysis", "pdg", "inspire", "units"]:
    _try_import(_mod)

# DQM tools — imported lazily so missing physics deps (tqdm, etc.) don't block loading
def __getattr__(name):
    if name == "dqm":
        from . import dqm
        return dqm
    raise AttributeError(f"module 'tools' has no attribute {name!r}")
