"""
Compatibility shim for BaseTool, RuntimeField, StateField.

Import priority:
  1. orchestral_ai  (pip-installed)
  2. Bundled orchestral/ package in THIS repo
  3. Pure stdlib fallback (no validation, but fully functional)

CRITICAL: The stub fallback is ONLY activated when we detect a bundled
(repo-local) implementation, not when a real pip install is present.
This prevents overriding a real orchestral install with no-op stubs.
"""
import sys
import os

_repo_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

def _is_bundled_install(module) -> bool:
    """Return True if `module` lives inside this repo (i.e. is the bundled stub)."""
    try:
        mod_file = getattr(module, "__file__", None) or ""
        if not mod_file:
            return False
        common = os.path.commonpath([os.path.abspath(mod_file), _repo_root])
        return common == _repo_root
    except (ValueError, OSError):
        return False


try:
    from orchestral_ai import BaseTool, RuntimeField, StateField  # type: ignore
    # Successfully imported a real pip install — use it as-is, no stubs
except ImportError:
    try:
        import orchestral  # type: ignore
        from orchestral.tools.base.tool import BaseTool  # type: ignore
        from orchestral.tools.base.field_utils import RuntimeField, StateField  # type: ignore

        if _is_bundled_install(orchestral):
            # The bundled stub is active — replace with no-op shims so tools work
            # without a real orchestral install
            def RuntimeField(description="", default=None, **kw):  # noqa: F811
                return default

            def StateField(description="", default=None, **kw):    # noqa: F811
                return default

            class BaseTool:                                         # noqa: F811
                """Minimal BaseTool: kwargs become instance attributes."""
                def __init__(self, **kwargs):
                    for k, v in kwargs.items():
                        setattr(self, k, v)
                def run(self):
                    raise NotImplementedError
        # else: real orchestral pip install found — keep imported classes intact

    except ImportError:
        # Pure stdlib fallback — no external deps at all
        class BaseTool:                                             # noqa: F811
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)
            def run(self):
                raise NotImplementedError

        def RuntimeField(description="", default=None, **kw):      # noqa: F811
            return default

        def StateField(description="", default=None, **kw):        # noqa: F811
            return default

__all__ = ["BaseTool", "RuntimeField", "StateField"]
