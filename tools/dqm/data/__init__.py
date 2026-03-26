# Lazy imports — load only when requested
__all__ = ["CMSDQMFetchTool", "DQMPreprocessorTool"]

def __getattr__(name):
    import importlib
    _map = {
        "CMSDQMFetchTool":     (".cms_dqm_fetch",    "CMSDQMFetchTool"),
        "DQMPreprocessorTool": (".dqm_preprocessor", "DQMPreprocessorTool"),
    }
    if name in _map:
        mod_rel, cls = _map[name]
        m = importlib.import_module(mod_rel, package=__name__)
        return getattr(m, cls)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
