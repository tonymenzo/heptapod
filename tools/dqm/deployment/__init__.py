__all__ = ["DQMRealtimeMonitorTool", "DQMAlertDispatcherTool", "DQMAdaptiveThresholdTool"]

def __getattr__(name):
    import importlib
    _map = {
        "DQMRealtimeMonitorTool":   (".realtime_monitor",   "DQMRealtimeMonitorTool"),
        "DQMAlertDispatcherTool":   (".alert_dispatcher",   "DQMAlertDispatcherTool"),
        "DQMAdaptiveThresholdTool": (".adaptive_threshold", "DQMAdaptiveThresholdTool"),
    }
    if name in _map:
        mod_rel, cls = _map[name]
        m = importlib.import_module(mod_rel, package=__name__)
        return getattr(m, cls)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
