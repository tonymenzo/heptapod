"""
ML4DQM — Machine Learning tools for CMS Data Quality Monitoring.
Extends the HEPTAPOD framework with a full data→train→deploy pipeline.

Tools are imported lazily so this package loads even without optional
dependencies (torch, pydantic, orchestral_ai) installed — matching the
pattern used throughout the rest of the HEPTAPOD tool suite.

Import individual tools directly:
    from tools.dqm.data.cms_dqm_fetch import CMSDQMFetchTool
    from tools.dqm.training.autoencoder_train import DQMAutoencoderTrainTool
    ...

Or import from this package (requires dependencies to be installed):
    from tools.dqm import CMSDQMFetchTool
"""

__all__ = [
    "CMSDQMFetchTool",
    "DQMPreprocessorTool",
    "DQMAutoencoderTrainTool",
    "DQMTransformerTrainTool",
    "DQMModelEvaluatorTool",
    "DQMRealtimeMonitorTool",
    "DQMAlertDispatcherTool",
    "DQMAdaptiveThresholdTool",
]


def __getattr__(name):
    """Lazy import — only load a tool class when it is actually requested."""
    _map = {
        "CMSDQMFetchTool":         ("tools.dqm.data.cms_dqm_fetch",        "CMSDQMFetchTool"),
        "DQMPreprocessorTool":     ("tools.dqm.data.dqm_preprocessor",     "DQMPreprocessorTool"),
        "DQMAutoencoderTrainTool": ("tools.dqm.training.autoencoder_train", "DQMAutoencoderTrainTool"),
        "DQMTransformerTrainTool": ("tools.dqm.training.transformer_train", "DQMTransformerTrainTool"),
        "DQMModelEvaluatorTool":   ("tools.dqm.training.model_evaluator",   "DQMModelEvaluatorTool"),
        "DQMRealtimeMonitorTool":  ("tools.dqm.deployment.realtime_monitor","DQMRealtimeMonitorTool"),
        "DQMAlertDispatcherTool":  ("tools.dqm.deployment.alert_dispatcher","DQMAlertDispatcherTool"),
    }
    if name in _map:
        import importlib
        module_path, class_name = _map[name]
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    raise AttributeError(f"module 'tools.dqm' has no attribute {name!r}")
