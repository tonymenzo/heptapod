__all__ = ["DQMAutoencoderTrainTool", "DQMTransformerTrainTool", "DQMModelEvaluatorTool"]

def __getattr__(name):
    import importlib
    _map = {
        "DQMAutoencoderTrainTool": (".autoencoder_train", "DQMAutoencoderTrainTool"),
        "DQMTransformerTrainTool": (".transformer_train",  "DQMTransformerTrainTool"),
        "DQMModelEvaluatorTool":   (".model_evaluator",    "DQMModelEvaluatorTool"),
    }
    if name in _map:
        mod_rel, cls = _map[name]
        m = importlib.import_module(mod_rel, package=__name__)
        return getattr(m, cls)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
