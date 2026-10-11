"""Lazy backend registry; importing this module never loads model weights."""


def model_class(config):
    family = config["model"].get("family", "qwen2_5_vl")
    if family == "gemma3":
        from ..gemma3 import Classifier
    elif family in ("qwen2_5_vl", "qwen2vl", "qwen"):
        from ..vlm import Classifier
    else:
        raise ValueError(f"Unsupported model family: {family!r}")
    return Classifier


def load_model(config, adapter=None, training=False):
    return model_class(config)(config, adapter=adapter, training=training)
