"""Binary inference entry point, preserving existing model adapters."""


def load_classifier(config, adapter=None, training=False):
    if config["model"].get("family") == "gemma3":
        from ..gemma3 import Classifier
    else:
        from ..vlm import Classifier
    return Classifier(config, adapter=adapter, training=training)
