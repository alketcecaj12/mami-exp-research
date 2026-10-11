"""Binary Task A inference; model loading is shared across tasks."""
from ..models.registry import load_model


def load_classifier(config, adapter=None, training=False):
    return load_model(config, adapter=adapter, training=training)
