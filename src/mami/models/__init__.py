"""Model backends shared by Task A and Task B.

Task-specific prompts and output parsing belong in binary/ and multilabel/.
"""
from .registry import load_model, model_class

__all__ = ["load_model", "model_class"]
