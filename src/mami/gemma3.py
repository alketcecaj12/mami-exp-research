"""Backward-compatible import for the Gemma 3 backend."""
from .models.gemma3 import Classifier
from .binary.prompts import PROMPT, parse_label

__all__ = ["Classifier", "PROMPT", "parse_label"]
