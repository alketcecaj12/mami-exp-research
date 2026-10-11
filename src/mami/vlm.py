"""Backward-compatible import for the Qwen2.5-VL backend."""
from .models.qwen import Classifier
from .binary.prompts import PROMPT, LABELS, parse_label

__all__ = ["Classifier", "PROMPT", "LABELS", "parse_label"]
