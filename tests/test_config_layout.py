"""Configuration copies preserve experiment settings and legacy paths."""
from pathlib import Path
import yaml
import pytest

PAIRS = [
    ("configs/full_corrected.yaml", "configs/training/full_corrected.yaml"),
    ("configs/mac_stable.yaml", "configs/training/mac_stable.yaml"),
    ("configs/expanded.yaml", "configs/training/expanded.yaml"),
    ("configs/gemma3_multilabel.yaml", "configs/multilabel/gemma3.yaml"),
    ("configs/qwen_multilabel.yaml", "configs/multilabel/qwen.yaml"),
    ("configs/qwen7b.yaml", "configs/binary/qwen7b.yaml"),
    ("configs/full_corrected.yaml", "configs/binary/qwen3b.yaml"),
    ("configs/full_corrected.yaml", "configs/clustering/dataset.yaml"),
]

@pytest.mark.parametrize("old,new", PAIRS)
def test_config_copies_preserve_values(old, new):
    root = Path(__file__).resolve().parents[1]
    assert yaml.safe_load((root / old).read_text()) == yaml.safe_load((root / new).read_text())
