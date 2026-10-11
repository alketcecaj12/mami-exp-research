"""Regression tests for reorganized package entry points."""


def test_data_public_import():
    from mami.data import load_splits
    from mami.data.loader import load_splits as implementation
    assert load_splits is implementation


def test_training_legacy_import():
    from mami.train import train_lora
    from mami.training.lora import train_lora as implementation
    assert train_lora is implementation
