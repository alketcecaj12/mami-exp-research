"""Registry tests do not download or initialize model weights."""
import pytest
from mami.models.registry import model_class


def test_gemma3_backend():
    from mami.gemma3 import Classifier
    assert model_class({"model": {"family": "gemma3"}}) is Classifier


def test_qwen_backend_and_default():
    from mami.vlm import Classifier
    assert model_class({"model": {"family": "qwen2_5_vl"}}) is Classifier
    assert model_class({"model": {}}) is Classifier


def test_unknown_family_fails_explicitly():
    with pytest.raises(ValueError, match="Unsupported model family"):
        model_class({"model": {"family": "mistral4"}})


def test_legacy_imports_resolve_to_new_backends():
    from mami.gemma3 import Classifier as OldGemma
    from mami.vlm import Classifier as OldQwen
    from mami.models.gemma3 import Classifier as NewGemma
    from mami.models.qwen import Classifier as NewQwen
    assert OldGemma is NewGemma
    assert OldQwen is NewQwen
