"""Clustering corpus module imports without model downloads."""
from mami.clustering.corpus import load_corpus, digest
from mami.data.loader import load_splits


def test_clustering_corpus_imports():
    assert callable(load_corpus)
    assert callable(load_splits)
    assert digest(b"abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_unit_vector_normalizes_and_rejects_invalid_values():
    import numpy as np
    import pytest
    from mami.clustering.cluster_memes import unit_vector
    vector = np.zeros(768, dtype=np.float32)
    vector[0] = 4.0
    result = unit_vector(vector)
    assert result.shape == (768,)
    assert result[0] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        unit_vector(np.ones(12))
    with pytest.raises(ValueError):
        unit_vector(np.full(768, np.nan))


def test_clustering_cli_help_without_loading_model():
    import subprocess
    import sys
    result = subprocess.run([sys.executable, "-m", "mami.clustering.cluster_memes", "--help"],
                            capture_output=True, text=True, check=True)
    assert "--cluster-only" in result.stdout
    assert "--splits" in result.stdout
