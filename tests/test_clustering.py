"""Clustering corpus module imports without model downloads."""
from mami.clustering.corpus import load_corpus, digest
from mami.data.loader import load_splits


def test_clustering_corpus_imports():
    assert callable(load_corpus)
    assert callable(load_splits)
    assert digest(b"abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
