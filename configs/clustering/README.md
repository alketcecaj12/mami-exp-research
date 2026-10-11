# Clustering configuration

The clustering CLI reads the **dataset/model YAML** supplied via `--config`. Clustering parameters such as `--k`, `--seed`, `--splits`, and `--limit` are command-line arguments, **not YAML keys**. `dataset.yaml` is an exact copy of `../full_corrected.yaml`, provided as a workflow-specific entry point; adjust `data.root` for your machine. Do not edit the copied YAML independently without recording the change.

Run from the repository root:

```bash
python -m mami.clustering.cluster_memes --config configs/clustering/dataset.yaml --output outputs_clustering_smoke80 --splits validation --limit 80 --k 5 10 --seed 42
```

Use a **new output directory** for a different corpus or parameter sweep. After embeddings have been generated, reuse them without loading the embedding model:

```bash
python -m mami.clustering.cluster_memes --config configs/clustering/dataset.yaml --output outputs_clustering_smoke80 --splits validation --limit 80 --k 5 10 --seed 42 --cluster-only
```

Clustering uses frozen EmbeddingGemma representations, KMeans, and label-blind fitting. Human labels appear in review galleries only for descriptive inspection. `--cluster-only` requires existing embedding files for both image and multimodal modes. The script writes CSV metrics, assignments, and HTML atlases. Its output directory should not be committed to Git.
