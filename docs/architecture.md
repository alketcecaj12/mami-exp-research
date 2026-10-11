# Repository architecture

This research repository separates three workflows while reusing dataset and model infrastructure.

| Location | Responsibility |
| --- | --- |
| `src/mami/binary/` | MAMI Task A binary prompts, inference, evaluation and baselines |
| `src/mami/multilabel/` | MAMI Task B subtype prompts, parsing, inference, evaluation and offline re-evaluation |
| `src/mami/models/` | Shared lazy backend registry and Gemma 3 / Qwen implementations |
| `src/mami/data/` | Split-aware image loading, validation and content-overlap checking |
| `src/mami/training/` | LoRA training and resumable checkpoints |
| `src/mami/clustering/` | Label-blind embeddings, KMeans clustering, stability metrics and human-review atlases |
| `src/mami/cli.py` | Main `mami` command entry point for classification and training |
| `configs/{binary,multilabel,training,clustering}/` | Workflow-specific YAML copies and clustering instructions |
| `tests/` | Import, parsing, evaluation, configuration and CLI smoke tests |

## Compatibility

Legacy YAML files in `configs/` remain usable; workflow-specific files are currently copies, not aliases. The older `mami.vlm`, `mami.gemma3` and `mami.train` module imports remain supported through wrappers. The dataset import `from mami.data import load_splits` is preserved by the `data/` package.

## Reproducibility and boundaries

Use the same corpus splits, model revision, prompts, and evaluation settings for fair comparisons. Training and evaluation are distinct workflows; clustering is exploratory and does not train on reference labels. Clustering outputs include image-only and multimodal embeddings and human-review galleries. These outputs do **not** provide audience reactions or cross-platform user interaction data. The current automated tests are not substitutes for full model or embedding experiments.

## Quick checks

```bash
python -m pytest tests/ -v
python -m mami.clustering.cluster_memes --help
```

See `configs/clustering/README.md` for a small clustering run. See the root `README.md` for the original corrected binary research experiment and its reported results.
