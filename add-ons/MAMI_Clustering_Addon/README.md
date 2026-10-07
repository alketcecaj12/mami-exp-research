# MAMI visual and thematic clustering pilot

Research question: What recurring visual motifs and themes appear in this meme corpus, and how do they overlap with dataset misogyny labels?

This is an exploratory extension of your MAMI research project. It uses frozen EmbeddingGemma 2 representations, not Qwen or LoRA. It does not modify existing code, configurations, adapters or classification results. Actual EmbeddingGemma inference must first be checked on your Mac. The model was newly released when this addon was prepared.

## 1. Separate environment

Extract MAMI_Clustering_Addon into Downloads, next to mami_research. From Terminal:

```bash
cd /Users/alket/Downloads/mami_research
python3 -m venv .venv-clustering
source .venv-clustering/bin/activate
python -m pip install --upgrade pip
python -m pip install --upgrade -r ../MAMI_Clustering_Addon/requirements.txt
python -m pip freeze > ../MAMI_Clustering_Addon/environment_mac.txt
```

The separate environment keeps the existing Qwen environment intact. Initial dependencies follow the new model's current library support rather than guessing minimum versions. Keep environment_mac.txt for reproducibility; do not upgrade during a run. The script also records the embedding dependency versions and pins the model to the resolved Hugging Face commit on its first run.

## 2. Smoke test: 80 randomly selected memes, four clusters

```bash
caffeinate -i python ../MAMI_Clustering_Addon/cluster_memes.py \
  --config configs/full_corrected.yaml \
  --output outputs_clustering_smoke80 \
  --limit 80 --k 4

open outputs_clustering_smoke80/index.html
```

The first run downloads model files. Inference is local; images are not sent to a cloud inference service. It selects MPS when available, otherwise CPU, and uses float32. It disables the unused audio encoder. Image-only and combined image/transcription vectors are extracted separately. The smoke sample is random with seed 42 and is not stratified using labels.

Send embedding_run.json, clustering_run.json, cluster_metrics.csv and modality_comparison.csv, plus a screenshot of one gallery. If an error occurs, send the full traceback and environment_mac.txt; do not alter the working Qwen environment to fix this addon. If MPS is unsupported, a separate output folder with --device cpu can test CPU execution.

## 3. Full exploratory corpus, after the smoke test succeeds

```bash
caffeinate -i python ../MAMI_Clustering_Addon/cluster_memes.py \
  --config configs/full_corrected.yaml \
  --output outputs_clustering_full \
  --k 10 20 30

open outputs_clustering_full/index.html
```

All three corrected splits are included by default. With your current data, expect 10,998 retained records (8,999 train + 999 validation + 1,000 test), unless data changed. The existing split loader is reused, including its exact-content overlap removal. Same-split duplicates and near-duplicates can remain; their influence should be examined during interpretation. Record counts are counts in this curated corpus, not prevalence across platforms or the internet.

Including test content here is intentional exploratory corpus mapping, not classifier evaluation. If you want to exclude it, use --splits train validation with a separate output folder. Corpus-wide discoveries must not later be represented as independent confirmation on an untouched test corpus.

## 4. Cache and repeat analyses

The script caches each successful embedding. If interrupted, rerun the same command and output directory in the same environment. Completed cached embeddings are reused, and remaining items are encoded. Do not run two commands into the same output folder simultaneously.

To try a different number of clusters without loading the model:

```bash
python ../MAMI_Clustering_Addon/cluster_memes.py \
  --config configs/full_corrected.yaml \
  --output outputs_clustering_full \
  --cluster-only --k 15 25
```

The corpus must match the saved manifest, including paths, labels, image/text hashes and sampling settings. A changed corpus requires a new output folder. Existing review worksheets are preserved. The top-level index and run summary describe the most recent clustering command; earlier k-specific galleries remain in their folders.

## Method and interpretation

1. Embeddings: google/embeddinggemma-2, full 768 dimensions, float32, unit normalized. No fine-tuning. EXIF orientation is applied and images are converted to RGB. Image-only input has one image placeholder; combined input adds the documented clustering prefix and supplied transcription. Image-only still includes visible text in the meme.
2. Clustering: ordinary KMeans on normalized vectors, n_init=10. This is not spherical KMeans. Cluster counts 10/20/30 are exploratory resolutions, not known numbers of themes. Every example is assigned; KMeans does not automatically identify noise or overlapping themes.
3. Stability check: repeat KMeans with seed+1 and report adjusted Rand index. This only assesses sensitivity to initialization at a fixed k. It does not establish stability under resampling or prove semantic validity.
4. Structure check: cosine silhouette on at most 2,000 sampled examples. This describes separation in embedding space, not thematic correctness. Do not select k from this score alone or from misogyny-label purity.
5. Modality comparison: adjusted Rand index between the image-only and combined partitions at each k. Cluster numbers are arbitrary and must not be matched directly across modes or k.
6. Interpretation: each gallery shows five nearest-centroid examples first, up to three peripheral examples, up to three random examples, then all remaining members. Label values and composition are hidden behind expandable details. No label is supplied to the encoder or clustering algorithm.

For each cluster, complete human_review.csv with a provisional theme, visual motifs, possible template/watermark effects, interpretation, counterexamples and confidence. Look for hostility, endorsement, quotation, criticism, satire and ambiguity without assuming similar imagery has the same stance. Avoid calling a cluster misogynous merely because it contains many positive labels. Review representative and peripheral cases. A second independent reviewer can help challenge interpretations.

No themes are automatically named, and no 2D map is used to determine clusters. No audience-reaction analysis is performed: that requires comments, replies, timestamps, platform identifiers and conversational context absent from the current dataset. A later cross-platform study needs a separately designed and appropriately sourced corpus.

## Files

- index.html and atlas_MODE_kK/: offline gallery; open the root index in your browser.
- thumbnails/: local previews used by all galleries. Keep the output folder together.
- atlas_MODE_kK/human_review.csv: blank interpretation worksheet; edit a copy or preserve it in place.
- assignments_MODE_kK.csv: every record's cluster plus metadata and original dataset label.
- embeddings_image.npy and embeddings_multimodal.npy: matrices aligned to corpus.json.
- cluster_metrics.csv: cluster counts, silhouette and initialization stability.
- modality_comparison.csv: agreement between image and combined partitions.
- corpus.json, data_report.json, embedding_run.json, clustering_run.json: provenance and completion information.
- cache/: reusable per-item embeddings.

The full galleries contain corpus images and transcriptions. Keep them local unless dataset sharing terms permit redistribution. Cluster assignments and interpretation notes can be shared separately.

## Verification

Syntax, finite-vector validation, cluster output generation, HTML escaping, representative selection and preservation of review worksheets were checked on a small synthetic corpus. No real model inference or MPS execution was performed in the hosted development environment. The 80-example smoke test is required to establish compatibility on your Mac. Synthetic testing does not validate research quality.

Model interface and methodology reference: https://huggingface.co/google/embeddinggemma-2
