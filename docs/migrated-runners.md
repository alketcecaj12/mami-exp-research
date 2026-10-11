# Migrated experimental runners

These commands use the active package implementations, rather than the historical copies under `add-ons/`:

```bash
python -m mami.models.llava --config configs/llava.yaml --split validation --limit 8
python -m mami.models.smolvlm --config configs/smolvlm2.yaml --split validation --limit 8
python -m mami.binary.score_thresholds --help
```

LLaVA and SmolVLM remain **zero-shot comparison pipelines**, not interchangeable with the Qwen training runner. Use the model-specific configuration, downloaded weights and appropriate memory/precision settings. The threshold experiment computes normalized likelihoods for two fixed candidate answer sequences; these are **not calibrated probabilities**. Refer to the historical add-on READMEs on the archive branch for the original protocol and full commands.

These modules were migrated with minimal changes and require local smoke testing before scientific use. Run `python -m pytest tests/ -q` and the `--help` commands first. The archive branch `archive/pre-root-cleanup-20261011` preserves the original add-ons and experiment outputs.
