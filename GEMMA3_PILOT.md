# Gemma 3 MAMI pilot (experimental)

This branch adds Gemma 3 4B to the existing `mami` CLI without replacing Qwen.

## Prerequisites

Accept the Gemma model license at https://huggingface.co/google/gemma-3-4b-it and authenticate using `hf auth login`.

```bash
git fetch origin
git switch feature/gemma3-pilot
python -m pip install -e '.[vlm,train]'
mami inspect --config configs/gemma3_pilot.yaml
mami download --config configs/gemma3_pilot.yaml
```

## Pilot

First confirm 1-image inference, then evaluate 100 unchanged test examples:

```bash
mami vlm --config configs/gemma3_pilot.yaml --split test --limit 1
mami vlm --config configs/gemma3_pilot.yaml --split test --limit 100
```

Output: `outputs_gemma3_pilot/zero_shot_multimodal_test.csv` and `vlm_test_metrics.json`.

Then train on 100 training examples (not the test images):

```bash
mami train --config configs/gemma3_pilot.yaml --limit 100
mami vlm --config configs/gemma3_pilot.yaml --split test --limit 100 --adapter outputs_gemma3_pilot/lora_adapter
```

The existing CLI overwrites the common `vlm_test_metrics.json` when running a second evaluation. Copy/rename the zero-shot metrics before evaluating the adapter. The raw prediction files remain separate.

## Research cautions

- This is an *unvalidated* implementation. No Gemma inference or training was executed on the user's Mac.
- Gemma 3 4B multimodal float32 LoRA on MPS may exceed memory; stop after the one-example smoke test if memory pressure is excessive.
- The existing training code checks chat-template prefix equality; Gemma templates may require adjustment to answer masking before training works.
- The current `--limit 100` selects the first 100 rows, not a stratified sample. Use the same test filenames for the Qwen comparison.
- The Qwen training implementation currently includes attention projections outside the vision tower, including multimodal projector layers if names match. Verify trainable parameters before interpreting LoRA comparisons.
- The prompt is shared with Qwen, but tokenizer and chat templates differ.
- Zero-shot evaluation should be completed and archived before LoRA training.
