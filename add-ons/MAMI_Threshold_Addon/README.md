# Qwen LoRA threshold experiment

This standalone script uses your existing MAMI project and trained Qwen2.5-VL-3B adapters. It changes no installed project files, YAML files, adapters or previous results. No extra packages or retraining are required.

## First run: eight-example smoke test

Extract this folder into `/Users/alket/Downloads`, then run:

```bash
cd /Users/alket/Downloads/mami_research
source .venv/bin/activate
caffeinate -i python ../MAMI_Threshold_Addon/score_thresholds.py \
  --config configs/qwen_seed43.yaml \
  --adapter outputs_qwen_seed43/lora_adapter \
  --output outputs_threshold_seed43_smoke8 \
  --limit 8
```

The model should load on MPS with float32. The script explicitly sets float32 for this new scoring experiment without editing your YAML. Use that precision consistently for all three adapters. The smoke test verifies execution, not classification quality. Send `summary.json`, `scores.csv` and `run.json` from the new output folder before starting the full batch.

## Full validation runs, after the smoke test succeeds

Run sequentially to avoid loading multiple models into your Mac's memory:

```bash
caffeinate -i python ../MAMI_Threshold_Addon/score_thresholds.py \
  --config configs/full_corrected.yaml \
  --adapter outputs_full_corrected/lora_adapter \
  --output outputs_threshold_seed42

caffeinate -i python ../MAMI_Threshold_Addon/score_thresholds.py \
  --config configs/qwen_seed43.yaml \
  --adapter outputs_qwen_seed43/lora_adapter \
  --output outputs_threshold_seed43

caffeinate -i python ../MAMI_Threshold_Addon/score_thresholds.py \
  --config configs/qwen_seed44.yaml \
  --adapter outputs_qwen_seed44/lora_adapter \
  --output outputs_threshold_seed44
```

Each output directory must be new. If interrupted, completed rows remain in scores.csv and run.json reports failure. There is no resume feature; use a new output directory for a retry. Incomplete runs must not be treated as full validation results.

## What is being scored?

For every validation meme, the model receives the existing image, transcription and classification prompt. It evaluates both exact candidate responses, NOT_MISOGYNOUS and MISOGYNOUS. The gold label is recorded for evaluation but is never supplied to the model.

For each candidate, the script sums the autoregressive log probabilities of its tokens plus the Qwen end-of-answer token. Prompt tokens and the template's trailing newline are excluded. Token alignment is checked against the generation prompt. Different candidate lengths are deliberately not averaged: these are complete-sequence likelihoods. Label wording and tokenization can affect these scores; this experiment fixes both.

The relative score is:

    score_1 = exp(logp_1) / (exp(logp_0) + exp(logp_1))

The implementation uses a stable sigmoid of logp_1 - logp_0. This is probability mass normalized over two specific answer sequences, NOT a calibrated probability that the meme is misogynous. Both sequences could have low absolute likelihood. Raw log likelihoods and token counts are retained for inspection.

Classify as positive when score_1 >= threshold. Threshold 0.5 chooses the more likely complete candidate. This can disagree with the old greedy generation method, which chooses tokens one at a time and may generate invalid strings. Candidate scoring forces a binary choice; zero invalid labels does not establish that the model is reliable.

There are two sequential forward passes per meme. Existing preprocessing is reused, including its fast/slow image processor configuration. Full-sequence logits are computed and only the answer slices converted to float32 for log-softmax. Runtime and memory use need verification on your Mac.

## Outputs

- scores.csv: labels, both sequence log likelihoods, token counts, relative scores, predictions at 0.5 and image/text hashes. Flushed after each example.
- thresholds.csv: confusion counts, precision, recall, false-positive rate, accuracy and macro-F1 at every unique score and a fixed 0.01 grid. A threshold just above 1 includes the all-negative endpoint. Blank precision means no positives were predicted, not perfect precision.
- summary.json: metrics at 0.5 and smoke-test indicator. No threshold is selected automatically.
- run.json: completion status, actual device/dtype, prompt, resolved configuration, versions, adapter/source hashes and recorded base-model information. The base weights themselves are not rehashed.
- data_report.json: output of the existing corrected split loader. The loader reads all splits for its existing overlap checks; this script scores validation only.

## Research protocol

1. Confirm smoke-test execution and plausible scores.
2. Score the complete validation split for all three seeds. Preserve all runs.
3. Compare false positives and recall across thresholds, including the 0.5 scoring baseline and the separately reported original generation baseline.
4. Decide the operating objective explicitly (e.g. maximum recall subject to a chosen false-positive-rate limit). Do not select a seed or threshold by test performance.
5. Freeze the selection rule and thresholds before further test evaluation. Performance measured on the same validation data used to select a threshold is optimistic; independent evaluation is needed. Our test data have already been inspected, so subsequent test comparisons are exploratory. A fresh held-out dataset would provide stronger confirmation.

For fixed scores, raising the threshold cannot increase the number of false positives, but recall cannot increase either. Precision can fluctuate; higher thresholds do not guarantee improved precision or macro-F1.

## Verification and limits

The delivered script was syntax-checked and its probability conversion and threshold counting were tested against hand-calculated examples, ties and all-negative predictions. Actual Qwen inference, token alignment and MPS execution must be verified by the smoke test on your Mac. This is an experimental addon, not evidence that thresholding will improve held-out performance.

References:
- Hugging Face Qwen2.5-VL model documentation: https://huggingface.co/docs/transformers/model_doc/qwen2_5_vl
- scikit-learn threshold tuning guidance: https://scikit-learn.org/stable/modules/classification_threshold.html
