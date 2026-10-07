# SmolVLM2 zero-shot comparison for MAMI

This add-on evaluates `HuggingFaceTB/SmolVLM2-2.2B-Instruct` using the existing corrected MAMI splits, original Qwen task instruction, supplied transcription, strict label parser and evaluation metrics. It adds a separate runner; it does not enable SmolVLM training or alter Qwen code. Do not pass this model to the existing `mami train` or `mami vlm` commands: those remain Qwen-specific.

## Install on your Mac

Extract `MAMI_SmolVLM_Addon.zip` in Downloads so the installer is at `~/Downloads/MAMI_SmolVLM_Addon/install_addon.py`. From the existing project:

```bash
cd /Users/alket/Downloads/mami_research
source .venv/bin/activate
python ../MAMI_SmolVLM_Addon/install_addon.py
python -m pip install 'num2words==0.5.14'
```

The installer creates three new files only: `src/mami/smolvlm.py`, `configs/smolvlm2.yaml`, and `SMOLVLM_README.md`. It derives data paths and seed from your existing `configs/full_corrected.yaml`, checks for split-specific image directories, and refuses to overwrite differing existing files. Your editable project installation makes the new module available without reinstalling the project. Python 3.11 and the existing Transformers 4.57.1 environment are used; no torch/Transformers upgrade or Flash Attention installation is requested.

## Download and inspect

Ensure sufficient free disk space before downloading (about 10 GB free is a reasonable initial allowance; actual cache usage can vary).

```bash
df -h .
mami download --config configs/smolvlm2.yaml
mami inspect --config configs/smolvlm2.yaml
```

The existing download command is model-independent and records the resolved model commit in `models/smolvlm2-2.2b/model_info.json`. Expected corrected counts: 8,999 training, 999 validation and 1,000 test. Investigate differences before comparing scores.

## Small execution check

```bash
python -m mami.smolvlm --config configs/smolvlm2.yaml --split validation --limit 8
```

Results go to `outputs_smolvlm2/smoke_8/`. A valid output must be exactly `MISOGYNOUS` or `NOT_MISOGYNOUS` after whitespace trimming. Invalid outputs remain errors; do not extract a convenient label from a longer response. The first eight examples are an execution check, not a representative performance estimate. Do not tune the prompt to obtain high scores on them.

## Full validation

Once the small run confirms that execution completes, evaluate the frozen configuration:

```bash
caffeinate -i python -m mami.smolvlm --config configs/smolvlm2.yaml --split validation
```

Keep the Mac plugged in with the lid open. Upload these files for analysis:

- `outputs_smolvlm2/vlm_validation_metrics.json`
- `outputs_smolvlm2/zero_shot_multimodal_validation_raw.csv`
- `outputs_smolvlm2/vlm_validation_config.json`

The runner also writes the simplified prediction CSV, data report and exact prompt. Each prediction is saved as it completes. An interrupted run preserves partial raw results, but inference resumption is not implemented. Re-running the same split/limit is blocked unless `--overwrite` is explicitly supplied. Full evaluation and smoke runs use separate directories. An explicit overwrite clears prior metrics for that run so a failed new run cannot leave old metrics appearing current.

## Methodological choices

- Float32 on the automatically selected device (MPS on the user's Mac), eager attention, batch size one. Approximately 8.8 GB for 2.2B float32 weights alone; activations, loading overhead and macOS require additional memory. Actual speed and peak memory must be measured locally.
- SmolVLM's native slow image processor, with longest edge fixed to 1,024 pixels. Native patch size and splitting settings are retained. Qwen's pixel-budget settings do not transfer directly to SmolVLM. The resolved image processor configuration is saved with the results.
- The original task prompt is imported from `mami.vlm.PROMPT` at runtime, not the temporary shorter 7B prompt. Its exact content is saved. The supplied transcription is appended with the same prefix. No human label is included in the prompt.
- Greedy generation with one beam and 16 new tokens, matching the original answer budget. The model's own chat template and processor are used. The result parser and metric implementation are imported from the project.
- SmolVLM runs in float32 whereas the recorded Qwen inference used float16. This compares practical model pipelines, not parameter count alone. Document this precision difference and the different native preprocessing.
- This is a new exploratory model-family comparison after the original test results were examined. Establish settings on validation; avoid using the already examined test set to support new confirmatory claims without qualification.
- SmolVLM performance is unknown until measured. No claim of superior reliability, accuracy, or successful Mac execution is made by this package.

## Test evaluation (later, with settings fixed)

```bash
python -m mami.smolvlm --config configs/smolvlm2.yaml --split test
```

This uses distinct test filenames. Existing Qwen output directories are never targeted by the supplied configuration.

## Validation of this add-on

The package was checked for Python syntax, installer idempotence/conflict handling, and output/invalid-response handling using a mocked model. Model-weight inference on Apple Silicon was not available in the development environment. The eight-example run is required to verify the actual local processor/model/device combination.

## References

- Model card: https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct
- Transformers 4.57.1 SmolVLM documentation: https://huggingface.co/docs/transformers/v4.57.1/model_doc/smolvlm

Retain the saved commit and software versions with reported results. To remove the add-on, remove only its three newly created project files; keep outputs if they form part of the research record.
