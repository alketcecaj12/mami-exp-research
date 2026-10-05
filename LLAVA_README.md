# LLaVA-1.5-7B zero-shot MAMI add-on

This adds a separate inference runner for llava-hf/llava-1.5-7b-hf to the corrected mami_research project. It preserves the Qwen and SmolVLM implementations and outputs. LoRA training is not included.

## Install

Extract this folder in Downloads. From the existing activated environment:

    cd /Users/alket/Downloads/mami_research
    source .venv/bin/activate
    python ../MAMI_LLaVA_Addon/install_addon.py

The installer adds src/mami/llava.py, configs/llava.yaml and LLAVA_README.md only. It copies the corrected data paths and seed from configs/full_corrected.yaml. A repeat installation is safe when the files are identical; differing files are not overwritten. No library upgrade is required by this add-on.

## Download

Check disk space first. Allow at least 20 GB free as an initial working allowance, preferably more for system operation; actual temporary/cache usage can vary.

    df -h .
    mami download --config configs/llava.yaml
    mami inspect --config configs/llava.yaml

The generic download command records the resolved model revision. The corrected split sizes should be 8999 train, 999 validation and 1000 test. Keep the existing Qwen adapters and results.

## Eight-example execution check

    python -m mami.llava --config configs/llava.yaml --split validation --limit 8

Outputs are isolated in outputs_llava/smoke_8. Examine raw responses and invalid counts, not only accuracy. This sample is not a representative performance estimate.

## Balanced development check

    caffeinate -i python -m mami.llava --config configs/llava.yaml --split train --balanced 100

This evaluates 50 positive and 50 negative retained TRAINING examples without fine-tuning. Selection is deterministic: sort by filename within class, sample each class using the experiment seed plus the label, and shuffle using the experiment seed. Output is in outputs_llava/dev_balanced_100. The prediction CSV records the selected identifiers.

This is a development diagnostic, not validation or test evidence. Do not report its score as held-out performance. Retain negative results and do not repeatedly change prompts until this small sample scores well.

Upload the files within that directory:
- vlm_train_metrics.json
- zero_shot_multimodal_train_raw.csv
- vlm_train_config.json

## Full validation, once the configuration is fixed

    caffeinate -i python -m mami.llava --config configs/llava.yaml --split validation

Outputs go to outputs_llava:
- vlm_validation_metrics.json
- zero_shot_multimodal_validation_raw.csv
- zero_shot_multimodal_validation.csv
- vlm_validation_config.json
- data_report.json
- prompt.txt

Each completed prediction is saved. A crash preserves partial raw results; resumption is not implemented. Repeat runs are refused unless --overwrite is explicitly supplied. Overwrite removes old metrics for that run before execution. Smoke, development and full validation outputs are separate.

## Protocol

The runner imports the original classification instruction and exact-label parser from mami.vlm and the metrics and split loader from the existing project. It appends the supplied transcription with the original prefix. No reference label enters the inference prompt.

Defaults:
- Native macOS PyTorch MPS when available; otherwise CUDA or CPU.
- Float16 weights and floating inputs, eager attention, batch size one.
- Native LLaVA chat template and slow image processor, do_pad=True.
- Processor patch size and feature selection match model configuration. CLIP's additional CLS token is accounted for.
- Native image resize/crop configuration is retained (not Qwen's pixel bounds or SmolVLM's longest-edge setting).
- Greedy generation, one beam, 16 new tokens.
- Exact MISOGYNOUS/NOT_MISOGYNOUS parsing. Other answers remain invalid, not silently reinterpreted.

Software versions, model revision, runner hash, resolved device, prompt, preprocessing configuration and completion status are saved. Precision and preprocessing differ across model pipelines and must be disclosed in comparisons. The model is not guaranteed to outperform Qwen. Its performance and numerical behaviour on this Mac must be measured.

Seven-billion-scale float16 weights require roughly 14 GB plus vision components, activations, loading overhead and system memory. This is an estimate, not a measured peak. Do not switch the whole model to float32 on the 36 GB Mac without assessing memory.

The prior test set has already been examined. New model comparisons are exploratory. Do not select models on that test set and then describe it as untouched.

## Checks performed

Python syntax, installer idempotence and conflict handling, deterministic balanced sampling, invalid-response scoring and output isolation were checked locally with a mocked inference model. No LLaVA weights were run on Apple Silicon in the development environment.

## References

- https://huggingface.co/llava-hf/llava-1.5-7b-hf
- https://huggingface.co/docs/transformers/model_doc/llava

Use python -m mami.llava for inference. The existing mami vlm and mami train commands remain Qwen-specific. To remove the add-on, remove its three new project files only; retain research outputs if needed.
