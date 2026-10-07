MAMI training numerical stability fix

Extract this ZIP next to your existing mami_research folder.
From your existing activated mami_research terminal:

python ../MAMI_Training_Fix/install_fix.py .
mami train --config configs/mac_stable.yaml --limit 8

If the smoke test succeeds, start full training (fresh base model):
mami train --config configs/mac_stable.yaml

For a future interruption of this full run, resume with:
mami train --config configs/mac_stable.yaml --resume outputs_fp32/training_checkpoint.pt

Do not resume the 8-example checkpoint into a full run. Dataset/settings checks
will reject this. The original failed run had no intermediate checkpoint and
cannot be resumed. Its old outputs/lora_adapter is probably the smoke adapter.

After full training completes:
mami vlm --config configs/mac_stable.yaml --split validation --adapter outputs_fp32/lora_adapter

Results: outputs_fp32/vlm_validation_metrics.json

Changes:
- Training loads float32 directly; inference keeps previous auto precision.
- Checkpoints save adapter weights, optimizer, RNG states and progress at every
  completed optimizer step (8 examples with your existing settings).
- The last checkpoint is replaced atomically, after finite-gradient and
  finite-parameter checks. Resume requires identical data and settings.
- Checkpointing uses non-reentrant autograd to handle frozen inputs.
- Finite-loss errors identify the example; no examples are silently skipped.
- mac_stable.yaml uses a separate outputs_fp32 folder, preserving prior results.
- No new dependency installation or model download is necessary.

Float32 roughly doubles base-weight memory versus float16 and may be slower.
This targets a likely precision problem; it does not guarantee the cause is fixed.
If there is another failure, provide the full error, especially file_name.
An unchanged deterministic run can hit the same error again on resume.
Do not disable Mac memory safety limits or skip a troublesome training row.

Checks in the authoring environment: Python compilation, existing lightweight
regression tests, installer verification and source review. Mac GPU inference,
float32 training and live checkpoint resume still require local verification.
