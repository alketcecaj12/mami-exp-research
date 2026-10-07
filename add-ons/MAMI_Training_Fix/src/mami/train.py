"""Minimal batch-size-one LoRA training; loss is only on the answer tokens."""
from pathlib import Path
from PIL import Image
from .vlm import Classifier, LABELS


def train_lora(config, frame, resume=None):
    import torch
    from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
    classifier = Classifier(config, training=True)
    cfg = config["training"]
    torch.manual_seed(config["experiment"]["seed"])
    # Target language attention only; keep the vision encoder frozen.
    targets = [name for name, module in classifier.model.named_modules()
               if "visual" not in name and name.split(".")[-1] in {"q_proj", "k_proj", "v_proj", "o_proj"}
               and isinstance(module, torch.nn.Linear)]
    if not targets:
        raise ValueError("No language attention modules found")
    model = get_peft_model(classifier.model, LoraConfig(
        r=cfg["rank"], lora_alpha=cfg["alpha"], lora_dropout=0.05,
        target_modules=targets, bias="none", task_type="CAUSAL_LM"))
    model.print_trainable_parameters()
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.train()
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=cfg["learning_rate"])
    optimizer.zero_grad()
    accumulation = cfg["accumulation_steps"]
    import hashlib
    import json
    # Include image bytes and metadata so resume refuses a changed dataset.
    digest = hashlib.sha256()
    for row in frame.itertuples():
        digest.update(json.dumps([row.file_name, row.text, int(row.label)]).encode())
        digest.update(Path(row.image_path).read_bytes())
    signature = {"data_sha256": digest.hexdigest(), "training": cfg,
                 "model": config["model"], "seed": config["experiment"]["seed"]}
    start_epoch, start_position = 0, 0
    destination = Path(config["experiment"]["output"])
    destination.mkdir(parents=True, exist_ok=True)
    checkpoint = destination / "training_checkpoint.pt"

    def cpu_copy(value):
        if torch.is_tensor(value):
            return value.detach().cpu().clone()
        if isinstance(value, dict):
            return {k: cpu_copy(v) for k, v in value.items()}
        if isinstance(value, list):
            return [cpu_copy(v) for v in value]
        return value

    def save_checkpoint(epoch, position):
        state = {"adapter": cpu_copy(get_peft_model_state_dict(model)),
                 "optimizer": cpu_copy(optimizer.state_dict()),
                 "epoch": epoch, "position": position, "signature": signature,
                 "cpu_rng": torch.get_rng_state()}
        if classifier.device == "mps":
            state["mps_rng"] = torch.mps.get_rng_state().cpu()
        if classifier.device == "cuda":
            state["cuda_rng"] = torch.cuda.get_rng_state_all()
        temporary = checkpoint.with_suffix(".tmp")
        torch.save(state, temporary)
        temporary.replace(checkpoint)

    if resume:
        state = torch.load(resume, map_location="cpu", weights_only=True)
        if state["signature"] != signature:
            raise ValueError("Resume requires the same data, seed, model and training settings")
        set_peft_model_state_dict(model, state["adapter"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch, start_position = state["epoch"], state["position"]
        torch.set_rng_state(state["cpu_rng"])
        if "mps_rng" in state and classifier.device == "mps":
            torch.mps.set_rng_state(state["mps_rng"])
        if "cuda_rng" in state and classifier.device == "cuda":
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        print(f"Resuming epoch {start_epoch + 1}, next example {start_position + 1}")
    else:
        save_checkpoint(0, 0)
    for epoch in range(start_epoch, cfg["epochs"]):
        ordered = frame.sample(frac=1, random_state=config["experiment"]["seed"] + epoch).reset_index(drop=True)
        for position, row in enumerate(ordered.itertuples()):
            if epoch == start_epoch and position < start_position:
                continue
            with Image.open(row.image_path) as source:
                image = source.convert("RGB")
            messages = classifier.messages(image, row.text)
            prefix = classifier.encode(messages)
            messages.append({"role": "assistant", "content": [{"type": "text", "text": LABELS[row.label]}]})
            batch = classifier.encode(messages, generation=False)
            prefix_length = prefix["input_ids"].shape[1]
            if not torch.equal(prefix["input_ids"][0], batch["input_ids"][0, :prefix_length]):
                raise ValueError("Chat template prefix mismatch; answer masking would be incorrect")
            labels = batch["input_ids"].clone()
            labels[:, :prefix_length] = -100
            labels[batch["attention_mask"] == 0] = -100
            # Normalize the final partial accumulation group by its actual size.
            group_start = (position // accumulation) * accumulation
            group_size = min(accumulation, len(ordered) - group_start)
            loss = model(**batch, labels=labels).loss
            if not torch.isfinite(loss):
                raise ValueError(f"Non-finite loss at epoch={epoch+1}, example={position+1}, file={row.file_name}. "
                                 f"Last clean checkpoint: {checkpoint}. No example was skipped.")
            (loss / group_size).backward()
            if (position + 1) % accumulation == 0 or position + 1 == len(ordered):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
                if any(not torch.isfinite(p).all() for p in model.parameters() if p.requires_grad):
                    raise ValueError(f"Non-finite adapter weights; last clean checkpoint: {checkpoint}")
                optimizer.zero_grad()
                # Save only at completed optimizer steps, when no gradients are pending.
                next_epoch, next_position = (epoch + 1, 0) if position + 1 == len(ordered) else (epoch, position + 1)
                save_checkpoint(next_epoch, next_position)
            if position % 10 == 0 or position + 1 == len(ordered):
                print(f"epoch={epoch + 1} example={position + 1}/{len(ordered)} loss={loss.item():.4f}", flush=True)
    destination = Path(config["experiment"]["output"]) / "lora_adapter"
    destination.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(destination)
    classifier.processor.save_pretrained(destination)
    print(f"Saved adapter to {destination}")
