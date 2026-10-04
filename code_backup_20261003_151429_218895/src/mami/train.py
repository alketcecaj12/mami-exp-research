"""Minimal batch-size-one LoRA training; loss is only on the answer tokens."""
from pathlib import Path
from PIL import Image
from .vlm import Classifier, LABELS


def train_lora(config, frame):
    import torch
    from peft import LoraConfig, get_peft_model
    classifier = Classifier(config)
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
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.train()
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=cfg["learning_rate"])
    optimizer.zero_grad()
    accumulation = cfg["accumulation_steps"]
    for epoch in range(cfg["epochs"]):
        ordered = frame.sample(frac=1, random_state=config["experiment"]["seed"] + epoch).reset_index(drop=True)
        for position, row in enumerate(ordered.itertuples()):
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
                raise ValueError("Non-finite loss; try float32 or a CUDA training machine")
            (loss / group_size).backward()
            if (position + 1) % accumulation == 0 or position + 1 == len(ordered):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
            if position % 10 == 0:
                print(f"epoch={epoch + 1} example={position + 1}/{len(ordered)} loss={loss.item():.4f}", flush=True)
    destination = Path(config["experiment"]["output"]) / "lora_adapter"
    destination.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(destination)
    classifier.processor.save_pretrained(destination)
    print(f"Saved adapter to {destination}")
