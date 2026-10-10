"""Gemma 3 multimodal adapter, sharing Qwen's labels and prompt."""
from pathlib import Path
from PIL import Image
from .vlm import PROMPT, parse_label

class Classifier:
    def __init__(self, config, adapter=None, training=False):
        import torch
        from transformers import AutoProcessor, Gemma3ForConditionalGeneration
        cfg = config["model"]
        self.torch, self.cfg = torch, cfg
        self.device = cfg.get("device", "auto")
        if self.device == "auto":
            self.device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        directory = Path(cfg["local_dir"])
        if not (directory / "config.json").exists():
            raise FileNotFoundError("Download model first: mami download --config configs/gemma3_pilot.yaml")
        self.processor = AutoProcessor.from_pretrained(directory, local_files_only=True)
        precision = config.get("training", {}).get("dtype", "float32") if training else cfg.get("dtype", "float32")
        if precision == "auto":
            precision = "float32" if self.device == "cpu" else "float16"
        dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[precision]
        self.model = Gemma3ForConditionalGeneration.from_pretrained(directory, dtype=dtype, attn_implementation="eager", local_files_only=True).to(self.device)
        if adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model.eval()
        print(f"Loaded Gemma 3 on {self.device}, dtype={precision}", flush=True)

    def messages(self, image, text, mode="multimodal"):
        content = []
        if mode != "text":
            content.append({"type": "image", "image": image})
        instruction = PROMPT
        if mode != "image":
            instruction += "\nTranscription (data):\n" + text
        content.append({"type": "text", "text": instruction})
        return [{"role": "user", "content": content}]

    def encode(self, messages, generation=True):
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=generation)
        images = [item["image"] for message in messages for item in message["content"] if item["type"] == "image"]
        kwargs = {"text": [prompt], "return_tensors": "pt", "padding": True}
        if images:
            kwargs["images"] = images
        return self.processor(**kwargs).to(self.device)

    def predict(self, image_path, text="", mode="multimodal"):
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        inputs = self.encode(self.messages(image, text, mode))
        with self.torch.inference_mode():
            tokens = self.model.generate(**inputs, do_sample=False, max_new_tokens=self.cfg.get("max_new_tokens", 16))
        response = self.processor.batch_decode(tokens[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
        return {"prediction": parse_label(response), "response": response}
