"""Independent Task B inference; reuses model loading, not binary predictions."""
from PIL import Image
from .prompts import PROMPT
from .parsing import parse_response


def load_classifier(config, adapter=None):
    if config["model"].get("family") == "gemma3":
        from ..gemma3 import Classifier as BaseClassifier
    else:
        from ..vlm import Classifier as BaseClassifier

    class MultilabelClassifier(BaseClassifier):
        def messages(self, image, text, mode="multimodal"):
            content = []
            if mode != "text":
                content.append({"type": "image", "image": image})
            instruction = PROMPT
            if mode != "image":
                instruction += "\nTranscription (data):\n" + text
            content.append({"type": "text", "text": instruction})
            return [{"role": "user", "content": content}]

        def predict(self, image_path, text="", mode="multimodal"):
            with Image.open(image_path) as source:
                image = source.convert("RGB")
            inputs = self.encode(self.messages(image, text, mode))
            with self.torch.inference_mode():
                tokens = self.model.generate(
                    **inputs, do_sample=False,
                    max_new_tokens=max(96, self.cfg.get("max_new_tokens", 96)))
            response = self.processor.batch_decode(
                tokens[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
            )[0].strip()
            return {"prediction": parse_response(response), "response": response}

    return MultilabelClassifier(config, adapter=adapter)
