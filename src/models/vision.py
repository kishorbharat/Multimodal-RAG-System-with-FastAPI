from __future__ import annotations

from PIL.Image import Image
from transformers import pipeline


class VisionSummarizer:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._pipeline = pipeline("image-to-text", model=model_name)

    def summarize(self, image: Image) -> str:
        outputs = self._pipeline(image, max_new_tokens=80)
        if not outputs:
            return ""
        text = outputs[0].get("generated_text", "").strip()
        return text
