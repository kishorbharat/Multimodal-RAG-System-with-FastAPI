from __future__ import annotations

import torch
from PIL.Image import Image
from transformers import pipeline

from src.config import settings


def _vision_dtype(device: str) -> torch.dtype:
    """Use fp16 on CUDA (T4/A100) for ~2x throughput; fp32 everywhere else."""
    return torch.float16 if device == "cuda" else torch.float32


class VisionSummarizer:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        device = settings.device
        self._pipeline = pipeline(
            "image-to-text",
            model=model_name,
            device=device,
            torch_dtype=_vision_dtype(device),
        )

    def summarize(self, image: Image) -> str:
        outputs = self._pipeline(image, max_new_tokens=80)
        if not outputs:
            return ""
        text = outputs[0].get("generated_text", "").strip()
        return text
