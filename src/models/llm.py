from __future__ import annotations

import os

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from src.config import settings


class LocalFallbackChatModel:
    """Simple wrapper over transformers text2text pipeline with a chat-like invoke API."""

    def __init__(self, model_name: str) -> None:
        import torch
        from langchain_huggingface import HuggingFacePipeline
        from transformers import AutoConfig, pipeline

        from src.config import settings

        config = AutoConfig.from_pretrained(model_name)
        task = "text2text-generation" if config.is_encoder_decoder else "text-generation"
        device = settings.device

        pipe_kwargs = {
            "task": task,
            "model": model_name,
            "max_new_tokens": 512,
            "do_sample": False,
            "device": device,
            # fp16 on CUDA (T4 / A100) halves VRAM and speeds up inference
            "torch_dtype": torch.float16 if device == "cuda" else torch.float32,
        }
        if task == "text-generation":
            pipe_kwargs["return_full_text"] = False

        pipe = pipeline(**pipe_kwargs)
        self._llm = HuggingFacePipeline(pipeline=pipe)

    def invoke(self, prompt: str):
        return self._llm.invoke(prompt)


def load_llm() -> BaseChatModel | LocalFallbackChatModel:
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=settings.openai_model, temperature=0)
    return LocalFallbackChatModel(model_name=settings.local_llm_model)


def invoke_llm(llm: BaseChatModel | LocalFallbackChatModel, prompt: str) -> str:
    if hasattr(llm, "_llm"):
        result = llm.invoke(prompt)
        return str(result).strip()

    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content.strip() if hasattr(response, "content") else str(response).strip()
