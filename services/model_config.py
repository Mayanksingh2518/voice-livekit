"""Resolve which LLM provider to use.

Both providers are driven through the OpenAI client — Gemini exposes an OpenAI-compatible
endpoint that supports the two things this project depends on: tool calling and structured
outputs. That keeps one code path instead of two SDKs.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

_DEFAULT_MODEL = {"gemini": "gemini-flash-latest", "openai": "gpt-4o-mini"}


@dataclass(frozen=True)
class LLMConfig:
    provider: str
    api_key: str
    base_url: str | None
    model: str

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    @property
    def litellm_model(self) -> str:
        """How LiteLLM names this model — used by Opik's in-process judge metrics."""
        return f"gemini/{self.model}" if self.provider == "gemini" else self.model


def _provider() -> str:
    explicit = os.getenv("LLM_PROVIDER", "").strip().lower()
    if explicit in _DEFAULT_MODEL:
        return explicit
    # Infer from whichever key is present, preferring Gemini since it has a free tier.
    if os.getenv("GEMINI_API_KEY"):
        return "gemini"
    return "openai"


def llm_config(model_override: str | None = None) -> LLMConfig:
    provider = _provider()
    if provider == "gemini":
        return LLMConfig(
            provider="gemini",
            api_key=os.getenv("GEMINI_API_KEY", ""),
            base_url=os.getenv("GEMINI_BASE_URL", GEMINI_BASE_URL),
            model=model_override or os.getenv("LLM_MODEL") or _DEFAULT_MODEL["gemini"],
        )
    return LLMConfig(
        provider="openai",
        api_key=os.getenv("OPENAI_API_KEY", ""),
        base_url=os.getenv("OPENAI_BASE_URL") or None,
        model=model_override or os.getenv("LLM_MODEL") or _DEFAULT_MODEL["openai"],
    )


def analysis_config() -> LLMConfig:
    """The post-call analyser may run a different model from the live agent."""
    return llm_config(os.getenv("ANALYSIS_MODEL") or None)
