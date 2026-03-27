"""LLM backend configuration.

Supports Ollama (local) and Claude API (cloud) backends.
Configuration is loaded from environment variables or .env files.
"""

from __future__ import annotations

import os
from typing import Literal

from langchain_core.language_models import BaseChatModel


def get_llm(
    backend: Literal["ollama", "claude"] = "ollama",
    model: str | None = None,
    temperature: float = 0.0,
    base_url: str | None = None,
) -> BaseChatModel:
    """Create an LLM instance.

    Parameters
    ----------
    backend : str
        "ollama" for local Ollama, "claude" for Anthropic Claude API.
    model : str, optional
        Model name. Defaults: ollama="qwen3:32b", claude="claude-sonnet-4-20250514".
    temperature : float
        Sampling temperature.
    base_url : str, optional
        Ollama server URL. Default from OLLAMA_BASE_URL env var or localhost.

    Returns
    -------
    BaseChatModel
        A LangChain chat model ready for use with LangGraph.
    """
    if backend == "ollama":
        from langchain_ollama import ChatOllama

        model = model or os.environ.get("OLLAMA_MODEL", "qwen3:32b")
        base_url = base_url or os.environ.get(
            "OLLAMA_BASE_URL", "http://localhost:11434"
        )
        return ChatOllama(
            model=model,
            base_url=base_url,
            temperature=temperature,
        )

    elif backend == "claude":
        from langchain_anthropic import ChatAnthropic

        model = model or os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY environment variable required for Claude backend"
            )
        return ChatAnthropic(
            model=model,
            temperature=temperature,
            api_key=api_key,
            max_tokens=4096,
        )

    else:
        raise ValueError(f"Unknown backend: {backend!r}. Use 'ollama' or 'claude'.")
