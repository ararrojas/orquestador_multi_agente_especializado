"""Fábrica de modelos. El proveedor se elige por variable de entorno
(`LLM_MODEL`, formato `proveedor:modelo` de `init_chat_model`), p. ej.:

    anthropic:claude-opus-5-5          (ANTHROPIC_API_KEY)
    google_genai:gemini-3.5-flash-lite      (GOOGLE_API_KEY, tiene free tier)

`LLM_RPM` limita las peticiones por minuto (útil con cuotas gratuitas).
"""

from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.rate_limiters import InMemoryRateLimiter

load_dotenv()

DEFAULT_MODEL = "anthropic:claude-opus-5-5"


@lru_cache(maxsize=4)
def get_llm(model: str | None = None) -> BaseChatModel:
    # timeout + pocos reintentos: si el proveedor no responde, fallar con un error claro
    # en vez de quedarse colgado varios minutos.
    kwargs = {"max_tokens": 8000, "timeout": float(os.getenv("LLM_TIMEOUT", "60")), "max_retries": 2}
    rpm = float(os.getenv("LLM_RPM", "0") or 0)
    if rpm > 0:
        kwargs["rate_limiter"] = InMemoryRateLimiter(requests_per_second=rpm / 60, check_every_n_seconds=0.2)
    return init_chat_model(model or os.getenv("LLM_MODEL", DEFAULT_MODEL), **kwargs)


def message_text(message) -> str:
    """Texto plano de un mensaje (ignora bloques de thinking / tool_use)."""
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    parts = []
    for block in content or []:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "".join(parts).strip()
