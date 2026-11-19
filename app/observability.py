"""Capa de observabilidad: Arize Phoenix (OpenInference/OTel) o LangSmith.

- Phoenix (por defecto): si existe `PHOENIX_COLLECTOR_ENDPOINT`, se registra un
  TracerProvider OTLP y se instrumenta LangChain/LangGraph con OpenInference. Cada
  nodo del grafo, cada llamada al LLM (con tokens de entrada/salida y modelo, de
  donde Phoenix calcula el costo) y cada herramienta queda como un span.
- LangSmith: si `LANGSMITH_TRACING=true` y `LANGSMITH_API_KEY`, LangChain envía las
  trazas solo; aquí solo se usa `traceable` para el span raíz del job.

`traced(name)` decora las funciones del worker (ejecución de un job / reanudación HITL)
para que cada ejecución tenga un span raíz con el `job_id` como session.id.

`UsageMetadataCallbackHandler` cuenta los tokens de cada job; el gate de riesgo lo usa
para detectar ejecuciones de "costo muy alto".
"""

from __future__ import annotations

import functools
import json
import logging
import os
from contextlib import contextmanager
from typing import Any, Callable

from langchain_core.callbacks import UsageMetadataCallbackHandler

log = logging.getLogger(__name__)

PROJECT_NAME = os.getenv("PHOENIX_PROJECT_NAME", os.getenv("LANGSMITH_PROJECT", "pre7-multiagente"))
_backend: str | None = None


def init_observability() -> str:
    """Inicializa el backend de trazas una sola vez por proceso. Devuelve su nombre."""
    global _backend
    if _backend is not None:
        return _backend

    endpoint = os.getenv("PHOENIX_COLLECTOR_ENDPOINT")
    if endpoint:
        from openinference.instrumentation.langchain import LangChainInstrumentor
        from phoenix.otel import register

        if not endpoint.rstrip("/").endswith("/v1/traces"):
            endpoint = endpoint.rstrip("/") + "/v1/traces"
        provider = register(project_name=PROJECT_NAME, endpoint=endpoint, batch=True,
                            set_global_tracer_provider=True, verbose=False)
        LangChainInstrumentor().instrument(tracer_provider=provider)
        _backend = "phoenix"
    elif os.getenv("LANGSMITH_TRACING", "").lower() == "true" and os.getenv("LANGSMITH_API_KEY"):
        _backend = "langsmith"  # LangChain lee las variables de entorno y exporta solo
    else:
        _backend = "none"
    log.info("observabilidad: %s (proyecto %s)", _backend, PROJECT_NAME)
    return _backend


@contextmanager
def job_context(job_id: str, **metadata: Any):
    """Propaga session.id y metadatos a todos los spans hijos (Phoenix agrupa por sesión
    la ejecución inicial y la reanudación tras la aprobación humana)."""
    if _backend == "phoenix":
        from openinference.instrumentation import using_metadata, using_session

        with using_session(job_id), using_metadata({"job_id": job_id, **metadata}):
            yield
    else:
        yield


def traced(name: str) -> Callable:
    """Decorador para funciones async: abre un span raíz (CHAIN) con input/output."""

    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            if _backend == "phoenix":
                from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes
                from opentelemetry import trace
                from opentelemetry.trace import Status, StatusCode

                tracer = trace.get_tracer("app")
                with tracer.start_as_current_span(name) as span:
                    span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND,
                                       OpenInferenceSpanKindValues.CHAIN.value)
                    span.set_attribute(SpanAttributes.INPUT_VALUE, _short(kwargs or args[1:]))
                    try:
                        result = await fn(*args, **kwargs)
                    except Exception as exc:
                        span.record_exception(exc)
                        span.set_status(Status(StatusCode.ERROR, str(exc)[:200]))
                        raise
                    span.set_attribute(SpanAttributes.OUTPUT_VALUE, _short(result))
                    span.set_status(Status(StatusCode.OK))
                    return result
            if _backend == "langsmith":
                from langsmith import traceable

                return await traceable(name=name, run_type="chain")(fn)(*args, **kwargs)
            return await fn(*args, **kwargs)

        return wrapper

    return decorator


def _short(value: Any, limit: int = 4000) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False, default=str)
    except TypeError:
        text = str(value)
    return text[:limit]


# --------------------------------------------------------------------------- #
# Conteo de tokens por job (registro en memoria del proceso worker)
# --------------------------------------------------------------------------- #
_usage: dict[str, UsageMetadataCallbackHandler] = {}


def usage_handler(job_id: str) -> UsageMetadataCallbackHandler:
    handler = _usage[job_id] = UsageMetadataCallbackHandler()
    return handler


def tokens_used(job_id: str) -> int:
    handler = _usage.get(job_id)
    if handler is None:
        return 0
    return sum(int(u.get("total_tokens", 0)) for u in handler.usage_metadata.values())


def release_usage(job_id: str) -> int:
    total = tokens_used(job_id)
    _usage.pop(job_id, None)
    return total


def flush() -> None:
    """Fuerza el envío de los spans pendientes (al apagar el worker)."""
    if _backend == "phoenix":
        from opentelemetry import trace

        provider = trace.get_tracer_provider()
        if hasattr(provider, "force_flush"):
            provider.force_flush()
