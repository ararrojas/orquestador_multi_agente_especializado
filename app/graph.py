"""Grafo de producción: el orquestador multi-agente del M6 + gate de riesgo/HITL,
compilado con un checkpointer de Redis (`AsyncRedisSaver`).

    START → supervisor ⇄ researcher / analyst → synthesizer → validator ─┬─▶ supervisor (refinar)
                                                                          └─▶ risk_gate ─┬─▶ END
                                                                                         └─▶ human_approval ⏸
                                                                                               └─▶ publish_report → END

El `thread_id` de LangGraph es el `job_id`: cada job tiene su historial de checkpoints
en Redis y una ejecución pausada se reanuda desde cualquier worker.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

from langgraph.graph import END, StateGraph

from app.hitl import human_approval, publish_report_node, risk_gate, route_after_gate, wants_publication
from graph import add_orchestrator, initial_state
from state import OrchestratorState

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
CHECKPOINT_TTL_MIN = int(os.getenv("CHECKPOINT_TTL_MIN", str(7 * 24 * 60)))


class ProductionState(OrchestratorState, total=False):
    publish_requested: bool        # la petición implica la herramienta con efectos secundarios
    force_approval: bool           # el cliente exige revisión humana
    critical_reasons: list[str]    # por qué el gate consideró la tarea crítica
    tokens_at_gate: int
    approval: dict                 # decisión humana (approved, reviewer, comment, decided_at)
    published_to: str              # ruta del informe publicado


def build_production_graph(checkpointer, llm=None):
    if llm is None:
        from llm import get_llm

        llm = get_llm()
    g = StateGraph(ProductionState)
    add_orchestrator(g, llm, on_done="risk_gate")
    g.add_node("risk_gate", risk_gate)
    g.add_node("human_approval", human_approval)  # destinos declarados en su Command[...]
    g.add_node("publish_report", publish_report_node)
    g.add_conditional_edges("risk_gate", route_after_gate)
    g.add_edge("publish_report", END)
    return g.compile(checkpointer=checkpointer, name="orquestador-multiagente")


def production_input(question: str, *, publish: bool | None = None, require_approval: bool = False) -> dict:
    return {
        **initial_state(question),
        "publish_requested": wants_publication(question) if publish is None else publish,
        "force_approval": require_approval,
        "critical_reasons": [],
    }


@asynccontextmanager
async def redis_checkpointer(url: str = REDIS_URL) -> AsyncIterator:
    """AsyncRedisSaver con índices creados y TTL (requiere Redis 8 / Redis Stack:
    usa los módulos RedisJSON y RediSearch)."""
    from langgraph.checkpoint.redis.aio import AsyncRedisSaver

    async with AsyncRedisSaver.from_conn_string(
        url, ttl={"default_ttl": CHECKPOINT_TTL_MIN, "refresh_on_read": True}
    ) as saver:
        await saver.asetup()
        yield saver
