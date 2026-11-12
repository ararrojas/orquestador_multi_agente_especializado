"""Esquema de estado compartido del orquestador multi-agente.

`OrchestratorState` hereda de `MessagesState` (canal `messages` con reducer
`add_messages`) y añade campos estructurados para que el Supervisor sepa en todo
momento *qué agente aportó qué*, cuántos pasos lleva el sistema y si la tarea
está terminada.
"""

from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict

from langgraph.graph import MessagesState

# Nombres de nodo del grafo. El Supervisor solo puede devolver uno de estos.
AgentName = Literal["researcher", "analyst"]
Route = Literal["researcher", "analyst", "synthesizer"]


class Contribution(TypedDict):
    """Una aportación atómica de un especialista (trazabilidad por agente)."""

    agent: AgentName          # quién aportó
    step: int                 # en qué paso global del grafo
    instruction: str          # qué le pidió el Supervisor
    output: str               # qué devolvió
    tools_used: list[str]     # herramientas que ejecutó para obtenerlo


def merge_counts(left: dict[str, int], right: dict[str, int]) -> dict[str, int]:
    """Reducer: suma contadores por agente en vez de sobrescribirlos."""
    merged = dict(left or {})
    for key, value in (right or {}).items():
        merged[key] = merged.get(key, 0) + value
    return merged


class OrchestratorState(MessagesState):
    # --- Control de flujo (lo escribe el Supervisor) ---
    next_agent: Route                 # a qué nodo enrutar ahora
    current_instruction: str          # instrucción acotada para el especialista
    supervisor_reasoning: str         # por qué tomó esa decisión (auditoría)

    # --- Trazabilidad (lo escriben los especialistas; append-only) ---
    contributions: Annotated[list[Contribution], operator.add]
    agent_calls: Annotated[dict[str, int], merge_counts]

    # --- Condiciones de parada ---
    step_count: int                   # pasos del Supervisor consumidos
    revision_count: int               # rondas de refinamiento pedidas por el validador

    # --- Síntesis y validación ---
    draft_answer: str                 # borrador del sintetizador
    validation_feedback: str          # crítica del validador (vacía si aprobó)
    task_completed: bool              # True cuando el validador aprueba o se agota el presupuesto
    final_answer: str
