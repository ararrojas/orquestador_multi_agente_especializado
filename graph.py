"""Grafo principal: topología jerárquica Supervisor → especialistas → síntesis → validación.

    START → supervisor ─┬─▶ researcher ──┐
                        ├─▶ analyst ─────┤ (vuelven al supervisor)
                        └─▶ synthesizer → validator ─┬─▶ END (aprobado / límite)
                                                     └─▶ supervisor (refinar)
"""

from __future__ import annotations

from typing import Literal

from langgraph.graph import END, START, StateGraph

from agents import (
    make_analyst_node,
    make_research_node,
    make_supervisor_node,
    make_synthesizer_node,
    make_validator_node,
)
from state import OrchestratorState


def route_from_supervisor(state: OrchestratorState) -> Literal["researcher", "analyst", "synthesizer"]:
    return state["next_agent"]


def route_from_validator(state: OrchestratorState) -> Literal["supervisor", "done"]:
    return "done" if state.get("task_completed") else "supervisor"


def add_orchestrator(g: StateGraph, llm, *, on_done: str = END) -> StateGraph:
    """Añade los nodos y aristas del orquestador a `g`.

    `on_done` es el nodo al que se va cuando el validador cierra la tarea
    (END en el M6; el gate de riesgo/HITL en la API de producción).
    """
    g.add_node("supervisor", make_supervisor_node(llm))
    g.add_node("researcher", make_research_node(llm))
    g.add_node("analyst", make_analyst_node(llm))
    g.add_node("synthesizer", make_synthesizer_node(llm))
    g.add_node("validator", make_validator_node(llm))

    g.add_edge(START, "supervisor")
    g.add_conditional_edges("supervisor", route_from_supervisor, ["researcher", "analyst", "synthesizer"])
    g.add_edge("researcher", "supervisor")
    g.add_edge("analyst", "supervisor")
    g.add_edge("synthesizer", "validator")
    g.add_conditional_edges("validator", route_from_validator, {"supervisor": "supervisor", "done": on_done})
    return g


def build_graph(llm=None, *, checkpointer=None):
    """Compila el grafo. `llm` es inyectable (útil para tests con modelos falsos)."""
    if llm is None:
        from llm import get_llm

        llm = get_llm()
    return add_orchestrator(StateGraph(OrchestratorState), llm).compile(checkpointer=checkpointer)


def initial_state(question: str) -> dict:
    return {
        "messages": [("user", question)],
        "contributions": [],
        "agent_calls": {},
        "step_count": 0,
        "revision_count": 0,
        "validation_feedback": "",
        "task_completed": False,
    }
