"""Lógica común a los especialistas: aislamiento de contexto + trazabilidad."""

from __future__ import annotations

import warnings

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.prebuilt import create_react_agent

from llm import message_text
from state import AgentName, Contribution, OrchestratorState

SPECIALIST_RECURSION_LIMIT = 12  # tope de iteraciones ReAct internas por invocación

# Callback opcional para mostrar progreso en vivo (main.py lo activa; los tests no).
progress_hook = None


def _report(text: str) -> None:
    if progress_hook:
        progress_hook(text)


def original_question(state: OrchestratorState) -> str:
    """Primera petición humana (la del usuario), no los mensajes internos."""
    for m in state["messages"]:
        if isinstance(m, HumanMessage):
            return message_text(m)
    return ""


def build_specialist_node(name: AgentName, llm, tools, system_prompt: str, build_context):
    """Crea un nodo de grafo que ejecuta un agente ReAct con herramientas acotadas.

    `build_context(state) -> str` decide QUÉ ve el especialista: solo su instrucción
    y el contexto mínimo necesario (no el historial completo del sistema).
    """
    # La consigna pide create_react_agent; en LangGraph 1.x sigue funcionando (deprecado a favor
    # de langchain.agents.create_agent, con la misma semántica).
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        agent = create_react_agent(llm, tools, prompt=system_prompt, name=name)

    def node(state: OrchestratorState) -> dict:
        task = HumanMessage(content=build_context(state))
        _report(f"   … {name} trabajando")
        inner = [task]
        for update in agent.stream(
            {"messages": [task]}, {"recursion_limit": SPECIALIST_RECURSION_LIMIT}, stream_mode="updates"
        ):
            for delta in update.values():
                for m in (delta or {}).get("messages", []):
                    inner.append(m)
                    for call in getattr(m, "tool_calls", None) or []:
                        _report(f"   ↳ {name} llama {call['name']}({call['args']})")
        output = message_text(inner[-1]) or "(sin salida)"
        tools_used = [m.name for m in inner if isinstance(m, ToolMessage)]
        contribution: Contribution = {
            "agent": name,
            "step": state.get("step_count", 0),
            "instruction": state.get("current_instruction", ""),
            "output": output,
            "tools_used": tools_used,
        }
        return {
            "contributions": [contribution],
            "agent_calls": {name: 1},
            # Al historial global solo sube el resultado final, etiquetado con el autor.
            "messages": [AIMessage(content=output, name=name)],
        }

    node.__name__ = f"{name}_node"
    return node


def format_contributions(state: OrchestratorState, agent: AgentName | None = None) -> str:
    items = [c for c in state.get("contributions", []) if agent is None or c["agent"] == agent]
    if not items:
        return "(ninguna todavía)"
    return "\n\n".join(
        f"### {c['agent']} · paso {c['step']} · herramientas: {', '.join(c['tools_used']) or 'ninguna'}\n"
        f"Instrucción: {c['instruction']}\n{c['output']}"
        for c in items
    )
